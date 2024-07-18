# /bin/python3
Author = "xuwj"

import gi
import configparser
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# import proto.task_pb2 as task_pb2

from common.FPS import PERF_DATA
import ctypes
import cupy as cp
import cv2 

import threading
import os, sys
import pyds
import json
import time
import queue

from sort import BoxTracker

GPU_ID = 0

MAX_NUM_SOURCES = 16

MUXER_OUTPUT_WIDTH=1920
MUXER_OUTPUT_HEIGHT=1080

PGIE_CONFIG_FILE = '/root/apps/ai_server/cfg/config_infer_primary_yoloV8n.txt'
TRACKER_CONFIG_FILE = "/root/apps/ai_server/cfg/dstest_tracker_config.txt"


def get_data_GPU(gst_buffer, frame_meta):
    # Create dummy owner object to keep memory for the image array alive
    owner = None
    # Getting Image data using nvbufsurface
    # the input should be address of buffer and batch_id
    # Retrieve dtype, shape of the array, strides, pointer to the GPU buffer, and size of the allocated memory
    data_type, shape, strides, dataptr, size = pyds.get_nvds_buf_surface_gpu(hash(gst_buffer), frame_meta.batch_id)
    # dataptr is of type PyCapsule -> Use ctypes to retrieve the pointer as an int to pass into cupy
    ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
    ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    # Get pointer to buffer and create UnownedMemory object from the gpu buffer
    c_data_ptr = ctypes.pythonapi.PyCapsule_GetPointer(dataptr, None)
    unownedmem = cp.cuda.UnownedMemory(c_data_ptr, size, owner)
    # Create MemoryPointer object from unownedmem, at index 0
    memptr = cp.cuda.MemoryPointer(unownedmem, 0)
    # Create cupy array to access the image data. This array is in GPU buffer
    n_frame_gpu = cp.ndarray(shape=shape, dtype=data_type, memptr=memptr, strides=strides, order='C')
    # Initialize cuda.stream object for stream synchronization
    # stream = cp.cuda.stream.Stream(null=True) # Use null stream to prevent other cuda applications from making illegal memory access of buffer
    # Modify the red channel to add blue tint to image
    
    # with stream:
    return n_frame_gpu.get()

class Plumber():
    GetRunningState_list = [[] for i in range(MAX_NUM_SOURCES)]
    num_sources = 0
    task_queue = queue.Queue()
    stand_by = False

    def __init__(self, logging):
        self.logging = logging
        self.g_source_id_list = [0] * MAX_NUM_SOURCES
        self.g_eos_list = [False] * MAX_NUM_SOURCES
        self.g_source_enabled = [False] * MAX_NUM_SOURCES
        self.g_source_bin_list = [None] * MAX_NUM_SOURCES
        self.g_sink_bin_list = [None] * MAX_NUM_SOURCES
        self.g_queue_bin_list = [None] * MAX_NUM_SOURCES
        self.g_convert_bin_list = [None] * MAX_NUM_SOURCES
        self.g_filter_bin_list = [None] * MAX_NUM_SOURCES
        # self.src_task_count = [0] * MAX_NUM_SOURCES

        self.trackers = {}
        self.streammuxs = {}

        # self.add_task_/que_flag = 1

        self.lock = threading.Lock()

        self.perf_data = PERF_DATA(8)
        self.tiler_size = json.load(open("/root/apps/ai_server/cfg/config.json", 'r')) 
        t = threading.Thread(target=self.run, args=())
        t.start()

    # def tiler_sink_pad_buffer_probe(self,pad,info,u_data):     
    #     src_id = 0
    #     frame_number=0

    #     gst_buffer = info.get_buffer()    
    #     batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    #     l_frame = batch_meta.frame_meta_list
    #     while l_frame is not None:

    #         try:
    #             frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
    #         except StopIteration:
    #             break

    #         # src_id = frame_meta.source_id
    #         src_id = frame_meta.pad_index

    #         frame_number = frame_meta.frame_num

    #         # if frame_number % 2 == 0:
    #         #     n_frame_cpu = get_data_GPU(gst_buffer, frame_meta)
    #         #     img_arr = cv2.cvtColor(n_frame_cpu, cv2.COLOR_RGBA2BGRA)

    #         #     # 保存图片
    #         #     img_path = "/root/apps/ai_server/stream{}_frame{}.jpg".format(src_id, frame_number)
    #         #     cv2.imwrite(img_path, img_arr)

    #         #     if frame_number >= 1000:
    #         #         del_img_path = "/root/apps/ai_server/stream{}_frame{}.jpg".format(src_id, frame_number - 1000)
    #         #         if os.path.exists(del_img_path):
    #         #             os.remove(del_img_path)

    #         # if task_queue.qsize() < 16:
    #         #     add_task_que_flag = 1

    #         if self.add_task_que_flag:
    #             # 获取锁
    #             # lock.acquire()
    #             # try:                        
    #                 for jpg_src_id in range(1):
    #                     # 遍历当前路径, 获取所有jpg
    #                     img_dir = "/root/apps/ai_server/stream"+str(src_id)
                        
    #                     for f in os.listdir(img_dir):
    #                         if f.endswith("src.jpg"):
    #                             img_name = f[:-4]
    #                             src_img_path = os.path.join(img_dir, img_name[:-3] + "src.jpg")
    #                             roi_img_path = os.path.join(img_dir, img_name[:-3] + "roi.jpg")
    #                             # print("src_img_path=", src_img_path)
    #                             # print("roi_img_path=", roi_img_path)

    #                             # 判断图片是否存在
    #                             if not os.path.exists(src_img_path) or not os.path.exists(roi_img_path):
    #                                 if os.path.exists(src_img_path):
    #                                     os.remove(src_img_path)
    #                                 if os.path.exists(roi_img_path):
    #                                     os.remove(roi_img_path)
    #                                 # print("img not exist")
    #                                 break

    #                             self.lock.acquire()
    #                             try:
    #                                 with open(os.path.join(img_dir, img_name[:-3] + "src.jpg"), 'rb') as f:
    #                                     src_img = f.read()

    #                                 with open(os.path.join(img_dir, img_name[:-3] + "roi.jpg"), 'rb') as f:
    #                                     roi_img = f.read()

    #                                 jpg_src_id = int(img_name.split("_")[0][6:])
    #                                 k_obj = int(img_name.split("_")[2])

    #                                 # task_notify = task_pb2.TaskNotify()

    #                                 # # print("debug==================", GetRunningState_list[jpg_src_id])
    #                                 # if Plumber.GetRunningState_list[jpg_src_id]:
    #                                 #     task_notify.collect_notify.channel.id = Plumber.GetRunningState_list[jpg_src_id][0]
    #                                 #     task_notify.collect_notify.channel.name = Plumber.GetRunningState_list[jpg_src_id][1]
    #                                 #     task_notify.collect_notify.channel.url = Plumber.GetRunningState_list[jpg_src_id][2]
                                    
    #                                 # task_notify.collect_notify.picture.id = k_obj
    #                                 # task_notify.collect_notify.picture.srcData = src_img
    #                                 # task_notify.collect_notify.picture.roiData = roi_img
    #                                 # task_notify.collect_notify.picture.type = task_pb2.PICTURE_TYPE_PERSON
    #                                 # # print(len(src_img), len(roi_img))

    #                                 # if self.src_task_count[jpg_src_id] >= 100:
    #                                 #     self.src_task_count[jpg_src_id] = 1
    #                                 # else:
    #                                 #     self.src_task_count[jpg_src_id] +=1
    #                                 # # print(src_task_count)
                                    

    #                                 # # 获取文件夹待发送图片数量
    #                                 # folder_imgs_num = len(os.listdir("/root/apps/ai_server/stream"+str(jpg_src_id)))
    #                                 # skip_num = folder_imgs_num // 2 // 100 + 1
    #                                 # # print("skip_num: ", skip_num)

    #                                 # if Plumber.task_queue.qsize() < 1000 and self.src_task_count[jpg_src_id] % skip_num == 0:
    #                                 #     Plumber.task_queue.put(task_notify)

    #                                 # 删除图片
    #                                 # os.remove(os.path.join(img_dir, img_name[:-3] + "src.jpg"))
    #                                 # os.remove(os.path.join(img_dir, img_name[:-3] + "roi.jpg"))
    #                                 # 保存图片
    #                                 # with open(os.path.join("/root/apps/ai_server/tmp", img_name[:-3] + "src.jpg"), 'wb') as f:
    #                                 #     f.write(src_img)
    #                                 # with open(os.path.join("/root/apps/ai_server/tmp", img_name[:-3] + "roi.jpg"), 'wb') as f:
    #                                 #     f.write(roi_img)
    #                             finally:
    #                                 self.lock.release()
                                    
    #                             # shutil.move(os.path.join(img_dir, img_name[:-3] + "src.jpg"), "tmp/")
    #                             # shutil.move(os.path.join(img_dir, img_name[:-3] + "roi.jpg"), "tmp/")

    #                             break # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    #                 # if task_queue.qsize() > 30:
    #                 #     add_task_que_flag = 0

    #             # finally:
    #             #     # 释放锁
                    
    #         try:
    #             l_frame = l_frame.next
    #         except StopIteration:
    #             break

    #     return Gst.PadProbeReturn.OK

    def nvdrmvideosink_probe(self, pad, info, u_data):        
        src_id = 0 
        frame_number=0
        gst_buffer = info.get_buffer()    
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:

            try:
                frame_meta = pyds.glist_get_nvds_frame_meta(l_frame.data)
            except StopIteration:
                break
            # src_id = frame_meta.source_id
            src_id = frame_meta.pad_index
            self.perf_data.update_fps("stream"+str(src_id))

            frame_number = frame_meta.frame_num

            # if frame_number % 2 == 0:
            n_frame_cpu = get_data_GPU(gst_buffer, frame_meta)
            img_arr = cv2.cvtColor(n_frame_cpu, cv2.COLOR_RGBA2BGRA)


            # # 保存图片
            # img_path = "/root/apps/ai_server/stream{}_frame{}.jpg".format(src_id, frame_number)
            # cv2.imwrite(img_path, img_arr)

            # if frame_number >= 1000:
            #     del_img_path = "/root/apps/ai_server/stream{}_frame{}.jpg".format(src_id, frame_number - 1000)
            #     if os.path.exists(del_img_path):
            #         os.remove(del_img_path)

            l_obj = frame_meta.obj_meta_list

            display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            display_meta.num_labels = 1
            py_nvosd_text_params = display_meta.text_params[0]
            # Setting display text to be shown on screen
            # Note that the pyds module allocates a buffer for the string, and the
            # memory will not be claimed by the garbage collector.
            # Reading the display_text field here will return the C address of the
            # allocated string. Use pyds.get_string() to get the string content.
            chanel_id = 0
            chanel_id = Plumber.GetRunningState_list[src_id][0]
            py_nvosd_text_params.display_text = Plumber.GetRunningState_list[src_id][1]

            # Now set the offsets where the string should appear
            py_nvosd_text_params.x_offset = 10
            py_nvosd_text_params.y_offset = 12

            # Font , font-color and font-size
            py_nvosd_text_params.font_params.font_name = "Serif"
            py_nvosd_text_params.font_params.font_size = 10
            # set(red, green, blue, alpha); set to White
            py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

            # Text background color
            py_nvosd_text_params.set_bg_clr = 1
            # set(red, green, blue, alpha); set to Black
            py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.7)
            # Using pyds.get_string() to get display_text as string
            # print(pyds.get_string(py_nvosd_text_params.display_text))
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

            while l_obj is not None:
                try:
                    # Casting l_obj.data to pyds.NvDsObjectMeta
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                    # # 列出对象的所有属性和方法
                    # print(dir(obj_meta))
                except StopIteration:
                    break

                # 判断对象是否已消失
                object_id = obj_meta.object_id

                if obj_meta.tracker_confidence > 0.2: # 判断是否目标存在
                    # print(f"Object ID {object_id} disappeared {frame_number} frames ago")
                    # if object_id in self.trackers.keys(): # 判断trackers是否保存先前信息
                    #     print(f"frame_number_now: {frame_number}\t\t latest frame_num: {self.trackers[object_id].frame_number}")
                    #     if frame_number - self.trackers[object_id].frame_number > 18: #判断是否达到发送要求
                    #         self.trackers[object_id].send(save=True)
                    #         self.trackers.pop(object_id)
                    obj_meta.text_params.text_bg_clr.set(0.5, 0.0, 0.5, 0.6)  # 设置显示背景颜色

                    rect_params = obj_meta.rect_params
                    top = int(rect_params.top)
                    left = int(rect_params.left)
                    width = int(rect_params.width)
                    height = int(rect_params.height)
                    class_id = obj_meta.class_id
                    conf = obj_meta.confidence
                    
                    # print(f"Tracking Object ID {object_id} at ({left}, {top}, {width}, {height})")

                    if left < 10 or top < 10 or left + width > 1920 - 10 or top + height > 1080 - 10:
                        l_obj = l_obj.next
                        continue

                    # 计算框中点
                    center_x = left + width / 2
                    center_y = top + height / 2

                    if object_id in self.trackers.keys():
                        self.trackers[object_id].update(img_arr, object_id, conf, top, left, width, height, center_x, center_y, class_id, src_id, frame_number)
                    else:
                        self.trackers[object_id] = BoxTracker(img_arr, object_id, conf, top, left, width, height, center_x, center_y, class_id, src_id, frame_number)

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
            
            n2d = []
            for key in self.trackers.keys():
                if frame_number - self.trackers[key].frame_number > 18: #判断是否达到发送要求
                    flag, msg = self.trackers[key].send(save=True)
                    # print(msg)
                    if frame_number - self.trackers[key].frame_number > 120 or flag:
                        n2d.append(key)
            for index in n2d:
                self.trackers.pop(index)

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    def stop_release_source(self, source_id):
        #Attempt to change status of source to be released 
        state_return = self.g_source_bin_list[source_id].set_state(Gst.State.NULL)

        if state_return == Gst.StateChangeReturn.SUCCESS:
            self.logging.info("LBK_STATE CHANGE SUCCESS\n")
            pad_name = "sink_%u" % source_id
            self.logging.info(pad_name)
            #Retrieve sink pad to be released
            sinkpad = self.streammux.get_static_pad(pad_name)
            #Send flush stop event to the sink pad, then release from the streammux
            sinkpad.send_event(Gst.Event.new_flush_stop(False))
            self.streammux.release_request_pad(sinkpad)

            demux_padname = "src_%u" % source_id
            demux_srcpad = self.nvstreamdemux.get_static_pad(demux_padname)
            self.nvstreamdemux.release_request_pad(demux_srcpad)

            self.logging.info("LBK_STATE CHANGE SUCCESS\n")
            #Remove the source bin from the pipeline
            self.pipeline.remove(self.g_source_bin_list[source_id])
            Plumber.num_sources -= 1
            self.pipeline.remove(self.g_sink_bin_list[source_id])
            self.pipeline.remove(self.g_queue_bin_list[source_id])
            self.pipeline.remove(self.g_convert_bin_list[source_id])
            self.pipeline.remove(self.g_filter_bin_list[source_id])

            return True

        elif state_return == Gst.StateChangeReturn.FAILURE:
            self.logging.info("LBK_STATE CHANGE FAILURE\n")
            return False
        
        elif state_return == Gst.StateChangeReturn.ASYNC:
            state_return = self.g_source_bin_list[source_id].get_state(Gst.CLOCK_TIME_NONE)
            pad_name = "sink_%u" % source_id
            self.logging.info(pad_name)
            sinkpad = self.streammux.get_static_pad(pad_name)
            # sinkpad.send_event(Gst.Event.new_flush_stop(False))
            self.streammux.release_request_pad(sinkpad)
            self.logging.info("LBK_STATE CHANGE ASYNC\n")
            self.pipeline.remove(self.g_source_bin_list[source_id])
            Plumber.num_sources -= 1
            self.pipeline.remove(self.g_sink_bin_list[source_id])
            self.pipeline.remove(self.g_queue_bin_list[source_id])
            self.pipeline.remove(self.g_convert_bin_list[source_id])
            self.pipeline.remove(self.g_filter_bin_list[source_id])
            return True

    def delete_sources(self,f):
        def wrapper(source_id):
            # f() # 无需执行
            try:
                #Disable the source
                self.g_source_enabled[source_id] = False
                #Release the source
                self.logging.info("Calling Stop %d " % source_id)
                reval = self.stop_release_source(source_id)

                if reval == 1:
                    return True
            except Exception as e:
                self.logging.info(f"error {e}")
                return False
        return wrapper 

    def make_element(self, element_name, i):
        element = Gst.ElementFactory.make(element_name, element_name)
        if not element:
            sys.stderr.write(" Unable to create {0}".format(element_name))
        element.set_property("name", "{0}-{1}".format(element_name, str(i)))
        return element

    def on_pad_added(self, src, pad, des):
        vpad = des.get_static_pad("sink")
        pad.link(vpad)


    def create_source_bin(self,index,uri):
        self.logging.info("Creating source bin")

        # Create a source GstBin to abstract this bin's content from the rest of the
        # pipeline
        self.g_source_id_list[index] = index
        bin_name="source-bin-%02d" %index
        self.logging.info(bin_name)
        nbin=Gst.Bin.new(bin_name)
        if not nbin:
            self.logging.info(" Unable to create source bin \n")
        self.pipeline.add(nbin)

        src = Gst.ElementFactory.make("rtspsrc", "src-"+str(index))
        src.set_property("location", uri)
        src.set_property("drop-on-latency", True)
        Gst.Bin.add(nbin,src)

        queuev1 = Gst.ElementFactory.make("queue2", "queue-"+str(index))
        src.connect("pad-added", self.on_pad_added, queuev1)
        Gst.Bin.add(nbin,queuev1)

        depay = Gst.ElementFactory.make("rtph265depay", "depay-"+str(index))
        Gst.Bin.add(nbin,depay)

        parse = Gst.ElementFactory.make("h265parse", "parse-"+str(index))
        Gst.Bin.add(nbin,parse)

        decode = Gst.ElementFactory.make("nvv4l2decoder", "decoder-"+str(index))
        # decode.set_property("enable-max-performance", True)
        decode.set_property("drop-frame-interval", 5)      # important!!!!!!!!!!!!!!!!!!!!!!!!!!!
        decode.set_property("num-extra-surfaces", 0)
        Gst.Bin.add(nbin,decode)

        # creating nvvidconv
        nvvideoconvert = self.make_element("nvvideoconvert", index)
        Gst.Bin.add(nbin,nvvideoconvert)

        caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
        filter = self.make_element("capsfilter", index)
        filter.set_property("caps", caps1)
        Gst.Bin.add(nbin,filter)

        nbin.add_pad(Gst.GhostPad.new_no_target("src",Gst.PadDirection.SRC))

        queuev1.link(depay)
        depay.link(parse)
        parse.link(decode)
        decode.link(nvvideoconvert)
        nvvideoconvert.link(filter)

        # decoder_src_pad = decode.get_static_pad("src")
        filter_src_pad = filter.get_static_pad("src")
        bin_ghost_pad=nbin.get_static_pad("src")
        # bin_ghost_pad.set_target(decoder_src_pad)
        bin_ghost_pad.set_target(filter_src_pad)
        sinkpad = self.streammux.get_request_pad("sink_%u" % index)
        # sinkpad = drmsink.get_static_pad("sink")
        bin_ghost_pad.link(sinkpad)

        self.g_source_enabled[index] = True

        return nbin

    def add_sources(self, f):
        def wrapper(data):
            # f() #无需执行
            try:
                self.pipeline.set_state(Gst.State.PAUSED)
                time.sleep(0.5)
                # self.nvstreamdemux.set_state(Gst.State.NULL)
                # time.sleep(0.5)
                # streammux.set_state(Gst.State.NULL)
                # time.sleep(0.5)
                # streammux.set_property("batch-size", g_num_sources+1)

                source_id = data[0]    

                #Enable the source
                self.g_source_enabled[source_id] = True

                self.logging.info("Calling Start %d " % source_id)

                #Create a uridecode bin with the chosen source id
                source_bin = self.create_source_bin(source_id, data[1])

                if (not source_bin):
                    sys.stderr.write("Failed to create source bin. Exiting.")
                    exit(1)
                
                #Add source bin to our list and to pipeline
                self.g_source_bin_list[source_id] = source_bin

                # pipeline nvstreamdemux -> queue -> nvvidconv -> nvosd -> (if Jetson) nvegltransform -> nveglgl
                # Creating EGLsink
                if True:
                    self.logging.info("Creating fakesink \n")
                    # sink = make_element("nv3dsink", source_id)
                    sink = self.make_element("fakesink", source_id)
                    if not sink:
                        sys.stderr.write(" Unable to create nv3dsink \n")
                    sink.set_property('enable-last-sample', 0)
                    sink.set_property('sync', 0)
                    
                self.pipeline.add(sink)
                self.g_sink_bin_list[source_id] = sink

                # creating queue
                queue = self.make_element("queue", source_id)

                self.pipeline.add(queue)
                self.g_queue_bin_list[source_id] = queue

                # creating nvosd
                # nvdsosd = make_element("nvdsosd", source_id)
                # pipeline.add(nvdsosd)

                # connect nvstreamdemux -> queue
                # padname = "src_%u" % source_id
                # demuxsrcpad = self.nvstreamdemux.get_request_pad(padname)
                # if not demuxsrcpad:
                #     sys.stderr.write("Unable to create demux src pad \n")

                # queuesinkpad = queue.get_static_pad("sink")
                # if not queuesinkpad:
                #     sys.stderr.write("Unable to create queue sink pad \n")
                # demuxsrcpad.link(queuesinkpad)


                # connect  queue -> nvvidconv -> nvosd -> nveglgl
                # queue.link(nvvideoconvert)
                # nvvideoconvert.link(filter)
                # filter.link(sink)
                # queue.link(sink)

                # sink.set_property("sync", 0)
                # sink.set_property("qos", 0)
                    
                # sink_sinkpad = sink.get_static_pad("sink")
                # if not sink_sinkpad:
                #     sys.stderr.write(" Unable to get sink pad of nvosd \n")
                # sink_sinkpad.add_probe(Gst.PadProbeType.BUFFER, self.tiler_sink_pad_buffer_probe, 0)

                #Set state of source bin to playing
                state_return = self.g_source_bin_list[source_id].set_state(Gst.State.PLAYING)

                if state_return == Gst.StateChangeReturn.SUCCESS:
                    self.logging.info("STATE CHANGE SUCCESS\n")
                    Plumber.num_sources += 1

                elif state_return == Gst.StateChangeReturn.FAILURE:
                    self.logging.info("STATE CHANGE FAILURE\n")
                
                elif state_return == Gst.StateChangeReturn.ASYNC:
                    state_return = self.g_source_bin_list[source_id].get_state(Gst.CLOCK_TIME_NONE)
                    Plumber.num_sources += 1

                elif state_return == Gst.StateChangeReturn.NO_PREROLL:
                    self.logging.info("STATE CHANGE NO PREROLL\n")
                    Plumber.num_sources += 1

                self.pipeline.set_state(Gst.State.PLAYING)
                
                return True
            except Exception as e:
                self.logging.info(f"error:{e}")
                return False
            
        return wrapper

    def add_task(self, f):
        def wrapper(args):
            service_id = args[0]
            dev_id = args[1]
            root = args[2]
                
            self.logging.info(f"Creating streammux_{service_id}\n ")
            # Create nvstreammux instance to form batches from one or more sources.
            streammux = Gst.ElementFactory.make("nvstreammux", f"Stream-muxer-{service_id}")
            self.streammux_dict[service_id] = streammux
            if not streammux:
                sys.stderr.write(" Unable to create NvStreamMux \n")

            streammux.set_property("batched-push-timeout", 200000)                   
            streammux.set_property("batch-size", 16)           # important!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            streammux.set_property("gpu_id", dev_id)

            self.pipeline.add(streammux)
            streammux.set_property("live-source", 1)

            self.logging.info("Creating Pgie \n ")
            pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
            if not pgie:
                sys.stderr.write(" Unable to create pgie \n")

            self.logging.info("Creating nvtracker \n ")
            tracker = Gst.ElementFactory.make("nvtracker", "tracker")
            if not tracker:
                sys.stderr.write(" Unable to create tracker \n")

            tracker_srcpad = tracker.get_static_pad("src")
            if not tracker_srcpad:
                sys.stderr.write(" Unable to get sink pad of nvosd \n")
            tracker_srcpad.add_probe(Gst.PadProbeType.BUFFER, self.nvdrmvideosink_probe, 0)

            self.logging.info("Creating nvstreamdemux \n ")
            self.nvstreamdemux = Gst.ElementFactory.make("nvstreamdemux", "nvstreamdemux")
            if not self.nvstreamdemux:
                sys.stderr.write(" Unable to create nvstreamdemux \n")

            self.logging.info("Creating tee \n ")
            tee = Gst.ElementFactory.make("tee", "tee")
            if not tee:
                sys.stderr.write(" Unable to create tee \n")

            tee_q0 = Gst.ElementFactory.make("queue", "tee_q0")
            # tee_q1 = Gst.ElementFactory.make("queue", "tee_q1")

            self.logging.info("Creating tiler \n ")
            tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
            tiler.set_property('rows', self.tiler_size["tiler_row"])
            tiler.set_property('columns', self.tiler_size["tiler_col"])
            tiler.set_property('width', self.tiler_size["tiler_w"])
            tiler.set_property('height', self.tiler_size["tiler_h"])
            if not tiler:
                sys.stderr.write(" Unable to create tiler \n")

            self.logging.info("Creating nvosd \n ")
            nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
            if not nvosd:
                sys.stderr.write(" Unable to create nvosd \n")

            # creating nvvidconv
            out_videoconvert = Gst.ElementFactory.make("nvvideoconvert", "out_videoconvert")

            out_caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), width=1920, height=1080")
            out_filter = Gst.ElementFactory.make("capsfilter", "capsfilter")
            out_filter.set_property("caps", out_caps)


            self.logging.info("Creating sink \n ")
            # drmsink = Gst.ElementFactory.make("nvdrmvideosink", "drmsink")
            # # drmsink = Gst.ElementFactory.make("nv3dsink", "drmsink")
            # drmsink.set_property('enable-last-sample', 0)
            # drmsink.set_property('sync', 0)
            # if not drmsink:
            #     sys.stderr.write(" Unable to create drmsink \n")

            sink = Gst.ElementFactory.make("fakesink", "sink")
            sink.set_property("signal-handoffs", True)
            sink.set_property("silent", False)
                
            # 定义handoff信号的回调函数
            def on_handoff(fakesink, buffer, pad):
                pass
                # print(f"Buffer received with PTS: {buffer.pts}")

            # 连接handoff信号
            sink.connect("handoff", on_handoff)


            streammux.set_property('live-source', 1)
            #Set streammux width and height
            streammux.set_property('width', MUXER_OUTPUT_WIDTH)
            streammux.set_property('height', MUXER_OUTPUT_HEIGHT)
            #Set pgie configuration file paths
            pgie.set_property('config-file-path', PGIE_CONFIG_FILE)

            #Set properties of tracker
            config = configparser.ConfigParser()
            config.read(TRACKER_CONFIG_FILE)
            config.sections()

            # 设置保留帧数
            # tracker.set_property("max-shadow-tracking-duration", 20)
            for key in config['tracker']:
                if key == 'tracker-width' :
                    tracker_width = config.getint('tracker', key)
                    tracker.set_property('tracker-width', tracker_width)
                if key == 'tracker-height' :
                    tracker_height = config.getint('tracker', key)
                    tracker.set_property('tracker-height', tracker_height)
                if key == 'gpu-id' :
                    tracker_gpu_id = config.getint('tracker', key)
                    tracker.set_property('gpu_id', tracker_gpu_id)
                if key == 'll-lib-file' :
                    tracker_ll_lib_file = config.get('tracker', key)
                    tracker.set_property('ll-lib-file', tracker_ll_lib_file)
                if key == 'll-config-file' :
                    tracker_ll_config_file = config.get('tracker', key)
                    tracker.set_property('ll-config-file', tracker_ll_config_file)
                if key == 'enable-batch-process' :
                    tracker_enable_batch_process = config.getint('tracker', key)
                    tracker.set_property('enable_batch_process', tracker_enable_batch_process)

            #Set necessary properties of the nvinfer element, the necessary ones are:
            pgie_batch_size=pgie.get_property("batch-size")
            if(pgie_batch_size < MAX_NUM_SOURCES):
                self.logging.info("WARNING: Overriding infer-config batch-size",pgie_batch_size," with number of sources ", Plumber.num_sources," \n")
            pgie.set_property("batch-size",MAX_NUM_SOURCES)

            self.logging.info("Adding elements to Pipeline \n")
            self.pipeline.add(pgie)
            self.pipeline.add(tracker)
            self.pipeline.add(self.nvstreamdemux)
            self.pipeline.add(tee)
            self.pipeline.add(tee_q0)
            # self.pipeline.add(tee_q1)
            self.pipeline.add(tiler)
            self.pipeline.add(nvosd)
            self.pipeline.add(out_videoconvert)
            self.pipeline.add(out_filter)
            self.pipeline.add(sink)

            self.logging.info("Linking elements in the Pipeline \n")
            streammux.link(pgie)
            pgie.link(tracker)
            tracker.link(tee)

            tee.link(tee_q0)
            tee_q0.link(tiler)

            # tee.link(tee_q1)
            tee.link(self.nvstreamdemux)

            
            tiler.link(nvosd)
            nvosd.link(sink)
            # out_videoconvert.link(out_filter)
            # out_filter.link(drmsink)
        return wrapper

    def bus_call(self, bus, message, loop):
        # global g_eos_list
        t = message.type
        if t == Gst.MessageType.EOS:
            sys.stdout.write("End-of-stream\n")
            loop.quit()
        elif t==Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            sys.stderr.write("Warning: %s: %s\n" % (err, debug))
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            sys.stderr.write("Error: %s: %s\n" % (err, debug))
            loop.quit()
        elif t == Gst.MessageType.ELEMENT:
            struct = message.get_structure()
            #Check for stream-eos message
            if struct is not None and struct.has_name("stream-eos"):
                parsed, stream_id = struct.get_uint("stream-id")
                if parsed:
                    #Set eos status of stream to True, to be deleted in delete-sources
                    self.logging.info("Got EOS from stream %d" % stream_id)
                    self.g_eos_list[stream_id] = True
        return True

    def run(self,):
        # os.system("rm /root/apps/ai_server/*.jpg")

        # Standard GStreamer initialization
        Gst.init(None)

        # Create gstreamer elements */
        # Create Pipeline element that will form a connection of other elements
        self.logging.info("Creating Pipeline \n ")
        self.pipeline = Gst.Pipeline()

        if not self.pipeline:
            sys.stderr.write(" Unable to create Pipeline \n")



        "=============================================================="
        "=============================================================="


        self.logging.info("Creating streammux \n ")
        # Create nvstreammux instance to form batches from one or more sources.
        self.streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
        if not self.streammux:
            sys.stderr.write(" Unable to create NvStreamMux \n")

        self.streammux.set_property("batched-push-timeout", 200000)                   
        self.streammux.set_property("batch-size", 16)           # important!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.streammux.set_property("gpu_id", GPU_ID)

        self.pipeline.add(self.streammux)
        self.streammux.set_property("live-source", 1)

        self.logging.info("Creating Pgie \n ")
        pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
        if not pgie:
            sys.stderr.write(" Unable to create pgie \n")

        self.logging.info("Creating nvtracker \n ")
        tracker = Gst.ElementFactory.make("nvtracker", "tracker")
        if not tracker:
            sys.stderr.write(" Unable to create tracker \n")

        tracker_srcpad = tracker.get_static_pad("src")
        if not tracker_srcpad:
            sys.stderr.write(" Unable to get sink pad of nvosd \n")
        tracker_srcpad.add_probe(Gst.PadProbeType.BUFFER, self.nvdrmvideosink_probe, 0)

        # self.logging.info("Creating nvstreamdemux \n ")
        # self.nvstreamdemux = Gst.ElementFactory.make("nvstreamdemux", "nvstreamdemux")
        # if not self.nvstreamdemux:
        #     sys.stderr.write(" Unable to create nvstreamdemux \n")

        self.logging.info("Creating tee \n ")
        tee = Gst.ElementFactory.make("tee", "tee")
        if not tee:
            sys.stderr.write(" Unable to create tee \n")

        tee_q0 = Gst.ElementFactory.make("queue", "tee_q0")
        # tee_q1 = Gst.ElementFactory.make("queue", "tee_q1")

        self.logging.info("Creating tiler \n ")
        tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
        tiler.set_property('rows', self.tiler_size["tiler_row"])
        tiler.set_property('columns', self.tiler_size["tiler_col"])
        tiler.set_property('width', self.tiler_size["tiler_w"])
        tiler.set_property('height', self.tiler_size["tiler_h"])
        if not tiler:
            sys.stderr.write(" Unable to create tiler \n")

        self.logging.info("Creating nvosd \n ")
        nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        if not nvosd:
            sys.stderr.write(" Unable to create nvosd \n")

        # creating nvvidconv
        out_videoconvert = Gst.ElementFactory.make("nvvideoconvert", "out_videoconvert")

        out_caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), width=1920, height=1080")
        out_filter = Gst.ElementFactory.make("capsfilter", "capsfilter")
        out_filter.set_property("caps", out_caps)


        self.logging.info("Creating sink \n ")
        # drmsink = Gst.ElementFactory.make("nvdrmvideosink", "drmsink")
        # # drmsink = Gst.ElementFactory.make("nv3dsink", "drmsink")
        # drmsink.set_property('enable-last-sample', 0)
        # drmsink.set_property('sync', 0)
        # if not drmsink:
        #     sys.stderr.write(" Unable to create drmsink \n")

        sink = Gst.ElementFactory.make("fakesink", "sink")
        sink.set_property("signal-handoffs", True)
        sink.set_property("silent", False)
        
        # 定义handoff信号的回调函数
        def on_handoff(fakesink, buffer, pad):
            pass
            # print(f"Buffer received with PTS: {buffer.pts}")

        # 连接handoff信号
        sink.connect("handoff", on_handoff)


        self.streammux.set_property('live-source', 1)
        #Set streammux width and height
        self.streammux.set_property('width', MUXER_OUTPUT_WIDTH)
        self.streammux.set_property('height', MUXER_OUTPUT_HEIGHT)
        #Set pgie configuration file paths
        pgie.set_property('config-file-path', PGIE_CONFIG_FILE)

        #Set properties of tracker
        config = configparser.ConfigParser()
        config.read(TRACKER_CONFIG_FILE)
        config.sections()

        # 设置保留帧数
        # tracker.set_property("max-shadow-tracking-duration", 20)
        for key in config['tracker']:
            if key == 'tracker-width' :
                tracker_width = config.getint('tracker', key)
                tracker.set_property('tracker-width', tracker_width)
            if key == 'tracker-height' :
                tracker_height = config.getint('tracker', key)
                tracker.set_property('tracker-height', tracker_height)
            if key == 'gpu-id' :
                tracker_gpu_id = config.getint('tracker', key)
                tracker.set_property('gpu_id', tracker_gpu_id)
            if key == 'll-lib-file' :
                tracker_ll_lib_file = config.get('tracker', key)
                tracker.set_property('ll-lib-file', tracker_ll_lib_file)
            if key == 'll-config-file' :
                tracker_ll_config_file = config.get('tracker', key)
                tracker.set_property('ll-config-file', tracker_ll_config_file)
            if key == 'enable-batch-process' :
                tracker_enable_batch_process = config.getint('tracker', key)
                tracker.set_property('enable_batch_process', tracker_enable_batch_process)

        #Set necessary properties of the nvinfer element, the necessary ones are:
        pgie_batch_size=pgie.get_property("batch-size")
        if(pgie_batch_size < MAX_NUM_SOURCES):
            self.logging.info("WARNING: Overriding infer-config batch-size",pgie_batch_size," with number of sources ", Plumber.num_sources," \n")
        pgie.set_property("batch-size",MAX_NUM_SOURCES)

        self.logging.info("Adding elements to Pipeline \n")
        self.pipeline.add(pgie)
        self.pipeline.add(tracker)
        # self.pipeline.add(self.nvstreamdemux)
        self.pipeline.add(tee)
        self.pipeline.add(tee_q0)
        # self.pipeline.add(tee_q1)
        self.pipeline.add(tiler)
        self.pipeline.add(nvosd)
        self.pipeline.add(out_videoconvert)
        self.pipeline.add(out_filter)
        self.pipeline.add(sink)

        self.logging.info("Linking elements in the Pipeline \n")
        self.streammux.link(pgie)
        pgie.link(tracker)
        tracker.link(tee)

        tee.link(tiler)
        # tee_q0.link(tiler)

        # tee.link(tee_q1)
        # tee.link(self.nvstreamdemux)

        
        tiler.link(nvosd)
        nvosd.link(sink)
        # out_videoconvert.link(out_filter)
        # out_filter.link(drmsink)
        
        
        "=============================================================="
        "=============================================================="

        # perf callback function to print fps every 5 sec
        GLib.timeout_add(5000, self.perf_data.perf_print_callback)

        # filter1.link(nvstreamdemux)

        # create an event loop and feed gstreamer bus mesages to it
        loop = GLib.MainLoop()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect ("message", self.bus_call, loop)    

        self.pipeline.set_state(Gst.State.PAUSED)

        self.logging.info("Starting pipeline \n")
        if Plumber.stand_by == False:
            Plumber.stand_by = True
        # start play back and listed to events		
        self.pipeline.set_state(Gst.State.PLAYING)

        try:
            loop.run()
        # except:
        #     # 继续运行
        #     pass

        except Exception as error:
            print("error:", error)
            pass

        # cleanup
        self.logging.info("Exiting app\n")
        self.pipeline.set_state(Gst.State.NULL)