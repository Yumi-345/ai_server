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


def check_pipeline_elements(pipeline):
    for element in pipeline.iterate_elements():
        state_change_return, current, pending = element.get_state(1 * Gst.SECOND)
        print(f"Element {element.get_name()} state: {current}, Pending state: {pending}")
        if state_change_return == Gst.StateChangeReturn.FAILURE:
            print(f"Element {element.get_name()} failed to change state.")
            return False
    return True


def stop_pipeline(pipeline, timeout=5):
    if pipeline:
        # pipeline.set_state(Gst.State.PAUSED)
        # while True:
        #     state_change_return, current, pending = pipeline.get_state(0.1 * Gst.SECOND)
        #     # print(f"Current pipeline state: {current}, Pending state: {pending}")
        #     if current == Gst.State.PAUSED:
        #         print("Pipeline paused successfully.")
        #         break
        #     if time.time() - start_time > timeout:
        #         print("Failed to pause pipeline within the timeout period.")
        #         break

        pipeline.set_state(Gst.State.NULL)
        start_time = time.time()
        while True:
            state_change_return, current, pending = pipeline.get_state(0.1 * Gst.SECOND)
            # print(f"Current pipeline state: {current}, Pending state: {pending}")
            if current == Gst.State.NULL:
                print("Pipeline stopped successfully.")
                break
            if time.time() - start_time > timeout:
                print("Failed to stop pipeline within the timeout period.")
                break

# Function to start the pipeline
def start_pipeline(pipeline, timeout=5):
    if pipeline:
        pipeline.set_state(Gst.State.PLAYING)
        start_time = time.time()
        while True:
            state_change_return, current, pending = pipeline.get_state(0.1 * Gst.SECOND)
            # print(f"Current pipeline state: {current}, Pending state: {pending}")
            if current == Gst.State.PLAYING:
                print("Pipeline started successfully.")
                # check_pipeline_elements(pipeline)
                return True
            if time.time() - start_time > timeout:
                print("Failed to start pipeline within the timeout period.")
                if not check_pipeline_elements(pipeline):
                    print("One or more elements failed to change state.")
                break
        return False


def remove_element_from_pipeline(pipeline, element, timeout=5):
    if pipeline and element:
        pipeline.remove(element)
        element.set_state(Gst.State.NULL)
        start_time = time.time()
        while True:
            state_change_return, current, pending = element.get_state(1 * Gst.SECOND)
            print(f"Current element state: {current}, Pending state: {pending}")
            if current == Gst.State.NULL:
                print(f"Element {element.get_name()} removed and set to NULL state.")
                break
            if time.time() - start_time > timeout:
                print(f"Failed to set element {element.get_name()} to NULL state within the timeout period.")
                break


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
    task_list = []
    channel_list = []
    task_channel_dict = {}

    num_sources = 0
    # task_queue = queue.Queue()
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
        self.tees = {}

        # self.add_task_/que_flag = 1

        self.trackers_lock = threading.Lock()

        self.perf_data = PERF_DATA(8)
        self.tiler_size = json.load(open("/root/apps/ai_server/cfg/config.json", 'r')) 
        t = threading.Thread(target=self.run, args=())
        t.start()

    def nvdrmvideosink_probe(self, pad, info, u_data):
        # 通过u_data实现不同task的跟踪，避免跟踪混乱
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

            n_frame_cpu = get_data_GPU(gst_buffer, frame_meta)
            img_arr = cv2.cvtColor(n_frame_cpu, cv2.COLOR_RGBA2BGRA)

            l_obj = frame_meta.obj_meta_list

            while l_obj is not None:
                try:
                    # Casting l_obj.data to pyds.NvDsObjectMeta
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                    # # 列出对象的所有属性和方法
                    # print(dir(obj_meta))
                except StopIteration:
                    break

                object_id = obj_meta.object_id
                u_id = f"{u_data}_{src_id}_{object_id}"

                if obj_meta.tracker_confidence > 0.2: # 判断是否目标存在
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
                    try:
                        self.trackers_lock.acquire()
                        if u_id in self.trackers.keys():
                            self.trackers[u_id].update(img_arr, u_id, conf, top, left, width, height, center_x, center_y, class_id, src_id, frame_number)
                        else:
                            self.trackers[u_id] = BoxTracker(img_arr, u_id, conf, top, left, width, height, center_x, center_y, class_id, src_id, frame_number)
                    except Exception as e:
                        print(f"error in update trackers: {e}")
                    finally:
                        self.trackers_lock.release()
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
            try:
                self.trackers_lock.acquire()
                n2d = []
                for key in self.trackers.keys():
                    if frame_number - self.trackers[key].frame_number > 18: #判断是否达到发送要求
                        flag, msg = self.trackers[key].send(save=True)
                        # print(msg)
                        if frame_number - self.trackers[key].frame_number > 120 or flag:
                            n2d.append(key)
                for index in n2d:
                    self.trackers.pop(index)
            except Exception as e:
                print(f"error in pop tracker: {e}")
            finally:
                self.trackers_lock.release()

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    def make_element(self, element_name, i):
        element = Gst.ElementFactory.make(element_name, element_name)
        if not element:
            sys.stderr.write(" Unable to create {0}".format(element_name))
        element.set_property("name", "{0}-{1}".format(element_name, str(i)))
        return element

    def on_pad_added(self, src, pad, des):
        vpad = des.get_static_pad("sink")
        pad.link(vpad)

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

    def stop_release_source(self, source_id):
        #Attempt to change status of source to be released 
        state_return = self.g_source_bin_list[source_id].set_state(Gst.State.NULL)
        stop_pipeline(self.pipeline)

        if state_return == Gst.StateChangeReturn.SUCCESS:
            self.logging.info("LBK_STATE CHANGE SUCCESS\n")
            # pad_name = "sink_%u" % source_id
            # self.logging.info(pad_name)
            # #Retrieve sink pad to be released
            # sinkpad = self.streammux.get_static_pad(pad_name)
            # #Send flush stop event to the sink pad, then release from the streammux
            # sinkpad.send_event(Gst.Event.new_flush_stop(False))
            # self.streammux.release_request_pad(sinkpad)

            # demux_padname = "src_%u" % source_id
            # demux_srcpad = self.nvstreamdemux.get_static_pad(demux_padname)
            # self.nvstreamdemux.release_request_pad(demux_srcpad)

            self.logging.info("LBK_STATE CHANGE SUCCESS\n")
            #Remove the source bin from the pipeline
            self.pipeline.remove(self.g_source_bin_list[source_id])
            # Plumber.num_sources -= 1
            # self.pipeline.remove(self.g_sink_bin_list[source_id])
            # self.pipeline.remove(self.g_queue_bin_list[source_id])
            # self.pipeline.remove(self.g_convert_bin_list[source_id])
            # self.pipeline.remove(self.g_filter_bin_list[source_id])

            self.pipeline.remove(self.pipeline.get_by_name(f"tee_{source_id}"))
            # self.pipeline.remove(self.pipeline.get_by_name(f"queue_channel_{source_id}"))
            # self.pipeline.remove(self.pipeline.get_by_name(f"fakesink_channel_{source_id}"))
            for service_id in Plumber.task_channel_dict.keys():
                if source_id in Plumber.task_channel_dict[service_id]:
                    self.pipeline.remove(self.pipeline.get_by_name(f"channel{source_id}_task{service_id}"))
                    self.streammuxs[service_id].release_request_pad(self.streammuxs[service_id].get_static_pad(f"sink_{source_id}"))
                    Plumber.task_channel_dict[service_id].remove(source_id)
            try:
                self.trackers_lock.acquire()
                self.trackers.clear()
            except Exception as e:
                print(f"error in stop src pop tracker: {e}")
            finally:
                self.trackers_lock.release()
            start_pipeline(self.pipeline)
            return True

        elif state_return == Gst.StateChangeReturn.FAILURE:
            self.logging.info("LBK_STATE CHANGE FAILURE\n")
            return False
        
        elif state_return == Gst.StateChangeReturn.ASYNC:
            # state_return = self.g_source_bin_list[source_id].get_state(Gst.CLOCK_TIME_NONE)
            # pad_name = "sink_%u" % source_id
            # self.logging.info(pad_name)
            # sinkpad = self.streammux.get_static_pad(pad_name)
            # # sinkpad.send_event(Gst.Event.new_flush_stop(False))
            # self.streammux.release_request_pad(sinkpad)
            self.logging.info("LBK_STATE CHANGE ASYNC\n")
            self.pipeline.remove(self.g_source_bin_list[source_id])
            # Plumber.num_sources -= 1
            # self.pipeline.remove(self.g_sink_bin_list[source_id])
            # self.pipeline.remove(self.g_queue_bin_list[source_id])
            # self.pipeline.remove(self.g_convert_bin_list[source_id])
            # self.pipeline.remove(self.g_filter_bin_list[source_id])

            self.pipeline.remove(self.pipeline.get_by_name(f"tee_{source_id}"))
            # self.pipeline.remove(self.pipeline.get_by_name(f"queue_channel_{source_id}"))
            # self.pipeline.remove(self.pipeline.get_by_name(f"fakesink_channel_{source_id}"))
            for service_id in Plumber.task_channel_dict.keys():
                if source_id in Plumber.task_channel_dict[service_id]:
                    self.pipeline.remove(self.pipeline.get_by_name(f"channel{source_id}_task{service_id}"))
                    self.streammuxs[service_id].release_request_pad(self.streammuxs[service_id].get_static_pad(f"sink_{source_id}"))
                    Plumber.task_channel_dict[service_id].remove(source_id)
        
            try:
                self.trackers_lock.acquire()
                self.trackers.clear()
            except Exception as e:
                print(f"error in stop src pop tracker: {e}")
            finally:
                self.trackers_lock.release()
            start_pipeline(self.pipeline)

            return True

    def create_source_bin(self,source_id,source_url):
        self.logging.info("Creating source bin")

        # Create a source GstBin to abstract this bin's content from the rest of the
        # pipeline
        self.g_source_id_list[source_id] = source_id
        bin_name="source-bin-%02d" %source_id
        self.logging.info(bin_name)
        nbin=Gst.Bin.new(bin_name)
        if not nbin:
            self.logging.info(" Unable to create source bin \n")
        self.pipeline.add(nbin)

        src = Gst.ElementFactory.make("rtspsrc", "src-"+str(source_id))
        src.set_property("location", source_url)
        src.set_property("drop-on-latency", True)
        Gst.Bin.add(nbin,src)

        queuev1 = Gst.ElementFactory.make("queue2", "queue-"+str(source_id))
        src.connect("pad-added", self.on_pad_added, queuev1)
        Gst.Bin.add(nbin,queuev1)

        depay = Gst.ElementFactory.make("rtph265depay", "depay-"+str(source_id))
        Gst.Bin.add(nbin,depay)

        parse = Gst.ElementFactory.make("h265parse", "parse-"+str(source_id))
        Gst.Bin.add(nbin,parse)

        decode = Gst.ElementFactory.make("nvv4l2decoder", "decoder-"+str(source_id))
        # decode.set_property("enable-max-performance", True)
        decode.set_property("drop-frame-interval", 5)      # important!!!!!!!!!!!!!!!!!!!!!!!!!!!
        decode.set_property("num-extra-surfaces", 0)
        Gst.Bin.add(nbin,decode)

        # creating nvvidconv
        nvvideoconvert = self.make_element("nvvideoconvert", source_id)
        Gst.Bin.add(nbin,nvvideoconvert)

        caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
        filter = self.make_element("capsfilter", source_id)
        filter.set_property("caps", caps1)
        Gst.Bin.add(nbin,filter)

        tee = Gst.ElementFactory.make("tee", f"tee_{source_id}")
        self.pipeline.add(tee) # 特别注意这里，因为tee已经需要放在bin的外部了，所以这里是添加到pipeline里边而不是nbin
        # Gst.Bin.add(nbin,tee)

        # queue = Gst.ElementFactory.make("queue", f"queue_channel_{source_id}")
        # self.pipeline.add(queue)

        # sink = Gst.ElementFactory.make("fakesink", f"fakesink_channel_{source_id}")
        # sink.set_property("signal-handoffs", True)
        # sink.set_property("sync", False)
        # self.pipeline.add(sink)


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

        # sinkpad = self.streammux.get_request_pad("sink_%u" % source_id)
        # # sinkpad = drmsink.get_static_pad("sink")
        # bin_ghost_pad.link(sinkpad)

        sinkpad = tee.get_static_pad("sink")
        bin_ghost_pad.link(sinkpad)
        
        
        # bin_ghost_pad.link(tee)

        # tee.link(queue)
        # queue.link(sink)

        self.g_source_enabled[source_id] = True
        self.tees[source_id] = tee

        return nbin


    def add_task(self, f):
        def wrapper(*args):
            try:
                    
                service_id = args[0]
                dev_id = args[1]
                root = args[2]

                self.pipeline.set_state(Gst.State.PAUSED)
                # stop_pipeline(self.pipeline)
                time.sleep(0.5)

                self.logging.info(f"Creating streammux_{service_id}\n ")
                # Create nvstreammux instance to form batches from one or more sources.
                streammux = Gst.ElementFactory.make("nvstreammux", f"Stream-muxer-{service_id}")
                self.streammuxs[service_id] = streammux
                if not streammux:
                    sys.stderr.write(" Unable to create NvStreamMux \n")

                streammux.set_property("batched-push-timeout", 200000)                   
                streammux.set_property("batch-size", 16)           # important!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                streammux.set_property("gpu_id", dev_id)

                self.pipeline.add(streammux)
                streammux.set_property("live-source", 1)

                self.logging.info("Creating Pgie \n ")
                pgie = Gst.ElementFactory.make("nvinfer", f"primary_inference{service_id}")
                if not pgie:
                    sys.stderr.write(" Unable to create pgie \n")

                self.logging.info("Creating nvtracker \n ")
                tracker = Gst.ElementFactory.make("nvtracker", f"tracker{service_id}")
                if not tracker:
                    sys.stderr.write(" Unable to create tracker \n")

                tracker_srcpad = tracker.get_static_pad("src")
                if not tracker_srcpad:
                    sys.stderr.write(" Unable to get sink pad of nvosd \n")
                tracker_srcpad.add_probe(Gst.PadProbeType.BUFFER, self.nvdrmvideosink_probe, service_id)

                self.logging.info("Creating sink \n ")
                # drmsink = Gst.ElementFactory.make("nvdrmvideosink", "drmsink")
                # # drmsink = Gst.ElementFactory.make("nv3dsink", "drmsink")
                # drmsink.set_property('enable-last-sample', 0)
                # drmsink.set_property('sync', 0)
                # if not drmsink:
                #     sys.stderr.write(" Unable to create drmsink \n")

                sink = Gst.ElementFactory.make("fakesink", f"task_fakesink_{service_id}")
                sink.set_property("signal-handoffs", True)
                sink.set_property("sync", False)
                # sink.set_property("silent", False)
                    
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
                self.pipeline.add(sink)

                self.logging.info("Linking elements in the Pipeline \n")
                streammux.link(pgie)
                pgie.link(tracker)
                tracker.link(sink)

                self.pipeline.set_state(Gst.State.PLAYING)
                # start_pipeline(self.pipeline)
                # time.slppe(0.5)
                return True 
            except Exception as e:
                self.logging.info(f"error in add task {service_id}:{e}")
                self.pipeline.set_state(Gst.State.PLAYING)
                # start_pipeline(self.pipeline)
                return False
        return wrapper

    def remove_task(self, f):
        def wrapper(*args):
            service_id = args[0]
            try:
                # self.pipeline.set_state(Gst.State.PAUSED)
                stop_pipeline(self.pipeline)
                # time.sleep(0.5)
                self.pipeline.remove(self.pipeline.get_by_name(f"Stream-muxer-{service_id}"))
                self.pipeline.remove(self.pipeline.get_by_name(f"primary_inference{service_id}"))
                self.pipeline.remove(self.pipeline.get_by_name(f"tracker{service_id}"))
                self.pipeline.remove(self.pipeline.get_by_name(f"task_fakesink_{service_id}"))
                for channel_id in self.task_channel_dict[service_id]:
                    self.pipeline.remove(self.pipeline.get_by_name(f"channel{channel_id}_task{service_id}"))
                    self.tees[channel_id].release_request_pad(self.tees[channel_id].get_static_pad(f"src_{service_id+1}"))
                
                try:
                    self.trackers_lock.acquire()
                    self.trackers.clear()
                except Exception as e:
                    print(f"error in remove task pop tracker: {e}")
                finally:
                    self.trackers_lock.release()

                # self.pipeline.set_state(Gst.State.PLAYING)
                start_pipeline(self.pipeline)
                return True
            except Exception as e:
                self.logging.info(f"error in remove tasks: {e}")
                # self.pipeline.set_state(Gst.State.PLAYING)
                start_pipeline(self.pipeline)
                return False
        return wrapper

    def enable_channel(self, f):
        def wrapper(*args):
            channel_id = args[0]
            channel_name = args[1]
            channel_url = args[2]
            try:
                self.pipeline.set_state(Gst.State.PAUSED)
                # stop_pipeline(self.pipeline)
                time.sleep(0.5)

                #Enable the source
                self.g_source_enabled[channel_id] = True

                self.logging.info("Calling Start %d " % channel_id)

                #Create a uridecode bin with the chosen source id
                source_bin = self.create_source_bin(channel_id, channel_url)

                if (not source_bin):
                    sys.stderr.write("Failed to create source bin. Exiting.")
                    exit(1)
                
                #Add source bin to our list and to pipeline
                self.g_source_bin_list[channel_id] = source_bin

                # pipeline nvstreamdemux -> queue -> nvvidconv -> nvosd -> (if Jetson) nvegltransform -> nveglgl
                # Creating EGLsink
                # if True:
                #     self.logging.info("Creating fakesink \n")
                #     # sink = make_element("nv3dsink", source_id)
                #     sink = self.make_element("fakesink", channel_id)
                #     if not sink:
                #         sys.stderr.write(" Unable to create nv3dsink \n")
                #     sink.set_property('enable-last-sample', 0)
                #     sink.set_property('sync', 0)
                    
                # self.pipeline.add(sink)
                # self.g_sink_bin_list[channel_id] = sink

                # creating queue
                # queue = self.make_element("queue", channel_id)

                # self.pipeline.add(queue)
                # self.g_queue_bin_list[channel_id] = queue

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
                state_return = self.g_source_bin_list[channel_id].set_state(Gst.State.PLAYING)

                if state_return == Gst.StateChangeReturn.SUCCESS:
                    self.logging.info("STATE CHANGE SUCCESS\n")
                    Plumber.num_sources += 1

                elif state_return == Gst.StateChangeReturn.FAILURE:
                    self.logging.info("STATE CHANGE FAILURE\n")
                
                elif state_return == Gst.StateChangeReturn.ASYNC:
                    state_return = self.g_source_bin_list[channel_id].get_state(Gst.CLOCK_TIME_NONE)
                    Plumber.num_sources += 1

                elif state_return == Gst.StateChangeReturn.NO_PREROLL:
                    self.logging.info("STATE CHANGE NO PREROLL\n")
                    Plumber.num_sources += 1

                self.pipeline.set_state(Gst.State.PLAYING)
                # start_pipeline(self.pipeline)
                
                return True
            except Exception as e:
                self.logging.info(f"error in enable: {e}")
                self.pipeline.set_state(Gst.State.PLAYING)
                # start_pipeline(self.pipeline)
                return False
            
        return wrapper

    def disable_channel(self,f):
        def wrapper(*args):
            source_id = args[0]
            try:
                #Disable the source
                self.g_source_enabled[source_id] = False
                #Release the source
                self.logging.info("Calling Stop %d " % source_id)
                reval = self.stop_release_source(source_id)

                if reval == 1:
                    return True
            except Exception as e:
                self.logging.info(f"error in disable channel_{source_id}: {e}")
                return False
        return wrapper 

    def bind_chan_task(self,f):
        # 有可能存在需要暂停视频流bin的操作
        def wrapper(*args):
            service_id = args[0]
            channel_id = args[1]
            channel_name = args[2]
            config = args[3]
            try:
                # self.pipeline.set_state(Gst.State.PAUSED)
                # self.pipeline.set_state(Gst.State.NULL)
                # time.sleep(0.5)
                # stop_pipeline(self.g_source_bin_list[channel_id])
                stop_pipeline(self.pipeline)
                sinkpad = self.streammuxs[service_id].get_request_pad(f"sink_{channel_id}")
                
                queue = Gst.ElementFactory.make("queue", f"channel{channel_id}_task{service_id}")
                self.pipeline.add(queue)

                # self.tees[channel_id].link(queue)
                tee = self.tees[channel_id]
                tee_src_pad = tee.get_request_pad(f"src_{service_id+1}")

                queue_sink_pad = queue.get_static_pad("sink")
                queue_src_pad = queue.get_static_pad("src")

                tee_src_pad.link(queue_sink_pad)
                queue_src_pad.link(sinkpad)

                # self.pipeline.set_state(Gst.State.PLAYING)
                # start_pipeline(self.g_source_bin_list[channel_id])
                start_pipeline(self.pipeline)
                return True
            except Exception as e:
                self.logging.info(f"error in bind: {e}")
                return False
        return wrapper
    
    def unbind_chan_task(self, f):
        def wrapper(*args):
            print("***********************************************************")
            service_id = args[0]
            channel_id = args[1]
            try:
                # stop_pipeline(self.g_source_bin_list[channel_id])
                stop_pipeline(self.pipeline)
                queue = self.pipeline.get_by_name(f"channel{channel_id}_task{service_id}")

                if queue:
                    remove_element_from_pipeline(self.pipeline, queue)
                    # self.pipeline.remove(queue)
                    # queue.set_state(Gst.State.NULL)

                tee = self.tees[channel_id]
                streammux = self.streammuxs[service_id]

                tee_src_pad = tee.get_static_pad(f"src_{service_id+1}")
                if tee_src_pad:
                    tee.release_request_pad(tee_src_pad)

                sinkpad = streammux.get_static_pad(f"sink_{channel_id}")
                if sinkpad:
                    streammux.release_request_pad(sinkpad)

                try:
                    self.trackers_lock.acquire()
                    self.trackers.clear
                except Exception as e:
                    print(f"error in unbind pop tracker: {e}")
                finally:
                    self.trackers_lock.release()

                # start_pipeline(self.g_source_bin_list[channel_id])
                start_pipeline(self.pipeline)
                    
                return True
            except Exception as e:
                self.logging.info(f"error in unbind task{service_id} channel{channel_id}: {e}")
                # self.pipeline.set_state(Gst.State.PLAYING)
                return False

        return wrapper

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


        # self.logging.info("Creating streammux \n ")
        # # Create nvstreammux instance to form batches from one or more sources.
        # self.streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
        # if not self.streammux:
        #     sys.stderr.write(" Unable to create NvStreamMux \n")

        # self.streammux.set_property("batched-push-timeout", 200000)                   
        # self.streammux.set_property("batch-size", 16)           # important!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # self.streammux.set_property("gpu_id", GPU_ID)

        # self.pipeline.add(self.streammux)
        # self.streammux.set_property("live-source", 1)

        # self.logging.info("Creating Pgie \n ")
        # pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
        # if not pgie:
        #     sys.stderr.write(" Unable to create pgie \n")

        # self.logging.info("Creating nvtracker \n ")
        # tracker = Gst.ElementFactory.make("nvtracker", "tracker")
        # if not tracker:
        #     sys.stderr.write(" Unable to create tracker \n")

        # tracker_srcpad = tracker.get_static_pad("src")
        # if not tracker_srcpad:
        #     sys.stderr.write(" Unable to get sink pad of nvosd \n")
        # tracker_srcpad.add_probe(Gst.PadProbeType.BUFFER, self.nvdrmvideosink_probe, 0)

        # # self.logging.info("Creating nvstreamdemux \n ")
        # # self.nvstreamdemux = Gst.ElementFactory.make("nvstreamdemux", "nvstreamdemux")
        # # if not self.nvstreamdemux:
        # #     sys.stderr.write(" Unable to create nvstreamdemux \n")

        # self.logging.info("Creating tee \n ")
        # tee = Gst.ElementFactory.make("tee", "tee")
        # if not tee:
        #     sys.stderr.write(" Unable to create tee \n")

        # tee_q0 = Gst.ElementFactory.make("queue", "tee_q0")
        # # tee_q1 = Gst.ElementFactory.make("queue", "tee_q1")

        # self.logging.info("Creating tiler \n ")
        # tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
        # tiler.set_property('rows', self.tiler_size["tiler_row"])
        # tiler.set_property('columns', self.tiler_size["tiler_col"])
        # tiler.set_property('width', self.tiler_size["tiler_w"])
        # tiler.set_property('height', self.tiler_size["tiler_h"])
        # if not tiler:
        #     sys.stderr.write(" Unable to create tiler \n")

        # self.logging.info("Creating nvosd \n ")
        # nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        # if not nvosd:
        #     sys.stderr.write(" Unable to create nvosd \n")

        # # creating nvvidconv
        # out_videoconvert = Gst.ElementFactory.make("nvvideoconvert", "out_videoconvert")

        # out_caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), width=1920, height=1080")
        # out_filter = Gst.ElementFactory.make("capsfilter", "capsfilter")
        # out_filter.set_property("caps", out_caps)


        # self.logging.info("Creating sink \n ")
        # # drmsink = Gst.ElementFactory.make("nvdrmvideosink", "drmsink")
        # # # drmsink = Gst.ElementFactory.make("nv3dsink", "drmsink")
        # # drmsink.set_property('enable-last-sample', 0)
        # # drmsink.set_property('sync', 0)
        # # if not drmsink:
        # #     sys.stderr.write(" Unable to create drmsink \n")

        # sink = Gst.ElementFactory.make("fakesink", "sink")
        # sink.set_property("signal-handoffs", True)
        # sink.set_property("silent", False)
        
        # # 定义handoff信号的回调函数
        # def on_handoff(fakesink, buffer, pad):
        #     pass
        #     # print(f"Buffer received with PTS: {buffer.pts}")

        # # 连接handoff信号
        # sink.connect("handoff", on_handoff)


        # self.streammux.set_property('live-source', 1)
        # #Set streammux width and height
        # self.streammux.set_property('width', MUXER_OUTPUT_WIDTH)
        # self.streammux.set_property('height', MUXER_OUTPUT_HEIGHT)
        # #Set pgie configuration file paths
        # pgie.set_property('config-file-path', PGIE_CONFIG_FILE)

        # #Set properties of tracker
        # config = configparser.ConfigParser()
        # config.read(TRACKER_CONFIG_FILE)
        # config.sections()

        # # 设置保留帧数
        # # tracker.set_property("max-shadow-tracking-duration", 20)
        # for key in config['tracker']:
        #     if key == 'tracker-width' :
        #         tracker_width = config.getint('tracker', key)
        #         tracker.set_property('tracker-width', tracker_width)
        #     if key == 'tracker-height' :
        #         tracker_height = config.getint('tracker', key)
        #         tracker.set_property('tracker-height', tracker_height)
        #     if key == 'gpu-id' :
        #         tracker_gpu_id = config.getint('tracker', key)
        #         tracker.set_property('gpu_id', tracker_gpu_id)
        #     if key == 'll-lib-file' :
        #         tracker_ll_lib_file = config.get('tracker', key)
        #         tracker.set_property('ll-lib-file', tracker_ll_lib_file)
        #     if key == 'll-config-file' :
        #         tracker_ll_config_file = config.get('tracker', key)
        #         tracker.set_property('ll-config-file', tracker_ll_config_file)
        #     if key == 'enable-batch-process' :
        #         tracker_enable_batch_process = config.getint('tracker', key)
        #         tracker.set_property('enable_batch_process', tracker_enable_batch_process)

        # #Set necessary properties of the nvinfer element, the necessary ones are:
        # pgie_batch_size=pgie.get_property("batch-size")
        # if(pgie_batch_size < MAX_NUM_SOURCES):
        #     self.logging.info("WARNING: Overriding infer-config batch-size",pgie_batch_size," with number of sources ", Plumber.num_sources," \n")
        # pgie.set_property("batch-size",MAX_NUM_SOURCES)

        # self.logging.info("Adding elements to Pipeline \n")
        # self.pipeline.add(pgie)
        # self.pipeline.add(tracker)
        # # self.pipeline.add(self.nvstreamdemux)
        # self.pipeline.add(tee)
        # self.pipeline.add(tee_q0)
        # # self.pipeline.add(tee_q1)
        # self.pipeline.add(tiler)
        # self.pipeline.add(nvosd)
        # self.pipeline.add(out_videoconvert)
        # self.pipeline.add(out_filter)
        # self.pipeline.add(sink)

        # self.logging.info("Linking elements in the Pipeline \n")
        # self.streammux.link(pgie)
        # pgie.link(tracker)
        # tracker.link(tee)

        # tee.link(tiler)
        # # tee_q0.link(tiler)

        # # tee.link(tee_q1)
        # # tee.link(self.nvstreamdemux)

        
        # tiler.link(nvosd)
        # nvosd.link(sink)
        # # out_videoconvert.link(out_filter)
        # # out_filter.link(drmsink)
        
        
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

        # start play back and listed to events		
        self.pipeline.set_state(Gst.State.PLAYING)

        try:
            loop.run()
        except Exception as error:
            self.logging.info("error:", error)

        if Plumber.stand_by == False:
            Plumber.stand_by = True            

        # cleanup
        self.logging.info("Exiting app\n")
        self.pipeline.set_state(Gst.State.NULL)