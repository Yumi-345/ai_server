# /bin/python3
Author = "xuwj"

import gi
import configparser
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib


from common.FPS import PERF_DATA

import cv2 

import threading
import sys
import pyds
import time

from sort import BoxTracker

from utils import start_pipeline, stop_pipeline, remove_element_from_pipeline, get_data_GPU


MUXER_OUTPUT_WIDTH=1920
MUXER_OUTPUT_HEIGHT=1080

PGIE_CONFIG_FILE = '/root/apps/ai_server/cfg/config_infer_primary_yoloV8n.txt'
TRACKER_CONFIG_FILE = "/root/apps/ai_server/cfg/dstest_tracker_config.txt"



class Plumber():
    task_list = []
    channel_list = []
    task_channel_dict = {}

    stand_by = False

    def __init__(self, logging):
        self.logging = logging

        # self.g_source_bin_list = {}
        # self.streammuxs = {}
        # self.tees = {}

        self.tracker_dict = {}
        self.infer_pipeline_dict = {}
        self.source_pipeline_dict = {}
        self.source_flag_dict = {}
        self.tracker_lock = threading.Lock()

        self.perf_data = PERF_DATA(0)
        # self.tiler_size = json.load(open("/root/apps/ai_server/cfg/config.json", 'r')) 
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
                    
                    if left < 10 or top < 10 or left + width > 1920 - 10 or top + height > 1080 - 10:
                        l_obj = l_obj.next
                        continue

                    # 计算框中点
                    center_x = left + width / 2
                    center_y = top + height / 2
                    try:
                        self.tracker_lock.acquire()
                        if u_id in self.tracker_dict.keys():
                            self.tracker_dict[u_id].update(img_arr, u_id, conf, top, left, width, height, center_x, center_y, class_id, src_id, frame_number)
                        else:
                            self.tracker_dict[u_id] = BoxTracker(img_arr, u_id, conf, top, left, width, height, center_x, center_y, class_id, src_id, frame_number)
                    except Exception as e:
                        self.logging.info(f"error in update tracker_dict: {e}")
                    finally:
                        self.tracker_lock.release()
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
            try:
                self.tracker_lock.acquire()
                n2d = []
                for key in self.tracker_dict.keys():
                    if frame_number - self.tracker_dict[key].frame_number > 18: #判断是否达到发送要求
                        flag, msg = self.tracker_dict[key].send(save=True)
                        # print(msg)
                        if frame_number - self.tracker_dict[key].frame_number > 120 or flag:
                            n2d.append(key)
                for index in n2d:
                    self.tracker_dict.pop(index)
            except Exception as e:
                self.logging.info(f"error in pop tracker: {e}")
            finally:
                self.tracker_lock.release()

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


    def source_bus_call(self, bus, message, source_pipeline, channel_id):

        t = message.type
        if t == Gst.MessageType.EOS:
            sys.stdout.write("************End-of-stream\n")
            self.source_flag_dict[channel_id] = False
            # time.sleep(5)
            # start_pipeline(source_pipeline)
            # loop.quit()
        elif t==Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            sys.stderr.write("Warning: %s: %s\n" % (err, debug))
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            self.source_flag_dict[channel_id] = False
            sys.stderr.write("Error********: %s: %s\n" % (err, debug))
            # time.sleep(5)
            # start_pipeline(source_pipeline)
            # loop.quit()
        elif t == Gst.MessageType.ELEMENT:
            struct = message.get_structure()
            #Check for stream-eos message
            if struct is not None and struct.has_name("stream-eos"):
                parsed, stream_id = struct.get_uint("stream-id")
                if parsed:
                    #Set eos status of stream to True, to be deleted in delete-sources
                    self.logging.info("Got EOS from stream %d" % stream_id)
        return True

    def bus_call(self, bus, message, loop):
        # global g_eos_list
        t = message.type
        if t == Gst.MessageType.EOS:
            sys.stdout.write("End-of-stream\n")
            # loop.quit()
        elif t==Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            sys.stderr.write("Warning: %s: %s\n" % (err, debug))
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            sys.stderr.write("Error: %s: %s\n" % (err, debug))
            # loop.quit()
        elif t == Gst.MessageType.ELEMENT:
            struct = message.get_structure()
            #Check for stream-eos message
            if struct is not None and struct.has_name("stream-eos"):
                parsed, stream_id = struct.get_uint("stream-id")
                if parsed:
                    #Set eos status of stream to True, to be deleted in delete-sources
                    self.logging.info("Got EOS from stream %d" % stream_id)
        return True

    def stop_release_source(self, source_id):
        #Attempt to change status of source to be released 
        source_pipe = self.source_pipeline_dict[source_id]
        state_return = source_pipe.set_state(Gst.State.NULL)
        # stop_pipeline(self.pipeline)

        if state_return == Gst.StateChangeReturn.SUCCESS:
            self.logging.info("LBK_STATE CHANGE SUCCESS\n")
            for service_id in Plumber.task_channel_dict.keys():
                if source_id in Plumber.task_channel_dict[service_id]:
                    infer_pipe = self.infer_pipeline_dict[service_id]
                    
                    appsrc = infer_pipe.get_by_name(f"appsrc_channel{source_id}_task{service_id}")
                    if appsrc:
                        infer_pipe.remove(appsrc)
                        appsrc.set_state(Gst.State.NULL)
                        # appsrc.unref()
                    
                    queue_right = infer_pipe.get_by_name(f"queue_channel{source_id}_task{service_id}_right")
                    if queue_right:
                        infer_pipe.remove(queue_right)
                        queue_right.set_state(Gst.State.NULL)
                        # queue_right.unref()
                    
                    streammux = infer_pipe.get_by_name(f"Stream-muxer-{service_id}")
                    if streammux:
                        streammux.release_request_pad(streammux.get_static_pad(f"sink_{source_id}"))

                    Plumber.task_channel_dict[service_id].remove(source_id)
            
            self.source_pipeline_dict.pop(source_id)
            source_pipe.unref()
            # try:
            #     self.tracker_lock.acquire()
            #     self.tracker_dict.clear()
            # except Exception as e:
            #     self.logging.info(f"error in stop src clean tracker: {e}")
            # finally:
            #     self.tracker_lock.release()
            # start_pipeline(self.pipeline)
            return True

        elif state_return == Gst.StateChangeReturn.FAILURE:
            self.logging.info("LBK_STATE CHANGE FAILURE\n")
            return False
        
        elif state_return == Gst.StateChangeReturn.ASYNC:
            self.logging.info("LBK_STATE CHANGE SUCCESS\n")
            for service_id in Plumber.task_channel_dict.keys():
                if source_id in Plumber.task_channel_dict[service_id]:
                    infer_pipe = self.infer_pipeline_dict[service_id]
                    
                    appsrc = infer_pipe.get_by_name(f"appsrc_channel{source_id}_task{service_id}")
                    if appsrc:
                        infer_pipe.remove(appsrc)
                        appsrc.set_state(Gst.State.NULL)
                        # appsrc.unref()
                    
                    queue_right = infer_pipe.get_by_name(f"queue_channel{source_id}_task{service_id}_right")
                    if queue_right:
                        infer_pipe.remove(queue_right)
                        queue_right.set_state(Gst.State.NULL)
                        # queue_right.unref()
                    
                    streammux = infer_pipe.get_by_name(f"Stream-muxer-{service_id}")
                    if streammux:
                        streammux.release_request_pad(streammux.get_static_pad(f"sink_{source_id}"))

                    Plumber.task_channel_dict[service_id].remove(source_id)
            
            self.source_pipeline_dict.pop(source_id)
            source_pipe.unref()
            # try:
            #     self.tracker_lock.acquire()
            #     self.tracker_dict.clear()
            # except Exception as e:
            #     self.logging.info(f"error in stop src clean tracker: {e}")
            # finally:
            #     self.tracker_lock.release()
            # start_pipeline(self.pipeline)
            return True

    def create_source_bin(self,source_pipeline, source_id,source_url):
        self.logging.info("Creating source bin")

        # Create a source GstBin to abstract this bin's content from the rest of the
        # pipeline
        bin_name = f"source-bin-{source_id}"
        self.logging.info(bin_name)
        nbin = Gst.Bin.new(bin_name)
        if not nbin:
            self.logging.info(" Unable to create source bin \n")
        source_pipeline.add(nbin)

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

        caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA, width=1920, height=1080")
        filter = self.make_element("capsfilter", source_id)
        filter.set_property("caps", caps1)
        Gst.Bin.add(nbin,filter)

        tee = Gst.ElementFactory.make("tee", f"tee_{source_id}")
        source_pipeline.add(tee) # 特别注意这里，因为tee已经需要放在bin的外部了，所以这里是添加到pipeline里边而不是nbin
        # Gst.Bin.add(nbin,tee)

        ###########################################################################
        # queue = Gst.ElementFactory.make("queue", f"queue_channel_{source_id}")
        # self.pipeline.add(queue)

        # sink = Gst.ElementFactory.make("fakesink", f"fakesink_channel_{source_id}")
        # # sink.set_property("signal-handoffs", True)
        # # sink.set_property("sync", False)
        # sink.set_property('enable-last-sample', 0) #奇怪的属性，设置了就能运行
        # self.pipeline.add(sink)
        ############################################################################

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

        # tee.link(queue)
        # queue.link(sink)

        # self.tees[source_id] = tee

        return nbin
        # return source_pipeline


    def add_task(self, f):
        def wrapper(*args):
            try:

                service_id = args[0]
                dev_id = args[1]
                root = args[2]

                pipeline = Gst.Pipeline.new(f"infer_pipe{service_id}")

                bus = pipeline.get_bus()
                bus.add_signal_watch()
                bus.connect ("message", self.bus_call, None)    



                pipeline.set_state(Gst.State.PAUSED)
                # stop_pipeline(pipeline)
                # time.sleep(0.5)

                self.logging.info(f"Creating streammux_{service_id}\n ")
                # Create nvstreammux instance to form batches from one or more sources.
                streammux = Gst.ElementFactory.make("nvstreammux", f"Stream-muxer-{service_id}")
                # self.streammuxs[service_id] = streammux
                if not streammux:
                    sys.stderr.write(" Unable to create NvStreamMux \n")

                streammux.set_property("batched-push-timeout", 200000)                   
                streammux.set_property("batch-size", 16)           # important!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                streammux.set_property("gpu_id", dev_id)
                streammux.set_property("live-source", 1)
                
                #Set streammux width and height
                streammux.set_property('width', MUXER_OUTPUT_WIDTH)
                streammux.set_property('height', MUXER_OUTPUT_HEIGHT)

                streammux.set_property("sync-inputs", True) #很重要！！！！！！！！！！！！！！！！

                pipeline.add(streammux)

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
                # sink.set_property("signal-handoffs", True)
                # sink.set_property("sync", False)
                sink.set_property('enable-last-sample', 0)

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
                # pgie_batch_size=pgie.get_property("batch-size")
                # if(pgie_batch_size < MAX_NUM_SOURCES):
                #     self.logging.info("WARNING: Overriding infer-config batch-size",pgie_batch_size," with number of sources ", Plumber.num_sources," \n")
                # pgie.set_property("batch-size",MAX_NUM_SOURCES)

                self.logging.info("Adding elements to Pipeline \n")
                pipeline.add(pgie)
                pipeline.add(tracker)
                pipeline.add(sink)

                self.logging.info("Linking elements in the Pipeline \n")
                streammux.link(pgie)
                pgie.link(tracker)
                tracker.link(sink)

                pipeline.set_state(Gst.State.PLAYING)
                # start_pipeline(pipeline)
                # time.slppe(0.5)
                self.infer_pipeline_dict[service_id] = pipeline
                return True 
            except Exception as e:
                self.logging.info(f"error in add task {service_id}:{e}")
                # self.pipeline.set_state(Gst.State.PLAYING)
                # start_pipeline(pipeline)
                # time.sleep(0.5)
                return False
        return wrapper

    def remove_task(self, f):
        def wrapper(*args):
            service_id = args[0]
            try:
                infer_pipe = self.infer_pipeline_dict[service_id]
                stop_pipeline(infer_pipe)

                for channel_id in Plumber.task_channel_dict[service_id]:
                    source_pipe = self.source_pipeline_dict[channel_id]

                    # *********************此处有可能需要先暂停视频流*****************
                    tee = source_pipe.get_by_name(f"tee_{channel_id}")
                    if tee:
                        tee.release_request_pad(tee.get_static_pad(f"src_{service_id}"))

                    queue_left = source_pipe.get_by_name(f"queue_channel{channel_id}_task{service_id}_left")
                    if queue_left:
                        source_pipe.remove(queue_left)
                        queue_left.set_state(Gst.State.NULL)
                        # queue_left.unref()

                    appsink = source_pipe.get_by_name(f"appsink_channel{channel_id}_task{service_id}")
                    if appsink:
                        source_pipe.remove(appsink)
                        appsink.set_state(Gst.State.NULL)
                        # appsink.unref()

                self.infer_pipeline_dict.pop(service_id)
                infer_pipe.unref()

                # try:
                #     self.tracker_lock.acquire()
                #     self.tracker_dict.clear()
                # except Exception as e:
                #     self.logging.info(f"error in remove task clean tracker: {e}")
                # finally:
                #     self.tracker_lock.release()

                # self.pipeline.set_state(Gst.State.PLAYING)
                # start_pipeline(self.pipeline)
                return True
            except Exception as e:
                self.logging.info(f"error in remove tasks: {e}")
                # self.pipeline.set_state(Gst.State.PLAYING)
                # start_pipeline(self.pipeline)
                return False
        return wrapper

    def enable_channel(self, f):
        def wrapper(*args):
            channel_id = args[0]
            channel_name = args[1]
            channel_url = args[2]
            self.source_flag_dict.clear()
            try:
                # self.pipeline.set_state(Gst.State.PAUSED)
                # stop_pipeline(self.pipeline)
                # time.sleep(0.5)

                source_pipeline = Gst.Pipeline.new(f"source_pipe_{channel_id}")
                source_pipeline.set_state(Gst.State.PAUSED)

                bus = source_pipeline.get_bus()
                bus.add_signal_watch()
                bus.connect("message", self.source_bus_call, source_pipeline, channel_id)
                
                self.logging.info("Calling Start %d " % channel_id)

                #Create a uridecode bin with the chosen source id
                source_bin = self.create_source_bin(source_pipeline, channel_id, channel_url)

                if (not source_bin):
                    sys.stderr.write("Failed to create source bin. Exiting.")
                    exit(1)
                
                #Add source bin to our list and to pipeline
                # self.g_source_bin_list[channel_id] = source_bin

                #Set state of source bin to playing
                state_return = source_pipeline.set_state(Gst.State.PLAYING)
                # time.sleep(0.5)


                if state_return == Gst.StateChangeReturn.SUCCESS:
                    self.logging.info("STATE CHANGE SUCCESS\n")

                elif state_return == Gst.StateChangeReturn.FAILURE:
                    self.logging.info("STATE CHANGE FAILURE\n")
                
                elif state_return == Gst.StateChangeReturn.ASYNC:
                    state_return = source_bin.get_state(Gst.CLOCK_TIME_NONE)

                elif state_return == Gst.StateChangeReturn.NO_PREROLL:
                    self.logging.info("STATE CHANGE NO PREROLL\n")

                # self.pipeline.set_state(Gst.State.PLAYING)
                # start_pipeline(self.pipeline)
                if channel_id in self.source_flag_dict.keys():
                    if not self.source_flag_dict[channel_id]:
                        source_pipeline.set_state(Gst.State.NULL)
                        return False       

                self.perf_data.add_stream(channel_id)
                self.source_pipeline_dict[channel_id] = source_pipeline

                # time.sleep(0.5)
                return True
            except Exception as e:
                self.logging.info(f"error in enable: {e}")
                # self.pipeline.set_state(Gst.State.PLAYING)
                # start_pipeline(self.pipeline)
                return False
            
        return wrapper

    def disable_channel(self,f):
        def wrapper(*args):
            channel_id = args[0]
            try:
                #Release the source
                self.logging.info("Calling Stop %d " % channel_id)
                reval = self.stop_release_source(channel_id)

                if reval == 1:
                    self.perf_data.remove_stream(channel_id)
                    return True
            except Exception as e:
                self.logging.info(f"error in disable channel_{channel_id}: {e}")
                return True #即使没有成功，链接结构大概率也已被破坏
        return wrapper 

    def bind_chan_task(self,f):
        # 有可能存在需要暂停视频流bin的操作
        def wrapper(*args):
            service_id = args[0]
            channel_id = args[1]
            channel_name = args[2]
            config = args[3]
            try:
                infer_pipe = self.infer_pipeline_dict[service_id]
                source_pipe = self.source_pipeline_dict[channel_id]

                stop_pipeline(source_pipe, infer_pipe)
                
                # #####################################>>>获取元素<<<###################################
                tee = source_pipe.get_by_name(f"tee_{channel_id}")

                appsink = Gst.ElementFactory.make("appsink", f"appsink_channel{channel_id}_task{service_id}")
                appsink.set_property('emit-signals', True)
                appsink.set_property('sync', False)
                # appsink.set_property('async', True)
                appsink.set_property("enable-last-sample", False)
                
                appsrc = Gst.ElementFactory.make("appsrc", f"appsrc_channel{channel_id}_task{service_id}")
                # appsrc.set_property('is-live', True)
                # appsrc.set_property('format', Gst.Format.TIME)
                # appsrc.set_property("block", True)
                appsrc.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA, width=1920, height=1080")) #非常重要！！！！！！！

                # tee后边也要一个queue，不然在对新任务进行绑定时，没有缓冲造成重启失败
                queue_left = Gst.ElementFactory.make("queue", f"queue_channel{channel_id}_task{service_id}_left")
                queue_right = Gst.ElementFactory.make("queue", f"queue_channel{channel_id}_task{service_id}_right")

                streammux = infer_pipe.get_by_name(f"Stream-muxer-{service_id}")
                # #############################>>>获取元素结束<<<###########################

                source_pipe.add(appsink)
                source_pipe.add(queue_left)
                infer_pipe.add(appsrc)
                infer_pipe.add(queue_right)

                # ################################>>>获取pad<<<#############################################
                tee_src_pad = tee.get_request_pad(f"src_{service_id}")

                appsink_sink_pad = appsink.get_static_pad("sink")
                appsrc_src_pad = appsrc.get_static_pad("src")

                queue_left_sink_pad = queue_left.get_static_pad("sink")
                queue_left_src_pad = queue_left.get_static_pad("src")

                queue_right_sink_pad = queue_right.get_static_pad("sink")
                queue_right_src_pad = queue_right.get_static_pad("src")

                if streammux:
                    sinkpad = streammux.get_request_pad(f"sink_{channel_id}")
                # ################################>>>获取pad结束<<<###########################################                


                tee_src_pad.link(queue_left_sink_pad)
                queue_left_src_pad.link(appsink_sink_pad)

                def on_new_sample(sink, src):
                    sample = sink.emit("pull-sample")
                    buffer = sample.get_buffer()
                    # 将buffer推送到appsrc
                    src.emit("push-buffer", buffer)
                    return Gst.FlowReturn.OK
                
                appsink.connect("new-sample", on_new_sample, appsrc)

                appsrc_src_pad.link(queue_right_sink_pad)
                queue_right_src_pad.link(sinkpad)

                try:
                    self.tracker_lock.acquire()
                    self.tracker_dict.clear()
                except Exception as e:
                    self.logging.info(f"error in remove task clean tracker: {e}")
                finally:
                    self.tracker_lock.release()
                
                start_pipeline(source_pipe, infer_pipe)
                return True
            except Exception as e:
                self.logging.info(f"error in bind: {e}")
                start_pipeline(source_pipe, infer_pipe)
                return False
        return wrapper
    
    def unbind_chan_task(self, f):
        def wrapper(*args):
            service_id = args[0]
            channel_id = args[1]
            try:
                infer_pipe = self.infer_pipeline_dict[service_id]
                source_pipe = self.source_pipeline_dict[channel_id]

                # stop_pipeline(source_pipe, infer_pipe)

                tee = source_pipe.get_by_name(f"tee_{channel_id}")
                queue_left = source_pipe.get_by_name(f"queue_channel{channel_id}_task{service_id}_left")
                appsink = source_pipe.get_by_name(f"appsink_channel{channel_id}_task{service_id}")
                
                appsrc = infer_pipe.get_by_name(f"appsrc_channel{channel_id}_task{service_id}")
                queue_right = infer_pipe.get_by_name(f"queue_channel{channel_id}_task{service_id}_right")
                streammux = infer_pipe.get_by_name(f"Stream-muxer-{service_id}")
                
                if queue_left:
                    source_pipe.remove(queue_left)
                    queue_left.set_state(Gst.State.NULL)
                    # queue_left.unref()

                if appsink:
                    source_pipe.remove(appsink)
                    appsink.set_state(Gst.State.NULL)
                    # appsink.unref()

                if appsrc:
                    infer_pipe.remove(appsrc)
                    appsrc.set_state(Gst.State.NULL)
                    # appsrc.unref()

                if queue_right:
                    infer_pipe.remove(queue_right)
                    queue_right.set_state(Gst.State.NULL)
                    # queue_right.unref()

                tee_src_pad = tee.get_static_pad(f"src_{service_id}")
                if tee_src_pad:
                    tee.release_request_pad(tee_src_pad)

                sinkpad = streammux.get_static_pad(f"sink_{channel_id}")
                if sinkpad:
                    streammux.release_request_pad(sinkpad)

                # try:
                #     self.tracker_lock.acquire()
                #     self.tracker_dict.clear()
                # except Exception as e:
                #     self.logging.info(f"error in unbind clean tracker: {e}")
                # finally:
                #     self.tracker_lock.release()

                # start_pipeline(source_pipe, infer_pipe)
                    
                return True
            except Exception as e:
                self.logging.info(f"error in unbind task{service_id} channel{channel_id}: {e}")
                start_pipeline(source_pipe, infer_pipe)
                return True #即使失败，链接结构也被破坏了
        return wrapper

    def run(self,):
        # os.system("rm /root/apps/ai_server/*.jpg")

        # Standard GStreamer initialization
        Gst.init(None)

        # Create gstreamer elements */
        # Create Pipeline element that will form a connection of other elements
        # self.logging.info("Creating Pipeline \n ")
        # self.pipeline = Gst.Pipeline.new("infer_pipe")

        # if not self.pipeline:
        #     sys.stderr.write(" Unable to create Pipeline \n")

        # perf callback function to print fps every 5 sec
        GLib.timeout_add(5000, self.perf_data.perf_print_callback)

        # create an event loop and feed gstreamer bus mesages to it
        loop = GLib.MainLoop()
        # bus = self.pipeline.get_bus()
        # bus.add_signal_watch()
        # bus.connect ("message", self.bus_call, loop)    

        # self.pipeline.set_state(Gst.State.PAUSED)

        # self.logging.info("Starting pipeline \n")

        # start play back and listed to events		
        # self.pipeline.set_state(Gst.State.PLAYING)

        try:
            loop.run()
        except Exception as error:
            self.logging.info("error:", error)

        # if Plumber.stand_by == False:
        #     Plumber.stand_by = True            

        # cleanup
        self.logging.info("Exiting app\n")
        self.pipeline.set_state(Gst.State.NULL)