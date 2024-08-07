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
import gc

from sort import BoxTracker

from utils import start_pipeline, stop_pipeline, get_data_GPU, check_pipeline_elements, SafeLock


MUXER_OUTPUT_WIDTH=1920
MUXER_OUTPUT_HEIGHT=1080

PGIE_CONFIG_FILE = '/root/apps/ai_server/cfg/config_infer_primary_yoloV8n.txt'
TRACKER_CONFIG_FILE = "/root/apps/ai_server/cfg/dstest_tracker_config.txt"

MAX_TIMES_FOR_ENABLE = 3


class Plumber():
    task_list = []
    channel_list = []
    task_channel_dict = {}
    service_lock = SafeLock(15)

    stand_by = False

    def __init__(self, logging):
        self.logging = logging

        self.tracker_dict = {}
        self.infer_pipeline_dict = {}
        self.source_pipeline_dict = {}
        self.enable_channel_times_dict = {}
        self.tracker_lock = threading.Lock()

        self.perf_data = PERF_DATA(0)

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
            # time.sleep(5)
            # try:
            #     Plumber.service_lock.acquire()
            #     if self.enable_channel_times_dict[channel_id] <= 0:
            #         self.logging.info(f"can not enable source {channel_id} with try {MAX_TIMES_FOR_ENABLE} times, disable it")
            #         self.disable_channel(None)(channel_id)
            #         Plumber.channel_list.remove(channel_id)
            #     else:
            #         start_pipeline(source_pipeline)
            #         self.enable_channel_times_dict[channel_id] -= 1
            # except Exception as e:
            #     self.logging.info(f"error in re-disable or re-enable {channel_id}")
            # finally:
            #     Plumber.service_lock.release()
            # loop.quit()

        elif t==Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            sys.stderr.write("Warning: %s: %s\n" % (err, debug))

        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            sys.stderr.write("Error: %s: %s\n" % (err, debug))
            # time.sleep(5)
            # try:
            #     Plumber.service_lock.acquire()
            #     if self.enable_channel_times_dict[channel_id] <= 0:
            #         self.logging.info(f"can not enable source {channel_id} with try {MAX_TIMES_FOR_ENABLE} times, disable it")
            #         self.disable_channel(None)(channel_id)
            #         Plumber.channel_list.remove(channel_id)
            #     else:
            #         start_pipeline(source_pipeline)
            #         self.enable_channel_times_dict[channel_id] -= 1
            # except Exception as e:
            #     self.logging.info(f"error in re-disable or re-enable {channel_id}")
            # finally:
            #     Plumber.service_lock.release()
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

    # def stop_release_source(self, source_id):
    #     #Attempt to change status of source to be released 
    #     source_pipe = self.source_pipeline_dict[source_id]
    #     state_return = source_pipe.set_state(Gst.State.NULL)

    #     if state_return == Gst.StateChangeReturn.SUCCESS or state_return == Gst.StateChangeReturn.ASYNC:
    #         self.logging.info("LBK_STATE CHANGE SUCCESS\n")
    #         for service_id in Plumber.task_channel_dict.keys():
    #             if source_id in Plumber.task_channel_dict[service_id]:
    #                 infer_pipe = self.infer_pipeline_dict[service_id]

    #                 streammux = infer_pipe.get_by_name(f"Stream-muxer-{service_id}")
    #                 if streammux:
    #                     streammux_sink_pad = streammux.get_static_pad(f"sink_{source_id}")
    #                     streammux_sink_pad.send_event(Gst.Event.new_flush_stop(True))
    #                     streammux.release_request_pad(streammux_sink_pad)

    #                 queue_right = infer_pipe.get_by_name(f"queue_channel{source_id}_task{service_id}_right")
    #                 appsrc = infer_pipe.get_by_name(f"appsrc_channel{source_id}_task{service_id}")

    #                 if appsrc:
    #                     appsrc.set_state(Gst.State.NULL)
    #                     infer_pipe.remove(appsrc)
                    
    #                 if queue_right:
    #                     queue_right.set_state(Gst.State.NULL)
    #                     infer_pipe.remove(queue_right)

    #                 Plumber.task_channel_dict[service_id].remove(source_id)
            
    #         self.source_pipeline_dict.pop(source_id)
    #         source_pipe.unref()
    #         return True

    #     elif state_return == Gst.StateChangeReturn.FAILURE:
    #         self.logging.info("LBK_STATE CHANGE FAILURE\n")
    #         return False

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

        filter_src_pad = filter.get_static_pad("src")
        bin_ghost_pad=nbin.get_static_pad("src")
        bin_ghost_pad.set_target(filter_src_pad)

        sinkpad = tee.get_static_pad("sink")
        bin_ghost_pad.link(sinkpad)

        return nbin

    def add_task(self, f):
        def wrapper(*args):
            try:

                service_id = args[0]
                dev_id = args[1]
                root = args[2]

                if service_id in self.infer_pipeline_dict.keys():
                    pipeline = self.infer_pipeline_dict[service_id]
                else:
                    pipeline = Gst.Pipeline.new(f"infer_pipe{service_id}")

                    bus = pipeline.get_bus()
                    bus.add_signal_watch()
                    bus.connect ("message", self.bus_call, None)

                    pipeline.set_state(Gst.State.NULL)

                    self.logging.info(f"Creating streammux_{service_id}\n ")
                    # Create nvstreammux instance to form batches from one or more sources.
                    streammux = Gst.ElementFactory.make("nvstreammux", f"Stream-muxer-{service_id}")
                    if not streammux:
                        sys.stderr.write(" Unable to create NvStreamMux \n")

                    streammux.set_property("batched-push-timeout", 200000)                   
                    streammux.set_property("batch-size", 16)           # important!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    streammux.set_property("gpu_id", dev_id)
                    streammux.set_property("live-source", 1)

                    #Set streammux width and height
                    streammux.set_property('width', MUXER_OUTPUT_WIDTH)
                    streammux.set_property('height', MUXER_OUTPUT_HEIGHT)

                    # 设置为False之后可以动态链接，但是断开之后再重连会连接不上，设置为True则绑定需要暂停管道，猜测是因为要求数据同步，所以需要初始化管道 
                    streammux.set_property("sync-inputs", False) #很重要！！！！！！！！！！！！！！！！

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
                    sink = Gst.ElementFactory.make("fakesink", f"task_fakesink_{service_id}")
                    sink.set_property('enable-last-sample', 0)
                    sink.set_property("sync", 0)

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

                    self.logging.info("Adding elements to Pipeline \n")
                    pipeline.add(pgie)
                    pipeline.add(tracker)
                    pipeline.add(sink)

                    self.logging.info("Linking elements in the Pipeline \n")
                    streammux.link(pgie)
                    pgie.link(tracker)
                    tracker.link(sink)
                    self.infer_pipeline_dict[service_id] = pipeline

                pipeline.set_state(Gst.State.PLAYING)
                return True 
            except Exception as e:
                self.logging.info(f"error in add task {service_id}:{e}")
                return False
        return wrapper

    def remove_task(self, f):
        def wrapper(*args):
            service_id = args[0]
            try:
                infer_pipe = self.infer_pipeline_dict[service_id]

                for channel_id in Plumber.task_channel_dict[service_id]:
                    source_pipe = self.source_pipeline_dict[channel_id]
                    stop_pipeline(source_pipe)

                    tee = source_pipe.get_by_name(f"tee_{channel_id}")
                    queue_left = source_pipe.get_by_name(f"queue_task{service_id}_left")
                    appsink = source_pipe.get_by_name(f"appsink_task{service_id}")

                    appsrc = infer_pipe.get_by_name(f"appsrc_channel{channel_id}")
                    queue_right = infer_pipe.get_by_name(f"queue_channel{channel_id}_right")
                    streammux = infer_pipe.get_by_name(f"Stream-muxer-{service_id}")

                    sinkpad = streammux.get_static_pad(f"sink_{channel_id}")
                    sinkpad.send_event(Gst.Event.new_flush_stop(True))
                    streammux.release_request_pad(sinkpad)

                    tee_src_pad = tee.get_static_pad(f"src_{service_id}")
                    tee_src_pad.send_event(Gst.Event.new_flush_stop(False))
                    tee.release_request_pad(tee_src_pad)
                    
                    source_pipe.set_state(Gst.State.PLAYING)

                    if queue_left:
                        queue_left.set_state(Gst.State.NULL)
                        source_pipe.remove(queue_left)

                    if appsink:
                        appsink.set_state(Gst.State.NULL)
                        source_pipe.remove(appsink)

                    if appsrc:
                        appsrc.set_state(Gst.State.NULL)
                        infer_pipe.remove(appsrc)

                    if queue_right:
                        queue_right.set_state(Gst.State.NULL)
                        infer_pipe.remove(queue_right)

                    # time.sleep(0.3) # 预留一定间隔防止状态轮转太快导致的pipe崩溃或死锁

                # stop_pipeline(infer_pipe) # 停止再启动则造成内存泄漏，奇奇怪怪

                # streammux = infer_pipe.get_by_name(f"Stream-muxer-{service_id}")
                # pgie = infer_pipe.get_by_name(f"primary_inference{service_id}")
                # tracker = infer_pipe.get_by_name(f"tracker{service_id}")
                # sink = infer_pipe.get_by_name(f"task_fakesink_{service_id}")

                # infer_pipe.remove(streammux)
                # infer_pipe.remove(pgie)
                # infer_pipe.remove(tracker)
                # infer_pipe.remove(sink)

                # self.infer_pipeline_dict.pop(service_id)
                # infer_pipe.unref()
                # del infer_pipe

                try:
                    self.tracker_lock.acquire()
                    self.tracker_dict.clear()
                except Exception as e:
                    self.logging.info(f"error in remove task clean tracker: {e}")
                finally:
                    self.tracker_lock.release()

                return True
            except Exception as e:
                self.logging.info(f"error in remove tasks: {e}")
                return False
        return wrapper

    def enable_channel(self, f):
        def wrapper(*args):
            channel_id = args[0]
            channel_name = args[1]
            channel_url = args[2]
            try:
                self.enable_channel_times_dict[channel_id] = MAX_TIMES_FOR_ENABLE

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

                self.perf_data.add_stream(channel_id)
                self.source_pipeline_dict[channel_id] = source_pipeline
                return True
            except Exception as e:
                self.logging.info(f"error in enable: {e}")
                return False
            
        return wrapper

    def disable_channel(self,f):
        def wrapper(*args):
            channel_id = args[0]
            try:
                #Release the source
                self.logging.info("Calling Stop %d " % channel_id)
                # reval = self.stop_release_source(channel_id)
                source_pipe = self.source_pipeline_dict[channel_id]

                stop_pipeline(source_pipe)

                for service_id in Plumber.task_channel_dict.keys():
                    if channel_id in Plumber.task_channel_dict[service_id]:
                        infer_pipe = self.infer_pipeline_dict[service_id]

                        streammux = infer_pipe.get_by_name(f"Stream-muxer-{service_id}")
                        if streammux:
                            streammux_sink_pad = streammux.get_static_pad(f"sink_{channel_id}")
                            streammux_sink_pad.send_event(Gst.Event.new_flush_stop(True))
                            streammux.release_request_pad(streammux_sink_pad)

                        queue_right = infer_pipe.get_by_name(f"queue_channel{channel_id}_right")
                        appsrc = infer_pipe.get_by_name(f"appsrc_channel{channel_id}")

                        if appsrc:
                            appsrc.set_state(Gst.State.NULL)
                            infer_pipe.remove(appsrc)

                        if queue_right:
                            queue_right.set_state(Gst.State.NULL)
                            infer_pipe.remove(queue_right)

                        Plumber.task_channel_dict[service_id].remove(channel_id)

                self.source_pipeline_dict.pop(channel_id)
                source_pipe.unref()

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

                stop_pipeline(source_pipe)
                
                # #####################################>>>获取元素<<<###################################
                tee = source_pipe.get_by_name(f"tee_{channel_id}")

                appsink = Gst.ElementFactory.make("appsink", f"appsink_task{service_id}")
                appsink.set_property('emit-signals', True)
                appsink.set_property('sync', False)
                # appsink.set_property('async', True)
                appsink.set_property("enable-last-sample", False)
                # appsink.set_property("drop", True)
                appsink.set_property("max_buffers", 50)
                
                appsrc = Gst.ElementFactory.make("appsrc", f"appsrc_channel{channel_id}")
                appsrc.set_property('is-live', True)
                # appsrc.set_property('format', Gst.Format.TIME)
                # appsrc.set_property("block", True)
                caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA, width=1920, height=1080")
                appsrc.set_property("caps", caps) #非常重要！！！！！！！

                # tee后边也要一个queue，不然在对新任务进行绑定时，没有缓冲造成重启失败
                queue_left = Gst.ElementFactory.make("queue2", f"queue_task{service_id}_left")
                # queue_left.set_property("max-size-buffers", 0)
                # queue_left.set_property("max-size-bytes", 0)
                # queue_left.set_property("max-size-time", 100000000)

                queue_right = Gst.ElementFactory.make("queue2", f"queue_channel{channel_id}_right")
                # queue_right.set_property("max-size-buffers", 0)
                # queue_right.set_property("max-size-bytes", 0)
                # queue_right.set_property("max-size-time", 100000000)

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

                sinkpad = streammux.get_request_pad(f"sink_{channel_id}")
                # ################################>>>获取pad结束<<<###########################################                
                

                tee_src_pad.link(queue_left_sink_pad)
                queue_left_src_pad.link(appsink_sink_pad)

                def on_new_sample(sink, src):
                    sample = sink.emit("pull-sample")
                    buffer = sample.get_buffer()
                    # 将buffer推送到appsrc
                    src.emit("push-buffer", buffer)
                    # print("*********************************")
                    return Gst.FlowReturn.OK
                
                appsink.connect("new-sample", on_new_sample, appsrc)

                appsrc_src_pad.link(queue_right_sink_pad)
                queue_right_src_pad.link(sinkpad)
                
                # appsink.set_state(Gst.State.PLAYING)
                # queue_left.set_state(Gst.State.PLAYING)
                appsrc.set_state(Gst.State.PLAYING)
                queue_right.set_state(Gst.State.PLAYING)

                try:
                    self.tracker_lock.acquire()
                    self.tracker_dict.clear()
                except Exception as e:
                    self.logging.info(f"error in remove task clean tracker: {e}")
                finally:
                    self.tracker_lock.release()

                source_pipe.set_state(Gst.State.PLAYING)
                # start_pipeline(source_pipe)
                return True
            except Exception as e:
                self.logging.info(f"error in bind: {e}")
                source_pipe.set_state(Gst.State.PLAYING)
                # start_pipeline(source_pipe)
                return False
        return wrapper

    def unbind_chan_task(self, f):
        def wrapper(*args):
            service_id = args[0]
            channel_id = args[1]
            try:
                infer_pipe = self.infer_pipeline_dict[service_id]
                source_pipe = self.source_pipeline_dict[channel_id]

                stop_pipeline(source_pipe)

                tee = source_pipe.get_by_name(f"tee_{channel_id}")
                queue_left = source_pipe.get_by_name(f"queue_task{service_id}_left")
                appsink = source_pipe.get_by_name(f"appsink_task{service_id}")
                
                appsrc = infer_pipe.get_by_name(f"appsrc_channel{channel_id}")
                queue_right = infer_pipe.get_by_name(f"queue_channel{channel_id}_right")
                streammux = infer_pipe.get_by_name(f"Stream-muxer-{service_id}")

                if streammux:
                    sinkpad = streammux.get_static_pad(f"sink_{channel_id}")
                    sinkpad.send_event(Gst.Event.new_flush_stop(True))
                    streammux.release_request_pad(sinkpad)

                if tee:
                    tee_src_pad = tee.get_static_pad(f"src_{service_id}")
                    tee_src_pad.send_event(Gst.Event.new_flush_stop(False))
                    tee.release_request_pad(tee_src_pad)

                source_pipe.set_state(Gst.State.PLAYING)
                # start_pipeline(source_pipe)

                if queue_left:
                    queue_left.set_state(Gst.State.NULL)
                    source_pipe.remove(queue_left)

                if appsink:
                    appsink.set_state(Gst.State.NULL)
                    source_pipe.remove(appsink)

                if appsrc:
                    appsrc.set_state(Gst.State.NULL)
                    infer_pipe.remove(appsrc)

                if queue_right:
                    queue_right.set_state(Gst.State.NULL)
                    infer_pipe.remove(queue_right)

                try:
                    self.tracker_lock.acquire()
                    self.tracker_dict.clear()
                except Exception as e:
                    self.logging.info(f"error in remove task clean tracker: {e}")
                finally:
                    self.tracker_lock.release()

                return True
            except Exception as e:
                self.logging.info(f"error in unbind task{service_id} channel{channel_id}: {e}")
                source_pipe.set_state(Gst.State.PLAYING)
                # start_pipeline(source_pipe)
                return True #即使失败，链接结构也被破坏了
        return wrapper

    def run(self,):
        # Standard GStreamer initialization
        Gst.init(None)

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
        # self.pipeline.set_state(Gst.State.NULL)


# nohup python3 -u app.py >> output.log 2>&1 &
# nohup python3 -u grpc_client.py >> out_client.log 2>&1 &
