# /bin/python3
Author = "xuwj"

import gi
import configparser
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib


from common.FPS import PERF_DATA

import cv2 
import time
import threading
import sys
import pyds
import numpy as np

import ctypes
import cupy as cp

from utils.sort import BoxTracker

from utils.utils import start_pipeline, stop_pipeline, get_data_GPU, check_pipeline_elements, SafeLock

from service.PCN import PCNServer

from collections import defaultdict, deque


MUXER_OUTPUT_WIDTH=1920
MUXER_OUTPUT_HEIGHT=1080

PGIE_CONFIG_FILE = '/root/apps/ai_server/cfg/test.txt'
TRACKER_CONFIG_FILE = "/root/apps/ai_server/cfg/dstest_tracker_config.txt"

# MAX_TIMES_FOR_ENABLE = 3
frame_cache = defaultdict(lambda:deque(maxlen=50))

SERVICE_DICT = {
    0:"PCNDetect",
    1:"PCNDetect",
    2:"PCNDetect",
    3:"PCNDetect"
}

def on_pad_added(src, pad, des):
    vpad = des.get_static_pad("sink")
    pad.link(vpad)

def decodebin_child_added(child_proxy,Object,name,user_data):
    print("Decodebin child added:", name, "\n")
    if(name.find("decodebin") != -1):
        Object.connect("child-added",decodebin_child_added,user_data)   
    if(name.find("nvv4l2decoder") != -1):
        Object.set_property("gpu_id", 0)
        Object.set_property("drop-frame-interval", 5) # 帧率的设置在这里，通过设置每几帧丢弃（or取?）一帧来达到改变帧率的目的。


class Plumber():
    task_list = []
    channel_list = []
    task_channel_dict = {}
    service_lock = SafeLock(15)

    stand_by = False

    def __init__(self, logging):
        self.logging = logging

        self.tracker_dict = {}
        self.infer_server_dict = {}
        self.source_pipeline_dict = {}
        self.tracker_lock = threading.Lock()

        self.perf_data = PERF_DATA(0)

        t = threading.Thread(target=self.run, args=())
        t.start()

    def tee_sink_probe(self, pad, info, u_data):
        # 获取 Gst.Buffer
        buffer = info.get_buffer()
        timestamp = buffer.pts / Gst.SECOND
        # print(f"======={u_data}:======={timestamp}")

        # Create dummy owner object to keep memory for the image array alive
        owner = None
        # Getting Image data using nvbufsurface
        # the input should be address of buffer and batch_id
        # Retrieve dtype, shape of the array, strides, pointer to the GPU buffer, and size of the allocated memory
        data_type, shape, strides, dataptr, size = pyds.get_nvds_buf_surface_gpu(hash(buffer), 0)
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
        stream = cp.cuda.stream.Stream(null=True) # Use null stream to prevent other cuda applications from making illegal memory access of buffer
        # Modify the red channel to add blue tint to image
        
        with stream:
            n_frame_cpu = n_frame_gpu.get()
            img_arr = cv2.cvtColor(n_frame_cpu, cv2.COLOR_RGBA2BGRA)
            frame_cache[u_data].append((img_arr, timestamp))
            # cv2.imwrite(f"/root/apps/ai_server/output/stream{timestamp}_src.jpg", img_arr)
        # stream.synchronize()

        return Gst.PadProbeReturn.OK


    
    def source_bus_call(self, bus, message, source_pipeline, channel_id):
        t = message.type
        if t == Gst.MessageType.EOS:
            sys.stdout.write("************End-of-stream\n")
            # time.sleep(60)
            # try:
            #     Plumber.service_lock.acquire()
            #     source_pipeline.set_state(Gst.State.PLAYING)
            #     # if self.enable_channel_times_dict[channel_id] <= 0:
            #     #     self.logging.info(f"can not enable source {channel_id} with try {MAX_TIMES_FOR_ENABLE} times, disable it")
            #     #     self.disable_channel(None)(channel_id)
            #     #     Plumber.channel_list.remove(channel_id)
            #     # else:
            #     #     start_pipeline(source_pipeline)
            #     #     self.enable_channel_times_dict[channel_id] -= 1
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

    def create_source_bin(self,source_pipeline, source_id, source_url):
        self.logging.info("Creating source bin")

        # Create a source GstBin to abstract this bin's content from the rest of the
        # pipeline
        bin_name = f"source-bin-{source_id}"
        self.logging.info(bin_name)

        # =====================================================================================
        # nbin = Gst.Bin.new(bin_name)
        # if not nbin:
        #     self.logging.info(" Unable to create source bin \n")
        # source_pipeline.add(nbin)

        # src = Gst.ElementFactory.make("rtspsrc", "src-"+str(source_id))
        # src.set_property("location", source_url)
        # src.set_property("drop-on-latency", True)
        # Gst.Bin.add(nbin,src)

        # rtph265depay = Gst.ElementFactory.make("rtph265depay", "depay-"+str(source_id))
        # src.connect("pad-added", self.on_pad_added, rtph265depay)
        # Gst.Bin.add(nbin,rtph265depay)

        # queuev1 = Gst.ElementFactory.make("queue2", "queue-"+str(source_id))
        # Gst.Bin.add(nbin,queuev1)

        # parse = Gst.ElementFactory.make("h265parse", "parse-"+str(source_id))
        # Gst.Bin.add(nbin,parse)

        # decode = Gst.ElementFactory.make("nvv4l2decoder", "decoder-"+str(source_id))
        # # decode.set_property("enable-max-performance", True)
        # decode.set_property("drop-frame-interval", 5)      # important!!!!!!!!!!!!!!!!!!!!!!!!!!! 这里影响帧率
        # decode.set_property("num-extra-surfaces", 0)
        # Gst.Bin.add(nbin,decode)

        # # creating nvvidconv
        # nvvideoconvert = Gst.ElementFactory.make("nvvideoconvert", f"nvvideoconvert_{source_id}")
        # nvvideoconvert.set_property("gpu-id", 0)
        # Gst.Bin.add(nbin,nvvideoconvert)

        # # videorate = Gst.ElementFactory.make("videorate", f"videorate_{source_id}")
        # # Gst.Bin.add(nbin,videorate)

        # caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA, width=1920, height=1080")
        # filter1 = Gst.ElementFactory.make("capsfilter", f"capsfilter_{source_id}")
        # filter1.set_property("caps", caps1)
        # Gst.Bin.add(nbin,filter1)

        # tee = Gst.ElementFactory.make("tee", f"tee_{source_id}")
        # source_pipeline.add(tee) # 特别注意这里，因为tee已经需要放在bin的外部了，所以这里是添加到pipeline里边而不是nbin

        # nbin.add_pad(Gst.GhostPad.new_no_target("src",Gst.PadDirection.SRC))

        # rtph265depay.link(queuev1)
        # queuev1.link(parse)
        # parse.link(decode)
        # decode.link(nvvideoconvert)
        # nvvideoconvert.link(filter1)
        # # videorate.link(filter1)

        # src_pad = filter1.get_static_pad("src")
        # bin_ghost_pad=nbin.get_static_pad("src")
        # bin_ghost_pad.set_target(src_pad)

        # sinkpad = tee.get_static_pad("sink")
        # bin_ghost_pad.link(sinkpad)
        # ===============================================================================================

        nbin=Gst.ElementFactory.make("uridecodebin", bin_name)
        if not bin:
            sys.stderr.write(" Unable to create uri decode bin \n")
        nbin.set_property("uri",source_url)
        # nbin.set_property("uri","rtsp://admin:avcit12345678@172.20.18.252/ch48/main/av_stream")
        # nbin.set_property("uri","rtsp://172.20.38.202:8558/test")
        # nbin.set_property("uri","file:///opt/nvidia/deepstream/deepstream-6.2/samples/streams/sample_1080p_h264.mp4")
        source_pipeline.add(nbin)

        # creating nvvidconv
        nvvideoconvert = Gst.ElementFactory.make("nvvideoconvert", f"nvvideoconvert_{source_id}")
        nvvideoconvert.set_property("gpu-id", 0)
        source_pipeline.add(nvvideoconvert)

        videorate = Gst.ElementFactory.make("videorate", f"videorate_{source_id}")
        videorate.set_property('drop-only', True)
        source_pipeline.add(videorate)

        caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA, width=1920, height=1080") # 添加(memory:NVMM) 会导致探针处的数据格式不正确，不添加则拉流失败
        # caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA, width=1920, height=1080, framerate=15/1") #对AVCIT摄像头无效

        filter1 = Gst.ElementFactory.make("capsfilter", f"capsfilter_{source_id}")
        filter1.set_property("caps", caps1)
        source_pipeline.add(filter1)

        tee = Gst.ElementFactory.make("tee", f"tee_{source_id}")
        source_pipeline.add(tee) 

        tee_sinkpad = tee.get_static_pad("sink")
        tee_sinkpad.add_probe(Gst.PadProbeType.BUFFER, self.tee_sink_probe, source_id)

        nbin.connect("pad-added", on_pad_added, nvvideoconvert)
        nbin.connect("child-added", decodebin_child_added, source_id)

        # queuev1.link(nvvideoconvert)
        nvvideoconvert.link(videorate)
        videorate.link(filter1)
        filter1.link(tee)

        return nbin

    def add_task(self, f):
        def wrapper(*args):
            try:
                service_id = args[0]
                dev_id = args[1]
                root = args[2]
                if service_id == 0:
                    pass                    
                elif service_id == 1:
                    pass
                elif service_id == 2:
                    pass
                elif service_id == 3:
                    pass
                elif service_id == 4:
                    pass
                elif service_id == 5:
                    pass
                elif service_id == 6:
                    pass
                elif service_id == 7:
                    pass
                elif service_id == 8:
                    pass
                elif service_id == 9:
                    pass
                elif service_id == 10:
                    pass
                elif int(service_id) in [11,12,13]:
                    server = PCNServer(service_id, dev_id, root, self.logging, self.perf_data)
                    self.infer_server_dict[service_id] = server
                elif service_id == 14:
                    pass
                elif service_id == 15:
                    pass
                elif service_id == 16:
                    pass
                elif service_id == 17:
                    pass
                elif service_id == 18:
                    pass
                elif service_id == 19:
                    pass
                else:
                    return False

                return True 
            except Exception as e:
                self.logging.info(f"error in add task {service_id}:{e}")
                return False
        return wrapper

    def remove_task(self, f):
        def wrapper(*args):
            service_id = args[0]
            try:
                # =========================================================
                server = self.infer_server_dict[service_id]
                infer_pipe = server.pipeline
                # ===========================================================

                infer_pipe.set_state(Gst.State.NULL)
                # check_pipeline_elements(infer_pipe)

                for channel_id in Plumber.task_channel_dict[service_id]:
                    source_pipe = self.source_pipeline_dict[channel_id]

                    # source_pipe.set_state(Gst.State.NULL)

                    tee = source_pipe.get_by_name(f"tee_{channel_id}")
                    queue_left = source_pipe.get_by_name(f"queue_task{service_id}_left")
                    appsink = source_pipe.get_by_name(f"appsink_task{service_id}")

                    # appsrc = infer_pipe.get_by_name(f"appsrc_channel{channel_id}")
                    # queue_right = infer_pipe.get_by_name(f"queue_channel{channel_id}_right")

                    streammux = infer_pipe.get_by_name(f"Stream-muxer-{service_id}")

                    if streammux:
                        sinkpad = streammux.get_static_pad(f"sink_{channel_id}")
                        sinkpad.send_event(Gst.Event.new_flush_stop(True))
                        streammux.release_request_pad(sinkpad)

                    if tee:
                        tee_src_pad = tee.get_static_pad(f"src_{service_id}")
                        # tee_src_pad.send_event(Gst.Event.new_flush_stop(False))
                        tee.release_request_pad(tee_src_pad)

                    # source_pipe.set_state(Gst.State.PLAYING)

                    if queue_left:
                        queue_left.set_state(Gst.State.NULL)
                        source_pipe.remove(queue_left)

                    if appsink:
                        appsink.set_state(Gst.State.NULL)
                        source_pipe.remove(appsink)

                    # if appsrc:
                    #     appsrc.set_state(Gst.State.NULL)
                    #     infer_pipe.remove(appsrc)

                    # if queue_right:
                    #     queue_right.set_state(Gst.State.NULL)
                    #     infer_pipe.remove(queue_right)


                # stop_pipeline(infer_pipe) # 停止再启动则造成内存泄漏，奇奇怪怪

                # =======================================================

                server.release()
                self.infer_server_dict.pop(service_id)
                # =======================================================

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
                source_pipe = self.source_pipeline_dict[channel_id]
                source_pipe.set_state(Gst.State.NULL)

                for service_id in Plumber.task_channel_dict.keys():
                    if channel_id in Plumber.task_channel_dict[service_id]:
                        # ======================================================
                        server = self.infer_server_dict[service_id]
                        infer_pipe = server.pipeline
                        # ===================================================

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
                        server.unbind(channel_id, service_id)

                self.source_pipeline_dict.pop(channel_id)
                del source_pipe

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
                # ===============================================================
                server = self.infer_server_dict[service_id]
                server.bind(service_id, channel_id, channel_name, config ,frame_cache)

                infer_pipe = server.pipeline
                # server.bind(channel_id, service_id) 
                # ================================================================


                source_pipe = self.source_pipeline_dict[channel_id]
                source_pipe.set_state(Gst.State.NULL)

                # #####################################>>>获取元素<<<###################################
                tee = source_pipe.get_by_name(f"tee_{channel_id}")

                appsink = Gst.ElementFactory.make("appsink", f"appsink_task{service_id}")
                appsink.set_property('emit-signals', True)
                # appsink.set_property('sync', False) # 当为False,切后推理速度小于拉流速度时，可能导致延时积累 
                appsink.set_property("enable-last-sample", False)
                appsink.set_property("drop", True)
                appsink.set_property("max_buffers", 10)
                # appsink.set_property("max-lateness", 1e+9)

                appsrc = Gst.ElementFactory.make("appsrc", f"appsrc_channel{channel_id}")
                appsrc.set_property('is-live', True)
                caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA, width=1920, height=1080, framerate=30/1")
                appsrc.set_property("caps", caps) #非常重要！！！！！！！

                # tee后边也要一个queue，不然在对新任务进行绑定时，没有缓冲造成重启失败
                queue_left = Gst.ElementFactory.make("queue", f"queue_task{service_id}_left")
                # queue_left.set_property("flush-on-eos", True)
                # queue_left.set_property("max-size-buffers", 10)
                # queue_left.set_property("leaky", 2)

                queue_right = Gst.ElementFactory.make("queue", f"queue_channel{channel_id}_right")
                # queue_right.set_property("flush-on-eos", True)
                # queue_right.set_property("max-size-buffers", 10)
                # queue_right.set_property("leaky", 2)

                # def on_overrun(queue):
                #     print(f"\n================={queue.name} overrun detected======================\n")
                # queue_left.connect("overrun", on_overrun)
                # queue_right.connect("overrun", on_overrun)

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

                source_pipe.set_state(Gst.State.PLAYING)

                return True
            except Exception as e:
                self.logging.info(f"error in bind: {e}")
                source_pipe.set_state(Gst.State.PLAYING)
                return False
        return wrapper

    def unbind_chan_task(self, f):
        def wrapper(*args):
            service_id = args[0]
            channel_id = args[1]
            try:
                # ===========================================================
                server = self.infer_server_dict[service_id]
                infer_pipe = server.pipeline
                # =========================================================

                # infer_pipe = self.infer_pipeline_dict[service_id]
                source_pipe = self.source_pipeline_dict[channel_id]

                # stop_pipeline(source_pipe)
                # source_pipe.set_state(Gst.State.NULL)

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
                    # tee_src_pad.send_event(Gst.Event.new_flush_stop(False))
                    tee.release_request_pad(tee_src_pad)

                if queue_left:
                    queue_left.set_state(Gst.State.NULL)
                    source_pipe.remove(queue_left)

                if appsink:
                    # appsink.emit("end-of-stream")
                    appsink.set_state(Gst.State.NULL)
                    source_pipe.remove(appsink)

                if appsrc:
                    appsrc.emit("end-of-stream")
                    appsrc.set_state(Gst.State.NULL)
                    infer_pipe.remove(appsrc)

                if queue_right:
                    queue_right.set_state(Gst.State.NULL)
                    infer_pipe.remove(queue_right)

                # source_pipe.set_state(Gst.State.PLAYING)
                # start_pipeline(source_pipe)
                # check_pipeline_elements(source_pipe)
                server.unbind(channel_id, service_id)

                return True
            except Exception as e:
                self.logging.info(f"error in unbind task{service_id} channel{channel_id}: {e}")
                return True #即使失败，链接结构也被破坏了
        return wrapper
    
    def get_media_info(self, f):
        def wrapper(*args):
            try:
                pass
            except Exception as e:
                return False
        return wrapper

    def run(self,):
        # Standard GStreamer initialization
        Gst.init(None)

        # perf callback function to print fps every 5 sec
        GLib.timeout_add(5000, self.perf_data.perf_print_callback)

        # create an event loop and feed gstreamer bus mesages to it
        loop = GLib.MainLoop()

        try:
            loop.run()
        except Exception as error:
            self.logging.info("error:", error)

        # if Plumber.stand_by == False:
        #     Plumber.stand_by = True            

        # cleanup
        self.logging.info("Exiting app\n")


# nohup python3 -u app.py >> output.log 2>&1 &
# nohup python3 -u grpc_client.py >> out_client.log 2>&1 &
