# /bin/python3
Author = "xuwj"

import gi
import configparser
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import sys
sys.path.append('/root/apps')
sys.path.append('/root/apps/ai_server')

import pyds
import cv2 
import time
import queue
import threading
import numpy as np

from service.base_server import BaseServer
# from pipe import Plumber
from utils.sort import BoxTracker

from utils.utils import SafeLock


MUXER_OUTPUT_WIDTH=1920
MUXER_OUTPUT_HEIGHT=1080

PGIE_CONFIG_FILE = '/root/apps/ai_server/cfg/yolov8n.txt'
TRACKER_CONFIG_FILE = "/root/apps/ai_server/cfg/dstest_tracker_config.txt"
# 在这里实现一个继承与boxtracker的类，基础功能复用boxtracker，发送规则则根据每个server不同而自定义 


# 将 [left, top, width, height] 转换为 [x_min, y_min, x_max, y_max]
def lthw_to_xyxy(boxes):
    return np.hstack((boxes[:, 0:1], boxes[:, 1:2], boxes[:, 0:1] + boxes[:, 2:3], boxes[:, 1:2] + boxes[:, 3:4]))

# 计算box1和box2之间的重合度矩阵
def compute_overlap_matrix(boxes1, boxes2):
    boxes1_xyxy = lthw_to_xyxy(boxes1)
    boxes2_xyxy = lthw_to_xyxy(boxes2)
    
    # 计算交集区域
    x_min_inter = np.maximum(boxes1_xyxy[:, np.newaxis, 0], boxes2_xyxy[:, 0])
    y_min_inter = np.maximum(boxes1_xyxy[:, np.newaxis, 1], boxes2_xyxy[:, 1])
    x_max_inter = np.minimum(boxes1_xyxy[:, np.newaxis, 2], boxes2_xyxy[:, 2])
    y_max_inter = np.minimum(boxes1_xyxy[:, np.newaxis, 3], boxes2_xyxy[:, 3])
    
    # 交集面积
    inter_area = np.clip(x_max_inter - x_min_inter, 0, None) * np.clip(y_max_inter - y_min_inter, 0, None)
    
    # box1和box2的面积
    box1_area = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]) * (boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1])
    box2_area = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]) * (boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1])
    
    # 计算重合度（交集面积除以最小的box面积）
    overlap_matrix = inter_area / np.minimum(box1_area[:, np.newaxis], box2_area)
    
    return overlap_matrix

# 找到每个box1的最佳匹配box2，设置最低重合度阈值
def find_best_match(boxes1, boxes2, overlap_threshold=0.5):
    overlap_matrix = compute_overlap_matrix(boxes1, boxes2)
    
    # 找到每个box1的最佳匹配（重合度最大）
    best_matches = np.argmax(overlap_matrix, axis=1)
    best_overlaps = overlap_matrix[np.arange(overlap_matrix.shape[0]), best_matches]
    
    # 如果最佳匹配的重合度小于阈值，则视为未匹配
    best_matches[best_overlaps < overlap_threshold] = -1  # -1表示未匹配
    best_overlaps[best_overlaps < overlap_threshold] = 0  # 低于阈值的重合度设为0
    
    return best_matches, best_overlaps


class MultiBoxTransfer:
    def __init__(self):
        self.queue = queue.Queue(maxsize=30)
        self.is_alive = True
        self.track_class_list = [0, 1, 3, 4, 5, 7]
        self.match_pairs = [0, 2]
        self.tracker_dict = {}
        self.box_lock = SafeLock(10)
        t = threading.Thread(target=self.run, daemon=True)
        t.start()

    def put(self, msg):
        if self.queue.full():
            print(f"多目标跟踪queue已满，舍弃最旧元素")
            self.queue.get()
        self.queue.put(msg)

    def run(self, ):
        while True:
            try:
                if not self.is_alive:
                    break
                if not self.queue.empty():
                    msg = self.queue.get()
                    img_arr = msg["img"]
                    src_id = msg["src_id"]
                    frame_number = msg["frame_number"]
                    service_id = msg["service_id"]

                    best_matches = None
                    if all([key in msg["objs"].keys() for key in self.match_pairs]):
                        boxes0 = np.array(msg["objs"][self.match_pairs[0]])[:, 2:6]
                        boxes1 = np.array(msg["objs"][self.match_pairs[1]])[:, 2:6]
                        best_matches, best_ious = find_best_match(boxes0, boxes1, 0.9)
                        # print(boxes0, boxes1, best_matches, best_ious)
                    for key in msg["objs"].keys():
                        if key in self.track_class_list:
                            objs = msg["objs"][key]
                            for index, obj in enumerate(objs):
                                child_obj = None
                                u_id = f"{service_id}_{src_id}_{obj[0]}"
                                if key == self.match_pairs[0]:
                                    if best_matches is not None:
                                    #对进行了匹配的目标进行处理
                                        if best_matches[index] != -1:
                                            child_obj = msg["objs"][self.match_pairs[-1]][best_matches[index]]
                                
                                # 更新tracker或者初始化tracker
                                self.box_lock.acquire()
                                if u_id in self.tracker_dict.keys():
                                    self.tracker_dict[u_id].update(
                                        img_arr, 
                                        u_id, 
                                        *obj[1:], 
                                        key, 
                                        src_id, 
                                        frame_number, 
                                        child_obj
                                        )
                                else:
                                    self.tracker_dict[u_id] = BoxTracker(
                                        img_arr, 
                                        u_id, 
                                        *obj[1:], 
                                        key, 
                                        src_id, 
                                        frame_number, 
                                        None,  # config预留位置
                                        child_obj
                                        )
                                self.box_lock.release()
                        
                    # 清除目标
                    self.box_lock.acquire()        
                    n2d = []
                    for key in self.tracker_dict.keys():
                        if str(src_id) == key.split("_")[1]:
                            if frame_number - self.tracker_dict[key].frame_number > 20: #判断是否达到发送要求
                                flag, info = self.tracker_dict[key].send(save=True)
                                # print(info)
                                if frame_number - self.tracker_dict[key].frame_number > 100 or flag:
                                    n2d.append(key)
                    for index in n2d:
                        self.tracker_dict.pop(index)
                    self.box_lock.release()

            except Exception as e:
                print(f"error in multi box transfer:{e}")

    def release(self, ):
        self.is_alive = False
        self.tracker_dict.clear()

    def unbind(self, channel_id, task_id):
        # self.tracker_dict.clear()
        # 清除目标
        self.box_lock.acquire()        
        n2d = []
        for key in self.tracker_dict.keys():
            if str(channel_id) == key.split("_")[1]:
                n2d.append(key)
        for index in n2d:
            self.tracker_dict.pop(index)
        self.box_lock.release()


    # def __del__(self):
    #     print(f"真正的死亡就是：被遗忘，仿佛从来没有来过")


class SleepingWorkstation(BaseServer):
    def __init__(self, service_id, dev_id, root, logging, perf_data):
        self.service_id = service_id
        self.dev_id = dev_id
        self.root = root
        self.logging = logging
        # self.channel_box_dict = {}
        self.tracker_dict = {}
        self.perf_data = perf_data
        # self.tracker_lock = SafeLock(10)
        
        # ==============
        self.config_dict = {}
        self.multi_box = MultiBoxTransfer()

        self.pipeline = self.create_pipeline()

    def nvdrmvideosink_probe(self, pad, info, u_data):
        # 此处单线程会阻塞，导致处理速度变慢，考虑多线程处理多任务
        src_id = 0 
        frame_number=0
        gst_buffer = info.get_buffer()
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

        l_frame = batch_meta.frame_meta_list

        while l_frame is not None:
            # ==================
            msg = {}
            # =====================
            try:
                frame_meta = pyds.glist_get_nvds_frame_meta(l_frame.data)
            except StopIteration:
                break
            src_id = frame_meta.source_id
            # src_id = frame_meta.pad_index
            self.perf_data.update_fps("stream"+str(src_id))

            frame_number = frame_meta.frame_num
            t0 = time.time()

            n_frame_cpu = self.get_data_GPU(gst_buffer, frame_meta)
            img_arr = cv2.cvtColor(n_frame_cpu, cv2.COLOR_RGBA2BGRA)
            # print(f"=========================={time.time()-t0}================")

            l_obj = frame_meta.obj_meta_list
            
            # =============
            msg["src_id"] = src_id
            msg["frame_number"] = frame_number
            msg["img"] = img_arr
            msg["objs"] = {}
            msg["service_id"] = u_data
            # =====================

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
                    # obj_meta.text_params.text_bg_clr.set(0.5, 0.0, 0.5, 0.6)  # 设置显示背景颜色

                    rect_params = obj_meta.rect_params
                    top = int(rect_params.top)
                    left = int(rect_params.left)
                    width = int(rect_params.width)
                    height = int(rect_params.height)
                    class_id = obj_meta.class_id
                    confidence = obj_meta.confidence

                    if left < 10 or top < 10 or left + width > 1920 - 10 or top + height > 1080 - 10:
                        l_obj = l_obj.next
                        continue

                    # 计算框中点
                    center_x = left + width / 2
                    center_y = top + height / 2

                    if class_id in msg["objs"].keys():
                        msg["objs"][class_id].append([object_id, confidence, left, top, width, height, center_x, center_y])
                    else:
                        msg["objs"][class_id] = [[object_id, confidence, left, top, width, height, center_x, center_y]]
                    # ===================
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
            # ==============
            self.multi_box.put(msg)
            # ============

            try:
                l_frame = l_frame.next
            except StopIteration:
                break
        return Gst.PadProbeReturn.OK

    def bus_call(self, bus, message):
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
    
    def create_pipeline(self, ):
        pipeline = Gst.Pipeline.new(f"infer_pipe{self.service_id}")

        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect ("message", self.bus_call)

        pipeline.set_state(Gst.State.NULL)

        self.logging.info(f"Creating streammux_{self.service_id}\n ")
        # Create nvstreammux instance to form batches from one or more sources.
        streammux = Gst.ElementFactory.make("nvstreammux", f"Stream-muxer-{self.service_id}")
        if not streammux:
            sys.stderr.write(" Unable to create NvStreamMux \n")

        '''
        这里batch大于视频流个数会导致NVstreammux每个流取一帧后开始等待，直至超时；
        一旦超时设置的帧率小于实际拉流帧率，进而导致延时积累
        '''
        streammux.set_property("batched-push-timeout", 30000) # 这里的处理速度如果小于发送的速度，会导致延迟
        # streammux.set_property("frame-duration", 0)
        streammux.set_property("batch-size", 8) # 重要点，此处的batch-size,将会影响探针处的l_frame个数，也就是说探针如果不设置为多线程，将会影响并发性能

        streammux.set_property("gpu_id", self.dev_id)
        streammux.set_property("live-source", 1)

        #Set streammux width and height
        streammux.set_property('width', MUXER_OUTPUT_WIDTH)
        streammux.set_property('height', MUXER_OUTPUT_HEIGHT)

        '''
        sync-inputs设置为False之后可以动态链接，但是断开之后再重连会连接不上(通过drop-pipeline-eos设置为True可以解决)
        设置为True则绑定需要暂停管道，猜测是因为要求数据同步，所以需要初始化管道
        '''
        # streammux.set_property("drop-pipeline-eos", True)
        streammux.set_property("sync-inputs", False) #很重要！！！！！！！！！！！！！！！！

        pipeline.add(streammux)

        self.logging.info("Creating Pgie \n ")
        pgie = Gst.ElementFactory.make("nvinfer", f"primary_inference{self.service_id}")
        if not pgie:
            sys.stderr.write(" Unable to create pgie \n")

        self.logging.info("Creating nvtracker \n ")
        tracker = Gst.ElementFactory.make("nvtracker", f"tracker_{self.service_id}")
        if not tracker:
            sys.stderr.write(" Unable to create tracker \n")

        tracker_srcpad = tracker.get_static_pad("src")
        if not tracker_srcpad:
            sys.stderr.write(" Unable to get sink pad of nvosd \n")
        tracker_srcpad.add_probe(Gst.PadProbeType.BUFFER, self.nvdrmvideosink_probe, self.service_id)

        self.logging.info("Creating sink \n ")
        sink = Gst.ElementFactory.make("fakesink", f"task_fakesink_{self.service_id}")
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

        pipeline.set_state(Gst.State.PLAYING)
        return pipeline

    def bind(self, task_id, channel_id, channel_name, config):
        self.config_dict[channel_id] = config

    def release(self):
        # if self.probe_id is not None:
        #     tracker = self.pipeline.get_by_name(f"tracker_{self.service_id}")
        #     tracker_srcpad = tracker.get_static_pad("src")
        #     tracker_srcpad.remove_probe(self.probe_id)
        self.multi_box.release()
        # del self.multi_box

    def unbind(self, channel_id, task_id):
        self.multi_box.unbind(channel_id, task_id)

    def __del__(self):
        print(f"真正的死亡就是：被遗忘，仿佛从没来过")