
#!/usr/bin/env python3
# ls tmp/stream0*.jpg|wc && ls tmp/stream1*.jpg|wc && ls tmp/stream2*.jpg|wc && ls tmp/stream3*.jpg|wc && ls tmp/stream4*.jpg|wc  && ls tmp/stream5*.jpg|wc  && ls tmp/stream6*.jpg|wc  && ls tmp/stream7*.jpg|wc
# ls stream0|wc && ls stream1|wc && ls stream2|wc && ls stream3|wc && ls stream4|wc  && ls stream5|wc  && ls stream6|wc  && ls stream7|wc
import copy
import json
import os
# modeset_cmd = "sudo modprobe nvidia-drm modeset=1"
# os.system(modeset_cmd)
# install_cmd = "sudo /opt/nvidia/deepstream/deepstream/install.sh"
# os.system(install_cmd)

import queue
import shutil
import sys
# sys.path.append('/root/apps')
# sys.path.append('/home/avcit/.local/lib/python3.8/site-packages')

import cv2
import multiprocessing

import numpy as np
sys.path.append('../')
import gi
import configparser
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
from ctypes import *
import time
from common.is_aarch_64 import is_aarch64
from common.FPS import PERF_DATA
import pyds
import logging

import ctypes
import cupy as cp

from pipe import Plumber

# 配置日志记录器
logging.basicConfig(filename='/root/apps/ai_server/deepstream.log', level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# ==================================多进程=================================
def worker(conn):
    
    k_src = None
    frame_num = None

    all_obj = {}
    alreay_counted = []
    last_frame_obj = None

    config = None
    if not config:
        with open("/root/apps/ai_server/cfg/config.json", 'r') as f:
            config = json.load(f) 
    
    while True:
        try:
            # 子进程会一直等待父进程发送数据
            data = conn.get()
            # continue
            last_frame_obj = data

            # 获取当前的k_src 和 frame_num
            for k in last_frame_obj.keys():
                k_src = last_frame_obj[k]["src_id"]
                frame_num = last_frame_obj[k]["frame_num"]
                break

            # 更新进all_obj
            for k in last_frame_obj.keys():

                if k == -1:
                    continue

                if k not in all_obj.keys():
                    all_obj[k] = [last_frame_obj[k]]
                else:
                    all_obj[k].append(last_frame_obj[k])

            sum_count = 0
            need_to_del = []
            for k_obj in all_obj.keys():

                if k_obj in alreay_counted:
                    need_to_del.append(k_obj)
                    # print("already counted", k_obj)
                    continue 

                obj_trace = []
                obj_conf = []
                obj_bbox = []
                obj_framenum = []
                obj_class = []
                for i in range(len(all_obj[k_obj])):
                    obj_trace.append(all_obj[k_obj][i]["trace"])
                    obj_conf.append(all_obj[k_obj][i]["conf"])
                    obj_bbox.append(all_obj[k_obj][i]["bbox"])
                    obj_framenum.append(all_obj[k_obj][i]["frame_num"])
                    obj_class.append(all_obj[k_obj][i]["class_id"])

                if frame_num - obj_framenum[-1] > 120:
                    need_to_del.append(k_obj)
                    # print("大于500", k_obj)
                    continue

                if not (len(all_obj[k_obj]) > 3) or 0 not in obj_class[-10:]:
                    # print("短于3, 且没人", k_obj)
                    continue  

                min_x = 1920
                min_y = 1080
                max_x = 0
                max_y = 0
                for point in obj_trace:
                    if point[0] < min_x:
                        min_x = point[0]
                    if point[1] < min_y:
                        min_y = point[1]
                    if point[0] > max_x:
                        max_x = point[0]
                    if point[1] > max_y:
                        max_y = point[1]
                if max_x - min_x < config["move_pix"] and max_y - min_y < config["move_pix"]:
                    # print("距离短于10", k_obj)
                    continue

                need_to_pack = False
                if frame_num - obj_framenum[-1] > config["track_time_s"]*6 :
                    need_to_pack = True

                if need_to_pack:
                # 这个k_obj需要截图
                    # 找到这个k_obj的trace, 
                    trace = obj_trace
                    cof_list = obj_conf[-1000:]

                    # 找到置信度的排名前五的索引
                    indexed_values = list(enumerate(cof_list))
                    sorted_indices = sorted(indexed_values, key=lambda x: (-x[1], x[0]))
                    top_five_indices = [index for index, _ in sorted_indices[:5]]

                    trace_mid_idx = -1
                    max_area = 0
                    # 哪个面积最大
                    for bbox_i in top_five_indices:
                        top, left, width, height = obj_bbox[-1000:][bbox_i]
                        if max_area < width*height:
                            max_area = width*height
                            trace_mid_idx = bbox_i
                    
                    # if trace_mid_idx == -1:
                    #     print("没有找到合适的bbox", k_obj)

                    trace_mid_frame = obj_framenum[-1000:][trace_mid_idx]

                    top, left, width, height = 0, 0, 0, 0

                    # 找到最近的图
                    near_frame = None
                    idx = None
                    for i in range(200):

                        img_name = "/root/apps/ai_server/stream{}_frame{}.jpg".format(k_src, trace_mid_frame+i)
                        if os.path.exists(img_name):
                            near_frame = trace_mid_frame+i
                            if near_frame in obj_framenum:
                                # 找到索引
                                idx = obj_framenum.index(near_frame)
                                top, left, width, height = obj_bbox[idx]

                            if top or left or width or height:
                                break

                        img_name = "/root/apps/ai_server/stream{}_frame{}.jpg".format(k_src, trace_mid_frame-i)
                        if os.path.exists(img_name):
                            near_frame = trace_mid_frame-i
                            if near_frame in obj_framenum:
                                # 找到索引
                                idx = obj_framenum.index(near_frame)
                                top, left, width, height = obj_bbox[idx]

                            if top or left or width or height:
                                break

                    # 判断top, left, width, height是否有效
                    if top == 0 and left == 0 and width == 0 and height == 0:
                        continue

                    alreay_counted.append(k_obj)
                    if len(alreay_counted) > 1000:
                        alreay_counted.pop(0)

                    need_to_del.append(k_obj)

                    try:
                        # cv2读取图片
                        src_data = cv2.imread(f"/root/apps/ai_server/stream{k_src}_frame{near_frame}.jpg")
                        roi_data = copy.deepcopy(src_data[max(0,top-5): min(1080,top+height+5), max(0,left-5): min(1920,left+width+5)])
                    except:
                        continue

                    # 画框
                    cv2.rectangle(src_data, (left, top), (left+width, top+height), (0, 255, 0), 3)

                    # 画轨迹
                    try:
                        for i in range(len(trace)-1):
                                cv2.line(src_data, (int(trace[i][0]), int(trace[i][1])),
                                            (int(trace[i+1][0]), int(trace[i+1][1])), (0, 255, 0), 3)
                    except:
                        # print("draw line error")
                        pass

                    # 写标题
                    cv2.putText(src_data, "{}".format(k_obj), (left+10, top-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                    # 保存图片
                    cv2.imwrite(f"/root/apps/ai_server/stream{k_src}/stream{k_src}_frame{near_frame}_{k_obj}_roi_tmp.jpg", roi_data)
                    cv2.imwrite(f"/root/apps/ai_server/stream{k_src}/stream{k_src}_frame{near_frame}_{k_obj}_src_tmp.jpg", src_data)
                    shutil.move(f"/root/apps/ai_server/stream{k_src}/stream{k_src}_frame{near_frame}_{k_obj}_roi_tmp.jpg", f"/root/apps/ai_server/stream{k_src}/stream{k_src}_frame{near_frame}_{k_obj}_roi.jpg")
                    shutil.move(f"/root/apps/ai_server/stream{k_src}/stream{k_src}_frame{near_frame}_{k_obj}_src_tmp.jpg", f"/root/apps/ai_server/stream{k_src}/stream{k_src}_frame{near_frame}_{k_obj}_src.jpg")

                    

                    # 每次处理 1个k_obj
                    sum_count += 1
                    if sum_count == 1:
                        break

            # 删除已经处理过的k_obj
            for k_obj in need_to_del:
                all_obj.pop(k_obj)

        except EOFError:
            print("Worker received EOF, exiting...")
            # 继续处理数据
            continue
    print("Worker exiting...")
# ==================================多进程================================end


# ======================================grpc_main=====================================
import threading
import grpc
import time
from concurrent import futures
import proto.task_pb2 as task_pb2
import proto.global_pb2 as global_pb2
from proto.task_pb2_grpc import (
    TaskServiceServicer,
    add_TaskServiceServicer_to_server,
)  # 替换实际的服务名


class TaskServiceImpl(TaskServiceServicer):

    def GetRunningState(self, request, context):
        state = task_pb2.RunningState()
        for i in range(len(Plumber.GetRunningState_list)):

            if Plumber.GetRunningState_list[i] == []:
                continue

            state.channels.append(
                task_pb2.Channel(id=Plumber.GetRunningState_list[i][0], 
                                 name=Plumber.GetRunningState_list[i][1], 
                                 url=Plumber.GetRunningState_list[i][2])
                )

        return state

    def Enable(self, request, context):

        # global stand_by

        logging.info(">>>accept enabel " + str(request.collect_alg.channel.url))
        id = request.collect_alg.channel.id
        name = request.collect_alg.channel.name
        url = request.collect_alg.channel.url

        if not Plumber.stand_by:
            time.sleep(5)
            # return global_pb2.CommRes(code=1, msg="not stand by")

        # 使用cv2尝试打开rtsp
        # os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;5000"
        # cap = cv2.VideoCapture(url)
        # if not cap.isOpened():
        #     logging.info(">>>rtsp open failed")
        #     return global_pb2.CommRes(code=1, msg="rtsp open failed")
        
        if Plumber.num_sources == 8:
            logging.info(">>>max support 8 source")
            return global_pb2.CommRes(code=1, msg="max support 8 source")

        if [id, name, url] in Plumber.GetRunningState_list:
            logging.info(">>>rtsp already added")
            return global_pb2.CommRes(code=1, msg="rtsp already added")

        if [id, name, url] not in Plumber.GetRunningState_list:
            for i in range(len(Plumber.GetRunningState_list)):
                if Plumber.GetRunningState_list[i] == []:
                    Plumber.GetRunningState_list[i] = [id, name, url]

                    reval = add_sources([i, url])
                    
                    if reval == 1:
                        logging.info(">>>rtsp success added")
                        return global_pb2.CommRes(code=0, msg="success")
                    else:
                        logging.info(">>>fail check add_sources func")
                        return global_pb2.CommRes(code=1, msg="fail check add_sources func")
                    
    def Disable(self, request, context):
        """TODO"""

        if Plumber.num_sources == 1:
            logging.info(">>>at least one rtsp source")
            return global_pb2.CommRes(code=1, msg="at least one rtsp source")

        logging.info(">>>accept enabel " + str(request.collect_alg.channel.id))
        id = request.collect_alg.channel.id
        name = request.collect_alg.channel.name
        url = request.collect_alg.channel.url

        for i in range(len(Plumber.GetRunningState_list)):
            if Plumber.GetRunningState_list[i] == []:
                continue
            if id == Plumber.GetRunningState_list[i][0]:
                Plumber.GetRunningState_list[i] = []

                reval = delete_sources(i)
                time.sleep(0.5)
                if reval == 1:
                    logging.info(">>>rtsp success removed")
                    return global_pb2.CommRes(code=0, msg="success")
                else:
                    logging.info(">>>fail check delete_sources func")
                    return global_pb2.CommRes(code=1, msg="fail check delete_sources func")
        logging.info(">>>rtsp not exsit")
        return global_pb2.CommRes(code=1, msg="rtsp not exsit")

    def Reset(self, request, context):
        cmd = r"ps -ef | grep ai_server_fullscreen_good.py | grep -v grep | awk '{print $2}' | xargs kill -9"
        os.system(cmd)
        
        return global_pb2.CommRes(code=0, msg="success")
    
    def Notify(self, request, context):
        """TODO"""
        # global lock
        try:
            while context.is_active():
                # lock.acquire()
                # try:
                    time.sleep(0.08)
                    if not Plumber.task_queue.empty():
                        yield Plumber.task_queue.get()
                # finally:
                    # 释放锁
                    # lock.release()
                
        except GeneratorExit as e:
            print(f"Error occurred: {e}")
        finally:
            print("Subscribe ended")
            return

def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10)
    )  # 创建gRPC服务器实例
    add_TaskServiceServicer_to_server(
        TaskServiceImpl(), server
    )  # 将服务实现添加到服务器上

    # 指定监听地址和服务端口
    server.add_insecure_port("[::]:50151")

    server.start()  # 启动服务器
    logging.info("Server started on port 50051...")
    server.wait_for_termination()  # 阻塞等待，直到服务器关闭

t = threading.Thread(target=serve, args=())
t.start()

# ======================================grpc_main=====================================end


# ==================================多进程=================================
# 创建一个双向通信的管道
child_conn_0 = multiprocessing.Queue()
child_conn_1 = multiprocessing.Queue()
child_conn_2 = multiprocessing.Queue()
child_conn_3 = multiprocessing.Queue()
child_conn_4 = multiprocessing.Queue()
child_conn_5 = multiprocessing.Queue()
child_conn_6 = multiprocessing.Queue()
child_conn_7 = multiprocessing.Queue()
child_conn_8 = multiprocessing.Queue()
child_conn_9 = multiprocessing.Queue()
child_conn_10 = multiprocessing.Queue()
child_conn_11 = multiprocessing.Queue()
child_conn_12 = multiprocessing.Queue()
child_conn_13 = multiprocessing.Queue()
child_conn_14 = multiprocessing.Queue()
child_conn_15 = multiprocessing.Queue()
all_conn = [child_conn_0, child_conn_1, child_conn_2, child_conn_3,
            child_conn_4, child_conn_5, child_conn_6, child_conn_7,
            child_conn_8, child_conn_9, child_conn_10, child_conn_11,
            child_conn_12, child_conn_13, child_conn_14, child_conn_15]

# 创建子进程
p0 = multiprocessing.Process(target=worker, args=(child_conn_0,))
p1 = multiprocessing.Process(target=worker, args=(child_conn_1,))
p2 = multiprocessing.Process(target=worker, args=(child_conn_2,))
p3 = multiprocessing.Process(target=worker, args=(child_conn_3,))
p4 = multiprocessing.Process(target=worker, args=(child_conn_4,))
p5 = multiprocessing.Process(target=worker, args=(child_conn_5,))
p6 = multiprocessing.Process(target=worker, args=(child_conn_6,))
p7 = multiprocessing.Process(target=worker, args=(child_conn_7,))
p8 = multiprocessing.Process(target=worker, args=(child_conn_8,))
p9 = multiprocessing.Process(target=worker, args=(child_conn_9,))
p10 = multiprocessing.Process(target=worker, args=(child_conn_10,))
p11 = multiprocessing.Process(target=worker, args=(child_conn_11,))
p12 = multiprocessing.Process(target=worker, args=(child_conn_12,))
p13 = multiprocessing.Process(target=worker, args=(child_conn_13,))
p14 = multiprocessing.Process(target=worker, args=(child_conn_14,))
p15 = multiprocessing.Process(target=worker, args=(child_conn_15,))

p0.start()
p1.start()
p2.start()
p3.start()
p4.start()
p5.start()
p6.start()
p7.start()
p8.start()
p9.start()
p10.start()
p11.start()
p12.start()
p13.start()
p14.start()
p15.start()
# ====================================多进程=================================end


# =====================程序开始执行=================================
plumber = Plumber(all_conn, logging)

@plumber.add_sources
def add_sources():
    # print("decorator add sources")
    pass

@plumber.delete_sources
def delete_sources():
    # print("decorator delete sources")
    pass