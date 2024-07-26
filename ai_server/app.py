#!/usr/bin/env python3

import sys
sys.path.append('/root/apps')
sys.path.append('/root/apps/ai_server')

# sys.path.append('../')
import gi
gi.require_version('Gst', '1.0')
# from ctypes import *
import time
import logging

from pipe import Plumber


# 配置日志记录器
logging.basicConfig(filename='/root/apps/ai_server/deepstream.log', level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(message)s')


# ======================================grpc_main=====================================
import threading
import grpc
from concurrent import futures
from proto.alg_core.core_pb2 import (
    EnableChannelReq,
    DisableChannelReq,
    AddTaskReq,
    RemoveTaskReq,
    BindChanTaskReq,
    UnbindChanTaskReq,
    Event,
)
import proto.alg_core.common_pb2 as common_pb2
from proto.alg_core.common_pb2 import MediaInfo
from proto.alg_core.core_pb2_grpc import (
    CoreServiceServicer,
    add_CoreServiceServicer_to_server,
)
from google.protobuf import empty_pb2


LOCK = threading.Lock()

class TaskServiceImpl(CoreServiceServicer):

    def AddTask(self, request:AddTaskReq, context):
        try:
            LOCK.acquire()
            service_id = request.service_id
            dev_id = request.dev_id
            root = request.root
            logging.info(f">>>accept add task {service_id}")
            # if not Plumber.stand_by:
            #     time.sleep(2)

            if service_id in Plumber.task_list:
                logging.info(f">>>task {service_id} already added")

            else:
                reval = add_task(service_id, dev_id, root)
                if reval == 1:
                    logging.info(f">>>task{service_id} success added")
                    Plumber.task_list.append(service_id)
                else:
                    logging.info(f">>>failled add task{service_id}")
        except Exception as e:
            logging.info(f"Error processing add task: {e}")
        finally:
            LOCK.release()
        return empty_pb2.Empty()
    
    def RemoveTask(self, request:RemoveTaskReq, context):
        try:
            LOCK.acquire()
            service_id  = request.service_id
            if service_id in Plumber.task_list:
                reval = remove_task(service_id)
                if reval:
                    Plumber.task_channel_dict.pop(service_id)
                    Plumber.task_list.remove(service_id)
                    logging.info(f"success remove task{service_id}")
                else:
                    logging.info(f"faild remove task{service_id}")
            else:
                logging.info(f"task{service_id} not exist")
        except Exception as e:
            logging.info(f"error in remove task:{e}")
        finally:
            LOCK.release()
        return empty_pb2.Empty()

    def EnableChannel(self, request: EnableChannelReq, context):
        try:
            LOCK.acquire()
            channel_id = request.channel_id
            channel_name = request.channel_name
            channel_url = request.channel_url
            logging.info(f">>>accept enabel channel {channel_id}")

            if not Plumber.stand_by:
                time.sleep(2)
                # return global_pb2.CommRes(code=1, msg="not stand by")

            # 使用cv2尝试打开rtsp
            # os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;5000"
            # cap = cv2.VideoCapture(url)
            # if not cap.isOpened():
            #     logging.info(">>>rtsp open failed")
            #     return global_pb2.CommRes(code=1, msg="rtsp open failed")
            
            if channel_id in Plumber.channel_list:
                logging.info(f">>>channel{channel_id} already added")

            else:
                reval = enable_channel(channel_id, channel_name, channel_url)
                if reval == 1:
                    logging.info(f">>>channel{channel_id} success added")
                    Plumber.channel_list.append(channel_id)
                else:
                    logging.info(f">>>failled enable channel {channel_id}")
        except Exception as e:
            logging.info(f"Error processing EnableChannel: {e}")
        finally:
            LOCK.release()
        return empty_pb2.Empty()

    def DisableChannel(self, request:DisableChannelReq, context):
        try:
            LOCK.acquire()
            channel_id = request.channel_id
            if channel_id in Plumber.channel_list:
                reval = disable_channel(channel_id)
                if reval:
                    logging.info(f"success disable channel{channel_id}")
                    Plumber.channel_list.remove(channel_id)
                else:
                    logging.info(f"faild disable channel{channel_id}")
            else:
                logging.info(f"channel{channel_id} not in enabled")
        except Exception as e :
            logging.info(f"error in disablechannel:{e}")
        finally:
            LOCK.release()
        return empty_pb2.Empty()
    
    def BindChanTask(self, request:BindChanTaskReq, context):
        try:
            LOCK.acquire()
            service_id = request.service_id
            channel_id = request.channel_id
            channel_name = request.channel_name
            config = request.config
            if service_id in Plumber.task_list and channel_id in Plumber.channel_list:
                if service_id in Plumber.task_channel_dict.keys():
                    if channel_id in Plumber.task_channel_dict[service_id]:
                        logging.info(f"already bind task{service_id} with channel{channel_id}")
                    else:
                        revel = bind_chan_task(service_id, channel_id, channel_name, config)
                        if revel:
                            Plumber.task_channel_dict[service_id].append(channel_id)
                else:
                    revel = bind_chan_task(service_id, channel_id, channel_name, config)
                    if revel:
                        Plumber.task_channel_dict[service_id] = [channel_id]
            else:
                logging.info(f"task{service_id} or channel{channel_id} not added befor bind")
        except Exception as e:
            logging.info(f"Error processing bind task{service_id} and channel{channel_id}: {e}")
        finally:
            LOCK.release()
        return empty_pb2.Empty()
    
    def UnbindChanTask(self, request:UnbindChanTaskReq, context):
        try:
            LOCK.acquire()
            service_id = request.service_id
            channel_id = request.channel_id
            if service_id in Plumber.task_channel_dict.keys():
                if channel_id in Plumber.task_channel_dict[service_id]:
                    revel = unbind_chan_task(service_id, channel_id)
                    if revel:
                        Plumber.task_channel_dict[service_id].remove(channel_id)
                        logging.info(f"succesful unbind task{service_id} channel{channel_id}")
                    else:
                        logging.info(f"faild unbind task{service_id} channel{channel_id}")
                        return empty_pb2.Empty()
                else:
                    logging.info(f"channel{channel_id} is not bind with task{service_id}")
            else:
                logging.info(f"task{service_id} is not in pipeline, fiald unbind")
        except Exception as e:
            logging.info(f"Error processing unbind task{service_id} and channel{channel_id}: {e}")
        finally:
            LOCK.release()
        return empty_pb2.Empty()


def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10)
    )  # 创建gRPC服务器实例
    add_CoreServiceServicer_to_server(
        TaskServiceImpl(), server
    )  # 将服务实现添加到服务器上

    # 指定监听地址和服务端口
    server.add_insecure_port("[::]:50051")

    server.start()  # 启动服务器
    logging.info("Server started on port 50051...")
    server.wait_for_termination()  # 阻塞等待，直到服务器关闭

t = threading.Thread(target=serve, args=())
t.start()

# ======================================grpc_main=====================================end


# =====================程序开始执行=================================
plumber = Plumber(logging)

@plumber.add_task
def add_task():
    pass

@plumber.remove_task
def remove_task():
    pass

@plumber.enable_channel
def enable_channel():
    # print("decorator add sources")
    pass

@plumber.disable_channel
def disable_channel():
    # print("decorator delete sources")
    pass

@plumber.bind_chan_task
def bind_chan_task():
    pass

@plumber.unbind_chan_task
def unbind_chan_task():
    pass

# @plumber.get_media_info
# def get_media_info():
#     pass
