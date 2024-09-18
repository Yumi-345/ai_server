# -*- coding: utf-8 -*-

# import grpc
# import time
# from concurrent import futures
# import proto.task_pb2 as task_pb2
# import proto.global_pb2 as global_pb2
# from proto.task_pb2_grpc import (
#     TaskServiceStub,
#     add_TaskServiceServicer_to_server,
# )

# from google.protobuf import empty_pb2

import grpc
import time
from proto.alg_core.core_pb2 import (
    EnableChannelReq,
    DisableChannelReq,
    AddTaskReq,
    RemoveTaskReq,
    BindChanTaskReq,
    UnbindChanTaskReq,
    Event,
)
from proto.alg_core.core_pb2_grpc import CoreServiceStub
from google.protobuf import empty_pb2

MAX_MESSAGE_LENGTH = 256*1024*1024  # 可根据具体需求设置，此处设为256M

PORT = 50051

def create_channel():
    return grpc.insecure_channel(target=f"172.20.38.202:{PORT}")

# 创建客户端
stub = CoreServiceStub(create_channel())



url_17_floor = [
    "rtsp://admin:avcit12345678@172.20.18.252:554/ch33/main/av_stream", # 17F消防电梯前室
    "rtsp://admin:avcit12345678@172.20.18.252/ch34/main/av_stream", # 17F南C办公区
    "rtsp://admin:avcit12345678@172.20.18.252/ch35/main/av_stream", # 17F南A办公区
    "rtsp://admin:avcit12345678@172.20.18.252/ch36/main/av_stream", # 17F北A办公区
    "rtsp://admin:avcit12345678@172.20.18.252/ch37/main/av_stream", # 17F北B办公区
    "rtsp://admin:avcit12345678@172.20.18.252/ch38/main/av_stream", # 17F楼梯口
    "rtsp://admin:avcit12345678@172.20.18.252/ch39/main/av_stream", # 17F南B办公区
    "rtsp://admin:avcit12345678@172.20.18.252/ch40/main/av_stream", # 17F南D办公区
    "rtsp://admin:avcit12345678@172.20.18.252/ch41/main/av_stream", # 17F南C办公区
    "rtsp://admin:avcit12345678@172.20.18.252/ch42/main/av_stream", # 17F电梯厅
    "rtsp://admin:avcit12345678@172.20.18.252/ch43/main/av_stream", # 17F西过道
    "rtsp://admin:avcit12345678@172.20.18.252/ch44/main/av_stream", # 17F东过道
    "rtsp://admin:avcit12345678@172.20.18.252/ch45/main/av_stream", # Camera 01
]

other = [
    "rtsp://admin:avcit12345678@172.20.18.252/ch46/main/av_stream",#1楼消防梯前室
    "rtsp://admin:avcit12345678@172.20.18.252/ch47/main/av_stream",
    "rtsp://admin:avcit12345678@172.20.18.252/ch48/main/av_stream",
    "rtsp://admin:avcit12345678@172.20.18.252/ch49/main/av_stream",
    "rtsp://admin:avcit12345678@172.20.18.252/ch50/main/av_stream",
]


t = 20
num = 8
while 1:
    # 调用enable接口
    for index, url in enumerate(url_17_floor + other):
        if index > num-1:
            break
        counter_vehicle_alg = EnableChannelReq(channel_id=index, channel_name="穷哈哈", channel_url=url)

        resp_enable = stub.EnableChannel(counter_vehicle_alg)
        print(url)

    # print(f"休息{t}s\n\n")
    # time.sleep(t)

    print("add task")
    stub.AddTask(AddTaskReq(service_id=0, dev_id=0, root="test"))
    # stub.AddTask(AddTaskReq(service_id=1, dev_id=0, root="test"))
    # print(f"休息{t}s\n\n")
    # time.sleep(t)

    print("bind")
    for index in range(num):
        stub.BindChanTask(BindChanTaskReq(service_id=0, channel_id=index, channel_name="穷哈哈", config="test"))


    # stub.BindChanTask(BindChanTaskReq(service_id=0, channel_id=0, channel_name="穷哈哈", config="test"))
    # stub.BindChanTask(BindChanTaskReq(service_id=0, channel_id=1, channel_name="穷哈哈", config="test"))
    # stub.BindChanTask(BindChanTaskReq(service_id=0, channel_id=2, channel_name="穷哈哈", config="test"))
    # stub.BindChanTask(BindChanTaskReq(service_id=0, channel_id=3, channel_name="穷哈哈", config="test"))

    # stub.BindChanTask(BindChanTaskReq(service_id=1, channel_id=0, channel_name="穷哈哈", config="test"))
    # stub.BindChanTask(BindChanTaskReq(service_id=1, channel_id=1, channel_name="穷哈哈", config="test"))
    # stub.BindChanTask(BindChanTaskReq(service_id=1, channel_id=2, channel_name="穷哈哈", config="test"))
    # stub.BindChanTask(BindChanTaskReq(service_id=1, channel_id=3, channel_name="穷哈哈", config="test"))

    print(f"休息{t}s\n\n")
    time.sleep(t)

    # print("unbind")
    # for index in range(num):
    #     stub.UnbindChanTask(UnbindChanTaskReq(service_id=0, channel_id=index))

    # stub.UnbindChanTask(UnbindChanTaskReq(service_id=0, channel_id=0))
    # stub.UnbindChanTask(UnbindChanTaskReq(service_id=0, channel_id=1))
    # stub.UnbindChanTask(UnbindChanTaskReq(service_id=0, channel_id=2))
    # stub.UnbindChanTask(UnbindChanTaskReq(service_id=0, channel_id=3))

    # stub.UnbindChanTask(UnbindChanTaskReq(service_id=1, channel_id=0))
    # stub.UnbindChanTask(UnbindChanTaskReq(service_id=1, channel_id=1))
    # stub.UnbindChanTask(UnbindChanTaskReq(service_id=1, channel_id=2))
    # stub.UnbindChanTask(UnbindChanTaskReq(service_id=1, channel_id=3))
    # print(f"休息{t}s\n\n")
    # time.sleep(t)

    # print("remove task")
    # stub.RemoveTask(RemoveTaskReq(service_id=0))
    # stub.RemoveTask(RemoveTaskReq(service_id=1))
    # print(f"休息{t}s\n\n")
    # time.sleep(t)

    # print("disable channel")
    # stub.DisableChannel(DisableChannelReq(channel_id=0))
    # stub.DisableChannel(DisableChannelReq(channel_id=1))
    # stub.DisableChannel(DisableChannelReq(channel_id=2))
    # stub.DisableChannel(DisableChannelReq(channel_id=3))
    
    print(f"休息{t}s\n\n")
    time.sleep(t)
