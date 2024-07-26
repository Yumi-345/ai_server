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

def create_channel():
    return grpc.insecure_channel(target="127.0.0.1:50051")

# 创建客户端
stub = CoreServiceStub(create_channel())



url_9_floor = [
        "rtsp://admin:avcit12345678@192.168.19.100/ch39/main/av_stream",
        "rtsp://admin:avcit12345678@192.168.19.100/ch33/main/av_stream",
        # "rtsp://admin:avcit12345678@192.168.19.100/ch34/main/av_stream", 拉流失败
        # "rtsp://admin:avcit12345678@192.168.19.100/ch35/main/av_stream",
        # "rtsp://admin:avcit12345678@192.168.19.100/ch36/main/av_stream"
        ]

url_12_floor = [
    "rtsp://admin:avcit12345678@192.168.19.100/ch43/main/av_stream",
    "rtsp://admin:avcit12345678@192.168.19.100/ch40/main/av_stream",
    "rtsp://admin:avcit12345678@192.168.19.100/ch46/main/av_stream",
    "rtsp://admin:avcit12345678@192.168.19.100/ch45/main/av_stream",
    "rtsp://admin:avcit12345678@192.168.19.100/ch44/main/av_stream",
    "rtsp://admin:avcit12345678@192.168.19.100/ch41/main/av_stream",
]

aomen = [
    "rtsp://192.168.31.121:8554/test",
]



# 调用enable接口
for index, url in enumerate(url_12_floor + url_9_floor):
    if index > 3:
        break
    counter_vehicle_alg = EnableChannelReq(channel_id=index, channel_name="穷哈哈", channel_url=url,)


    resp_enable = stub.EnableChannel(counter_vehicle_alg)
    print(url)
    # print(resp_enable)
    # time.sleep(1)

print("add task")
stub.AddTask(AddTaskReq(service_id=0, dev_id=0, root="test"))
# # stub.AddTask(AddTaskReq(service_id=1, dev_id=0, root="test"))


print("bind")
stub.BindChanTask(BindChanTaskReq(service_id=0, channel_id=0, channel_name="穷哈哈", config="test"))
stub.BindChanTask(BindChanTaskReq(service_id=0, channel_id=1, channel_name="穷哈哈", config="test"))
stub.BindChanTask(BindChanTaskReq(service_id=0, channel_id=2, channel_name="穷哈哈", config="test"))
stub.BindChanTask(BindChanTaskReq(service_id=0, channel_id=3, channel_name="穷哈哈", config="test"))
# stub.BindChanTask(BindChanTaskReq(service_id=0, channel_id=4, channel_name="穷哈哈", config="test"))
# stub.BindChanTask(BindChanTaskReq(service_id=0, channel_id=5, channel_name="穷哈哈", config="test"))
# stub.BindChanTask(BindChanTaskReq(service_id=0, channel_id=6, channel_name="穷哈哈", config="test"))
# stub.BindChanTask(BindChanTaskReq(service_id=0, channel_id=7, channel_name="穷哈哈", config="test"))

# stub.BindChanTask(BindChanTaskReq(service_id=1, channel_id=0, channel_name="穷哈哈", config="test"))
# stub.BindChanTask(BindChanTaskReq(service_id=1, channel_id=1, channel_name="穷哈哈", config="test"))
# stub.BindChanTask(BindChanTaskReq(service_id=1, channel_id=2, channel_name="穷哈哈", config="test"))
# stub.BindChanTask(BindChanTaskReq(service_id=1, channel_id=3, channel_name="穷哈哈", config="test"))

# print("unbind")
# stub.UnbindChanTask(UnbindChanTaskReq(service_id=0, channel_id=0))
# stub.UnbindChanTask(UnbindChanTaskReq(service_id=0, channel_id=1))
# stub.UnbindChanTask(UnbindChanTaskReq(service_id=0, channel_id=2))
# stub.UnbindChanTask(UnbindChanTaskReq(service_id=0, channel_id=3))

# print("remove task")
# stub.RemoveTask(RemoveTaskReq(service_id=0))
# stub.RemoveTask(RemoveTaskReq(service_id=1))

# print("disable channel")
# stub.DisableChannel(DisableChannelReq(channel_id=0))
# stub.DisableChannel(DisableChannelReq(channel_id=1))
# stub.DisableChannel(DisableChannelReq(channel_id=2))
# stub.DisableChannel(DisableChannelReq(channel_id=3))