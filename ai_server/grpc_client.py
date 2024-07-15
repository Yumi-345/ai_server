import grpc
import time
from concurrent import futures
import proto.task_pb2 as task_pb2
import proto.global_pb2 as global_pb2
from proto.task_pb2_grpc import (
    TaskServiceStub,
    add_TaskServiceServicer_to_server,
)
from google.protobuf import empty_pb2

MAX_MESSAGE_LENGTH = 256*1024*1024  # 可根据具体需求设置，此处设为256M

def create_channel():
    return grpc.insecure_channel(target="127.0.0.1:50151", 
                                #  options=[  ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                #             ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),]
                                )

# 创建客户端
stub = TaskServiceStub(create_channel())

names_9_floor = [
        "销售部",
        "技术部",
        "茶水区",
        "售前部",
        "右展厅",
    ]

url_9_floor = [
        "rtsp://admin:avcit12345678@192.168.19.100/ch39/main/av_stream",
        "rtsp://admin:avcit12345678@192.168.19.100/ch33/main/av_stream",
        # "rtsp://admin:avcit12345678@192.168.19.100/ch34/main/av_stream", 拉流失败
        # "rtsp://admin:avcit12345678@192.168.19.100/ch35/main/av_stream",
        # "rtsp://admin:avcit12345678@192.168.19.100/ch36/main/av_stream"
        ]

names_12_floor = [
    "邓国威工位上摄像头",
    "甄志明工位上摄像头",
    "严峻恒工位上摄像头",
    "蔡伟健工位上摄像头",
    "12楼门口",
    "楼上41",
]

url_12_floor = [
    "rtsp://admin:avcit12345678@192.168.19.100/ch43/main/av_stream",
    "rtsp://admin:avcit12345678@192.168.19.100/ch40/main/av_stream",
    "rtsp://admin:avcit12345678@192.168.19.100/ch46/main/av_stream",
    "rtsp://admin:avcit12345678@192.168.19.100/ch45/main/av_stream",
    "rtsp://admin:avcit12345678@192.168.19.100/ch44/main/av_stream",
    "rtsp://admin:avcit12345678@192.168.19.100/ch41/main/av_stream",
]



# 调用enable接口
# for i in range(len([1])):
for index, url in enumerate(url_12_floor+url_9_floor):
    if index > 7:
        break
    # counter_vehicle_alg = task_pb2.CollectAlg(channel = task_pb2.Channel(id=i, name="笑哈哈"+str(i), url="rtsp://admin:avcit12345678@192.168.1.8" + str(i) + "/h265/ch1/main/av_stream"))
    # counter_vehicle_alg = task_pb2.CollectAlg(channel = task_pb2.Channel(id=i, name="笑哈哈"+str(i), url="rtsp://admin:avcit12345678@192.168.19.100/ch40/main/av_stream"))
    # counter_vehicle_alg = task_pb2.CollectAlg(channel = task_pb2.Channel(id=i, name="笑哈哈"+str(i), url="rtsp://admin:avcit12345678@192.168.19.100/ch39/main/av_stream"))
    # counter_vehicle_alg = task_pb2.CollectAlg(channel = task_pb2.Channel(id=7, name="笑哈哈"+str(7), url="rtsp://admin:avcit12345678@192.168.19.100/ch33/main/av_stream"))
    # counter_vehicle_alg = task_pb2.CollectAlg(channel = task_pb2.Channel(id=i, name="笑哈哈"+str(i), url="rtsp://192.168.31.191:551/2160"))

    # 9楼
    # counter_vehicle_alg = task_pb2.CollectAlg(channel = task_pb2.Channel(id=i, name=names_9_floor[i], url=url_9_floor[i]))

    # counter_vehicle_alg.lines.append(global_pb2.Line(id=0, name="笑哈哈", x1=225*100//1920, y1=359*100//1080, x2=930*100//1920, y2=414*100//1080))
    # counter_vehicle_alg.lines.append(global_pb2.Line(id=1, name="笑哈哈", x1=976*100//1920, y1=518*100//1080, x2=1600*100//1920, y2=280*100//1080))

    # area = global_pb2.Area(id=0, name="笑哈哈")
    # area.points.append(global_pb2.Point(x=10, y=20))
    # counter_vehicle_alg.areas.append(area)

    counter_vehicle_alg = task_pb2.CollectAlg(channel = task_pb2.Channel(id=index, name="笑哈哈"+str(index), url=url))


    resp_enable = stub.Enable(task_pb2.AlgConfig(collect_alg=counter_vehicle_alg))
    time.sleep(1)
    print(url)
    print(resp_enable)


# resp_enable = stub.Disable(

#     task_pb2.AlgConfig(
#         collect_alg=task_pb2.CollectAlg(
#             channel=task_pb2.Channel(
#                 id=27, name="笑哈哈", url="rtsp://admin:avcit12345678@192.168.19.100/ch39/main/av_stream"
#             ),
#         )
#     )
# )
# print(resp_enable)

# 调用Notify接口，这个是流式接口
# resp = stub.Notify(empty_pb2.Empty())
# while True:
#     resp.next()
#     print(1)

# resp = stub.GetRunningState(empty_pb2.Empty())
# print(resp)

# reval = stub.Reset(empty_pb2.Empty())
# print(reval)