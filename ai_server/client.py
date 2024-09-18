"""
Author: AlfredNG
Date: 2024-07-15 09:35:35
Description: 
Copyright: Copyright (c) 2024-Present AVCIT
"""

import grpc
from proto.alg_core.core_pb2 import EnableChannelReq, DisableChannelReq,UnbindChanTaskReq
from proto.alg_core.core_pb2_grpc import CoreServiceStub
from google.protobuf import empty_pb2


def create_channel():
    """可以使用总线或者网络"""
    return grpc.insecure_channel("localhost:50051")
    """总线如下"""
    # return grpc.insecure_channel("unix:///tmp/alg-service.sock")


# 创建客户端
stub = CoreServiceStub(create_channel())
try:
    stub.UnbindChanTask(UnbindChanTaskReq(service_id=1,channel_id=1))
except Exception as e:
    print(f"Error processing UnbindChanTask: {e}")
# 调用接口
# try:
#     stub.EnableChannel(
#         EnableChannelReq(
#             channel_id=1,
#             channel_name="天河区",
#             channel_url="rtsp://192.168.31.222/1080",
#         )
#     )
# except Exception as e:
#     print(f"Error processing EnableChannel: {e}")
# stub.DisableChannel(DisableChannelReq(channel_id=1))
