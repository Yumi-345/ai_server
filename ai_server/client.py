"""
Author: AlfredNG
Date: 2024-07-15 09:35:35
Description: 
Copyright: Copyright (c) 2024-Present AVCIT
"""

import grpc
from proto.alg_core.core_pb2 import EnableChannelReq, DisableChannelReq
from proto.alg_core.core_pb2_grpc import CoreServiceStub
from google.protobuf import empty_pb2


def create_channel():
    """可以使用总线或者网络"""
    return grpc.insecure_channel("localhost:50051")
    """总线如下"""
    # return grpc.insecure_channel("unix:///tmp/aicore-service.sock")


# 创建客户端
stub = CoreServiceStub(create_channel())

# 调用接口
stub.EnableChannel(
    EnableChannelReq(
        channel_id=1,
        channel_name="天河区",
        channel_url="rtsp://192.168.31.222/1080",
    )
)

stub.DisableChannel(DisableChannelReq(channel_id=1))
