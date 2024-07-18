"""
Author: AlfredNG
Date: 2024-07-15 09:35:35
Description: 
Copyright: Copyright (c) 2024-Present AVCIT
"""

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
from proto.alg_core.common_pb2 import MediaInfo
from proto.alg_core.core_pb2_grpc import (
    CoreServiceServicer,
    add_CoreServiceServicer_to_server,
)
from google.protobuf import empty_pb2


class CoreServiceImpl(CoreServiceServicer):

    def AddTask(self, request: AddTaskReq, context):
        return

    def RemoveTask(self, request: RemoveTaskReq, context):
        return

    def EnableChannel(self, request: EnableChannelReq, context):
        try:
            print(
                f"Received EnableChannel request with channel_id: {request.channel_id}"
            )
            return empty_pb2.Empty()
        except Exception as e:
            print(f"Error processing EnableChannel: {e}")

    def DisableChannel(self, request: DisableChannelReq, context):
        try:
            print(
                f"Received DisableChannel request with channel_id: {request.channel_id}"
            )
            return empty_pb2.Empty()
        except Exception as e:
            print(f"Error processing DisableChannel: {e}")

    def BindChanTask(self, request: BindChanTaskReq, context):
        return

    def UnbindChanTask(self, request: UnbindChanTaskReq, context):
        return

    def GetMediaInfo(self, request: MediaInfo, context):
        return


def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10)
    )  # 创建gRPC服务器实例
    add_CoreServiceServicer_to_server(
        CoreServiceImpl(), server
    )  # 将服务实现添加到服务器上

    # 指定监听地址和服务端口
    server.add_insecure_port("[::]:50051")

    server.start()  # 启动服务器
    print("Server started on port 50051...")
    server.wait_for_termination()  # 阻塞等待，直到服务器关闭


if __name__ == "__main__":
    serve()
