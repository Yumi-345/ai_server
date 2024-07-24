# -*- coding: utf-8 -*-
"""
Author: AlfredNG
Date: 2024-07-15 09:35:35
Description: 
Copyright: Copyright (c) 2024-Present AVCIT
"""

import grpc
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
import proto.alg_core.common_pb2 as common_pb2
import time
import json

# import proto_bak.task_pb2 as task_pb2

# task_notify = task_pb2.TaskNotify()


# task_notify.collect_notify.channel.id = Plumber.GetRunningState_list[jpg_src_id][0]
# task_notify.collect_notify.channel.name = Plumber.GetRunningState_list[jpg_src_id][1]
# task_notify.collect_notify.channel.url = Plumber.GetRunningState_list[jpg_src_id][2]

# task_notify.collect_notify.picture.id = k_obj
# task_notify.collect_notify.picture.srcData = src_img
# task_notify.collect_notify.picture.roiData = roi_img
# task_notify.collect_notify.picture.type = task_pb2.PICTURE_TYPE_PERSON


def create_channel():
    """可以使用总线或者网络"""
    return grpc.insecure_channel("localhost:50051")
    """总线如下"""
    # return grpc.insecure_channel("unix:///tmp/aicore-service.sock")


# 创建客户端
stub = CoreServiceStub(create_channel())

# 调用接口
# stub.EnableChannel(
#     EnableChannelReq(
#         channel_id=1,
#         channel_name="天河区",
#         channel_url="rtsp://192.168.31.222/1080",
#     )
# )

# stub.DisableChannel(DisableChannelReq(channel_id=1))
# event = Event(
#     # event_type=EventType.EVENT_TYPE_NONCAR_OCCUPY_CAP,
#     # json_str={"te":"waet"},
#     # srcs=[b'src1', b'src2'],
#     # trains=[b'train1', b'train2'],
#     # rois=[b'roi1', b'roi2']
# )
# event.json_str = json.dumps({"test":[123]})
# stub.PublishEvent(event)


# stub.UnbindChanTask(UnbindChanTaskReq(service_id=0, channel_id=0))
stub.UnbindChanTask(UnbindChanTaskReq(service_id=1,channel_id=1))
# stub.RemoveTask(RemoveTaskReq(service_id=0))
# stub.DisableChannel(DisableChannelReq(channel_id=0))