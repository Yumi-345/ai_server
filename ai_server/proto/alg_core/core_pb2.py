# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/alg_core/core.proto
# Protobuf Python Version: 4.25.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
from proto.alg_core import common_pb2 as proto_dot_alg__core_dot_common__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x19proto/alg_core/core.proto\x12\x08\x61lg_core\x1a\x1bgoogle/protobuf/empty.proto\x1a\x1bproto/alg_core/common.proto\"S\n\nAddTaskReq\x12\x12\n\nservice_id\x18\x01 \x01(\x03\x12\x0e\n\x06\x64\x65v_id\x18\x02 \x01(\x05\x12\x0c\n\x04root\x18\x03 \x01(\t\x12\x13\n\x0bmodel_files\x18\x04 \x03(\t\"#\n\rRemoveTaskReq\x12\x12\n\nservice_id\x18\x01 \x01(\x03\"Q\n\x10\x45nableChannelReq\x12\x12\n\nchannel_id\x18\x01 \x01(\x03\x12\x14\n\x0c\x63hannel_name\x18\x02 \x01(\t\x12\x13\n\x0b\x63hannel_url\x18\x03 \x01(\t\"\'\n\x11\x44isableChannelReq\x12\x12\n\nchannel_id\x18\x01 \x01(\x03\"\xbd\x02\n\x11\x43hanServiceConfig\x12(\n\ttask_type\x18\x01 \x01(\x0e\x32\x15.alg_core.ServiceType\x12\x1b\n\x03roi\x18\x02 \x01(\x0b\x32\x0e.alg_core.Area\x12\x11\n\tbox_min_w\x18\x03 \x01(\x03\x12\x11\n\tbox_min_h\x18\x04 \x01(\x03\x12\x15\n\rdet_threshold\x18\x05 \x01(\x02\x12\x17\n\x0fmatch_threshold\x18\x06 \x01(\x02\x12\x16\n\x0e\x61larm_interval\x18\x07 \x01(\x03\x12\x14\n\x0cpgie_inteval\x18\x08 \x01(\x03\x12\x12\n\nwindow_len\x18\t \x01(\x03\x12\x12\n\nwindow_hit\x18\n \x01(\x03\x12\x1b\n\x13\x63rowd_num_threshold\x18\x0b \x01(\x03\x12\x18\n\x10\x66\x61\x63\x65qa_threshold\x18\x0c \x01(\x02\"_\n\x0f\x42indChanTaskReq\x12\x12\n\nservice_id\x18\x01 \x01(\x03\x12\x12\n\nchannel_id\x18\x02 \x01(\x03\x12\x14\n\x0c\x63hannel_name\x18\x03 \x01(\t\x12\x0e\n\x06\x63onfig\x18\x04 \x01(\t\";\n\x11UnbindChanTaskReq\x12\x12\n\nservice_id\x18\x01 \x01(\x03\x12\x12\n\nchannel_id\x18\x02 \x01(\x03\"n\n\x05\x45vent\x12\'\n\nevent_type\x18\x01 \x01(\x0e\x32\x13.alg_core.EventType\x12\x10\n\x08json_str\x18\x02 \x01(\t\x12\x0c\n\x04srcs\x18\x03 \x03(\x0c\x12\x0e\n\x06trains\x18\x04 \x03(\x0c\x12\x0c\n\x04rois\x18\x05 \x03(\x0c*\xee\x07\n\x0bServiceType\x12\x15\n\x11SERVICE_TYPE_NONE\x10\x00\x12%\n!SERVICE_TYPE_SMART_SITE_CAR_COVER\x10\x01\x12%\n!SERVICE_TYPE_SMART_SITE_CAR_WHEEL\x10\x02\x12-\n)SERVICE_TYPE_SMART_SITE_CAR_LINCESE_PLATE\x10\x03\x12)\n%SERVICE_TYPE_SMART_SITE_PERSON_HELMET\x10\x04\x12-\n)SERVICE_TYPE_SMART_SITE_PERSON_REFLECTIVE\x10\x05\x12*\n&SERVICE_TYPE_SAMRT_SITE_PERSON_SMOKING\x10\x06\x12)\n%SERVICE_TYPE_SMART_SITE_ENV_BARE_SOIL\x10\x07\x12%\n!SERVICE_TYPE_SMART_SITE_ENV_SMOKE\x10\x08\x12%\n!SERVICE_TYPE_SMART_SITE_ENV_FLAME\x10\t\x12(\n$SERVICE_TYPE_SMART_SITE_PERSON_NIGHT\x10\n\x12\x17\n\x13SERVICE_TYPE_OBJCNT\x10\x0b\x12\x17\n\x13SERVICE_TYPE_OBJCAP\x10\x0c\x12\x1e\n\x1aSERVICE_TYPE_PERSON_OBJCAP\x10\r\x12\x1b\n\x17SERVICE_TYPE_CAR_OBJCAP\x10\x0e\x12\x1e\n\x1aSERVICE_TYPE_NONCAR_OBJCAP\x10\x0f\x12 \n\x1cSERVICE_TYPE_FACERECOGNITION\x10\x10\x12\x1a\n\x16SERVICE_TYPE_ELECINTRU\x10\x11\x12\x1b\n\x17SERVICE_TYPE_WALKDOGDET\x10\x12\x12\x1e\n\x1aSERVICE_TYPE_PERSON_WANDER\x10\x13\x12\x1e\n\x1aSERVICE_TYPE_PARK_RIDE_DET\x10\x14\x12\x1e\n\x1aSERVICE_TYPE_NONCAR_OCCUPY\x10\x15\x12\x1c\n\x18SERVICE_TYPE_SHIP_OBJCAP\x10\x16\x12\x1a\n\x16SERVICE_TYPE_CROWD_DET\x10\x17\x12\x1d\n\x19SERVICE_TYPE_FALLDOWN_DET\x10\x18\x12!\n\x1dSERVICE_TYPE_RUBBISH_HEAP_DET\x10\x19\x12%\n!SERVICE_TYPE_RUBBISH_OVERFLOW_DET\x10\x1a\x12\x1a\n\x16SERVICE_TYPE_SMOKE_DET\x10\x1b\x12\x19\n\x15SERVICE_TYPE_FIRE_DET\x10\x1c*\xa5\x04\n\tEventType\x12\x14\n\x10\x45VENT_TYPE_ALARM\x10\x00\x12\x1b\n\x17\x45VENT_TYPE_OBJECT_COUNT\x10\x01\x12\x1d\n\x19\x45VENT_TYPE_OBJECT_CAPTURE\x10\x02\x12\x1c\n\x18\x45VENT_TYPE_OBJECT_DETECT\x10\x03\x12\x1c\n\x18\x45VENT_TYPE_ELEC_INTR_LAD\x10\x04\x12\x1b\n\x17\x45VENT_TYPE_WALK_DOG_DET\x10\x05\x12\x1f\n\x1b\x45VENT_TYPE_FACE_RECOGNITION\x10\x06\x12\x1b\n\x17\x45VENT_TYPE_SHIP_CAPTURE\x10\x07\x12 \n\x1c\x45VENT_TYPE_NONCAR_OCCUPY_CAP\x10\x08\x12\"\n\x1e\x45VENT_TYPE_RUBISH_HEAP_DET_CAP\x10\t\x12\'\n#EVENT_TYPE_RUBISH_OVER_FLOW_DET_CAP\x10\n\x12 \n\x1c\x45VENT_TYPE_FALL_DOWN_DET_CAP\x10\x0b\x12!\n\x1d\x45VENT_TYPE_SMOKE_FIRE_DET_CAP\x10\x0c\x12\x1c\n\x18\x45VENT_TYPE_CROWN_DET_CAP\x10\r\x12%\n!EVENT_TYPE_SIT_PERSON_SMOKING_CAP\x10\x0e\x12\x1c\n\x18\x45VENT_TYPE_PERSON_WANDER\x10\x0f\x12\x18\n\x14\x45VENT_TYPE_PARK_RIDE\x10\x10\x32\xd2\x04\n\x0b\x43oreService\x12\x37\n\x07\x41\x64\x64Task\x12\x14.alg_core.AddTaskReq\x1a\x16.google.protobuf.Empty\x12=\n\nRemoveTask\x12\x17.alg_core.RemoveTaskReq\x1a\x16.google.protobuf.Empty\x12\x43\n\rEnableChannel\x12\x1a.alg_core.EnableChannelReq\x1a\x16.google.protobuf.Empty\x12\x45\n\x0e\x44isableChannel\x12\x1b.alg_core.DisableChannelReq\x1a\x16.google.protobuf.Empty\x12\x41\n\x0c\x42indChanTask\x12\x19.alg_core.BindChanTaskReq\x1a\x16.google.protobuf.Empty\x12\x45\n\x0eUnbindChanTask\x12\x1b.alg_core.DisableChannelReq\x1a\x16.google.protobuf.Empty\x12?\n\x10PublishMediaInfo\x12\x16.google.protobuf.Empty\x1a\x13.alg_core.MediaInfo\x12;\n\x0cGetMediaInfo\x12\x16.google.protobuf.Empty\x1a\x13.alg_core.MediaInfo\x12\x37\n\x0cPublishEvent\x12\x16.google.protobuf.Empty\x1a\x0f.alg_core.EventB\x11Z\x0f./proto/algcoreb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'proto.alg_core.core_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  _globals['DESCRIPTOR']._options = None
  _globals['DESCRIPTOR']._serialized_options = b'Z\017./proto/algcore'
  _globals['_SERVICETYPE']._serialized_start=934
  _globals['_SERVICETYPE']._serialized_end=1940
  _globals['_EVENTTYPE']._serialized_start=1943
  _globals['_EVENTTYPE']._serialized_end=2492
  _globals['_ADDTASKREQ']._serialized_start=97
  _globals['_ADDTASKREQ']._serialized_end=180
  _globals['_REMOVETASKREQ']._serialized_start=182
  _globals['_REMOVETASKREQ']._serialized_end=217
  _globals['_ENABLECHANNELREQ']._serialized_start=219
  _globals['_ENABLECHANNELREQ']._serialized_end=300
  _globals['_DISABLECHANNELREQ']._serialized_start=302
  _globals['_DISABLECHANNELREQ']._serialized_end=341
  _globals['_CHANSERVICECONFIG']._serialized_start=344
  _globals['_CHANSERVICECONFIG']._serialized_end=661
  _globals['_BINDCHANTASKREQ']._serialized_start=663
  _globals['_BINDCHANTASKREQ']._serialized_end=758
  _globals['_UNBINDCHANTASKREQ']._serialized_start=760
  _globals['_UNBINDCHANTASKREQ']._serialized_end=819
  _globals['_EVENT']._serialized_start=821
  _globals['_EVENT']._serialized_end=931
  _globals['_CORESERVICE']._serialized_start=2495
  _globals['_CORESERVICE']._serialized_end=3089
# @@protoc_insertion_point(module_scope)
