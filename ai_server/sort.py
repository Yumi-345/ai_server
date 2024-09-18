Author = "xuwj"

import cv2 
import copy

import grpc
from proto.alg_core.common_pb2 import Box
from proto.alg_core.core_pb2 import Event, EventType, TaskTypeEnum
from proto.alg_core.picture_pb2 import PictureType
from proto.alg_core.core_pb2_grpc import CoreServiceStub

import time


MOVE_PIX = 80


def create_channel():
    return grpc.insecure_channel("172.20.253.34:50053")


class BoxTracker():
    def __init__(self, img_arr, u_id, conf, left, top, width, height, center_x, center_y, class_id, src_id, frame_number, config, child_obj):
        self.n2d = False
        self.img_arr = [img_arr]
        self.conf = [conf]
        self.top = [top]
        self.left = [left]
        self.width = [width]
        self.height = [height]
        self.area = [width * height]
        # self.center_x = center_x
        # self.center_y = center_y
        self.trace = [[center_x, center_y]]

        self.u_id = u_id
        self.class_id = class_id
        self.src_id = src_id

        self.frame_number = frame_number
        self.config = config
        self.child_obj = [child_obj]

        self.stub = CoreServiceStub(create_channel())

        self.last_time = time.time()


    def update(self, img_arr, u_id, conf, left, top, width, height, center_x, center_y, class_id, src_id, frame_number, child_obj):
        # print(f"======================{class_id}===========================")
        # if width > self.config.width & height > self.config.height & conf > self.config.conf:
        if 1:
        
            self.frame_number = frame_number
            self.trace.append([center_x, center_y])

            area = width * height

            if len(self.conf) < 5:
                self.img_arr.append(img_arr)
                self.conf.append(conf)
                self.top.append(top)
                self.left.append(left)
                self.width.append(width)
                self.height.append(height)
                self.area.append(area)
                self.child_obj.append(child_obj)

            else:
                min_conf = min(self.conf)
                if conf > min_conf:
                    index = self.conf.index(min_conf)
                    
                    self.img_arr[index] = img_arr
                    self.conf[index] = conf
                    self.top[index] = top
                    self.left[index] = left
                    self.width[index] = width
                    self.height[index] = height
                    self.area[index] = area
                    self.child_obj[index] = child_obj

    def send(self,save=False):
        if len(self.trace) < 12:
            return 0, "暂不发送，轨迹小于12"
        
        min_x = 1920
        min_y = 1080
        max_x = 0
        max_y = 0
        for point in self.trace:
            if point[0] < min_x:
                min_x = point[0]
            if point[1] < min_y:
                min_y = point[1]
            if point[0] > max_x:
                max_x = point[0]
            if point[1] > max_y:
                max_y = point[1]
        if max_x - min_x < MOVE_PIX and max_y - min_y < MOVE_PIX:
            # print("距离短于10", k_obj)
            return 0, f"暂不发送，距离短于{MOVE_PIX}"

        index = self.area.index(max(self.area))

        img_train = self.img_arr[index]
        img_arr = copy.deepcopy(self.img_arr[index] )
        # conf = self.conf[index] 
        top = self.top[index] 
        left = self.left[index] 
        width = self.width[index] 
        height = self.height[index] 
        # area = self.area[index] 

        child_obj = self.child_obj[index]

        if top == 0 and left == 0 and width == 0 and height == 0:
            return 0, "暂不发送，无效的框"
        
        try:
            roi_data = copy.deepcopy(img_arr[max(0,top-5): min(1080,top+height+5), max(0,left-5): min(1920,left+width+5)])
            if child_obj is not None:
                roi_data1 = copy.deepcopy(img_arr[max(0,child_obj[3]-5): min(1080,child_obj[3]+child_obj[5]+5), max(0,child_obj[2]-5): min(1920,child_obj[2]+child_obj[4]+5)])

        except:
            print()
        
        # 画框、画轨迹、写标题
        try:
            cv2.rectangle(img_arr, (left, top), (left+width, top+height), (0, 255, 0), 3)
            if child_obj is not None:
                cv2.rectangle(img_arr, (child_obj[2], child_obj[3]), (child_obj[2]+child_obj[4], child_obj[3]+child_obj[5]), (0, 255, 0), 3)
            for i in range(len(self.trace)-1):
                    cv2.line(img_arr, (int(self.trace[i][0]), int(self.trace[i][1])),
                                (int(self.trace[i+1][0]), int(self.trace[i+1][1])), (0, 255, 0), 3)
            cv2.putText(img_arr, "{}".format(self.u_id), (left+10, top-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        except:
            return 0, "draw error"

        if save:
            # 保存图片
            # cv2.imwrite(f"/root/apps/ai_server/output/stream{self.src_id}_frame{self.frame_number}_task{self.u_id}_roi.jpg", roi_data)
            cv2.imwrite(f"/root/apps/ai_server/output/stream{self.src_id}_frame{self.frame_number}_task{self.u_id}_src.jpg", img_arr)
            # cv2.imwrite(f"/root/apps/ai_server/output2/stream{self.src_id}_frame{self.frame_number}_task{self.u_id}_src.jpg", img_arr)
        
        try:
            _, train_img_encode = cv2.imencode(".jpg", img_train)
            _, src_img_encode = cv2.imencode(".jpg", img_arr)
            _, roi_img_encode = cv2.imencode(".jpg", roi_data)

            train_img_bytes = train_img_encode.tobytes()
            src_img_bytes = src_img_encode.tobytes()
            roi_img_bytes = roi_img_encode.tobytes()

            if child_obj is not None:
                _, roi_img_encode1 = cv2.imencode(".jpg", roi_data1)
                roi_img_bytes1 = roi_img_encode1.tobytes()
            # msg = {
            #     "service_id" : int(self.u_id.split("_")[0]),
            #     "channel_id" : self.src_id,
            #     "object_id" : int(self.u_id.split("_")[2])
            # }
            event = Event(
                event_type=EventType.ValueType(1), 
                channel_id=self.src_id,
                srcs=[src_img_bytes],
                trains=[train_img_bytes],
                rois=[roi_img_bytes], 
                box=Box(x1=int(left), y1=int(top), x2=int(left+width), y2=int(top+height)), 
                property="None", 
                task_id=TaskTypeEnum.ValueType(5), # 暂定
                pic_type=PictureType.ValueType(0)  # 暂定
            )
            # self.stub.PublishEvent(event)
            return 1, "发送成功"
        except Exception as e:
            print(f"error: {e}")
            # print(f"error: 服务端链接失败，未发送")
            return 1, "send failed"

    # def __del__(self):
    #     print("=============================================")