Author = "xuwj"

import cv2 
import copy

MOVE_PIX = 80

class BoxTracker():
    def __init__(self, img_arr, object_id, conf, top, left, width, height, center_x, center_y, class_id, src_id, frame_number):
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

        self.object_id = object_id
        self.class_id = class_id
        self.src_id = src_id

        self.frame_number = frame_number

    def update(self, img_arr, object_id, conf, top, left, width, height, center_x, center_y, class_id, src_id, frame_number):
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

    def send(self,save=False):
        if len(self.trace) < 4:
            return 0, "暂不发送，轨迹小于3"
        
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

        img_arr = self.img_arr[index] 
        # conf = self.conf[index] 
        top = self.top[index] 
        left = self.left[index] 
        width = self.width[index] 
        height = self.height[index] 
        # area = self.area[index] 

        if top == 0 and left == 0 and width == 0 and height == 0:
            return 0, "暂不发送，无效的框"
        
        try:
            roi_data = copy.deepcopy(img_arr[max(0,top-5): min(1080,top+height+5), max(0,left-5): min(1920,left+width+5)])
        except:
            print()
        
        # 画框、画轨迹、写标题
        try:
            cv2.rectangle(img_arr, (left, top), (left+width, top+height), (0, 255, 0), 3)
            for i in range(len(self.trace)-1):
                    cv2.line(img_arr, (int(self.trace[i][0]), int(self.trace[i][1])),
                                (int(self.trace[i+1][0]), int(self.trace[i+1][1])), (0, 255, 0), 3)
            cv2.putText(img_arr, "{}".format(self.object_id), (left+10, top-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        except:
            # print("draw error")
            pass

        if save:
            # 保存图片
            cv2.imwrite(f"/root/apps/ai_server/output/stream{self.src_id}_frame{self.frame_number}_{self.object_id}_roi.jpg", roi_data)
            cv2.imwrite(f"/root/apps/ai_server/output/stream{self.src_id}_frame{self.frame_number}_{self.object_id}_src.jpg", img_arr)
        return 1, "发送成功"
