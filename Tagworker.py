import cv2
import numpy as np
from Tagsolve import Solve_position

def timer(func):
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result=func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__}运行时间：{end - start}s")
        return result
    return wrapper
class Worker():
    def __init__(self,cap,model,using_csi,using_v4l2,imgHeight,imgWidth) -> None:
        self.cap=cap
        self.model=model
        self.img=None
        self.detections_yolo=None
        self.using_csi=using_csi
        self.using_v4l2=using_v4l2
        self.imgWidth=imgWidth
        self.imgHeight=imgHeight
    
    @timer
    def camera_cap(self):
        "摄像头捕捉图像"
        if self.using_csi and self.using_v4l2:
            _,frame=self.cap.__next__()
            data=frame.data
            npdata=np.frombuffer(data,dtype=np.uint16)
            bayerIMG = npdata.reshape(self.imgHeight, self.imgWidth, 1)/64
            grayIMG = np.zeros([self.imgHeight, self.imgWidth, 3], dtype='uint16')
            grayIMG[:,:,0]=grayIMG[:,:,1]=grayIMG[:,:,2]=bayerIMG[:,:,0]
            img=np.uint8(grayIMG)
        else:
            _, img = self.cap.read()
        return img
    @timer
    def yolo_detect(self):
        "yolo模型计算,返回运算结果"
        results = self.model(self.img)
        return results.xyxy[0].to('cpu').numpy()
    def draw_rect(self):
        "绘制yolo检测框,返回框坐标"
        detect_list = []
        for detection in self.detections_yolo:
            xmin, ymin, xmax, ymax, conf, classItem = detection[:6]
            if int(classItem) == 0 and conf > 0.3:
                detect_list.append([int(xmin), int(ymin), int(xmax), int(ymax), conf])

        if len(detect_list) > 0:
            FirstItem = detect_list[0]
            cv2.rectangle(self.img, (FirstItem[0], FirstItem[1]), (FirstItem[2], FirstItem[3]), (50, 255, 50), 3,
                        lineType=cv2.LINE_AA)
            rect = np.array([[FirstItem[0], FirstItem[1]], [FirstItem[0], FirstItem[3]],
                                [FirstItem[2], FirstItem[1]], [FirstItem[2], FirstItem[3]]], dtype=np.int32)
    
            return rect
        else:
            return None
    @timer
    def workonce(self):
        "进行一轮工作"
        self.img=self.camera_cap()
        self.detections_yolo=self.yolo_detect()
        if self.detections_yolo is not None: 
            # 过滤模型，获取classItem=0的识别结果
            rect = self.draw_rect()
            if rect is not None:
                print("detection")
                print(rect)
                solver=Solve_position(rect)
                return solver.get_location(),self.img                
        return None,self.img
        