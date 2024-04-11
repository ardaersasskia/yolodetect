import cv2,os
import numpy as np
from .Tagsolve import Solve_position

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
    def __init__(self,model,using_csi=False,using_v4l2=True,imgHeight=720,imgWidth=1280,cap=None,testflag=False,testdirectory=None) -> None:
        self.cap=cap
        self.model=model
        self.testflag=testflag
        self.testdirectory=testdirectory
        self.testpositions=[]
        self.testposition=None
        self.img=None
        self.images=[]
        self.img_with_rect=None
        self.detections_yolo=None
        self.using_csi=using_csi
        self.using_v4l2=using_v4l2
        self.imgWidth=imgWidth
        self.imgHeight=imgHeight
        if self.testflag:
            self.file_read(testdirectory)
    def file_read(self,directory):
        "从文件夹中读取图片"
        self.images=[]
        for filename in os.listdir(directory):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(directory, filename)
                img = cv2.imread(img_path)
                self.images.append(img)
                base_name=os.path.splitext(filename)[0]
                # 去除usb_
                x_y_=base_name.split('_')[1]
                # 取得x,y
                x_=float(x_y_.split('y')[0].split('x')[1])
                y_=float(x_y_.split('y')[1])
                self.testpositions.append({'x':x_,'y':y_})
                print(f"{filename} loaded")
        return None
    def file_pop(self):
        "从内存中读取图片"
        img = self.images.pop()
        self.imgWidth=img.shape[1]
        self.imgHeight=img.shape[0]
        testposition=self.testpositions.pop()
        return img,testposition
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
            if conf > 0.3:
                detect_list.append([int(xmin), int(ymin), int(xmax), int(ymax), conf,classItem])

        if len(detect_list) > 0:
            FirstItem = detect_list[0]
            cv2.rectangle(self.img_with_rect, (FirstItem[0], FirstItem[1]), (FirstItem[2], FirstItem[3]), (50, 255, 50), 3,
                        lineType=cv2.LINE_AA)
            rect = np.array([[FirstItem[0], FirstItem[1]], [FirstItem[2], FirstItem[1]],
                                [FirstItem[2], FirstItem[3]], [FirstItem[0], FirstItem[3]]], dtype=np.int32)
    
            return rect,FirstItem[5]
        else:
            return None
    def find_contour(self,rect,isoutside):
        "opencv寻找轮廓"
        outer=1.05
        xmin=rect[0][0]
        ymin=rect[0][1]
        xmax=rect[2][0]
        ymax=rect[2][1]
        width=xmax-xmin
        height=ymax-ymin
        center=(int((xmin+xmax)/2),int((ymin+ymax)/2))
        xmin=center[0]-int(width*outer/2)
        ymin=center[1]-int(height*outer/2)
        xmax=center[0]+int(width*outer/2)
        ymax=center[1]+int(height*outer/2)
        
        img_mini=self.img[ymin:ymax,xmin:xmax]
        cv2.imshow('mini',img_mini)
        # # 进行高斯模糊
        # blured = cv2.GaussianBlur(img_mini, (3, 3), 0)
        gray = cv2.cvtColor(img_mini, cv2.COLOR_BGR2GRAY)
        # 二值化
        _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        cv2.imshow('binary',binary)
        # 查找轮廓
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        final_counters=[]
        for i,contour in enumerate(contours):
            # hierarchy[0][i][y]
            # i为这组轮廓点的编号
            # y=0表示同一层次下一个轮廓编号
            # y=1表示同一层次上一个轮廓编号
            # y=2表示它的第一个子轮廓编号
            # y=3表示它的父轮廓编号
            if len(contour)>100 and hierarchy[0][i][3]==-1:
                final_counters.append(contour)
            if len(final_counters)>=2:
                break
        if len(final_counters)==2:
            final_counter=final_counters[1]
        else:
            final_counter=final_counters[0]
        if final_counter is not None:
            rect = cv2.minAreaRect(final_counter)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            for point in box:
                point[0] += xmin
                point[1] += ymin
            #print(box)
            #cv2.drawContours(img_mini,[box],0,(0,0,255),2)
            # final_counter的坐标是相对于img_mini的
            # 需要转换为相对于img的坐标
            # final_counter的xim,ymin,xmax,ymax四个点对应图像四角
            #print(f"contours:{len(final_counters)}")
            #cv2.drawContours(img_mini, final_counter, -1, (0, 0, 255), 2)
            #cv2.imshow('contours',img_mini)
            cv2.drawContours(self.img_with_rect, [box], 0, (0, 0, 255), 2)
            #cv2.imshow('img_with_rect',self.img_with_rect)
            return True,box
        else:
            return False,rect
    @timer
    def workonce(self):
        "进行一轮工作"
        if self.testflag:
            self.img,self.testposition=self.file_pop()
            self.img_with_rect=self.img.copy()
            print(f"testposition:{self.testposition}")
        else:
            self.img=self.img_with_rect=self.camera_cap()
        self.detections_yolo=self.yolo_detect()
        if self.detections_yolo is not None: 
            # 过滤模型，获取classItem=0的识别结果
            rect,classItem = self.draw_rect()
            if rect is not None:
                #print("detection")
                #print(rect)
                if classItem==0:
                    isoutside=True
                else:
                    isoutside=False
                
                contour_ret,rect=self.find_contour(rect,isoutside)
                solver=Solve_position(rect,isoutside)
                return solver.get_location(contour_ret),cv2.flip(self.img_with_rect,-1)                
        return None,self.img_with_rect
        