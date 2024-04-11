from threading import Thread,Event
import queue
import cv2
import numpy as np
import torch
import os,time

# ... (其余import和functions，保持不变)
def circshift(matrix, shiftnum1, shiftnum2):
    h, w = matrix.shape
    matrix = np.vstack((matrix[(h - shiftnum1):, :], matrix[:(h - shiftnum1), :]))
    matrix = np.hstack((matrix[:, (w - shiftnum2):], matrix[:, :(w - shiftnum2)]))
    return matrix

class Detect_Tag:
    def __init__(self) -> None:
        pass
    def boxarea(self, box):
        x1 = box[0][0]
        x2 = box[3][0]
        y1 = box[0][1]
        y2 = box[3][1]
        return (x2 - x1) * (y2 - y1)
    def inside(self, rectt, x, y):
        """判断点x,y是否在矩形rectt中心3/4区域内"""
        x1 = rectt[0][0]
        x2 = rectt[3][0]
        y1 = rectt[0][1]
        y2 = rectt[3][1]

        p = 0.75
        if x1 * p + x2 * (1 - p) < x < x1 * (1 - p) + x2 * p and y1 * p + y2 * (1 - p) < y < y1 * (1 - p) + y2 * p:
            return True
        return False

    def gettvec(self, box, h, cameraMatrix, distCoeffs):
        """进行位姿解算"""
        objPoints = np.array([[h, h, 0],
                                [-h, h, 0],
                                [-h, -h, 0],
                                [h, -h, 0]], dtype=np.double)

        square = []
        for i in range(0, 4):
            square.append(box[i][0] ** 2 + box[i][1] ** 2)
        arr_aa = np.array(square)
        minindex = np.argmin(arr_aa)

        # 顶点坐标旋转（右下 左下 左上 右上）
        box_2 = circshift(box, 2 - minindex, 0)
        #print('外接矩形box:', box_2)
        box_2 = np.array(box_2, dtype=np.double)

        # retval_1,rvec_1,tvec_1,_ = cv2.solvePnPRansac(objPoints, box_2, cameraMatrix, distCoeffs, False, cv2.SOLVEPNP_P3P)
        retval_1, rvec_1, tvec_1 = cv2.solvePnP(objPoints, box_2, cameraMatrix, distCoeffs, False, cv2.SOLVEPNP_P3P)

        return tvec_1[2] * 0.01, -tvec_1[0] * 0.01, -tvec_1[1] * 0.01, True

    def getRvec(self, box, h, cameraMatrix, distCoeffs):
        """进行位姿解算与旋转"""
        objPoints = np.array([[h, h, 0],
                                [-h, h, 0],
                                [-h, -h, 0],
                                [h, -h, 0]], dtype=np.double)

        square = []
        for i in range(0, 4):
            square.append(box[i][0] ** 2 + box[i][1] ** 2)
        arr_aa = np.array(square)
        minindex = np.argmin(arr_aa)

        # 顶点坐标旋转（右下 左下 左上 右上）
        box_2 = circshift(box, 2 - minindex, 0)
        #print('外接矩形box:', box_2)
        box_2 = np.array(box_2, dtype=np.double)

        # retval_1,rvec_1,tvec_1,_ = cv2.solvePnPRansac(objPoints, box_2, cameraMatrix, distCoeffs, False, cv2.SOLVEPNP_P3P)
        retval_1, rvec_1, tvec_1 = cv2.solvePnP(objPoints, box_2, cameraMatrix, distCoeffs, False, cv2.SOLVEPNP_P3P)

        R, _ = cv2.Rodrigues(rvec_1)
        euler_angles = cv2.RQDecomp3x3(R)[0]
        yaw = euler_angles[0]
        pitch = euler_angles[1]
        roll = euler_angles[2]
        return yaw, pitch, roll, True

    # 处理图片
    def get_location(self, contours, hierarchy, yolobox, cameraMatrix, distCoeffs, retVal=0):
        final_contour = []

        for ic in range(len(contours)):
            contour = contours[ic]
            rect_area = cv2.contourArea(contour)
            approx = cv2.approxPolyDP(contour, .03 * cv2.arcLength(contour, True), True)
            if rect_area > self.boxarea(yolobox) * 0.2 and len(approx) == 4:
                M = cv2.moments(contour)
                x = int(M['m10'] / M['m00'])
                y = int(M['m01'] / M['m00'])
                if self.inside(yolobox, x, y):
                    if hierarchy[0][ic][2] == -1:
                        final_contour = contour

        if len(final_contour) < 1:
            # print("no contour detected!")
            rect_r = cv2.minAreaRect(yolobox)
            yolobox = cv2.boxPoints(rect_r)
            return self.gettvec(yolobox, 3, cameraMatrix, distCoeffs)
        # print("contour detected!")
        rect = cv2.minAreaRect(final_contour)
        self.box_1 = cv2.boxPoints(rect)

        if retVal == 0:
            return self.gettvec(self.box_1, 2.3, cameraMatrix, distCoeffs)
        elif retVal == 1:
            return self.getRvec(self.box_1, 2.3, cameraMatrix, distCoeffs)
class FileRead(Thread):
    def __init__(self, path,yolo_img_queue,opencv_img_queue,event):
        Thread.__init__(self)
        self.path = path
        self.yolo_img_queue = yolo_img_queue
        self.opencv_img_queue = opencv_img_queue    
        self.event = event

    def run(self):
        files = os.listdir(self.path)
        for file in files:
            img_path = os.path.join(path, file)
            img = cv2.imread(img_path)       
            self.yolo_img_queue.put(img)
            self.opencv_img_queue.put(img)
            start_event.set()
            print('file read process')
            
        self.yolo_img_queue.put(None)  # 发送None作为YOLO线程的停止信号
        self.opencv_img_queue.put(None)  # 发送None作为OpenCV线程的停止信号
class CameraRead(Thread):
    def __init__(self,yolo_img_queue,opencv_img_queue,event,runflag,test_path="dataset/TAG/display"):
        Thread.__init__(self)
        self.yolo_img_queue = yolo_img_queue
        self.opencv_img_queue = opencv_img_queue    
        self.event = event
        self.test_path = test_path
        self.runflag = runflag

    def run(self):
        cap=cv2.VideoCapture(index=1,apiPreference=cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,600)
        cap.set(cv2.CAP_PROP_FPS,60)
        ret,frame=cap.read()
        while runflag:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            t1=time.time()
            ret,frame=cap.read()  
            if self.yolo_img_queue.not_full and self.opencv_img_queue.not_full:
                self.yolo_img_queue.put(frame)
                self.opencv_img_queue.put(frame)
                t2=time.time()
                print('camera read time:',t2-t1)
            else:
                continue
            start_event.set()
            #print('camera read process')

        self.yolo_img_queue.put(None)  # 发送None作为YOLO线程的停止信号
        self.opencv_img_queue.put(None)  # 发送None作为OpenCV线程的停止信号
class YOLODetectThread(Thread):
    def __init__(self, img_queue, result_queue, model,runevent,yolo_ready_event,opencv_ready_event,runflag):
        Thread.__init__(self)
        self.img_queue = img_queue
        self.result_queue = result_queue
        self.model = model
        self.runevent=runevent
        self.yolo_ready_event=yolo_ready_event
        self.opencv_ready_event=opencv_ready_event
        self.runflag=runflag
        

    def run(self):
        while runflag:
            self.runevent.wait()
           #  print('yolo start')
            self.yolo_ready_event.clear()
            
            img = self.img_queue.get()
            t1=time.time()
            if img is None:
                self.result_queue.put(None)
                break
            
            results = self.model(img[:,:,::-1])
            
            # print(results.pandas().xyxy[0]['class'])
            self.result_queue.put((img,results.xyxy[0].to('cpu').numpy()))
            self.yolo_ready_event.set()
            t2=time.time()
            print('yolo time:',t2-t1)
            if self.opencv_ready_event.is_set():
                self.runevent.set()
            else:
                self.runevent.clear()
            # print('yolo process')

class OpenCVProcessThread(Thread):
    def __init__(self, img_queue, result_queue, cameraMatrix, distCoeffs,runevent,yolo_ready_event,opencv_ready_event,runflag):
        Thread.__init__(self)
        self.img_queue = img_queue
        self.result_queue = result_queue
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs
        self.runevent=runevent
        self.yolo_ready_event=yolo_ready_event
        self.opencv_ready_event=opencv_ready_event
        self.runflag=runflag
    def run(self):
        while runflag:
            self.runevent.wait()
            # print('opencv start')
            self.opencv_ready_event.clear()
            img = self.img_queue.get()
            t1=time.time()
            if img is None:
                self.result_queue.put(None)
                break
             # 转为灰度图像
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 进行高斯模糊
            blured = cv2.GaussianBlur(gray, (3, 3), 0)
        # 二值化
            _, binary = cv2.threshold(blured, 80, 255, cv2.THRESH_BINARY_INV)
        # 查找轮廓
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # 将结果放入队列
            self.result_queue.put((contours, hierarchy))
            self.opencv_ready_event.set()
            t2=time.time()
            print('opencv time:',t2-t1)
            if self.yolo_ready_event.is_set():
                self.runevent.set()
            else:
                self.runevent.clear()
           #  print('opencv process')


if __name__ == '__main__':
    # camera parameters
    f = 5.4
    fx = 2.50453554e+03
    fy = 2.50706889e+03
    cx = 1.23265844e+03
    cy = 9.45947471e+02
    # 畸变系数
    cameraMatrix = np.array([[fx, 0, cx],
                             [0, fy, cy],
                             [0, 0, 1]], dtype=np.double)
    distCoeffs = np.array([[-1.28826679e-01, 4.12884075e-01, 2.12200329e-03, 7.82174557e-04, -9.14304350e-01]],
                          dtype=np.double)

    # 初始化队列和设备

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型，路径需要修改
    # model = torch.hub.load('./', 'custom', './pretrained/yolov5n.pt',source='local', force_reload=False)
    model = torch.hub.load('./', 'custom', 'yolov5n_20240320_2.pt',source='local', force_reload=False)
    #model = torch.hub.load('./', 'custom', './pretrained/balanced400.pt',source='local', force_reload=False)
    model = model.to(device)
   
    yolo_img_queue = queue.Queue(maxsize=2)
    opencv_img_queue = queue.Queue(maxsize=2)
    yolo_result_queue = queue.Queue()
    opencv_result_queue = queue.Queue()
    runevent=Event()
    runevent.set()
    start_event=Event()
    start_event.clear()
    yolo_ready_event=Event()
    opencv_ready_event=Event()
    yolo_ready_event.set()
    opencv_ready_event.set()
    path = "dataset/TAG/display"
    # testimg=cv2.imread(os.path.join(path,"23.jpg"))
    # test_result=model(os.path.join(path,"23.jpg"))
    # print(test_result.pandas().xyxy[0]['class'])
    runflag=True
    yolo_thread = YOLODetectThread(yolo_img_queue, yolo_result_queue, model,runevent,yolo_ready_event,opencv_ready_event,runflag)
    opencv_thread = OpenCVProcessThread(opencv_img_queue, opencv_result_queue, cameraMatrix, distCoeffs,runevent,yolo_ready_event,opencv_ready_event,runflag)
    # file_thread = FileRead(path,yolo_img_queue,opencv_img_queue,start_event)
    # file_thread.start()
    camera_thread = CameraRead(yolo_img_queue, opencv_img_queue, start_event,runflag)
    camera_thread.start()
    start_event.wait()
    yolo_thread.start()
    opencv_thread.start()
    
    while runflag:
        # 等待两个线程的结果
        t1=time.time()
        yolo_result = yolo_result_queue.get()
        tyolo=time.time()
        opencv_result = opencv_result_queue.get()
        if yolo_result is not None: 
            img,detections_yolo = yolo_result
            
            # 过滤模型，获取classItem=0的识别结果
            newList = []
            for detection in detections_yolo:
                xmin, ymin, xmax, ymax, conf, classItem = detection[:6]
                if conf > 0.35:
                    newList.append([int(xmin), int(ymin), int(xmax), int(ymax), conf])
                

            if len(newList) > 0:
                for listItem in newList:
                    cv2.rectangle(img, (listItem[0], listItem[1]), (listItem[2], listItem[3]), (50, 255, 50), 3,
                          lineType=cv2.LINE_AA)
                
                    rect = np.array([[listItem[0], listItem[1]], [listItem[0], listItem[3]],
                                 [listItem[2], listItem[1]], [listItem[2], listItem[3]]], dtype=np.int32)
                a = Detect_Tag()
                # img = cv2.resize(img, (1272, 972))
                cv2.imshow("img", img)
                k=cv2.waitKey(1)
                if k == ord('q'):
                    runflag = False
                    break

                
                if opencv_result is not None:
                # Use the contours and hierarchy to finalize the location
                    (contours_opencv, hierarchy_opencv) = opencv_result
                    result_location = a.get_location(contours_opencv, hierarchy_opencv, rect, cameraMatrix, distCoeffs)
                    #print(result_location)
                    #cv2.waitKey(0)
                    # cv2.destroyAllWindows() 
            else:
                print("No detection")
                print(detections_yolo)
                # img = cv2.resize(img, (1272, 972))
                cv2.imshow("img", img)
                k=cv2.waitKey(1)
                if k == ord('q'):
                    runflag = False
                    break
            # cv2.destroyAllWindows()
        elif yolo_result == None and opencv_result == None:
            break
        t3 = time.time()
        print('load_yolo_time: ',tyolo-t1)
        print("total_time:", t3 - t1)

    cv2.destroyAllWindows()
    


