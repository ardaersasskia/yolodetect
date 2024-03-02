import cv2
import numpy as np
import collections
import torch

#  red : 255, 0, 0
#  blue : 0. 112, 192
#  green : 0, 176, 80
# camera 2.88mm
fx=1.32372830e+03
fy=1.32757714e+03
cx=1.36991960e+03
cy=1.01136064e+03
distCoeffs = np.array([[1.78066165e-01,-2.55983163e-01,-8.34980941e-04,1.98172701e-04,1.21655276e-01]], dtype=np.double)

K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]], dtype=np.double)  #内参矩阵
cameraMatrix = K

def check(contours, w, h):
    dist = []
    for contour in contours:
        M=cv2.moments(contour)
        x=int(M['m10'] / M['m00'])
        y=int(M['m01'] / M['m00'])
        dist.append((x-w/2)**2 + (y-h/2)**2)
    
    id = dist.index(min(dist))
    return contours[id]
     
class Detect_Tag:
    def __init__(self, path = 'test.png') -> None:
        self.img = cv2.imread(path)
        # cv2.imshow('origin', self.img)
        
    def getColorList(self):
        dict = collections.defaultdict(list)
 
        # 黑色
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 46])
        color_list = []
        color_list.append(lower_black)
        color_list.append(upper_black)
        dict['black'] = color_list

        # 红色2
        lower_red = np.array([0, 43, 46])
        upper_red = np.array([10, 255, 255])
        color_list = []
        color_list.append(lower_red)
        color_list.append(upper_red)
        dict['red'] = color_list
 
        # 绿色
        lower_green = np.array([35, 43, 46])
        upper_green = np.array([77, 255, 255])
        color_list = []
        color_list.append(lower_green)
        color_list.append(upper_green)
        dict['green'] = color_list
 
        # 蓝色
        lower_blue = np.array([100, 43, 46])
        upper_blue = np.array([124, 255, 255])
        color_list = []
        color_list.append(lower_blue)
        color_list.append(upper_blue)
        dict['blue'] = color_list
 
        return dict
 
    # 处理图片
    def get_color(self, color):
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        color_dict = self.getColorList()
        mask = cv2.inRange(hsv, color_dict[color][0], color_dict[color][1])
        binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        binary = cv2.dilate(binary, None, iterations=2)
        contours, heirarchy = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # cv2.drawContours(self.img, contours,-1, (0, 255, 0), 2)
        w, h, _ = self.img.shape
        final_contour = []
        for contour in contours:
            rect_area = cv2.contourArea(contour)
            approx = cv2.approxPolyDP(contour, .03 * cv2.arcLength(contour, True), True)
            if rect_area > 1000 and len(approx)==4 :
                cv2.drawContours(self.img, contour,-1, (0, 255, 0), 2)
                final_contour.append(contour)
        contour_ = check(final_contour, w, h)
        rect = cv2.minAreaRect(contour_)        # cv::RotatedRect
        self.box_1 = cv2.boxPoints(rect)                     # cv::Rect   左上 左下 右上 右下
        box = np.int0(cv2.boxPoints(rect))
        print(box)
        cv2.drawContours(self.img, [box], 0, (255, 0, 0), 2)

        cv2.namedWindow("resultfinal",0)
        cv2.resizeWindow("resultfinal",800,600)
        cv2.imshow('resultfinal', self.img)
        cv2.waitKey(0)
                
        cv2.waitKey(0)

    # 将self.box_1改为函数参数box1,用于接收神经网络识别得到的四个顶点
    def cal_Distance(self, color, box1):
        if color == "red":
            h = 4.0
        elif color == "blue":
            h = 2.0
        else:
            h = 0.75
        objPoints= np.array([[h, h, 0],
                      [-h, h, 0],
                      [-h, -h, 0],
                      [h, -h, 0]], dtype=np.double)
        
        def circshift(matrix, shiftnum1, shiftnum2):
            h, w = matrix.shape
            matrix = np.vstack((matrix[(h-shiftnum1):, :], matrix[:(h-shiftnum1), :]))
            matrix = np.hstack((matrix[:, (w-shiftnum2):], matrix[:, :(w-shiftnum2)]))
            return matrix
        
        square = []
        for i in range(0, 4):
            square.append(box1[i][0]**2 + box1[i][1]**2)
        arr_aa = np.array(square)
        minindex = np.argmin(arr_aa)
        box_2 = circshift(box1, 2-minindex, 0)
        print('外接矩形坐标值：', box_2)
        box_2 = np.array(box_2, dtype=np.double)
        
        retval_1, rvec_1, tvec_1 = cv2.solvePnP(objPoints, box_2, cameraMatrix, distCoeffs, False, cv2.SOLVEPNP_P3P)
        print("用外接矩形测量tvel=", tvec_1)
        return tvec_1

    
        
       
import os
 
if __name__ == '__main__':
    device = torch.device("cuda")
    # 加载模型，路径需要修改
    model = torch.hub.load('D:/PythonProjs/TAG', 'custom', 'D:/PythonProjs/TAG/pretrained/balanced200.pt',
                           source='local', force_reload=False)
    model = model.to(device)

    path = "D:/PythonProjs/TAG/dataset/TAG/test/whiteBalanced/img1"
    files = os.listdir(path)
    for i, file in enumerate(files):
        img_path = os.path.join(path, file)
        img = cv2.imread(img_path)
        results = model(img_path)
        # 过滤模型
        xmins = results.pandas().xyxy[0]['xmin']
        ymins = results.pandas().xyxy[0]['ymin']
        xmaxs = results.pandas().xyxy[0]['xmax']
        ymaxs = results.pandas().xyxy[0]['ymax']
        classList = results.pandas().xyxy[0]['class']
        confidences = results.pandas().xyxy[0]['confidence']
        newList = []
        for xmin, ymin, xmax, ymax, classItem, conf in zip(xmins, ymins, xmaxs, ymaxs, classList, confidences):
            if classItem == 0 and conf > 0.3:
                newList.append([int(xmin), int(ymin), int(xmax), int(ymax), conf])

        if len(newList) > 0:
            listItem = newList[0]
            cv2.rectangle(img, (listItem[0], listItem[1]), (listItem[2], listItem[3]), (50, 255, 50), 1,
                          lineType=cv2.LINE_AA)
            # cv2.putText(img, "conf:" + str(listItem[4]), (listItem[0], listItem[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL,
            #            1, (50, 255, 255))

            rect = np.array([[listItem[0], listItem[1]], [listItem[0], listItem[3]],
                             [listItem[2], listItem[1]], [listItem[2], listItem[3]]], dtype=np.int32)
            rect_r = cv2.minAreaRect(rect)
            box1 = cv2.boxPoints(rect_r)  # cv::Rect   左上 左下 右上 右下
            a = Detect_Tag(path=img_path)

            print("图片名称：", file)
            a.cal_Distance("red", box1)
            print("")
            img = cv2.resize(img, (1272, 972))
            cv2.imshow(file, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No detection:" + file)
