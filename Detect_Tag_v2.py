#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import collections
import torch
import time
import os
# from white_balance import white_balance

def circshift(matrix, shiftnum1, shiftnum2):
    h, w = matrix.shape
    matrix = np.vstack((matrix[(h - shiftnum1):, :], matrix[:(h - shiftnum1), :]))
    matrix = np.hstack((matrix[:, (w - shiftnum2):], matrix[:, :(w - shiftnum2)]))
    return matrix

class Detect_Tag:
    def __init__(self, img) -> None:
        self.img = img

        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
        self.img = cv2.filter2D(self.img, -1, kernel=kernel)

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
        print('外接矩形box:', box_2)
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
        print('外接矩形box:', box_2)
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
    def get_location(self, color, yolobox, cameraMatrix, distCoeffs, retVal = 0):
        # 转为灰度图像
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # 进行高斯模糊
        gr = cv2.GaussianBlur(gray, (3, 3), 0)
        # 将高斯模糊灰度图转换为二值图，参数ret为阈值，参数binary为转换后的二值图
        ret, binary = cv2.threshold(gr, 80, 255, cv2.THRESH_BINARY_INV)  # outdoor: 200
        # 寻找轮廓，contours为轮廓点，hierarchy为轮廓层次信息
        # hierarchy[0][i][y]
        # i为这组轮廓点的编号
        # y=0表示同一层次下一个轮廓编号
        # y=1表示同一层次上一个轮廓编号
        # y=2表示它的第一个子轮廓编号
        # y=3表示它的父轮廓编号
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        w, h, _ = self.img.shape
        final_contour = []

        for ic in range(len(contours)):
            # 取出一组轮廓点
            contour = contours[ic]
            # 计算其包围面积
            rect_area = cv2.contourArea(contour)
            # 计算轮廓的近似多边形  
            approx = cv2.approxPolyDP(contour, .03 * cv2.arcLength(contour, True), True)
            # 如果approx的点数为4说明这个多边形为四边形  为什么面积判断是1000而不是采用yolo识别框面积的50%
            if rect_area > 1000 and len(approx) == 4:
                # 计算这个轮廓的矩
                # m00为面积
                M = cv2.moments(contour)
                # 计算中心坐标x,y
                x = int(M['m10'] / M['m00'])
                y = int(M['m01'] / M['m00'])
                # 如果轮廓中心在yolobox的中心范围内
                if self.inside(yolobox, x, y):
                    # 测试发现在将hierarchy[0][ic][0] > -1改为hierarchy[0][ic][0] == -1时可以将多个框轮廓识别出来
                    # 认为应该将判断条件改为hierarchy[0][ic][0] == -1 and hierarchy[0][ic][2] == -1 
                    # 即选中每层内框的最后一个轮廓，判断是否是红色的内框使用面积是否大于yolo识别框面积的50%
                    if hierarchy[0][ic][0] > -1 and hierarchy[0][ic][2] == -1:
                        final_contour = contour
                        cv2.drawContours(self.img, final_contour, -1, (50, 255, 255), 3)
                        cv2.namedWindow("contours",0)
                        cv2.resizeWindow("contours", 1272, 972)
                        cv2.imshow('contours', self.img)
                        cv2.waitKey(0)

        if len(final_contour) < 1:
            print("no contour detected!")
            rect_r = cv2.minAreaRect(yolobox)
            yolobox = cv2.boxPoints(rect_r)
            return self.gettvec(yolobox, 3, cameraMatrix, distCoeffs)

        rect = cv2.minAreaRect(final_contour)  # cv::RotatedRect
        self.box_1 = cv2.boxPoints(rect)  # cv::Rect   左上 右上 右下 左下

        if retVal == 0:
            return self.gettvec(self.box_1, 2.3, cameraMatrix, distCoeffs)
        elif retVal == 1:
            return self.getRvec(self.box_1, 2.3, cameraMatrix, distCoeffs)



if __name__ == '__main__':
    # camera parameters
    f = 5.4
    fx = 2.50453554e+03
    fy = 2.50706889e+03
    cx = 1.23265844e+03
    cy = 9.45947471e+02
    # 畸变系数
    distCoeffs = np.array([[-1.28826679e-01, 4.12884075e-01, 2.12200329e-03, 7.82174557e-04, -9.14304350e-01]],
                          dtype=np.double)

    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.double)
    cameraMatrix = K

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型，路径需要修改
    model = torch.hub.load('./', 'custom', './pretrained/balanced400.pt',
                           source='local', force_reload=False)
    model = model.to(device)

    path = "dataset\\TAG\\display"
    files = os.listdir(path)
    for i, file in enumerate(files):
        img_path = os.path.join(path, file)
        img = cv2.imread(img_path)
        results = model(img_path)
        # 过滤模型，获取classItem=0的识别结果
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
            cv2.rectangle(img, (listItem[0], listItem[1]), (listItem[2], listItem[3]), (50, 255, 50), 3,
                          lineType=cv2.LINE_AA)

            rect = np.array([[listItem[0], listItem[1]], [listItem[0], listItem[3]],
                             [listItem[2], listItem[1]], [listItem[2], listItem[3]]], dtype=np.int32)

            a = Detect_Tag(img)
            result = a.get_location("red", rect, cameraMatrix, distCoeffs)
            print("result: ", result)
            print("图片名称：", file)
            print("")

            img = cv2.resize(img, (1272, 972))
            cv2.imshow(file, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No detection:" + file)


