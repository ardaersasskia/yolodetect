import cv2
import numpy as np
import math

class Solve_position:
    def __init__(self,rect,isoutside=True) -> None:
        self.rect=rect
        self.isoutside=isoutside
        # 畸变系数
        # fx = 2.50453554e+03
        # fy = 2.50706889e+03
        # cx = 1.23265844e+03
        # cy = 9.45947471e+02
        fx = 2.10540759e+03
        fy = 2.10584118e+03
        cx = 1.88670808e+03
        cy = 1.03368485e+03
        self.cameraMatrix=np.array([[fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1]], dtype=np.double)
        self.distCoeffs = np.array([[0.07998255 ,-0.11608255 , 0.00055299 ,-0.00232755  ,0.05645519]],
                        dtype=np.double)
    def circshift(self,matrix, shiftnum1, shiftnum2):
        h, w = matrix.shape
        matrix = np.vstack((matrix[(h - shiftnum1):, :], matrix[:(h - shiftnum1), :]))
        matrix = np.hstack((matrix[:, (w - shiftnum2):], matrix[:, :(w - shiftnum2)]))
        return matrix

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
        objPoints = np.array([[ h,  h, 0],
                              [-h,  h, 0],
                              [-h, -h, 0],
                              [ h, -h, 0]], dtype=np.double)

        square = []
        for i in range(0, 4):
            square.append(box[i][0] ** 2 + box[i][1] ** 2)
        arr_aa = np.array(square)
        minindex = np.argmin(arr_aa)

        # 顶点坐标旋转（右下 左下 左上 右上）
        #print(box)
        #print(minindex)
        box_2 = self.circshift(box, 2 - minindex, 0)
        #print('外接矩形box:', box_2)
        box_2 = np.array(box_2, dtype=np.double)

        # retval_1,rvec_1,tvec_1,_ = cv2.solvePnPRansac(objPoints, box_2, cameraMatrix, distCoeffs, False, cv2.SOLVEPNP_P3P)
        _, _, tvec_1 = cv2.solvePnP(objPoints, box_2, cameraMatrix, distCoeffs, False, cv2.SOLVEPNP_P3P)

        return tvec_1[2] , -tvec_1[0] , -tvec_1[1] , True

    # 处理图片
    def get_location(self,counter_ret,h_pre):
        rect_r = cv2.minAreaRect(self.rect)
        yolobox = cv2.boxPoints(rect_r)
        area=cv2.contourArea(self.rect)/1000
        
        if self.isoutside:
            if counter_ret:
                h = 0.0543*math.log(area) + 1.7706
            else:
                h=2.5
        else:
            h= -2e-07*area*area - 0.0001*area + 0.7962

        with open('log.csv','a') as f:
            f.write(f'{area},{h},')
        return self.gettvec(yolobox, h, self.cameraMatrix, self.distCoeffs)