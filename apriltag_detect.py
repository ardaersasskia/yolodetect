#!/usr/bin/env python
# coding: UTF-8
#import apriltag
import pupil_apriltags as apriltag     # for windows
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
#at_detector = apriltag.Detector(apriltag.DetectorOptions(families='tag36h11 tag25h9 tag16h5') )
at_detector = apriltag.Detector(families='tag25h9')  #for windows

i=0
while(1):
    # 获得图像
    ret, frame = cap.read()
    # 检测按键
    k=cv2.waitKey(1)
    if k==27:
        break
    elif k==ord('s'):
        cv2.imwrite('E:/OpenCV_pic/'+str(i)+'.jpg', frame)
        i+=1
    # 检测apriltag
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = at_detector.detect(gray)
    for tag in tags:
        cv2.circle(frame, tuple(tag.corners[0].astype(int)), 4, (255, 0, 0), 2) # left-top
        cv2.circle(frame, tuple(tag.corners[1].astype(int)), 4, (255, 0, 0), 2) # right-top
        cv2.circle(frame, tuple(tag.corners[2].astype(int)), 4, (255, 0, 0), 2) # right-bottom
        cv2.circle(frame, tuple(tag.corners[3].astype(int)), 4, (255, 0, 0), 2) # left-bottom
    # 显示检测结果
    cv2.imshow('capture', frame)

cap.release()
cv2.destroyAllWindows()
