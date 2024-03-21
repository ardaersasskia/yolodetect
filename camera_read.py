import cv2
import socket
import time

# 创建一个UDP套接字
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 打开摄像头
cap = cv2.VideoCapture(1,cv2.CAP_V4L2)

while True:
    # 读取摄像头帧
    ret, frame = cap.read()

    if not ret:
        break

    # 将帧转换为JPEG编码
    jpeg_data = cv2.imencode(".jpg", frame)[1].tobytes()

    # 发送图片数据
    sock.sendto(jpeg_data, ("127.0.0.1", 12345))

    cv2.waitKey(16)

# 释放摄像头资源
cap.release()

print("摄像头已关闭")
