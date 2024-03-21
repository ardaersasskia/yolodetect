import cv2
import socket
import time
from jetcam.csi_camera import CSICamera
# 创建一个UDP套接字
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 打开摄像头
cap = CSICamera(width=1280,height=720,capture_width=1280,capture_height=720,capture_fps=60)

while True:
    # 读取摄像头帧
    frame = cap.read()

    # 将帧转换为JPEG编码
    jpeg_data = cv2.imencode(".jpg", frame)[1].tobytes()

    # 发送图片数据
    sock.sendto(jpeg_data, ("127.0.0.1", 12345))
    cv2.imshow("Frame", frame)
    if cv2.waitKey(16) == 27:
        break

# 释放摄像头资源
cap.release()

print("摄像头已关闭")
