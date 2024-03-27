import cv2
import socket
import numpy as np
from jetcam.csi_camera import CSICamera
# 创建一个UDP套接字
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('localhost', 12345))

# 打开摄像头
cap = CSICamera(width=1280,height=720,capture_width=1280,capture_height=720,capture_fps=60)
server_socket.listen(1)
client_socket, client_address = server_socket.accept()
print(f"Client connected: {client_address}")
while True:
    # 读取摄像头帧
    frame = cap.read()

    # 将帧转换为JPEG编码
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    result, encoded_frame = cv2.imencode('.jpg', frame, encode_param)
    # 发送图片数据
    data = np.array(encoded_frame)
    string_data = data.tostring()
    client_socket.sendall(string_data)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(16) == 27:
        break
client_socket.close()
server_socket.close()
# 释放摄像头资源
cap.release()

print("摄像头已关闭")
