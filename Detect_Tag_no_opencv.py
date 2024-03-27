from multiprocessing import Process, Queue, Event, Value
import cv2
import torch

from Tagworker import Worker
# 使用Gstreamer调用csi摄像头的参数,可按需更改sensor-id,width,height,framerate等
# drop = true 为溢出缓存时丢弃而不是等待，可有效降低摄像头延迟
video_device='nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=(fraction)60/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! appsink drop=true'

# 在jetson xavier nx上使用以下参数
in_jetson_xavier_nx=True
# 使用csi摄像头时，使用以下参数
using_csi = True
# 使用usb摄像头时，使用以下参数
# using_csi = False
# 在pc上使用以下参数
# in_jetson_xavier_nx=False
if __name__ == '__main__':
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载模型，路径需要修改
    model = torch.hub.load('./', 'custom', './yolov5n_20240320_2.pt',
                           source='local', force_reload=False)
    # 初始化摄像头
    if in_jetson_xavier_nx:
        if using_csi:
            cap = cv2.VideoCapture(video_device,cv2.CAP_GSTREAMER)
        else:
            cap = cv2.VideoCapture(1,cv2.CAP_V4L2)
    else:
        cap=cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret,frame=cap.read()
    # 初始化worker
    myworker=Worker(cap,model)

    while True:
        # 进行一帧运算
        result,img=myworker.workonce()
        # img = cv2.resize(img, (1272, 972))
        cv2.imshow("img", img)
        if result is not None:
            #cv2.waitKey(0)
            print(result)
        k=cv2.waitKey(1)
        if k == 27:
            cv2.destroyAllWindows()
            break
    cap.release()

    


