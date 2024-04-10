from multiprocessing import Process, Queue, Event, Value
import cv2
import torch
from v4l2py.device import Device,BufferType,Memory
from Tagworker import Worker


# 在jetson xavier nx上使用以下参数
in_jetson_xavier_nx=False
# 使用csi摄像头时，使用以下参数
using_csi = True
using_v4l2=False
# 使用usb摄像头时，使用以下参数
# using_csi = False
# 在pc上使用以下参数
# in_jetson_xavier_nx=False
if __name__ == '__main__':
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载模型，路径需要修改
    model = torch.hub.load('./', 'custom', './pretrained/20240402.pt',
                           source='local', force_reload=False)
    # 初始化摄像头
    if in_jetson_xavier_nx:
        if using_csi:
            if using_v4l2:
                cam = Device.from_id(0)
                cam.open()
                print(cam.info.card)
                print(cam.get_format(BufferType.VIDEO_CAPTURE))
                cap=enumerate(cam)
                _,frame=cap.__next__()
            else:
                # 使用Gstreamer调用csi摄像头的参数,可按需更改sensor-id,width,height,framerate等
                # drop = true 为溢出缓存时丢弃而不是等待，可有效降低摄像头延迟
                camera_width=3280
                camera_height=1848
                fps=28
                picture_width=3280
                picture_height=1848
                video_device=f'nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width={camera_width}, height={camera_height}, format=NV12, framerate=(fraction){fps}/1 ! nvvidconv ! video/x-raw, format=(string)BGRx , width={picture_width},height={picture_height}! videoconvert ! appsink max-buffers=1 drop=true'
                cap = cv2.VideoCapture(video_device,cv2.CAP_GSTREAMER)
                ret,frame=cap.read()
        else:
            # usb摄像头
            camera_width=3840
            camera_height=2160
            cap = cv2.VideoCapture(1,cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,camera_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    else:
        # pc测试
        camera_width=1280
        camera_height=720
        cap=cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
        ret,frame=cap.read()
    # 初始化worker
    myworker=Worker(cap,model,using_csi,using_v4l2,camera_height,camera_width)

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

    


