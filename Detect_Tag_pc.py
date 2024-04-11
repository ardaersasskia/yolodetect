from multiprocessing import Process, Queue, Event, Value
import cv2
import torch
from Tagwork.Tagworker import Worker


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
    model = torch.hub.load('./', 'custom', './pretrained/20240406yolov5n400.pt',
                           source='local', force_reload=False)

    camera_width=3840
    camera_height=2160
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 初始化worker
    myworker=Worker(cap=cap,
                    model=model,
                    imgHeight=camera_height,
                    imgWidth=camera_width,
                    testflag=False,
                    testdirectory='./usb_picture')

    while True:
        # 进行一帧运算
        result,img=myworker.workonce()
        img = cv2.resize(img, (1280, 720))
        cv2.imshow("img", img)
        if result is not None:
            #cv2.waitKey(0)
            print(result)
        k=cv2.waitKey(16)
        if k == 27:
            cv2.destroyAllWindows()
            break
    cap.release()

    


