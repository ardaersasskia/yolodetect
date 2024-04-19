import cv2
import torch
from Tagwork.Tagworker import Worker

if __name__ == '__main__':
    # 设备  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载模型，路径需要修改
    model = torch.hub.load('./', 'custom', './pretrained/20240406yolov5n400.pt',
                           source='local', force_reload=False)
    # cap=cv2.VideoCapture('WIN_20240402_14_22_42_Pro.mp4')
    camera_width=1920
    camera_height=1080
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    # 初始化摄像头
    myworker=Worker(model=model,
                    cap=cap,
                    testflag=False)
    while True:
        # 进行一帧运算
        result,img=myworker.workonce()
        width,height=img.shape[1],img.shape[0]
        if width>1920 or height>1080:
            img = cv2.resize(img, (int(width/4),int(height/4)))
        #img = cv2.resize(img, (int(myworker.imgWidth/4),int(myworker.imgHeight/4)))
        
        cv2.imshow("img", img)
        if result is None:
            k=cv2.waitKey(16)
            if k == 27:
                cv2.destroyAllWindows()
                break
            continue
        if result is not None:         
            print(result)
        k=cv2.waitKey(16)
        if k == 27:
            cv2.destroyAllWindows()
            break

    


