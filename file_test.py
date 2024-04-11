import cv2
import torch
from Tagworker import Worker

if __name__ == '__main__':
    # 设备 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载模型，路径需要修改
    model = torch.hub.load('./', 'custom', './pretrained/20240406yolov5n400.pt',
                           source='local', force_reload=False)
    # 初始化摄像头
    myworker=Worker(model=model,
                    testflag=True,
                    testdirectory='./usb_picture')
    with open('log.csv','w') as f:
        f.write('true x,x,true y,y\n')
    while True:  
        # 进行一帧运算
        result,img=myworker.workonce()
        img = cv2.resize(img, (int(myworker.imgWidth/4),int(myworker.imgHeight/4)))
        testp=myworker.testposition
        cv2.imshow("img", img)
        if result is not None:
            #cv2.waitKey(0)
            print(result)
            with open('log.csv','a') as f:
                f.write(f'{testp["x"]},{result[0][0]},{testp["y"]},{result[1][0]}\n')
        k=cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()
            break

    


