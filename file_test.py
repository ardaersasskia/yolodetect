import cv2
import torch
from Tagwork.Tagworker import Worker

if __name__ == '__main__':
    # 设备 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载模型，路径需要修改
    model = torch.hub.load('./', 'custom', './pretrained/20240406yolov5n400.pt',
                           source='local', force_reload=False)
    # 初始化摄像头
    myworker=Worker(model=model,
                    testflag=True,
                    testdirectory='./dataset/testpic')
    with open('log.csv','w') as f:
        f.write('area,h,true x,compute_x,true y,compute_y,true z,compute_z,delta_x,delta_y,delta_z \n')
    while True:
        # 进行一帧运算
        result,img=myworker.workonce()
        if result is None:
            cv2.destroyAllWindows()
            break
        img = cv2.resize(img, (int(myworker.imgWidth/4),int(myworker.imgHeight/4)))
        testp=myworker.testposition
        cv2.imshow("img", img)
        if result is not None:         
            #cv2.waitKey(0)
            print(result)
            with open('log.csv','a') as f:
                true_x=testp["x"]
                true_y=testp["y"]
                true_z=testp["z"]
                compute_x=result[0][0]
                compute_y=result[1][0]
                compute_z=-result[2][0]
                delta_x=compute_x-true_x
                delta_y=compute_y-true_y
                delta_z=compute_z-true_z               
                f.write(f'{true_x},{compute_x},{true_y},{compute_y},{true_z},{compute_z},{delta_x},{delta_y},{delta_z}\n')
        k=cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()
            break

    


