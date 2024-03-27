from multiprocessing import Process, Queue, Event, Value
import cv2
import torch

from Tagworker import Worker

if __name__ == '__main__':
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载模型，路径需要修改
    model = torch.hub.load('./', 'custom', './yolov5n_20240320_2.pt',
                           source='local', force_reload=False)
    cap = cv2.VideoCapture(0)
    myworker=Worker(cap,model)

    while True:
        result,img=myworker.workonce()
        # img = cv2.resize(img, (1272, 972))
        cv2.imshow("img", img)
        if result is not None:
            cv2.waitKey(0)
        k=cv2.waitKey(1)
        if k == 27:
            cv2.destroyAllWindows()
            break
    cap.release()

    


