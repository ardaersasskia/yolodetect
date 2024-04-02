import cv2,os
camera_width=3840
camera_height=2160
fps=30
cap=cv2.VideoCapture(1,cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,camera_height)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,camera_width)
cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter.fourcc('M','J','P','G'))
# cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter.fourcc('Y','U','Y','V'))
cap.set(cv2.CAP_PROP_FPS,fps)
cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
cap.set(cv2.CAP_PROP_POS_FRAMES,0)
ret,frame=cap.read()
while True:
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret,frame=cap.read()
    if ret:
        print(frame.shape)
        cv2.imshow(f"usb_camera {camera_width}x{camera_height}",cv2.resize(frame,(640,360)))
        
    else:
        print('no frame')
    k=cv2.waitKey(1)
    if k == 27:
        break
    elif k==ord('t'):
        position=input('Tag position: ')
        filename='usb_'+position+'.jpg'
        picture_path=os.path.join('./usb_picture',filename)
        cv2.imwrite(picture_path,frame)
cv2.destroyAllWindows()
cap.release()