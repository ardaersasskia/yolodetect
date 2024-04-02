import cv2,time,os
camera_width=3280
camera_height=1848
fps=28
picture_width=3280
picture_height=1848
video_device=f'nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width={camera_width}, height={camera_height}, format=NV12, framerate=(fraction){fps}/1 ! nvvidconv ! video/x-raw, format=(string)BGRx , width={picture_width},height={picture_height}! videoconvert ! appsink max-buffers=1 drop=true'
cap = cv2.VideoCapture(video_device,cv2.CAP_GSTREAMER)
# cap.set(cv2.CAP_PROP_GSTREAMER_QUEUE_LENGTH,1)
ret,_=cap.read()
while True:
    ret,frame=cap.read()
    if ret:
        print(frame.shape)
        cv2.imshow(f"csi_camera {camera_width}x{camera_height}",cv2.resize(frame,(640,480)))
    else:
        print('no frame')
    k=cv2.waitKey(1)
    if k == 27:
        break
    elif k == ord('t'):
        position=input('Tag position: ')
        filename='csi'+position+'.jpg'
        picture_path=os.path.join('./csi_picture',filename)
        cv2.imwrite(picture_path,frame)
cv2.destroyAllWindows()
cap.release()

