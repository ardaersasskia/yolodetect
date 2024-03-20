import cv2,time
cap=cv2.VideoCapture(0)
# while True:
#     t0=time.time()
#     ret,frame=cap.read()
#     t1=time.time()
#     cv2.imshow("frame",frame)
#     cv2.waitKey(16)
#     if ret:
#         print("读取视频帧时间：",t1-t0)
# ret,frame=cap.read()
# print("read first")
# cv2.waitKey(5000)
# t0=time.time()
# ret,frame=cap.read()
# t1=time.time()
# print("读取视频帧时间：",t1-t0)
while True:
    #cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret,frame=cap.read()
    print(frame.shape)
    cv2.imshow("frame",frame)
    k=cv2.waitKey(1)
    if k == ord('q'):
        break
cv2.destroyAllWindows()

