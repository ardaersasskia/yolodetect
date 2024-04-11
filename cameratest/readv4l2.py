from v4l2py.device import Device,BufferType,Memory
import numpy as np
import time
import cv2
imgHeight=1848
imgWidth=3280
cam = Device.from_id(0)
cam.open()
print(cam.info.card)
print(cam.get_format(BufferType.VIDEO_CAPTURE))
cam.controls.bypass_mode=0
# cam.create_buffers(BufferType.VIDEO_CAPTURE,Memory.MMAP,1)
# cam.stream_on(BufferType.VIDEO_CAPTURE)
cap=enumerate(cam)

grayIMG = np.zeros([imgHeight, imgWidth, 3], dtype='uint8')
# 消耗起始缓存
i,_ = cap.__next__()
i,_ = cap.__next__()
i,_ = cap.__next__()
i,_ = cap.__next__()
i,_ = cap.__next__()
i,_ = cap.__next__()
while True:
    t1=time.time()
    i,_ = cap.__next__()
    i,frame = cap.__next__()
    t2=time.time()
    # print(i)
    npdata=np.frombuffer(frame.data,dtype=np.uint16)
    bayerimg=npdata.reshape(imgHeight, imgWidth)/64
    grayIMG[:,:,0]=grayIMG[:,:,1]=grayIMG[:,:,2]=bayerimg
    preview=cv2.resize(grayIMG,(640,360))
    # grayIMG=np.uint8(grayIMG)
    # rgbIMG = cv2.cvtColor(bayerIMG, cv2.COLOR_BayerRG2RGB)
    # rgbIMG = np.uint16(rgbIMG*32)
    #rgbIMG = rgbIMG.astype(np.uint8)
    t3=time.time()
    print('read time:',t2-t1)
    print('process time:',t3-t2)
    cv2.imshow(f'v4l2 csi {imgWidth}x{imgHeight}',preview)
    if cv2.waitKey(1) ==27:
        break
    
cv2.destroyAllWindows()
cam.close()

