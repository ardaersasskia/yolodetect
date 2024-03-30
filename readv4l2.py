from v4l2py.device import Device,BufferType,Memory
import numpy as np
import time
import cv2
imgHeight=720
imgWidth=1280
cam = Device.from_id(0)
cam.open()
print(cam.info.card)
print(cam.get_format(BufferType.VIDEO_CAPTURE))
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
    i,frame = cap.__next__()
    print(i)
    npdata=np.frombuffer(frame.data,dtype=np.uint16)
    bayerIMG = npdata.reshape(imgHeight, imgWidth, 1)/64
    grayIMG[:,:,0]=grayIMG[:,:,1]=grayIMG[:,:,2]=bayerIMG[:,:,0]
    # grayIMG=np.uint8(grayIMG)
    # rgbIMG = cv2.cvtColor(bayerIMG, cv2.COLOR_BayerRG2RGB)
    # rgbIMG = np.uint16(rgbIMG*32)
    #rgbIMG = rgbIMG.astype(np.uint8)
    t2=time.time()
    print('process time:',t2-t1)
    cv2.imshow('img',grayIMG)
    if cv2.waitKey(1) == 27:
        break
    
cv2.destroyAllWindows()
cam.close()

