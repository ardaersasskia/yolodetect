# Camera Latency

使用gstreamer的命令会有不可避免的5帧缓冲，60fps下大约为80ms

![1711557682423](image/camera_latency/1711557682423.png)

这样一来就能解释为何会有如此大的摄像头延时了

![1711557771806](image/camera_latency/1711557771806.png)

下一步可以尝试使用[argus](https://docs.nvidia.com/jetson/l4t-multimedia/group__LibargusAPI.html)？