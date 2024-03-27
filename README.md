# 单目视觉测距

目前进度为Detect_Tag_no_opencv.py

移除opencv的轮廓检测部分，增加csi摄像头支持，暂时修改回串行

将程序分为以下三个部分：

* [Detect_Tag_no_opencv.py](Detect_Tag_no_opencv.py)(程序主体)
* [Tagsolve.py](Tagsolve.py)(负责从识别框解算位置姿态)
* [Tagworker.py](Tagworker.py)(负责从读取图片到输出识别框)

# Camera Latency

使用csi摄像头时延时依然很大，检测Gstreamer pipe耗时仅在7-12ms左右

![1711558489285](image/README/1711558489285.png)

发现官方论坛中有说明使用gstreamer的命令会有不可避免的5帧缓冲，60fps下大约为80ms的延时，与测试结果相符，这样一来就能解释为何会有如此大的摄像头延时了

![1711557682423](image/camera_latency/1711557682423.png)

根据官方说明，下一步可以尝试使用[argus](https://docs.nvidia.com/jetson/l4t-multimedia/group__LibargusAPI.html)？

> Libargus is designed to address a number of fundamental requirements:
>
> * Support for a wide variety of use cases (traditional photography, computational photography, video, computer vision, and other application areas.) To this end, libargus is a frame-based API; every capture is triggered by an explicit request that specifies exactly how the capture is to be performed.
> * Support for multiple platforms, including L4T and Android.
> * Efficient and simple integration into applications and larger frameworks. In support of this, libargus delivers images with EGLStreams, which are directly supported by other system components such as OpenGL and Cuda, and which ***require no buffer copies during delivery to the consumer***.
> * Expansive metadata along with each output image.
> * Support for multiple sensors, including both separate control over independent sensors and access to synchronized multi-sensor configurations. (The latter are unsupported in the current release. When support is added, it will be available on only some NVIDIA platforms.)
> * Version stability and extensibility, which are provided by unchanging virtual interfaces and the ability for vendors to add specialized extension interfaces.

坏消息 nvarguscamerasrc 已经时使用了argus，看来要试试通过v4l2来绕过这个板载ISP的4帧缓存![1711563548461](image/README/1711563548461.png)
