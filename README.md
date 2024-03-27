# 单目视觉测距

目前进度为Detect_Tag_no_opencv.py 

移除opencv的轮廓检测部分，增加csi摄像头支持，暂时修改回串行

将程序分为以下三个部分：

* [Detect_Tag_no_opencv.py](Detect_Tag_no_opencv.py)(程序主体)
* [Tagsolve.py](Tagsolve.py)(负责从识别框解算位置姿态)
* [Tagworker.py](Tagworker.py)(负责从读取图片到输出识别框)
