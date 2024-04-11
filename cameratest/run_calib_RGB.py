# -*- coding: utf-8 -*-
"""
Calibrate the Camera with Zhang Zhengyou Method.
Picture File Folder: "./pic/RGB_camera_calib_img/", Without Distortion. 

By You Zhiyuan, 2022.07.04, zhiyuanyou@foxmail.com
"""

import os

from calibrate_helper import Calibrator


def main():
    img_dir = "./cameratest/chessboard"
    shape_inner_corner = (10, 7)
    size_grid = 0.015
    # create calibrator
    calibrator = Calibrator(img_dir, shape_inner_corner, size_grid)
    # calibrate the camera
    mat_intri, coff_dis = calibrator.calibrate_camera()
    print("Camera Matrix: \n", mat_intri)
    print("Distortion Coefficients: \n", coff_dis)

if __name__ == '__main__':
    main()
