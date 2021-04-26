# -*- coding: utf-8 -*-

import numpy as np
import cv2

# 参数
cap_region_x_begin = 0.5  # 起点/总宽度
cap_region_y_end = 0.8
threshold = 60  # 二值化阈值
blurValue = 41  # 高斯模糊参数
bgSubThreshold = 50
learningRate = 0

# 主函数
imname = "test1.jpg"
frame = cv2.imread(imname, cv2.IMREAD_COLOR)
frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
# 双边滤波
frame = cv2.bilateralFilter(frame, 5, 50, 100)
frame = cv2.flip(frame, 1)  # 翻转  0:沿X轴翻转(垂直翻转)   大于0:沿Y轴翻转(水平翻转)   小于0:先沿X轴翻转，再沿Y轴翻转，等价于旋转180°
# 画手势区域标记矩形框
cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
              (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (0, 0, 255), 2)
# 经过双边滤波后的初始化窗口
cv2.imshow('original', frame)

# 主要操作
if isBgCaptured == 1:  # isBgCaptured == 1 表示已经捕获背景
    img = removeBG(frame)  # 移除背景
    img = img[0:int(cap_region_y_end * frame.shape[0]),
          int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # 剪切右上角矩形框区域
    cv2.imshow('mask', img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将移除背景后的图像转换为灰度图
    blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)  # 加高斯模糊
    cv2.imshow('blur', blur)
    ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)  # 二值化处理
    cv2.imshow('binary', thresh)

    # 寻找最大轮廓，认为此轮廓是手
    thresh1 = copy.deepcopy(thresh)
    _, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 每次找到的轮廓总数
    length = len(contours)

    maxArea = -1
    if length > 0:
        for i in range(length):  # 找到最大的轮廓（根据面积）
            temp = contours[i]
            area = cv2.contourArea(temp)  # 计算轮廓区域面积
            if area > maxArea:
                maxArea = area
                ci = i
        # 得出最大的轮廓区域
        res = contours[ci]
        # 得出点集（组成轮廓的点）的凸包
        hull = cv2.convexHull(res)
        drawing = np.zeros(img.shape, np.uint8)
        # 画出最大区域轮廓
        cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
        # 画出凸包轮廓
        cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

        moments = cv2.moments(res)  # 求最大区域轮廓的各阶矩
        humoments = cv2.HuMoments(moments)
        print("Hu矩：")
        print(humoments)