# -*- coding: utf-8 -*-

'''
 测试了一种避免脸部干扰的思路，把脸部直接剪除，
'''

import numpy as np
import cv2

imname = "test1.jpg"
img = cv2.imread(imname, cv2.IMREAD_COLOR)
cv2.imshow("原始图像", img)  # "image" 参数为图像显示窗口的标题, img是待显示的图像数据
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread(imname, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 识别输入图片中的人脸对象.返回对象的矩形尺寸
# faces：表示检测到的人脸目标序列
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100))
for (x, y, w, h) in faces:
    if w + h > 200:  # //针对这个图片画出最大的外框
        img2 = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 4)
cv2.imshow('人脸检测', img2)

img = cv2.imread(imname, cv2.IMREAD_COLOR)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 把图像转换到HSV色域
(_h, _s, _v) = cv2.split(hsv)  # 图像分割, 分别获取h, s, v 通道分量图像
skin3 = np.zeros(_h.shape, dtype=np.uint8)  # 根据源图像的大小创建一个全0的矩阵,用于保存图像数据
(x0, y0) = _h.shape  # 获取源图像数据的长和宽
# 遍历图像, 判断HSV通道的数值, 如果在指定范围中, 则置把新图像的点设为255,否则设为0
for (x, y, w, h) in faces:
    for i in range(0, x0):
        for j in range(0, y0):
            if (_h[i][j] > 11) and (_h[i][j] < 20) and (_s[i][j] > 28) and (_s[i][j] < 255) and (_v[i][j] > 50) and (
                 _v[i][j] < 255):
                skin3[i][j] = 255
            else:
                skin3[i][j] = 0
    for i in range(x, x+w):
        for j in range(y, y+h):
            skin3[i][j] = 0
    cv2.imshow(imname + " Skin3 HSV", skin3)
    cv2.waitKey(0)  # 等待键盘输入,参数表示等待时间,单位毫秒.0表示无限期等待
    cv2.destroyAllWindows()  # 销毁所有cv创建的窗口