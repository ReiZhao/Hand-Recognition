# -*- coding: utf-8 -*-

import numpy as np
import cv2
'''
第一种肤色检测方式，基于YCrCb颜色空间的Cr分量+Otsu法阈值分割算法
'''


# 此种方法最差
def skin_yuv_cr():
    img = cv2.imread(imname, cv2.IMREAD_COLOR)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)  # 把图像转换到YUV色域
    (y, cr, cb) = cv2.split(ycrcb)  # 图像分割, 分别获取y, cr, br通道图像

    # 高斯滤波, cr 是待滤波的源图像数据, (5,5)是值窗口大小, 0 是指根据窗口大小来计算高斯函数标准差
    cr1 = cv2.GaussianBlur(cr, (5, 5), 0)  # 对cr通道分量进行高斯滤波
    # 根据OTSU算法求图像阈值, 对图像进行二值化
    _, skin1 = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # cv2.imshow("image CR", cr1)
    cv2.imshow("Skin Cr+OSTU", skin1)
    return


# 目前此种方法最好
def skin_yuv_crcb():
    img = cv2.imread(imname, cv2.IMREAD_COLOR)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)  # 把图像转换到YUV色域
    # (y, cr, cb) = cv2.split(ycrcb)  # 图像分割, 分别获取y, cr, br通道分量图像
    lower_skin = np.array([100, 133, 77])
    upper_skin = np.array([255, 173, 127])
    skin2 = cv2.inRange(ycrcb, lower_skin, upper_skin)
    # cv2.imshow(imname, img)
    cv2.imshow(imname + " Skin2 Cr+Cb", skin2)
    return


# 此种方法较好
def skin_hsv():
    # 肤色检测之三: HSV中 7<H<20 28<S<256 50<V<256
    img = cv2.imread(imname, cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 把图像转换到HSV色域
    # (_h, _s, _v) = cv2.split(hsv)  # 图像分割, 分别获取h, s, v 通道分量图像
    lower_skin = np.array([7, 28, 50])
    upper_skin = np.array([20, 255, 255])
    skin3 = cv2.inRange(hsv, lower_skin, upper_skin)

    # cv2.imshow(imname, img)
    cv2.imshow(imname + " Skin3 HSV", skin3)
    return


# 人脸框选算法
def face_select():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img = cv2.imread(imname, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 识别输入图片中的人脸对象.返回对象的矩形尺寸
    # faces：表示检测到的人脸目标序列
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100))
    for (x, y, w, h) in faces:
        if w + h > 200:  # //针对这个图片画出最大的外框
            img2 = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 4)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

    cv2.imshow('img', img2)
    return


if __name__ == "__main__":
    imname = "test1.jpg"
    '''
        cv2.IMREAD_COLOR 表示读入一副彩色图像, alpha 通道被忽略, 默认值
        cv2.IMREAD_ANYCOLOR 表示读入一副彩色图像
        cv2.IMREAD_GRAYSCALE 表示读入一副灰度图像
        cv2.IMREAD_UNCHANGED 表示读入一幅图像，并且包括图像的 alpha 通道
    '''
    img = cv2.imread(imname, cv2.IMREAD_COLOR)
    cv2.imshow("image", img)  # "image" 参数为图像显示窗口的标题, img是待显示的图像数据
    # skin_yuv_cr()
    skin_yuv_crcb()
    # skin_hsv()
    # face_select()
    cv2.waitKey(0)  # 等待键盘输入,参数表示等待时间,单位毫秒.0表示无限期等待

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()  # 销毁所有cv创建的窗口