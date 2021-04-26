# -*- coding: utf-8 -*-

import numpy as np
import cv2
cap = cv2.VideoCapture(0)

while(1):
    ret, img = cap.read()
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)  # 把图像转换到YUV色域
    (y, cr, cb) = cv2.split(ycrcb)  # 图像分割, 分别获取y, cr, br通道图像
    # 高斯滤波, cr 是待滤波的源图像数据, (5,5)是值窗口大小, 0 是指根据窗口大小来计算高斯函数标准差
    cr1 = cv2.GaussianBlur(cr, (5, 5), 0)  # 对cr通道分量进行高斯滤波
    # 根据OTSU算法求图像阈值, 对图像进行二值化
    _, skin1 = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 形态学运算处理
    # 定义结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # 开运算用于移除由图像噪音形成的斑点
    skin1 = cv2.morphologyEx(skin1, cv2.MORPH_OPEN, kernel)

    # 闭运算用来连接被误分为许多小块的对象，
    skin1 = cv2.morphologyEx(skin1, cv2.MORPH_CLOSE, kernel)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 识别输入图片中的人脸对象.返回对象的矩形尺寸
    # faces：表示检测到的人脸目标序列
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100))
    for (x, y, w, h) in faces:
        if w + h > 200:  # //针对这个图片画出最大的外框
            skin1 = cv2.rectangle(skin1, (x, y), (x + w, y + h), (255, 255, 255), 4)
    cv2.imshow("Face", skin1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()