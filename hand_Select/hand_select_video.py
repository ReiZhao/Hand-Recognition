# -*- coding: utf-8 -*-

'''

    核心思路：先进行人脸剔除，然后依据肤色建模进行手部区域的动态框选

'''

import numpy as np
import cv2
cap = cv2.VideoCapture(0)
# history = 10
fgbg = cv2. createBackgroundSubtractorMOG2(detectShadows=False)
# fgbg.setHistory(history)
while(1):
    img_hand = 0
    skin = 0
    ret, img = cap.read()
    # 求原始图像的灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 前景分离
    # img = fgbg.apply(img)
    # '''
    # 把图像转换到YUV色域
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # 图像分割, 分别获取y, cr, br通道图像
    (y, cr, cb) = cv2.split(ycrcb)
    # 高斯滤波, cr 是待滤波的源图像数据, (5,5)是值窗口大小, 0 是指根据窗口大小来计算高斯函数标准差
    cr1 = cv2.GaussianBlur(cr, (5, 5), 0)  # 对cr通道分量进行高斯滤波

    # skin1 肤色区域帧
    # 根据OTSU算法求图像阈值, 对图像进行二值化
    _, skin1 = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 定义结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # 开运算用于移除由图像噪音形成的斑点
    skin1 = cv2.morphologyEx(skin1, cv2.MORPH_OPEN, kernel)
    # 闭运算用来连接被误分为许多小块的对象，
    skin1 = cv2.morphologyEx(skin1, cv2.MORPH_CLOSE, kernel)
    # '''
    ''' （ycrcb 双通道划分 效果不好）
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)  # 把图像转换到YUV色域
    # (y, cr, cb) = cv2.split(ycrcb)  # 图像分割, 分别获取y, cr, br通道分量图像
    lower_skin = np.array([150, 133, 67])
    upper_skin = np.array([255, 193, 127])
    skin2 = cv2.inRange(ycrcb, lower_skin, upper_skin)
    
    # 定义结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # 开运算用于移除由图像噪音形成的斑点
    skin2 = cv2.morphologyEx(skin2, cv2.MORPH_OPEN, kernel)
    # 闭运算用来连接被误分为许多小块的对象，
    skin2 = cv2.morphologyEx(skin2, cv2.MORPH_CLOSE, kernel)
    '''
    ''' HSV 效果同样不好
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 把图像转换到HSV色域
    (_h, _s, _v) = cv2.split(hsv)  # 图像分割, 分别获取h, s, v 通道分量图像
    lower_skin = np.array([2, 50, 50])
    upper_skin = np.array([15, 255, 255])
    skin3 = cv2.inRange(hsv, lower_skin, upper_skin)

    # 定义结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # 开运算用于移除由图像噪音形成的斑点
    skin3 = cv2.morphologyEx(skin3, cv2.MORPH_OPEN, kernel)
    # 闭运算用来连接被误分为许多小块的对象，
    skin3 = cv2.morphologyEx(skin3, cv2.MORPH_CLOSE, kernel)
    '''
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cv2.imshow("hand", skin1)
    # 识别输入图片中的人脸对象.返回对象的矩形尺寸
    # faces：表示检测到的人脸目标序列
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100))
    for (x, y, w, h) in faces:
        if w + h > 200:  # //针对这个图片画出最大的外框
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 4)

            # 裁剪掉脸部区域
            # skin1_face:裁剪掉脸部之后的图片帧
            skin1 = skin1[:, x+w:]
            img_hand = img[:, x+w-100:]


    # 搜寻闭合轮廓,并选取其中的最大轮廓，认为这就是手部区域
    # hand 手部区域帧
    hand, contours, hierarchy = cv2.findContours(skin1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 记录每一个闭合轮廓的面积
    S = np.zeros(len(contours))

    for i in range(0, len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        S[i] = w * h
        if i == 0:
            S_max = 0
            i_max = 0
    for i in range(0, len(contours)):
        if S[i] > S_max:
            S_max = S[i]
            i_max = i

    # 剔除提取轮廓时失败帧的影响
    if len(contours) == 0:
        continue

    # 返回最大轮廓的参数
    x0, y0, w0, h0 = cv2.boundingRect(contours[i_max])
    cv2.rectangle(img_hand, (x0, y0), (x0 + w0, y0 + h0), (0, 255, 0), 5)

    # 裁剪得到的手部轮廓为一张新的图片
    # hand = hand[y0:y0 + h0, x0:x0 + w0]

    # 形态学处理手部区域图片
    # 腐蚀图像
    # hand = cv2.erode(hand, kernel)
    # 膨胀图像
    # hand = cv2.dilate(hand, kernel)
    # 闭运算用来连接被误分为许多小块的对象，
    hand = cv2.morphologyEx(hand, cv2.MORPH_CLOSE, kernel)
    # 开运算用于移除由图像噪音形成的斑点
    # hand = cv2.morphologyEx(hand, cv2.MORPH_OPEN, kernel)
    # 显示原图像
    cv2.imshow("Original", img)
    cv2.imshow("FACE", img_hand)
    # 显示提取到的手部
    cv2.imshow("HAND", hand)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 关闭摄像头 释放内存
cap.release()
cv2.destroyAllWindows()