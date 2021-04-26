# -*- coding: utf-8 -*-

from numpy import*
import cv2
import copy
import math

threshold = 230  # 二值化阈值
blurValue = 41  # 高斯模糊参数
bgSubThreshold = 50


# 需要给函数添一个次数input，这样每次可以换行
def write_file(file_name, source):
    """
    保存识别的手势Hu矩参数，

    input:  file_name：写入的文件名
            source：数据来源
            s_n：数据批次（第几行）
    output:
    """
    f = open(file_name, "a")
    m, n = shape(source)
    # for s_n in range(smaple_n):
    for i in range(m):
        tmp = []
        for j in range(n):
            souce_temp = ("%.12f" % (source[i, j]))
            tmp.append(str(souce_temp))
        f.write("\t".join(tmp) + ",")

    # 写入完一个手势的7个Hu矩后换行
    f.write("\n")
    f.close()


if __name__ == "__main__":
        imname = "test_image.jpg"
        '''
            cv2.IMREAD_COLOR 表示读入一副彩色图像, alpha 通道被忽略, 默认值
            cv2.IMREAD_ANYCOLOR 表示读入一副彩色图像
            cv2.IMREAD_GRAYSCALE 表示读入一副灰度图像
            cv2.IMREAD_UNCHANGED 表示读入一幅图像，并且包括图像的 alpha 通道
        '''
        img = cv2.imread(imname, cv2.IMREAD_COLOR)
        cv2.imshow("image", img)  # "image" 参数为图像显示窗口的标题, img是待显示的图像数据
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将移除背景后的图像转换为灰度图
        # cv2.imshow("gray", gray)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)  # 加高斯模糊
        # cv2.imshow('blur', blur)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)  # 二值化处理
        # cv2.imshow('binary_image', thresh)
        # 寻找最大轮廓，认为此轮廓是手
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
            res_max = contours[ci]

            # 得出点集（组成轮廓的点）的凸包
            hull = cv2.convexHull(res_max)
            # 创建画图用数组
            drawing = zeros(img.shape, uint8)
            # 画出最大区域轮廓（绿色）|
            cv2.drawContours(drawing, [res_max], 0, (0, 255, 0), 2)
            cv2.imshow('contours', drawing)
            # 求手势区域轮廓的各阶矩
            moments = cv2.moments(res_max)
            # 求手势区域轮廓的Hu矩
            humoments = cv2.HuMoments(moments)
            print("Hu矩：")
            print(humoments)
        write_file("hand_feature.txt", humoments)
        write_file("hand_feature.txt", humoments)
        cv2.waitKey(0)  # 等待键盘输入,参数表示等待时间,单位毫秒.0表示无限期等待

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()  # 销毁所有cv创建的窗口
