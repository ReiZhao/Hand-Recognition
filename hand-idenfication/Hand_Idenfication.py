# -*- coding: utf-8 -*-

'''
   核心思路：框定手势出现的区域，通过剔除背景找到手势，然后计算手势的特征向量（各种矩），最后标出手势的重心，手指尖
   该文件为初始版手势识别算法，其中主函数中的许多语句未封装进单独的小函数中！2019。04。21
'''
import cv2
from numpy import*
import copy
import math
import DNN_Classifier as dnn

# 参数
cap_region_x_begin = 0.5  # 起点/总宽度
cap_region_y_end = 0.8
threshold = 60  # 二值化阈值
blurValue = 41  # 高斯模糊参数
bgSubThreshold = 50
learningRate = 0

# 变量
isBgCaptured = 0  # 布尔类型, 背景是否被更新
triggerSwitch = False  # 如果正确，手势图像将被捕获
sampleSwitch = False  # 如果正确，提取一个作为模版的轮廓图
matchingSwitch = False  # 如果正确，进行模版匹配
# 自适应混合高斯背景建模，背景滤波，搜寻前景
bgModel = cv2.createBackgroundSubtractorMOG2(50, bgSubThreshold)


def printThreshold(thr):
    print("! Changed threshold to " + str(thr))


# 移除背景
def removeBG(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate)  # 计算前景掩膜
    kernel = ones((3, 3), uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)  # 使用特定的结构元素来侵蚀图像。
    res = cv2.bitwise_and(frame, frame, mask=fgmask)  # 使用掩膜移除静态背景
    return res


if __name__ == "__main__":
    # 相机/摄像头
    camera = cv2.VideoCapture(0)
    camera.set(10, 200)  # 设置视频属性
    cv2.namedWindow('trackbar')  # 设置窗口名字
    cv2.resizeWindow("trackbar", 640, 200)  # 重新设置窗口尺寸

    # createTrackbar是Opencv中的API，其可在显示图像的窗口中快速创建一个滑动控件，用于手动调节阈值，具有非常直观的效果。
    cv2.createTrackbar('threshold', 'trackbar', threshold, 100, printThreshold)
    count_step = 0
    while camera.isOpened():
        ret, frame = camera.read()
        # 返回滑动条上的位置的值（即实时更新阈值）
        threshold = cv2.getTrackbarPos('threshold', 'trackbar')
        # frame = cv2.cvtColor(frame,cv2.COLOR_RGB2YCrCb)
        # 双边滤波
        frame = cv2.bilateralFilter(frame, 5, 50, 100)
        frame = cv2.flip(frame, 1)  # 翻转  0:沿X轴翻转(垂直翻转)   大于0:沿Y轴翻转(水平翻转)   小于0:先沿X轴翻转，再沿Y轴翻转，等价于旋转180°
        # 画手势区域标记矩形框
        cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),(frame.shape[1], int(cap_region_y_end * frame.shape[0])), (0, 0, 255), 2)
        # 经过双边滤波后的初始化窗口
        cv2.imshow('original', frame)
        # 移除背景
        img = removeBG(frame)

        # 主要操作
        """
        # isBgCaptured == 1 表示开始执行手势框取
        """
        if isBgCaptured == 1:

            # 剪切右上角矩形框区域
            img = img[0:int(cap_region_y_end * frame.shape[0]), int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]
            # cv2.imshow('mask', img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将移除背景后的图像转换为灰度图
            blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)  # 加高斯模糊
            # cv2.imshow('blur', blur)
            ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)  # 二值化处理
            cv2.imshow('binary imagine', thresh)
            """
                开始执行轮廓框选，并计算HU矩，画出重心和指尖
            """
            if triggerSwitch:

                thresh1 = copy.deepcopy(thresh)
                # 寻找最大轮廓，认为此轮廓是手
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
                    res_max = contours[ci]

                    # 得出点集（组成轮廓的点）的凸包
                    hull = cv2.convexHull(res_max)
                    # 创建画图用数组
                    drawing = zeros(img.shape, uint8)
                    # 画出最大区域轮廓（绿色）| 把手势的外轮廓给画出来
                    cv2.drawContours(drawing, [res_max], 0, (0, 255, 0), 2)
                    # 画出凸包轮廓（红色）| 把手势轮廓包围起来的一个不规则凸多边形
                    cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)
                    # 求手势区域轮廓的各阶矩
                    moments = cv2.moments(res_max)
                    # 求手势区域轮廓的Hu矩
                    humoments = cv2.HuMoments(moments)
                    # count_step += 1
                    print("Hu矩：")
                    print(humoments)
                    dnn.Predict(humoments)
                    # print("当前手势提取次数：", count_step )
                    # write_file("hand_feature_wzz.txt", humoments)
                    # cv2.imwrite("good.jpg", thresh1)
                    # 求重心，画出重心
                    center = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
                    cv2.circle(drawing, center, 8, (0, 0, 255), -1)

                    # 寻找指尖
                    fingerRes = []  # 存储指尖信息的数组
                    max = 0
                    count = 0
                    notice = 0
                    cnt = 0
                    for i in range(len(res_max)):  # res，存储手势轮廓的几何信息
                        temp = res_max[i]
                        # 计算重心到手势轮廓边缘的距离
                        dist = (temp[0][0] - center[0])*(temp[0][0] - center[0])+(temp[0][1] - center[1])*(temp[0][1] - center[1])
                        if dist > max:
                            max = dist
                            notice = i
                        if dist != max:
                            count = count + 1
                            if count > 40:
                                count = 0
                                max = 0
                                flag = False
                                # 低于手心的点不算
                                if center[1] < res_max[notice][0][1]:
                                    continue
                                    # 离得太近的不算
                                for j in range(len(fingerRes)):
                                    # 使用绝对值，阈值若超过40很有可能会无法识别拇指
                                    if abs(res_max[notice][0][0]-fingerRes[j][0]) < 38:
                                        flag = True
                                        break
                                if flag:  # 找不到手势的时候就退出
                                    continue
                                fingerRes.append(res_max[notice][0])
                                # 画出指尖
                                cv2.circle(drawing, tuple(res_max[notice][0]), 8, (255, 0, 0), -1)
                                cv2.line(drawing, center, tuple(res_max[notice][0]), (255, 0, 0), 2)
                                cnt = cnt + 1

                    # 输出完成手部区域框取和手指识别后的图
                    cv2.imshow('output', drawing)
                    # 输出找到的指尖量
                    print("当前找到的指尖数量：", cnt)
                    triggerSwitch = False
            """
                将提取到的手势作为匹配模版
            """
            if sampleSwitch:
                res = res_max
                sampleSwitch = False
            """
                调用opencv自带的matchShapes()函数进行模版匹配
            """
            if matchingSwitch:
                print("匹配结果为：")
                # 调用库函数进行模版匹配
                match_result = cv2.matchShapes(res_max, res, 1, 0.0)
                print(match_result)
                # 创建一个矩形，在图片上写文字，参数依次定义了文字类型，高，宽，字体厚度
                font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体
                # 匹配阈值暂定为 0.15
                if match_result <= 0.15:
                    drawing = cv2.putText(drawing, "matching!", (0, 40), font, 1.2, (255, 255, 255), 2)
                if match_result > 0.15:
                    drawing = cv2.putText(drawing, "no_matching!", (0, 40), font, 1.2, (255, 255, 255), 2)
                cv2.imshow('output', drawing)
                matchingSwitch = False

        # 输入的键盘值
        k = cv2.waitKey(10)
        if k == 27:  # 按下ESC退出，并为文件更新一行
            f = open("hand_feature_wzz.txt", "a")
            f.write("\n")
            f.close()
            break
        # 按下'b'会开始执行手势区域划分等具体工作
        elif k == ord('b'):
            isBgCaptured = 1
            print('!!!Background Captured!!!')
        # 按下'r'会更新背景，适用于镜头对焦出问题的情况
        elif k == ord('r'):
            bgModel = cv2.createBackgroundSubtractorMOG2(50, bgSubThreshold)
            triggerSwitch = False
            # isBgCaptured = 0
            print('!!!Reset BackGround!!!')
        # 按下'c'会进行手势特征提取
        elif k == ord('c'):
            triggerSwitch = True
            print('!!!Catch hand!!!')
        # 按下'm'会进行手势匹配识别
        elif k == ord('s'):
            sampleSwitch = True
            print('!!! get a sample !!!')
        elif k == ord('m'):
            matchingSwitch = True
            print('!!! Matching... !!!')

    camera.release()
    cv2.destroyAllWindows()
