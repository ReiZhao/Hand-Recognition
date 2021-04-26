import asyncio
import json
import cv2
from numpy import*
import math
import DNN_Classifier as dnn
import asyncws
import time
import threading

"""
    与java服务器实现远程通信，完成手势控制功能
"""

ACCESS_TOKEN = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiI3ZDg2Mzc3NzIwYjQ0M2YyOWI2MzE2ZTdmMjI3Njc0OCIsImlhdCI6MTU0MzYwMTY1OCwiZXhwIjoxODU4OTYxNjU4fQ.uSatzdHOC-ozC9OnI0pUk63Mtuawy7bauRG6k-swP9g'


# insideCurtain:cover.curtain_158d000202e843 outsideCurtain:cover.curtain_158d000202e545
# cover.open_cover cover.close_cover cover.set_cover_position


def projector_in(index):
    return json.dumps(
        {"id": index, "type": "call_service", "domain": "switch", "service": "turn_on", "service_data": {
            "entity_id": "switch.projector"
        }}
    )


def projector_out(index):
    return json.dumps(
        {"id": index, "type": "call_service", "domain": "switch", "service": "turn_off", "service_data": {
            "entity_id": "switch.projector"
        }}
    )


def close_left_light(index):
    return json.dumps(
        {"id": index, "type": "call_service", "domain": "switch", "service": "turn_off", "service_data": {
            "entity_id": "switch.wall_switch_left_158d00023f0a73"
        }}
    )


def open_left_light(index):
    return json.dumps(
        {"id": index, "type": "call_service", "domain": "switch", "service": "turn_on", "service_data": {
            "entity_id": "switch.wall_switch_left_158d00023f0a73"
        }}
    )


def close_right_light(index):
    return json.dumps(
        {"id": index, "type": "call_service", "domain": "switch", "service": "turn_off", "service_data": {
            "entity_id": "switch.wall_switch_right_158d00023f0a73"
        }}
    )


def open_right_light(index):
    return json.dumps(
        {"id": index, "type": "call_service", "domain": "switch", "service": "turn_on", "service_data": {
            "entity_id": "switch.wall_switch_right_158d00023f0a73"
        }}
    )



def open_curtain_white(index):
    return json.dumps(
        {"id": index, "type": "call_service", "domain": "cover", "service": "open_cover", "service_data": {
            "entity_id": "cover.curtain_158d000202e843_2"
        }}
    )


def open_curtain_black(index):
    return json.dumps(
        {"id": index, "type": "call_service", "domain": "cover", "service": "open_cover", "service_data": {
            "entity_id": "cover.curtain_158d000202e545_2"
        }}
    )


def close_curtain_white(index):
    return json.dumps(
        {"id": index, "type": "call_service", "domain": "cover", "service": "close_cover", "service_data": {
            "entity_id": "cover.curtain_158d000202e843_2"
        }}
    )


def close_curtain_black(index):
    return json.dumps(
        {"id": index, "type": "call_service", "domain": "cover", "service": "close_cover", "service_data": {
            "entity_id": "cover.curtain_158d000202e545_2"
        }}
    )


def open_curtain_white_position(index, position):
    return json.dumps(
        {"id": index, "type": "call_service", "domain": "cover", "service": "set_cover_position", "service_data": {
            "entity_id": "cover.curtain_158d000202e843_2", "position": position
        }}
    )


def open_curtain_black_position(index, position):
    return json.dumps(
        {"id": index, "type": "call_service", "domain": "cover", "service": "set_cover_position", "service_data": {
            "entity_id": "cover.curtain_158d000202e545_2", "position": position
        }}
    )


async def main():
    """Simple WebSocket client for Home Assistant."""
    websocket = await asyncws.connect('ws://192.168.50.208:8123/api/websocket')

    # await websocket.send(json.dumps(
    #     {'type': 'auth',
    #      'access_token': ACCESS_TOKEN}
    # ))

    # await websocket.send(json.dumps(
    #     {'id': 1, 'type': 'subscribe_events', 'event_type': 'state_changed'}
    # ))



    # await  websocket.send(json.dumps(
    #     {"id": 3, "type": "call_service", "domain": "light", "service": "turn_off", "service_data": {
    #         "entity_id": "light.yeelight_rgb_7c49eb138d2d"
    #     }}
    # ))

    # await websocket.send(json.dumps(
    #     {'id': 3, 'type': 'get_states'}
    # ))

    # await websocket.send(json.dumps(
    #     {'id': 4, 'type': 'get_config'}
    # ))

    # await websocket.send(json.dumps(
    #     {'id': 5, 'type': 'get_services'}
    # ))

    # while True:
    #     message = await websocket.recv()
    #     if message is None:
    #         break
    #     print(message)
    cap_region_x_begin = 0.5  # 起点/总宽度
    cap_region_y_end = 0.8
    threshold = 60  # 二值化阈值
    blurValue = 41  # 高斯模糊参数
    bgSubThreshold = 50
    learningRate = 0
    P_stable = 60  # 距离判定基数
    P_radio = 60  # 距离判定中距离与等级的比值
    hand_idenfication_flag = 10  # 手势识别基准数
    Image_center = array([310, 350])  # 计算手势距离用的标定点参数

    # 变量
    isBgCaptured = 0  # 布尔类型, 背景是否被更新
    triggerSwitch = False  # 如果正确，手势图像将被捕获
    sampleSwitch = False  # 如果正确，提取一个作为模版的轮廓图
    matchingSwitch = False  # 如果正确，进行模版匹配

    # 自适应混合高斯背景建模，背景滤波，搜寻前景
    bgModel = cv2.createBackgroundSubtractorMOG2(50, bgSubThreshold)
    # 定义形态学滤波需要的结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    index = 1



    def printThreshold(thr):
        print("! Changed threshold to " + str(thr))

    def removeBG(frame):
        """
            背景移除函数
            先计算前景掩膜，再利用定义的kernal腐蚀原图像提取前景
        """
        fgmask = bgModel.apply(frame, learningRate=learningRate)  # 计算前景掩膜
        kernel = ones((3, 3), uint8)
        fgmask = cv2.erode(fgmask, kernel, iterations=1)  # 使用特定的结构元素来侵蚀图像。
        res = cv2.bitwise_and(frame, frame, mask=fgmask)  # 使用掩膜移除静态背景
        return res

    def camera_open():
        """
            摄像头开启函数
            开启后重新调整窗口尺寸至（640，200）
        """
        # 相机/摄像头
        camera = cv2.VideoCapture(0)
        camera.set(10, 200)  # 设置视频属性
        cv2.namedWindow('trackbar')  # 设置窗口名字
        cv2.resizeWindow("trackbar", 640, 200)  # 重新设置窗口尺寸

        # createTrackbar是Opencv中的API，其可在显示图像的窗口中快速创建一个滑动控件，用于手动调节阈值，具有非常直观的效果。
        cv2.createTrackbar('threshold', 'trackbar', threshold, 100, printThreshold)
        return camera

    def image_pre_op(image):
        """
            图像预处理函数：
            input: 待处理图像帧
            output: 经过形态学滤波后的图像帧
        """
        # 双边滤波
        frame = cv2.bilateralFilter(image, 5, 50, 100)
        img = removeBG(frame)
        return img

    def image_roi(image):
        """
            手势识别区剪裁函数：
            input: 待处理图像帧
            output: 经过剪裁后的图像帧，只包含右上角
        """
        # 剪切右上角矩形框区域
        img_temp = image[0:int(cap_region_y_end * frame.shape[0]),
                   int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]
        # 返回剪切框的大小
        img_shape = img_temp.shape
        # 将移除背景后的图像转换为灰度图
        gray = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
        # 加高斯模糊
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        # 二值化处理
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)

        # 开运算用于移除由图像噪音形成的斑点
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        # 闭运算用来连接被误分为许多小块的对象，
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        return thresh, img_shape

    def hand_select(image):
        """
            函数内容：手势框选，手势特征提取，手心计算，手指尖计算
            input：
                thresh1：预处理，裁剪后的手势区域图

            output：
                hand_label：手势识别结果
                res_max：最大手势轮廓
                drawing ：识别结果可视化图像
                cnt：识别的指尖数量
                center: 手心坐标

        """
        # 预先创建返回变量，防止未找到手势时报错
        center = array([0, 0])
        # 寻找最大轮廓，认为此轮廓是手
        _, contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
            # 求手势区域轮廓的中心矩
            moments = cv2.moments(res_max)

            # 求重心，画出重心
            center = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))

        return center

    def hand_select_accuracy(image, center, img_shape):
        """
            函数名称：手势精确框选
                    在第一次框选的基础上依据手心位置进行第二次框选，提高准确度
            input
                image: 预处理后的灰度图
                center ：第一次框选寻找到的手势中心
                img_shape: 裁剪区域
        """
        # 根据人手一般大小推断，裁剪掉多余掉手臂
        image = image[:(center[1] + 160), :]
        # 寻找最大轮廓，认为此轮廓是手
        _, contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # 每次找到的轮廓总数
        length = len(contours)

        # 预先初始化返回变量，防止未找到手势时报错
        humoments = [[0], [0], [0], [0], [-1], [-1], [-1]]
        cnt = 0
        drawing = zeros(img_shape, uint8)  # 创建画图用数组

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

            # 画出最大区域轮廓（绿色）| 把手势的外轮廓给画出来
            cv2.drawContours(drawing, [res_max], 0, (0, 255, 0), 2)
            # 画出凸包轮廓（红色）| 把手势轮廓包围起来的一个不规则凸多边形
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)
            # 求手势区域轮廓的各阶矩
            moments = cv2.moments(res_max)
            # 求手势区域轮廓的Hu矩
            humoments = cv2.HuMoments(moments)
            # 手势识别

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
                dist = (temp[0][0] - center[0]) * (temp[0][0] - center[0]) + (temp[0][1] - center[1]) * (
                        temp[0][1] - center[1])
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
                            if abs(res_max[notice][0][0] - fingerRes[j][0]) < 40:
                                flag = True
                                break
                        if flag:  # 找不到手势的时候就退出
                            continue
                        fingerRes.append(res_max[notice][0])
                        # 画出指尖
                        cv2.circle(drawing, tuple(res_max[notice][0]), 8, (255, 0, 0), -1)
                        cv2.line(drawing, center, tuple(res_max[notice][0]), (255, 0, 0), 2)
                        # 指尖计数+1
                        cnt = cnt + 1

        return humoments, drawing, cnt

    def hand_idenfication(humoments):
        """
            函数内容：手势识别
            input：
                humoments：手势的Hu矩

            output：
                hand_label：手势识别结果
        """
        # 识别手势
        hand_label, hand_specie = dnn.Predict(humoments)

        return hand_label, hand_specie

    def hand_follow(center):
        """
            函数内容：手势中心跟踪，相对距离测算
            input：
                center：手势的中心坐标

            output：
                hand_label：手势识别结果
        """
        # 输出找到的手部重心坐标
        print("----------手势中心坐标:", center)
        # 计算手势中心的移动距离，实现实时跟踪手势
        Image_center = array([310, 350])

        P_absolute0 = center - Image_center
        P_absolute = math.hypot(P_absolute0[0], P_absolute0[1])

        print("----------手势中心绝对移动距离:", P_absolute)

        return P_absolute


    def hand_contorl(P_absolute):
        """
            函数内容：
                    依据手心坐标至标定点距离，计算手心相对移动距离，作为控制信号的判定标准
        """
        P_level = 0
        if P_absolute - P_stable >= 10:
            P_level = (P_absolute - P_stable) / P_radio
            P_level = int(P_level) + 1

        return P_level

    """
        以下内容为实现switch功能使用的函数体
    
    """
    def six():
        websocket.send(open_left_light(index))
        print("six")

    def five():
        websocket.send(open_left_light(index))
        print("five")

    def two():
        print("two")
        # 跟踪手势中心的移动，并通过计算移动量判断控制信号的输出等级
        P_absolute = hand_follow(center)
        P_level = hand_contorl(P_absolute)
        print("\n", P_level)
        # print("----------调整等级：", P_level)
        # 输出手势中心center， 输出手势距标定点的距离P_absolute
        print(center)
        print(P_absolute)

    def one():
        print("one")

    def good():
        print("good")

    def punch():
        print("punch")

    def other():
        print("None")

    def hand_label_result(num, index, num_past):
        numbers = {
            0: six,
            1: five,
            4: two,
            5: one,
            6: good,
            7: punch
        }

        method = numbers.get(num, other)

        if num != num_past:
            method()

        index +=1
        return index

    """
        以上内容为实现switch功能使用的函数体
    """
    print("----------------摄像头初始化--------------")
    print("-----------------执行主程序--------------")
    # 循环标记，错开追踪手势和识别手势的时间，降低CPU功耗
    camera = camera_open()
    count = 0
    hand_label = 8
    P_level_copy = 0
    num_past = 8
    hand_specie = "None"
    while camera.isOpened():

        ret, frame = camera.read()
        # 返回滑动条上的位置的值（即实时更新阈值）
        threshold = cv2.getTrackbarPos('threshold', 'trackbar')

        frame = cv2.flip(frame, 1)  # 翻转  0:沿X轴翻转(垂直翻转)   大于0:沿Y轴翻转(水平翻转)   小于0:先沿X轴翻转，再沿Y轴翻转，等价于旋转180°

        # 画手势区域标记矩形框, 以及中心点标记处
        cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                      (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (0, 0, 255), 2)
        cv2.circle(frame, (940, 370), 4, (218, 112, 214), 3)
        cv2.circle(frame, (940, 370), 2, (218, 112, 214), -1)

        # 初始化窗口
        cv2.imshow('original', frame)

        # 主要操作

        if isBgCaptured == 1:
            """
                isBgCaptured == 1 表示开始执行手势框取
            """
            # 图像帧预处理
            img = image_pre_op(frame)

            # 提取右上角作为手势识别区
            thresh, img_shape = image_roi(img)

            cv2.imshow('binary imagine', thresh)

            if triggerSwitch:
                """
                    开始执行轮廓框选，并计算HU矩，画出重心和指尖
                """
                # print("-----------------提取Hu矩---------------：")
                # 执行手势提取，识别，手心手尖提取
                center = hand_select(thresh)
                Hu_moments, drawing, cnt = hand_select_accuracy(thresh, center, img_shape)
                # 输出完成手部区域框取和手指识别后的图
                # cv2.imshow('output', drawing)
                # 输出找到的指尖量
                # print("----------当前找到的指尖数量:", cnt)

                # 识别手势每20次循环才执行一次
                if count == hand_idenfication_flag:
                    # print("\n-----------------识别手势---------------：")
                    hand_label, hand_specie = hand_idenfication(Hu_moments)
                    print("\n", hand_label, "\n")
                    count = 0

                    # 依据不同的hand_label结果执行相应的控制指令,当指令找到重复当hand_label时不再发送信息
                    if hand_label != num_past:
                        # 手势六
                        if hand_label == 0:
                            await websocket.send(projector_out(index))
                            print("six")
                            index += 1
                        # 手势五
                        elif hand_label == 1:
                            print("five")
                            await websocket.send(open_left_light(index))
                            index += 1
                            await websocket.send(open_right_light(index))
                            index += 1
                        # 手势一
                        elif hand_label == 5:
                            print("one")
                            await websocket.send(projector_in(index))
                            index += 1
                        # 手势good
                        elif hand_label == 6:
                            print("good")
                            await websocket.send(close_left_light(index))
                            index += 1
                            await websocket.send(close_right_light(index))
                            index += 1
                        # 手势punch
                        elif hand_label == 7:
                            print("punch")
                            index += 1
                        # index = hand_label_result(hand_label, index, num_past)
                        num_past = hand_label
                    print("num_past", num_past)
                    print("index", index)

                if hand_label == 4:
                    print("two")
                    # 跟踪手势中心的移动，并通过计算移动量判断控制信号的输出等级
                    P_absolute = hand_follow(center)
                    P_level = hand_contorl(P_absolute)
                    print("\n", P_level)
                    # print("----------调整等级：", P_level)
                    # 输出手势中心center， 输出手势距标定点的距离P_absolute
                    print(center)
                    print(P_absolute)
                    ''''
                    if P_level != P_level_copy:
                        if P_level == 0:
                            await websocket.send(close_curtain_white(index))
                        elif P_level == 1:
                            await websocket.send(open_curtain_white_position(index, 30))
                        elif P_level == 2:
                            await websocket.send(open_curtain_white_position(index, 60))
                        else:
                            await websocket.send(open_curtain_white(index))
                     
                        index += 1
            
                        P_level_copy = P_level
                        '''
                    # time.sleep(5)
                """"# 找寻到目标手势后开始持续跟踪,目前设定为手势two
                if hand_label == 5:
                    # await  websocket.send(open_curtain_black_position(2, 0))
                    # await  websocket.send(projector_in(index))
                    index+=1
                """
                # 创建一个矩形，在图片上写文字，参数依次定义了文字类型，高，宽，字体厚度
                font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体
                # 输出完成手部区域框取和手指识别后的图
                drawing = cv2.putText(drawing, hand_specie, (0, 40), font, 1.2, (255, 255, 255), 2)
                cv2.imshow('output', drawing)
                # 计数标志位递增
                count += 1
                # 手控运行开关，测试时需关闭注释
                # triggerSwitch = False

        # 输入的键盘值
        k = cv2.waitKey(10)
        if k == 27:  # 按下ESC退出，并为文件更新一行
            print("========Thank you for using my Program!=======")
            break
        # 按下'b'会开始执行手势区域划分等具体工作
        elif k == ord('b'):
            isBgCaptured = 1
            print('======Background Captured======')
        # 按下'r'会更新背景，适用于镜头对焦出问题的情况
        elif k == ord('r'):
            bgModel = cv2.createBackgroundSubtractorMOG2(50, bgSubThreshold)
            triggerSwitch = False
            # isBgCaptured = 0
            print('=======Reset BackGround=====')
        # 按下'c'会进行手势特征提取
        elif k == ord('c'):
            triggerSwitch = True
            print('======Catch hand======')
    # 施放内存，结束程序
    camera.release()
    cv2.destroyAllWindows()

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
loop.close()
