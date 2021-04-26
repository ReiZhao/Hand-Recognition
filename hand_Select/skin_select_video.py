import numpy as np
import cv2
cap = cv2.VideoCapture(0)

while(1):
    ret, img = cap.read()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 把图像转换到HSV色域
    (_h, _s, _v) = cv2.split(hsv)  # 图像分割, 分别获取h, s, v 通道分量图像
    lower_skin = np.array([10, 28, 50])
    upper_skin = np.array([20, 255, 255])
    skin3 = cv2.inRange(hsv, lower_skin, upper_skin)

    cv2.imshow("img", img)
    cv2.imshow(" Skin3 HSV", skin3)

    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)  # 把图像转换到YUV色域
    # (y, cr, cb) = cv2.split(ycrcb)  # 图像分割, 分别获取y, cr, br通道分量图像
    lower_skin = np.array([150, 133, 67])
    upper_skin = np.array([255, 193, 127])
    skin2 = cv2.inRange(ycrcb, lower_skin, upper_skin)
    # cv2.imshow(imname, img)

    cv2.imshow(" Skin2 Cr+Cb", skin2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()