import cv2
import time
import os
import numpy as np


def ktof(val):
    return round(((1.8 * ktoc(val) + 32.0)), 2)

def ktoc(val):
    return round(((val - 27315) / 100.0), 2)

def raw_to_8bit(data):
    cv2.normalize(data, data, 0, 65535, cv2.NORM_MINMAX)
    np.right_shift(data, 8, data)
    return cv2.cvtColor(np.uint8(data), cv2.COLOR_GRAY2RGB)

time_data = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
os.mkdir("img/"+str(time_data))
# time.sleep(2)
# os.mkdir("img/TH"+str(time_data))
time.sleep(2)

# cap_TH  = cv2.VideoCapture(2)
cap_th  = cv2.VideoCapture(2)
cap_RGB = cv2.VideoCapture(0)

cap_th.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"Y16 "))
cap_th.set(cv2.CAP_PROP_CONVERT_RGB, 0)

time.sleep(2)
# 使用 XVID 編碼
fourcc_RGB  = cv2.VideoWriter_fourcc(*'XVID')
fourcc_TH = cv2.VideoWriter_fourcc(*'XVID')

out_RGB = cv2.VideoWriter("img/"+str(time_data)+'.avi', fourcc_RGB, 23, (640, 480))
out_TH = cv2.VideoWriter("img/"+str(time_data)+'_th.avi', fourcc_TH, 23, (640, 480))
i = 0
j = 0

while(cap_RGB.isOpened()):

    ret_RGB, frame_RGB = cap_RGB.read()
    ret, frame = cap_th.read()

    # ret_TH, frame_TH = cap_TH.read()

    if ret_RGB and ret:
        frame = cv2.resize(frame[:-3,:], (640, 480))
        tempImage = frame.copy()
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(tempImage[0:480,0:640])
        maxLoc = (maxLoc[0], maxLoc[1]) #x1 y1
        # cv2.circle(frame,(maxLoc[0], maxLoc[1]), 5, (0,255,255), -1)

        # frame_RGB = cv2.flip(frame_RGB, 0)
        # frame = cv2.flip(frame, 0)

        # frame_RGB[0,0,0],frame_RGB[0,0,1]= str(ktoc(maxVal)).split('.') #溫度寫在000
        # frame_RGB[0,0,2] = 0
        # print(frame_RGB[0,0])
        # print(ktoc(maxVal))
        # out_RGB.write(frame_RGB)
        frame = raw_to_8bit(frame)
        print(frame.shape)
        print(frame[0,0])
        # out_TH.write(frame)

        cv2.imshow('frame',frame_RGB)
        cv2.imshow('frame2 ',frame)

        
        if  j%(100)==0:
            # cv2.imwrite('img/'+str(time_data)+'/Image'+str(i)+".jpg", frame_RGB)
            # cv2.imwrite('img/TH'+str(time_data)+'/Image'+str(i)+".jpg", frame_TH)
            i+=1
            j=0

        j+=1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        # 釋放所有資源
        cap_RGB.release()
        out_RGB.release()
        break
cv2.destroyAllWindows()
