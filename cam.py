import cv2
import time
import os

time_data = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
os.mkdir("img/"+str(time_data))
# time.sleep(2)
# os.mkdir("img/TH"+str(time_data))
time.sleep(2)

# cap_TH  = cv2.VideoCapture(2)
cap_RGB = cv2.VideoCapture(0)
time.sleep(2)
# 使用 XVID 編碼
fourcc_RGB  = cv2.VideoWriter_fourcc(*'XVID')
# fourcc_TH = cv2.VideoWriter_fourcc(*'XVID')

out_RGB = cv2.VideoWriter("img/"+str(time_data)+'.avi', fourcc_RGB, 23, (640, 480))
# out_TH = cv2.VideoWriter('output_TH_1.avi', fourcc_TH, 23, (640, 480))
i = 0
j = 0

while(cap_RGB.isOpened()):

    ret_RGB, frame_RGB = cap_RGB.read()
    # ret_TH, frame_TH = cap_TH.read()

    if ret_RGB:
        frame_RGB = cv2.flip(frame_RGB, 0)
        # frame_TH = cv2.flip(frame_TH, 0)

        out_RGB.write(frame_RGB)
        # out_TH.write(frame_TH)

        cv2.imshow('frame',frame_RGB)
        # cv2.imshow('frame',frame_TH)

        
        if  j%(100)==0:
            cv2.imwrite('img/'+str(time_data)+'/Image'+str(i)+".jpg", frame_RGB)
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
