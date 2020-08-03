import cv2
img = cv2.imread('Image2.jpg')
cv2.imshow('frame',img)
img[0][0][0]=87
img3=[0,1,2,3]
img3[0:4]
print(img.shape)
print(img3[0:4])
cv2.waitKey(0)
cv2.destroyAllWindows()



# import random
# import time
# import cv2
# time_data = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
# def ktof(val):
#     return round(((1.8 * ktoc(val) + 32.0)), 2)

# def ktoc(val):
#     return round(((val - 27315) / 100.0), 2)


# capture = cv2.VideoCapture("save/2020-07-24_20_28_25.avi")
# cap_th = cv2.VideoCapture(2)
# cap_th.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"Y16 "))
# cap_th.set(cv2.CAP_PROP_CONVERT_RGB, 0)

# fourcc_RGB  = cv2.VideoWriter_fourcc(*'XVID')
# out_RGB = cv2.VideoWriter("img/"+str(time_data)+'.avi', fourcc_RGB, 23, (640, 480))


# if capture.isOpened() and cap_th.isOpened():
#     while True:
#         ret, prev = capture.read()
#         ret2, frame = cap_th.read()

#         frame = cv2.resize(frame[:-3,:], (640, 480))
#         frame = cv2.flip(frame, 0)
#         tempImage = frame.copy()
#         minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(tempImage[0:480,0:640])
#         print(ktoc(maxVal))

#         prev[0,0,0],prev[0,0,1]= str(ktoc(maxVal)).split('.') #溫度寫在000
#         prev[0,0,2] = 0
#         if ret and ret2 :
#             cv2.imshow('video', prev)
#             cv2.imshow('video2', frame)
#             out_RGB.write(prev)

#         else:
#             break
#         if cv2.waitKey(43) & 0xFF == ord('q'):
#             out_RGB.release()
#             capture.release()
#             cap_th.release()

#             break
# cv2.destroyAllWindows()



# import cv2


# capture = cv2.VideoCapture("save/2020-07-24_20_28_25.avi")




# if capture.isOpened() :
#     while True:
#         ret, prev = capture.read()
       
#         if ret  : 
#             print(prev.shape)
#             print(prev[479][639])
#             cv2.imshow('video', prev)
            
#         else:
#             break
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             capture.release()
#             break
# cv2.destroyAllWindows()
