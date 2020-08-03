import argparse

import torch.backends.cudnn as cudnn

from models.experimental import *
from utils.datasets import *
from utils.utils import *






import cv2
import numpy as np
import os,sys
import time
from datetime import datetime
import imutils
from configparser import ConfigParser
#import face_recognition
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtWidgets import QLineEdit,QFormLayout,QDesktopWidget
from PyQt5.QtCore import QDate, QTime, QDateTime, Qt, QTimer
from PyQt5.QtGui import QPixmap,QColor, QPen,QPalette


  
port=0

logo_img = cv2.imread("NHRI_Logo.jpg", -1)
logo_img = cv2.resize( logo_img, (256,256))
facelist = [logo_img]*5


'''
def rgb_face(image):
    global facelist,templist,distance
    X_face_locations = face_recognition.face_locations(image , model= 'hog')
    
    faces = []
    
    for y1,x2,y2,x1 in X_face_locations:
        
        
        
        
        faces.append((x1,y1,x2,y2))
        
        
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0,255), 2)
        
            
        
    return image,faces

def get_face(rgb_im,faces):
    global facelist
    if faces == []:
        return rgb_im
    for face in faces:
        
        (x1,y1,x2,y2) = face
        
        faceimg = cv2.resize( rgb_im[y1:y2,x1:x2], (256,256))
                                
        facelist.append(faceimg)
        facelist = facelist[-5:]
        

'''


class RGBVideo(QtCore.QObject):
    
    image_rgb = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, camera_port=port, parent=None):
        super().__init__(parent)
        # self.camera = cv2.VideoCapture(camera_port)
        # self.width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        # self.height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # print (self.width,self.height)
        self.timer = QtCore.QBasicTimer()

    def start_recording(self):
        self.timer.start(0, self)

    def timerEvent(self, event):
        if (event.timerId() != self.timer.timerId()):
            return

        # read, data = self.camera.read()
        if read:
            # detect()
            self.image_rgb.emit(detect())

class RecordMessage(QtCore.QObject):
    
    
    image_message = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        

        self.timer = QtCore.QBasicTimer()

    def start_recording(self):
        self.timer.start(0, self)

    def timerEvent(self, event):
        global facelist
        
        
        
        faces = np.hstack((facelist[4],facelist[3],facelist[2],facelist[1],facelist[0]))
        
        if (event.timerId() != self.timer.timerId()):
            return

        
        self.image_message.emit(faces)

class FaceDetectionWidget(QtWidgets.QWidget):
    def __init__(self,  parent=None):
        super().__init__(parent)
       
        self.image = QtGui.QImage()
        self._red = (0, 0, 255)
        self._width = 2
        self._min_size = (30, 30)

    

    def image_rgb_slot(self, image_rgb):
        
        
        
        image_rgb=cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
        
        image_temp = image_rgb.copy()
        
        #image_rgb,faces = rgb_face(image_rgb)
        
        
        
        #get_face(image_temp,faces)
        
        
        image_rgb=cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
       
        
        
        
        self.image = self.get_qimage(image_rgb)
        if self.image.size() != self.size():
            self.setFixedSize(self.image.size())

        self.update()
        
    

    def get_qimage(self, image: np.ndarray):
        height, width, colors = image.shape
        bytesPerLine = 3 * width
        QImage = QtGui.QImage

        image = QImage(image.data,
                       width,
                       height,
                       bytesPerLine,
                       QImage.Format_RGB888)

        image = image.rgbSwapped()
        return image

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()
        


class MessageWidget(QtWidgets.QWidget):
    def __init__(self,  parent=None):
        super().__init__(parent)
        
        self.image = QtGui.QImage()
        self._red = (0, 0, 255)
        self._width = 2
        self._min_size = (30, 30)

    

    def message_data_slot(self, image_message):

        
        
        self.image = self.get_qimage(image_message)
        if self.image.size() != self.size():
            self.setFixedSize(self.image.size())

        self.update()

    def get_qimage(self, image: np.ndarray):
        height, width, colors = image.shape
        bytesPerLine = 3 * width
        QImage = QtGui.QImage

        image = QImage(image.data,
                       width,
                       height,
                       bytesPerLine,
                       QImage.Format_RGB888)

        image = image.rgbSwapped()
        return image

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()


class MainWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
    
        self.initUI()
    def initUI(self):
        self.img = np.ndarray(())
        
        
        self.timer_id = 0
        self.label = QtWidgets.QLabel("")
        
        self.label_start = QtWidgets.QLabel("Start")
        
        
        self.label.setFont(QtGui.QFont("Roman times",30,QtGui.QFont.Bold))
        
        
        self.label.setAlignment(Qt.AlignCenter)
        
        
        
        
        
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), Qt.white)
        self.setPalette(p)
        
        
        
        self.face_detection_widget = FaceDetectionWidget()
        
        self.message_widget = MessageWidget()
        
        
        
        
        # TODO: set video port
        
        
        self.rgb_video = RGBVideo()
        
        self.record_message = RecordMessage()

        
        
        image_rgb_slot = self.face_detection_widget.image_rgb_slot
        
        image_message_slot = self.message_widget.message_data_slot
        
        
        self.rgb_video.image_rgb.connect(image_rgb_slot)
        self.record_message.image_message.connect(image_message_slot)
        
        
        self.run_button = QtWidgets.QPushButton("Start",self)
        

        mainlayout = QtWidgets.QHBoxLayout()
        layout2 = QtWidgets.QVBoxLayout()
        messagelayout = QtWidgets.QVBoxLayout()
        layout = QtWidgets.QHBoxLayout()
        layout1 = QtWidgets.QHBoxLayout()
        layout3 = QtWidgets.QVBoxLayout()
        
        
        

        layout.addWidget(self.face_detection_widget)
        
        
        
        
        layout2.addWidget(self.label,0,Qt.AlignTop)
        
        
        layout2.addWidget(self.run_button)
        
        
        
        layout1.addLayout(layout)
        
        
        layout3.addLayout(layout1)
        layout3.addWidget(self.message_widget)
        mainlayout.addLayout(layout3)
        mainlayout.addLayout(layout2)
        
        
        
        #self.thermal_video.start_recording()
        #self.rgb_video.start_recording()
        #self.record_message.start_recording()
        #self.timer()
        #self.openSlot()
        #self.openSlot1()
        
        self.run_button.clicked.connect(self.rgb_video.start_recording)
        self.run_button.clicked.connect(self.record_message.start_recording)
        self.run_button.clicked.connect(self.timer)
        
        
        self.setLayout(mainlayout)
        
    
        
        
    def timer(self):
        self.timer_id = self.startTimer(1000, timerType = QtCore.Qt.VeryCoarseTimer)

    
    def timerEvent(self, event):
        self.label.setText(time.strftime(" %Y年%m月%d日 \n\n %H:%M:%S"))
        
        
    



























def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = torch_utils.time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        # label = '%s %.2f' % (names[int(cls)], conf)
                        label = '%s' % (str.capitalize(names[int(cls)])) #字首大寫 去準確率
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                # cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration
                return im0

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))
    return im0

if __name__ == '__main__':
    #python detect.py --source 2020-07-24_20_28_25.avi --weights runs/exp17/weights/best.pt --conf 0.65

    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--weights', nargs='+', type=str, default='runs/exp17/weights/best.pt', help='model.pt path(s)')

    # parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='2020-07-24_20_28_25.avi', help='source')  # file/folder, 0 for webcam

    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    # with torch.no_grad():
    #     if opt.update:  # update all models (to fix SourceChangeWarning)
    #         for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov3-spp.pt']:
    #             detect()
    #             create_pretrained(opt.weights, opt.weights)
    #     else:
    #         detect()
    app = QtWidgets.QApplication(sys.argv)
    screen = QDesktopWidget().screenGeometry()
    main_window = QtWidgets.QMainWindow()
    main_widget = MainWidget()
    main_window.setCentralWidget(main_widget)
    main_window.setGeometry(0, 0, 400, 400)
    #main_window.setWindowIcon(QtGui.QIcon("Logo.jpg"))
    main_window.setWindowTitle('UI Test')
    
    #main_window.resize(screen.width(),screen.height())
    
    #main_window.showFullScreen()
    
    main_window.show()
    sys.exit(app.exec_())
