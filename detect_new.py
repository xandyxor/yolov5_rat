import argparse

import torch.backends.cudnn as cudnn

from models.experimental import *
from utils.datasets import *
from utils.utils import *

#=======================================================================================================
#=======================================================================================================
import numpy as np
import cv2

def fix_xy(x,y):
    # 修正廣角RGB相機 與紅外線攝影機之鏡頭視角差異 公式用於RGB x1y1 = (180,50) x2y2 = (580,350)對應紅外線攝影機x1y1=(0,0) x2y2=(640,480)
    x2 = 1.6*int(x)-224
    y2 = 1.6*int(y)-80
    if x2<0:
        x2 = 0
    if x2>640:
        x2= 640
    if y2<0:
        y2 = 0
    if y2>480:
        y2=480
    return x2,y2

def fix_xyxy(x1,y1,x2,y2):
    x1,y1 = fix_xy(x1,y1)
    x2,y2 = fix_xy(x2,y2)
    return int(x1),int(y1),int(x2),int(y2)

en_mode = 0
colorMapType = 0

def ktof(val):
    return round(((1.8 * ktoc(val) + 32.0)), 2)

def ktoc(val):
    return round(((val - 27315) / 100.0), 2)

def display_temperatureC(img, val_k, loc, color):
    val = ktoc(val_k)
    x, y = loc
    if  en_mode == 1:
        cv2.putText(img,"{0:.1f} degC".format(val), loc, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.line(img, (x - 20, y), (x + 20, y), color, 3)
        cv2.line(img, (x, y - 20), (x, y + 20), color, 3)
    
    return  val   

def raw_to_8bit(data):
    cv2.normalize(data, data, 0, 65535, cv2.NORM_MINMAX)
    np.right_shift(data, 8, data)
    return cv2.cvtColor(np.uint8(data), cv2.COLOR_GRAY2RGB)

gcm = np.array([[[255, 255, 255]],
    [[253, 253, 253]],
    [[251, 251, 251]],
    [[249, 249, 249]],
    [[247, 247, 247]],
    [[245, 245, 245]],
    [[243, 243, 243]],
    [[241, 241, 241]],
    [[239, 239, 239]],
    [[237, 237, 237]],
    [[235, 235, 235]],
    [[233, 233, 233]],
    [[231, 231, 231]],
    [[229, 229, 229]],
    [[227, 227, 227]],
    [[225, 225, 225]],
    [[223, 223, 223]],
    [[221, 221, 221]],
    [[219, 219, 219]],
    [[217, 217, 217]],
    [[215, 215, 215]],
    [[213, 213, 213]],
    [[211, 211, 211]],
    [[209, 209, 209]],
    [[207, 207, 207]],
    [[205, 205, 205]],
    [[203, 203, 203]],
    [[201, 201, 201]],
    [[199, 199, 199]],
    [[197, 197, 197]],
    [[195, 195, 195]],
    [[193, 193, 193]],
    [[191, 191, 191]],
    [[189, 189, 189]],
    [[187, 187, 187]],
    [[185, 185, 185]],
    [[183, 183, 183]],
    [[181, 181, 181]],
    [[179, 179, 179]],
    [[177, 177, 177]],
    [[175, 175, 175]],
    [[173, 173, 173]],
    [[171, 171, 171]],
    [[169, 169, 169]],
    [[167, 167, 167]],
    [[165, 165, 165]],
    [[163, 163, 163]],
    [[161, 161, 161]],
    [[159, 159, 159]],
    [[157, 157, 157]],
    [[155, 155, 155]],
    [[153, 153, 153]],
    [[151, 151, 151]],
    [[149, 149, 149]],
    [[147, 147, 147]],
    [[145, 145, 145]],
    [[143, 143, 143]],
    [[141, 141, 141]],
    [[139, 139, 139]],
    [[137, 137, 137]],
    [[135, 135, 135]],
    [[133, 133, 133]],
    [[131, 131, 131]],
    [[129, 129, 129]],
    [[126, 126, 126]],
    [[124, 124, 124]],
    [[122, 122, 122]],
    [[120, 120, 120]],
    [[118, 118, 118]],
    [[116, 116, 116]],
    [[114, 114, 114]],
    [[112, 112, 112]],
    [[110, 110, 110]],
    [[108, 108, 108]],
    [[106, 106, 106]],
    [[104, 104, 104]],
    [[102, 102, 102]],
    [[100, 100, 100]],
    [[ 98,  98,  98]],
    [[ 96,  96,  96]],
    [[ 94,  94,  94]],
    [[ 92,  92,  92]],
    [[ 90,  90,  90]],
    [[ 88,  88,  88]],
    [[ 86,  86,  86]],
    [[ 84,  84,  84]],
    [[ 82,  82,  82]],
    [[ 80,  80,  80]],
    [[ 78,  78,  78]],
    [[ 76,  76,  76]],
    [[ 74,  74,  74]],
    [[ 72,  72,  72]],
    [[ 70,  70,  70]],
    [[ 68,  68,  68]],
    [[ 66,  66,  66]],
    [[ 64,  64,  64]],
    [[ 62,  62,  62]],
    [[ 60,  60,  60]],
    [[ 58,  58,  58]],
    [[ 56,  56,  56]],
    [[ 54,  54,  54]],
    [[ 52,  52,  52]],
    [[ 50,  50,  50]],
    [[ 48,  48,  48]],
    [[ 46,  46,  46]],
    [[ 44,  44,  44]],
    [[ 42,  42,  42]],
    [[ 40,  40,  40]],
    [[ 38,  38,  38]],
    [[ 36,  36,  36]],
    [[ 34,  34,  34]],
    [[ 32,  32,  32]],
    [[ 30,  30,  30]],
    [[ 28,  28,  28]],
    [[ 26,  26,  26]],
    [[ 24,  24,  24]],
    [[ 22,  22,  22]],
    [[ 20,  20,  20]],
    [[ 18,  18,  18]],
    [[ 16,  16,  16]],
    [[ 14,  14,  14]],
    [[ 12,  12,  12]],
    [[ 10,  10,  10]],
    [[  8,   8,   8]],
    [[  6,   6,   6]],
    [[  4,   4,   4]],
    [[  2,   2,   2]],
    [[  0,   0,   0]],
    [[  9,   0,   0]],
    [[ 16,   0,   2]],
    [[ 24,   0,   4]],
    [[ 31,   0,   6]],
    [[ 38,   0,   8]],
    [[ 45,   0,  10]],
    [[ 53,   0,  12]],
    [[ 60,   0,  14]],
    [[ 67,   0,  17]],
    [[ 74,   0,  19]],
    [[ 82,   0,  21]],
    [[ 89,   0,  23]],
    [[ 96,   0,  25]],
    [[103,   0,  27]],
    [[111,   0,  29]],
    [[118,   0,  31]],
    [[120,   0,  36]],
    [[121,   0,  41]],
    [[122,   0,  46]],
    [[123,   0,  51]],
    [[124,   0,  56]],
    [[125,   0,  61]],
    [[126,   0,  66]],
    [[127,   0,  71]],
    [[128,   1,  76]],
    [[129,   1,  81]],
    [[130,   1,  86]],
    [[131,   1,  91]],
    [[132,   1,  96]],
    [[133,   1, 101]],
    [[134,   1, 106]],
    [[135,   1, 111]],
    [[136,   1, 116]],
    [[136,   1, 121]],
    [[137,   2, 125]],
    [[137,   2, 130]],
    [[137,   3, 135]],
    [[138,   3, 139]],
    [[138,   3, 144]],
    [[138,   4, 149]],
    [[139,   4, 153]],
    [[139,   5, 158]],
    [[139,   5, 163]],
    [[140,   5, 167]],
    [[140,   6, 172]],
    [[140,   6, 177]],
    [[141,   7, 181]],
    [[141,   7, 186]],
    [[137,  10, 189]],
    [[132,  13, 191]],
    [[127,  16, 194]],
    [[121,  19, 196]],
    [[116,  22, 198]],
    [[111,  25, 200]],
    [[106,  28, 203]],
    [[101,  31, 205]],
    [[ 95,  34, 207]],
    [[ 90,  37, 209]],
    [[ 85,  40, 212]],
    [[ 80,  43, 214]],
    [[ 75,  46, 216]],
    [[ 69,  49, 218]],
    [[ 64,  52, 221]],
    [[ 59,  55, 223]],
    [[ 49,  57, 224]],
    [[ 47,  60, 225]],
    [[ 44,  64, 226]],
    [[ 42,  67, 227]],
    [[ 39,  71, 228]],
    [[ 37,  74, 229]],
    [[ 34,  78, 230]],
    [[ 32,  81, 231]],
    [[ 29,  85, 231]],
    [[ 27,  88, 232]],
    [[ 24,  92, 233]],
    [[ 22,  95, 234]],
    [[ 19,  99, 235]],
    [[ 17, 102, 236]],
    [[ 14, 106, 237]],
    [[ 12, 109, 238]],
    [[ 12, 112, 239]],
    [[ 12, 116, 240]],
    [[ 12, 119, 240]],
    [[ 12, 123, 241]],
    [[ 12, 127, 241]],
    [[ 12, 130, 242]],
    [[ 12, 134, 242]],
    [[ 12, 138, 243]],
    [[ 13, 141, 243]],
    [[ 13, 145, 244]],
    [[ 13, 149, 244]],
    [[ 13, 152, 245]],
    [[ 13, 156, 245]],
    [[ 13, 160, 246]],
    [[ 13, 163, 246]],
    [[ 13, 167, 247]],
    [[ 13, 171, 247]],
    [[ 14, 175, 248]],
    [[ 15, 178, 248]],
    [[ 16, 182, 249]],
    [[ 18, 185, 249]],
    [[ 19, 189, 250]],
    [[ 20, 192, 250]],
    [[ 21, 196, 251]],
    [[ 22, 199, 251]],
    [[ 23, 203, 252]],
    [[ 24, 206, 252]],
    [[ 25, 210, 253]],
    [[ 27, 213, 253]],
    [[ 28, 217, 254]],
    [[ 29, 220, 254]],
    [[ 30, 224, 255]],
    [[ 39, 227, 255]],
    [[ 53, 229, 255]],
    [[ 67, 231, 255]],
    [[ 81, 233, 255]],
    [[ 95, 234, 255]],
    [[109, 236, 255]],
    [[123, 238, 255]],
    [[137, 240, 255]],
    [[151, 242, 255]],
    [[165, 244, 255]],
    [[179, 246, 255]],
    [[193, 248, 255]],
    [[207, 249, 255]],
    [[221, 251, 255]],
    [[235, 253, 255]],
    [[ 24, 255, 255]]], dtype=("uint8"))

#=======================================================================================================
#=======================================================================================================

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




#=======================================================================================================
#=======================================================================================================

    # 建立 VideoCapture 物件
    cap_th  = cv2.VideoCapture(2)
    # cap.set(cv2.CAP_PROP_FPS, 60)
    cap_th.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"Y16 "))
    cap_th.set(cv2.CAP_PROP_CONVERT_RGB, 0)

#=======================================================================================================
#=======================================================================================================

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
        deg_tmp = 0
        deg_i = 0

        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
#=======================================================================================================
#=======================================================================================================
            ret, frame = cap_th.read()
            # cv2.rectangle(frame_RGB, (180,50), (580,350), (0,255,0), 1)
            
            if ret:

                frame = cv2.resize(frame[:-3,:], (640, 480))
                frame = cv2.flip(frame, 0)

                tempImage = frame.copy()
                frame = cv2.LUT(raw_to_8bit(frame), gcm)

                
#=======================================================================================================
#=======================================================================================================
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
                        label = '%s' % (str.capitalize(names[int(cls)]))


                        #========================================================
                        # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        if ret:
                            # imgt = tempImage[int(xyxy[0])+60:int(xyxy[2])+90,int(xyxy[1])+60:int(xyxy[3])+90]
                            # print(int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3]))
                            xyxy_temp = fix_xyxy(int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3]))
                            # print(xyxy_temp)
                            imgt = tempImage[int(xyxy_temp[1]):int(xyxy_temp[3]),int(xyxy_temp[0]):int(xyxy_temp[2])]
                            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(imgt)
                            # maxLoc = (maxLoc[0] +int(xyxy[0]), maxLoc[1] + int(xyxy[1]))
                            maxLoc = (maxLoc[0] + int(xyxy_temp[0]), maxLoc[1] + int(xyxy_temp[1])) #x1 y1
                            deg = display_temperatureC(imgt, maxVal, maxLoc, (0, 255, 0)) #displays max temp at max temp location on image
                            if deg_i >=15 or deg_i%15 ==0:
                                deg_i = 0
                                deg_tmp = round(deg,1)
                            deg_i = deg_i +1     
                            plot_one_box(xyxy, im0, label=label+"  "+str(deg_tmp), color=colors[int(cls)], line_thickness=3)


                            # cv2.rectangle(frame, (int(xyxy[0])+60,int(xyxy[1])+90), (int(xyxy[2])+60,int(xyxy[3])+90), (0,255,0), 1)
                            cv2.rectangle(frame, (int(xyxy_temp[0]),int(xyxy_temp[1])), (int(xyxy_temp[2]),int(xyxy_temp[3])), (0,255,0), 1)
                            text = str(deg_tmp)
                            cv2.putText(frame, text, (int(xyxy_temp[0]), int(xyxy_temp[1])), cv2.FONT_HERSHEY_PLAIN,1, (0, 255, 0), 1, cv2.LINE_AA)
                        else:
                        #========================================================
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)


            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))
            view_img = True
            # save_img = True
            # Stream results
            if view_img:
                #xyxy=[140,50,540,350]
                #cv2.rectangle(im0, (xyxy[0],xyxy[1]), (xyxy[2],xyxy[3]), (0,255,0), 1)

                cv2.imshow(p, im0)
                cv2.imshow("src",frame)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
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

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov3-spp.pt']:
                detect()
                create_pretrained(opt.weights, opt.weights)
        else:
            detect()
