# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import csv
import os
import platform
import sys
from pathlib import Path
#from prettytable import PrettyTable

import torch
import csv
import math
import mathfunction
import numpy as np
from numpy import dot
from numpy.linalg import norm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_csv=False,  # save results in CSV format
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    #save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir = Path(Path(project) / name)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / 'predictions.csv'

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            data = {'Image Name': image_name, 'Prediction': prediction, 'Confidence': confidence}
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            '''
            print('im:{0}'.format(im.shape[2:]))
            print('det:{0}'.format(det))
            print('im0:{0}'.format(im0))
            print('im0[0].shape:{0} im0[0]:{1}'.format(im0[0].shape,im0[0]))
            print('im0[1].shape:{0} im0[1]:{1}'.format(im0[1].shape,im0[1]))
            print('im0[2].shape:{0} im0[2]:{1}'.format(im0[2].shape,im0[2]))'''

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                '''
                t = PrettyTable(['Class name', 'Number'])
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    t.add_row([names[int(c)],n.item()])
                t.add_row(['Total',len(det)])
                #print(t)'''

                #print(im0)
                #print(im0.shape)

                #0거실 1주방 2서재(방) 3침실 4화장실
                room = ['거실', '주방', '서재', '침실', '화장실']
                #'0소파','1티비','2화분','3식탁','4의자','5음식','6식기','7주방기구','8침대','9전자제품','10책','11시계','12싱크','13변기','14화장실도구','15기타도구'
                #testname = ['소파','티비','화분','식탁','의자','음식','식기','주방기구','침대','전자제품','책','시계','싱크','변기','화장실도구','기타도구']
                testname = [i for i in range(80)]
                test = [0 for _ in range(80)]
                
                for x1, y1, x2, y2, a, b in det[:, 0:6]: # x=가로 y=세로 a=정확도 b=클래스네임
                    
                    bbw = abs(int(x2) - int(x1))
                    bbh = abs(int(y2) - int(y1))
                    imgw = int(gn[2])
                    imgh = int(gn[3])
                    imgper = ((bbw*bbh)/(imgw*imgh))*100
                    #print('사물:{0} 정확도:{1} 가로:{2} 세로:{3} gn:{4} per:{5}%'.format(names[int(b)],float(a),bbw,bbh,gn,imgper))
                    imgperscore = float(a) * 10

                    test[int(b)] += imgperscore

                    '''
                    #57소파(Couch)
                    if(int(b) in [57]):
                      test[0] += imgperscore
                    #62티비
                    elif(int(b) in [62]):
                      test[1] += imgperscore
                    #58화분 75꽃병
                    elif(int(b) in [58, 75]):
                      test[2] += imgperscore
                    #60책상(식탁)
                    elif(int(b) in [60]):
                      test[3] += imgperscore
                    #56의자
                    elif(int(b) in [56]):
                      test[4] += imgperscore
                    #음식
                    elif(int(b) in [46, 47, 48, 49, 50, 51, 52, 53, 54, 55]):
                      test[5] += imgperscore
                    #40와인잔 41컵 42포크 43칼 44스푼 45그릇
                    elif(int(b) in [40, 41, 42, 43, 44, 45]):
                      test[6] += imgperscore
                    #68렌지 69오븐 70토스터기 72냉장고(주방기구)
                    elif(int(b) in [68, 69, 70, 72]):
                      test[7] += imgperscore
                    #59침대
                    elif(int(b) in [59]):
                      test[8] += imgperscore
                    #63노트북, 64마우스, 66키보드, 67휴대폰
                    elif(int(b) in [63, 64, 66, 67]):
                      test[9] += imgperscore
                    #73책
                    elif(int(b) in [73]):
                      test[10] += imgperscore
                    #74시계
                    elif(int(b) in [74]):
                        test[11] += imgperscore
                    #71싱크
                    elif(int(b) in [71]):
                      test[12] += imgperscore
                    #61변기
                    elif(int(b) in [61]):
                      test[13] += imgperscore
                    #78헤어드라이어 79칫솔
                    elif(int(b) in [78, 79]):
                      test[14] += imgperscore
                    #65리모콘 76가위 77테디베어
                    elif(int(b) in [65, 76, 77]):
                      test[15] += imgperscore
                    '''

                #print('test:{0}'.format(test))
                #test = mathfunction.softmax(np.array(test))
                #print('test:{0}'.format(test))
                #print('testsum:{0}'.format(sum(test)))

                '''
                #각 방의 분포 데이터
                room_data = [
                [8.957712,0.541918,2.123330,1.144542,2.620160,0.001783,0.055224,0.057142,0.390340,0.037010,0.108476,0.009977,0.000000,0.008077,0.000000,0.016347]
                ,[0.046614,0.040107,0.818372,1.489861,2.040686,0.016224,0.430649,2.120751,0.002880,0.006128,0.008137,0.012671,0.323565,0.011867,0.001615,0.008837]
                ,[1.024636,0.481016,0.899136,1.559899,4.315187,0.005602,0.036886,0.047682,0.068522,0.336604,0.606592,0.035517,0.030793,0.000000,0.000000,0.034474]
                ,[0.831382,0.098943,1.333623,0.108237,1.036016,0.000000,0.008508,0.027868,16.342582,0.033946,0.049429,0.018795,0.006646,0.012154,0.000000,0.027260]
                ,[0.014012,0.026913,0.597798,0.023581,0.080071,0.003199,0.070363,0.065485,0.001078,0.007850,0.001644,0.007820,2.450248,4.887437,0.000732,0.000000]
                ]

                #유클리드 거리로 유사도 계산
                similarity_score = [0,0,0,0,0]
                for i in range(5):
                  temp = 0
                  for ii in range(16):
                    temp += pow((test[ii]-room_data[i][ii]), 2)
                  similarity_score[i] = math.sqrt(temp)
                
                #유클리드 거리 리스트(작을수록 유사도 높음)
                #print(similarity_score)
                t2 = PrettyTable(['Room name', 'Euclidean distance'])
                for c in range(5):
                    t2.add_row([room[c],similarity_score[c]])
                #print(t2)

                room_index = similarity_score.index(min(similarity_score))
                this_room = room[room_index]
                print('이 방은 "%s"입니다.' % (this_room))
                '''

                '''import dataAll
                import numpy as np
                from numpy import dot
                from numpy.linalg import norm

                def cos_sim(A, B):
                  return dot(A, B)/(norm(A)*norm(B))

                similarity_score = [0 for _ in range(1500)]

                count = 0
                for ci in dataAll.data_all:
                  similarity_score[count] = cos_sim(ci, test)
                  count += 1
                resultData = similarity_score.index(max(similarity_score)).any()
                print(resultData)
                print(similarity_score)
                    

                #print(room_data)'''

                # Living_room_data.csv
                # Kitchen_data.csv
                # Library_data.csv
                # Bedroom_data.csv
                # Bathroom_data.csv
                
                '''
                data_list_num = 0
                data_list = ['T_Living_room_data3.csv','T_Kitchen_data2.csv','T_Library_data2.csv','T_Bedroom_data2.csv','T_Bathroom_data2.csv']
                if not os.path.exists(data_list[data_list_num]):
                  with open(data_list[data_list_num],'w') as file:
                    write = csv.writer(file)
                    write.writerow(testname)
                  with open(data_list[data_list_num],'a') as file:
                    write = csv.writer(file)
                    write.writerow(test)
                else:
                  with open(data_list[data_list_num],'a') as file:
                    write = csv.writer(file)
                    write.writerow(test)'''
                
                if not os.path.exists('room_data.csv'):
                  with open('room_data.csv','w') as file:
                    write = csv.writer(file)
                    write.writerow(testname)
                    

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f'{names[c]}'
                    confidence = float(conf)
                    confidence_str = f'{confidence:.2f}'

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-csv', action='store_true', help='save results in CSV format')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
