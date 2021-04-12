import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from utils.torch_utils import select_device, load_classifier, time_synchronized

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import rospy
from utils.datasets import letterbox
# from utils.general import letterbox


class YoloROS:
    def __init__(self, save_img=False):
        rospy.init_node("yolo")
        out, source, weights, save_txt, imgsz = \
            opt.output, opt.source, opt.weights, opt.save_txt, opt.img_size
        # webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

        # Initialize
        device = select_device(opt.device)
        self.opt = opt
        if os.path.exists(out):
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder
        half = device.type != 'cpu'  # half precision only supported on CUDA
        self.half = half

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
        if half:
            model.half()  # to FP16

        # Second-stage classifier
        # classify = False
        # if classify:
        #     modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        #     modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        #     modelc.to(device).eval()

        # Get names and colors
        self.names = model.module.names if hasattr(model, 'module') else model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

        # Run inference
        # t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

        self.model = model
        self.device = device
        self.imgsz = imgsz
        self.compressed_img = False
        if (opt.pub_topic is None):
            self.pub_topic = opt.img_topic + "_detected" #新话题名  等于  原话题名_detected 所以rviz观测应该选新话题名
        else:
            self.pub_topic = opt.pub_topic
        self.publisher = rospy.Publisher(self.pub_topic, Image, queue_size=2)
        if ("compressed" in opt.img_topic):
            self.compressed_img = True
            rospy.Subscriber(opt.img_topic, CompressedImage, callback=self.on_image, queue_size=1, buff_size=14745600)
        else:
            rospy.Subscriber(opt.img_topic, Image, callback=self.on_image, queue_size=1, buff_size=14745600)
        self.cv_bridge = CvBridge()
        rospy.loginfo("Initialized. Waiting for image message %s." % opt.img_topic)
        rospy.spin()
        #这句话的意思是循环且监听反馈函数（callback）。循环就是指程序运行到这里，就会一直在这里循环了。监听反馈函数的意思是，如果这个节点有callback函数，那写一句ros::spin()在这里，就可以在有对应消息到来的时候，运行callback函数里面的内容。
        #就目前而言，以我愚见，我觉得写这句话适用于写在程序的末尾（因为写在这句话后面的代码不会被执行），适用于订阅节点，且订阅速度没有限制的情况
        #参考https://blog.csdn.net/datase/article/details/79742421
    
    def on_image(self, img_msg):
        # rospy.loginfo("receive image msg")
        if (self.compressed_img):
            im0 = self.cv_bridge.compressed_imgmsg_to_cv2(img_msg, 'bgr8')
        else:
            im0 = self.cv_bridge.imgmsg_to_cv2(img_msg, 'bgr8')

        img = None
        print(im0.shape, end=" ")
        if (len(im0.shape) == 2):
            im0 = cv2.merge([im0, im0, im0])
            img = letterbox(im0, self.imgsz, color=114)[0][:, :, ::-1].transpose(2,0,1)
        elif (len(im0.shape) == 3):
            if (im0.shape[0] != 3):
                img = letterbox(im0, self.imgsz)[0]
                img = img[:, :, ::-1].transpose(2,0,1)
            else:
                img = letterbox(im0[::-1, :,:].transpose(1,2,0), self.imgsz)[0].transpose(2,0,1)
        # img = letterbox(im0, self.imgsz)[0]
        # img = im0[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        print(img.shape, end=" ")







        # Inference
        t1 = time_synchronized()
        pred = self.model(img, augment=self.opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes, agnostic=self.opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        # if self.classify:
        #     pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            # if webcam:  # batch_size >= 1
            #     p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            # else:
            #     p, s, im0 = path, '', im0s

            # save_path = str(Path(out) / Path(p).name)
            # txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            # s += '%gx%g ' % img.shape[2:]  # print string
            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                # for c in det[:, -1].unique():
                #     n = (det[:, -1] == c).sum()  # detections per class
                #     s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    # if save_txt:  # Write to file
                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #     with open(txt_path + '.txt', 'a') as f:
                    #         f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    # if save_img or view_img:  # Add bbox to image
                    #     label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, im0, color=self.colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print('\rcost time: %.3fs ( %.1f fps )' % ((t2 - t1), 1.0/(t2-t1)), end="", flush=True)

            # Stream results
            # if self.view_img:
            #     cv2.namedWindow('yolo results', 0)
            #     cv2.imshow('yolo results', im0)
            #     if cv2.waitKey(1) == ord('q'):  # q to quit
            #         raise StopIteration
            if self.opt.publish:
                publish_msg = self.cv_bridge.cv2_to_imgmsg(im0, 'bgr8')
                publish_msg.header = img_msg.header
                self.publisher.publish(publish_msg)

            # Save results (image with detections)
    #         if save_img:
    #             if dataset.mode == 'images':
    #                 cv2.imwrite(save_path, im0)


    # if save_txt or save_img:
    #     print('Results saved to %s' % Path(out))
    #     if platform == 'darwin' and not self.opt.update:  # MacOS
    #         os.system('open ' + save_path)

    # print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--publish', action='store_true', help='publish detection image result')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--img_topic', type=str, help='receive image topic name', default="/rgb_right/image_raw")
    parser.add_argument('--pub_topic', type=str, help='publish image topic name', default=None)
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        yolo = YoloROS()
