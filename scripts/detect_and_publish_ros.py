#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import argparse
from sys import platform
from sensor_msgs.msg import Image, CompressedImage

from cv_bridge import CvBridge

import rospy

from models import *  # set ONNX_EXPORT in models.py

from utils.datasets import *
from utils.utils import *

from utils.datasets import letterbox

from object_msgs.msg import BoundingBox
from object_msgs.msg import BoundingBoxes

class YoloROS:
    def __init__(self, save_img=False):
        rospy.init_node("yolo")
        imgsz = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)        
        out, source, weights, half = opt.output, opt.source, opt.weights, opt.half#删除了属性view_img，因为文末没有了
        webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt') 

        # Initialize
        device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
        self.opt = opt##后加的
        if os.path.exists(out):
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder 

        # Initialize model
        model = Darknet(opt.cfg, imgsz)##Darknet来自文件./models.py 

        # Load weights
        attempt_download(weights)
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        else:  # darknet format
            _ = load_darknet_weights(model, weights)              

        # Second-stage classifier
        # classify = False
        # if classify:
        #     modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        #     modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        #     modelc.to(device).eval()

        # Eval mode
        model.to(device).eval()

        # Export mode
        if ONNX_EXPORT:
            img = torch.zeros((1, 3) + imgsz)  # (1, 3, 320, 192)
            torch.onnx.export(model, img, 'weights/export.onnx', verbose=False, opset_version=11)

            # Validate exported model
            import onnx
            model = onnx.load('weights/export.onnx')  # Load the ONNX model
            onnx.checker.check_model(model)  # Check that the IR is well formed
            print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
            return

        # Half precision
        half = half and device.type != 'cpu'  # half precision only supported on CUDA
        self.half = half##后加的
        if half:
            model.half()

        # detect.py中Set Dataloader这里没有，不用从文件夹读取图片
            
        # Get classes and colors
        self.classes = load_classes(parse_data_cfg(opt.data)['names'])
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.classes))]              

        # Run inference
        #t0 = time.time()

        ##后加的
        self.classes = load_classes(parse_data_cfg(opt.data)['names'])
        self.model = model
        self.device = device
        self.imgsz = imgsz
        self.compressed_img = False ##以safe为前缀的变量　增加布尔型变量：是否压缩了图片
        if (opt.pub_topic is None):
            self.pub_topic = opt.img_topic + "_detected" ##新话题名  等于  原话题名_detected 所以rviz观测应该选新话题名
        else:
            self.pub_topic = opt.pub_topic
        self.publisher = rospy.Publisher(self.pub_topic, BoundingBoxes, queue_size=2)  #发布bbox话题
        # self.publisher = rospy.Publisher(self.pub_topic, Image, queue_size=2)##发布新话题，图片

        ##后加的　如果原话题中有compressed怎么处理
        if ("compressed" in opt.img_topic):
            self.compressed_img = True
            rospy.Subscriber(opt.img_topic, CompressedImage, callback=self.on_image, queue_size=1, buff_size=14745600)
        else:
            rospy.Subscriber(opt.img_topic, Image, callback=self.on_image, queue_size=1, buff_size=14745600)
        self.cv_bridge = CvBridge()

        ##后加的
        rospy.loginfo("Initialized. Waiting for image message %s." % opt.img_topic)
        rospy.spin()  

    def on_image(self, img_msg):
        # rospy.loginfo("receive image msg")

        ##后加的　如果原话题中有compressed怎么处理
        if (self.compressed_img):
            im0 = self.cv_bridge.compressed_imgmsg_to_cv2(img_msg, 'bgr8')
        else:
            im0 = self.cv_bridge.imgmsg_to_cv2(img_msg, 'bgr8')

        ##后加的
        img = None
        # print(im0.shape)#, end=" ")
        if (len(im0.shape) == 2):##如果原图是二维的灰度图
            im0 = cv2.merge([im0, im0, im0])
            img = letterbox(im0, self.imgsz, color=114)[0][:, :, ::-1].transpose(2,0,1)
            img = np.ascontiguousarray(img, dtype=np.float16 if self.half else np.float32)  # uint8 to fp16/fp32
            img /= 255.0
            # cv2.imshow('img', im0)
            # cv2.waitKey(0)
        elif (len(im0.shape) == 3):##如果原图是三维的彩色
            if (im0.shape[0] != 3):##如果原图不是ＣＨＷ或ＣＷＨ,即Ｃ在最后一维
                img = letterbox(im0, self.imgsz)[0]
                img = img[:, :, ::-1].transpose(2,0,1)##img[:, :, ::-1]最后一维的Ｃ通道bgr顺序到rgb
                img = np.ascontiguousarray(img, dtype=np.float16 if self.half else np.float32)  # uint8 to fp16/fp32
                img = img / 255.0
            else:
                img = letterbox(im0[::-1, :,:].transpose(1,2,0), self.imgsz)[0].transpose(2,0,1)
        # img = letterbox(im0, self.imgsz)[0]
        # img = im0[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)##ascontiguousarray函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快。



        t0 = time.time()

        # Get on_imagedetections
        img = torch.from_numpy(img).to(self.device)##只有self.这不同
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t1 = time.time()##抄的yolov5这里
        img =img.float()##后加这句话没报133行错误了
        pred = self.model(img)[0]##只有self.这不同
        ##报错RuntimeError: Input type (torch.cuda.ByteTensor) and weight type (torch.FloatTensor) should be the same

        if opt.half:
            pred = pred.float()

        # Apply NMS
        # print(img.shape, type(img), img.dtype)
        # print(im0.shape, type(im0), im0.dtype)
        # print(pred.shape, type(pred), pred.dtype)
        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.nms_thres)##只有self.这不同
        ##报错./utils/utils.py, line 486RuntimeError: Output 0 of UnbindBackward is a view and its base or another view of its base has been modified inplace. This view is the output of a function that returns multiple views. Such functions do not allow the output views to be modified inplace. You should replace the inplace operation by an out-of-place one.
        ##报错RuntimeError:UnbindBackward的输出0是一个视图，其基视图或其基视图的另一个视图已被就地修改。此视图是返回多个视图的函数的输出。这样的函数不允许在原地修改输出视图。你应该用一个不合适的操作来代替原地操作。
        t2 = time.time()##抄的yolov5这里
        # print('\rcost time: %.3fs ( %.1f fps )' % ((t2 - t1), 1.0/(t2-t1)), end="", flush=True) 
        # Apply Classifier
        # if self.classify:
        #     pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        
        bboxs = BoundingBoxes()
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                ##img_size被改写成imgsz
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round() 

                # Write results
                for *xyxy, conf, _, cls in det:
                    bbox = BoundingBox()
                    label = '%s %.2f' % (self.classes[int(cls)], conf)
                    # label = "object"
                    # print("class idx:", cls)
                    plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)])
                    bbox.Class = self.classes[int(cls)]
                    bbox.probability = conf
                    xywh = xyxy2xywh(torch.tensor(xyxy).view(1, 4)).view(-1).tolist()#xyxy2xywh(xyxy)
                    bbox.x = xywh[0]
                    bbox.y = xywh[1]
                    bbox.w = xywh[2]
                    bbox.h = xywh[3]
            # Print time (inference + NMS)
            # print('\rcost time: %.3fs ( %.1f fps )' % ((t2 - t1), 1.0/(t2-t1)), end="", flush=True)   ##用到前面pred的t2-t1                
                    bboxs.bounding_boxes.append(bbox)
                    # print(i, len(bboxs.bounding_boxes))

        if self.opt.publish:
            bboxs.header = img_msg.header
            cv2.imshow("Image window", im0)
            cv2.waitKey(10)
            self.publisher.publish(bboxs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp-r.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/custom.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/final.pt', help='path to weights file')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')##pixel像素
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
#    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--publish', action='store_true', help='publish detection image result') ##
    parser.add_argument('--img_topic', type=str, help='receive image topic name', default="/thermal/image_raw/compressed") ##
    parser.add_argument('--pub_topic', type=str, help='publish image topic name', default=None)  ##
    opt = parser.parse_args()
    print(opt)
    ##运行时会显示Namespace(cfg='./cfg/yolov3-spp-r.cfg', conf_thres=0.3, data='./data/custom.data', device='', fourcc='mp4v', half=False, img_size=416, img_topic='/thermal/image_raw/compressed', nms_thres=0.5, output='output', pub_topic=None, publish=True, source='data/samples', weights='./weights/final.pt')


    with torch.no_grad():
        yolo = YoloROS()

