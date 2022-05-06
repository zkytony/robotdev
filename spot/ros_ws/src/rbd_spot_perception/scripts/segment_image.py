#!/usr/bin/env python
# Stream images through Spot
#
# Usage examples:
#
# rosrun rbd_spot_perception stream_image.py


import time
import argparse

import rospy
from rbd_spot_robot.utils import ros_utils
from sensor_msgs.msg import Image

from google.protobuf.json_format import MessageToDict, ParseDict
import numpy as np

import rbd_spot
import pandas as pd
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from bosdyn.api import image_pb2

import torch

import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import cv2
import random
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
            'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
            'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
            'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
            'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
            'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
            'teddy bear', 'hair drier', 'toothbrush']

def get_coloured_mask(mask):
    """
    random_colour_masks
      parameters:
        - image - predicted masks
      method:
        - the masks of each predicted object is given random colour for visualization
    """
    colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask == 1] = colours[random.randrange(0,10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask

def make_callback():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    model.to(device)
    publisher = rospy.Publisher("/spot/segment/image", Image, queue_size=1)
    def callback(msg):
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.gca()

        ax.axis('off')

        img = ros_utils.convert(msg)
        
        try:
            float_img = np.array(img, dtype=np.float32)
            torch_img = torch.tensor(img).float()
            torch_img /= 255
            torch_img = torch_img.permute(2, 0, 1)
            cuda_img = torch_img.cuda(device)
            pred  = model([cuda_img])
            pred_score = list(pred[0]['scores'].detach().cpu().numpy())
            pred_thresholded = [pred_score.index(x) for x in pred_score if x>0.5]
            pred_t = pred_thresholded[-1] if len(pred_thresholded) > 0 else 0
            masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
            pred_class = [class_names[i] for i in list(pred[0]['labels'].cpu().numpy())]
            pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
            masks = masks[:pred_t+1]
            boxes = pred_boxes[:pred_t+1]
            pred_cls = pred_class[:pred_t+1]
            for i in range(len(masks)):
                rgb_mask = get_coloured_mask(masks[i])
                img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
                box_pt1 = [int(j) for j in boxes[i][0]]
                box_pt2 = [int(j) for j in boxes[i][1]]
                cv2.rectangle(img, box_pt1, box_pt2,color=(0, 1, 0), thickness=3)
                cv2.putText(img,pred_cls[i], box_pt1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),thickness=2)
        except:
            print("Unable to detect object")

        
        result_bytes = img.flatten().tobytes()
        msg.data = result_bytes

        publisher.publish(msg)

    return callback


def main():
    rospy.init_node('segment')
    sub = rospy.Subscriber("/spot/stream_image/hand_color_image/image", Image, make_callback(), queue_size=1, buff_size=2**23)
    rospy.spin()

if __name__ == "__main__":
    main()
