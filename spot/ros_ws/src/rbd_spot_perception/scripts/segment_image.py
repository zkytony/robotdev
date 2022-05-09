#!/usr/bin/env python
# Stream images through Spot
#
# Usage examples:
#
# rosrun rbd_spot_perception stream_image.py

import time
import argparse

from cv2 import bitwise_and

import rospy
from rbd_spot_robot.utils import ros_utils
from sensor_msgs.msg import Image

from google.protobuf.json_format import MessageToDict, ParseDict
import numpy as np
import pandas as pd

import spot_driver.ros_helpers
import rbd_spot
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

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

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()
model.to(device)
publisher = rospy.Publisher("/spot/segment/image", Image, queue_size=1)

def segment_and_publish(msg):
    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    ax.axis('off')

    color_img_msg, _ = spot_driver.ros_helpers._getImageMsg(msg[0], msg[0].shot.acquisition_time)
    depth_img_msg, _ = spot_driver.ros_helpers._getImageMsg(msg[1], msg[1].shot.acquisition_time)
    img = ros_utils.convert(color_img_msg)
    depth_img = ros_utils.convert(depth_img_msg)
    
    float_img = np.array(img, dtype=np.float32)
    torch_img = torch.tensor(img).float()
    torch_img /= 255
    torch_img = torch_img.permute(2, 0, 1)
    cuda_img = torch_img.cuda(device)
    pred  = model([cuda_img])
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_thresholded = [pred_score.index(x) for x in pred_score if x>0.8]
    if len(pred_thresholded) > 0:
        pred_t = pred_thresholded[-1] if len(pred_thresholded) > 0 else 0
        masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()[:pred_t+1]
        bottle_mask = np.full(masks[0].shape, False)
        pred_class = [class_names[i] for i in list(pred[0]['labels'][:pred_t+1].cpu().numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'][:pred_t+1].detach().cpu().numpy())]
        for i, mask in enumerate(masks):
            rgb_mask = get_coloured_mask(mask)
            img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
            if pred_class[i] == "knife":
                print("HERE")
                bottle_mask = np.logical_or(bottle_mask, mask)
            box_pt1 = [int(j) for j in pred_boxes[i][0]]
            box_pt2 = [int(j) for j in pred_boxes[i][1]]
            cv2.rectangle(img, box_pt1, box_pt2,color=(0, 1, 0), thickness=3)
            cv2.putText(img,pred_class[i], box_pt1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),thickness=2)
        depth_img = cv2.bitwise_and(depth_img, depth_img, mask=np.where(bottle_mask, np.uint8(255), np.uint8(0)))
    else:
        print("Failed to detect objects")
    
    result_bytes = img.flatten().tobytes()
    depth_result_bytes = depth_img.flatten().tobytes()
    color_img_msg.data = result_bytes
    depth_img_msg.data = depth_result_bytes

    publisher.publish(depth_img_msg)


def get_images():
    conn = rbd_spot.SpotSDKConn(sdk_name="StreamImageClient")
    image_client = rbd_spot.image.create_client(conn)

    sources_result, _used_time = rbd_spot.image.listImageSources(image_client)
    print("ListImageSources took %.3fs" % _used_time)
    sources_df = pd.DataFrame(rbd_spot.image.sources_result_to_dict(sources_result))
    print("Available image sources:")
    print(sources_df, "\n")


    # Check if the sources are valid
    color_source = "hand_color_image"
    depth_source = "hand_depth_in_hand_color_frame"
    sources = [color_source, depth_source]

    image_requests = rbd_spot.image.build_image_requests(
        sources, quality=75, fmt="RAW")

    # Stream the image through specified sources
    _start_time = time.time()
    while True:
        try:
            result, time_taken = rbd_spot.image.getImage(image_client, image_requests)
            segment_and_publish(result)
            print(time_taken)
        finally:
            if rospy.is_shutdown():
                sys.exit(1)


def main():
    rospy.init_node("segment_image")
    get_images()

if __name__ == "__main__":
    main()
