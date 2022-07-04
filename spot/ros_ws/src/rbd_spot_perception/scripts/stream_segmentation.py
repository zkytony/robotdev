#!/usr/bin/env python
#
# Stream object segmentation results using Mask RCNN

import time
import sys
import argparse

import cv2
import rospy
import torch
import torchvision

import rbd_spot
from rbd_spot_perception.utils.vision.detector import (COCO_CLASS_NAMES,
                                                       maskrcnn_filter_by_score)


def main():
    parser = argparse.ArgumentParser(description="stream segmentation with Mask RCNN")
    parser.add_argument("--camera", type=str, help="camera set to stream images from.",
                        choices=['front', 'left', 'right', 'back', 'hand'],
                        default='hand')
    parser.add_argument("-q", "--quality", type=int,
                        help="image quality [0-100]", default=75)
    formats = ["UNKNOWN", "JPEG", "RAW", "RLE"]
    parser.add_argument("-f", "--format", type=str, default="RAW",
                        help="format", choices=formats)
    parser.add_argument("-t", "--timeout", type=float, help="time to keep streaming")
    parser.add_argument("-p", "--pub", action="store_true", help="publish as ROS messages")
    parser.add_argument("-r", "--rate", type=float,
                        help="maximum number of detections per second", default=3.0)

    args, _ = parser.parse_known_args()
    conn = rbd_spot.SpotSDKConn(sdk_name="StreamSegmentationClient")
    image_client = rbd_spot.image.create_client(conn)

    # Make image requests for specified camera
    if args.camera == "hand":
        sources = ["hand_color_image", "hand_depth_in_hand_color_frame"]
    elif args.camera == "front":
        sources = ["frontleft_fisheye_image",
                   "frontleft_depth_in_visual_frame",
                   "frontright_fisheye_image",
                   "frontright_depth_in_visual_frame"]
    else:
        sources = [f"{args.camera}_fisheye_image",
                   f"{args.camera}_depth_in_visual_frame"]

    print(f"Will stream images from {sources}")
    image_requests = rbd_spot.image.build_image_requests(
        sources, quality=args.quality, fmt=args.format)

    print("Loading model...")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    model.to(device)

    # Stream the image through specified sources
    _start_time = time.time()
    while True:
        try:
            result, time_taken = rbd_spot.image.getImage(image_client, image_requests)
            print("GetImage took: {:.3f}".format(time_taken))

            # get the images
            images_to_feed = []
            if args.camera == "front":
                # we run the detection for both frontleft and frontright;
                # we also need to rotate the images by 90 degrees counter-clockwise
                frontleft_fisheye = rbd_spot.image.imgarray_from_response(result[0], conn)   # frontleft
                frontright_fisheye = rbd_spot.image.imgarray_from_response(result[2], conn)  # frontright
                frontleft_fisheye = cv2.rotate(frontleft_fisheye, cv2.ROTATE_90_COUNTERCLOCKWISE)
                frontright_fisheye = cv2.rotate(frontright_fisheye, cv2.ROTATE_90_COUNTERCLOCKWISE)
                images_to_feed = [frontleft_fisheye, frontright_fisheye]
            else:
                images_to_feed = [rbd_spot.image.imgarray_from_response(result[0], conn)]

            # run through model
            predictions = []
            for image in images_to_feed:
                image_input = torch.div(torch.tensor(image), 255)
                if device.type == 'cuda':
                    image_input = image_input.cuda(device)
                pred = model([image_input.permute(2, 0, 1)])[0]
                pred = maskrcnn_filter_by_score(pred, 0.7)
                predictions.append(pred)
                # Print out a summary
                print("detected objects: {}".format(list(sorted(COCO_CLASS_NAMES[l] for l in pred['labels']))))

            # if args.pub:
            #     rbd_spot.image.ros_publish_image_result(conn, result, publishers)
            _used_time = time.time() - _start_time
            if args.timeout and _used_time > args.timeout:
                break

        finally:
            if args.pub and rospy.is_shutdown():
                sys.exit(1)



if __name__ == "__main__":
    main()
