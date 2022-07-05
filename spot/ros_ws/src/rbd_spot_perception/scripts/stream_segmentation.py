#!/usr/bin/env python
#
# Stream object segmentation results using Mask RCNN

import time
import sys
import argparse
import numpy as np

import cv2
import rospy
import struct
import torch
import torchvision
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header

import rbd_spot
from rbd_spot_perception.utils.vision.detector import (COCO_CLASS_NAMES,
                                                       maskrcnn_filter_by_score,
                                                       maskrcnn_draw_result)

def get_intrinsics(P):
    return dict(fx=P[0],
                fy=P[5],
                cx=P[2],
                cy=P[6])

class SegmentationPublisher:
    def __init__(self, camera, mask_threshold=0.7):
        self._camera = camera
        self._mask_threshold = mask_threshold
        self._segimg_pub = rospy.Publisher(f"/spot/segmentation/{camera}/result", Image, queue_size=10)
        self._segpcl_pub = rospy.Publisher(f"/spot/segmentation/{camera}/result_points", PointCloud2, queue_size=10)

    def publish_result(self, pred, visual_img, depth_img, caminfo):
        """
        Args:
            pred (Tensor): output of MaskRCNN model
            visual_img (np.ndarray): Image from the visual source (not rotated)
            depth_img (np.ndarray): Image from the corresponding depth source
        """
        # because the prediction is based on an upright image, we need to make sure
        # the drawn result is on an upright image
        if self._camera == "front":
            visual_img_upright = torch.tensor(cv2.rotate(visual_img, cv2.ROTATE_90_CLOCKWISE)).permute(2, 0, 1)
            result_img = maskrcnn_draw_result(pred, visual_img_upright)
        else:
            result_img = maskrcnn_draw_result(pred, torch.tensor(visual_img).permute(2, 0, 1))
        result_img_msg = rbd_spot.image.imgmsg_from_imgarray(result_img.permute(1, 2, 0).numpy())
        result_img_msg.header.stamp = caminfo.header.stamp
        result_img_msg.header.frame_id = caminfo.header.frame_id
        self._segimg_pub.publish(result_img_msg)
        rospy.loginfo("Published segmentation result (image)")

        # For each mask, obtain a set of points.
        masks = pred['masks'].squeeze()
        masks = masks.reshape(-1, masks.shape[-2], masks.shape[-1])   # make sure shape is (N, H, W) where N is number of masks
        masks = torch.greater(masks, self._mask_threshold)
        # We need to roate the masks cw by 90 deg if camera is front
        if self._camera == "front":
            masks = torch.rot90(masks, -1, (1,2))
        points = []
        for i, mask in enumerate(masks):
            mask_coords = mask.nonzero().cpu().numpy()  # works with boolean tensor too
            mask_coords_T = mask_coords.T
            mask_visual = visual_img[mask_coords_T[0], mask_coords_T[1], :].reshape(-1, 3)  # colors on the mask
            mask_depth = depth_img[mask_coords_T[0], mask_coords_T[1]]  # depth on the mask

            v, u = mask_coords_T[0], mask_coords_T[1]
            I = get_intrinsics(caminfo.P)
            z = mask_depth / 1000.0
            x = (u - I['cx']) * z / I['fx']
            y = (v - I['cy']) * z / I['fy']
            rgb = [struct.unpack('I', struct.pack('BBBB',
                                                  mask_visual[i][0],
                                                  mask_visual[i][1],
                                                  mask_visual[i][2], 255))[0]
                   for i in range(len(mask_visual))]
            points.extend([[x[i], y[i], z[i], rgb[i]]
                           for i in range(len(mask_visual))])
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('rgb', 12, PointField.UINT32, 1)]
        header = Header()
        header.stamp = caminfo.header.stamp
        header.frame_id = caminfo.header.frame_id
        pc2 = point_cloud2.create_cloud(header, fields, points)
        # static transform is already published by ros_publish_image_result, so no need here.
        self._segpcl_pub.publish(pc2)
        rospy.loginfo("Published segmentation result (points)")



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

    # create ros publishers; we publish: (1) raw image (2) depth (3) image with
    # segmentation drawn (4) segmentation point cloud
    if args.pub:
        # create ros publishers; we publish: (1) raw image (2) depth (3) camera info
        # (4) image with segmentation result drawn (5) segmentation point cloud
        # The first 3 are done through rbd_spot.image, while the last two are
        # handled by SegmentationPublisher.
        rospy.init_node("stream_segmentation")
        image_publishers = rbd_spot.image.ros_create_publishers(sources, name_space="segmentation")
        seg_publisher = SegmentationPublisher(args.camera)
        rate = rospy.Rate(args.rate)

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

            if args.pub:
                rbd_spot.image.ros_publish_image_result(conn, result, image_publishers)

            # Get visual, depth and camera info
            if args.camera == "front":
                # contains each element is (Image, CameraInfo)
                visual_depths = [(result[0], result[1]),
                                 (result[2], result[3])]
            else:
                visual_depths = [(result[0], result[1])]

            # run through model
            for visual_response, depth_response in visual_depths:
                visual_msg, caminfo = rbd_spot.image.imgmsg_from_response(visual_response, conn)
                depth_msg, caminfo = rbd_spot.image.imgmsg_from_response(depth_response, conn)

                image = rbd_spot.image.imgarray_from_imgmsg(visual_msg)
                if args.camera != "hand":
                    # grayscale image; make it 3 channels
                    image = cv2.merge([image, image, image])
                depth_image = rbd_spot.image.imgarray_from_imgmsg(depth_msg)

                image_input = torch.tensor(image)
                if args.camera == "front":
                    # we need to rotate the images by 90 degrees ccw to make it upright
                    image_input = torch.tensor(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
                image_input = torch.div(image_input, 255)
                if device.type == 'cuda':
                    image_input = image_input.cuda(device)

                pred = model([image_input.permute(2, 0, 1)])[0]
                pred = maskrcnn_filter_by_score(pred, 0.7)
                # Print out a summary
                print("detected objects: {}".format(list(sorted(COCO_CLASS_NAMES[l] for l in pred['labels']))))
                if len(pred['labels']) > 0:
                    if args.pub:
                        seg_publisher.publish_result(pred, image, depth_image, caminfo)
            if args.pub:
                rate.sleep()
            _used_time = time.time() - _start_time
            if args.timeout and _used_time > args.timeout:
                break

        finally:
            if args.pub and rospy.is_shutdown():
                sys.exit(1)



if __name__ == "__main__":
    main()
