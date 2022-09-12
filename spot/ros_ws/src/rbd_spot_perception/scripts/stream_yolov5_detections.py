#!/usr/bin/env python
#
# Stream object segmentation results using Mask RCNN
#
# Note that because the 3D projection contains only
# part of the object, the 3D bounding box will not be
# accurate. You might want to rely on the 3D pose only.

# This is another implementation of stream_segmentation.py
# But instead of using maskrcnn it uses a custom trained YoloV5
# And most of the design decision is taken from original stream_segmentation.py file

import time
import sys
import argparse
import numpy as np
import random
import os

import cv2
import rospy
import struct
import torch
import torchvision
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from vision_msgs.msg import BoundingBox3D
from geometry_msgs.msg import Point, Quaternion, Vector3
from std_msgs.msg import Header
from std_msgs.msg import ColorRGBA
from rbd_spot_perception.msg import (SimpleDetection2D,
                                     SimpleDetection2DArray,
                                     SimpleDetection3D,
                                     SimpleDetection3DArray)

import rbd_spot
from rbd_spot_perception.utils.vision.detector import (COCO_CLASS_NAMES,
                                                       maskrcnn_filter_by_score,
                                                       maskrcnn_draw_result,
                                                       bbox3d_from_points)
from rbd_spot_perception.utils.math import R2d

ABS_FILE_PATH = os.path.dirname(os.path.abspath(__file__))

def get_intrinsics(P):
    return dict(fx=P[0],
                fy=P[5],
                cx=P[2],
                cy=P[6])


def make_bbox3d_msg(center, sizes):
    if len(center) == 7:
        x, y, z, qx, qy, qz, qw = center
        q = Quaternion(x=qx, y=qy, z=qz, w=qw)
    else:
        x, y, z = center
        q = Quaternion(x=0, y=0, z=0, w=1)
    s1, s2, s3 = sizes
    msg = BoundingBox3D()
    msg.center.position = Point(x=x, y=y, z=z)
    msg.center.orientation = q
    msg.size = Vector3(x=s1, y=s2, z=s3)
    return msg

def make_bbox3d_marker_msg(center, sizes, marker_id, header):
    if len(center) == 7:
        x, y, z, qx, qy, qz, qw = center
        q = Quaternion(x=qx, y=qy, z=qz, w=qw)
    else:
        x, y, z = center
        q = Quaternion(x=0, y=0, z=0, w=1)
    s1, s2, s3 = sizes
    marker = Marker()
    marker.header = header
    marker.id = marker_id
    marker.type = Marker.CUBE
    marker.pose.position = Point(x=x, y=y, z=z)
    marker.pose.orientation = q
    # The actual bounding box seems tooo large - for now just draw the center;
    marker.scale = Vector3(x=s1, y=s2, z=s3)  #0.2, y=0.2, z=0.2)
    marker.action = Marker.MODIFY
    marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.3)
    return marker

def rotate_bbox2d_90(bbox, dim, d="counterclockwise"):
    """deg: angle in degrees; d specifies direction, either
    'counterclockwise' or 'clockwise'"""
    if d != "counterclockwise" and d != "clockwise":
        raise ValueError("Invalid direction", d)
    x1, y1, x2, y2 = bbox
    h, w = dim
    if d == "clockwise":
        return np.array([w - y1, x1,
                         w - y2, x2])
    else:
        return np.array([y1, h - x1,
                         y2, h - x2])


class SegmentationPublisher:
    def __init__(self, camera, mask_threshold=0.7):
        self._camera = camera
        self._mask_threshold = mask_threshold
        # Publishes the image with segmentation drawn
        self._segimg_pub = rospy.Publisher(f"/spot/segmentation/{camera}/result", Image, queue_size=10)
        # Publishes the point cloud of the back-projected segmentations
        self._segpcl_pub = rospy.Publisher(f"/spot/segmentation/{camera}/result_points", PointCloud2, queue_size=10)
        # Publishes bounding boxes of detected objects with reasonable filtering done.
        self._segdet3d_pub = rospy.Publisher(f"/spot/segmentation/{camera}/result_boxes3d", SimpleDetection3DArray,  queue_size=10)
        # Publishes the detected classes, confidence scores, and boudning boxes
        self._segdet2d_pub = rospy.Publisher(f"/spot/segmentation/{camera}/result_boxes2d", SimpleDetection2DArray, queue_size=10)
        # Bounding box visualization markers
        self._segbox_markers_pub = rospy.Publisher(f"/spot/segmentation/{camera}/result_boxes_viz", MarkerArray, queue_size=10)

    def publish_result(self, yolopred, yoloPandaDataframe, visual_img, depth_img, caminfo):
        """
        Args:
            yolopred : output of yolov5 model
            yoloPandaDataframe: Panda dataframe for all bounding boxes with prediction
            visual_img (np.ndarray): Image from the visual source (not rotated)
            depth_img (np.ndarray): Image from the corresponding depth source
        """
        # publish the 2D bounding boxes
        det2d_msgs = []

        # render generates the numpy array of yolo predicted image with label and bounding boxes
        yoloRender = yolopred.render()

        # Publishing result of predicted 2d bounding boxes
        for _, row in yoloPandaDataframe.iterrows():
            det2d = SimpleDetection2D(label=row['name'], score=np.float32(row['confidence']),x1=int(row['xmin']), y1=int(row['ymin']),x2=int(row['xmax']), y2=int(row['ymax']))
            det2d_msgs.append(det2d)

        det2d_array = SimpleDetection2DArray(header=caminfo.header, detections=det2d_msgs)
        # # rospy.loginfo("Publishing det2d_array = ", det2d_array)
        self._segdet2d_pub.publish(det2d_array)

        # Publishing image with 2d bounding box
        result_img_msg = rbd_spot.image.imgmsg_from_imgarray(yoloRender[0])
        result_img_msg.header.stamp = caminfo.header.stamp
        result_img_msg.header.frame_id = caminfo.header.frame_id
        self._segimg_pub.publish(result_img_msg)
        rospy.loginfo("Published segmentation result (image)")

        points = []
        markers = []
        det3d_msgs = []

        # for mask in yoloMasks:
        # The logic of the point cloud and 3d bounding box generation is
        # it is sampling a small subset area from the 2d bounding box detected by YoloV5
        # and making 3d point cloud and corrosponding bounding box from that

        for _, row in yoloPandaDataframe.iterrows():
            xmin = int(row['xmin'])
            ymin = int(row['ymin'])
            xmax = int(row['xmax'])
            ymax = int(row['ymax'])
            predictedLabel = row['name']
            confidence = np.float32(row['confidence'])

            sample_length = min( (ymax-ymin), (xmax-xmin))
            sample_length = sample_length * 0.8 # Change this value to make the sample window of depth calculation area
            sample_length = int(sample_length)
            y_coords_list = []
            x_coords_list = []

            y_start = ymin + ((ymax-ymin - sample_length) // 2)
            y_end = y_start + sample_length

            x_start = xmin + ((xmax-xmin - sample_length) // 2)
            x_end = x_start + sample_length

            # Populating all the points or pixels inside the sample area
            # TODO: Should be a better way to do this. Find that.
            for i in range (y_start, y_end):
                for j in range(x_start, x_end):
                    y_coords_list.append(int(i))
                    x_coords_list.append(int(j))

            y_coords = np.array(y_coords_list)
            x_coords = np.array(x_coords_list)
            mask_visual = visual_img[y_coords, x_coords, :].reshape(-1, 3)  # colors on the mask
            mask_depth = depth_img[y_coords, x_coords]  # depth on the mask

            v, u = y_coords, x_coords

            I = get_intrinsics(caminfo.P)
            z = mask_depth / 1000.0
            x = (u - I['cx']) * z / I['fx']
            y = (v - I['cy']) * z / I['fy']
            # filter out points too close to the gripper (most likely noise)
            keep_indices = np.argwhere(z > 0.1).flatten()
            z = z[keep_indices]
            if len(z) == 0:
                continue  # we won't have points for this mask
            x = x[keep_indices]
            y = y[keep_indices]
            color = [struct.unpack('I', struct.pack('BBBB',
                                                    mask_visual[i][2],
                                                    mask_visual[i][1],
                                                    mask_visual[i][0], 255))[0]
                   for i in keep_indices]
            # The points for a single detection mask
            mask_points = [[x[i], y[i], z[i], color[i]]
                           for i in range(len(x))]
            points.extend(mask_points)
            try:
                box_center, box_sizes = bbox3d_from_points([x, y, z], axis_aligned=True, no_rotation=True)
                bbox3d_msg = make_bbox3d_msg(box_center, box_sizes)
                label = predictedLabel
                score = confidence
                det3d = SimpleDetection3D(label=label,
                                          score=score,
                                          box=bbox3d_msg)
                det3d_msgs.append(det3d)
                markers.append(make_bbox3d_marker_msg(box_center, box_sizes, 1000 + i, result_img_msg.header))
            except Exception as ex:
                rospy.logerr(f"Error: {ex}")

        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('rgb', 12, PointField.UINT32, 1)]
        pc2 = point_cloud2.create_cloud(caminfo.header, fields, points)
        # static transform is already published by ros_publish_image_result, so no need here.
        self._segpcl_pub.publish(pc2)
        rospy.loginfo("Published segmentation result (points)")
        # publish bounding boxes and markers
        det3d_array = SimpleDetection3DArray(header=caminfo.header,
                                             detections=det3d_msgs)
        self._segdet3d_pub.publish(det3d_array)
        rospy.loginfo("Published segmentation result (bboxes)")
        markers_array = MarkerArray(markers=markers)
        self._segbox_markers_pub.publish(markers_array)
        rospy.loginfo("Published segmentation result (markers)")



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
        rospy.init_node(f"stream_segmentation_{args.camera}")
        image_publishers = rbd_spot.image.ros_create_publishers(sources, name_space="segmentation")
        seg_publisher = SegmentationPublisher(args.camera)
        rate = rospy.Rate(args.rate)

    print(f"Will stream images from {sources}")
    image_requests = rbd_spot.image.build_image_requests(
        sources, quality=args.quality, fmt=args.format)

    print("Loading model...")
    yolomodel = torch.hub.load('ultralytics/yolov5', 'custom', path=f"{ABS_FILE_PATH}/../models/yolov5/yolov5_lab_custom/best.pt")
    # yolomodel.conf = 0.45
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

                yolomodelPrediction = yolomodel(image)
                yoloPandaDataframe = yolomodelPrediction.pandas().xyxy[0]

                if args.pub:
                    seg_publisher.publish_result(yolomodelPrediction, yoloPandaDataframe, image, depth_image, caminfo)
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
