#!/usr/bin/env python
# Stream images through Spot
#
# Usage examples:
#
# rosrun rbd_spot_perception stream_image.py

import time
import argparse
from tkinter.font import ROMAN
from xxlimited import foo

from cv2 import bitwise_and

import rospy
from rbd_spot_robot.utils import ros_utils
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose

from google.protobuf.json_format import MessageToDict, ParseDict
import numpy as np
import pandas as pd

import spot_driver.ros_helpers
from bosdyn.client.frame_helpers import *
import rbd_spot
import sys
import rbd_spot_robot.state
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
import open3d as o3d

from scipy.spatial.transform import Rotation as R

import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util

from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder, blocking_stand, block_until_arm_arrives
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client import math_helpers

from bosdyn.api import arm_command_pb2
import bosdyn.api.gripper_command_pb2
from bosdyn.client.frame_helpers import *
from bosdyn.api import geometry_pb2

import traceback
import time

def move_arm(config, pos):
    """A simple example of using the Boston Dynamics API to command Spot's arm."""

    # See hello_spot.py for an explanation of these lines.
    bosdyn.client.util.setup_logging(config.verbose)

    sdk = bosdyn.client.create_standard_sdk('ArmSpotClient')
    robot = sdk.create_robot(config.hostname)
    robot.authenticate(config.username, config.password)
    robot.time_sync.wait_for_sync()

    assert robot.has_arm(), "Robot requires an arm to run this example."

    # Verify the robot is not estopped and that an external application has registered and holds
    # an estop endpoint.
    assert not robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, " \
                                    "such as the estop SDK example, to configure E-Stop."

    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)

    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    lease = lease_client.acquire()
    try:
        with bosdyn.client.lease.LeaseKeepAlive(lease_client):
            # Now, we are ready to power on the robot. This call will block until the power
            # is on. Commands would fail if this did not happen. We can also check that the robot is
            # powered at any point.
            robot.logger.info("Powering on robot... This may take several seconds.")
            robot.power_on(timeout_sec=20)
            assert robot.is_powered_on(), "Robot power on failed."
            robot.logger.info("Robot powered on.")

            # Tell the robot to stand up. The command service is used to issue commands to a robot.
            # The set of valid commands for a robot depends on hardware configuration. See
            # SpotCommandHelper for more detailed examples on command building. The robot
            # command service requires timesync between the robot and the client.
            robot.logger.info("Commanding robot to stand...")
            command_client = robot.ensure_client(RobotCommandClient.default_service_name)
            blocking_stand(command_client, timeout_sec=10)
            robot.logger.info("Robot standing.")

            time.sleep(2.0)

            # Move the arm to a spot in front of the robot, and open the gripper.


            # Send the request
            cmd_id = command_client.robot_command(command)
            robot.logger.info('Moving arm to position 1.')

            # Wait until the arm arrives at the goal.
            block_until_arm_arrives_with_prints(robot, command_client, cmd_id)

            # Move the arm to a different position
            hand_ewrt_flat_body.z = 0

            flat_body_Q_hand.w = 0.707
            flat_body_Q_hand.x = 0.707
            flat_body_Q_hand.y = 0
            flat_body_Q_hand.z = 0

            flat_body_T_hand2 = geometry_pb2.SE3Pose(position=hand_ewrt_flat_body,
                                                     rotation=flat_body_Q_hand)
            odom_T_hand = odom_T_flat_body * math_helpers.SE3Pose.from_obj(flat_body_T_hand2)

            arm_command = RobotCommandBuilder.arm_pose_command(
                odom_T_hand.x, odom_T_hand.y, odom_T_hand.z, odom_T_hand.rot.w, odom_T_hand.rot.x,
                odom_T_hand.rot.y, odom_T_hand.rot.z, ODOM_FRAME_NAME, seconds)

            # Close the gripper
            gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(0.0)

            # Build the proto
            command = RobotCommandBuilder.build_synchro_command(gripper_command, arm_command)

            # Send the request
            cmd_id = command_client.robot_command(command)
            robot.logger.info('Moving arm to position 2.')

            # Wait until the arm arrives at the goal.
            # Note: here we use the helper function provided by robot_command.
            block_until_arm_arrives(command_client, cmd_id)

            robot.logger.info('Done.')

            # Power the robot off. By specifying "cut_immediately=False", a safe power off command
            # is issued to the robot. This will attempt to sit the robot before powering off.
            robot.power_off(cut_immediately=False, timeout_sec=20)
            assert not robot.is_powered_on(), "Robot power off failed."
            robot.logger.info("Robot safely powered off.")
    finally:
        # If we successfully acquired a lease, return it.
        lease_client.return_lease(lease)


def block_until_arm_arrives_with_prints(robot, command_client, cmd_id):
    """Block until the arm arrives at the goal and print the distance remaining.
        Note: a version of this function is available as a helper in robot_command
        without the prints.
    """
    while True:
        feedback_resp = command_client.robot_command_feedback(cmd_id)
        robot.logger.info(
            'Distance to go: ' +
            '{:.2f} meters'.format(feedback_resp.feedback.synchronized_feedback.arm_command_feedback
                                   .arm_cartesian_feedback.measured_pos_distance_to_goal) +
            ', {:.2f} radians'.format(
                feedback_resp.feedback.synchronized_feedback.arm_command_feedback.
                arm_cartesian_feedback.measured_rot_distance_to_goal))

        if feedback_resp.feedback.synchronized_feedback.arm_command_feedback.arm_cartesian_feedback.status == arm_command_pb2.ArmCartesianCommand.Feedback.STATUS_TRAJECTORY_COMPLETE:
            robot.logger.info('Move complete.')
            break
        time.sleep(0.1)

class SpotTablePerception():

    class_names = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]


    def __init__(self):
        # set up mask rcnn
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.model.to(self.device)

        # segmented image publisher
        self.image_publisher = rospy.Publisher("/spot/segment/image", Image, queue_size=1)

        # for keeping track of the current observations
        self.current_point_cloud = None
        self.current_arm_pose = None

        # sdk clients
        image_conn = rbd_spot.SpotSDKConn(sdk_name="StreamImageClient")
        self.image_client = rbd_spot.image.create_client(image_conn)
        state_conn = rbd_spot.SpotSDKConn(sdk_name="ArmPoseClient")
        self.state_client = rbd_spot_robot.state.create_client(state_conn)

        # Check if the sources are valid
        color_source = "hand_color_image"
        depth_source = "hand_depth_in_hand_color_frame"
        sources = [color_source, depth_source]
        self.image_requests = rbd_spot.image.build_image_requests(
            sources, quality=75, fmt="RAW")

        self.point_cloud = o3d.geometry.PointCloud()
        self.transform = np.identity(4)

    def close_robot_and_sit(self):
        # Power the robot off. By specifying "cut_immediately=False", a safe power off command
        # is issued to the robot. This will attempt to sit the robot before powering off.
        self.robot.power_off(cut_immediately=False, timeout_sec=20)
        assert not self.robot.is_powered_on(), "Robot power off failed."
        self.robot.logger.info("Robot safely powered off.")
        # If we successfully acquired a lease, return it.
        self.lease_client.return_lease(self.lease)
        self.lease_keep_alive.shutdown()

    def stand(self):
        self.robot.logger.info("Commanding robot to stand...")
        blocking_stand(self.command_client, timeout_sec=10)
        self.robot.logger.info("Robot standing.")

    def get_lease_and_command_client(self):
        # See hello_spot.py for an explanation of these lines.
        sdk = bosdyn.client.create_standard_sdk('ArmSpotClient')
        self.robot = sdk.create_robot("138.16.161.12")
        self.robot.authenticate("user", "97qp5bwpwf2c")
        self.robot.time_sync.wait_for_sync()

        assert self.robot.has_arm(), "Robot requires an arm to run this example."

        # Verify the robot is not estopped and that an external application has registered and holds
        # an estop endpoint.
        assert not self.robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, " \
                                        "such as the estop SDK example, to configure E-Stop."

        self.lease_client = self.robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
        self.lease = self.lease_client.acquire()
        self.lease_keep_alive = bosdyn.client.lease.LeaseKeepAlive(self.lease_client)

        # Now, we are ready to power on the robot. This call will block until the power
        # is on. Commands would fail if this did not happen. We can also check that the robot is
        # powered at any point.
        self.robot.logger.info("Powering on robot... This may take several seconds.")
        self.robot.power_on(timeout_sec=20)
        assert self.robot.is_powered_on(), "Robot power on failed."
        self.robot.logger.info("Robot powered on.")

        # Tell the robot to stand up. The command service is used to issue commands to a robot.
        # The set of valid commands for a robot depends on hardware configuration. See
        # SpotCommandHelper for more detailed examples on command building. The robot
        # command service requires timesync between the robot and the client.
        self.robot.logger.info("Commanding robot to stand...")
        self.command_client = self.robot.ensure_client(RobotCommandClient.default_service_name)

    def build_arm_command(self, pos, rot, gripper_angle):
        """
        pos: [x, y, z] in odom
        rot: [x, y, z, w] in odom
        gripper_angle: gripper angle in [0, 1]
        """
        # Make the arm pose RobotCommand
        # Build a position to move the arm to (in meters, relative to and expressed in the gravity aligned body frame).
        # hand_ewrt_flat_body = geometry_pb2.Vec3(x=pos[0], y=pos[1], z=pos[2])

        # Rotation as a quaternion
        # odom_Q_hand = geometry_pb2.Quaternion(w=rot[0], x=rot[1], y=rot[2], z=rot[3])

        # odom_T_hand = geometry_pb2.SE3Pose(position=hand_ewrt_flat_body,
        #                                         rotation=odom_Q_hand)

        # duration in seconds
        seconds = 2

        arm_command = RobotCommandBuilder.arm_pose_command(
            pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3], ODOM_FRAME_NAME, seconds)

        # Make the open gripper RobotCommand
        gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(gripper_angle)

        # Combine the arm and gripper commands into one RobotCommand
        return RobotCommandBuilder.build_synchro_command(gripper_command, arm_command)

    def grab_bottle(self):
        # mean = [-4.95875445, -0.74238825,  3.21221821]
        mean, cov = self.point_cloud.compute_mean_and_covariance()
        mean[2] += 0.2
        command = self.build_arm_command(mean, [0, 0, 0, 1], 1.0)
        print(self.lease_client.list_leases())
        cmd_id = self.command_client.robot_command(command)
        self.robot.logger.info('Moving arm to position 2.')

        # Wait until the arm arrives at the goal.
        # Note: here we use the helper function provided by robot_command.
        block_until_arm_arrives(self.command_client, cmd_id)

    def stow_arm(self):
        stow = RobotCommandBuilder.arm_stow_command()

        # Issue the command via the RobotCommandClient
        cmd_id = self.command_client.robot_command(stow)
        self.robot.logger.info('Stowing arm')

        # Wait until the arm arrives at the goal.
        # Note: here we use the helper function provided by robot_command.
        block_until_arm_arrives(self.command_client, cmd_id)

    def deploy_arm(self):
        unstow = RobotCommandBuilder.arm_ready_command()

        # Issue the command via the RobotCommandClient
        cmd_id = self.command_client.robot_command(unstow)
        self.robot.logger.info('Stowing arm')

        # Wait until the arm arrives at the goal.
        # Note: here we use the helper function provided by robot_command.
        block_until_arm_arrives(self.command_client, cmd_id)

    def segment_and_publish(self, msg):
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
        cuda_img = torch_img.cuda(self.device)
        pred  = self.model([cuda_img])
        pred_score = list(pred[0]['scores'].detach().cpu().numpy())
        pred_thresholded = [pred_score.index(x) for x in pred_score if x>0.8]
        found_bottle = False
        if len(pred_thresholded) > 0:
            pred_t = pred_thresholded[-1] if len(pred_thresholded) > 0 else 0
            masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()[:pred_t+1]
            bottle_mask = np.full(masks[0].shape, False)
            pred_class = [self.class_names[i] for i in list(pred[0]['labels'][:pred_t+1].cpu().numpy())]
            for i, mask in enumerate(masks):
                if pred_class[i] == "bottle":
                    found_bottle = True
                    bottle_mask = np.logical_or(bottle_mask, mask)
                    break
            binary_mask = np.where(bottle_mask, np.uint8(255), np.uint8(0))
            if np.sum(binary_mask) > 500:
                depth_img = cv2.bitwise_and(depth_img, depth_img, mask=binary_mask)
        else:
            print("Failed to detect objects")

        if found_bottle:
            intrinsics = msg[1].source.pinhole.intrinsics
            shape = depth_img.shape
            o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
                shape[0], shape[1],
                intrinsics.focal_length.x, intrinsics.focal_length.y,
                intrinsics.principal_point.x, intrinsics.principal_point.y
            )
            o3d_img = o3d.geometry.Image(img)
            print(depth_img.dtype)
            o3d_depth_img = o3d.geometry.Image(depth_img)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_img, o3d_depth_img, convert_rgb_to_intensity = False)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_intrinsics)

            # flip the orientation, so it looks upright, not upside-down
            pcd.transform([[0,0,1,0],[-1,0,0,0],[0,-1,0,0],[0,0,0,1]])
            print(self.transform)
            new_points = np.asarray(pcd.points)
            print(new_points[:10, :])
            pcd.transform(self.transform)
            new_points = np.asarray(pcd.points)
            if new_points.shape[0] > 100:
                old_points = np.asarray(self.point_cloud.points)
                self.point_cloud.points = o3d.utility.Vector3dVector(np.concatenate((old_points, new_points), 0))
                self.point_cloud.random_down_sample(0.5)
                print(self.point_cloud.compute_mean_and_covariance())
                o3d.visualization.draw_geometries([self.point_cloud])  
            
        result_bytes = img.flatten().tobytes()
        print(np.sum(depth_img) / np.sum(np.where(bottle_mask, 1, 0)))
        depth_result_bytes = depth_img.flatten().tobytes()
        color_img_msg.data = result_bytes
        depth_img_msg.data = depth_result_bytes

        self.image_publisher.publish(depth_img_msg)


    def add_new_point_cloud(self):
        pass

    def get_image(self):
        # Stream the image through specified sources
        try:
            result, time_taken = rbd_spot.image.getImage(self.image_client, self.image_requests)
            print(time_taken)
            return result
        finally:
            if rospy.is_shutdown():
                sys.exit(1)

    def get_arm_pose(self):
        state = rbd_spot_robot.state.getRobotState(self.state_client)
        snapshot = state.kinematic_state.transforms_snapshot
        tform = get_a_tform_b(snapshot, "hand", "odom")
        print(tform)
        self.transform = np.linalg.inv(tform.to_matrix())
    
    @staticmethod
    def get_matrix_for_transform(transform):
        rot = transform.rotation
        pos = transform.position
        rot_mat = R.from_quat([rot.x, rot.y, rot.z, rot.w])
        print(rot_mat.as_euler("xyz"), pos)
        translation_vec = np.array([pos.x, pos.y, pos.z])
        return SpotTablePerception.get_matrix_from_rot_trans(rot_mat, translation_vec)

    @staticmethod
    def get_matrix_from_rot_trans(rot, trans):
        mat = np.zeros((4, 4))
        mat[:3, :3] = rot.as_matrix()
        mat[3, :3] = trans
        mat[3, 3] = 1
        return mat


    def run(self):
        rospy.init_node("segment_image")
        print("HERE")
        self.get_lease_and_command_client()
        self.stand()

        try:
            while True:
                self.get_arm_pose()
                msg = self.get_image()
                self.segment_and_publish(msg)
                # self.deploy_arm()
                self.grab_bottle()
                self.stow_arm()
                break
        finally:
            self.close_robot_and_sit()

if __name__ == "__main__":
    obj = SpotTablePerception()
    obj.run()

