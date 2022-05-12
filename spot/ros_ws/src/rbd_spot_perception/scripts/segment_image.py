#!/usr/bin/env python
# Stream images through Spot
#
# Usage examples:
#
# rosrun rbd_spot_perception stream_image.py

import pdb
from select import select
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
from bosdyn.client.docking import blocking_dock_robot
from bosdyn.api.geometry_pb2 import SE2Velocity, SE2VelocityLimit, Vec2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2

from bosdyn.client.world_object import WorldObjectClient
from bosdyn.api import world_object_pb2

import traceback
import time


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
        # Create robot object with a world object client.
        # Create the world object client.
        fiducial_conn = rbd_spot.SpotSDKConn(sdk_name="FiducialClient")
        self.fiducial_client = fiducial_conn.ensure_client(WorldObjectClient.default_service_name)
        
        # Time sync is necessary so that time-based filter requests can be converted.

        # Check if the sources are valid
        color_source = "hand_color_image"
        depth_source = "hand_depth_in_hand_color_frame"
        sources = [color_source, depth_source]
        self.image_requests = rbd_spot.image.build_image_requests(
            sources, quality=75, fmt="RAW")

        self.point_cloud = o3d.geometry.PointCloud()
        self.transform_mat = np.identity(4)
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
        # self.robot = sdk.create_robot("138.16.161.12")
        self.robot = sdk.create_robot("138.16.161.2")
        # self.robot.authenticate("user", "97qp5bwpwf2c")
        self.robot.authenticate("user", "dungnydsc8su")
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

    def build_arm_command_odom(self, pos, rot, gripper_angle):
        """
        pos: [x, y, z] in odom
        rot: [x, y, z, w] in odom
        gripper_angle: gripper angle in [0, 1]
        """
        # duration in seconds
        seconds = 2

        arm_command = RobotCommandBuilder.arm_pose_command(
            pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3], ODOM_FRAME_NAME, seconds)

        # Make the open gripper RobotCommand
        gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(gripper_angle)

        # Combine the arm and gripper commands into one RobotCommand
        return RobotCommandBuilder.build_synchro_command(gripper_command, arm_command)

    def build_arm_command_no_rot(self, pos, gripper_angle):
        """
        pos: [x, y, z] in odom
        gripper_angle: gripper angle in [0, 1]
        """
        # TODO
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
        command = self.build_arm_command_odom(mean, [0, 0, 0, 1], 1.0)
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
                found_bottle = False
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
            o3d_depth_img = o3d.geometry.Image(depth_img)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_img, o3d_depth_img, convert_rgb_to_intensity = False)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_intrinsics)

            # flip the orientation, so it looks upright, not upside-down
            pcd.transform([[0,0,1,0],[-1,0,0,0],[0,-1,0,0],[0,0,0,1]])
            print(self.transform)
            new_points = np.asarray(pcd.points)
            pcd.transform(self.transform_mat)

            mean, _ = pcd.compute_mean_and_covariance()
            if mean[0] <= 0 or np.linalg.norm(mean > 1):
                found_bottle = False
            else:
                self.add_new_point_cloud(pcd, visualize=True)
                
            
        result_bytes = img.flatten().tobytes()
        print(np.sum(depth_img) / np.sum(np.where(bottle_mask, 1, 0)))
        depth_result_bytes = depth_img.flatten().tobytes()
        color_img_msg.data = result_bytes
        depth_img_msg.data = depth_result_bytes

        return found_bottle


    def add_new_point_cloud(self, pcd, visualize=False):
        new_points = np.asarray(pcd.points)
        old_points = np.asarray(self.point_cloud.points)
        self.point_cloud.points = o3d.utility.Vector3dVector(np.concatenate((old_points, new_points), 0))
        self.point_cloud.random_down_sample(0.5)
        if visualize:
            o3d.visualization.draw_geometries([self.point_cloud])  

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
        self.transform_mat = np.linalg.inv(tform.to_matrix())
        self.transform = tform.inverse()
    
    def dock_robot(self, dock_id):
        blocking_dock_robot(self.robot, dock_id)

    def get_mobility_params(self):
        speed_limit = SE2VelocityLimit(max_vel=SE2Velocity(
            linear=Vec2(x=0.5, y=0.5, angular=1)))
        mobility_params = spot_command_pb2.MobilityParams(
            vel_limit=speed_limit, locomotion_hint=spot_command_pb2.HINT_AUTO)

        return mobility_params

    def go_to_gpe_body(self, goal_pose, heading):
        print(goal_pose)
        self.relative_move(goal_pose[0], goal_pose[1], heading, ODOM_FRAME_NAME, self.command_client, self.state_client)

    def relative_move(self, dx, dy, dyaw, frame_name, robot_command_client, robot_state_client, stairs=False):
        transforms = self.state_client.get_robot_state().kinematic_state.transforms_snapshot

        # Build the transform for where we want the robot to be relative to where the body currently is.
        body_tform_goal = math_helpers.SE2Pose(x=dx, y=dy, angle=dyaw)
        # We do not want to command this goal in body frame because the body will move, thus shifting
        # our goal. Instead, we transform this offset to get the goal position in the output frame
        # (which will be either odom or vision).
        out_tform_body = get_se2_a_tform_b(transforms, frame_name, BODY_FRAME_NAME)
        out_tform_goal = out_tform_body * body_tform_goal

        # Command the robot to go to the goal point in the specified frame. The command will stop at the
        # new position.
        robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=out_tform_goal.x, goal_y=out_tform_goal.y, goal_heading=out_tform_goal.angle,
            frame_name=frame_name, params=RobotCommandBuilder.mobility_params(stair_hint=stairs))
        end_time = 10.0
        cmd_id = robot_command_client.robot_command(lease=None, command=robot_cmd,
                                                    end_time_secs=time.time() + end_time)
        # Wait until the robot has reached the goal.
        feedback = robot_command_client.robot_command_feedback(cmd_id)
        mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback
        traj_feedback = mobility_feedback.se2_trajectory_feedback
        if (traj_feedback.status == traj_feedback.STATUS_AT_GOAL and
                traj_feedback.body_movement_status == traj_feedback.BODY_STATUS_SETTLED):
            print("Arrived at the goal.")
            return True
        time.sleep(5)

    def get_fiducials(self, offset):
        self.fiducial_poses = {}
    
        # Get all fiducial objects (an object of a specific type).
        request_fiducials = [world_object_pb2.WORLD_OBJECT_APRILTAG]
        fiducial_objects = self.fiducial_client.list_world_objects(
        object_type=request_fiducials).world_objects

        fiducials = []

        for fiducial in fiducial_objects:
            fiducial_number = fiducial.name.split("_")[-1]
            fiducial_name = "fiducial_"+str(fiducial_number)

            fiducials.append(fiducial_name)

            print()
            snapshot = fiducial.transforms_snapshot 
            
            fiducial_transform = get_a_tform_b(snapshot, fiducial_name, BODY_FRAME_NAME)
            fiducial_transform_inv = fiducial_transform.inverse()

            fid_origin = np.array(fiducial_transform_inv.transform_point(0, 0, 0))
            goal_pos = np.array(fiducial_transform_inv.transform_point(0, 0, offset))
            x_axis = np.array([1, 0, 0])
            goal_heading_vec = fid_origin - goal_pos
            goal_heading_vec = goal_heading_vec / np.linalg.norm(goal_heading_vec)
            goal_rot = np.arccos(np.dot(x_axis, goal_heading_vec))

            self.fiducial_poses[fiducial_name] = (goal_pos, goal_rot)

        print(self.fiducial_poses)

    def go_to_fid(self, fid_id):
        offsets = [1, 0.75]
        for i in range(2):
            self.get_fiducials(offsets[i])
            try:
                fid = self.fiducial_poses["fiducial_" + str(fid_id)]
            except:
                print("fid not found")
                self.close_robot_and_sit()
                return False

            self.go_to_gpe_body(fid[0], fid[1])

        return True

        
    def run(self):
        rospy.init_node("segment_image")
        print("HERE")
        self.get_lease_and_command_client()
        self.stand()

        self.go_to_fid(523)

        self.get_arm_pose()
        msg = self.get_image()
        if self.segment_and_publish(msg):
            self.grab_bottle()
            self.stow_arm()
        else:
            print("didn't find bottle")
        self.close_robot_and_sit()

        # while True:
        #     msg = self.get_image()
        #     self.segment_and_publish(msg)
        
        # self.dock_robot(521)

if __name__ == "__main__":
    obj = SpotTablePerception()

