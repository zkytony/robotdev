# Copyright (c) 2022 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Tutorial to show how to use Spot's arm.
"""
import argparse
import sys
import time

import cv2
import numpy as np

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import estop_pb2, geometry_pb2, image_pb2, manipulation_api_pb2
from bosdyn.client.estop import EstopClient
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, get_vision_tform_body, math_helpers
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from constrained_manipulation_helper import *
from bosdyn.api import robot_command_pb2
from bosdyn.client import robot_command
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)
from bosdyn.client.robot_state import RobotStateClient

g_image_click = None
g_image_display = None


def verify_estop(robot):
    """Verify the robot is not estopped"""

    client = robot.ensure_client(EstopClient.default_service_name)
    if client.get_status().stop_level != estop_pb2.ESTOP_LEVEL_NONE:
        error_message = "Robot is estopped. Please use an external E-Stop client, such as the" \
        " estop SDK example, to configure E-Stop."
        robot.logger.error(error_message)
        raise Exception(error_message)


def arm_object_grasp(config, sdk, robot, lease_client, robot_state_client, image_client, manipulation_api_client, command_client):
    """A simple example of using the Boston Dynamics API to command Spot's arm."""

    # See hello_spot.py for an explanation of these lines.

    # Take a picture with a camera
    robot.logger.info('Getting an image from: ' + config.image_source)
    image_responses = image_client.get_image_from_sources([config.image_source])

    if len(image_responses) != 1:
        print('Got invalid number of images: ' + str(len(image_responses)))
        print(image_responses)
        assert False

    image = image_responses[0]
    if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
        dtype = np.uint16
    else:
        dtype = np.uint8
    img = np.fromstring(image.shot.image.data, dtype=dtype)
    if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
        img = img.reshape(image.shot.image.rows, image.shot.image.cols)
    else:
        img = cv2.imdecode(img, -1)

    # Show the image to the user and wait for them to click on a pixel
    robot.logger.info('Click on an object to start grasping...')
    image_title = 'Click to grasp'
    cv2.namedWindow(image_title)
    cv2.setMouseCallback(image_title, cv_mouse_callback)

    global g_image_click, g_image_display
    g_image_display = img
    cv2.imshow(image_title, g_image_display)
    while g_image_click is None:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            # Quit
            print('"q" pressed, exiting.')
            exit(0)

    robot.logger.info('Picking object at image location (' + str(g_image_click[0]) + ', ' +
                      str(g_image_click[1]) + ')')

    pick_vec = geometry_pb2.Vec2(x=g_image_click[0], y=g_image_click[1])

    # Build the proto
    grasp = manipulation_api_pb2.PickObjectInImage(
        pixel_xy=pick_vec, transforms_snapshot_for_camera=image.shot.transforms_snapshot,
        frame_name_image_sensor=image.shot.frame_name_image_sensor,
        camera_model=image.source.pinhole)

    # Optionally add a grasp constraint.  This lets you tell the robot you only want top-down grasps or side-on grasps.
    add_grasp_constraint(config, grasp, robot_state_client)

    # Ask the robot to pick up the object
    grasp_request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)

    # Send the request
    cmd_response = manipulation_api_client.manipulation_api_command(
        manipulation_api_request=grasp_request)

    # Get feedback from the robot
    while True:
        feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
            manipulation_cmd_id=cmd_response.manipulation_cmd_id)

        # Send the request
        response = manipulation_api_client.manipulation_api_feedback_command(
            manipulation_api_feedback_request=feedback_request)

        print('Current state: ',
              manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state))

        if response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED or response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED:
            break

        time.sleep(0.25)

    robot.logger.info('Finished grasp.')
    time.sleep(4.0)



def cv_mouse_callback(event, x, y, flags, param):
    global g_image_click, g_image_display
    clone = g_image_display.copy()
    if event == cv2.EVENT_LBUTTONUP:
        g_image_click = (x, y)
    else:
        # Draw some lines on the image.
        #print('mouse', x, y)
        color = (30, 30, 30)
        thickness = 2
        image_title = 'Click to grasp'
        height = clone.shape[0]
        width = clone.shape[1]
        cv2.line(clone, (0, y), (width, y), color, thickness)
        cv2.line(clone, (x, 0), (x, height), color, thickness)
        cv2.imshow(image_title, clone)


def add_grasp_constraint(config, grasp, robot_state_client):
    # There are 3 types of constraints:
    #   1. Vector alignment
    #   2. Full rotation
    #   3. Squeeze grasp
    #
    # You can specify more than one if you want and they will be OR'ed together.

    # For these options, we'll use a vector alignment constraint.
    use_vector_constraint = config.force_top_down_grasp or config.force_horizontal_grasp

    # Specify the frame we're using.
    grasp.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME

    if use_vector_constraint:
        if config.force_top_down_grasp:
            # Add a constraint that requests that the x-axis of the gripper is pointing in the
            # negative-z direction in the vision frame.

            # The axis on the gripper is the x-axis.
            axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)

            # The axis in the vision frame is the negative z-axis
            axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=-1)

        if config.force_horizontal_grasp:
            # Add a constraint that requests that the y-axis of the gripper is pointing in the
            # positive-z direction in the vision frame.  That means that the gripper is constrained to be rolled 90 degrees and pointed at the horizon.

            # The axis on the gripper is the y-axis.
            axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=0, y=1, z=0)

            # The axis in the vision frame is the positive z-axis
            axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=1)

        # Add the vector constraint to our proto.
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(
            axis_on_gripper_ewrt_gripper)
        constraint.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(
            axis_to_align_with_ewrt_vo)

        # We'll take anything within about 10 degrees for top-down or horizontal grasps.
        constraint.vector_alignment_with_tolerance.threshold_radians = 0.17

    elif config.force_45_angle_grasp:
        # Demonstration of a RotationWithTolerance constraint.  This constraint allows you to
        # specify a full orientation you want the hand to be in, along with a threshold.
        #
        # You might want this feature when grasping an object with known geometry and you want to
        # make sure you grasp a specific part of it.
        #
        # Here, since we don't have anything in particular we want to grasp,  we'll specify an
        # orientation that will have the hand aligned with robot and rotated down 45 degrees as an
        # example.

        # First, get the robot's position in the world.
        robot_state = robot_state_client.get_robot_state()
        vision_T_body = get_vision_tform_body(robot_state.kinematic_state.transforms_snapshot)

        # Rotation from the body to our desired grasp.
        body_Q_grasp = math_helpers.Quat.from_pitch(0.785398)  # 45 degrees
        vision_Q_grasp = vision_T_body.rotation * body_Q_grasp

        # Turn into a proto
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.rotation_with_tolerance.rotation_ewrt_frame.CopyFrom(vision_Q_grasp.to_proto())

        # We'll accept anything within +/- 10 degrees
        constraint.rotation_with_tolerance.threshold_radians = 0.17

    elif config.force_squeeze_grasp:
        # Tell the robot to just squeeze on the ground at the given point.
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.squeeze_grasp.SetInParent()


def run_constrained_manipulation(config, sdk, robot, lease_client, robot_state_client, image_client, manipulation_api_client, command_client):
    """A simple example of using the Boston Dynamics API to run a
       constrained manipulation task."""

    print(
        "Start doing constrained manipulation. Make sure Object of interest is grasped before starting."
    )

    # Build constrained manipulation command
    # You can build the task type of interest by using functions
    # defined in constrained_manipulation_helper.py
    # The input to this function is a normalized task velocity in range [-1, 1].
    # The normalized task velocity is scaled as a function of the force limit
    # (See the constrained_manipulation_helper.py for more details)
    # For heavier tasks, consider specifying the force or torque limit as well.
    if (config.task_type == 'crank'):
        command = construct_crank_task(config.task_velocity, force_limit=config.force_limit)
    elif (config.task_type == 'lever'):
        command = construct_lever_task(config.task_velocity, force_limit=config.force_limit,
                                       torque_limit=config.torque_limit)
    elif (config.task_type == 'left_handed_ballvalve'):
        command = construct_left_handed_ballvalve_task(config.task_velocity,
                                                       force_limit=config.force_limit,
                                                       torque_limit=config.torque_limit)
    elif (config.task_type == 'right_handed_ballvalve'):
        command = construct_right_handed_ballvalve_task(config.task_velocity,
                                                        force_limit=config.force_limit,
                                                        torque_limit=config.torque_limit)
    elif (config.task_type == 'cabinet'):
        command = construct_cabinet_task(config.task_velocity, force_limit=config.force_limit)
    elif (config.task_type == 'wheel'):
        command = construct_wheel_task(config.task_velocity, force_limit=config.force_limit)
    elif (config.task_type == 'drawer'):
        command = construct_drawer_task(config.task_velocity, force_limit=config.force_limit)
    elif (config.task_type == 'knob'):
        command = construct_knob_task(config.task_velocity, torque_limit=config.torque_limit)
    else:
        print("Unspecified task type. Exit.")
        return


    # Check to see if robot is powered on.
    assert robot.is_powered_on(), "Robot must be powered on."
    robot.logger.info("Robot powered on.")
    # Check if the gripper is already holding the object.
    is_gripper_holding = robot_state_client.get_robot_state(
    ).manipulator_state.is_gripper_holding_item
    assert is_gripper_holding, "Gripper is not holding the object. If it is, override the gripper state holding."

    command_client = robot.ensure_client(robot_command.RobotCommandClient.default_service_name)

    # Note that the take lease API is used, rather than acquire. Using acquire is typically a
    # better practice, but in this example, a user might want to switch back and forth between
    # using the tablet and using this script. Using take allows for directly hijacking control
    # away from the tablet.
    lease_client.take()
    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        command.full_body_command.constrained_manipulation_request.end_time.CopyFrom(
            robot.time_sync.robot_timestamp_from_local_secs(time.time() + 10))
        command_client.robot_command_async(command)
        time.sleep(15.0)

def open_gripper(options, sdk, robot, lease_client, robot_state_client, image_client, manipulation_api_client, command_client):
     # Close the gripper
    input("here we go!")
    gripper_command = RobotCommandBuilder.claw_gripper_open_command()
    # Send the request
    #command = RobotCommandBuilder.build_synchro_command(gripper_command, arm_command)
    cmd_id = command_client.robot_command(gripper_command)
    robot.logger.info('Moving arm to position 1.')

    # Wait until the arm arrives at the goal.
    block_until_arm_arrives_with_prints(robot, command_client, cmd_id)
    input("i am here yo")


def main(argv):
    """Command line interface."""
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('-i', '--image-source', help='Get image from source',
                        default='frontleft_fisheye_image')
    parser.add_argument('-t', '--force-top-down-grasp',
                        help='Force the robot to use a top-down grasp (vector_alignment demo)',
                        action='store_true')
    parser.add_argument('-f', '--force-horizontal-grasp',
                        help='Force the robot to use a horizontal grasp (vector_alignment demo)',
                        action='store_true')
    parser.add_argument(
        '-r', '--force-45-angle-grasp',
        help='Force the robot to use a 45 degree angled down grasp (rotation_with_tolerance demo)',
        action='store_true')
    parser.add_argument('-s', '--force-squeeze-grasp',
                        help='Force the robot to use a squeeze grasp', action='store_true')

    parser.add_argument(
        '--task-type', help='Specify the task type to manipulate.', default='crank', choices=[
            'lever', 'left_handed_ballvalve', 'right_handed_ballvalve', 'crank', 'wheel', 'cabinet',
            'drawer', 'knob'
        ])
    parser.add_argument('--task-velocity', help='Desired task velocity', type=float, default=0.5)
    parser.add_argument('--force-limit', help='Max force to be applied along task dimensions',
                        type=float, default=40)
    parser.add_argument('--torque-limit', help='Max force to be applied along task dimensions',
                        type=float, default=5.0)
    options = parser.parse_args(argv)

    num = 0
    if options.force_top_down_grasp:
        num += 1
    if options.force_horizontal_grasp:
        num += 1
    if options.force_45_angle_grasp:
        num += 1
    if options.force_squeeze_grasp:
        num += 1

    if num > 1:
        print("Error: cannot force more than one type of grasp.  Choose only one.")
        sys.exit(1)

    try:

        sdk = bosdyn.client.create_standard_sdk('ArmObjectGraspClient')
        robot = sdk.create_robot(options.hostname)

        bosdyn.client.util.setup_logging(options.verbose)
        bosdyn.client.util.authenticate(robot)

        lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
        robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
        image_client = robot.ensure_client(ImageClient.default_service_name)

        manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)

        with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):

            robot.time_sync.wait_for_sync()

            assert robot.has_arm(), "Robot requires an arm to run this example."

            # Verify the robot is not estopped and that an external application has registered and holds
            # an estop endpoint.
            verify_estop(robot)


            # Now, we are ready to power on the robot. This call will block until the power
            # is on. Commands would fail if this did not happen. We can also check that the robot is
            # powered at any point.
            robot.logger.info("Powering on robot... This may take a several seconds.")
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


            robot_state = robot_state_client.get_robot_state()
            print(robot_state.kinematic_state.transforms_snapshot.child_to_parent_edge_map["hand"].parent_tform_child)
            input("open gripper")
            open_gripper(options, sdk, robot, lease_client, robot_state_client, image_client, manipulation_api_client, command_client)
            arm_object_grasp(options, sdk, robot, lease_client, robot_state_client, image_client, manipulation_api_client, command_client)
            run_constrained_manipulation(options, sdk, robot, lease_client, robot_state_client, image_client, manipulation_api_client, command_client)
        return True
    except Exception as exc:  # pylint: disable=broad-except
        logger = bosdyn.client.util.get_logger()
        logger.exception("Threw an exception")
        return False


if __name__ == '__main__':
    if not main(sys.argv[1:]):
        sys.exit(1)
