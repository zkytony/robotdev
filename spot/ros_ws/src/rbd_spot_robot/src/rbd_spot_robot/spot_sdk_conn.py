import os
from dataclasses import dataclass, field

from bosdyn.client import create_standard_sdk, ResponseError, RpcError

import rospy
import logging
import tf2_ros
from spot_driver.ros_helpers import populateTransformStamped


@dataclass(init=True)
class SpotSDKConn:
    """Establishes connection with Spot SDK and
    creates standard objects every program that
    uses Spot SDK needs such as 'robot'"""
    # Assumes you have successfully run source setup_spot.sh
    sdk_name: str
    hostname: str = os.environ['SPOT_IP']
    username: str = "user"
    password: str = os.environ['SPOT_USER_PASSWORD']
    logto: str = "rosout"

    def __post_init__(self):
        self.logger = logging.getLogger(self.logto)
        try:
            sdk = create_standard_sdk(self.sdk_name)
        except Exception as e:
            self.logger.error("Error creating SDK object: %s", e)
            return

        self.robot = sdk.create_robot(self.hostname)
        try:
            self.robot.authenticate(self.username, self.password)
            self.robot.start_time_sync()
        except RpcError as err:
            self.logger.error("Failed to communicate with robot: %s", err)
            return

    def ensure_client(self, service_name):
        return self.robot.ensure_client(service_name)


# class SpotSDKClientWithTF(SpotSDKClient):
#     def __init__(self, sdk_name, name='client'):
#         super(SpotSDKClientWithTF, self).__init__(sdk_name, name=name)

#         # NOTE: THE FOLLOWING is BORROWED from spot_driver/spot_ros.py
#         self.camera_static_transform_broadcaster = tf2_ros.StaticTransformBroadcaster()
#         # Static transform broadcaster is super simple and just a latched publisher. Every time we add a new static
#         # transform we must republish all static transforms from this source, otherwise the tree will be incomplete.
#         # We keep a list of all the static transforms we already have so they can be republished, and so we can check
#         # which ones we already have
#         self.camera_static_transforms = []

#         # Spot has 2 types of odometries: 'odom' and 'vision'
#         # The former one is kinematic odometry and the second one is a combined odometry of vision and kinematics
#         # These params enables to change which odometry frame is a parent of body frame and to change tf names of each odometry frames.
#         self.mode_parent_odom_tf = rospy.get_param('~mode_parent_odom_tf', 'odom') # 'vision' or 'odom'
#         self.tf_name_kinematic_odom = rospy.get_param('~tf_name_kinematic_odom', 'odom')
#         self.tf_name_raw_kinematic = 'odom'
#         self.tf_name_vision_odom = rospy.get_param('~tf_name_vision_odom', 'vision')
#         self.tf_name_raw_vision = 'vision'
#         if self.mode_parent_odom_tf != self.tf_name_raw_kinematic and self.mode_parent_odom_tf != self.tf_name_raw_vision:
#             rospy.logerr('rosparam \'~mode_parent_odom_tf\' should be \'odom\' or \'vision\'.')
#             return


#     def populate_camera_static_transforms(self, image_data, spot_wrapper):
#         """Check data received from one of the image tasks and use the transform snapshot to extract the camera frame
#         transforms. This is the transforms from body->frontleft->frontleft_fisheye, for example. These transforms
#         never change, but they may be calibrated slightly differently for each robot so we need to generate the
#         transforms at runtime.

#         Args:
#         image_data: Image protobuf data from the wrapper
#         """
#         # We exclude the odometry frames from static transforms since they are not static. We can ignore the body
#         # frame because it is a child of odom or vision depending on the mode_parent_odom_tf, and will be published
#         # by the non-static transform publishing that is done by the state callback
#         excluded_frames = [self.tf_name_vision_odom, self.tf_name_kinematic_odom, "body"]
#         for frame_name in image_data.shot.transforms_snapshot.child_to_parent_edge_map:
#             if frame_name in excluded_frames:
#                 continue
#             parent_frame = image_data.shot.transforms_snapshot.child_to_parent_edge_map.get(frame_name).parent_frame_name
#             existing_transforms = [(transform.header.frame_id, transform.child_frame_id) for transform in self.camera_static_transforms]
#             if (parent_frame, frame_name) in existing_transforms:
#                 # We already extracted this transform
#                 continue

#             transform = image_data.shot.transforms_snapshot.child_to_parent_edge_map.get(frame_name)
#             local_time = spot_wrapper.robotToLocalTime(image_data.shot.acquisition_time)
#             tf_time = rospy.Time(local_time.seconds, local_time.nanos)
#             static_tf = populateTransformStamped(tf_time, transform.parent_frame_name, frame_name,
#                                                  transform.parent_tform_child)
#             self.camera_static_transforms.append(static_tf)
#             self.camera_static_transform_broadcaster.sendTransform(self.camera_static_transforms)
