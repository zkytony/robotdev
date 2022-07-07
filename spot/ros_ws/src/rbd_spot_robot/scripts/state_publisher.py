#!/usr/bin/env python
#
# Gets robot state and publishes as tf transform
import rospy
from tf2_ros import TFMessage
from sensor_msgs.msg import JointState
import argparse
import rbd_spot

def main():
    parser = argparse.ArgumentParser(description="Publish Spot RobotState as TF messages")
    parser.add_argument("--root-frame", type=str, help="The name of the root frame of the TF tree",
                        default="body")
    args, _ = parser.parse_known_args()

    rospy.init_node("spot_state_tf_publisher")
    conn = rbd_spot.SpotSDKConn(sdk_name="StateTFPublisher")
    robot_state_client = rbd_spot.state.create_client(conn)
    tf_pub = rospy.Publisher('/tf', TFMessage, queue_size=10)
    js_pub = rospy.Publisher('/joint_states', JointState, queue_size=10)

    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        state = rbd_spot.state.getRobotState(robot_state_client)
        tf_msg = rbd_spot.state.get_tf_from_state(state, conn, args.root_frame)
        js_msg = rbd_spot.state.get_joint_state_from_state(state, conn)

        tf_pub.publish(tf_msg)
        js_pub.publish(js_msg)
        print("published")
        rate.sleep()

if __name__ == "__main__":
    main()
