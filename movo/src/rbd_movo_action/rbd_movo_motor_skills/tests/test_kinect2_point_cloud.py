import rospy
from sensor_msgs.msg import PointCloud2

if __name__ == "__main__":
    print("Waiting for message from kinect")
    rospy.init_node("test_kinect2_pc")
    message = rospy.wait_for_message("/kinect2/sd/points",
                                     PointCloud2,
                                     timeout=15)
    import pdb; pdb.set_trace()
