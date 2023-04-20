import rospy
from geometry_msgs.msg import Pose, PoseStamped
from tf.transformations import euler_from_quaternion, quaternion_from_euler

import tf2_ros
import tf2_geometry_msgs  # **Do not use geometry_msgs. Use this instead for PoseStamped
import tf



def move_to_fiducial():

    rospy.init_node('move_to_fiducial', anonymous=True)

    listener = tf.TransformListener()
    fiducial_frame = "persistant_fiducial_523"

    rospy.sleep(1)

    (trans,rot) = listener.lookupTransform('base_link', fiducial_frame, rospy.Time(0))
    print(trans,rot)

    pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)


    rospy.sleep(1)

    goal = PoseStamped()

    goal.header.seq = 10
    goal.header.stamp = rospy.Time.now()
    goal.header.frame_id = "base_link"

    goal.pose.position.x = -2.3244099617004395
    goal.pose.position.y = -0.6164484024047852
    goal.pose.position.z = 0

    goal.pose.orientation.x = 0.0
    goal.pose.orientation.y = 0.0
    goal.pose.orientation.z = 0.9080171092073034
    goal.pose.orientation.w = 0.4189330846171164

    #pub.publish(goal)

def transform_pose(input_pose, from_frame, to_frame):

    # Test Case
    rospy.init_node("transform_test")

    # **Assuming /tf2 topic is being broadcasted
    tf_buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_buffer)

    #############
    my_listener = tf.TransformListener()
    rospy.sleep(1)
    (trans,rot) = my_listener.lookupTransform(to_frame, from_frame, rospy.Time(0))
    print(from_frame + " poses in frame of " + to_frame)
    print(trans,rot)
    print("\n")
    ##############

    pose_stamped = tf2_geometry_msgs.PoseStamped()
    pose_stamped.pose = input_pose
    pose_stamped.header.frame_id = from_frame
    pose_stamped.header.stamp = rospy.Time(0)

    try:
        # ** It is important to wait for the listener to start listening. Hence the rospy.Duration(1)
        output_pose_stamped = tf_buffer.transform(pose_stamped, to_frame, rospy.Duration(1))

        #output_pose_stamped.pose.position.x = -2.1020445823669434
        #output_pose_stamped.pose.position.y = -0.972100555896759
        #output_pose_stamped.pose.position.z = 0

        #output_pose_stamped.pose.position.x = -1.2914777994155884
        #output_pose_stamped.pose.position.y = -1.98142671585083
        output_pose_stamped.pose.position.z = 0

        print("approach raw pose in frame of " + to_frame)
        print(output_pose_stamped.pose)
        eulers = list(euler_from_quaternion([output_pose_stamped.pose.orientation.x,output_pose_stamped.pose.orientation.y,output_pose_stamped.pose.orientation.z,output_pose_stamped.pose.orientation.w]))
        #print(eulers)
        corrected_eulers = eulers
        corrected_eulers[0] = 0
        corrected_eulers[1] = 0
        #print(corrected_eulers)
        corrected_quaternion = quaternion_from_euler(corrected_eulers[0],corrected_eulers[1],corrected_eulers[2])
        #print("correct quaternion")
        #print(corrected_quaternion)

        output_pose_stamped.pose.orientation.x = corrected_quaternion[0]
        output_pose_stamped.pose.orientation.y = corrected_quaternion[1]
        output_pose_stamped.pose.orientation.z = corrected_quaternion[2]
        output_pose_stamped.pose.orientation.w = corrected_quaternion[3]

        print("correct pose:")
        print(output_pose_stamped.pose)
        
        pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        rospy.sleep(1)
        pub.publish(output_pose_stamped)


        br = tf.TransformBroadcaster()
        for i in range(5):
            br.sendTransform((output_pose_stamped.pose.position.x, output_pose_stamped.pose.position.y, output_pose_stamped.pose.position.z),
                 (output_pose_stamped.pose.orientation.x,output_pose_stamped.pose.orientation.y,output_pose_stamped.pose.orientation.z,output_pose_stamped.pose.orientation.w),
                 rospy.Time.now(),
                 "approach_location",
                 'map')
            rospy.sleep(1)
        
        return output_pose_stamped

    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        raise


if __name__ == '__main__':
    try:
        #move_to_fiducial()
            
        my_pose = Pose()
        my_pose.position.x = 0
        my_pose.position.y = 0
        my_pose.position.z = 1
        my_pose.orientation.x = 0
        my_pose.orientation.y = 0
        my_pose.orientation.z = 0
        my_pose.orientation.w = 1

        transformed_pose = transform_pose(my_pose, "fiducial_523", "map")
        #print(transformed_pose)

    except rospy.ROSInterruptException:
        pass