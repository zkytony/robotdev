from rbd_movo_motor_skills.utils.ros_utils import quat_diff_angle_relative, to_degrees


def test_quat_diff_angle():
    # examples come from https://quaternions.online/
    q1 = [0, 0, 0, 1]
    q2 = [0, 0.707, 0, 0.707]
    q3 = [0, 0.383, 0, 0.924]
    assert round(to_degrees(abs(quat_diff_angle_relative(q1, q2)))) == 90.0
    assert round(to_degrees(abs(quat_diff_angle_relative(q1, q3)))) == 45.0
    q4 = [0, 0.609, 0, -0.793]
    q5 = [-0.158, -0.588, 0.205, 0.766]
    assert round(to_degrees(abs(quat_diff_angle_relative(q4, q5)))) == 30.0


    q6 = [-0.005, 0.609, 0.007, -0.793]
    print(round(to_degrees(abs(quat_diff_angle_relative(q4, q6)))))
    assert round(to_degrees(abs(quat_diff_angle_relative(q4, q6)))) == 1.0

    q7 = [0.729, 0.576, 0.302, 0.214]
    q8 = (0.7243693112024411, 0.5885347916612598, 0.3117590946682811, 0.17810717808093052)#(0.7609753990475726, 0.5970478485791919, 0.2415943524759006, 0.07799023915270577)
    print(round(to_degrees(abs(quat_diff_angle_relative(q7, q8)))))
    assert round(to_degrees(abs(quat_diff_angle_relative(q7, q8)))) < 30.0

    print("pass.")

if __name__ == "__main__":
    test_quat_diff_angle()
