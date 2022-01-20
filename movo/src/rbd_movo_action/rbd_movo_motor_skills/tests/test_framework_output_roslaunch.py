from rbd_movo_motor_skills.motion_planning.framework import SkillManager
from rbd_movo_motor_skills.utils.ros_utils import ROSLaunchWriter

def test_roslaunch_writer():
    rlw = ROSLaunchWriter()
    _mb = ROSLaunchWriter.make_block
    _mt = ROSLaunchWriter.make_tag
    blocks = [
        _mt("arg", dict(name='camera_info_topic', default="/movo_camera/sd/camera_info")),
        _mt("arg", dict(name='camera_image_topic', default="/movo_camera/sd/image_color_rect")),
        _mt("arg", dict(name='marker_size', default="0.05")),
        _mb("node", dict(pkg="aruco_ros", type="marker_publisher", name="aruco_marker_publisher"),
           [_mt("remap", {'from':"/camera_info", 'to':"$(arg camera_info_topic)"}),
            _mt("remap", {'from':"/image", 'to':"$(arg camera_image_topic)"}),
            _mt("param", {'name':"/image_is_rectified", 'to':"True"})])
    ]
    rlw.add_blocks(blocks)
    print(rlw.dump())


def test():
    mgr = SkillManager()
    skill_file_path = "../cfg/skills/general.left_clearaway.skill"
    mgr.load(skill_file_path)


if __name__ == "__main__":
    test_roslaunch_writer()
    test()
