import rospy
from rbd_movo_motor_skills.motion_planning.framework import SkillManager

def test():
    rospy.set_param("skill/pkg_base_dir", "../")
    rospy.set_param("skill/pkg_name", "rbd_movo_motor_skills")
    skill_file_path = "scili8.livingroom.pickup_bottle_from_chair.skill"
    mgr = SkillManager(skill_file_path)
    mgr.run()


if __name__ == "__main__":
    test()
