import rospy
from rbd_movo_motor_skills.motion_planning.framework import SkillManager

def test():
    rospy.set_param("skill/pkg_base_dir", "../")
    rospy.set_param("skill/pkg_name", "rbm_movo_motor_skills")
    skill_file_path = "general.left_clearaway.skill"
    mgr = SkillManager(skill_file_path)
    mgr.run()


if __name__ == "__main__":
    test()
