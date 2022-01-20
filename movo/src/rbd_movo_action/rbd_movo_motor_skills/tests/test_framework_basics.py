from rbd_movo_motor_skills.motion_planning.framework import SkillManager

def test():
    skill_file_path = "general.left_clearaway.skill"
    mgr = SkillManager(skill_file_path)
    mgr.run()


if __name__ == "__main__":
    test()
