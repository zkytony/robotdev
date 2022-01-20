from rbd_movo_motor_skills.motion_planning.framework import SkillManager

def test():
    mgr = SkillManager("../")
    skill_file_path = "general.left_clearaway.skill"
    mgr.load(skill_file_path)
    mgr.start()


if __name__ == "__main__":
    test()
