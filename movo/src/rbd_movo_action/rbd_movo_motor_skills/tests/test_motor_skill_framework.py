from rbd_movo_motor_skills.motion_planning.framework import SkillManager


def test():
    mgr = SkillManager()
    skill_file_path = "../cfg/skills/general.left_clearaway.skill"
    mgr.load(skill_file_path)

if __name__ == "__main__":
    test()
