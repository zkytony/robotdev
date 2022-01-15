#!/usr/bin/env python

import sys
import rospy
import argparse
from rbd_movo_motor_skills.motion_planning.framework import SkillManager
from rbd_movo_motor_skills.shared.ros_util import *

def main():
    parser = argparse.ArgumentParser(description="Run Skill.")
    parser.add_argument("skill_file_path", type=str,
                        help="path to .skill file")
    args = parser.parse_args()

    skillmgr = SkillManager()
    skillmgr.load(args.skill_file_path)
    rospy.init_node("run_skill_%s" % skillmgr.skill_name)

    rate = 1
    while not rospy.is_shutdown():
        rospy.loginfo("Running skill node for %s" % skillmgr.skill_name)
        rospy.sleep(1/rate)


if __name__ == "__main__":
    main()
