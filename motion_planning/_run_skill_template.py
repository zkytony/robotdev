#!/usr/bin/env python
# This is a template only. Do not run this directly
# as it will not work.
# Copy the content of this file to the <ros_pkg>/scripts/run_skills.py
# and add the following into your CMakeLists.txt
#
# install(
#   PROGRAMS
#   scripts/run_skill.py
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

import sys
import rospy
import argparse
from framework import SkillManager
from shared.ros_util import *

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
