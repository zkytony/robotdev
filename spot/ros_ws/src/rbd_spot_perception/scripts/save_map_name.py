#!/usr/bin/env python
# A super simple node that just saves the map_name
# that is currently in an environment variable to
# a file called .map_name under the given path to
# directory in the argument. This makes other programs
# running in other shells be able to access the
# map name directly without asking for it from the
# user again.
#
# Convention: whether you are running mapping or localization,
# it is desirable to have the current map name stored in the
# .map_name file.
import rospy
import os
import sys

def main():
    rospy.init_node("save_map_name")
    map_name = os.environ['MAP_NAME']
    save_path = os.path.join(sys.argv[1], ".map_name")
    with open(save_path, "w") as f:
        f.write(map_name + "\n")
    rospy.loginfo(f"map name {map_name} saved to {save_path}.")

if __name__ == "__main__":
    main()
