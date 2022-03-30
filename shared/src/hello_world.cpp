#include "motion_planning/framework.h"
#include <iostream>
#include <string>

// to test this program run (from robotdev/movo):
// rosrun rbd_movo_motor_skills hello_world src/rbd_movo_action/rbd_movo_motor_skills/cfg/skills/general.left_clearaway.skill
int main(int argc, char **argv) {
    rbd::SkillManager mgr;
    std::cout << "Hello world!" << std::endl;
    std::cout << "Attempting to open " << std::string(argv[1]) << std::endl;
    mgr.Load(std::string(argv[1]));
}
