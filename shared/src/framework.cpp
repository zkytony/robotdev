#include "motion_planning/framework.h"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <assert.h>

namespace rbd
{

SkillManager::SkillManager() {
    //...
}

void SkillManager::Load(const std::string &skill_file_path) {

    YAML::Node skill_spec = YAML::LoadFile(skill_file_path);
    for (YAML::const_iterator it=skill_spec.begin(); it != skill_spec.end(); ++it) {
        YAML::Node checkpoint = *it;
        std::cout << checkpoint.Type() << std::endl;  // 4 means Map
    }

}



}
