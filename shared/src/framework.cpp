#include "motion_planning/framework.h"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <assert.h>
#include <stdexcept>

namespace rbd
{

SkillManager::SkillManager() {
    //...
}

void SkillManager::Load(const std::string &skill_file_path) {

    YAML::Node skill_spec = YAML::LoadFile(skill_file_path);

    // First, create a map from cue type to verifier and executor types
    if (!skill_spec["config"] || !skill_spec["skill"]) {
        throw std::runtime_error("Expects both 'config' and 'skill' in the skill file.");
    }



    // for (YAML::const_iterator it=skill_spec.begin(); it != skill_spec.end(); ++it) {
    //     YAML::Node checkpoint_spec = *it;

    //     if (checkpoint_spec["perception_cues"]) {

    //     }

    //     checkpoint_spec["name"];


    //     std::cout << checkpoint.Type() << std::endl;  // 4 means Map
    // }

}



}
