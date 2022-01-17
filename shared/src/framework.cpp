#include "motion_planning/framework.h"
#include <yaml-cpp/yaml.h>

#include <assert.h>
#include <stdexcept>
#include <iostream>
#include <map>
#include <string>

using std::string;

namespace rbd
{

// constant definitions
const string SkillWorker::NA = "NA";  // not applicable

std::ostream &operator <<(std::ostream &os, const skill_team_types_pair &p) {
    os << "(verifier: " << p.verifier << "; executor: " << p.executor << ")";
}


SkillManager::SkillManager() {
    //...
}

void SkillManager::Load(const string &skill_file_path) {

    YAML::Node skill_spec = YAML::LoadFile(skill_file_path);

    // Check if the spec is valid (has both config and skill)
    if (!skill_spec["config"] || !skill_spec["skill"]) {
        throw std::runtime_error("Expects both 'config' and 'skill' in the skill file.");
    }

    // First, create a map from cue type to verifier and executor types
    SkillTeamConfig config;   // maps from cue type to
    YAML::Node config_spec = skill_spec["config"];
    for (YAML::const_iterator it=config_spec.begin(); it != config_spec.end(); ++it) {
        string cue_type = it->first.as<string>();

        YAML::Node cue_spec = config_spec[cue_type];
        string verifier_type = cue_spec["verifier"].as<string>();
        string executor_type = cue_spec["executor"].as<string>();
        SkillTeamTypes team_types = {verifier_type, executor_type};
        config[cue_type] = team_types;
    }
    this->config_ = config;




    // for (YAML::const_iterator it=skill_spec.begin(); it != skill_spec.end(); ++it) {
    //     YAML::Node checkpoint_spec = *it;

    //     if (checkpoint_spec["perception_cues"]) {

    //     }

    //     checkpoint_spec["name"];


    //     std::cout << checkpoint.Type() << std::endl;  // 4 means Map
    // }

}



}
