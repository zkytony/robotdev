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

std::ostream &operator <<(std::ostream &os, const SkillTeamTypes &p) {
    os << "(verifier: " << p.verifier << "; executor: " << p.executor << ")";
}


Checkpoint::Checkpoint(const string &name,
                       const std::vector<Cue> &perception_cues,
                       const std::vector<Cue> &actuation_cues)
    : perception_cues_(perception_cues),
      actuation_cues_(actuation_cues),
      name_(name) {}


SkillManager::SkillManager() {}

void SkillManager::Load(const string &skill_file_path) {

    YAML::Node spec = YAML::LoadFile(skill_file_path);

    // Check if the spec is valid (has both config and skill)
    if (!spec["config"] || !spec["skill"]) {
        throw std::runtime_error("Expects both 'config' and 'skill' in the skill file.");
    }

    // First, create a map from cue type to verifier and executor types
    SkillTeamConfig config;   // maps from cue type to SkillTeamTypes (verifier, executor)
    YAML::Node config_spec = spec["config"];
    for (YAML::const_iterator it=config_spec.begin(); it!=config_spec.end(); ++it) {
        string cue_type = it->first.as<string>();

        YAML::Node cue_spec = config_spec[cue_type];
        string verifier_type = cue_spec["verifier"].as<string>();
        string executor_type = cue_spec["executor"].as<string>();
        SkillTeamTypes team_types = {verifier_type, executor_type};
        config[cue_type] = team_types;
    }
    this->config_ = config;

    // Then, parse the skill. Create the checkpoints
    YAML::Node skill_spec = spec["skill"];
    for (YAML::const_iterator it=skill_spec.begin(); it!=skill_spec.end(); ++it) {
        YAML::Node checkpoint_spec = *it;
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
