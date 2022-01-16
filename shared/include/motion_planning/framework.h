#ifndef RBD_MOTOR_SKILL_FRAMEWORK_H
#define RBD_MOTOR_SKILL_FRAMEWORK_H

#include <vector>

namespace rbd {

class Checkpoint {
public:
    Checkpoint();
};

typedef std::vector<Checkpoint> Skill;

// A skill is a list of checkpoints
class SkillManager {
public:
    SkillManager();

    Skill skill;  // skill to manage
    int cindex = -1;   // current checkpoint index

    bool is_initialized() const { return this->cindex >= 0; }

    void init();  // initialize
};

}

#endif  // RBD_MOTOR_SKILL_FRAMEWORK_H
