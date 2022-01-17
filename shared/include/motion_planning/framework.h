#ifndef RBD_MOTOR_SKILL_FRAMEWORK_H
#define RBD_MOTOR_SKILL_FRAMEWORK_H

/**
  *  A SkillManager manages the execution of a skill.  It is
  *  given a skill written in a yaml file, where the skill is
  *  specified as a list of "checkpoints". Each checkpoint
  *  specifies the perception cues and actuation cues that
  *  define the goal condition of reaching that checkpoint.
  *  Tolerance can be specified.
  *
  *  The SkillManager is aware of which checkpoint is the next
  *  goal to reach for the robot. Each checkpoint, in yaml,
  *  is defined as a dictionary as follows:
  *
  *  perception_cues:
  *      - type: ARTagPose
  *        pose: [x, y, z, qx, qy, qz, qw]
  *        id: ar_marker_4
  *        base_frame: base_link
  *
  *      - type: ARTagsCombo
  *        tags:
  *        - pose: [x, y, z, qx, qy, qz, qw]
  *          id: ar_marker_4
  *          base_frame: base_link
  *          type: reference
  *        - ...
  *
  *  actuation_cues:
  *      - type: JointPoses
  *        - pose: [j1 j2 j3 j4 j5 j6 j7]
  *        - group: left_arm
  *
  *      - type: EEPose
  *        - pose [x y z qx qy qz qw]
  *        - ee_frame: left_ee_link
  *        - group: left_arm
  *
  *      ...
  *
  *  Each cue has a type (e.g. ARTagPose, ARTagsCombo) and
  *  custom properties to define that cue. The propoerties
  *  are understood by the corresponding verifier and executor.
  *
  *  Note that order of cues does not matter within a checkpoint.
  *  Also, certain cue types can only appear once in a checkpoint.
  *
  *  A SkillManager parses a skill specification and
  *  internally converts it into a ROS launch file. Then, the
  *  SkillManager starts a subprocess to run that launch file
  *  which effectively starts the process of completing the skill.
  *
  *  When the SkillManager is "managing" a checkpoint, i.e. it is
  *  supervising the progress towards a checkpoint, SkillWorkers,
  *  either verifier or executor nodes will be spawned. After the
  *  checkpoint is completed, these nodes will be killed, if they
  *  are not used by the next checkpoint. For one cue type, only
  *  one node for the corresponding verifier and executor will be
  *  created. This node will communicate with the manager who will
  *  provide the correct cue to watch out for or the correct goal
  *  to execute towards.
  *
  *  Terminology
  *
  *  | ...
  *  | previous_checkpints (completed)
  *  | ...
  *  | checkpoint (doing)
  *  | ...
  *  | future_checkpoints (todo)
  *  | ...
  *
  *  If a function name is CamelCase, then this function
  *  will interact with other nodes (e.g. launch them).
  */

#include <vector>
#include <string>

namespace rbd {

enum class SkillWorkerStatus { kStarted, kWorking, kStopped, kError };

class Cue {
public:
    Cue();
    Cue(std::string &n):name(n) {}
    const std::string name;
};

class Goal {
public:
    Goal();
    // Creates a goal for achieving cue
    Goal(Cue c):cue(c) {};
    const Cue cue;
};


class SkillWorker {
public:
    void Start();
    void Stop();
};


class Verifier : SkillWorker {
public:
    Verifier();
    Verifier(Cue cue);
private:
    Cue cue_;
};


class Executor : SkillWorker {
public:
    Executor();
    Executor(Goal goal);
private:
    Goal goal_;
};


class Checkpoint {
public:
    Checkpoint();
};


// A skill is a list of checkpoints
typedef std::vector<Checkpoint> Skill;

class SkillManager {
public:
    SkillManager();

    Skill skill;  // skill to manage
    int cindex = -1;   // current checkpoint index

    bool isInitialized() const { return this->cindex >= 0; }

    void Init();  // initialize

    void Load(std::string &skill_file_path);

private:
    std::vector<Cue> cues_;
    std::vector<SkillWorker> workers_;

};

}

#endif  // RBD_MOTOR_SKILL_FRAMEWORK_H
