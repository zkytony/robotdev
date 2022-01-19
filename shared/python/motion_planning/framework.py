"""
Overall framework:

A SkillManager manages the execution of a skill.  It is
given a skill written in a yaml file, where the skill is
specified as a list of "checkpoints". Each checkpoint
specifies the perception cues and actuation cues that
define the goal condition of reaching that checkpoint.
Tolerance can be specified.

The SkillManager is aware of which checkpoint is the next
goal to reach for the robot. Each checkpoint, in yaml,
is defined as a dictionary as follows:

perception_cues:
    - type: ARTagPose
      pose: [x, y, z, qx, qy, qz, qw]
      id: ar_marker_4
      base_frame: base_link

    - type: ARTagsCombo
      tags:
      - pose: [x, y, z, qx, qy, qz, qw]
        id: ar_marker_4
        base_frame: base_link
        type: reference
      - ...

actuation_cues:
    - type: JointPoses
      - pose: [j1 j2 j3 j4 j5 j6 j7]
      - group: left_arm

    - type: EEPose
      - pose [x y z qx qy qz qw]
      - ee_frame: left_ee_link
      - group: left_arm

    ...

Each cue has a type (e.g. ARTagPose, ARTagsCombo) and
custom properties to define that cue. The propoerties
are understood by the corresponding verifier and executor.

Note that order of cues does not matter within a checkpoint.
Also, certain cue types can only appear once in a checkpoint.

A SkillManager parses a skill specification and
internally converts it into a ROS launch file. Then, the
SkillManager starts a subprocess to run that launch file
which effectively starts the process of completing the skill.

When the SkillManager is "managing" a checkpoint, i.e. it is
supervising the progress towards a checkpoint, SkillWorkers,
either verifier or executor nodes will be spawned. After the
checkpoint is completed, these nodes will be killed, if they
are not used by the next checkpoint. For one cue type, only
one node for the corresponding verifier and executor will be
created. This node will communicate with the manager who will
provide the correct cue to watch out for or the correct goal
to execute towards.

Terminology

| ...
| previous_checkpints (completed)
| ...
| checkpoint (doing)
| ...
| future_checkpoints (todo)
| ...

If a function name is CamelCase, then this function
will interact with other nodes (e.g. launch them).

/author: Kaiyu Zheng
"""

class Cue:
    def __init__(self, name):
        self._name = name

class Goal:
    def __init__(self, cue):
        self._cue = cue

class SkillWorker:
    """A Skill Worker is a class that can start and stop
    a ROS node."""
    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

class Verifier(SkillWorker):
    """A verifier's job is to verify if a cue is observed."""
    def __init__(self, cue):
        self._cue = cue

class Executor(SkillWorker):
    """An executor's job is to execute to achieve a goal,
    which is derived from a cue."""
    def __init__(self, goal):
        self._goal = goal

class Checkpoint:
    """Describes a checkpoint in a skill where we expect the
    robot to observe certain perception cues and actuation cues.
    Note that there is no ordering among the cues within a checkpoint."""
    def __init__(self, name, perception_cues, actuation_cues):
        self._name = name
        self._perception_cues = perception_cues
        self._actuation_cues = actuation_cues

class Skill:
    """A skill is a list of checkpoints"""
    def __init__(self, checkpoints):
        self._checkpoints = checkpoints

class SkillManager:
    """See documentation above."""
    def __init__(self):
        self._current_checkpoint_index = -1
        self._skill = None
        self._workers = set()    # the set of skill workers currently running
        self._config = {}        # the configuration; maps from cue type to (verifier_class, executor_class)

    def load(self, skill_file_path):
        pass

    def init(self):
        pass

    @property
    def is_initialized(self):
        return self._current_checkpoint_index >= 0
