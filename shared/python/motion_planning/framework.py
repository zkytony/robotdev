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
# Implementation details:
# - a 'cue' is a dictionary with required fields 'type' and 'args'

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
        """
        Args:
            cue (dict): cue a dictionary with required fields 'type' and 'args'
        """
        if type(cue) != dict\
           or "type" not in cue\
           or "args" not in cue:
            raise ValueError("cue must be a dictionary with 'type' and 'args' fields.")
        self._cue = cue


class Executor(SkillWorker):
    """An executor's job is to execute to achieve a goal,
    which is derived from a cue."""
    def __init__(self, cue):
        self.goal = self.make_goal(cue)

    @staticmethod
    def make_goal(cue):
        raise NotImplementedError


class Checkpoint:
    """Describes a checkpoint in a skill where we expect the
    robot to observe certain perception cues and actuation cues.
    Note that there is no ordering among the cues within a checkpoint."""
    def __init__(self, name, perception_cues, actuation_cues):
        """
        Args:
            name (str): name of the checkpoint
            perception_cues (list): list of cues
            actuation_cues (list): list of cues
        """
        self._name = name
        self._perception_cues = perception_cues
        self._actuation_cues = actuation_cues

class Skill:
    """A skill is a list of checkpoints"""
    def __init__(self, checkpoints):
        self._checkpoints = checkpoints

import yaml
class SkillManager:
    """See documentation above."""
    def __init__(self):
        self._current_checkpoint_index = -1
        self._skill = None
        self._workers = set()    # the set of skill workers currently running
        self._config = {}        # the configuration; maps from cue type to (verifier_class, executor_class)

    @property
    def is_initialized(self):
        return self._current_checkpoint_index >= 0

    def load(self, skill_file_path):
        """Loads the skill from the path"""
        # loads the skill file
        with open(skill_file_path) as f:
            spec = yaml.safe_load(f)
            SkillManager._validate_skill_spec(spec)

        # load config
        for cue_type in spec["config"]:
            cs = spec["config"][cue_type]
            self._config[cue_type] = (cs["verifier"], cs["executor"])

        # load skills (checkpoint specs)
        skill = []
        for ckspec in spec["skill"]:
            skill.append(Checkpoint(ckspec['name'],
                                    ckspec.get('perception_cues', []),
                                    ckspec.get('actuation_cues', [])))
        self._skill = skill

    @staticmethod
    def _validate_skill_spec(spec):
        assert "config" in spec, "spec must have 'config'"
        assert "skill" in spec, "spec must have 'skill'"
        for cue_type in spec["config"]:
            assert "verifier" in spec["config"][cue_type],\
                "cue type {} lacks verifier. If one is not needed, do 'verifier: \"NA\"'".format(cue_type)
            assert "executor" in spec["config"][cue_type],\
                "cue type {} lacks executor. If one is not needed, do 'executor: \"NA\"'".format(cue_type)
        for i, ckspec in enumerate(spec["skill"]):
            assert "name" in ckspec, "checkpoint {} has no name".format(i)
            assert "perception_cues" in ckspec\
                or "actuation_cues" in ckspec,\
                "checkpoint {} has neither perception cue nor actuation cue".format(i)
            if "perception_cues" in ckspec:
                assert type(ckspec["perception_cues"]) == list, "perception cues should be a list."
            if "actuation_cues" in ckspec:
                assert type(ckspec["actuation_cues"]) == list, "actuation cues should be a list."

    def init(self):
        """Initializes the skill manager. Will create a roslaunch file,
        and save that at a particular path."""

        pass
