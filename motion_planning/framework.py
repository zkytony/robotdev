# Motor Skill Planning & Creation framework
# /author: kaiyu zheng

import yaml
import os
import rospy
from collections import deque

class Cue:
    """A Cue specifies information that signals the
    achievement of a checkpoint."""
    ONE_PER_CHECKPOINT=False
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return "cue"

    def __eq__(self, other):
        if isinstance(other, Cue):
            return self.name == other.name
        return False

    def __hash__(self):
        return hash(self.name)


class Goal:
    """A goal is basically executor's notion of cue;
    When it is achieved, the verifier should be able
    to verify the satisfaction of a given cue"""
    def __init__(self, cue):
        self.cue = cue

    def __str__(self):
        return "Goal(%s)" % self.cue

    def __eq__(self, other):
        if isinstance(other, Goal):
            return self.cue == other.cue
        return False

    def __hash__(self):
        return hash(self.cue)


class SkillWorker:
    """Represents a ROS Node that plays the role of a worker
    in our Motor Skill framework"""
    def __init__(self, name):
        pass

    def to_launch_xml(self):
        """Returns an xml string for how to
        run this node."""
        raise NotImplementedError()

    def run(self):
        """Directly run this node in the current process"""
        raise NotImplementedError()


class Executor(SkillWorker):
    """An executor's job is to achieve a given goal,
    and communicate with the Manager."""
    def __init__(self, goal):
        pass


class Verifier(SkillWorker):
    """A verifier's job is to check for a given cue,
    and communicate with the Manager"""
    def __init__(self, cue):
        pass


class SkillManager:
    """A SkillManager manages the execution of a skill.  It is
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
    """
    def __init__(self, config={}):
        """
        Args:
            config (dict): Contains the following fields:
                - 'cue_handlers': a dictionary that maps from
                     each cue type (e.g. ARTagRelativePose) to
                     a tuple of (verifier_class, executor_class)
        """
        self._config = config
        self._skill_name = None
        self._skill_spec = None
        self._checkpoint_index = -1

        # Keeps track of what is running
        self._executors_running = {}
        self._verifiers_running = {}
        self._args = {}   # maps from checkpoint index to

    def load(self, skill_file_path):
        if self._skill_spec is None:
            with open(skill_file_path) as f:
                self._skill_spec = yaml.safe_load(f)
                self._skill_name = self._get_name(skill_file_path)
        else:
            raise ValueError("This skill manager has already loaded a skill."\
                             "You must start a new manager for a different skill.")

    def _get_name(self, skill_file_path):
        return os.path.basename(skill_file_path)

    @property
    def skill_name(self):
        return self._skill_name

    @property
    def checkpoint(self):
        """Returns the checkpoint of concern right now.
        That is, it is the checkpoint the robot wants
        to complete currently."""
        if self._checkpoint_index < 0:
            return None
        else:
            return self._skill_spec[self._checkpoint_index]

    @property
    def is_initialized(self):
        return self._checkpoint_index >= 0

    def init(self):
        """Initializes the manager"""
        if self.is_initialized:
            err_msg = "Skill manager for %s has been initialized." % self.skill_name
            rospy.logerr(err_msg)
            raise ValueError(err_msg)
        # Work on the next checkpoint
        self.WorkOnNextCheckpoint()

    def WorkOnNextCheckpoint(self):
        self._checkpoint_index += 1
        if self.checkpoint is None:
            # No more checkpoint. We are done.
            rospy.loginfo("No more checkpoints. Done.")
            return True

        # Prepare the verifiers and executors
        nodes = self._prepare(self.checkpoint)
        self._launch(nodes)

    def _prepare(self, checkpoint):
        """
        Prepares the verifiers and executors of a given checkpoint.
        Returns a list of skillworkers that can be later launched
        as individual ROS nodes.

        Args:
            checkpoint (dict): dictionary that contains necessary
                information to define the checkpoint's cues.
        Returns:
            list of SkillWorker objects.
        """
        perception_cues = checkpoint.get("perception_cues", [])
        action_cues = checkpoint.get("action_cues", [])
        _cue_handlers = self._config['cue_handlers']

        for cue_spec in perception_cues:
            cue_type = cue_spec['type']
            if cue_type not in _cue_handlers:
                err_msg = "Unable to recognize cue type '%s'" % cue_type
                rospy.logerr(err_msg)
                raise ValueError(err_msg)

            verifier_class, executor_class = _cue_handlers[cue_type]
