# motor skill planning & creation framework
# /author: kaiyu zheng

import yaml

class Cue:
    """A Cue specifies information that signals the
    achievement of a checkpoint."""
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


class Executor:
    """An executor's job is to achieve a given goal,
    and communicate with the Manager."""
    def __init__(self, goal):
        pass


class Verifier:
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

    A SkillManager parses a skill specification and
    internally converts it into a ROS launch file. Then, the
    SkillManager starts a subprocess to run that launch file
    which effectively starts the process of completing the skill.
    """
    def __init__(self):
        pass

    def loads(self):
        pass

    def init(self):
        pass
