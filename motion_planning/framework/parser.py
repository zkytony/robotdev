class SkillParser:
    """A skill is specified as a yaml file as a list
    of "checkpoints". Each checkpoint specifies the
    perception cues and actuation cues that define
    the goal condition of reaching that checkpoint.
    Tolerance can be specified.

    Example:

    - perception:
       ...

      actuation:
        ...

    Each cue has type and a corresponding 'verifier'.
    For example, one cue type could be ARTagCue, JointCue, and
    the corresponding resolver is ARTagVerifie, JointVerifierr. A
    verifier is a node that subscribes to certain topics and
    verifies if the specified cue is attained/accepted. If so,
    the verifier will publish to a designated topic to indicate
    success.

    **Note that for performance, for the same cue type, only
    one node for the verifiers will be created.**

    A SkillParser parses a skill specification and converts it
    into a ROS launch file which can be used to launch necessary
    cue verifiers.
    """
