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
# Implementation details / assumptions:
# - a 'cue' is a dictionary with required fields 'type' and 'args'.
#   Optionally, 'name'
#
# - there could be multiple cues with the same type, but different
#   verifier for each. So each verifier has a name. Same for executor

import yaml
import os
import rospy

from std_msgs.msg import String


class SkillWorker:
    """A Skill Worker is a class that can start and stop
    a ROS node."""
    def __init__(self):
        self.running = False

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

class Verifier(SkillWorker):
    """A verifier's job is to verify if a cue is observed.
    The verifier will publish whether a cue is reached to
    a designated topic specific to this verifier."""
    DONE = True
    NOT_DONE = False
    def __init__(self, name, cue):
        """
        Args:
            cue (dict): cue a dictionary with required fields 'type' and 'args'
        """
        super().__init__()
        if type(cue) != dict\
           or "type" not in cue\
           or "args" not in cue:
            raise ValueError("cue must be a dictionary with 'type' and 'args' fields.")
        self.cue = cue
        self.name = name

    @property
    def topic(self):
        raise NotImplementedError()

    @property
    def always_check(self):
        """If True, then the verifier will always be checking
        during the execution of the skill. Otherwise, the verifier
        will stop checking once it has successfully verified once.

        Default False; Override this function by your child class
        if necessary."""
        return False



class Executor(SkillWorker):
    """An executor's job is to execute to achieve a goal,
    which is derived from a cue."""
    def __init__(self, name, cue):
        super().__init__()
        if type(cue) != dict\
           or "type" not in cue\
           or "args" not in cue:
            raise ValueError("cue must be a dictionary with 'type' and 'args' fields.")
        self.goal = self.make_goal(cue)
        self.name = name

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

        Note that a cue is a dictionary with required fields 'type' and 'args'
        """
        self._name = name
        self._perception_cues = perception_cues
        self._actuation_cues = actuation_cues

    def setup(self, config):
        """Setup the verifier and executor objects using given config.
        These objects should be set up so that their nodes ready to be run.

        Args:
            config: maps from cue_type to (Verifier, Executor)
        Returns:
            List of SkillWorker objects """
        workers = []
        for cue in self._perception_cues:
            verifier_class, executor_class = config[cue['type']]
            name_prefix = self._name.replace(" ", "_").lower()
            if verifier_class != "NA":
                workers.append(verifier_class(name_prefix + "_Vrf", cue))
            if executor_class != "NA":
                workers.append(executor_class(name_prefix + "_Exe", cue))
        return workers


class Skill:
    """A skill is a list of checkpoints"""
    def __init__(self, name, checkpoints):
        self._name = name
        self.checkpoints = checkpoints
    @property
    def name(self):
        return self._name


class SkillManager:
    """See documentation above."""
    def __init__(self, pkg_base_dir, **kwargs):
        """
        When you call the 'start' method, a node named 'skill_manager' will be run.

        The node publishes to:

         - skill/name (std_msgs/String) name of skill

        Args:
            pkg_base_dir (str): Path to the root directory of the ROS package
                where you host your implementation of the verifier, executor classes.
                The skills should be stored under <pkg_base_dir>/cfg/skills
                The launch files for each skill will be saved under <pkg_base_dir>/launch/skills
        """
        self.pkg_base_dir = pkg_base_dir
        self._skill = None
        self._config = {}        # the configuration; maps from cue type to (verifier_class, executor_class)
        self._workers = set()    # the set of skill workers currently running

        # A dictionary maps from cue type to True or False, indicating whether a cue is
        # reached; By default, when a checkpoint is completed, this dictionary is empty.
        self._current_checkpoint_status = {}
        # Indicates which index in the checkpoints are we at now
        self._current_checkpoint_index = -1

        # publishers
        self._pub_skillname = rospy.Publisher("skill/name", String, queue_size=10)
        # parameters
        self._rate_skillname = kwargs.get("rate_skillname", 2)  # default 2hz
        self._rate_verification_check = kwargs.get("rate_verification_check", 10)  # default 10hz

    @property
    def skill(self):
        return self._skill

    @property
    def dir_skills(self):
        return os.path.join(self.pkg_base_dir, "cfg", "skills")

    def _path_to_skill(self, skill_file_relpath):
        return os.path.join(self.dir_skills, skill_file_relpath)

    def _init_dirs(self):
        if not os.path.exists(self.dir_launch):
            os.makedirs(self.dir_launch)

    @property
    def initialized(self):
        return self._current_checkpoint_index >= 0

    def load(self, skill_file_relpath):
        """Loads the skill from the path;
        Note that it is relative to <pkg_base_dir>/cfg/skills"""
        # loads the skill file
        with open(self._path_to_skill(skill_file_relpath)) as f:
            spec = yaml.safe_load(f)
            SkillManager._validate_skill_spec(spec)

        # load config
        for cue_type in spec["config"]:
            cs = spec["config"][cue_type]
            self._config[cue_type] = (cs["verifier"], cs["executor"])

        # load skills (checkpoint specs)
        checkpoints = []
        for ckspec in spec["skill"]:
            checkpoints.append(Checkpoint(ckspec['name'],
                                          ckspec.get('perception_cues', []),
                                          ckspec.get('actuation_cues', [])))
        self._skill = Skill(os.path.basename(skill_file_relpath),
                            checkpoints)

    @staticmethod
    def _validate_skill_spec(spec):
        assert "config" in spec, "spec must have 'config'"
        assert "skill" in spec, "spec must have 'skill'"
        cue_types = set()
        for cue_type in spec["config"]:
            assert "verifier" in spec["config"][cue_type],\
                "cue type {} lacks verifier. If one is not needed, do 'verifier: \"NA\"'".format(cue_type)
            assert "executor" in spec["config"][cue_type],\
                "cue type {} lacks executor. If one is not needed, do 'executor: \"NA\"'".format(cue_type)
            cue_types.add(cue_type)
        for i, ckspec in enumerate(spec["skill"]):
            assert "name" in ckspec, "checkpoint {} has no name".format(i)
            assert "perception_cues" in ckspec\
                or "actuation_cues" in ckspec,\
                "checkpoint {} has neither perception cue nor actuation cue".format(i)
            if "perception_cues" in ckspec:
                assert type(ckspec["perception_cues"]) == list, "perception cues should be a list."
                for c in ckcspec['perception_cues']:
                    assert c in cue_types, "cue type {} not in config".format(c)

            if "actuation_cues" in ckspec:
                assert type(ckspec["actuation_cues"]) == list, "actuation cues should be a list."
                for c in ckcspec['actuation_cues']:
                    assert c in cue_types, "cue type {} not in config".format(c)


    def start(self):
        """Starts the SkillManager node -> This means
        you want to execute the skill.
        """
        rospy.loginfo("Starting skill manager for {}".format(self._skill.name))
        rospy.init_node("skill_manager")

        # publish skill name
        rospy.Timer(rospy.Duration(1./self._rate_skillname),
                    lambda event: self._pub_skillname.publish(String(self._skill.name)))

        self._current_checkpoint_index = 0
        while self._current_checkpoint_index < len(self._skill.checkpoints):
            rospy.loginfo("*** CHECKPOINT {} ***".format(self._current_checkpoint_index))
            # Build workers and run them
            checkpoint = self._skill.checkpoints[self._current_checkpoint_index]
            workers = checkpoint.setup(self._config)
            for w in workers:
                w.start()
                self._workers.add(w)

            # Now, the workers have started. We just need to wait for the
            # verifiers to all pass.
            rospy.loginfo("Waiting for verifier")
            verifiers = set(w for w in workers if isinstance(w, Verifier))
            self._wait_for_verifiers(verifiers)

            # stop the workers
            for w in workers:
                w.stop()

            # reset state for the next checkpoint
            self._current_checkpoint_index += 1
            self._current_checkpoint_status = {}


    def _checkpoint_passed(self):
        return all(self._current_checkpoint_status[v] == Verifier.DONE
                   for v in self._current_checkpoint_status)

    def _verify_callback(self, m, args):
        """
        called when receiving a message from verifier
        Args:
            m (std_msgs.Bool or String)
            args: the first element is the verifier this callback corresponds to
        """
        verifier = args[0]
        assert verifier.name in self._current_checkpoint_status,\
            "error: Expecting verifier {}'s status to be tracked.".format(verifier.name)
        assert m.data == Verifier.DONE or m.data == Verifier.NOT_DONE,\
            "Unexpected verification messge {} from Verifier {}. Expected {} or {}"\
            .format(m.data, verifier.name, Verifier.DONE, Verifier.NOT_DONE)
        if verifier.always_check:
            self._current_checkpoint_status[verifier.name] = m.data
        else:
            if m.data == Verifier.DONE:
                self._current_checkpoint_status[verifier.name] = Verifier.DONE

    def _wait_for_verifiers(self, verifiers):
        assert len(self._current_checkpoint_status) == 0,\
            "Bad manager state: status of previous checkpoint is not cleared."
        self._current_checkpoint_status = {
            v.name: Verifier.NOT_DONE
            for v in verifiers
        }
        # Set up a subscriber to each verifier topic
        for v in verifiers:
            rospy.Subscriber(v.topic, String, self._verify_callback, (v,))

        rate = rospy.Rate(self._rate_verification_check)
        while not self._verification_passed():
            rate.sleep()
