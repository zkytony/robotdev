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
are not used by the next checkpoint. These node will communicate
with the manager to report progress.

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
#
# - The skill manager will publish the name of its skill and the
#   current checkpoint index under skill/name and skill/checkpoint.
#   Basically 'skill/' is the namespace for this framework.
#
# - Note that for each verifier node, we expect it to
#   publish String messages to topic '<vfr_node_name>/pass'

import yaml
import os
import signal
import rospy
import subprocess

from std_msgs.msg import String

class SkillManager(object):
    """See documentation above."""
    def __init__(self, skill_file_relpath, **kwargs):
        """
        When you call the 'run' method, a node named 'skill_manager' will be run.

        The node publishes to:

         - skill/name (std_msgs/String) name of skill

        Args:
            pkg_base_dir (str): Path to the root directory of the ROS package
                where you host your implementation of the verifier, executor classes.
                The skills should be stored under <pkg_base_dir>/cfg/skills
                The launch files for each skill will be saved under <pkg_base_dir>/launch/skills
        """
        ## `pkg_name` is the name of the package that contains verifier, executor implementations.
        ## `pkg_base_dir` is the path to the root diretory of this package. Both set
        ## from ROS parameter server
        self.pkg_name = None
        self.pkg_base_dir = None
        self._get_params()

        # publishers
        self._pub_skillname = rospy.Publisher("skill/name", String, queue_size=10)
        self._pub_checkpoint = rospy.Publisher("skill/checkpoint", String, queue_size=10)
        # parameters
        self._rate_info = kwargs.get("rate_info", 5)  # default 5hz
        self._rate_verification_check = kwargs.get("rate_verification_check", 10)  # default 10hz

        # Load Skill
        # 'self._config' is the configuration; maps from cue type to (verifier_class, executor_class)
        self._config, self._skill = self.load(skill_file_relpath)
        self._p_workers = {}    # Maps from worker process id (currently running) to work node name

        # A dictionary maps from cue type to True or False, indicating whether a cue is
        # reached; By default, when a checkpoint is completed, this dictionary is empty.
        self._current_checkpoint_status = {}
        # Indicates which index in the checkpoints are we at now
        self._current_checkpoint_index = -1


    @property
    def initialized(self):
        return self._current_checkpoint_index >= 0

    @property
    def skill(self):
        return self._skill

    @property
    def current_checkpoint(self):
        print(self._current_checkpoint_index)
        return self._skill.checkpoints[self._current_checkpoint_index]


    @property
    def dir_skills(self):
        return os.path.join(self.pkg_base_dir, "cfg", "skills")

    def load(self, skill_file_relpath):
        """Loads the skill from the path;
        Returns (config, skill) tuple. Does not alter manager's state.
        Note that it is relative to <pkg_base_dir>/cfg/skills"""
        # loads the skill file
        with open(self._path_to_skill(skill_file_relpath)) as f:
            spec = yaml.safe_load(f)
            SkillManager._validate_skill_spec(spec)

        # load config
        config = {}
        for cue_type in spec["config"]:
            cs = spec["config"][cue_type]
            config[cue_type] = (cs["verifier"], cs["executor"])

        # load skills (checkpoint specs)
        checkpoints = []
        for ckspec in spec["skill"]:
            checkpoints.append(Checkpoint(ckspec['name'],
                                          ckspec.get('perception_cues', []),
                                          ckspec.get('actuation_cues', [])))
        return config, Skill(os.path.basename(skill_file_relpath), checkpoints)


    def run(self):
        """Starts the SkillManager node -> This means
        you want to execute the skill.
        """
        if self._skill is None:
            raise ValueError("No skill. Cannot start manager.")

        rospy.init_node("skill_manager")
        rospy.loginfo("Initialized skill manager for {}".format(self._skill.name))

        # publish skill name
        rospy.Timer(rospy.Duration(1./self._rate_info),
                    lambda event: self._pub_skillname.publish(String(self._skill.name)))
        # publish current checkpoint
        rospy.Timer(rospy.Duration(1./self._rate_info),
                    lambda event: self._pub_checkpoint.publish(
                        String("Working on [{}/{}] {}".format(self._current_checkpoint_index,
                                                              len(self._skill.checkpoints),
                                                              self.current_checkpoint.name))))

        # Starts running the skill - run the verifier and executor
        # for each checkpoint and monitors the progress.
        self._current_checkpoint_index = 0
        while self._current_checkpoint_index < len(self._skill.checkpoints):
            rospy.loginfo("*** CHECKPOINT {} ***".format(self._current_checkpoint_index))
            # Build workers and run them
            checkpoint = self._skill.checkpoints[self._current_checkpoint_index]
            workers = checkpoint.setup(self._config)
            for worker_type, worker_node_executable, worker_node_name, args in workers:
                p = SkillWorker.start(self.pkg_name,
                                      worker_node_executable,
                                      worker_node_name,
                                      args)
                self._p_workers[p] = (worker_type, worker_node_executable)
            for p in self._p_workers:
                p.wait()

            # Now, the workers have started. We just need to wait for the
            # verifiers to all pass.
            rospy.loginfo("Waiting for verifier")
            vfr_node_names = set(w[2] for w in workers if w[0] == "verifier")
            self._wait_for_verifiers(vfr_node_names)

            # stop the workers
            for p in self._p_workers:
                worker_type, worker_node_executable = self._p_workers[p]
                rospy.loginfo("Stopping {} {}".format(worker_type, worker_node_executable))
                SkillWorker.stop(p)

            # reset state for the next checkpoint
            self._current_checkpoint_index += 1
            self._current_checkpoint_status = {}
            self._p_workers = {}

    def _get_params(self):
        def _g(p):
            import socket
            try:
                if rospy.has_param(p):
                    return rospy.get_param(p)
                else:
                    rospy.logerr("Required prarameter '{}' is not provided".format(p))
                    raise ValueError("Required prarameter '{}' is not provided".format(p))
            except socket.error:
                print("Unable to communicate with ROS master. Do you have 'roscore' running?")
                exit()

        self.pkg_base_dir = _g("skill/pkg_base_dir")
        self.pkg_name = _g("skill/pkg_name")

    def _path_to_skill(self, skill_file_relpath):
        return os.path.join(self.dir_skills, skill_file_relpath)

    def _init_dirs(self):
        if not os.path.exists(self.dir_launch):
            os.makedirs(self.dir_launch)

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
                for c in ckspec['perception_cues']:
                    assert c['type'] in cue_types, "cue type {} not in config".format(c)

            if "actuation_cues" in ckspec:
                assert type(ckspec["actuation_cues"]) == list, "actuation cues should be a list."
                for c in ckspec['actuation_cues']:
                    assert c['type'] in cue_types, "cue type {} not in config".format(c)

    def _verify_callback(self, m, args):
        """
        called when receiving a message from verifier. This will update the
        tracking of the status of the verifier that sent this message `m`.

        Args:
            m (std_msgs.Bool or String)
            args: the first element is the verifier's node name this callback corresponds to
        """
        vfr_node_name = args[0]
        assert vfr_node_name in self._current_checkpoint_status,\
            "error: Expecting verifier {}'s status to be tracked.".format(vfr_node_name)
        assert m.data == Verifier.DONE or m.data == Verifier.NOT_DONE,\
            "Unexpected verification messge {} from Verifier {}. Expected {} or {}"\
            .format(m.data, vfr_node_name, Verifier.DONE, Verifier.NOT_DONE)
        self._current_checkpoint_status[verifier.name] = m.data

    def _wait_for_verifiers(self, vfr_node_names):
        """
        Args:
            vfr_node_names (list): list of verifier node names.
                Note that for each verifier node, we expect it to
                publish String messages to topic 'vfr_node_name/pass'
        """
        assert len(self._current_checkpoint_status) == 0,\
            "Bad manager state: status of previous checkpoint is not cleared."
        self._current_checkpoint_status = {
            v: Verifier.NOT_DONE
            for v in vfr_node_names
        }
        # Set up a subscriber to each verifier topic
        for v in vfr_node_names:
            topic = "{}/pass".format(v)
            rospy.Subscriber(topic, String, self._verify_callback, (v,))

        rate = rospy.Rate(self._rate_verification_check)
        while not self._checkpoint_passed():
            for p in self._p_workers:
                if p.poll() is not None:
                    # p has terminated. This is unexpected.
                    self.stop_all_workers()
                    exit()
            rate.sleep()

    def _checkpoint_passed(self):
        return all(self._current_checkpoint_status[v] == Verifier.DONE
                   for v in self._current_checkpoint_status)

    def stop_all_workers(self):
        for p in self._p_workers:
            SkillWorker.stop(p)


class Checkpoint(object):
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
        self.name = name
        self._perception_cues = perception_cues
        self._actuation_cues = actuation_cues

    def setup(self, config):
        """Setup the verifier and executor objects using given config.
        These objects should be set up so that their nodes ready to be run.

        Args:
            config: maps from cue_type to (Verifier, Executor)
        Returns:
            List of (worker_node_executable, args) tuples. Note: We do not
            directly setup the SkillWorker objects here, because
            those objects are intended to be individual nodes, which
            will be created by separate processes (see SkillWorker.start)"""
        workers = []
        node_name_prefixes = set()
        for cue in self._perception_cues:
            verifier_node_executable, executor_node_executable = config[cue['type']]
            name_prefix = "{}_{}".format(cue['type'], self.name.replace(" ", "_").lower())
            if name_prefix in node_name_prefixes:
                raise ValueError("Node prefix {} already exists."\
                                 "Cannot start nodes of the same name."\
                                 .format(name_prefix))
            node_name_prefixes.add(name_prefix)
            if verifier_node_executable != "NA":
                node_name = name_prefix + "_Vfr"
                workers.append(("verifier", verifier_node_executable, node_name, cue))
            if executor_node_executable != "NA":
                node_name = name_prefix + "_Exe"
                workers.append(("executor", executor_node_executable, node_name, cue))
        return workers


class Skill(object):
    """A skill is a list of checkpoints"""
    def __init__(self, name, checkpoints):
        self._name = name
        self.checkpoints = checkpoints
    @property
    def name(self):
        return self._name


class SkillWorker(object):
    """A Skill Worker is a class that can start and stop
    a ROS node."""
    def __init__(self, name):
        self.name = name

    @staticmethod
    def start(pkg, node_executable, node_name, args):
        """
        Args:
            pkg (str): package with the worker's executable
            node_executable (str): the name of the executable node
            node_name (str): the unqiue name of this node. Note that
                multiple nodes of the same executable may run together,
                but they should have different names
            args (dict): dictionary. Will be converted into a yaml string
                to pass into the node.
        """
        return subprocess.Popen(["rosrun",
                                 pkg,
                                 node_executable,
                                 node_name,
                                 yaml.dump(args)])

    @staticmethod
    def stop(p):
        try:
            if p.poll() is None:  # process hasn't terminated yet
                os.kill(p.pid, signal.SIGINT)
        except OSError as e:
            if errno == 3:
                # No such process error. We ignore this.
                pass
            else:
                raise e


class Verifier(SkillWorker):
    """A verifier's job is to verify if a cue is observed.
    The verifier will publish whether a cue is reached to
    a designated topic specific to this verifier.

    The verifier will publish to <name>/pass topic."""
    DONE = True
    NOT_DONE = False
    def __init__(self, name, cue, rate=10):
        """
        Args:
            name (str): Name of the node for this verifier.
            cue (dict): cue a dictionary with required fields 'type' and 'args'
        """
        super(Verifier, self).__init__(name)
        if type(cue) != dict\
           or "type" not in cue\
           or "args" not in cue:
            raise ValueError("cue must be a dictionary with 'type' and 'args' fields.")
        self.cue = cue
        self.status = Verifier.NOT_DONE

        # Initialize the verifier node
        rospy.init_node(self.name)
        rospy.loginfo("Initialized verifier node {}".format(self.name))
        self.pub = rospy.Publisher(self.topic, String, queue_size=10)
        rospy.loginfo("Publishing to {}/pass...".format(self.name))
        rate = rospy.Rate(rate)
        while not rospy.is_shutdown():
            self.status = self._verify()
            self.pub.publish(String(self.status))
            rate.sleep()

    @property
    def topic(self):
        return "{}/pass".format(self.name)

    def _verify(self):
        """TO BE OVERRIDDEN"""
        return Verifier.NOT_DONE


class Executor(SkillWorker):
    """An executor's job is to execute to achieve a goal,
    which is derived from a cue."""
    def __init__(self, name, cue):
        super(Executor, self).__init__(name)
        if type(cue) != dict\
           or "type" not in cue\
           or "args" not in cue:
            raise ValueError("cue must be a dictionary with 'type' and 'args' fields.")
        self.goal = self.make_goal(cue)

    @staticmethod
    def make_goal(cue):
        raise NotImplementedError
