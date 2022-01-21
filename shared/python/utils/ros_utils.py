# ROS Utilities
import tf
import math

def IS_TAG(t):
    return len(t) == 2 or len(t[2]) == 0

class ROSLaunchWriter:
    """A ROS LAUNCH FILE CONSISTS OF
    'blocks'

    A 'tag' is a special kind of 'block' that
    contains nothing in it.

    Internally, we represent a block as:

    (block_type, <stuff> [, blocks])

    where <stuff> is a dictionary that specifies the options to
    the block (XML options).  And optionally, one can put
    'blocks' at the end which will make those blocks the children
    of the block `block_name`.
    """

    def __init__(self):
        self._blocks = []

    def add_tag(self, tag_type, options):
        """Adds a single, unnested <tag ... /> tag.
        `kwargs` are the options.
        For example, arg.

        options is a dictionary that specifies the XML tag options.

        Note that a tag is a special kind of 'block'"""
        self._blocks.append((tag_type, options))

    def add_block(self, block_type, options, blocks):
        """
        Adds a block. A block looks like:

        <block_type ...>
            <block> ... </block>
            .... [stuff in blocks]
        </block>

        Args:
            blocks (list): list of blocks.
        """
        self._blocks.append((block_name, options, blocks))

    def add_blocks(self, blocks):
        self._blocks.extend(blocks)

    @staticmethod
    def make_block(block_type, options, blocks):
        """Same specification as `add_block` except instead of
        adding the block into self._blocks, returns the block."""
        return (block_type, options, blocks)

    @staticmethod
    def make_tag(tag_type, options):
        """Same specification as `add_tag` except instead of
        adding the block into self._blocks, returns the block."""
        return (tag_type, options)

    def _dump_block(self, block, indent_level, indent_size=4):
        block_type = block[0]
        options = block[1]
        block_str = (" "*(indent_level*indent_size)) + "<" + block_type + " "
        for opt_name in options:
            opt_val = options[opt_name]
            block_str += "{}=\"{}\" ".format(opt_name, opt_val)
        if IS_TAG(block):
            block_str += "\>\n"
        else:
            block_str += ">\n"
            for subblock in block[2]:
                block_str += self._dump_block(subblock, indent_level+1)
            block_str += "\n</{}>\n".format(block_type)
        return block_str

    def dump(self, f=None, indent_size=4):
        """Outputs the roslaunch file to given file stream, if provided.
        Otherwise, returns the entire string of the XML file."""
        lines = "<?xml version=\"1.0\"?>\n"
        lines += "<launch>\n"
        for block in self._blocks:
            lines += self._dump_block(block, 0, indent_size=indent_size) + "\n"
        lines += "</launch>"
        if f is not None:
            f.writelines(lines)
        else:
            return "".join(lines)


def pose_to_tuple(pose):
    """
    Given a geometry_msgs/Pose message,
    returns a tuple (x, y, z, qx, qy, qz, qw)
    """
    x = pose.position.x
    y = pose.position.y
    z = pose.position.z
    qx = pose.orientation.x
    qy = pose.orientation.y
    qz = pose.orientation.z
    qw = pose.orientation.w
    return (x, y, z, qx, qy, qz, qw)

def euclidean_dist(p1, p2):
    return math.sqrt(sum([(a - b)** 2 for a, b in zip(p1, p2)]))

def vec_norm(v):
    return math.sqrt(sum(a**2 for a in v))


def quat_diff(q1, q2):
    """returns the quaternion space difference between two
    quaternion q1 and q2"""
    # reference: https://stackoverflow.com/a/22167097/2893053
    # reference: http://wiki.ros.org/tf2/Tutorials/Quaternions#Relative_rotations
    x1, y1, z1, w1 = q1
    q1_inv = (x1, y1, z1, -w1)
    qd = tf.transformations.quaternion_multiply(q2, q1_inv)
    return qd

def quat_diff_angle(q1, q2):
    """returns the angle (radians) between q1 and q2; signed"""
    # reference: https://stackoverflow.com/a/23263233/2893053
    qd = quat_diff(q1, q2)
    return 2*math.atan2(vec_norm(qd[:3]), qd[3])

def quat_diff_angle_relative(q1, q2):
    """returns the angle (radians) between q1 and q2;
    The angle will be the smallest one between the two
    vectors. It is the "intuitive" angular difference.
    Unsigned."""
    # reference: https://stackoverflow.com/a/23263233/2893053
    ad = quat_diff_angle(q1, q2)
    return min(2*math.pi - ad, ad)

def to_degrees(th):
    return th*180 / math.pi
