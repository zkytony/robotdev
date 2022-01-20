# ROS Utilities

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
