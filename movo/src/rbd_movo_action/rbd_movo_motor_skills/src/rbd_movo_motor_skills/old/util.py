import os
import sys
import math
import rbd_movo_motor_skills.old.common as common
import numpy as np
import copy

# Colors on terminal https://stackoverflow.com/a/287944/2893053
class bcolors:
    # source: https://godoc.org/github.com/whitedevops/colors

    ResetAll = "\033[0m"

    Bold       = "\033[1m"
    Dim        = "\033[2m"
    Underlined = "\033[4m"
    Blink      = "\033[5m"
    Reverse    = "\033[7m"
    Hidden     = "\033[8m"

    ResetBold       = "\033[21m"
    ResetDim        = "\033[22m"
    ResetUnderlined = "\033[24m"
    ResetBlink      = "\033[25m"
    ResetReverse    = "\033[27m"
    ResetHidden     = "\033[28m"

    Default      = "\033[39m"
    Black        = "\033[30m"
    Red          = "\033[31m"
    Green        = "\033[32m"
    Yellow       = "\033[33m"
    Blue         = "\033[34m"
    Magenta      = "\033[35m"
    Cyan         = "\033[36m"
    LightGray    = "\033[37m"
    DarkGray     = "\033[90m"
    LightRed     = "\033[91m"
    LightGreen   = "\033[92m"
    LightYellow  = "\033[93m"
    LightBlue    = "\033[94m"
    LightMagenta = "\033[95m"
    LightCyan    = "\033[96m"
    White        = "\033[97m"

    BackgroundDefault      = "\033[49m"
    BackgroundBlack        = "\033[40m"
    BackgroundRed          = "\033[41m"
    BackgroundGreen        = "\033[42m"
    BackgroundYellow       = "\033[43m"
    BackgroundBlue         = "\033[44m"
    BackgroundMagenta      = "\033[45m"
    BackgroundCyan         = "\033[46m"
    BackgroundLightGray    = "\033[47m"
    BackgroundDarkGray     = "\033[100m"
    BackgroundLightRed     = "\033[101m"
    BackgroundLightGreen   = "\033[102m"
    BackgroundLightYellow  = "\033[103m"
    BackgroundLightBlue    = "\033[104m"
    BackgroundLightMagenta = "\033[105m"
    BackgroundLightCyan    = "\033[106m"
    BackgroundWhite        = "\033[107m"

    DISABLED = False

    @staticmethod
    def s(color, content, bold=False):
        """Returns a string with color when shown on terminal.
        `color` is a constant in `bcolors` class."""
        if bcolors.DISABLED:
            return content
        else:
            bold = bcolors.Bold if bold else ""
            return bold + color + content + bcolors.ResetAll

# String with colors
def info(text, debug_level=0, bold=False):
    if common.DEBUG_LEVEL >= debug_level:
        print(bcolors.s(bcolors.Cyan, text, bold=bold))

def info2(text, debug_level=1, bold=False):
    # Used by default for debugging.
    if common.DEBUG_LEVEL >= debug_level:
        print(bcolors.s(bcolors.LightMagenta, text, bold=bold))

def error(text, debug_level=0):
    if common.DEBUG_LEVEL >= debug_level:
        print(bcolors.s(bcolors.Red, text))

def warning(text, debug_level=0):
    if common.DEBUG_LEVEL >= debug_level:
        print(bcolors.s(bcolors.Yellow, text))

def success(text, debug_level=0, bold=False):
    if common.DEBUG_LEVEL >= debug_level:
        print(bcolors.s(bcolors.Green, text, bold=bold))

def rgb_to_hex(rgb):
    r,g,b = rgb
    return '#%02x%02x%02x' % (int(r), int(g), int(b))

def hex_to_rgb(hx):
    """hx is a string, begins with #. ASSUME len(hx)=7."""
    if len(hx) != 7:
        raise ValueError("Hex must be #------")
    hx = hx[1:]  # omit the '#'
    r = int('0x'+hx[:2], 16)
    g = int('0x'+hx[2:4], 16)
    b = int('0x'+hx[4:6], 16)
    return (r,g,b)

# Printing
def print_banner(text, ch='=', length=78, color=None):
    """Source: http://code.activestate.com/recipes/306863-printing-a-bannertitle-line/"""
    spaced_text = ' %s ' % text
    banner = spaced_text.center(length, ch)
    if color is None:
        print(banner)
    else:
        print(bcolors.s(color, text))


def print_in_box(msgs, ho="=", vr="||", color=None):
    max_len = 0
    for msg in msgs:
        max_len = max(len(msg), max_len)
    if color is None:
        print(ho*(max_len+2*(len(vr)+1)))
    else:
        print(bcolors.s(color, ho*(max_len+2*(len(vr)+1))))
    for msg in msgs:
        if color is None:
            print(vr + " " + msg + " " + vr)
        else:
            print(bcolors.s(color, vr + " " + msg + " " + vr))
    if color is None:
        print(ho*(max_len+2*(len(vr)+1)))
    else:
        print(bcolors.s(color, ho*(max_len+2*(len(vr)+1))))


# Disable
def block_print():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enable_print():
    sys.stdout = sys.__stdout__

# Geometry
def euc_dist(p1, p2):
    return np.sqrt(np.sum((p1-p2)**2))

def pose_close(p1, p2, r=0.005):
    """
    p1 and p2 are geometry_msgs/Pose objects. Returns true
    if their positions are within the given radius.
    """
    dist = math.sqrt((p1.position.x - p2.position.x)**2
                     + (p1.position.y - p2.position.y)**2
                     + (p1.position.z - p2.position.z)**2)
    return dist <= 0.005

# A vector is a tuple or array of two points (start, end), each
# is N-dimensional np.array.
def unit_vector(v):
    """Return a vector that is a unit vector in the direction
    of `v`. Does not change origin."""
    return np.array([v[0], v[0] + (v[1] - v[0]) / np.linalg.norm(v[1] - v[0])])

def point_along(p, v2, t=1):
    """returns the end point of moving from p along the direction
    of v2 for t times the magnitutde of v2."""
    return p + (v2[1] - v2[0]) * t

def intersect(l1, l2):
    """Returns the point of intersection for the given two lines.
    A line is a tuple (p, v) where p = (x,y) and v = (p1, p2), i.e.
    two points."""
    p1, v1 = l1; x1, y1 = p1; dx1, dy1 = v1[1] - v1[0]
    p2, v2 = l2; x2, y2 = p2; dx2, dy2 = v2[1] - v2[0]

    A = np.array([[dx1, -dx2], [dy1, -dy2]])
    b = np.array([x2 - x1, y2 - y1])
    t, k = np.linalg.solve(A, b)
    p_int = point_along(p1, v1, t=t)
    return p_int


# Data
def downsample(arr1d, every=5):#final_len):
    result = []
    if len(arr1d) < every:
        return arr1d

    result.append(arr1d[0])
    for i in range(1, len(arr1d)-1):
        if i % every == 0:
            result.append(arr1d[i])
    result.append(arr1d[-1])

    return result

def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)


# Colors
def n_rand_colors(n, ctype=1):
    colors = []
    while len(colors) < n:
        colors = random_unique_color(colors)
    return colors

def random_unique_color(colors, ctype=1):
    """
    ctype=1: completely random
    ctype=2: red random
    ctype=3: blue random
    ctype=4: green random
    ctype=5: yellow random
    """
    if ctype == 1:
        color = "#%06x" % random.randint(0x444444, 0x999999)
        while color in colors:
            color = "#%06x" % random.randint(0x444444, 0x999999)
    elif ctype == 2:
        color = "#%02x0000" % random.randint(0xAA, 0xFF)
        while color in colors:
            color = "#%02x0000" % random.randint(0xAA, 0xFF)
    elif ctype == 4:  # green
        color = "#00%02x00" % random.randint(0xAA, 0xFF)
        while color in colors:
            color = "#00%02x00" % random.randint(0xAA, 0xFF)
    elif ctype == 3:  # blue
        color = "#0000%02x" % random.randint(0xAA, 0xFF)
        while color in colors:
            color = "#0000%02x" % random.randint(0xAA, 0xFF)
    elif ctype == 5:  # yellow
        h = random.randint(0xAA, 0xFF)
        color = "#%02x%02x00" % (h, h)
        while color in colors:
            h = random.randint(0xAA, 0xFF)
            color = "#%02x%02x00" % (h, h)
    else:
        raise ValueError("Unrecognized color type %s" % (str(ctype)))
    return color

def argsort(seq, key=None):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    if key is None:
        key = seq.__getitem__
    return sorted(range(len(seq)), key=key)
