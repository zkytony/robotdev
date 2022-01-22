import rospkg
import os

class ActionType:
    EXECUTE = 0
    CANCEL = 1
    # PAUSE = 2

DEBUG_LEVEL = 0

_rospack = rospkg.RosPack()
PKG_PATH = _rospack.get_path('rbd_movo_motor_skills')

def goal_file(name):
    return os.path.join(PKG_PATH, "cfg", "%s.yml" % name)
