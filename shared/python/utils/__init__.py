import os
import sys
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
def sinfo(text, bold=False):
    return(bcolors.s(bcolors.Cyan, text, bold=bold))

def sinfo2(text, debug_level=1, bold=False):
    return(bcolors.s(bcolors.LightMagenta, text, bold=bold))

def serror(text):
    return(bcolors.s(bcolors.Red, text))

def swarning(text):
    return(bcolors.s(bcolors.Yellow, text))

def ssuccess(text, bold=False):
    return(bcolors.s(bcolors.Green, text, bold=bold))
