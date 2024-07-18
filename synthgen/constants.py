import re
import numpy as np

# Regular expressions
REGEX_TIMESTAMP = "[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}"
REGEX_MKVFILE = re.compile("kinect_([0-9]+)_(" + REGEX_TIMESTAMP + ")\\.mkv")
REGEX_PVTXT = re.compile(".+_pv\\.txt")


# Atracsys
ATRACSYS_SERIAL = 3314649369913484584
ATRACSYS_ID = f"atracsys_{ATRACSYS_SERIAL}_"  # TODO has this a meaning?
CHECKER_BOARD_ID = 102

# Kinect
KINECT_WIDTH_PX = 2048
KINECT_HEIGHT_PX = 1536
KINECT_INTRINSICS = np.array([981.653, 979.243, 1027.85, 774.92])
KINECT_SERIAL_IDS = [
    "000156211512",
    "000385500312",
    "000389100312",
    "000426200312",
    "000434300312",
]

# Simulation workspace
WORKING_VOLUME_CENTER = np.array([0, 0, 1])
