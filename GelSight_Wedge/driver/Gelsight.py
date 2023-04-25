import yaml
from yaml.loader import SafeLoader
import re

import math
import numpy as np
import socket
import time
from math import pi, sin, cos

from urllib import request

import cv2
import _thread
from threading import Thread
from pose import Pose
from calibration.calibration import calibration
# from .tracking import Tracking
from GelSight_driver import GelSight



# read the xml file
with open("C:/Users/hasna/OneDrive/Documents/Year_3/3rd Year Project/GelSight_Wedge/config/gelsight_wedge_config.yaml") as f:
    config = yaml.load(f, Loader=SafeLoader)

stream = config["stream"]

# get the cropped image shape
cropped_w = stream["markers_w"]*stream["marker_distance"]
cropped_h = stream["markers_h"]*stream["marker_distance"]

selected_corners = stream["selected_corners"]
for i, corner in enumerate(selected_corners):
    selected_corners[i] = re.findall(r"\d+",corner)

selected_corners = np.array([
        [selected_corners[0], selected_corners[1]],
        [selected_corners[2], selected_corners[3]],
        [selected_corners[4], selected_corners[5]],
        [selected_corners[6], selected_corners[7]]], dtype="float32")

IP = stream["ip"]
Size = (cropped_h, cropped_w)
gelsight = GelSight(IP, selected_corners, output_sz=Size)
gelsight.start()

def test():
    while True:
        img = gelsight.stream.image
        # cv2.imshow("test", img)
        # cv2.waitKey(10)
        depth = gelsight.pc.depth
        # ref = gelsight.pc.frame0
        if depth is None:
            continue
        # if ref is None:
        #     continue
        cv2.imshow("frame", depth)

        c = cv2.waitKey(1) & 0xFF
        print(gelsight.pc.pose)
        if c == ord("q"):
            break

if __name__ == "__main__":
    try:
        test()
    finally:
        del gelsight