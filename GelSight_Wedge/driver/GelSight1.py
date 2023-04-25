import yaml
from yaml.loader import SafeLoader
import re

import math
import numpy as np
import socket
import time
from math import pi, sin, cos

import cv2
import _thread
from threading import Thread
from pose import Pose

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

Size = (640, 480)  # Update to match resolution of C310 camera
cap = cv2.VideoCapture(0)

def test():
    while True:
        ret, img = cap.read()
        if not ret:
            continue

        cv2.imshow("frame", img)

        c = cv2.waitKey(1) & 0xFF
        if c == ord("q"):
            break

if __name__ == "__main__":
    try:
        test()
    finally:
        cap.release()
        cv2.destroyAllWindows()
