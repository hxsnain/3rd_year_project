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

# read the xml file
with open("../config/gelsight_wedge_config.yaml") as f:
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

bool_quit = False



def calculateProjection(selected_corners, cropped_w, cropped_h):
    quadcorner = np.array(selected_corners, dtype="float32")
    dst_corner = np.array([
        [0, 0],
        [cropped_w - 1, 0],
        [cropped_w - 1, cropped_h - 1],
        [0, cropped_h - 1]], dtype="float32")

    warpMatrix = cv2.getPerspectiveTransform(quadcorner, dst_corner)
    return warpMatrix


class Streaming(object):
    def __init__(self, url):
        self.image = None
        self.url = url
        self.streaming = False
        self.stream_url = request.urlopen(self.url)
        # self.start_stream()
        self.M = calculateProjection(selected_corners, cropped_w, cropped_h)

    def __del__(self):
        self.stop_stream()

    # def start_stream(self):
    #     self.streaming = True
    #     self.stream = request.urlopen(self.url)

    def stop_stream(self):
        if self.streaming == True:
            self.stream_url.close()
        self.streaming = False

    def load_stream(self):
        bytess = b''
        while True:
            # if self.streaming == False:
            #     time.sleep(0.01)
            #     continue
            # data = self.stream_url.read()
            # img1 = np.frombuffer(data, np.uint8)
            # img_cv = cv2.imdecode(img1, cv2.IMREAD_ANYCOLOR)
            # self.image = cv2.warpPerspective(img_cv, self.M, (cropped_w, cropped_h))
            # cv2.imshow("test", self.image)
            # cv2.waitKey(1)

            bytess += self.stream_url.read(32767)

            a = bytess.find(b'\xff\xd8')  # JPEG start
            b = bytess.find(b'\xff\xd9')  # JPEG end

            if a != -1 and b != -1:
                jpg = bytess[a:b + 2]  # actual image
                bytess = bytess[b + 2:]  # other informations

                image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                self.image = cv2.warpPerspective(image, self.M, (cropped_w, cropped_h))


class GelSight(Thread):
    def __init__(self, IP, corners, tracking_setting=None, output_sz=(210, 270), pose_enable=True, tracking_enable=False):
        Thread.__init__(self)

        url = "http://{}:8080/?action=stream".format(IP)
        # url = "http://{}:8080/?action=snapshot".format(IP)
        self.stream = Streaming(url)
        _thread.start_new_thread(self.stream.load_stream, ())

        self.cali = calibration()
        self.corners = corners
        self.output_sz = output_sz
        self.tracking_setting = tracking_setting
        self.pose_enable = pose_enable
        self.tracking_enable = tracking_enable
        self.output_sz_pose = output_sz

        # K_tracking = 1
        # self.output_sz_tracking = (int(output_sz[0]//K_tracking), int(output_sz[1]//K_tracking))

        # Wait for video streaming
        self.wait_for_stream()

        # Start thread for calculating pose
        self.start_pose()

        # Start thread for tracking markers
        # self.start_tracking()



    def __del__(self):
        self.pc.running = False
        self.tc.running = False
        self.stream.stop_stream()
        print("stop_stream")

    def start_pose(self):
        self.pc = Pose(self.stream, self.cali, self.corners, self.output_sz_pose) # Pose class
        self.pc.start()

    # def start_tracking(self):
    #     if self.tracking_setting is not None:
    #         self.tc = Tracking(self.stream, self.tracking_setting, self.corners, self.output_sz_tracking, id=self.id) # Tracking class
    #         self.tc.start()

    def wait_for_stream(self):
        while True:
            img = self.stream.image
            if img is None:
                continue
            else:
                break
        print("GelSight image found")

    def run(self):
        print("Run GelSight driver")
        pass


