import cv2
import numpy as np
from math import pi, sin, cos
from threading import Thread
from find_marker import Matching
import time


class Tracking(Thread):
    def __init__(self, stream, tracking_setting, corners, output_sz, id='right'):
        Thread.__init__(self)

        self.stream = stream

        self.tracking_setting = tracking_setting
        self.m = Matching(*self.tracking_setting)

        self.corners = corners
        self.output_sz = output_sz

        self.running = False
        self.tracking_img = None

        self.slip_index_realtime = 0.

        self.flow = None

        self.id = id

    def __del__(self):
        pass

    def find_marker(self, frame, RESCALE=4):


    def tracking(self):
        m = self.m
        frame0 = None

        self.running = True

        cnt = 0
        while self.running:
            img = self.stream.image.copy()

            if img is None: continue
            ############################################################
            # # find marker masks
            mask = self.find_marker(img)
            self.mask = mask

            # # # # find marker centers
            mc = self.marker_center(mask, img)

            m.init(mc)

            m.run()

            flow = m.get_flow()
            ############################################################

            if frame0 is None:
                frame0 = img.copy()
                # frame0 = cv2.GaussianBlur(frame0, (int(63), int(63)), 0)

            diff = (img * 1.0 - frame0) * 4 + 127
            # trim(diff)

            self.diff_raw = diff.copy()

            (Ox, Oy, Cx, Cy, Occupied) = flow
            Ox, Oy, Cx, Cy = np.array(Ox), np.array(Oy), np.array(Cx), np.array(Cy)

            # draw flow
            self.draw_flow(diff, flow)

            # print(time.time()-tm)
            tm = time.time()

            # # Motor reaction based on the sliding information
            self.slip_index_realtime = float(np.mean(((Cx - Ox) ** 2 + (Cy - Oy) ** 2) ** 0.5))
            # # slip_index.put(slip_index_realtime)
            # print("ArrowMean CurveRight:", self.slip_index_realtime, end =" ")

            # self.tracking_img = (mask*1.0)
            self.tracking_img = diff / 255.

    def run(self):
        print("Run tracking algorithm")
        self.tracking()
        pass
