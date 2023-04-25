import copy
import numpy as np

import cv2
from threading import Thread
from fast_poisson import fast_poisson

from scipy.interpolate import griddata



# def sigmoid(x):
#     return (np.exp(x) / (1 + np.exp(x)))


class Pose(Thread):
    def __init__(self, stream, cali, corners, output_sz=(100, 130)):
        Thread.__init__(self)
        self.stream = stream
        self.cali = cali
        self.thresh = 0.01
        self.kernel1 = self.make_kernal(2,'circle')
        self.table2 = np.load('../calibration/table_smooth.npy')

        self.corners = corners
        self.output_sz = output_sz

        self.running = False
        self.frame_large = None
        self.diff_large = None

        self.pose = None
        self.mv = None
        self.frame0 = None
        self.frame = None
        self.raw = None
        self.area = 0
        self.depth = None

    def __del__(self):
        pass

    def make_kernal(self, n, k_type):
        if k_type == 'circle':
            kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
        else:
            kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (n, n))
        return kernal

    def img2grad(self, frame, frame0, blur_inverse):
        diff_temp1 = frame - frame0
        diff_temp2 = diff_temp1 * blur_inverse
        diff_temp2[:, :, 0] = (diff_temp2[:, :, 0] - self.cali.zeropoint[0]) / self.cali.lookscale[0]
        diff_temp2[:, :, 1] = (diff_temp2[:, :, 1] - self.cali.zeropoint[1]) / self.cali.lookscale[1]
        diff_temp2[:, :, 2] = (diff_temp2[:, :, 2] - self.cali.zeropoint[2]) / self.cali.lookscale[2]
        diff_temp3 = np.clip(diff_temp2, 0, 0.999)
        diff = (diff_temp3 * self.cali.bin_num).astype(int)

        grad_img = self.table2[diff[:, :, 0], diff[:, :, 1], diff[:, :, 2], :]

        return grad_img

    def img2depth(self, frame, frame0, marker_mask, blur_inverse):
        grad_img2 = self.img2grad(frame, frame0, blur_inverse)

        # x, y = np.meshgrid(np.linspace(0, marker_mask.shape[1] - 1, marker_mask.shape[1]),
        #                    np.linspace(0, marker_mask.shape[0] - 1, marker_mask.shape[0]))
        # unfill_x = x[marker_mask < 1].astype(int)
        # unfill_y = y[marker_mask < 1].astype(int)
        # points = np.stack((unfill_y, unfill_x), axis=1)
        grad_img2[:, :, 0] = grad_img2[:, :, 0] * (1 - marker_mask)
        grad_img2[:, :, 1] = grad_img2[:, :, 1] * (1 - marker_mask)
        # # grad_img2[:, :, 0] = grad_img2[:, :, 0] * (1 - marker_mask) * red_mask
        # # grad_img2[:, :, 1] = grad_img2[:, :, 1] * (1 - marker_mask) * red_mask
        #
        # values_1 = grad_img2[:, :, 0][marker_mask < 1]
        # values_2 = grad_img2[:, :, 1][marker_mask < 1]
        # grad_img2[:, :, 0] = griddata(points, values_1, (y, x), method='nearest')
        # grad_img2[:, :, 1] = griddata(points, values_2, (y, x), method='nearest')

        # zeros = np.zeros_like(dx)
        return fast_poisson(grad_img2[:,:,0], grad_img2[:,:,1])

    def marker_detection(self, raw_image_blur):
        m, n = raw_image_blur.shape[1], raw_image_blur.shape[0]
        raw_image_blur = cv2.pyrDown(raw_image_blur).astype(np.float32)
        ref_blur = cv2.GaussianBlur(raw_image_blur, (25, 25), 0)
        diff = ref_blur - raw_image_blur
        diff *= 16.0
        diff[diff < 0.] = 0.
        diff[diff > 255.] = 255.

        mask_b = diff[:, :, 0] > 150
        mask_g = diff[:, :, 1] > 150
        mask_r = diff[:, :, 2] > 150
        mask = (mask_b * mask_g + mask_b * mask_r + mask_g * mask_r) > 0
        mask = cv2.resize(mask.astype(np.uint8), (m, n))
        return mask

    def get_pose(self):

        self.running = True

        num = 0
        while self.running:
            img = self.stream.image.copy()
            frame = copy.deepcopy(img)
            raw = copy.deepcopy(img)

            if img is None: continue

            # Store first frame
            num += 1
            blur_inverse = 1
            if num == 1:
                frame0 = img.copy()
                self.frame0 = frame0
                marker = self.cali.mask_marker(frame0)
                keypoints = self.cali.find_dots(marker)
                marker_mask = self.cali.make_mask(frame0, keypoints)
                frame0 = cv2.inpaint(frame0, marker_mask, 3, cv2.INPAINT_TELEA)

                # red_mask = (frame0[:, :, 2] > 12).astype(np.uint8)
                frame0 = cv2.GaussianBlur(frame0.astype(np.float32), (3, 3), 0) + 1
                blur_inverse = 1 + ((np.mean(frame0) / frame0) - 1) * 2

            # self.frame = frame
            frame = cv2.GaussianBlur(frame.astype(np.float32), (3, 3), 0)
            marker_mask = self.marker_detection(frame)
            # marker_mask = self.cali.mask_marker(frame)

            # marker_mask = cv2.dilate(marker_mask, self.kernel1, iterations=1)

            depth = self.img2depth(frame, self.frame0, marker_mask, blur_inverse)
            depth[depth < 0] = 0

            # self.depth = depth.copy()

            depth_filter = np.zeros(depth.shape, dtype=np.uint8)
            depth_filter[depth > 0.5] = 255
            self.depth = depth_filter
            contours, _ = cv2.findContours(depth_filter, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            num_cnt = len(contours)
            self.area = 0

            if num_cnt >0:
                areas = [cv2.contourArea(c) for c in contours]
                max_index = np.argmax(areas)
                cnt = contours[max_index]
                area = cv2.contourArea(cnt)
                if area>10:
                    ellipse = cv2.fitEllipse(cnt)
                    color = (255, 255, 0)
                    width = 5
                    # img = np.zeros(test_img.shape,dtype=np.uint8)
                    frame = np.uint8(frame)
                    cv2.ellipse(self.frame, ellipse, (color), width)
                    # Record pose estimation
                    self.pose = (ellipse[0], ellipse[2])
                    self.area = area

                else:
                    # No contact
                    self.pose = None

            else:
                # No contact
                self.pose = None

            self.raw = raw


    def run(self):
        print("Run pose estimation")
        self.get_pose()
        pass