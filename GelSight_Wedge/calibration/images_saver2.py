import copy
import cv2
import numpy as np
import yaml
from yaml.loader import SafeLoader
import re

# read the xml file
with open("C:/Users/hasna/OneDrive/Documents/Year_3/3rd_year_project/GelSight_Wedge/config/gelsight_wedge_config.yaml") as f:
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

# directory for image saving
saved_dir = "C:/Users/hasna/OneDrive/Documents/Year_3/3rd_year_project/GelSight_Wedge/saved_imgs"

def calculateProjection(selected_corners, cropped_w, cropped_h):
    quadcorner = np.array(selected_corners, dtype="float32")
    dst_corner = np.array([
        [0, 0],
        [cropped_w - 1, 0],
        [cropped_w - 1, cropped_h - 1],
        [0, cropped_h - 1]], dtype="float32")

    warpMatrix = cv2.getPerspectiveTransform(quadcorner, dst_corner)
    return warpMatrix

# function to read frames from webcam
def readFromWebcam(capture):
    _, frame = capture.read()
    return frame

if __name__ == "__main__":

    # calculate the projection matrix
    M = calculateProjection(selected_corners, cropped_w, cropped_h)

    # open the webcam
    capture = cv2.VideoCapture(0)

    # show the images until
    N = 0

    # wait for the webcam to warm up
    for i in range(30):
        capture.read()

    # capture the first frame
    img_crop = readFromWebcam(capture)
    cv2.imwrite("C:/Users/hasna/OneDrive/Documents/Year_3/3rd_year_project/GelSight_Wedge/saved_imgs/ref.jpg", img_crop)

    WINDOW_NAME = "WINDOW_SHOW"
    cv2.imshow(WINDOW_NAME, img_crop)
    cv2.waitKey(27)

    num_saved_images = 0
    # capture and display images until 'esc' key is pressed
    while True:
        # read a frame from the webcam
        img_crop = readFromWebcam(capture)
        img_crop = cv2.resize(img_crop, (int(cropped_w), int(cropped_h)))

        img_show = copy.deepcopy(img_crop)
        string_show = "Have saved %d images" % num_saved_images
        cv2.putText(img_show, string_show, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, img_show)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('s'):
            saved_name = saved_dir + "/sample_%d.jpg" % num_saved_images
            img_resized = cv2.resize(img_crop, (640, 480))
            cv2.imwrite(saved_name, img_resized)
            num_saved_images += 1

    capture.release()