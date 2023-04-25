import numpy as np
import cv2
import yaml
from yaml.loader import SafeLoader
import copy

color_lists = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

# read the xml file
with open("C:/Users/hasna/OneDrive/Documents/Year_3/3rd_year_project/GelSight_Wedge/config/gelsight_wedge_config.yaml") as f:
    global config
    config = yaml.load(f, Loader=SafeLoader)

original_H = 480
original_W = 640
img_last_action = np.zeros((original_H, original_W, 3), dtype=np.uint8)

selected_corners = [(17, 78), (352, 153), (349, 370), (12, 420)]


def mouse_cb(action, x, y, *userdata):
    # num_corners = 0
    global img
    global img_last_action

    # when left mouse button is pressed, add one point
    if action == cv2.EVENT_LBUTTONDOWN:
        # save the previous image in case misclick
        if selected_corners.__len__() > 0:
            img_last_action = copy.deepcopy(img)
        cv2.circle(img, (x, y), 4, color_lists[selected_corners.__len__()])
        selected_corners.append((x, y))
        cv2.imshow("CORNER_SELECT", img)



    # if the right mouse button is pressed, remove the previous point
    if action == cv2.EVENT_RBUTTONDOWN:
        img = copy.deepcopy(img_last_action)
        remove = selected_corners.pop()
        cv2.imshow("CORNER_SELECT", img_last_action)

        # num_corners -= 1

    if selected_corners.__len__() == 4:
        print(selected_corners)
        selected_corners.clear()


if __name__ == "__main__":

    # capture video from the camera
    cap = cv2.VideoCapture(0)

    # show the images until
    N = 0

    # wait for the stream
    while True:
        ret, img = cap.read()
        if not ret:
            continue
        else:
            break

    #
    while N<10:
        ret, img = cap.read()
        N += 1

    # img = cv2.imread("/home/jackie/VisTac2Pose/1.jpeg")
    img_last_action = copy.deepcopy(img)
    WINDOW_NAME = "CORNER_SELECT"
    cv2.imshow(WINDOW_NAME, img)
    cv2.setMouseCallback(WINDOW_NAME, mouse_cb, 0)
    cv2.waitKey()