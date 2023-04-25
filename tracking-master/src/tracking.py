from lib import find_marker
import numpy as np
import cv2
import time
import marker_dectection
import sys
import setting

calibrate = False

if cv2.CAP_PROP_BACKEND == cv2.CAP_V4L2:
    cv2.CAP_PROP_BACKEND = cv2.CAP_V4L

cap = cv2.VideoCapture(2, cv2.CAP_V4L2)

# cap = cv2.VideoCapture(2)
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FPS))

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
print(cap.get(cv2.CAP_PROP_FPS))

ret, frame = cap.read()
print(frame.shape)  # (y,x) before rotate (H,W)


# Resize scale for faster image processing
setting.init()
RESCALE = setting.RESCALE

# Create Mathing Class
m = find_marker.Matching(
    N_=setting.N_,
    M_=setting.M_,
    fps_=setting.fps_,
    x0_=setting.x0_,
    y0_=setting.y0_,
    dx_=setting.dx_,
    dy_=setting.dy_,
)
"""
N_, M_: the row and column of the marker array
x0_, y0_: the coordinate of upper-left marker
dx_, dy_: the horizontal and vertical interval between adjacent markers
"""


def get_marker(frame):
    cv2.imshow("Frame", frame)
    frame = cv2.GaussianBlur(frame, (11, 11), 0)
    cv2.imshow("Gaussian", frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # print("Get marker size: ", frame.shape)
    ret, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("thresh", thresh)
    th2 = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # mask_erode = cv.GaussianBlur(mask_erode, (5,5),0 )
    mask_erode = cv2.erode(thresh, kernel=np.ones((3, 3), np.uint8), iterations=1)
    cv2.imshow("eroded", mask_erode)
    mask_dilate = cv2.dilate(mask_erode, kernel=np.ones((5, 5), np.uint8), iterations=5)
    mask_erode = cv2.erode(mask_dilate, kernel=np.ones((3, 3), np.uint8), iterations=1)

    mask_color = cv2.cvtColor(mask_erode, cv2.COLOR_GRAY2BGR)
    contours, hierarchy = cv2.findContours(
        mask_erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    MIN_A = 150
    true_centers = []
    for num, i in enumerate(contours):
        area = cv2.contourArea(i)
        if area > MIN_A:
            rect = cv2.minAreaRect(i)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            center = tuple(np.intp((box[0] + box[1] + box[2] + box[3]) / 4))
            cv2.circle(mask_color, center, 1, (255, 0, 0), 3)
            # cv2.putText(mask_color,"{}".format(area),center , cv2.FONT_HERSHEY_COMPLEX, 0.5,(0,0,255))
            cv2.drawContours(mask_color, [box], 0, (0, 0, 255), 2)
            # if setting.RESCALE == 2:
            #     if (
            #         center[1] > 10
            #         and center[1] < 310
            #         and center[0] > 10
            #         and center[0] < 230
            #     ):  # Crop 320, 240 (y,x)
            #         true_centers.append(center)

            # else:
            #     if (
            #         center[1] > 20
            #         and center[1] < 620
            #         and center[0] > 20
            #         and center[0] < 460
            #     ):  # Crop 640, 480 (y,x)
            true_centers.append(center)

    constructed = np.zeros(frame.shape)
    for center in true_centers:
        cv2.circle(constructed, center, 5, (255, 255, 255), -1)
    cv2.imshow("Mask_color", mask_color)
    cv2.imshow("Constructed", constructed)
    constructed = np.float32(constructed)
    constructed = cv2.cvtColor(constructed, cv2.COLOR_BGR2GRAY)
    return constructed


def get_rotation(flow, frame):
    # print(type(flow))
    # print(len(flow))

    (Ox, Oy, Cx, Cy, Occ) = flow
    Oxnp = np.array(Ox).flatten()
    Oynp = np.array(Oy).flatten()
    Cxnp = np.array(Cx).flatten()
    Cynp = np.array(Cy).flatten()

    A = np.empty((1, 2))
    bb = np.empty(1)
    midlist = []
    Olist = []
    Clist = []
    for i in range(len(Oxnp)):
        Ox = Oxnp[i]
        Oy = Oynp[i]
        Cx = Cxnp[i]
        Cy = Cynp[i]
        O = np.array([Ox, Oy])
        C = np.array([Cx, Cy])
        # Set threshold
        dist = np.linalg.norm(C - O)

        if dist > 5:
            mid = (int((Cx + Ox) / 2), int((Cy + Oy) / 2))
            a1 = [[(Cx - Ox), (Cy - Oy)]]
            b1 = [mid[0] * (Cx - Ox) + mid[1] * (Cy - Oy)]
            # print(a1, b1)
            A = np.concatenate((A, a1), axis=0)
            bb = np.concatenate((bb, b1), axis=0)
            midlist.append(mid)
            # print(mid)
            cv2.circle(frame, mid, 5, (155, 155, 255), 1)
            Olist.append(O)
            Clist.append(C)

    A = np.delete(A, 0, 0)
    bb = np.delete(bb, 0, 0)

    x = np.linalg.lstsq(A, bb, rcond=-1)
    (cx, cy) = x[0]
    center = (int(cx), int(cy))
    centernp = np.array(center)
    cv2.circle(frame, center, 3, (255, 255, 255), -1)

    angle_list = []
    moment_list = []
    for i, mid in enumerate(midlist):
        before_vec = Olist[i] - centernp
        after_vec = Clist[i] - centernp
        angle = np.degrees(
            np.arccos(
                (np.dot(before_vec, after_vec))
                / (np.linalg.norm(before_vec) * np.linalg.norm(after_vec))
            )
        )
        moment = np.cross(before_vec, after_vec)
        moment_list.append(moment)
        angle_list.append(angle)
        cv2.line(frame, center, mid, (255, 255, 255), 1)
    if len(midlist) > 5:
        moment_np = np.array(moment_list)
        ccw = np.count_nonzero(moment_np > 0)
        cw = np.count_nonzero(moment_np < 0)
        total = ccw + cw
        if abs(ccw - cw) / total < 0.5:
            rot = 0
        elif ccw < cw:
            print("here")
            rot = -1
        elif ccw > cw:
            rot = 1
        print("CCW", ccw, "CW", cw, "Rot", rot)

        angle = np.mean(np.array(angle_list))
        if angle == np.NaN:
            angle = 0
        else:
            angle = rot * angle

        cv2.putText(
            frame,
            "{} deg".format(angle),
            (50, 50),
            cv2.FONT_HERSHEY_COMPLEX,
            0.4,
            (255, 255, 0),
        )
    else:
        center = None
        angle = None
    return center, angle


# save video
# fourcc = cv2.VideoWriter_fourcc(*'XVID')

# if gelsight_version == 'HSR':
#     out = cv2.VideoWriter('output.mp4',fourcc, 30.0, (215,215))
# else:
#     out = cv2.VideoWriter('output.mp4',fourcc, 30.0, (1280//RESCALE,720//RESCALE))

# for i in range(30): ret, frame = cap.read()


# from pytouch.tasks import ContactArea

# start = time.time()
# while time.time() - start < 1:
#     ret, base1 = cap.read()
# # base1 = cv2.cvtColor(base1, cv2.COLOR_BGR2GRAY)
# # base1 = cv2.cvtColor(base1, cv2.COLOR_GRAY2BGR)
# contact_area = ContactArea(base=base1, contour_threshold=50)


ret, base_frame = cap.read()
base_frame = marker_dectection.init(base_frame)
base_frame = cv2.rotate(base_frame, cv2.ROTATE_90_CLOCKWISE)


##================================ Get the contact area
def get_contact(base_frame, frame):
    diff = cv2.absdiff(frame, base_frame)
    diff_invert = cv2.bitwise_not(diff)
    cv2.imshow("Inv", diff_invert)

    # diff[diff > 50] = 255
    diffB = diff[:, :, 0]
    diffMax = diff[:, :, 0].copy()
    diffG = diff[:, :, 1]
    diffR = diff[:, :, 2]

    diffmax = np.amax(diff, axis=2)
    ret, diffthresh = cv2.threshold(diffmax, 100, 255, cv2.THRESH_BINARY)
    diffthresh_erode = cv2.erode(diffthresh, (3, 3), iterations=3)
    diffthresh_dilate = cv2.dilate(diffthresh_erode, (5, 5), iterations=3)
    cv2.imshow("DiffM", diffmax)
    cv2.imshow("DiffThresh", diffthresh)
    cv2.imshow("DiffThreshDilate", diffthresh_dilate)

    # diffthreshdilate = cv2.dilate(diffthresh, kernel=np.ones((5, 5), np.uint8), iterations=5)
    # cv2.imshow("ThreshDilate", diffthreshdilate)
    # print(diffmax.shape)
    # print(diffmax)
    # diffmaxBGR = np.dstack((diffmax, diffmax, diffmax))

    # diffmaxBGR = cv2.cvtColor(diffmaxBGR, cv2.COLOR_GRAY2BGR)
    # cv2.imshow("DiffMax", diffmaxBGR)
    # cv2.imshow("DiffB", diffB)
    # cv2.imshow("DiffG", diffG)
    # cv2.imshow("DiffR", diffR)
    cv2.imshow("Diff", diff)


ret, frame = cap.read()
while True:
    tm = time.time()
    # capture frame-by-frame
    ret, frame = cap.read()
    # print(frame.shape)
    if not (ret):
        break
    frame = marker_dectection.init(frame)
    # print(frame.shape)

    frame_raw = frame.copy()

    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    ## FIXME:
    # if gelsight_version == "HSR":
    #     frame = marker_dectection.init_HSR(frame)
    # else:
    #     frame = marker_dectection.init(frame)
    # resize (or unwarp)

    # frame = marker_dectection.init_HSR(frame)
    # find marker masks FIXME: Change
    # mask = marker_dectection.find_marker(frame)

    mask = get_marker(frame)  # 640 x 480

    # find marker centers
    mc = marker_dectection.marker_center(mask, frame)

    if calibrate == False:
        tm = time.time()
        # # matching init
        m.init(mc)
        # # matching
        m.run()
        # # matching result
        """
        output: (Ox, Oy, Cx, Cy, Occupied) = flow
            Ox, Oy: N*M matrix, the x and y coordinate of each marker at frame 0
            Cx, Cy: N*M matrix, the x and y coordinate of each marker at current frame
            Occupied: N*M matrix, the index of the marker at each position, -1 means inferred. 
                e.g. Occupied[i][j] = k, meaning the marker mc[k] lies in row i, column j.
        """
        flow = m.get_flow()
        print(flow)
        ## Inspect the Flow
        # (Ox, Oy, Cx, Cy, Occ) = flow
        # print(flow)

        # for i in range(setting.N_):
        #     for k in range(setting.M_):

        # print(len(mc))
        # print(type(Ox), type(Ox),type(Cx), type(Cx), type(Occ))
        # print(flow.shape)

        # draw flow
        marker_dectection.draw_flow(frame, flow)
        # center, angle = get_rotation(flow, frame)

    mask_img = mask.astype(frame[0].dtype)

    mask_img = cv2.merge((mask_img, mask_img, mask_img))

    # cv2.imshow('raw',frame_raw)
    cv2.imshow("frame", frame)

    if calibrate:
        # Display the mask
        cv2.imshow("mask", mask_img)
        calibrate = False

    # out.write(frame)
    # print(frame.shape)

    # if time.time() - start > 1:
    #     m = find_marker.Matching(
    #         N_=setting.N_,
    #         M_=setting.M_,
    #         fps_=setting.fps_,
    #         x0_=setting.x0_,
    #         y0_=setting.y0_,
    #         dx_=setting.dx_,
    #         dy_=setting.dy_,
    #     )
    #     start = time.time()

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    # print(1/(time.time() - tm))


# When everything done, release the capture
cap.release()
# out.release()
cv2.destroyAllWindows()
