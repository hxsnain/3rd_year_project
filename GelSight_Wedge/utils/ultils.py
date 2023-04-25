from urllib import request
import numpy as np
import cv2

def downloadImg():
    global url
    with request.urlopen(url) as f:
        data = f.read()
        img1 = np.frombuffer(data, np.uint8)
        img_cv = cv2.imdecode(img1, cv2.IMREAD_ANYCOLOR)
        return img_cv