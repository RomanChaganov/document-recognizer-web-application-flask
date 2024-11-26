from app import TESSDATA_PATH
from collections import defaultdict

import cv2
import math
import numpy as np

from PIL import Image
from tesserocr import PyTessBaseAPI, PSM


def preprocess(filename, delete_stamp):
    image = cv2.imread(filename)

    if image is None:
        return image

    if delete_stamp:
        gray = remove_stamp(image)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    binary = binarization(gray, False)
    deskew_img = deskew(binary)
    deskew_img = _rotate_tesser(deskew_img)

    return deskew_img, image


def remove_stamp(image):
    K = _extrack_black(image)
    K = cv2.bitwise_not(K)
    # K = cv2.convertScaleAbs(K, alpha=2, beta=-70)

    return K


def _extrack_black(image):
    # CMYK, but only key channel!
    norm = np.float32(image) / 255.0
    K = 1 - np.max(norm, axis=-1)
    K = np.uint8(K * 255)

    return K


def binarization(gray, adaptive=False):
    if adaptive:
        B = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 10
        )
    else:
        ret, B = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return B


def deskew(binary):
    angel = _find_angel(binary)

    if angel != 0:
        binary = _rotate(binary, angel)

    return binary


def _rotate_tesser(image):
    img = Image.fromarray(image)
    with PyTessBaseAPI(psm=PSM.AUTO_OSD, path=TESSDATA_PATH, lang='rus+eng') as api:
        api.SetImage(img)
        it = api.AnalyseLayout()
        osd = it.Orientation()
        direction = osd[0]
    
    image = _rotate(image, 270 / direction) if direction else image
    return image


def _find_angel(binary):
    binary_not = cv2.bitwise_not(binary)
    width = binary_not.shape[1]

    lines = cv2.HoughLinesP(binary_not, 1, np.pi / 180, 100, int(width / 8), 20)
    if len(lines) == 0:
        return 0
    
    angels = list()
    for line in lines:
        x0, y0, x1, y1 = line[0]
        angels.append(math.atan2(y1 - y0, x1 - x0))
    
    counts = defaultdict(int)

    for angel in angels:     
        for key in counts:
            if abs(angel - key) <= 0.01:
                counts[key] += 1
                break
        else:
            counts[angel] += 1

    angel = max(counts, key=counts.get)
    angel = angel * 180 / math.pi

    return angel


def _rotate(binary, angel):
    if angel == 90:
        img = cv2.transpose(binary)
        img = cv2.flip(img, 1)
    elif angel == 180:
        img = cv2.flip(img, -1)
    elif angel == 270:
        img = cv2.transpose(binary)
        img = cv2.flip(img, 0)
    else:
        length = max(binary.shape)
        R = cv2.getRotationMatrix2D((length / 2, length / 2), angel, 1) 
        img = cv2.warpAffine(
            binary, R, (length, length), borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )

    return img
