import cv2
import numpy as np

WINDOW_NAME = 'frame'
WINDOW_FLAGS = cv2.WINDOW_AUTOSIZE


def rescale(img, scale):
    h, w, _ = img.shape
    h, w = int(h * scale), int(w * scale)
    return cv2.resize(img, (w, h))


def roi(img, x1, y1, x2, y2):
    return img[y1:y2, x1:x2]


def process_frame(src):
    print(src.shape)
    h, w, _ = src.shape

    # dst = rescale(src, 1)
    dst = roi(src, 3050, 482, 3910, 784)

    cv2.namedWindow(WINDOW_NAME, WINDOW_FLAGS)
    cv2.imshow(WINDOW_NAME, dst)
    cv2.waitKey(0)
