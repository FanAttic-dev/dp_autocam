import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm

WINDOW_NAME = 'frame'
WINDOW_FLAGS = cv2.WINDOW_NORMAL  # cv2.WINDOW_AUTOSIZE

yolo_args = {
    'device': 0,  # 0 if gpu else 'cpu'
    'imgsz': 960,
    'classes': None,  # [0] for ball only, None for all
    'conf': 0.15,
    'iou': 0.85
}


class BackgroundSubtractor:
    def __init__(self):
        # self.model = cv2.createBackgroundSubtractorMOG2()
        self.model = cv2.createBackgroundSubtractorKNN()

    def init(self, cap, iterations=10):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for _ in tqdm(range(iterations)):
            ret, frame = cap.read()
            if not ret:
                return
            self.apply(frame)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def apply(self, img):
        return self.model.apply(img)


def rescale(img, scale):
    h, w, _ = img.shape
    h, w = int(h * scale), int(w * scale)
    return cv2.resize(img, (w, h))


def roi(img, x1, y1, x2, y2):
    h, w, _ = img.shape
    x1 = min(max(0, x1), w-1)
    x2 = min(max(0, x2), w-1)
    y1 = min(max(0, y1), h-1)
    y2 = min(max(0, y2), h-1)
    return img[y1:y2, x1:x2]


def roi_16_9(img, x1, y1, w):
    img_h, img_w, _ = img.shape
    h = int(w/16*9)
    x2, y2 = x1 + w, y1 + h
    if y2 >= img_h:
        raise "roi_16_9: height too large, cannot keep 16:9 aspect ratio"
    return roi(img, x1, y1, x2, y2)


def detect_players(img):
    model_path = Path(f"./weights/yolov8_{yolo_args['imgsz']}.pt")
    model = YOLO(model_path)
    return model.predict(img, **yolo_args)


def coords_to_pts(coords):
    pts = np.array([[v["x"], v["y"]] for v in coords.values()], np.int32)
    return pts.reshape((-1, 1, 2))


def draw_lines(img, pts):
    return cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 255), thickness=5)


def draw_mask(img, pts):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, pts=[pts], color=(255, 255, 255))
    return cv2.bitwise_and(img, img, mask=mask)


def process_frame(src, coords, bgSubtractor=None):
    # cv2.namedWindow(WINDOW_NAME + 'original', cv2.WINDOW_NORMAL)
    # cv2.imshow(WINDOW_NAME + 'original', src)

    dst = src
    # dst = rescale(src, 1)
    # dst = roi(src, 3000, 400, 4200, 800)
    # dst = roi_16_9(src, 2500, 400, 1000)
    # dst = detect_players(dst)[0].plot()

    pts = coords_to_pts(coords)
    dst = draw_mask(dst, pts)
    dst = draw_lines(dst, pts)

    if bgSubtractor is not None:
        dst = bgSubtractor.apply(dst)

    if dst is not None:
        cv2.namedWindow(WINDOW_NAME, WINDOW_FLAGS)
        cv2.imshow(WINDOW_NAME, dst)

    key = cv2.waitKey(0)
    if key == ord('q'):
        return False

    return True
