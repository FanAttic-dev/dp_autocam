import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm

from constants import GAUSS_SIGMA, MORPH_CLOSE_ITERATIONS, WINDOW_NAME
from utils import coords_to_pts

yolo_args = {
    'device': 0,  # 0 if gpu else 'cpu'
    'imgsz': 960,
    'classes': None,  # [0] for ball only, None for all
    'conf': 0.05,
    'max_det': 3,
    'iou': 0.7
}


class BackgroundSubtractor:
    def __init__(self):
        # self.model = cv2.createBackgroundSubtractorMOG2()
        self.model = cv2.createBackgroundSubtractorKNN(history=1)

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


class ImageProcessor:
    def __init__(self):
        self.bgSubtractor = BackgroundSubtractor()
        # bgSubtractor.init(cap)

    def rescale(self, img, scale):
        h, w, _ = img.shape
        h, w = int(h * scale), int(w * scale)
        return cv2.resize(img, (w, h))

    def roi(self, img, x1, y1, x2, y2):
        h, w, _ = img.shape
        x1 = min(max(0, x1), w-1)
        x2 = min(max(0, x2), w-1)
        y1 = min(max(0, y1), h-1)
        y2 = min(max(0, y2), h-1)
        return img[y1:y2, x1:x2]

    def roi_16_9(self, img, x1, y1, w):
        img_h, img_w, _ = img.shape
        h = int(w/16*9)
        x2, y2 = x1 + w, y1 + h
        if y2 >= img_h:
            raise "roi_16_9: height too large, cannot keep 16:9 aspect ratio"
        return self.roi(img, x1, y1, x2, y2)

    def detect_players(self, img):
        model_path = Path(f"./weights/yolov8_{yolo_args['imgsz']}.pt")
        model = YOLO(model_path)
        return model.predict(img, **yolo_args)

    def detect_ball(self, img):
        model_path = Path(f"./weights/yolov8_{yolo_args['imgsz']}_ball.pt")
        model = YOLO(model_path)
        return model.track(img, **yolo_args, tracker="bytetrack.yaml")

    def draw_lines(self, img, pts):
        return cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 255), thickness=5)

    def draw_mask(self, img, pts):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, pts=[pts], color=(255, 255, 255))
        # se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # mask = cv2.dilate(mask, se, iterations=10)
        return cv2.bitwise_and(img, img, mask=mask)

    def draw_bounding_boxes(self, mask, dst):
        contours, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # cv2.drawContours(dst, contours, -1, (0, 255, 0), 1)
        bbs = []
        for i, contour in enumerate(contours):
            bb = cv2.boundingRect(contour)
            x, y, w, h = bb
            # roi = img[y:y+h, x:x+w]
            cv2.rectangle(dst, (x, y), (x+w, y+h), (0, 255, 255), 2)
            bbs.append(bb)

        return dst, bbs

    def process_frame(self, src, coords):
        dst = cv2.GaussianBlur(src, (3, 3), GAUSS_SIGMA)

        pts = coords_to_pts(coords)
        dst = self.draw_mask(dst, pts)
        # dst = self.rescale(src, 1)
        # dst = self.roi(dst, 3000, 400, 3960, 816)
        # dst = self.roi_16_9(src, 2500, 400, 1000)
        # dst = self.detect_players(dst)[0].plot()
        # dst = self.detect_ball(dst)[0].plot()
        # dst = self.draw_lines(dst, pts)

        mask = self.bgSubtractor.apply(dst)
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                se, iterations=MORPH_CLOSE_ITERATIONS)
        dst, bbs = self.draw_bounding_boxes(mask=mask, dst=src)

        return dst, mask, bbs
