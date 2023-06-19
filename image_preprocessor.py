import cv2
import numpy as np

from pathlib import Path
from utils import coords_to_pts


class ImagePreprocessor:
    GAUSS_SIGMA = 1

    def __init__(self):
        ...

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

    def draw_lines(self, img, pts):
        return cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 255), thickness=5)

    def draw_mask(self, img, pts):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, pts=[pts], color=(255, 255, 255))
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, se, iterations=10)
        return cv2.bitwise_and(img, img, mask=mask)

    def draw_bounding_boxes(self, img, bbs):
        for bb in bbs:
            x, y, w, h = bb.cpu().numpy().astype(int)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)

    def process_frame(self, src, coords):
        dst = cv2.GaussianBlur(src, (3, 3), ImagePreprocessor.GAUSS_SIGMA)

        pts = coords_to_pts(coords)
        dst = self.draw_mask(dst, pts)
        # dst = self.rescale(src, 1)
        # dst = self.roi(dst, 3000, 400, 3960, 816)
        # dst = self.roi_16_9(src, 2500, 400, 1000)
        # dst = self.detect_players(dst)[0].plot()
        # dst = self.detect_ball(dst)[0].plot()
        # dst = self.draw_lines(dst, pts)

        return dst
