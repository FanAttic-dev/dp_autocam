import cv2
import numpy as np

from utils import coords2pts
from constants import colors


class ImageProcessor:
    """Wrapper for image processing tasks."""
    @staticmethod
    def rescale(img, scale):
        h, w, _ = img.shape
        h, w = int(h * scale), int(w * scale)
        return cv2.resize(img, (w, h))

    @staticmethod
    def roi(img, x1, y1, x2, y2):
        h, w, _ = img.shape
        x1 = min(max(0, x1), w-1)
        x2 = min(max(0, x2), w-1)
        y1 = min(max(0, y1), h-1)
        y2 = min(max(0, y2), h-1)
        return img[y1:y2, x1:x2]

    @staticmethod
    def roi_16_9(img, x1, y1, w):
        img_h, img_w, _ = img.shape
        h = int(w/16*9)
        x2, y2 = x1 + w, y1 + h
        if y2 >= img_h:
            raise "roi_16_9: height too large, cannot keep 16:9 aspect ratio"
        return ImageProcessor.roi(img, x1, y1, x2, y2)

    @staticmethod
    def gaussian_blur(img, sigma=1):
        return cv2.GaussianBlur(img, (3, 3), sigma)

    @staticmethod
    def draw_mask(img, pitch_coords, margin):
        pts = coords2pts(pitch_coords)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, pts=[pts], color=colors["white"])

        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, se, iterations=margin)

        return cv2.bitwise_and(img, img, mask=mask)
