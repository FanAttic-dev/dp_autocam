
import cv2
import numpy as np
from image_processor import ImageProcessor
from utils import coords_to_pts
from constants import colors


class ImagePreprocessor:
    """ Class for preprocessing the incoming video frame. """

    def __init__(self, frame_orig, pitch_coords):
        self.pitch_coords = pitch_coords

        self.h, self.w, _ = frame_orig.shape
        margin_vert = 0.4
        margin_horiz = 0.33
        dst = np.array([
            [0, (1 - margin_vert) * self.h],  # left bottom
            [margin_horiz * self.w, margin_vert * self.h],  # left top
            [(1 - margin_horiz) * self.w, margin_vert * self.h],  # right top
            [self.w-1, (1 - margin_vert) * self.h]  # right bottom
        ], dtype=np.int32)

        self.H, _ = cv2.findHomography(coords_to_pts(pitch_coords), dst)

    @property
    def H_inv(self):
        return np.linalg.inv(self.H)

    def draw_mask(self, img, margin=1):
        pts = coords_to_pts(self.pitch_coords)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, pts=[pts], color=colors["white"])

        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, se, iterations=margin)

        return cv2.bitwise_and(img, img, mask=mask)

    def pitch2normalized(self, pts):
        return cv2.perspectiveTransform(pts.astype(np.float64), self.H)

    def normalized2pitch(self, pts):
        return cv2.perspectiveTransform(np.array(pts, dtype=np.float64), self.H_inv)

    def preprocess(self, img):
        img = self.draw_mask(img)
        # img = cv2.warpPerspective(img, self.H, (self.w, self.h))

        return img
