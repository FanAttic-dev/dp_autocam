import cv2
import numpy as np
from camera.top_down import TopDown
from utils.config import Config

from utils.constants import DT_INT, Color


class ImagePreprocessor:
    def __init__(self, frame_orig, top_down: TopDown, config: Config):
        pitch_corners = config.pitch_corners
        margins = config.dataset["mask_margins"]
        self.init_mask(frame_orig, top_down, pitch_corners, margins)

    def init_mask(self, frame_orig, top_down: TopDown, pitch_corners, margins):
        pts = pitch_corners.squeeze()
        pts_top_down = top_down.pts_screen2tdpts(pts)

        lb, lt, rt, rb = pts_top_down
        mt, mr, mb, ml = margins.values()
        lb += [ml, mb]
        lt += [ml, mt]
        rt += [mr, mt]
        rb += [mr, mb]

        pts_top_down = [lb, lt, rt, rb]
        pts = top_down.tdpts2screen(pts_top_down).astype(DT_INT)

        self.mask = np.zeros(frame_orig.shape[:2], dtype=np.uint8)
        cv2.fillPoly(self.mask, pts=[pts], color=Color.WHITE)

    def draw_mask(self, img):
        return cv2.bitwise_and(img, img, mask=self.mask)

    @staticmethod
    def overlay_white(img, alpha):
        white = (np.ones(img.shape) * 255).astype(np.uint8)
        return cv2.addWeighted(img, (1-alpha), white, alpha, 0.0)
