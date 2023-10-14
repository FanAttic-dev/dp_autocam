import cv2
import numpy as np
from utils.config import Config

from utils.helpers import coords2pts
from utils.constants import Color


class ImagePreprocessor:
    def __init__(self, frame_orig, top_down, config: Config):
        pitch_coords = config.pitch_coords
        margins = config.dataset["mask_margins"]
        self.init_mask(frame_orig, top_down, pitch_coords, margins)

    def init_mask(self, frame_orig, top_down, pitch_coords, margins):
        pts = coords2pts(pitch_coords).squeeze()
        pts_top_down = top_down.pts2top_down_points(pts)

        lb, lt, rt, rb = pts_top_down
        mt, mr, mb, ml = margins.values()
        lb += [ml, mb]
        lt += [ml, mt]
        rt += [mr, mt]
        rb += [mr, mb]

        pts_top_down = [lb, lt, rt, rb]
        pts = top_down.top_down_points2pts(pts_top_down).astype(np.int32)

        self.mask = np.zeros(frame_orig.shape[:2], dtype=np.uint8)
        cv2.fillPoly(self.mask, pts=[pts], color=Color.WHITE)

    def draw_mask(self, img):
        return cv2.bitwise_and(img, img, mask=self.mask)
