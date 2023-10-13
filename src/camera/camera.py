from abc import ABC, abstractmethod

import numpy as np
from utils.protocols import HasStats


class Camera(ABC, HasStats):
    FRAME_W = 1920
    FRAME_H = 1080
    FRAME_ASPECT_RATIO = FRAME_W/FRAME_H
    FRAME_CORNERS = np.array([
        [0, 0],
        [0, FRAME_H-1],
        [FRAME_W-1, FRAME_H-1],
        [FRAME_W-1, 0]
    ], dtype=np.int16)

    @property
    @abstractmethod
    def center(self):
        ...

    @property
    def corners_ang(self):
        return {
            "left top": [-self.fov_horiz_deg / 2, -self.fov_vert_deg / 2],
            "left bottom": [-self.fov_horiz_deg / 2, self.fov_vert_deg / 2],
            "right bottom": [self.fov_horiz_deg / 2, self.fov_vert_deg / 2],
            "right top": [self.fov_horiz_deg / 2, -self.fov_vert_deg / 2],
        }

    @abstractmethod
    def get_frame(self, frame_orig):
        ...

    @abstractmethod
    def pan(self, dx):
        ...

    @abstractmethod
    def get_corner_pts(self):
        ...

    @abstractmethod
    def update_by_bbs(self, bbs):
        ...

    @abstractmethod
    def get_stats(self) -> dict:
        ...

    def print(self):
        print(self.get_stats())
