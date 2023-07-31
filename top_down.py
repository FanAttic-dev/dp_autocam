import json
from pathlib import Path
import cv2
import numpy as np
from utils import apply_homography, coords_to_pts, load_json
from constants import colors


class TopDown:
    assets_path = Path('assets')
    pitch_model_path = assets_path / 'pitch_model.png'
    pitch_coords_path = assets_path / 'coords_pitch_model.json'

    def __init__(self, video_pitch_coords):
        self.pitch_model = cv2.imread(str(TopDown.pitch_model_path))
        self.pitch_coords = load_json(TopDown.pitch_coords_path)
        self.video_pitch_coords = video_pitch_coords
        self.H, _ = cv2.findHomography(coords_to_pts(video_pitch_coords),
                                       coords_to_pts(self.pitch_coords))

    def warp_frame(self, frame):
        return cv2.warpPerspective(
            frame, self.H, (self.pitch_model.shape[1], self.pitch_model.shape[0]))

    def warp_bbs(self, bbs):
        res = []
        for bb in bbs:
            x1, y1, x2, y2 = bb
            x1, y1 = apply_homography(self.H, x1, y1)
            x2, y2 = apply_homography(self.H, x2, y2)
            center_x = int((x1 + x2) / 2)
            center_y = int(y2)

            res.append((center_x, center_y))
        return res

    def check_bounds(self, x, y):
        h, w, _ = self.pitch_model.shape
        return x >= 0 and x < w and y >= 0 and y < h

    def draw_points_(self, pitch_model, pts):
        for pt in pts:
            x, y = pt
            if not self.check_bounds(x, y):
                print(x, y)
                continue
            cv2.circle(
                pitch_model, (x, y), radius=10, color=colors["teal"], thickness=-1)

    def draw_roi_(self, frame, camera):
        pts_warped = np.array(
            [apply_homography(self.H, x, y)
             for x, y in camera.get_corner_pts()],
            dtype=np.int32
        )
        cv2.polylines(frame, [pts_warped], isClosed=True,
                      color=colors["yellow"], thickness=5)
