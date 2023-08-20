import json
from pathlib import Path
import cv2
import numpy as np
from detector import YoloDetector
from utils import apply_homography, coords_to_pts, load_json
from constants import colors


class TopDown:
    assets_path = Path('assets')
    pitch_model_path = assets_path / 'pitch_model.png'
    pitch_coords_path = assets_path / 'coords_pitch_model.json'

    def __init__(self, video_pitch_coords, camera):
        self.pitch_model = cv2.imread(str(TopDown.pitch_model_path))
        self.pitch_coords = load_json(TopDown.pitch_coords_path)
        self.video_pitch_coords = video_pitch_coords
        self.H, _ = cv2.findHomography(coords_to_pts(video_pitch_coords),
                                       coords_to_pts(self.pitch_coords))
        self.camera = camera

    @property
    def H_inv(self):
        return np.linalg.inv(self.H)

    def warp_frame(self, frame):
        return cv2.warpPerspective(
            frame, self.H, (self.pitch_model.shape[1], self.pitch_model.shape[0]))

    def bbs2points(self, bbs):
        points = {
            "points": [],
            "cls": []
        }
        for bb, cls in zip(bbs["boxes"], bbs["cls"]):
            x1, y1, x2, y2 = bb
            x1, y1 = apply_homography(self.H, x1, y1)
            x2, y2 = apply_homography(self.H, x2, y2)
            center_x = int((x1 + x2) / 2)
            center_y = int(y2)

            points["points"].append((center_x, center_y))
            points["cls"].append(cls)
        return points

    def check_bounds(self, x, y):
        h, w, _ = self.pitch_model.shape
        return x >= 0 and x < w and y >= 0 and y < h

    def draw_bbs_(self, top_down_frame, bbs):
        points = self.bbs2points(bbs)
        for pt, cls in zip(points["points"], points["cls"]):
            # if not self.check_bounds(*pt):
            #     continue
            cv2.circle(
                top_down_frame,
                pt,
                radius=10,
                color=YoloDetector.cls2color[cls],
                thickness=-1
            )

    def draw_last_measurement_(self, top_down_frame):
        meas = np.array([[self.camera.measurement_last]], dtype=np.float32)
        meas_top_down_coord = cv2.perspectiveTransform(meas, self.H)[0][0]
        self.draw_points_(
            top_down_frame, [meas_top_down_coord], colors["violet"], radius=20)

    def draw_points_(self, top_down_frame, points, color, radius=10):
        for pt in points:
            x, y = pt
            cv2.circle(
                top_down_frame,
                (int(x), int(y)),
                radius=radius,
                color=color,
                thickness=-1
            )

    def draw_roi_(self, frame):
        x_min = self.video_pitch_coords["left bottom"]["x"]
        x_max = self.video_pitch_coords["right bottom"]["x"]
        y_max = self.video_pitch_coords["left bottom"]["y"]
        y_min = self.video_pitch_coords["left top"]["y"]

        pts_warped = []
        for x, y in self.camera.get_corner_pts():
            x = np.clip(x, x_min, x_max)
            y = np.clip(y, y_min, y_max)
            x_, y_ = apply_homography(self.H, x, y)
            pts_warped.append((x_, y_))
        pts_warped = np.array(pts_warped, dtype=np.int32)

        cv2.polylines(frame, [pts_warped], isClosed=True,
                      color=colors["yellow"], thickness=5)

    def get_frame(self, bbs):
        top_down_frame = self.pitch_model.copy()
        self.draw_roi_(top_down_frame)

        self.draw_bbs_(top_down_frame, bbs)
        self.draw_last_measurement_(top_down_frame)
        return top_down_frame
