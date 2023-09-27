from pathlib import Path
import cv2
import numpy as np
from detection.detector import YoloDetector
from utils.helpers import apply_homography, coords2pts, discard_extreme_points_, load_json
from utils.constants import INTERPOLATION_TYPE, colors


class TopDown:
    assets_path = Path('assets')
    pitch_model_path = assets_path / 'pitch_model.png'
    pitch_coords_path = assets_path / 'coords_pitch_model.json'

    def __init__(self, video_pitch_coords, camera):
        self.pitch_model = cv2.imread(str(TopDown.pitch_model_path))
        self.pitch_model_red = self.get_pitch_model_red()
        self.pitch_coords = load_json(TopDown.pitch_coords_path)
        self.video_pitch_coords = video_pitch_coords
        self.H, _ = cv2.findHomography(coords2pts(video_pitch_coords),
                                       coords2pts(self.pitch_coords))
        self.camera = camera

    @property
    def H_inv(self):
        return np.linalg.inv(self.H)

    def get_pitch_model_red(self):
        pitch = self.pitch_model.copy()
        pitch[:, :, 0] = 0
        pitch[:, :, 1] = 0
        return pitch

    def warp_frame(self, frame, overlay=False):
        frame_warped = cv2.warpPerspective(
            frame, self.H,
            (self.pitch_model.shape[1], self.pitch_model.shape[0]),
            flags=INTERPOLATION_TYPE
        )

        if overlay:
            frame_warped = cv2.add(
                frame_warped, (self.pitch_model_red * 0.5).astype(np.uint8))

        return frame_warped

    def pts2top_down_points(self, pts):
        return np.array([apply_homography(self.H, *pt) for pt in pts])

    def top_down_points2pts(self, pts):
        return np.array([apply_homography(self.H_inv, *pt) for pt in pts])

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

    def draw_bbs_(self, top_down_frame, bbs, discard_extremes=False):
        if len(bbs) == 0 or len(bbs["boxes"]) == 0:
            return

        points = self.bbs2points(bbs)

        if discard_extremes:
            discard_extreme_points_(points)

        for pt, cls in zip(points["points"], points["cls"]):
            cv2.circle(
                top_down_frame,
                pt,
                radius=15,
                color=YoloDetector.cls2color[cls],
                thickness=-1
            )

    def draw_screen_point_(self, top_down_frame, pt, color=colors["violet"], radius=30):
        if pt is None:
            return

        pt_x, pt_y = pt
        pt = np.array(
            [[[np.array(pt_x).item(), np.array(pt_y).item()]]], np.float32)
        pt_top_down_coord = cv2.perspectiveTransform(pt, self.H)[0][0]
        self.draw_points_(
            top_down_frame, [pt_top_down_coord], color, radius)

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
        margin = 50

        pitch_pts = coords2pts(self.video_pitch_coords)
        x_min, y_min = np.min(pitch_pts, axis=0)[0] - margin
        x_max, y_max = np.max(pitch_pts, axis=0)[0] + margin

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

        self.draw_bbs_(top_down_frame, bbs, discard_extremes=True)
        self.draw_screen_point_(
            top_down_frame, self.camera.players_filter.pos)
        return top_down_frame
