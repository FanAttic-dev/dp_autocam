from functools import cached_property
from pathlib import Path
import cv2
import numpy as np
from camera.camera import Camera
from utils.config import Config
from utils.constants import DT_FLOAT, DT_INT, INTERPOLATION_TYPE, Color
import utils.utils as utils


class TopDown:
    OVERLAY_OPACITY = 0.5

    assets_path = Path('assets')
    pitch_model_path = assets_path / 'pitch' / 'pitch_model.png'
    pitch_model_corners_path = assets_path / 'pitch' / 'pitch_model_corners.yaml'

    def __init__(self, pitch_orig_corners, camera: Camera):
        self.camera = camera
        self.pitch_orig_corners = pitch_orig_corners

        self.pitch_model = cv2.imread(str(TopDown.pitch_model_path))
        self.pitch_model_overlay = self._get_pitch_model_overlay()

    def _get_pitch_model_overlay(self):
        self.pitch_model_overlay = utils.mask_out_red_channel(self.pitch_model)
        return (self.pitch_model_overlay * TopDown.OVERLAY_OPACITY).astype(np.uint8)

    @cached_property
    def pitch_model_corners(self):
        pitch_model_corners = utils.load_yaml(
            TopDown.pitch_model_corners_path
        )
        return Config.load_pitch_corners(pitch_model_corners)

    @cached_property
    def H(self):
        H, _ = cv2.findHomography(
            self.pitch_orig_corners,
            self.pitch_model_corners
        )
        return H

    @cached_property
    def H_inv(self):
        return np.linalg.inv(self.H)

    def warp_frame(self, frame, overlay=False):
        frame_warped = cv2.warpPerspective(
            frame, self.H,
            (self.pitch_model.shape[1], self.pitch_model.shape[0]),
            flags=INTERPOLATION_TYPE
        )

        if overlay:
            frame_warped = cv2.add(frame_warped, self.pitch_model_overlay)

        return frame_warped

    def pts_screen2tdpts(self, pts):
        return np.array([utils.apply_homography(self.H, *pt) for pt in pts])

    def tdpts2screen(self, pts):
        return np.array([utils.apply_homography(self.H_inv, *pt) for pt in pts])

    def bbs_screen2tdpts(self, bbs):
        """Convert bounding boxes to top-down points (TDPts)."""
        tdpts = {
            "pts": [],
            "cls": []
        }
        for bb, cls in zip(bbs["boxes"], bbs["cls"]):
            x1, y1, x2, y2 = bb
            x1, y1 = utils.apply_homography(self.H, x1, y1)
            x2, y2 = utils.apply_homography(self.H, x2, y2)
            center_x = int((x1 + x2) / 2)
            center_y = int(y2)

            tdpts["pts"].append((center_x, center_y))
            tdpts["cls"].append(cls)
        return tdpts

    def check_bounds(self, x, y):
        h, w, _ = self.pitch_model.shape
        return x >= 0 and x < w and y >= 0 and y < h

    def draw_bbs_(self, frame_top_down, bbs, discard_extremes=False):
        if len(bbs) == 0 or len(bbs["boxes"]) == 0:
            return

        tdpts = self.bbs_screen2tdpts(bbs)

        if discard_extremes:
            utils.discard_extreme_tdpts_(tdpts)

        for pt, cls in zip(tdpts["pts"], tdpts["cls"]):
            cv2.circle(
                frame_top_down,
                pt,
                radius=15,
                color=Color.cls2color[cls],
                thickness=-1
            )

    def draw_screen_pt_(self, frame_top_down, pt, color=Color.VIOLET, radius=30):
        if pt is None:
            return

        pt_x, pt_y = pt
        pt = np.array(
            [[[np.array(pt_x).item(), np.array(pt_y).item()]]], DT_FLOAT)
        pt_top_down_coord = cv2.perspectiveTransform(pt, self.H)[0][0]
        self.draw_pts_(
            frame_top_down, [pt_top_down_coord], color, radius)

    def draw_pts_(self, frame_top_down, pts, color, radius=10):
        for pt in pts:
            x, y = pt
            cv2.circle(
                frame_top_down,
                (int(x), int(y)),
                radius=radius,
                color=color,
                thickness=-1
            )

    def draw_roi_(self, frame):
        margin = 50

        pitch_pts = self.pitch_orig_corners
        x_min, y_min = np.min(pitch_pts, axis=0)[0] - margin
        x_max, y_max = np.max(pitch_pts, axis=0)[0] + margin

        pts_warped = []
        for x, y in self.camera.get_pts_corners():
            x = np.clip(x, x_min, x_max)
            y = np.clip(y, y_min, y_max)
            x_, y_ = utils.apply_homography(self.H, x, y)
            pts_warped.append((x_, y_))
        pts_warped = np.array(pts_warped, dtype=DT_INT)

        cv2.polylines(frame, [pts_warped], isClosed=True,
                      color=Color.YELLOW, thickness=5)

    def get_frame(self, bbs, players_center=None):
        frame_top_down = self.pitch_model.copy()
        self.draw_roi_(frame_top_down)

        self.draw_bbs_(frame_top_down, bbs, discard_extremes=True)
        self.draw_screen_pt_(
            frame_top_down,
            players_center
        )

        return frame_top_down
