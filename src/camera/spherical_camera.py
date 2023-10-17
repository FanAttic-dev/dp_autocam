from functools import cached_property
import cv2
import numpy as np
from camera.camera import Camera
from camera.projective_camera import ProjectiveCamera
from utils.config import Config
from utils.constants import INTERPOLATION_TYPE, Color, DrawingMode
import utils.utils as utils


class SphericalCamera(ProjectiveCamera):
    SENSOR_DX = 10
    FOV_DX = 5

    def __init__(self, frame_orig, config: Config, ignore_bounds=Config.autocam["debug"]["ignore_bounds"]):
        super().__init__(frame_orig, config, ignore_bounds)

    @property  # TODO: make cached
    def lens_fov_vert_deg(self):
        return self.lens_fov_horiz_deg / Camera.FRAME_ASPECT_RATIO  # 99

    @property  # TODO: make cached
    def limits(self):
        return np.deg2rad(np.array([self.lens_fov_horiz_deg, self.lens_fov_vert_deg], dtype=np.float32)) / 2

    @cached_property
    def coords_spherical_frame(self):
        coords_screen_frame = self._get_coords_screen_frame()
        return self._screen2spherical(coords_screen_frame)

    @cached_property
    def coords_spherical_corners(self):
        coords_screen_frame = self._get_coords_screen_corners()
        return self._screen2spherical(coords_screen_frame)

    @property
    def coords_screen_fov(self):
        coords_spherical_fov = self.coords_spherical_frame * \
            (self.fov_rad / 2 / self.limits)

        coords_spherical_fov = self._gnomonic(coords_spherical_fov)
        return self._spherical2screen(coords_spherical_fov)

    @property
    def coords_screen_corners(self):
        coords_spherical_fov = self.coords_spherical_corners * \
            (self.fov_rad / 2 / self.limits)

        coords_spherical_fov = self._gnomonic(coords_spherical_fov)
        return self._spherical2screen(coords_spherical_fov)

    def get_coords_screen_borders(self, skip):
        coords_screen_frame = self._get_coords_screen_borders(skip)

        coords_spherical_borders = self._screen2spherical(coords_screen_frame)
        coords_spherical_fov = coords_spherical_borders * \
            (self.fov_rad / 2 / self.limits)

        coords_spherical_fov = self._gnomonic(coords_spherical_fov)
        coords_screen_fov = self._spherical2screen(coords_spherical_fov)
        if Config.autocam["correct_rotation"]:
            _, coords_screen_fov = self.correct_rotation(coords_screen_fov)
        return (coords_screen_fov * self.frame_orig_size).astype(np.int32)

    def _get_coords_screen_frame(self):
        xx, yy = np.meshgrid(np.linspace(0, 1, Camera.FRAME_W),
                             np.linspace(0, 1, Camera.FRAME_H))
        return np.array([xx.ravel(), yy.ravel()], dtype=np.float32).T

    def _get_coords_screen_corners(self):
        return np.array([
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 0]
        ], dtype=np.float32)

    def _get_coords_screen_borders(self, skip):
        w = Camera.FRAME_W // skip
        h = Camera.FRAME_H // skip
        pts_w = np.linspace(0, 1, w)
        pts_h = np.linspace(0, 1, h)

        top = np.array([
            pts_w.copy(),
            np.zeros(w)
        ]).T
        right = np.array([
            np.ones(h),
            pts_h.copy()
        ]).T
        bottom = np.array([
            np.flip(pts_w.copy()),
            np.ones(w)
        ]).T
        left = np.array([
            np.zeros(h),
            pts_h.copy()
        ]).T

        return np.concatenate([top, right, bottom, left])

    def _screen2spherical(self, coord_screen):
        """ In range: [0, 1], out range: [-FoV_lens/2, FoV_lens/2] """

        return (coord_screen * 2 - 1) * self.limits

    def _spherical2screen(self, coord_spherical):
        """ In range: [-FoV_lens/2, FoV_lens/2], out range: [0, 1] """

        x, y = coord_spherical.T
        horiz_limit, vert_limit = self.limits
        x = (x / horiz_limit + 1.) * 0.5
        y = (y / vert_limit + 1.) * 0.5
        return np.array([x, y], dtype=np.float32).T

    def coords2ptz(self, x, y, f=None):
        coords_screen = np.array(
            [x, y], dtype=np.float32) / self.frame_orig_size
        coords_spherical = self._screen2spherical(coords_screen)
        pan_deg, tilt_deg = np.rad2deg(
            self._gnomonic_inverse(coords_spherical, (0, 0))
        )
        f = f if f is not None else self.zoom_f
        return pan_deg, tilt_deg, f

    def ptz2coords(self, pan_deg, tilt_deg, f=None):
        coords_spherical = np.deg2rad(
            np.array([pan_deg, tilt_deg], dtype=np.float32)
        )
        coords_spherical = self._gnomonic(coords_spherical, (0, 0))
        coords_screen = self._spherical2screen(coords_spherical)
        return (coords_screen * self.frame_orig_size).astype(np.int16)

    def get_corner_pts(self, correct_rotation):
        pts = self.coords_screen_corners * self.frame_orig_size

        if correct_rotation:
            _, pts = self.correct_rotation(pts)

        return np.array(pts, dtype=np.int32)

    def roi2original(self, coords_screen_roi):
        frame_size = np.array([Camera.FRAME_W, Camera.FRAME_H], dtype=np.int16)
        coords_screen = coords_screen_roi / frame_size

        if Config.autocam["correct_rotation"]:
            _, coords_screen = self.correct_rotation(
                coords_screen,
                center=np.array([0.5, 0.5], dtype=np.float32)
            )

        coords_spherical_roi = self._screen2spherical(coords_screen)
        coords_spherical_roi = coords_spherical_roi * \
            (self.fov_rad / 2 / self.limits)
        coords_spherical_roi = self._gnomonic(coords_spherical_roi)
        coords_screen = self._spherical2screen(coords_spherical_roi)

        coords_screen = (coords_screen * self.frame_orig_size).astype(np.int16)
        return coords_screen

    def get_frame(self, frame_orig):
        coords = self.coords_screen_fov

        if Config.autocam["correct_rotation"]:
            _, coords = self.correct_rotation(coords)

        return self._remap(frame_orig, coords)

    def _remap(self, frame_orig, coords):
        """ In range: [0, 1], Out img range: [0, frame_size] """

        frame_orig_h, frame_orig_w, _ = frame_orig.shape
        frame_size = [Camera.FRAME_H, Camera.FRAME_W]

        map_x = coords[:, 0] * frame_orig_w
        map_y = coords[:, 1] * frame_orig_h
        map_x = np.reshape(map_x, frame_size)
        map_y = np.reshape(map_y, frame_size)

        return cv2.remap(frame_orig, map_x, map_y, interpolation=INTERPOLATION_TYPE)

    def _gnomonic(self, coord_spherical, center=None):
        """ 
        Converts latitude (tilt) and longtitude (pan) to x, y coordinates
        based on the Gnomonic Projection.

        In range: [-FoV_lens/2, FoV_lens/2], out range: [-FoV_lens/2, FoV_lens/2]
        """

        lambda_rad = coord_spherical.T[0]
        phi_rad = coord_spherical.T[1]

        if center is None:
            center = -np.deg2rad([self.pan_deg, self.tilt_deg])
        center_pan_rad, center_tilt_rad = center

        sin_phi = np.sin(phi_rad)
        cos_phi = np.cos(phi_rad)
        cos_c = np.sin(center_tilt_rad) * sin_phi + np.cos(center_tilt_rad) * \
            cos_phi * np.cos(lambda_rad - center_pan_rad)
        x = (cos_phi * np.sin(lambda_rad - center_pan_rad)) / cos_c
        y = (np.cos(center_tilt_rad) * sin_phi - np.sin(center_tilt_rad) *
             cos_phi * np.cos(lambda_rad - center_pan_rad)) / cos_c

        return np.array([x, y]).T

    def _gnomonic_inverse(self, coord_spherical, center=None):
        """ 
        Converts x, y coodinates obtained by the Gnomonic Projection
        to latitude (tilt) and longtitude (pan).

        In range: [-FoV_lens/2, FoV_lens/2], out range: [-FoV_lens/2, FoV_lens/2]
        """

        x = coord_spherical.T[0]
        y = coord_spherical.T[1]

        if center is None:
            center = -np.deg2rad([self.pan_deg, self.tilt_deg])
        center_pan_rad, center_tilt_rad = center

        rou = np.sqrt(x ** 2 + y ** 2)
        c = np.arctan(rou)
        sin_c = np.sin(c)
        cos_c = np.cos(c)

        lat = np.arcsin(
            cos_c * np.sin(center_tilt_rad) + (y * sin_c * np.cos(center_tilt_rad)) / rou)
        lon = center_pan_rad + np.arctan2(x * sin_c, rou * np.cos(center_tilt_rad)
                                          * cos_c - y * np.sin(center_tilt_rad) * sin_c)

        return np.array([lon, lat]).T

    def draw_roi_(self, frame_orig, color=Color.VIOLET, drawing_mode=DrawingMode.LINES):
        if drawing_mode == DrawingMode.LINES:
            skip = 1
            coords = self.get_coords_screen_borders(skip)
            cv2.polylines(frame_orig, [coords], True, color, thickness=5)
        elif drawing_mode == DrawingMode.CIRCLES:
            skip = 50
            coords = self.get_coords_screen_borders(skip)
            for x, y in coords:
                cv2.circle(frame_orig, [x, y], radius=5,
                           color=color, thickness=-1)

    def draw_frame_mask(self, frame_orig):
        skip = 50

        mask = np.zeros(frame_orig.shape[:2], dtype=np.uint8)
        pts = self.get_coords_screen_borders(skip)
        cv2.fillPoly(mask, [pts], 255)
        return cv2.bitwise_and(frame_orig, frame_orig, mask=mask)

    def draw_grid_(self, frame_orig, color=Color.BLUE):
        skip = 40

        frame_orig_w, frame_orig_h = self.frame_orig_size
        xx, yy = np.meshgrid(np.linspace(0, 1, frame_orig_w // skip),
                             np.linspace(0, 1, frame_orig_h // skip))
        coords = np.array([xx.ravel(), yy.ravel()], dtype=np.float32).T

        coords = self._screen2spherical(coords)
        coords = self._gnomonic(coords)
        coords = self._spherical2screen(coords)

        coords = (coords * self.frame_orig_size).astype(np.int32)

        for x, y in coords:
            cv2.circle(frame_orig, [x, y], radius=5,
                       color=color, thickness=-1)

    def draw_zoom_target_(self, frame_orig):
        target_w = 1000
        target_h = 200

        # w, h = self.frame_orig_size
        # w / self.lens_fov_horiz_deg

        x1 = self.frame_orig_center_x - target_w // 2
        y1 = self.frame_orig_center_y - target_h // 2
        x2 = self.frame_orig_center_x + target_w // 2
        y2 = self.frame_orig_center_y + target_h // 2

        cv2.rectangle(frame_orig, (x1, y1), (x2, y2),
                      color=Color.ORANGE, thickness=5)

    def process_input(self, key, mouseX, mouseY):
        is_alive = True
        if key == ord('8'):
            self.sensor_w += SphericalCamera.SENSOR_DX
        elif key == ord('2'):
            self.sensor_w -= SphericalCamera.SENSOR_DX
        elif key == ord('6'):
            self.lens_fov_horiz_deg += SphericalCamera.FOV_DX
        elif key == ord('4'):
            self.lens_fov_horiz_deg -= SphericalCamera.FOV_DX
        else:
            is_alive = super().process_input(key, mouseX, mouseY)
        return is_alive

    def get_stats(self):
        stats = {
            "Name": SphericalCamera.__name__,
            "f": f"{self.zoom_f:.2f}",
            "sensor_w": self.sensor_w,
            "lens_fov_horiz_deg": self.lens_fov_horiz_deg,
            "lens_fov_vert_deg": self.lens_fov_vert_deg,
            "pan_deg": f"{self.pan_deg:.4f}",
            "tilt_deg": f"{self.tilt_deg:.4f}",
            "fov_horiz_deg": self.fov_horiz_deg,
            "fov_vert_deg": self.fov_vert_deg,
            "center": self.center,
        }
        return stats
