from functools import cached_property
import cv2
import numpy as np
from camera.camera import Camera
from utils.config import Config
from utils.constants import DT_FLOAT, DT_INT, INTERPOLATION_TYPE, Color, DrawingMode
import utils.utils as utils


class SphericalCamera(Camera):
    SENSOR_DX = 10
    FOV_DX = 5
    DRAWING_STEP = 50

    @property  # could use @cached_property for optimization
    def lens_fov_vert_deg(self):
        return self.lens_fov_horiz_deg / Camera.FRAME_ASPECT_RATIO  # 99

    @property  # could use @cached_property for optimization
    def limits(self):
        limits = np.array(
            [self.lens_fov_horiz_deg, self.lens_fov_vert_deg],
            dtype=DT_FLOAT
        ) / 2
        return np.deg2rad(limits)

    def _screen2spherical(self, coord_screen):
        """ In range: [0, 1], out range: [-FoV_lens/2, FoV_lens/2] """
        return (coord_screen * 2 - 1) * self.limits

    def _spherical2screen(self, coord_spherical):
        """ In range: [-FoV_lens/2, FoV_lens/2], out range: [0, 1] """
        x, y = coord_spherical.T
        horiz_limit, vert_limit = self.limits
        x = (x / horiz_limit + 1.) * 0.5
        y = (y / vert_limit + 1.) * 0.5
        return np.array([x, y], dtype=DT_FLOAT).T

    def spherical2screen_fov(
        self,
        pts_spherical,
        correct_rotation=Config.autocam["correct_rotation"],
        normalized=False
    ):
        pts_spherical_fov = pts_spherical * \
            (self.fov_rad / 2 / self.limits)

        pts_spherical_fov = self._gnomonic(pts_spherical_fov)
        pts_screen_fov = self._spherical2screen(pts_spherical_fov)

        if correct_rotation:
            gnomonic_center = self.center / self.frame_orig_size
            _, pts_screen_fov = self.correct_rotation(
                pts_screen_fov,
                gnomonic_center
            )

        if normalized:
            return pts_screen_fov

        return (pts_screen_fov * self.frame_orig_size).astype(DT_INT)

    def screen2ptz(self, x, y, f=None):
        pts_screen = np.array(
            [x, y], dtype=DT_FLOAT) / self.frame_orig_size
        pts_spherical = self._screen2spherical(pts_screen)
        pan_deg, tilt_deg = np.rad2deg(
            self._gnomonic_inverse(pts_spherical, (0, 0))
        )
        f = f if f is not None else self.zoom_f
        return pan_deg, tilt_deg, f

    def ptz2screen(self, pan_deg, tilt_deg, f=None):
        pts_spherical = np.deg2rad(
            np.array([pan_deg, tilt_deg], dtype=DT_FLOAT)
        )
        pts_spherical = self._gnomonic(pts_spherical, (0, 0))
        pts_screen = self._spherical2screen(pts_spherical)
        return (pts_screen * self.frame_orig_size).astype(DT_INT)

    # region Frame points
    def _get_pts_frame_screen(self):
        xx, yy = np.meshgrid(np.linspace(0, 1, Camera.FRAME_W),
                             np.linspace(0, 1, Camera.FRAME_H))
        return np.array([xx.ravel(), yy.ravel()], dtype=DT_FLOAT).T

    @cached_property
    def pts_frame_spherical(self):
        pts_screen_frame = self._get_pts_frame_screen()
        return self._screen2spherical(pts_screen_frame)
    # endregion

    # region Corner points
    def _get_pts_corners_screen(self):
        return np.array([
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 0]
        ], dtype=DT_FLOAT)

    @cached_property
    def pts_corners_spherical(self):
        pts_screen_frame = self._get_pts_corners_screen()
        return self._screen2spherical(pts_screen_frame)

    def get_pts_corners(self, correct_rotation):
        return self.spherical2screen_fov(
            self.pts_corners_spherical,
            correct_rotation
        )
    # endregion

    # region Border points
    def _get_pts_borders_screen(self):
        w = Camera.FRAME_W // SphericalCamera.DRAWING_STEP
        h = Camera.FRAME_H // SphericalCamera.DRAWING_STEP
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

    @cached_property
    def pts_borders_spherical(self):
        pts_borders_screen = self._get_pts_borders_screen()
        return self._screen2spherical(pts_borders_screen)

    @property
    def pts_borders_screen_fov(self):
        return self.spherical2screen_fov(
            self.pts_borders_spherical
        )
    # endregion

    def roi2original(self, pts_roi_screen):
        pts_screen = pts_roi_screen / self.frame_roi_size
        pts_spherical = self._screen2spherical(pts_screen)

        return self.spherical2screen_fov(
            pts_spherical,
            normalized=False
        )

    def get_frame(self, frame_orig):
        pts_fov_screen = self.spherical2screen_fov(
            self.pts_frame_spherical,
            normalized=True
        )
        return self._remap(frame_orig, pts_fov_screen)

    def _remap(self, frame_orig, pts_screen):
        """ Creates a new frame by mapping from frame_orig based on pts.

        Args:
            frame_orig: Frame to be sampled from.
            pts: Mapping scheme.
                Range: [0, 1]

        Returns:
            Remapped image
                Range: [0, frame_size]
        """
        frame_orig_h, frame_orig_w, _ = frame_orig.shape

        map_x = pts_screen[:, 0] * frame_orig_w
        map_y = pts_screen[:, 1] * frame_orig_h
        map_x = np.reshape(map_x, self.frame_roi_size[::-1])
        map_y = np.reshape(map_y, self.frame_roi_size[::-1])

        return cv2.remap(frame_orig, map_x, map_y, interpolation=INTERPOLATION_TYPE)

    def _gnomonic(self, coord_spherical, center=None):
        """ Convert latitude (tilt) and longtitude (pan) to x, y coordinates.

        The conversion is based on the Gnomonic Projection.
        https://mathworld.wolfram.com/GnomonicProjection.html

        Args:
            coord_spherical:
                Range: [-FoV_lens/2, FoV_lens/2]
            center: Center of projection.

        Returns:
            x, y coordinates
                Range: [-FoV_lens/2, FoV_lens/2]
        """
        lambda_rad = coord_spherical.T[0]  # pan
        phi_rad = coord_spherical.T[1]  # tilt

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
        """Convert x, y coordinates to latitude (tilt) and longtitude (pan).

        It assumes that the x, y coordinates have been obtained by the Gnomonic Projection.
        https://mathworld.wolfram.com/GnomonicProjection.html

        Args:
            coord_spherical:
                In range: [-FoV_lens/2, FoV_lens/2]
                Out range: [-FoV_lens/2, FoV_lens/2]
            center: Center of projection.
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

    # -------------------------------------------------------------------------
    # Drawing

    def draw_roi_(self, frame_orig, color=Color.VIOLET, drawing_mode=DrawingMode.LINES):
        if drawing_mode == DrawingMode.LINES:
            pts = self.pts_borders_screen_fov
            cv2.polylines(frame_orig, [pts], True, color, thickness=5)
        elif drawing_mode == DrawingMode.CIRCLES:
            pts = self.pts_borders_screen_fov
            for x, y in pts:
                cv2.circle(frame_orig, [x, y], radius=5,
                           color=color, thickness=-1)

    def draw_frame_mask(self, frame_orig):
        mask = np.zeros(frame_orig.shape[:2], dtype=np.uint8)
        pts = self.pts_borders_screen_fov
        cv2.fillPoly(mask, [pts], 255)
        return cv2.bitwise_and(frame_orig, frame_orig, mask=mask)

    def draw_grid_(self, frame_orig, color=Color.BLUE):
        frame_orig_w, frame_orig_h = self.frame_orig_size
        xx, yy = np.meshgrid(np.linspace(0, 1, frame_orig_w // SphericalCamera.DRAWING_STEP),
                             np.linspace(0, 1, frame_orig_h // SphericalCamera.DRAWING_STEP))
        pts = np.array([xx.ravel(), yy.ravel()], dtype=DT_FLOAT).T

        pts = self._screen2spherical(pts)
        pts = self._gnomonic(pts)
        pts = self._spherical2screen(pts)

        pts = (pts * self.frame_orig_size).astype(DT_INT)

        for x, y in pts:
            cv2.circle(frame_orig, [x, y], radius=5,
                       color=color, thickness=-1)

    def process_input(self, key, mouse_pos):
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
            is_alive = super().process_input(key, mouse_pos)
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
