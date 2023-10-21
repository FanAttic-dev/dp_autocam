from functools import cached_property
import cv2
import numpy as np
from camera.camera import Camera
from utils.config import Config
from utils.constants import INTERPOLATION_TYPE, Color, DrawingMode
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
            dtype=np.float32
        ) / 2
        return np.deg2rad(limits)

    def coords_spherical2screen_fov(
        self,
        coords_spherical,
        correct_rotation=Config.autocam["correct_rotation"],
        normalized=False
    ):
        coords_spherical_fov = coords_spherical * \
            (self.fov_rad / 2 / self.limits)

        coords_spherical_fov = self._gnomonic(coords_spherical_fov)
        coored_screen_fov = self._spherical2screen(coords_spherical_fov)

        if correct_rotation:
            _, coored_screen_fov = self.correct_rotation(coored_screen_fov)

        if normalized:
            return coored_screen_fov

        return (coored_screen_fov * self.frame_orig_size).astype(np.int32)

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

    # region Coords Frame
    def _get_coords_screen_frame(self):
        xx, yy = np.meshgrid(np.linspace(0, 1, Camera.FRAME_W),
                             np.linspace(0, 1, Camera.FRAME_H))
        return np.array([xx.ravel(), yy.ravel()], dtype=np.float32).T

    @cached_property
    def coords_spherical_frame(self):
        coords_screen_frame = self._get_coords_screen_frame()
        return self._screen2spherical(coords_screen_frame)
    # endregion

    # region Coords Corners
    def _get_coords_screen_corners(self):
        return np.array([
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 0]
        ], dtype=np.float32)

    @cached_property
    def coords_spherical_corners(self):
        coords_screen_frame = self._get_coords_screen_corners()
        return self._screen2spherical(coords_screen_frame)

    @property
    def coords_screen_corners(self):
        coords_spherical_fov = self.coords_spherical_corners * \
            (self.fov_rad / 2 / self.limits)

        coords_spherical_fov = self._gnomonic(coords_spherical_fov)
        return self._spherical2screen(coords_spherical_fov)

    def get_corner_pts(self, correct_rotation):
        return self.coords_spherical2screen_fov(
            self.coords_spherical_corners,
            correct_rotation
        )
    # endregion

    # region Coords Corners
    def _get_coords_screen_borders(self):
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
    def coords_spherical_borders(self):
        coords_screen_frame = self._get_coords_screen_borders()
        return self._screen2spherical(coords_screen_frame)

    def get_coords_screen_borders(self):
        return self.coords_spherical2screen_fov(
            self.coords_spherical_borders
        )
    # endregion

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
        coords_screen_fov = self.coords_spherical2screen_fov(
            self.coords_spherical_frame,
            normalized=True
        )
        return self._remap(frame_orig, coords_screen_fov)

    def _remap(self, frame_orig, coords):
        """ Creates a new frame by mapping from frame_orig based on coords.

        Args:
            frame_orig: Frame to be sampled from.
            coords: Mapping scheme.
                Range: [0, 1]

        Returns:
            Remapped image
                Range: [0, frame_size]
        """
        frame_orig_h, frame_orig_w, _ = frame_orig.shape
        frame_size = [Camera.FRAME_H, Camera.FRAME_W]

        map_x = coords[:, 0] * frame_orig_w
        map_y = coords[:, 1] * frame_orig_h
        map_x = np.reshape(map_x, frame_size)
        map_y = np.reshape(map_y, frame_size)

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
            coords = self.get_coords_screen_borders()
            cv2.polylines(frame_orig, [coords], True, color, thickness=5)
        elif drawing_mode == DrawingMode.CIRCLES:
            coords = self.get_coords_screen_borders()
            for x, y in coords:
                cv2.circle(frame_orig, [x, y], radius=5,
                           color=color, thickness=-1)

    def draw_frame_mask(self, frame_orig):
        mask = np.zeros(frame_orig.shape[:2], dtype=np.uint8)
        pts = self.get_coords_screen_borders()
        cv2.fillPoly(mask, [pts], 255)
        return cv2.bitwise_and(frame_orig, frame_orig, mask=mask)

    def draw_grid_(self, frame_orig, color=Color.BLUE):
        frame_orig_w, frame_orig_h = self.frame_orig_size
        xx, yy = np.meshgrid(np.linspace(0, 1, frame_orig_w // SphericalCamera.DRAWING_STEP),
                             np.linspace(0, 1, frame_orig_h // SphericalCamera.DRAWING_STEP))
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
