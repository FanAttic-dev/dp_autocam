from functools import cached_property
import cv2
import numpy as np
from camera.camera import Camera
from camera.projective_camera import ProjectiveCamera
from utils.config import Config
from utils.constants import INTERPOLATION_TYPE, Color, DrawingMode


class SphericalCamera(ProjectiveCamera):
    ZOOM_DZ = 10
    SENSOR_DX = 10
    FOV_DX = 5

    def __init__(self, frame_orig, config: Config):
        self.lens_fov_horiz_deg = 115
        self.sensor_w = 36  # FX sensor size

        super().__init__(frame_orig, config)

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

    @property
    def coords_spherical_fov(self):
        coords = self.coords_spherical_frame * \
            (self.fov_rad / 2 / self.limits)

        return self._gnomonic_forward(coords)

    @property
    def coords_screen_fov(self):
        return self._spherical2screen(self.coords_spherical_fov)

    def _get_coords_screen_frame(self):
        xx, yy = np.meshgrid(np.linspace(0, 1, Camera.FRAME_W),
                             np.linspace(0, 1, Camera.FRAME_H))
        return np.array([xx.ravel(), yy.ravel()], dtype=np.float32).T

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

    def set_center(self, x, y, f=None):
        coords_screen = np.array(
            [x, y], dtype=np.float32) / self.frame_orig_size
        pan_deg, tilt_deg = np.rad2deg(self._screen2spherical(coords_screen))
        f = f if f is not None else self.zoom_f
        self.set_ptz(pan_deg, tilt_deg, f)

    @property
    def center(self):
        coords_screen = self._spherical2screen(
            np.array([self.pan_deg, self.tilt_deg], dtype=np.float32))
        return (coords_screen * self.frame_orig_size).astype(np.uint16)

    @property
    def fov_rad(self):
        return np.deg2rad(np.array([self.fov_horiz_deg, self.fov_vert_deg]), dtype=np.float32)

    @property
    def fov_horiz_deg(self):
        return np.rad2deg(2 * np.arctan(self.sensor_w / (2 * self.zoom_f)))

    @property
    def fov_vert_deg(self):
        return self.fov_horiz_deg / Camera.FRAME_ASPECT_RATIO

    def get_corner_pts(self):
        coords = self.coords_screen_fov * self.frame_orig_size
        coords = np.reshape(coords, (Camera.FRAME_H, Camera.FRAME_W, 2))

        lt = coords[0, 0]
        lb = coords[-1, 0]
        rb = coords[-1, -1]
        rt = coords[0, -1]

        return np.array([lt, lb, rb, rt], dtype=np.int32)

    def _gnomonic_forward(self, coord_spherical):
        """ In/out range: [-FoV_lens/2, FoV_lens/2] """

        lambda_rad = coord_spherical.T[0]
        phi_rad = coord_spherical.T[1]

        center_pan_rad = -np.deg2rad(self.pan_deg)
        center_tilt_rad = -np.deg2rad(self.tilt_deg)

        cos_c = np.sin(center_tilt_rad) * np.sin(phi_rad) + np.cos(center_tilt_rad) * \
            np.cos(phi_rad) * np.cos(lambda_rad - center_pan_rad)
        x = (np.cos(phi_rad) *
             np.sin(lambda_rad - center_pan_rad)) / cos_c
        y = (np.cos(center_tilt_rad) * np.sin(phi_rad) - np.sin(center_tilt_rad)
             * np.cos(phi_rad) * np.cos(lambda_rad - center_pan_rad)) / cos_c

        return np.array([x, y]).T

    def _gnomonic_inverse(self, coord_spherical):
        """ In/out range: [-FoV_lens/2, FoV_lens/2] """

        x = coord_spherical.T[0]
        y = coord_spherical.T[1]

        rou = np.sqrt(x ** 2 + y ** 2)
        c = np.arctan(rou)
        sin_c = np.sin(c)
        cos_c = np.cos(c)

        lat = np.arcsin(
            cos_c * np.sin(self.cp[1]) + (y * sin_c * np.cos(self.cp[1])) / rou)
        lon = self.cp[0] + np.arctan2(x * sin_c, rou * np.cos(self.cp[1])
                                      * cos_c - y * np.sin(self.cp[1]) * sin_c)

        return np.array([lon, lat]).T

    def _remap(self, frame_orig, coords):
        """ In range: [0, 1], Out img range: [0, frame_size] """

        frame_orig_h, frame_orig_w, _ = frame_orig.shape
        frame_size = [Camera.FRAME_H, Camera.FRAME_W]

        map_x = coords[:, 0] * frame_orig_w
        map_y = coords[:, 1] * frame_orig_h
        map_x = np.reshape(map_x, frame_size)
        map_y = np.reshape(map_y, frame_size)

        return cv2.remap(frame_orig, map_x, map_y, interpolation=INTERPOLATION_TYPE)

    def get_frame(self, frame_orig):
        return self._remap(frame_orig, self.coords_screen_fov)

    def get_roi_border_pts(self, skip=50):
        coords = self.coords_screen_fov
        coords = np.reshape(coords, (Camera.FRAME_H, Camera.FRAME_W, 2))
        coords = (coords * self.frame_orig_size).astype(np.int32)

        top = coords[0, ::skip, :]
        right = coords[::skip, -1, :]
        bottom = coords[-1, ::skip, :]
        left = coords[::skip, 0, :]

        return np.concatenate([top, right, np.flip(bottom, axis=0), left])

    def draw_roi_(self, frame_orig, color=Color.VIOLET, drawing_mode=DrawingMode.LINES):
        if drawing_mode == DrawingMode.LINES:
            skip = 5
            coords = self.get_roi_border_pts(skip)
            cv2.polylines(frame_orig, [coords], True, color, thickness=10)
        elif drawing_mode == DrawingMode.CIRCLES:
            skip = 50
            coords = self.get_roi_border_pts(skip)
            for x, y in coords:
                cv2.circle(frame_orig, [x, y], radius=5,
                           color=color, thickness=-1)

    def draw_frame_mask(self, frame_orig):
        skip = 50

        mask = np.zeros(frame_orig.shape[:2], dtype=np.uint8)
        pts = self.get_roi_border_pts(skip)
        cv2.fillPoly(mask, [pts], 255)
        return cv2.bitwise_and(frame_orig, frame_orig, mask=mask)

    def draw_grid_(self, frame_orig, color=Color.YELLOW):
        skip = 50

        frame_orig_w, frame_orig_h = self.frame_orig_size
        xx, yy = np.meshgrid(np.linspace(0, 1, frame_orig_w // skip),
                             np.linspace(0, 1, frame_orig_h // skip))
        coords = np.array([xx.ravel(), yy.ravel()], dtype=np.float32).T

        coords = self._screen2spherical(coords)
        coords = self._gnomonic_forward(coords)
        coords = self._spherical2screen(coords)

        coords = (coords * self.frame_orig_size).astype(np.int32)

        for x, y in coords:
            cv2.circle(frame_orig, [x, y], radius=5,
                       color=color, thickness=-1)

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
            "players_vel": self.players_filter.vel.squeeze(1),
        }
        return stats
