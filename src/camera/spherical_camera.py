import cv2
import numpy as np
from camera.projective_camera import ProjectiveCamera
from utils.config import Config
from utils.constants import colors


class SphericalCamera(ProjectiveCamera):
    SENSOR_W = 100
    ZOOM_DZ = 10
    SENSOR_W_DX = 10
    R_DX = .05

    def __init__(self, frame_orig, config: Config):
        self.r = 1
        super().__init__(frame_orig, config)
        self.sensor_w = SphericalCamera.SENSOR_W

    @property
    def fov_horiz_deg(self):
        return np.rad2deg(2 * np.arctan(self.sensor_w / (2 * self.zoom_f)))

    @property
    def fov_vert_deg(self):
        return self.fov_horiz_deg / 16 * 9

    def draw_grid_(self, frame_orig):
        lens_fov = 120

        step = 1
        limit = lens_fov / 2
        pans = np.arange(-limit, limit, step)
        tilts = np.arange(-limit, limit, step)
        for pan_deg in pans:
            for tilt_deg in tilts:
                x, y = self.ptz2coords(pan_deg, tilt_deg, self.zoom_f)
                cv2.circle(frame_orig, (x, y), radius=5,
                           color=colors["violet"], thickness=-1)

    def ptz2coords(self, pan_deg, tilt_deg, zoom_f):
        """ https://mathworld.wolfram.com/GnomonicProjection.html """
        lambda_deg = pan_deg
        lambda_rad = np.deg2rad(lambda_deg)
        phi_deg = tilt_deg
        phi_rad = np.deg2rad(phi_deg)

        lambda_0 = 0
        phi_1 = 0

        cos_c = np.sin(phi_1) * np.sin(phi_rad) + np.cos(phi_1) * \
            np.cos(phi_rad) * np.cos(lambda_rad - lambda_0)
        x = self.r * (np.cos(phi_rad) *
                      np.sin(lambda_rad - lambda_0)) / cos_c
        y = self.r * (np.cos(phi_1) * np.sin(phi_rad) - np.sin(phi_1)
                      * np.cos(phi_rad) * np.cos(lambda_rad - lambda_0)) / cos_c

        h, w, _ = self.frame_orig_shape
        x = x * w
        y = y * h
        return self.shift_coords(int(x), int(y))

    def coords2ptz(self, x, y):
        x, y = self.unshift_coords(x, y)
        h, w, _ = self.frame_orig_shape
        x = x / w
        y = y / h

        lambda_0 = 0
        phi_1 = 0

        rho = np.sqrt(x**2 + y**2)
        c = np.arctan(rho / self.r)

        phi_rad = np.arcsin(np.cos(c) * np.sin(phi_1) +
                            y * np.sin(c) * np.cos(phi_1) / rho)
        lambda_rad = lambda_0 + \
            np.arctan2(x * np.sin(c), (rho * np.cos(phi_1) *
                                       np.cos(c) - y * np.sin(phi_1) * np.sin(c)))

        return np.rad2deg(lambda_rad), np.rad2deg(phi_rad)

    def process_input(self, key, mouseX, mouseY):
        is_alive = True
        if key == ord('8'):
            self.sensor_w += SphericalCamera.SENSOR_W_DX
        elif key == ord('2'):
            self.sensor_w -= SphericalCamera.SENSOR_W_DX
        elif key == ord('6'):
            self.zoom(SphericalCamera.ZOOM_DZ)
        elif key == ord('4'):
            self.zoom(-SphericalCamera.ZOOM_DZ)
        elif key == ord('+'):
            self.r += SphericalCamera.R_DX
        elif key == ord('-'):
            self.r -= SphericalCamera.R_DX
        else:
            is_alive = super().process_input(key, mouseX, mouseY)
        return is_alive

    def get_stats(self):
        stats = {
            "Name": SphericalCamera.__name__,
            "f": f"{self.zoom_f:.2f}",
            "sensor_w": self.sensor_w,
            "r": self.r,
            "pan_deg": f"{self.pan_deg:.4f}",
            "tilt_deg": f"{self.tilt_deg:.4f}",
            "fov_horiz_deg": self.fov_horiz_deg,
            "fov_vert_deg": self.fov_vert_deg,
            "players_vel": self.players_filter.vel.squeeze(1),
        }
        return stats
