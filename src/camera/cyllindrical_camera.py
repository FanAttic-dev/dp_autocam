import cv2
import numpy as np
from camera.camera import Camera
from camera.projective_camera import ProjectiveCamera
from utils.config import Config
from utils.constants import INTERPOLATION_TYPE, Color


class CyllindricalCamera(ProjectiveCamera):
    SENSOR_W = 36

    def __init__(self, frame_orig, config: Config):
        super().__init__(frame_orig, config)
        self.sensor_w = CyllindricalCamera.SENSOR_W

    @property
    def center(self):
        return self.ptz2coords(self.pan_deg, self.tilt_deg)

    @property
    def corners_ang(self):
        return {
            "left top": [-self.fov_horiz_deg / 2, -self.fov_vert_deg / 2],
            "left bottom": [-self.fov_horiz_deg / 2, self.fov_vert_deg / 2],
            "right bottom": [self.fov_horiz_deg / 2, self.fov_vert_deg / 2],
            "right top": [self.fov_horiz_deg / 2, -self.fov_vert_deg / 2],
        }

    def get_corner_pts(self):
        pts = [
            self.ptz2coords(
                self.pan_deg + pan_deg,
                self.tilt_deg + tilt_deg)
            for pan_deg, tilt_deg in self.corners_ang.values()
        ]
        return np.array(pts, dtype=np.int32)

    def set_center(self, x, y, f=None):
        pan_deg, tilt_deg = self.coords2ptz(x, y)
        f = f if f is not None else self.zoom_f
        self.set_ptz(pan_deg, tilt_deg, f)

    @property
    def H_inv(self):
        return np.linalg.inv(self.H)

    @property
    def fov_horiz_deg(self):
        return np.rad2deg(2 * np.arctan(self.sensor_w / (2 * self.zoom_f)))

    @property
    def fov_vert_deg(self):
        return self.fov_horiz_deg / Camera.FRAME_ASPECT_RATIO

    def ptz2coords(self, theta_deg, phi_deg):
        theta_rad = np.deg2rad(theta_deg)
        x = np.tan(theta_rad) * self.cyllinder_radius

        phi_rad = np.deg2rad(phi_deg)
        y = np.tan(phi_rad) * \
            np.sqrt(self.cyllinder_radius**2 + x**2)
        return self.shift_coords(int(x), int(y))

    def coords2ptz(self, x, y):
        x, y = self.unshift_coords(x, y)
        pan_deg = np.rad2deg(np.arctan(x / self.cyllinder_radius))
        tilt_deg = np.rad2deg(
            np.arctan(y / (np.sqrt(self.cyllinder_radius**2 + x**2))))
        return pan_deg, tilt_deg

    def get_frame(self, frame_orig):
        return cv2.warpPerspective(
            frame_orig,
            self.H,
            (ProjectiveCamera.FRAME_W, ProjectiveCamera.FRAME_H),
            flags=INTERPOLATION_TYPE
        )

    def draw_roi_(self, frame_orig, color=Color.YELLOW):
        pts = self.get_corner_pts()
        cv2.polylines(frame_orig, [pts], True, color, thickness=10)

    def draw_frame_mask(self, frame_orig):
        mask = np.zeros(frame_orig.shape[:2], dtype=np.uint8)
        pts = self.get_corner_pts()
        cv2.fillPoly(mask, [pts], 255)
        return cv2.bitwise_and(frame_orig, frame_orig, mask=mask)

    def process_input(self, key, mouseX, mouseY):
        is_alive = True
        if key == ord('6'):
            self.cyllinder_radius += 50
        elif key == ord('4'):
            self.cyllinder_radius -= 50
        elif key == ord('8'):
            self.sensor_w += 10
        elif key == ord('2'):
            self.sensor_w -= 10
        else:
            is_alive = super().process_input(key, mouseX, mouseY)
        return is_alive

    def get_stats(self):
        stats = {
            "Name": CyllindricalCamera.__name__,
            "f": f"{self.zoom_f:.2f}",
            "sensor_w": self.sensor_w,
            "cyllinder_r": self.cyllinder_radius,
            "pan_deg": f"{self.pan_deg:.4f}",
            "tilt_deg": f"{self.tilt_deg:.4f}",
            # "fov_horiz_deg": self.fov_horiz_deg,
            # "fov_vert_deg": self.fov_vert_deg,
            "players_vel": self.players_filter.vel.squeeze(1),
            "players_std": np.sqrt(self.players_var) if self.players_var is not None else "",
        }
        return stats
