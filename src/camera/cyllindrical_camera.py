import cv2
import numpy as np
from camera.camera import Camera
from camera.projective_camera import ProjectiveCamera
from utils.config import Config
from utils.helpers import get_pitch_rotation_rad, rotate_pts


class CyllindricalCamera(ProjectiveCamera):
    SENSOR_W = 100

    def __init__(self, frame_orig, config: Config):
        super().__init__(frame_orig, config)
        self.sensor_w = CyllindricalCamera.SENSOR_W

    @property
    def H(self):
        src = self.get_corner_pts()
        dst = Camera.FRAME_CORNERS

        H, _ = cv2.findHomography(src, dst)

        if Config.autocam["correct_rotation"]:
            # TODO: use lookup table
            pitch_coords_orig = self.config.pitch_coords_pts
            pitch_coords_frame = cv2.perspectiveTransform(
                pitch_coords_orig.astype(np.float64), H)
            roll_rad = get_pitch_rotation_rad(pitch_coords_frame)

            src = np.array(rotate_pts(src, roll_rad), dtype=np.int32)
            H, _ = cv2.findHomography(src, dst)

        return H

    @property
    def center(self):
        return self.ptz2coords(self.pan_deg, self.tilt_deg)

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
        return self.fov_horiz_deg / 16 * 9

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

    def process_input(self, key, mouseX, mouseY):
        is_alive = True
        if key == ord('c'):
            self.cyllinder_radius += 10
        elif key == ord('v'):
            self.cyllinder_radius -= 10
        elif key == ord('+'):
            self.sensor_w += 1
        elif key == ord('-'):
            self.sensor_w -= 1
        else:
            is_alive = super().process_input(key, mouseX, mouseY)
        return is_alive

    def get_stats(self):
        stats = {
            "Name": CyllindricalCamera.__name__,
            "f": f"{self.zoom_f:.2f}",
            # "sensor_w": self.sensor_w,
            # "cyllinder_r": self.cyllinder_radius,
            "pan_deg": f"{self.pan_deg:.4f}",
            "tilt_deg": f"{self.tilt_deg:.4f}",
            # "fov_horiz_deg": self.fov_horiz_deg,
            # "fov_vert_deg": self.fov_vert_deg,
            "players_vel": self.players_filter.vel.squeeze(1),
            "players_std": np.sqrt(self.players_var) if self.players_var is not None else "",
        }
        return stats
