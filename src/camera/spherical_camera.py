import cv2
import numpy as np
from camera.projective_camera import ProjectiveCamera
from utils.config import Config


class SphericalCamera(ProjectiveCamera):
    def __init__(self, frame_orig, config: Config):
        super().__init__(frame_orig, config)

    @property
    def fov_horiz_deg(self):
        ...

    @property
    def fov_vert_deg(self):
        ...

    def coords2ptz(self, x, y):
        ...

    def ptz2coords(self, theta_deg, phi_deg, f):
        ...

    def process_input(self, key, mouseX, mouseY):
        return super().process_input(key, mouseX, mouseY)

    def get_stats(self):
        stats = {
            "Name": SphericalCamera.__name__,
            "f": f"{self.zoom_f:.2f}",
            "pan_deg": f"{self.pan_deg:.4f}",
            "tilt_deg": f"{self.tilt_deg:.4f}",
            "fov_horiz_deg": self.fov_horiz_deg,
            "fov_vert_deg": self.fov_vert_deg,
            "players_vel": self.players_filter.vel.squeeze(1),
            "players_std": np.sqrt(self.players_var),
        }
        return stats
