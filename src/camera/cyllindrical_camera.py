import cv2
import numpy as np
from camera.camera import Camera
from camera.projective_camera import ProjectiveCamera
from utils.config import Config
from utils.constants import INTERPOLATION_TYPE, Color
import utils.utils as utils


class CyllindricalCamera(ProjectiveCamera):
    def __init__(self, frame_orig, config: Config):
        self.cyllinder_radius = 1860
        super().__init__(frame_orig, config)

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
        ptz = self.coords2ptz(x, y, f)
        self.set_ptz(*ptz)

    def try_set_center(self, x, y, f=None):
        ptz = self.coords2ptz(x, y, f)
        return self.try_set_ptz(*ptz)

    @property
    def H_inv(self):
        return np.linalg.inv(self.H)

    @property
    def H(self):
        src = self.get_corner_pts()
        dst = Camera.FRAME_CORNERS

        if Config.autocam["correct_rotation"]:
            H = self.correct_rotation(src, dst)
        else:
            H, _ = cv2.findHomography(src, dst)

        return H

    def correct_rotation(self, src, dst):
        # TODO: use lookup table
        H, _ = cv2.findHomography(src, dst)
        pitch_coords_orig = self.config.pitch_coords_pts
        pitch_coords_frame = cv2.perspectiveTransform(
            pitch_coords_orig.astype(np.float64), H)
        roll_rad = utils.get_pitch_rotation_rad(pitch_coords_frame)

        src = np.array(utils.rotate_pts(src, roll_rad), dtype=np.int32)
        H, _ = cv2.findHomography(src, dst)
        return H

    def ptz2coords(self, pan_deg, tilt_deg):
        pan_rad = np.deg2rad(pan_deg)
        x = np.tan(pan_rad) * self.cyllinder_radius

        tilt_rad = np.deg2rad(tilt_deg)
        y = np.tan(tilt_rad) * \
            np.sqrt(self.cyllinder_radius**2 + x**2)
        return self.shift_coords(int(x), int(y))

    def coords2ptz(self, x, y, f=None):
        x, y = self.unshift_coords(x, y)
        pan_deg = np.rad2deg(np.arctan(x / self.cyllinder_radius))
        tilt_deg = np.rad2deg(
            np.arctan(y / (np.sqrt(self.cyllinder_radius**2 + x**2))))
        f = f if f is not None else self.zoom_f
        return pan_deg, tilt_deg, f

    def get_frame(self, frame_orig):
        return cv2.warpPerspective(
            frame_orig,
            self.H,
            (ProjectiveCamera.FRAME_W, ProjectiveCamera.FRAME_H),
            flags=INTERPOLATION_TYPE
        )

    def roi2original(self, pts):
        H_inv = np.linalg.inv(self.H)
        pts = pts.copy()

        for i in range(0, len(pts)):
            x, y = pts[i]
            pts[i] = utils.apply_homography(H_inv, x, y)

        return pts.astype(np.int16)

    def draw_roi_(self, frame_orig, color=Color.VIOLET):
        pts = self.get_corner_pts()
        cv2.polylines(frame_orig, [pts], True, color, thickness=10)

    def draw_frame_mask(self, frame_orig):
        mask = np.zeros(frame_orig.shape[:2], dtype=np.uint8)
        pts = self.get_corner_pts()
        cv2.fillPoly(mask, [pts], 255)
        return cv2.bitwise_and(frame_orig, frame_orig, mask=mask)

    def draw_grid_(self, frame_orig, color=Color.BLUE):
        step = 20
        fov_horiz_deg = 120
        fov_vert_deg = fov_horiz_deg / Camera.FRAME_ASPECT_RATIO

        frame_orig_w, frame_orig_h = self.frame_orig_size
        xx, yy = np.meshgrid(np.linspace(-fov_horiz_deg / 2, fov_horiz_deg / 2, frame_orig_w // step),
                             np.linspace(-fov_vert_deg / 2, fov_vert_deg / 2, frame_orig_h // step))
        coords = np.array([xx.ravel(), yy.ravel()], dtype=np.int16)

        for pan_deg, tilt_deg in coords.T:
            x, y = self.ptz2coords(pan_deg, tilt_deg)
            cv2.circle(frame_orig, [x, y], radius=5,
                       color=color, thickness=-1)

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
        }
        return stats
