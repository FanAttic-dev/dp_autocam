from abc import ABC, abstractmethod
import cv2

import numpy as np
from camera.PID import PID
from utils.config import Config
from utils.constants import DT_FLOAT, DT_INT, Color
from utils.protocols import HasStats
import utils.utils as utils


class Camera(ABC, HasStats):
    PAN_DX = .1
    TILT_DY = .1
    ZOOM_DZ = .5

    SENSOR_W = 36  # FX sensor width [mm]
    FRAME_W = 1920
    FRAME_H = 1080
    FRAME_ASPECT_RATIO = FRAME_W/FRAME_H
    FRAME_CORNERS = np.array([
        [0, 0],
        [0, FRAME_H-1],
        [FRAME_W-1, FRAME_H-1],
        [FRAME_W-1, 0]
    ], dtype=DT_INT)

    def __init__(
        self,
        frame_orig,
        config: Config,
        ignore_bounds=Config.autocam["debug"]["ignore_bounds"]
    ):
        h, w, _ = frame_orig.shape
        self.frame_orig_size = np.array([w, h], dtype=DT_INT)
        self.frame_orig_center_x = w // 2
        self.frame_orig_center_y = h // 2

        self.frame_roi_size = np.array(
            [Camera.FRAME_W, Camera.FRAME_H], dtype=DT_INT)

        self.config = config
        self.ignore_bounds = ignore_bounds
        self.sensor_w = Camera.SENSOR_W
        self.lens_fov_horiz_deg = config.dataset["camera_params"]["lens_fov_horiz_deg"]

        self.init_ptz(config)
        self.init_pid(config, *self.center, self.zoom_f)

    # region PTZ
    @property
    def ptz(self):
        return self.pan_deg, self.tilt_deg, self.zoom_f

    @abstractmethod
    def screen2ptz(self, x, y, f=None):
        """Convert x, y screen coordinates to pan-tilt-zoom spherical coordinates.

        Both screen and spherical coordinates are in original frame space.

        Returns:
            pan_deg, tilt_deg, zoom_f
        """
        ...

    @abstractmethod
    def ptz2screen(self, pan_deg, tilt_deg, f=None):
        """Convert ptz spherical coordinates to x, y screen coordinates.

        Both screen and spherical coordinates are in original frame space.

        Returns:
            x, y screen coordinates in original frame space.
        """
        ...

    def pan(self, dx):
        pan_deg = self.pan_deg + dx
        self.try_set_ptz(pan_deg, self.tilt_deg, self.zoom_f)

    def tilt(self, dy):
        tilt_deg = self.tilt_deg + dy
        self.try_set_ptz(self.pan_deg, tilt_deg, self.zoom_f)

    def zoom(self, dz):
        f = self.zoom_f + dz
        self.try_set_ptz(self.pan_deg, self.tilt_deg, f)

    def init_ptz(self, config: Config):
        camera_config = config.dataset["camera_params"]

        self.pan_deg_min = camera_config["pan_deg"]["min"]
        self.pan_deg_max = camera_config["pan_deg"]["max"]
        self.pan_deg_default = camera_config["pan_deg"]["default"]

        self.tilt_deg_min = camera_config["tilt_deg"]["min"]
        self.tilt_deg_max = camera_config["tilt_deg"]["max"]
        self.tilt_deg_default = camera_config["tilt_deg"]["default"]

        self.zoom_f_min = camera_config["zoom_f"]["min"]
        self.zoom_f_max = camera_config["zoom_f"]["max"]
        self.zoom_f_default = camera_config["zoom_f"]["default"]

        self.set_ptz(
            self.pan_deg_default,
            self.tilt_deg_default,
            self.zoom_f_default
        )

    def set_ptz(self, pan_deg, tilt_deg, zoom_f):
        self.pan_deg, self.tilt_deg, self.zoom_f = pan_deg, tilt_deg, zoom_f
        return self

    def try_set_ptz(self, pan_deg, tilt_deg, zoom_f):
        if not self.check_ptz(pan_deg, tilt_deg, zoom_f):
            return False

        self.set_ptz(pan_deg, tilt_deg, zoom_f)
        return True

    def check_ptz(self, pan_deg, tilt_deg, zoom_f):
        if self.ignore_bounds:
            return True

        is_valid = pan_deg >= self.pan_deg_min and pan_deg <= self.pan_deg_max and \
            tilt_deg >= self.tilt_deg_min and tilt_deg <= self.tilt_deg_max and \
            zoom_f >= self.zoom_f_min and zoom_f <= self.zoom_f_max

        if not is_valid:
            return False

        ptz_old = self.ptz
        self.set_ptz(pan_deg, tilt_deg, zoom_f)
        is_valid = self.check_corner_pts()
        self.set_ptz(*ptz_old)

        return is_valid

    def clip_ptz(self, pan_deg, tilt_deg, zoom_f):
        pan_deg = np.clip(pan_deg, self.pan_deg_min, self.pan_deg_max)
        tilt_deg = np.clip(tilt_deg, self.tilt_deg_min, self.tilt_deg_max)
        zoom_f = np.clip(zoom_f, self.zoom_f_min, self.zoom_f_max)
        return pan_deg, tilt_deg, zoom_f
    # endregion

    # region PID
    @property
    def pid_target(self):
        return self.pid_x.target, self.pid_y.target, self.pid_f.target

    @property
    def pid_signal(self):
        return self.pid_x.signal, self.pid_y.signal, self.pid_f.signal

    def init_pid(self, config: Config, x, y, f):
        pid_config = config.autocam["pid"]

        self.pid_x = PID(
            kp=pid_config["x"]["kp"],
            ki=pid_config["x"]["ki"],
            kd=pid_config["x"]["kd"]
        )
        self.pid_y = PID(
            kp=pid_config["y"]["kp"],
            ki=pid_config["y"]["ki"],
            kd=pid_config["y"]["kd"]
        )
        self.pid_f = PID(
            kp=pid_config["f"]["kp"],
            ki=pid_config["f"]["ki"],
            kd=pid_config["f"]["kd"]
        )

        self.pid_x.init(x)
        self.pid_y.init(y)
        self.pid_f.init(f)

    def update_pid(self, pid_x=None, pid_y=None, pid_f=None):
        self.pid_x.update(pid_x)
        self.pid_y.update(pid_y)
        self.pid_f.update(pid_f)
    # endregion

    # region FOV
    @property
    def fov_horiz_deg(self):
        return np.rad2deg(2 * np.arctan(self.sensor_w / (2 * self.zoom_f)))

    @property
    def fov_vert_deg(self):
        return self.fov_horiz_deg / Camera.FRAME_ASPECT_RATIO

    @property
    def fov_rad(self):
        return np.deg2rad(
            np.array([self.fov_horiz_deg, self.fov_vert_deg]),
            dtype=DT_FLOAT
        )

    def fov2f(self, fov_deg):
        """Calculate focal length based on field of view.

        https://www.edmundoptics.com/knowledge-center/application-notes/imaging/understanding-focal-length-and-field-of-view/
        """
        return self.sensor_w / (2 * np.tan(np.deg2rad(fov_deg) / 2))

    def screen_width_px2fov(self, px):
        """Calculate field of view given by width in screen space [px]."""
        frame_orig_width, _ = self.frame_orig_size
        return self.lens_fov_horiz_deg / frame_orig_width * px

    @abstractmethod
    def roi2original(self, pts_roi_screen):
        """Convert coordinates from camera frame space (ROI) to original frame space."""
        ...
    # endregion

    # region Center
    @property
    def center(self):
        return self.ptz2screen(self.pan_deg, self.tilt_deg)

    def set_center(self, x, y, f=None):
        ptz = self.screen2ptz(x, y, f)
        return self.set_ptz(*ptz)

    def try_set_center(self, x, y, f=None):
        ptz = self.screen2ptz(x, y, f)
        return self.try_set_ptz(*ptz)
    # endregion

    # region Corner Points
    @abstractmethod
    def get_pts_corners(self, correct_rotation: bool) -> np.ndarray:
        """Get corner points of current view in the space of the original frame."""
        ...

    def check_corner_pts(self):
        """Check whether the corner points of the current view lie within the original frame."""
        corner_pts = self.get_pts_corners(Config.autocam["correct_rotation"])
        w, h = self.frame_orig_size
        frame_box = np.array([0, 0, w-1, h-1])
        return utils.is_polygon_in_box(corner_pts, frame_box)

    def correct_rotation(self, pts, center=None):
        """Rotate points by the angle between the current frame and the back pitch line.

        Returns:
            H: Homography matrix mapping the current view from original frame space to output frame space.
            pts: rotated points
        """

        # Find homography that maps corner points
        # from original frame space to the output frame space.
        corner_pts = self.get_pts_corners(correct_rotation=False)
        H, _ = cv2.findHomography(corner_pts, Camera.FRAME_CORNERS)

        # Map pitch corners from original frame space to the output frame space.
        pitch_corners_orig = self.config.pitch_corners.astype(DT_FLOAT)
        pitch_corners_frame = cv2.perspectiveTransform(
            pitch_corners_orig, H
        )

        # Rotate points by the angle between the back line and the (horizontal) x-axis.
        roll_rad = utils.get_pitch_rotation_rad(pitch_corners_frame)
        pts = utils.rotate_pts(pts, roll_rad, center)
        return H, pts
    # endregion

    # region Drawing
    @abstractmethod
    def draw_grid_(self, frame_orig, color=Color.BLUE):
        ...

    @abstractmethod
    def draw_frame_mask(self, frame_orig):
        ...

    def draw_center_(self, frame_orig, color=Color.RED):
        cv2.circle(frame_orig, self.center,
                   radius=5, color=color, thickness=5)
    # endregion

    def process_input(self, key, mouse_pos):
        is_alive = True
        if key == ord('d'):
            self.pan(Camera.PAN_DX)
        elif key == ord('a'):
            self.pan(-Camera.PAN_DX)
        elif key == ord('w'):
            self.tilt(-Camera.TILT_DY)
        elif key == ord('s'):
            self.tilt(Camera.TILT_DY)
        elif key == ord('+'):
            self.zoom(self.__class__.ZOOM_DZ)
        elif key == ord('-'):
            self.zoom(-self.__class__.ZOOM_DZ)
        elif key == ord('f'):
            self.try_set_center(*mouse_pos)
        elif key == ord('q'):
            is_alive = False
        return is_alive

    @abstractmethod
    def get_stats(self) -> dict:
        ...

    def print(self):
        print(self.get_stats())
