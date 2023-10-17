from abc import abstractmethod
import cv2
import numpy as np
from camera.PID import PID
from camera.camera import Camera
from utils.config import Config
from utils.constants import Color
from filters.kalman_filter import KalmanFilterVel
from filters.particle_filter import ParticleFilter
import utils.utils as utils


class ProjectiveCamera(Camera):
    PAN_DX = 1
    TILT_DY = 1
    ZOOM_DZ = 1

    def __init__(self, frame_orig, config: Config):
        h, w, _ = frame_orig.shape
        self.frame_orig_size = np.array([w, h], dtype=np.uint16)
        self.frame_orig_center_x = w // 2
        self.frame_orig_center_y = h // 2

        self.config = config
        self.sensor_w = Camera.SENSOR_W
        self.lens_fov_horiz_deg = config.dataset["camera_params"]["lens_fov_horiz_deg"]

        self.init_ptz(config)
        self.init_pid(config, *self.center, self.zoom_f)

        self.init_dead_zone(config)

    @property
    def ptz(self):
        return self.pan_deg, self.tilt_deg, self.zoom_f

    @property
    def pid_target(self):
        return self.pid_x.target, self.pid_y.target, self.pid_f.target

    @property
    def pid_signal(self):
        return self.pid_x.signal, self.pid_y.signal, self.pid_f.signal

    def update_pid(self, pid_x=None, pid_y=None, pid_f=None):
        self.pid_x.update(pid_x)
        self.pid_y.update(pid_y)
        self.pid_f.update(pid_f)

    def check_ptz(self, pan_deg, tilt_deg, zoom_f):
        if Config.autocam["debug"]["ignore_bounds"]:
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

    def check_corner_pts(self):
        corner_pts = self.get_corner_pts()
        w, h = self.frame_orig_size
        frame_box = np.array([0, 0, w-1, h-1])
        return utils.polygon_lies_in_box(corner_pts, frame_box)

    def set_ptz(self, pan_deg, tilt_deg, zoom_f):
        self.pan_deg, self.tilt_deg, self.zoom_f = pan_deg, tilt_deg, zoom_f
        return self

    def try_set_ptz(self, pan_deg, tilt_deg, zoom_f):
        if not self.check_ptz(pan_deg, tilt_deg, zoom_f):
            return False

        self.set_ptz(pan_deg, tilt_deg, zoom_f)
        return True

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

    def init_dead_zone(self, config: Config):
        size = np.array(config.autocam["dead_zone"]["size"])

        center = np.array([
            ProjectiveCamera.FRAME_W // 2,
            ProjectiveCamera.FRAME_H // 2
        ])
        self.dead_zone = np.array([
            center - size // 2,  # start point (top left)
            center + size // 2  # end point (bottom right)
        ])

    @property
    def fov_horiz_deg(self):
        return np.rad2deg(2 * np.arctan(self.sensor_w / (2 * self.zoom_f)))

    @property
    def fov_vert_deg(self):
        return self.fov_horiz_deg / Camera.FRAME_ASPECT_RATIO

    @property
    def fov_rad(self):
        return np.deg2rad(np.array([self.fov_horiz_deg, self.fov_vert_deg]), dtype=np.float32)

    def fov2f(self, fov_deg):
        """ Calculates the focal length based on the field of view. """

        return self.sensor_w / (2 * np.tan(np.deg2rad(fov_deg) / 2))

    def screen_width_px2fov(self, px):
        """ Calculates the field of view given by a width in screen space [px]. """

        frame_orig_width, _ = self.frame_orig_size
        return self.lens_fov_horiz_deg / frame_orig_width * px

    @property
    def center(self):
        return self.ptz2coords(self.pan_deg, self.tilt_deg)

    @abstractmethod
    def coords2ptz(self, x, y, f=None):
        ...

    @abstractmethod
    def ptz2coords(self, pan_deg, tilt_deg, f=None):
        ...

    def set_center(self, x, y, f=None):
        ptz = self.coords2ptz(x, y, f)
        return self.set_ptz(*ptz)

    def try_set_center(self, x, y, f=None):
        ptz = self.coords2ptz(x, y, f)
        return self.try_set_ptz(*ptz)

    @property
    @abstractmethod
    def H(self):
        ...

    @property
    def H_inv(self):
        return np.linalg.inv(self.H)

    @property
    def is_meas_in_dead_zone(self):
        if not Config.autocam["dead_zone"]["enabled"]:
            return False

        meas = np.array([[self.ball_mu_last]], dtype=np.float32)
        meas_frame_coord = cv2.perspectiveTransform(meas, self.H)[0][0]
        return utils.pt_lies_in_box(meas_frame_coord, self.dead_zone)

    def shift_coords(self, x, y):
        x = x + self.frame_orig_center_x
        y = y + self.frame_orig_center_y
        return x, y

    def unshift_coords(self, x, y):
        x = x - self.frame_orig_center_x
        y = y - self.frame_orig_center_y
        return x, y

    @abstractmethod
    def draw_roi_(self, frame_orig, color=Color.VIOLET):
        ...

    @abstractmethod
    def draw_grid_(self, frame_orig, color=Color.BLUE):
        ...

    def draw_center_(self, frame_orig, color=Color.RED):
        cv2.circle(frame_orig, self.center,
                   radius=5, color=color, thickness=5)

    def draw_dead_zone_(self, frame):
        start, end = self.dead_zone
        cv2.rectangle(frame, start, end,
                      color=Color.RED, thickness=1)

    @abstractmethod
    def draw_frame_mask(self, frame_orig):
        ...

    @abstractmethod
    def get_frame(self, frame_orig):
        ...

    @abstractmethod
    def roi2original(self, pts):
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

    def process_input(self, key, mouseX, mouseY):
        is_alive = True
        if key == ord('d'):
            self.pan(ProjectiveCamera.PAN_DX)
        elif key == ord('a'):
            self.pan(-ProjectiveCamera.PAN_DX)
        elif key == ord('w'):
            self.tilt(-ProjectiveCamera.TILT_DY)
        elif key == ord('s'):
            self.tilt(ProjectiveCamera.TILT_DY)
        elif key == ord('+'):
            self.zoom(self.__class__.ZOOM_DZ)
        elif key == ord('-'):
            self.zoom(-self.__class__.ZOOM_DZ)
        elif key == ord('f'):
            self.try_set_center(mouseX, mouseY)
        elif key == ord('q'):
            is_alive = False
        return is_alive
