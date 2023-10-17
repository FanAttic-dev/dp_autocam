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
        self.config = config
        self.sensor_w = Camera.SENSOR_W
        self.lens_fov_horiz_deg = config.dataset["camera_params"]["lens_fov_horiz_deg"]
        self.init_ptz(config.dataset)

        h, w, _ = frame_orig.shape
        self.frame_orig_size = np.array([w, h], dtype=np.uint16)
        self.frame_orig_center_x = w // 2
        self.frame_orig_center_y = h // 2

        self.pid_x = PID(kp=0.03, ki=0.009)
        self.pid_y = PID()
        self.pid_f = PID(kp=0.01)
        center_x, center_y = self.center
        self.pid_x.init(center_x)
        self.pid_y.init(center_y)
        self.pid_f.init(self.zoom_f)

        self.ball_filter = ParticleFilter(Config.autocam["ball_pf"])
        self.ball_filter.init(self.center)
        self.ball_mu_last = self.center

        self.players_filter = KalmanFilterVel(
            Config.autocam["players_kf"]["dt"],
            Config.autocam["players_kf"]["std_acc"],
            Config.autocam["players_kf"]["std_meas"],
        )
        self.players_filter.set_pos(*self.center)

        self.is_initialized = False
        self.players_var = None
        self.u_last = None
        self.init_dead_zone()

    def set_ptz(self, pan_deg, tilt_deg, zoom_f):
        def _set_ptz(pan_deg, tilt_deg, zoom_f):
            self.pan_deg = np.clip(
                pan_deg,
                self.pan_deg_min,
                self.pan_deg_max,
            )
            self.tilt_deg = np.clip(
                tilt_deg,
                self.tilt_deg_min,
                self.tilt_deg_max
            )
            self.zoom_f = np.clip(
                zoom_f,
                self.zoom_f_min,
                self.zoom_f_max
            )

        def _check_corner_pts():
            corner_pts = self.get_corner_pts()
            w, h = self.frame_orig_size
            frame_box = np.array([0, 0, w-1, h-1])
            return utils.polygon_lies_in_box(corner_pts, frame_box)

        if Config.autocam["debug"]["ignore_bounds"]:
            self.pan_deg, self.tilt_deg, self.zoom_f = pan_deg, tilt_deg, zoom_f
            print(_check_corner_pts())
            return self

        pan_old, tilt_old, zoom_old = self.pan_deg, self.tilt_deg, self.zoom_f
        _set_ptz(pan_deg, tilt_deg, zoom_f)
        if not _check_corner_pts():
            _set_ptz(pan_old, tilt_old, zoom_old)

        return self

    def init_ptz(self, config):
        camera_config = config["camera_params"]

        self.pan_deg_min = camera_config["pan_deg"]["min"]
        self.pan_deg_max = camera_config["pan_deg"]["max"]
        self.pan_deg_default = camera_config["pan_deg"]["default"]

        self.tilt_deg_min = camera_config["tilt_deg"]["min"]
        self.tilt_deg_max = camera_config["tilt_deg"]["max"]
        self.tilt_deg_default = camera_config["tilt_deg"]["default"]

        self.zoom_f_min = camera_config["zoom_f"]["min"]
        self.zoom_f_max = camera_config["zoom_f"]["max"]
        self.zoom_f_default = camera_config["zoom_f"]["default"]

        self.pan_deg = self.pan_deg_default
        self.tilt_deg = self.tilt_deg_default
        self.zoom_f = self.zoom_f_default

    def init_dead_zone(self):
        size = np.array(Config.autocam["dead_zone"]["size"])

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
    @abstractmethod
    def center(self):
        ...

    @property
    @abstractmethod
    def set_center(self, x, y, f=None):
        ...

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

    def draw_ball_prediction_(self, frame_orig, color):
        x, y = self.ball_mu_last
        cv2.circle(frame_orig, (int(x), int(y)),
                   radius=4, color=color, thickness=5)

    def draw_ball_u_(self, frame_orig, color):
        if self.u_last is None:
            return

        u_x, u_y = self.u_last
        mu_x, mu_y = self.ball_mu_last
        pt1 = np.array([mu_x, mu_y], dtype=np.int32)
        pt2 = np.array([mu_x + u_x, mu_y + u_y], dtype=np.int32)
        cv2.line(frame_orig, pt1, pt2, color=color, thickness=2)

    def draw_dead_zone_(self, frame):
        start, end = self.dead_zone
        cv2.rectangle(frame, start, end,
                      color=Color.RED, thickness=1)

    @abstractmethod
    def draw_frame_mask(self, frame_orig):
        ...

    def draw_players_bb_(self, frame_orig, bbs, color=Color.TEAL):
        if len(bbs["boxes"]) == 0:
            return

        margin_px = Config.autocam["zoom"]["bb"]["margin_px"]
        x1, y1, x2, y2 = utils.get_bounding_box(bbs)
        x1 -= margin_px
        x2 += margin_px
        cv2.rectangle(frame_orig, (x1, y1), (x2, y2),
                      color, thickness=5)

    @abstractmethod
    def get_frame(self, frame_orig):
        ...

    @abstractmethod
    def roi2original(self, pts):
        ...

    def pan(self, dx):
        pan_deg = self.pan_deg + dx
        self.set_ptz(pan_deg, self.tilt_deg, self.zoom_f)

    def tilt(self, dy):
        tilt_deg = self.tilt_deg + dy
        self.set_ptz(self.pan_deg, tilt_deg, self.zoom_f)

    def zoom(self, dz):
        f = self.zoom_f + dz
        self.set_ptz(self.pan_deg, self.tilt_deg, f)

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
            self.set_center(mouseX, mouseY)
        elif key == ord('q'):
            is_alive = False
        return is_alive
