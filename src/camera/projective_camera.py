from abc import abstractmethod
import cv2
import numpy as np
from camera.PID import PID
from camera.camera import Camera
from utils.config import Config
from utils.constants import Color
from filters.kalman_filter import KalmanFilterVel
from filters.particle_filter import ParticleFilter
from utils.helpers import apply_homography, discard_extreme_boxes_, filter_bbs_ball, get_bounding_box, get_pitch_rotation_rad, lies_in_box, points_average, discard_extreme_points_, get_bb_center, lies_in_box_pt, points_variance


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

    def update_by_bbs(self, bbs, top_down):
        def measure_ball(bb_ball):
            """ Get the ball center point. """
            return get_bb_center(bb_ball)

        def measure_players(bbs):
            """ Get the players' center point in frame_orig space. """
            points = top_down.bbs2points(bbs)
            discard_extreme_points_(points)
            points_mu = points_average(points)
            self.players_var = points_variance(points, points_mu)
            x, y = apply_homography(top_down.H_inv, *points_mu)
            return np.array([x, y])

        def measure_zoom_var(ball_var):
            """ Map the PF variance to the camera zoom bounds. """
            ball_var = np.mean(ball_var)
            var_min = Config.autocam["zoom"]["var_min"]
            var_max = Config.autocam["zoom"]["var_max"]
            ball_var = np.clip(ball_var, var_min, var_max)

            # zoom is inversely proportional to the variance
            zoom_level = 1 - (ball_var - var_min) / (var_max - var_min)
            zoom_range = self.zoom_f_max - self.zoom_f_min
            f = self.zoom_f_min + zoom_level * zoom_range
            return f

        def measure_zoom_bbs(bbs):
            discard_extreme_boxes_(bbs)
            alpha = Config.autocam["zoom"]["bb"]["alpha"]
            margin_px = Config.autocam["zoom"]["bb"]["margin_px"]

            bb_x_min, _, bb_x_max, _ = get_bounding_box(bbs)
            bb_x_min -= margin_px
            bb_x_max += margin_px
            bb_width = bb_x_max - bb_x_min

            frame_orig_width, _ = self.frame_orig_size
            fov_target_deg = self.lens_fov_horiz_deg / frame_orig_width * bb_width
            f = self.fov2f(fov_target_deg)
            f = np.clip(f, self.zoom_f_min, self.zoom_f_max)
            return f

        def measure_u(balls_detected, players_center, ball_var):
            def measure_players_center():
                if not balls_detected and np.mean(ball_var) > Config.autocam["u_control"]["center"]["var_th"]:
                    return Config.autocam["u_control"]["center"]["alpha"] * (players_center - self.ball_mu_last)
                return np.array([0., 0.])

            def measure_players_movement():
                if not self.is_initialized:
                    self.players_filter.set_pos(*players_center)
                self.players_filter.predict()
                self.players_filter.update(*players_center)
                return Config.autocam["u_control"]["velocity"]["alpha"] * self.players_filter.vel.T[0]

            u = np.array([0., 0.])

            u += measure_players_center()
            u += measure_players_movement()

            return u

        bbs_ball = filter_bbs_ball(bbs)
        players_detected = len(bbs) > 0 and len(bbs["boxes"]) > 0
        balls_detected = len(bbs_ball) > 0 and len(bbs_ball['boxes']) > 0

        if not players_detected:
            return

        players_center = measure_players(bbs)

        # Incorporate measurements into PF
        if balls_detected:
            ball_centers = [measure_ball(bb_ball)
                            for bb_ball in bbs_ball['boxes']]
            self.ball_filter.update(players_center, ball_centers)
            self.ball_filter.resample()

        # Apply motion model with uncertainty to PF
        self.ball_filter.predict()

        # Get estimate
        ball_mu, ball_var = self.ball_filter.estimate

        # Set control input
        u = measure_u(balls_detected, players_center, ball_var)
        self.ball_filter.set_u(u)

        # Camera model
        mu_x, mu_y = ball_mu if not self.is_meas_in_dead_zone else (None, None)
        self.pid_x.update(mu_x)
        self.pid_y.update(mu_y)

        # f = measure_zoom_var(ball_var)
        f = measure_zoom_bbs(bbs)
        self.pid_f.update(f)

        pid_x = self.pid_x.get()
        pid_y = self.pid_y.get()
        pid_f = self.pid_f.get()
        self.set_center(pid_x, pid_y, pid_f)

        # Update variables
        self.ball_mu_last = ball_mu
        self.u_last = u
        self.is_initialized = True

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
            inner = np.array([corner_pts[0], corner_pts[2]],
                             dtype=np.int32).ravel()
            w, h = self.frame_orig_size
            outer = np.array([0, 0, w-1, h-1])
            return lies_in_box(inner, outer)

        if Config.autocam["debug"]["ignore_bounds"]:
            self.pan_deg, self.tilt_deg, self.zoom_f = pan_deg, tilt_deg, zoom_f
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
        return self.sensor_w / (2 * np.tan(np.deg2rad(fov_deg) / 2))

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
        return lies_in_box_pt(meas_frame_coord, self.dead_zone)

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
        margin_px = Config.autocam["zoom"]["bb"]["margin_px"]
        x1, y1, x2, y2 = get_bounding_box(bbs)
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
