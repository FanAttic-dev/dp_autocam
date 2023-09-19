import cv2
import numpy as np
from PID import PID
from constants import INTERPOLATION_TYPE, colors, params
from kalman_filter import KalmanFilterVel
from particle_filter import ParticleFilter
from utils import apply_homography, coords2pts, filter_bbs_ball, get_bounding_box, get_pitch_rotation_rad, points_average, discard_extreme_points_, get_bb_center, lies_in_rectangle, points_variance, rotate_pts


class Camera:
    def __init__(self):
        ...

    def get_frame(self, frame_orig):
        ...

    def pan(self, dx):
        ...

    def get_corner_pts(self):
        ...

    def update_by_bbs(self, bbs):
        ...


class PerspectiveCamera(Camera):
    SENSOR_W = 100
    PAN_DX = 1
    TILT_DY = 1
    ZOOM_DZ = 1

    FRAME_ASPECT_RATIO = 16/9
    FRAME_W = 1920
    FRAME_H = int(FRAME_W / FRAME_ASPECT_RATIO)
    FRAME_CORNERS = np.array([
        [0, 0],
        [0, FRAME_H-1],
        [FRAME_W-1, FRAME_H-1],
        [FRAME_W-1, 0]
    ], dtype=np.int16)

    def __init__(self, frame_orig, config, pan_deg=None, tilt_deg=None, zoom_f=None):
        self.sensor_w = PerspectiveCamera.SENSOR_W
        self.config = config
        self.load_config(config)

        pan_deg = pan_deg if pan_deg is not None else self.pan_deg_default
        tilt_deg = tilt_deg if tilt_deg is not None else self.tilt_deg_default
        zoom_f = zoom_f if zoom_f is not None else self.zoom_f_default
        self.set(pan_deg, tilt_deg, zoom_f)

        h, w, _ = frame_orig.shape
        self.frame_orig_shape = frame_orig.shape
        self.frame_orig_center_x = w // 2
        self.frame_orig_center_y = h // 2

        self.pid_x = PID(kp=0.03, ki=0.009)
        self.pid_y = PID()
        self.pid_f = PID(kp=0.01)
        center_x, center_y = self.center
        self.pid_x.init(center_x)
        self.pid_y.init(center_y)
        self.pid_f.init(self.zoom_f)

        self.ball_filter = ParticleFilter(params["ball_pf"])
        self.ball_filter.init(self.center)
        self.ball_mu_last = self.center

        self.players_filter = KalmanFilterVel(
            params["players_kf"]["dt"],
            params["players_kf"]["std_acc"],
            params["players_kf"]["std_meas"],
            0
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
            """ Get the players' center point. """
            points = top_down.bbs2points(bbs)
            discard_extreme_points_(points)
            points_mu = points_average(points)
            self.players_var = points_variance(points, points_mu)
            x, y = apply_homography(top_down.H_inv, *points_mu)
            return np.array([x, y])

        def measure_zoom_var(ball_var):
            """ Map the PF variance to the camera zoom bounds. """
            ball_var = np.mean(ball_var)
            var_min = params["zoom"]["var_min"]
            var_max = params["zoom"]["var_max"]
            ball_var = np.clip(ball_var, var_min, var_max)

            # zoom is inversely proportional to the variance
            zoom_level = 1 - (ball_var - var_min) / (var_max - var_min)
            zoom_range = self.zoom_f_max - self.zoom_f_min
            f = self.zoom_f_min + zoom_level * zoom_range
            return f

        def measure_zoom(bbs):
            discard_extreme_points_(bbs)
            bb_x_min, bb_y_min, bb_x_max, bb_y_max = get_bounding_box(bbs)

            corner_pts = self.get_corner_pts()
            roi_x_min, roi_y_min = corner_pts[0]
            roi_x_max, roi_y_max = corner_pts[2]

            dz = (bb_x_min - roi_x_min) + (roi_x_max - bb_x_max)
            f = self.pid_f.target + dz
            return f

        def measure_u(balls_detected, players_center, ball_var):
            def measure_players_center():
                if not balls_detected and np.mean(ball_var) > params["u_control"]["center"]["var_th"]:
                    return params["u_control"]["center"]["alpha"] * (players_center - self.ball_mu_last)
                return np.array([0., 0.])

            def measure_players_movement():
                if not self.is_initialized:
                    self.players_filter.set_pos(*players_center)
                self.players_filter.predict()
                self.players_filter.update(*players_center)
                return params["u_control"]["velocity"]["alpha"] * self.players_filter.vel.T[0]

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
        # f = measure_zoom_var(ball_var)
        mu_x, mu_y = ball_mu if not self.is_meas_in_dead_zone else (None, None)
        self.pid_x.update(mu_x)
        self.pid_y.update(mu_y)

        f = measure_zoom(bbs)
        self.pid_f.update(f)

        pid_x = self.pid_x.get()
        pid_y = self.pid_y.get()
        pid_f = self.pid_f.get()
        self.set_center(pid_x, pid_y, pid_f)

        # Update variables
        self.ball_mu_last = ball_mu
        self.u_last = u
        self.is_initialized = True

    @property
    def fov_horiz_deg(self):
        return np.rad2deg(2 * np.arctan(self.sensor_w / (2 * self.zoom_f)))

    @property
    def fov_vert_deg(self):
        return self.fov_horiz_deg / 16 * 9

    @property
    def center(self):
        return self.ptz2coords(self.pan_deg, self.tilt_deg, self.zoom_f)

    @property
    def corners_ang(self):
        return {
            "left top": [-self.fov_horiz_deg / 2, -self.fov_vert_deg / 2],
            "left bottom": [-self.fov_horiz_deg / 2, self.fov_vert_deg / 2],
            "right bottom": [self.fov_horiz_deg / 2, self.fov_vert_deg / 2],
            "right top": [self.fov_horiz_deg / 2, -self.fov_vert_deg / 2],
        }

    @property
    def H(self):
        src = self.get_corner_pts()
        dst = PerspectiveCamera.FRAME_CORNERS

        H, _ = cv2.findHomography(src, dst)

        if params["correct_rotation"]:
            # TODO: use lookup table
            pitch_coords_orig = coords2pts(self.config["pitch_coords"])
            pitch_coords_frame = cv2.perspectiveTransform(
                pitch_coords_orig.astype(np.float64), H)
            roll_rad = get_pitch_rotation_rad(pitch_coords_frame)

            src = np.array(rotate_pts(src, roll_rad), dtype=np.int32)
            H, _ = cv2.findHomography(src, dst)

        return H

    @property
    def H_inv(self):
        return np.linalg.inv(self.H)

    def set(self, pan_deg, tilt_deg, zoom_f):
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

    def load_config(self, config):
        camera_config = config["camera_params"]
        self.cyllinder_radius = camera_config["cyllinder_radius"]

        self.pan_deg_min = camera_config["pan_deg"]["min"]
        self.pan_deg_max = camera_config["pan_deg"]["max"]
        self.pan_deg_default = camera_config["pan_deg"]["default"]

        self.tilt_deg_min = camera_config["tilt_deg"]["min"]
        self.tilt_deg_max = camera_config["tilt_deg"]["max"]
        self.tilt_deg_default = camera_config["tilt_deg"]["default"]

        self.zoom_f_min = camera_config["zoom_f"]["min"]
        self.zoom_f_max = camera_config["zoom_f"]["max"]
        self.zoom_f_default = camera_config["zoom_f"]["default"]

    def set_center(self, x, y, f=None):
        pan_deg, tilt_deg = self.coords2ptz(x, y)
        f = f if f is not None else self.zoom_f
        self.set(pan_deg, tilt_deg, f)

    def init_dead_zone(self):
        size = np.array(params["dead_zone"]["size"])

        center = np.array([
            PerspectiveCamera.FRAME_W // 2,
            PerspectiveCamera.FRAME_H // 2
        ])
        self.dead_zone = np.array([
            center - size // 2,  # start point (top left)
            center + size // 2  # end point (bottom right)
        ])

    @property
    def is_meas_in_dead_zone(self):
        if not params["dead_zone"]["enabled"]:
            return False

        meas = np.array([[self.ball_mu_last]], dtype=np.float32)
        meas_frame_coord = cv2.perspectiveTransform(meas, self.H)[0][0]
        return lies_in_rectangle(meas_frame_coord, self.dead_zone)

    def shift_coords(self, x, y):
        x = x + self.frame_orig_center_x
        y = y + self.frame_orig_center_y
        return x, y

    def unshift_coords(self, x, y):
        x = x - self.frame_orig_center_x
        y = y - self.frame_orig_center_y
        return x, y

    def ptz2coords(self, theta_deg, phi_deg, f):
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

    def get_corner_pts(self):
        pts = [
            self.ptz2coords(
                self.pan_deg + pan_deg,
                self.tilt_deg + tilt_deg,
                self.zoom_f)
            for pan_deg, tilt_deg in self.corners_ang.values()
        ]

        return np.array(pts, dtype=np.int32)

    def draw_roi_(self, frame_orig, color=colors["yellow"]):
        pts = self.get_corner_pts()
        cv2.polylines(frame_orig, [pts], True, color, thickness=10)

    def draw_center_(self, frame_orig, color=colors["red"]):
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
                      color=colors["red"], thickness=1)

    def draw_players_bb(self, frame_orig, bbs, color=colors["blue"]):
        x1, y1, x2, y2 = get_bounding_box(bbs)
        cv2.rectangle(frame_orig, (x1, y1), (x2, y2),
                      color, thickness=5)

    def get_frame(self, frame_orig):
        return cv2.warpPerspective(
            frame_orig,
            self.H,
            (PerspectiveCamera.FRAME_W, PerspectiveCamera.FRAME_H),
            flags=INTERPOLATION_TYPE
        )

    def pan(self, dx):
        pan_deg = self.pan_deg + dx
        self.set(pan_deg, self.tilt_deg, self.zoom_f)

    def tilt(self, dy):
        tilt_deg = self.tilt_deg + dy
        self.set(self.pan_deg, tilt_deg, self.zoom_f)

    def zoom(self, dz):
        f = self.zoom_f + dz
        self.set(self.pan_deg, self.tilt_deg, f)

    def process_input(self, key, mouseX, mouseY):
        is_alive = True
        if key == ord('d'):
            self.pan(PerspectiveCamera.PAN_DX)
        elif key == ord('a'):
            self.pan(-PerspectiveCamera.PAN_DX)
        elif key == ord('w'):
            self.tilt(-PerspectiveCamera.TILT_DY)
        elif key == ord('s'):
            self.tilt(PerspectiveCamera.TILT_DY)
        elif key == ord('p'):
            self.zoom(PerspectiveCamera.ZOOM_DZ)
        elif key == ord('m'):
            self.zoom(-PerspectiveCamera.ZOOM_DZ)
        elif key == ord('+'):
            self.sensor_w += 1
        elif key == ord('-'):
            self.sensor_w -= 1
        elif key == ord('c'):
            self.cyllinder_radius += 10
        elif key == ord('v'):
            self.cyllinder_radius -= 10
        elif key == ord('f'):
            self.set_center(mouseX, mouseY)
        elif key == ord('q'):
            is_alive = False
        return is_alive

    def get_stats(self):
        stats = {
            "Name": PerspectiveCamera.__name__,
            "f": f"{self.zoom_f:.2f}",
            # "sensor_w": self.sensor_w,
            # "cyllinder_r": self.cyllinder_radius,
            "pan_deg": f"{self.pan_deg:.4f}",
            "tilt_deg": f"{self.tilt_deg:.4f}",
            # "fov_horiz_deg": self.fov_horiz_deg,
            # "fov_vert_deg": self.fov_vert_deg,
            "players_vel": self.players_filter.vel.squeeze(1),
            "players_std": np.sqrt(self.players_var),
        }
        return stats

    def print(self):
        print(self.get_stats())
