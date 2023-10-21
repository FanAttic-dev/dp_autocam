from algorithm.algo import Algo
from camera.camera import Camera
from camera.top_down import TopDown
from filters.kalman_filter import KalmanFilterVel
from filters.particle_filter import ParticleFilter
from utils.config import Config
from utils.constants import Color
import utils.utils as utils
import numpy as np
import cv2


class AutocamAlgo(Algo):
    def __init__(self, camera: Camera, top_down: TopDown, config: Config):
        self.camera = camera
        self.top_down = top_down

        self.init_filters(config)

        self.is_initialized = False
        self.players_var = None
        self.u_last = None

    def update_by_bbs(self, bbs):
        bbs_ball = self.filter_bbs_ball(bbs)
        players_detected = len(bbs) > 0 and len(bbs["boxes"]) > 0
        balls_detected = len(bbs_ball) > 0 and len(bbs_ball['boxes']) > 0

        if not players_detected:
            return

        utils.discard_extreme_boxes_(bbs)
        players_center = self.measure_players(bbs)

        # Incorporate measurements into PF
        if balls_detected:
            ball_centers = [self.measure_ball(bb_ball)
                            for bb_ball in bbs_ball['boxes']]
            self.ball_filter.update(players_center, ball_centers)
            self.ball_filter.resample()

        # Apply motion model with uncertainty to PF
        self.ball_filter.predict()

        # Get estimate
        ball_mu, ball_var = self.ball_filter.estimate
        f = self.measure_zoom(ball_var, bbs)

        # Set control input
        u = self.measure_u(balls_detected, players_center, ball_var)
        self.ball_filter.set_u(u)

        # Update camera
        self.try_update_camera(ball_mu, f)

        # Update variables
        self.ball_mu_last = ball_mu
        self.u_last = u
        self.is_initialized = True

    def try_update_camera(self, center, f=None):
        """Try to update the camera PID target.

        It first converts the coords to PTZ,
        then clips the PTZ to a given PTZ limits,
        and then verifies if the corner points would lie
        within the original frame.

        If the new target is valid, the camera PID is updated accordingly.
        Otherwise, the PID gets updated by its previous target.
        """

        center_ptz = self.camera.screen2ptz(*center, f)
        center_ptz = self.camera.clip_ptz(*center_ptz)
        is_valid = self.camera.check_ptz(*center_ptz)
        if not is_valid:
            target = (None, None, None)
            print("Update camera: target out of bounds")
        else:
            center_screen = self.camera.ptz2screen(*center_ptz)
            target = (*center_screen, f)

        self.camera.update_pid(*target)
        self.camera.set_center(*self.camera.pid_signal)
        return is_valid

    def measure_ball(self, bb_ball):
        """ Get the ball center point. """

        return utils.get_bb_center(bb_ball)

    def measure_players(self, bbs):
        """ Get the players' center point in frame_orig space. """

        points = self.top_down.bbs2points(bbs)
        points_mu = utils.points_average(points)
        self.players_var = utils.points_variance(points, points_mu)
        x, y = utils.apply_homography(self.top_down.H_inv, *points_mu)
        return np.array([x, y])

    def measure_zoom_var(self, ball_var):
        """ Maps the PF variance to the camera zoom bounds. """

        ball_var = np.mean(ball_var)
        var_min = Config.autocam["zoom"]["var_min"]
        var_max = Config.autocam["zoom"]["var_max"]
        ball_var = np.clip(ball_var, var_min, var_max)

        # zoom is inversely proportional to the variance
        zoom_level = 1 - (ball_var - var_min) / (var_max - var_min)
        zoom_range = self.camera.zoom_f_max - self.camera.zoom_f_min
        f = self.camera.zoom_f_min + zoom_level * zoom_range
        return f

    def measure_zoom_bb(self, bbs):
        """ Calculates the focal length based on the players' bounding box. """

        margin_px = Config.autocam["zoom"]["bb"]["margin_px"]

        bb_x_min, _, bb_x_max, _ = utils.get_bounding_box(bbs)
        bb_x_min -= margin_px
        bb_x_max += margin_px
        bb_width = bb_x_max - bb_x_min

        fov_target_deg = self.camera.screen_width_px2fov(bb_width)
        f = self.camera.fov2f(fov_target_deg)
        f = np.clip(f, self.camera.zoom_f_min, self.camera.zoom_f_max)
        return f

    def measure_zoom(self, ball_var, bbs):
        f_var = self.measure_zoom_var(ball_var)
        f_bb = self.measure_zoom_bb(bbs)
        # return (f_var + f_bb) / 2
        return max(f_var, f_bb)

    def measure_u(self, balls_detected, players_center, ball_var):
        def measure_players_center():
            alpha = Config.autocam["u_control"]["center"]["alpha"]
            var_th = Config.autocam["u_control"]["center"]["var_th"]

            if not balls_detected and np.mean(ball_var) > var_th:
                return alpha * (players_center - self.ball_mu_last)
            return np.array([0., 0.])

        def measure_players_movement():
            alpha = Config.autocam["u_control"]["velocity"]["alpha"]

            if not self.is_initialized:
                self.players_filter.set_pos(*players_center)
            self.players_filter.predict()
            self.players_filter.update(*players_center)
            return alpha * self.players_filter.vel.T[0]

        u = np.array([0., 0.])
        u += measure_players_center()
        u += measure_players_movement()

        return u

    def init_filters(self, config: Config):
        self.ball_filter = ParticleFilter(config.autocam["ball_pf"])
        self.ball_filter.init(self.camera.center)
        self.ball_mu_last = self.camera.center

        self.players_filter = KalmanFilterVel(
            config.autocam["players_kf"]["dt"],
            config.autocam["players_kf"]["std_acc"],
            config.autocam["players_kf"]["std_meas"],
        )
        self.players_filter.set_pos(*self.camera.center)

    def filter_bbs_ball(self, bbs):
        """ Returns only bbs of class ball. """

        bbs_ball = {
            "boxes": [],
            "cls": [],
            "ids": []
        }
        for i, (bb, cls) in enumerate(zip(bbs["boxes"], bbs["cls"])):
            if cls != 0:
                continue

            bbs_ball["boxes"].append(bb)
            bbs_ball["cls"].append(cls)

            if i >= len(bbs_ball["ids"]):
                continue

            bbs_ball["ids"].append(bbs_ball["ids"][i])
        return bbs_ball

    def draw_ball_u_(self, frame_orig, color):
        if self.u_last is None:
            return

        u_x, u_y = self.u_last
        mu_x, mu_y = self.ball_mu_last
        pt1 = np.array([mu_x, mu_y], dtype=np.int32)
        pt2 = np.array([mu_x + u_x, mu_y + u_y], dtype=np.int32)
        cv2.line(frame_orig, pt1, pt2, color=color, thickness=2)

    def draw_ball_prediction_(self, frame_orig, color):
        x, y = self.ball_mu_last
        cv2.circle(frame_orig, (int(x), int(y)),
                   radius=4, color=color, thickness=5)

    def draw_players_bb_(self, frame_orig, bbs, color=Color.TEAL):
        if len(bbs["boxes"]) == 0:
            return

        margin_px = Config.autocam["zoom"]["bb"]["margin_px"]
        x1, y1, x2, y2 = utils.get_bounding_box(bbs)
        x1 -= margin_px
        x2 += margin_px
        cv2.rectangle(frame_orig, (x1, y1), (x2, y2),
                      color, thickness=2)

    def get_stats(self):
        stats = {
            "Name": AutocamAlgo.__name__,
            "players_vel": self.players_filter.vel.squeeze(1),
            "players_std": np.sqrt(self.players_var) if self.players_var is not None else "",
        }
        return stats
