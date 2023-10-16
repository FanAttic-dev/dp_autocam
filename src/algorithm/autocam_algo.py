from algorithm.algo import Algo
from camera.projective_camera import ProjectiveCamera
from camera.top_down import TopDown
from utils.config import Config
import utils.helpers as helpers
import numpy as np


class AutocamAlgo(Algo):
    def __init__(self, camera: ProjectiveCamera, top_down: TopDown):
        self.camera = camera
        self.top_down = top_down

    def update_by_bbs(self, bbs):
        bbs_ball = self.filter_bbs_ball(bbs)
        players_detected = len(bbs) > 0 and len(bbs["boxes"]) > 0
        balls_detected = len(bbs_ball) > 0 and len(bbs_ball['boxes']) > 0

        if not players_detected:
            return

        players_center = self.measure_players(bbs)

        # Incorporate measurements into PF
        if balls_detected:
            ball_centers = [self.measure_ball(bb_ball)
                            for bb_ball in bbs_ball['boxes']]
            self.camera.ball_filter.update(players_center, ball_centers)
            self.camera.ball_filter.resample()

        # Apply motion model with uncertainty to PF
        self.camera.ball_filter.predict()

        # Get estimate
        ball_mu, ball_var = self.camera.ball_filter.estimate

        # Set control input
        u = self.measure_u(balls_detected, players_center, ball_var)
        self.camera.ball_filter.set_u(u)

        # Camera model
        mu_x, mu_y = ball_mu
        self.camera.pid_x.update(mu_x)
        self.camera.pid_y.update(mu_y)

        f = self.measure_zoom(ball_var, bbs)
        self.camera.pid_f.update(f)

        pid_x = self.camera.pid_x.get()
        pid_y = self.camera.pid_y.get()
        pid_f = self.camera.pid_f.get()
        self.camera.set_center(pid_x, pid_y, pid_f)

        # Update variables
        self.camera.ball_mu_last = ball_mu
        self.camera.u_last = u
        self.camera.is_initialized = True

    def measure_ball(self, bb_ball):
        """ Get the ball center point. """

        return helpers.get_bb_center(bb_ball)

    def measure_players(self, bbs):
        """ Get the players' center point in frame_orig space. """

        points = self.top_down.bbs2points(bbs)
        helpers.discard_extreme_points_(points)
        points_mu = helpers.points_average(points)
        self.players_var = helpers.points_variance(points, points_mu)
        x, y = helpers.apply_homography(self.top_down.H_inv, *points_mu)
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

        helpers.discard_extreme_boxes_(bbs)
        bb_x_min, _, bb_x_max, _ = helpers.get_bounding_box(bbs)
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
        return max(f_var, f_bb)

    def measure_u(self, balls_detected, players_center, ball_var):
        def measure_players_center():
            alpha = Config.autocam["u_control"]["center"]["alpha"]
            var_th = Config.autocam["u_control"]["center"]["var_th"]

            if not balls_detected and np.mean(ball_var) > var_th:
                return alpha * (players_center - self.camera.ball_mu_last)
            return np.array([0., 0.])

        def measure_players_movement():
            if not self.camera.is_initialized:
                self.camera.players_filter.set_pos(*players_center)
            self.camera.players_filter.predict()
            self.camera.players_filter.update(*players_center)
            return Config.autocam["u_control"]["velocity"]["alpha"] * self.camera.players_filter.vel.T[0]

        u = np.array([0., 0.])
        u += measure_players_center()
        u += measure_players_movement()

        return u

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
