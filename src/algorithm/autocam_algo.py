from algorithm.algo import Algo
from camera.camera import Camera
from camera.top_down import TopDown
from filters.kalman_filter import KalmanFilterVel
from filters.particle_filter import ParticleFilter
from utils.config import Config
from utils.constants import DT_INT, Color
import utils.utils as utils
import numpy as np
import cv2


class AutocamAlgo(Algo):
    def __init__(self, camera: Camera, top_down: TopDown, config: Config):
        self.camera = camera
        self.top_down = top_down

        self.ball_filter = ParticleFilter(config.autocam["ball_pf"])
        self.ball_filter.init(self.camera.center)

        self.players_filter = KalmanFilterVel(
            config.autocam["players_kf"]["dt"],
            config.autocam["players_kf"]["std_acc"],
            config.autocam["players_kf"]["std_meas"],
        )

        # Used for PF visualization and control input calculation.
        self.ball_mu_last = self.camera.center

        self.players_var = None  # Used for debug info only.
        self.u_last = None  # Used for visualization only.
        self.is_initialized = False

    def update_by_bbs(self, bbs):
        bbs_ball = self.filter_bbs_ball(bbs)

        players_detected = len(bbs) > 0 and len(bbs["boxes"]) > 0
        balls_detected = len(bbs_ball) > 0 and len(bbs_ball['boxes']) > 0
        if not players_detected:
            return

        utils.discard_extreme_bbs_(bbs)

        # Players center
        players_center = self.get_players_center(bbs)
        if not self.is_initialized:
            self.players_filter.set_pos(*players_center)
            self.is_initialized = True

        # Incorporate measurements into PF
        if balls_detected:
            ball_centers = [
                utils.get_bb_center(bb_ball) for bb_ball in bbs_ball['boxes']
            ]
            self.ball_filter.update(players_center, ball_centers)
            self.ball_filter.resample()

        # Apply motion model with uncertainty to PF
        self.ball_filter.predict()

        # Get PF estimate
        ball_mu, ball_var = self.ball_filter.estimate
        f = self.get_zoom_f(ball_var, bbs)

        # Set control input
        u = self.get_u(balls_detected, players_center, ball_var)
        self.ball_filter.set_u(u)

        # Update camera
        self.try_update_camera(ball_mu, f)

        # Update variables
        self.ball_mu_last = ball_mu
        self.u_last = u

    def try_update_camera(self, center, f=None):
        """Try to update the camera PID target.

        It first converts the points to PTZ,
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
            print("[try_update_camera] Camera target out of bounds.")
        else:
            center_screen = self.camera.ptz2screen(*center_ptz)
            target = (*center_screen, f)

        self.camera.update_pid(*target)
        self.camera.set_center(*self.camera.pid_signal)
        return is_valid

    def get_players_center(self, bbs):
        """Get the players' center point in frame_orig space."""
        tdpts = self.top_down.bbs_screen2tdpts(bbs)
        tdpts_mu = utils.pts_average(tdpts["pts"])
        self.players_var = utils.pts_variance(tdpts["pts"], tdpts_mu)

        x, y = utils.apply_homography(self.top_down.H_inv, *tdpts_mu)
        return np.array([x, y])

    def get_zoom_f(self, ball_var, bbs):
        f_var = self.get_zoom_by_ball_var(ball_var)
        f_bb = self.get_zoom_by_bbs(bbs)
        return max(f_var, f_bb)

    def get_zoom_by_ball_var(self, ball_var):
        """Map the PF variance to the camera zoom bounds."""
        ball_var = np.mean(ball_var)
        var_min = Config.autocam["zoom"]["var_min"]
        var_max = Config.autocam["zoom"]["var_max"]
        ball_var = np.clip(ball_var, var_min, var_max)

        # zoom is inversely proportional to the variance
        zoom_level = 1 - (ball_var - var_min) / (var_max - var_min)
        zoom_range = self.camera.zoom_f_max - self.camera.zoom_f_min
        f = self.camera.zoom_f_min + zoom_level * zoom_range
        return f

    def get_zoom_by_bbs(self, bbs):
        """Calculate the focal length based on the players' bounding box."""
        if not bbs or len(bbs["boxes"]) == 0:
            return self.camera.zoom_f

        margin_px = Config.autocam["zoom"]["bb"]["margin_px"]

        bb_x_min, _, bb_x_max, _ = utils.get_bbs_bounding_box(bbs)
        bb_x_min -= margin_px
        bb_x_max += margin_px
        bb_width = bb_x_max - bb_x_min

        fov_target_deg = self.camera.screen_width_px2fov(bb_width)
        f = self.camera.fov2f(fov_target_deg)
        f = np.clip(f, self.camera.zoom_f_min, self.camera.zoom_f_max)
        return f

    def get_u(self,
              balls_detected: bool,
              players_center: tuple[int, int],
              ball_var: tuple[float, float]):
        def get_center_vector():
            """Calculate the ball -> players' center vector.

            This is to avoid the PF to drift away when no ball detected
            for a longer period of time.

            Args:
                balls_detected: Whether any ball was detected.
                players_center: x, y coordinates of the players' center.
                ball_var: x, y variance of the particle filter 
                    used as a threshold when deciding wheter to start moving
                    the particles to the players' center.
            Returns:
                Ball -> players' center vector if ball_var exceeds 
                PF variance threshold, else [0, 0] vector.
            """

            alpha = Config.autocam["u_control"]["center"]["alpha"]
            var_th = Config.autocam["u_control"]["center"]["var_th"]

            if not balls_detected and np.mean(ball_var) > var_th:
                return alpha * (players_center - self.ball_mu_last)
            return np.array([0., 0.])

        def get_movement_vector():
            """Get the players' center velocity vector (multiplied by an alpha)."""
            alpha = Config.autocam["u_control"]["velocity"]["alpha"]

            self.players_filter.predict()
            self.players_filter.update(*players_center)

            return alpha * self.players_filter.vel.T[0]

        u = np.array([0., 0.])
        u += get_center_vector()
        u += get_movement_vector()

        return u

    def filter_bbs_ball(self, bbs):
        """Filters bbs of class ball."""
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
        pt1 = np.array([mu_x, mu_y], dtype=DT_INT)
        pt2 = np.array([mu_x + u_x, mu_y + u_y], dtype=DT_INT)
        cv2.line(frame_orig, pt1, pt2, color=color, thickness=2)

    def draw_ball_prediction_(self, frame_orig, color):
        x, y = self.ball_mu_last
        cv2.circle(frame_orig, (int(x), int(y)),
                   radius=4, color=color, thickness=5)

    def draw_players_bb_(self, frame_orig, bbs, color=Color.TEAL):
        if len(bbs["boxes"]) == 0:
            return

        margin_px = Config.autocam["zoom"]["bb"]["margin_px"]
        x1, y1, x2, y2 = utils.get_bbs_bounding_box(bbs)
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
