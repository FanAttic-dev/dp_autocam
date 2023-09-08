import cv2
import numpy as np
from PID import PID
from constants import colors
from particle_filter import ParticleFilter
from utils import apply_homography, average_point, get_bb_center, get_bounding_box, lies_in_rectangle


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
    SENSOR_W = 836  # 25
    CYLLINDER_RADIUS = 1000

    PAN_DX = 1
    DEFAULT_PAN_DEG = 12
    PAN_DEG_MIN = -60
    PAN_DEG_MAX = 60

    TILT_DY = 1
    DEFAULT_TILT_DEG = 9
    TILT_DEG_MIN = 5
    TILT_DEG_MAX = 12

    ZOOM_DZ = 60
    DEFAULT_F = 1003  # 30
    F_MIN = 900
    F_MAX = 1300

    FRAME_ASPECT_RATIO = 16/9
    FRAME_W = 1920
    FRAME_H = int(FRAME_W / FRAME_ASPECT_RATIO)
    FRAME_CORNERS = np.array([
        [0, 0],
        [0, FRAME_H-1],
        [FRAME_W-1, FRAME_H-1],
        [FRAME_W-1, 0]
    ], dtype=np.int16)

    def __init__(self, frame_orig, pan_deg=DEFAULT_PAN_DEG, tilt_deg=DEFAULT_TILT_DEG):
        h, w, _ = frame_orig.shape
        self.frame_orig_shape = frame_orig.shape
        self.frame_orig_center_x = w // 2
        self.frame_orig_center_y = h // 2
        self.set(pan_deg, tilt_deg)

        self.pid_x = PID()
        self.pid_y = PID()
        self.pid_f = PID(kp=0.005)
        center_x, center_y = self.center
        self.pid_x.init(center_x)
        self.pid_y.init(center_y)
        self.pid_f.init(PerspectiveCamera.DEFAULT_F)

        self.ball_model = ParticleFilter()
        self.ball_model.init(self.center)
        self.ball_estimate_last = self.center

        self.pause_measurements = False
        self.init_dead_zone()

    def update_by_bbs(self, bbs, bbs_ball, top_down):
        def measure_ball(bb_ball):
            """ Get the ball center point. """
            return get_bb_center(bb_ball)

        def measure_players(bbs):
            """ Get the players' center point. """
            pts = top_down.bbs2points(bbs)
            x, y = average_point(pts)
            x, y = apply_homography(top_down.H_inv, x, y)
            return np.array([x, y])

        def measure_zoom():
            """ Map the PF variance to the camera zoom bounds. """
            var = np.mean(self.ball_model.var)
            var_min = self.ball_model.std_pos ** 2
            var_max = self.ball_model.std_pos ** 2 * 10
            var = np.clip(var, var_min, var_max)

            # zoom is inversely proportional to the variance
            zoom_level = 1 - (var - var_min) / (var_max - var_min)
            f = PerspectiveCamera.F_MIN + zoom_level * PerspectiveCamera.F_MAX
            return f

        def measure_u(balls_detected, players_center):
            """ Move to the players' center if no measurement for a long time. """
            var = self.ball_model.var
            var_u_th = self.ball_model.std_pos ** 2 * 50
            if not balls_detected and np.mean(var) > var_u_th:
                u = players_center - self.ball_estimate_last
                return u

            return np.array([0, 0])

        players_detected = len(bbs) > 0 and len(bbs["boxes"]) > 0
        balls_detected = len(bbs_ball) > 0 and len(bbs_ball['boxes']) > 0

        if not players_detected:
            return

        players_center = measure_players(bbs)

        # Set control input
        u = measure_u(balls_detected, players_center)
        self.ball_model.set_u(u)

        # Apply motion model with uncertainty to PF
        self.ball_model.predict()

        if balls_detected:
            ball_centers = [measure_ball(bb_ball)
                            for bb_ball in bbs_ball['boxes']]

            # Incorporate measurements into PF
            self.ball_model.update(players_center, ball_centers)

        self.ball_model.resample()

        # Camera model
        mu = self.ball_model.mu
        self.ball_estimate_last = mu
        mu_x, mu_y = mu
        f = measure_zoom()

        self.pid_x.update(mu_x)
        self.pid_y.update(mu_y)
        self.pid_f.update(f)
        # is_in_dead_zone = self.is_meas_in_dead_zone(*mu)
        # print(f"Is in dead zone: {is_in_dead_zone}")

        pid_x = self.pid_x.get()
        pid_y = self.pid_y.get()
        pid_f = self.pid_f.get()
        self.set_center(pid_x, pid_y, pid_f)

    @property
    def fov_horiz_deg(self):
        return np.rad2deg(2 * np.arctan(PerspectiveCamera.SENSOR_W / (2 * self.f)))

    @property
    def fov_vert_deg(self):
        return self.fov_horiz_deg / 16 * 9

    @property
    def center(self):
        return self.ptz2coords(self.pan_deg, self.tilt_deg, self.f)

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
        return H

    @property
    def H_inv(self):
        return np.linalg.inv(self.H)

    def set(self, pan_deg, tilt_deg, f=DEFAULT_F):
        self.pan_deg = np.clip(
            pan_deg,
            PerspectiveCamera.PAN_DEG_MIN,
            PerspectiveCamera.PAN_DEG_MAX,
        )
        self.tilt_deg = np.clip(
            tilt_deg,
            PerspectiveCamera.TILT_DEG_MIN,
            PerspectiveCamera.TILT_DEG_MAX
        )
        self.f = np.clip(
            f,
            PerspectiveCamera.F_MIN,
            PerspectiveCamera.F_MAX
        )

    def set_center(self, x, y, f=None):
        pan_deg, tilt_deg = self.coords2ptz(x, y)
        f = f if f is not None else self.f
        self.set(pan_deg, tilt_deg, f)

    def init_dead_zone(self):
        self.dead_zone = np.array([
            [640, 0],  # start point (top left)
            [1280, 1079]  # end point (bottom right)
        ])

    def is_meas_in_dead_zone(self, meas_x, meas_y):
        meas = np.array([[[meas_x.item(), meas_y.item()]]], dtype=np.float32)
        meas_frame_coord = cv2.perspectiveTransform(meas, self.H)[0][0]
        return lies_in_rectangle(meas_frame_coord, self.dead_zone)

    def reset(self):
        self.set(PerspectiveCamera.DEFAULT_PAN_DEG,
                 PerspectiveCamera.DEFAULT_TILT_DEG)

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
        x = np.tan(theta_rad) * PerspectiveCamera.CYLLINDER_RADIUS

        phi_rad = np.deg2rad(phi_deg)
        y = np.tan(phi_rad) * \
            np.sqrt(PerspectiveCamera.CYLLINDER_RADIUS**2 + x**2)
        return self.shift_coords(int(x), int(y))

    def coords2ptz(self, x, y):
        x, y = self.unshift_coords(x, y)
        pan_deg = np.rad2deg(np.arctan(x / PerspectiveCamera.CYLLINDER_RADIUS))
        tilt_deg = np.rad2deg(
            np.arctan(y / (np.sqrt(PerspectiveCamera.CYLLINDER_RADIUS**2 + x**2))))
        return pan_deg, tilt_deg

    def get_corner_pts(self):
        pts = [
            self.ptz2coords(
                self.pan_deg + pan_deg,
                self.tilt_deg + tilt_deg,
                self.f)
            for pan_deg, tilt_deg in self.corners_ang.values()
        ]

        return np.array(pts, dtype=np.int32)

    def draw_roi_(self, frame_orig, color=colors["yellow"]):
        pts = self.get_corner_pts()
        cv2.polylines(frame_orig, [pts], True, color, thickness=10)

    def draw_center_(self, frame_orig, color=colors["red"]):
        cv2.circle(frame_orig, self.center,
                   radius=5, color=color, thickness=5)

    def draw_ball_prediction_(self, frame_orig, color=colors["violet"]):
        x, y = self.ball_model.mu
        cv2.circle(frame_orig, (int(x), int(y)),
                   radius=5, color=color, thickness=5)

    def draw_dead_zone_(self, frame):
        start, end = self.dead_zone
        cv2.rectangle(frame, start, end,
                      color=colors["yellow"], thickness=5)

    def get_frame(self, frame_orig):
        return cv2.warpPerspective(
            frame_orig,
            self.H,
            (PerspectiveCamera.FRAME_W, PerspectiveCamera.FRAME_H),
            flags=cv2.INTER_LINEAR
        )

    def pan(self, dx):
        pan_deg = self.pan_deg + dx
        self.set(pan_deg, self.tilt_deg, self.f)

    def tilt(self, dy):
        tilt_deg = self.tilt_deg + dy
        self.set(self.pan_deg, tilt_deg, self.f)

    def zoom(self, dz):
        f = self.f + dz
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
        elif key == ord('r'):
            self.reset()
        elif key == ord('f'):
            self.set_center(mouseX, mouseY)
        elif key == ord('q'):
            is_alive = False
        elif key == ord('n'):
            self.pause_measurements = not self.pause_measurements
        return is_alive

    def get_stats(self):
        stats = {
            "Name": PerspectiveCamera.__name__,
            "pan_deg": self.pan_deg,
            "tilt_deg": self.tilt_deg,
            "f": self.f,
            "fov_horiz_deg": self.fov_horiz_deg,
            "fov_vert_deg": self.fov_vert_deg,
        }
        return stats

    def print(self):
        print(self.get_stats())


class FixedHeightCamera(Camera):
    def __init__(self, full_img):
        self.full_img_h, self.full_img_w, _ = full_img.shape
        self.h = 450  # self.full_img_h
        self.center_x = self.full_img_w // 2
        self.center_y = self.full_img_h // 2 + 180
        self.w = int(self.h / 9 * 16)

    def check_bounds(self, x, y):
        return x >= 0 and x+self.w < self.full_img_w and y >= 0 and y+self.h < self.full_img_h

    def get_frame_origin(self, center_x, center_y):
        x = center_x - self.w // 2
        y = center_y - self.h // 2
        return x, y

    def get_frame(self, full_img):
        if full_img is None:
            return None

        x, y = self.get_frame_origin(self.center_x, self.center_y)
        if not self.check_bounds(x, y):
            return None

        return full_img[y:y+self.h, x:x+self.w]

    def pan(self, dx):
        return self.set_center_x(self.center_x + dx)

    def set_center_x(self, center_x):
        x, y = self.get_frame_origin(center_x, self.center_y)
        if not self.check_bounds(x, y):
            return False

        self.center_x = center_x
        return True

    def update_by_bbs(self, bbs):
        if not bbs:
            return

        bb_centers = []
        for bb in bbs:
            x, y, w, h = bb
            bb_center_x = x + w//2
            bb_center_y = y + h//2
            bb_centers.append((bb_center_x, bb_center_y))

        center_x = sum(map(
            lambda bb_center: bb_center[0], bb_centers)) // len(bb_centers)

        if self.check_bounds(self.get_frame_x(center_x)):
            self.center_x = center_x
