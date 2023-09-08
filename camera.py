import cv2
import numpy as np
from PID import PID
from constants import colors
from dynamics import Dynamics
from kalman_filter import KalmanFilterAcc, KalmanFilterVel
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
    PAN_DX = 1
    TILT_DY = 1
    ZOOM_DZ = 60
    SENSOR_W = 836  # 25
    DEFAULT_F = 1003  # 30
    MIN_F = 1003
    MAX_F = 2083
    CYLLINDER_RADIUS = 1000
    DEFAULT_PAN_DEG = 12
    MIN_PAN_DEG = -60
    MAX_PAN_DEG = 60
    DEFAULT_TILT_DEG = 9
    MIN_TILT_DEG = 9  # 5
    MAX_TILT_DEG = 9  # 12
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

        # self.model = Dynamics(dt=0.1, accel_rate=0.1, decel_rate=0.1)
        self.model = PID()
        # self.model = KalmanFilterVel(
        #     dt=0.1, std_acc=0.1, std_meas=100, decel_rate=1)
        center_x, center_y = self.center
        self.model.init(center_x)

        self.ball_model = ParticleFilter()
        self.ball_model.init(self.center)
        self.ball_estimate_last = self.center
        # self.ball_model = KalmanFilterVel(
        #     dt=0.1, std_acc=0.1, std_meas=0.05, decel_rate=0.1)

        self.pause_measurements = False
        self.init_dead_zone()

    @property
    def variance_threshold(self):
        return self.ball_model.std_pos ** 2 * 100

    def update_by_bbs(self, bbs, bbs_ball, top_down):
        def measure_ball(bb_ball):
            return get_bb_center(bb_ball)

        def measure_players(bbs):
            pts = top_down.bbs2points(bbs)
            x, y = average_point(pts)
            x, y = apply_homography(top_down.H_inv, x, y)
            return x, y

        def measure_zoom():
            self.ball_model.var

        # def measure_zoom(bbs):
        #     bb_x_min, bb_y_min, bb_x_max, bb_y_max = get_bounding_box(bbs)

        #     corner_pts = self.get_corner_pts()
        #     roi_x_min, roi_y_min = corner_pts[0]
        #     roi_x_max, roi_y_max = corner_pts[2]

        #     dz = (bb_x_min - roi_x_min) + (roi_x_max - bb_x_max)

        #     return dz

        players_detected = len(bbs) > 0 and len(bbs["boxes"]) > 0
        balls_detected = len(bbs_ball) > 0 and len(bbs_ball['boxes']) > 0

        if not players_detected:
            return

        players_center = np.array(measure_players(bbs))
        ball_centers = []

        # Move to the players' center if no measurement for a long time
        # var = self.ball_model.var
        # if not balls_detected and np.mean(var) > self.variance_threshold:
        #     u = players_center - self.ball_estimate_last
        #     self.ball_model.set_u(u)
        # else:
        #     self.ball_model.reset_u()

        self.ball_model.predict()

        if balls_detected:
            # Distance from estimate to measurement
            ball_centers = [measure_ball(bb_ball)
                            for bb_ball in bbs_ball['boxes']]

            # Incorporate measurements
            self.ball_model.update(players_center, ball_centers)

        self.ball_model.resample()

        mu = self.ball_model.mu
        self.ball_estimate_last = mu

        # is_in_dead_zone = self.is_meas_in_dead_zone(*mu)
        # print(f"Is in dead zone: {is_in_dead_zone}")

        # Camera model
        self.model.set_target(mu[0])
        self.model.update()

        center_x, center_y = self.center
        pid_x = self.model.get()
        self.set_center(pid_x, center_y)

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
            PerspectiveCamera.MIN_PAN_DEG,
            PerspectiveCamera.MAX_PAN_DEG,
        )
        self.tilt_deg = np.clip(
            tilt_deg,
            PerspectiveCamera.MIN_TILT_DEG,
            PerspectiveCamera.MAX_TILT_DEG
        )
        self.f = np.clip(
            f,
            PerspectiveCamera.MIN_F,
            PerspectiveCamera.MAX_F
        )

    def set_center(self, x, y):
        pan_deg, tilt_deg = self.coords2ptz(x, y)
        self.set(pan_deg, tilt_deg, self.f)

    def init_dead_zone(self):
        # left = self.model.signal - self.model.th
        # right = self.model.signal + self.model.th
        # return np.array([
        #     [int(left), 0],  # start point (top left)
        #     # end point (bottom right)
        #     [int(right), self.frame_orig_shape[0]-1]
        # ])
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

    def print(self):
        print(f"pan_deg = {self.pan_deg}")
        print(f"tilt_deg = {self.tilt_deg}")
        print(f"f = {self.f}")
        print(f"fov_horiz = {self.fov_horiz_deg}")
        print(f"fov_vert = {self.fov_vert_deg}")
        print()

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
