import cv2
import numpy as np


class Camera:
    def __init__(self):
        ...

    def get_frame(self, full_img):
        ...

    def pan(self, dx):
        ...


class PerspectiveCamera(Camera):
    SENSOR_W = 10
    CYLLINDER_RADIUS = 1000
    FRAME_W = 1920
    FRAME_ASPECT_RATIO = 16/9

    def __init__(self, full_img):
        self.full_img_h, self.full_img_w, _ = full_img.shape
        self.center_x = self.full_img_w // 2
        self.center_y = self.full_img_h // 2
        self.reset()

    @property
    def fov_horiz_deg(self):
        return np.rad2deg(2 * np.arctan(PerspectiveCamera.SENSOR_W / (2 * self.f)))

    @property
    def fov_vert_deg(self):
        return self.fov_horiz_deg / 16 * 9

    def reset(self):
        self.pan_deg = 0
        self.tilt_deg = 0
        self.f = 12

    def shift_coords(self, x, y):
        x = x + self.center_x
        y = y + self.center_y
        return x, y

    def get_coords(self, theta_deg, phi_deg, f):
        theta_rad = np.deg2rad(theta_deg)
        x = np.tan(theta_rad) * PerspectiveCamera.CYLLINDER_RADIUS

        phi_rad = np.deg2rad(phi_deg)
        y = np.tan(phi_rad) * \
            np.sqrt(PerspectiveCamera.CYLLINDER_RADIUS**2 + x**2)
        return self.shift_coords(x, y)

    def get_corner_coords(self, pan_deg, tilt_deg, f):
        pts = []
        for theta_deg in [-self.fov_horiz_deg // 2, self.fov_horiz_deg // 2]:
            for phi_deg in [-self.fov_vert_deg // 2, self.fov_vert_deg // 2]:
                x, y = self.get_coords(
                    pan_deg + theta_deg,
                    tilt_deg + phi_deg,
                    f
                )
                pts.append([x, y])
        return pts

    def check_coord_bounds(self, x, y):
        return x >= 0 and x < self.full_img_w and y >= 0 and y < self.full_img_h

    def check_ptz_bounds(self, pan_deg, tilt_deg, f):
        coords = self.get_corner_coords(pan_deg, tilt_deg, f)
        return all([self.check_coord_bounds(x, y) for x, y in coords])

    def draw_roi_(self, full_img):
        for theta_deg in range(-int(self.fov_horiz_deg) // 2, int(self.fov_horiz_deg) // 2):
            for phi_deg in [-self.fov_vert_deg / 2, self.fov_vert_deg / 2]:
                x, y = self.get_coords(
                    self.pan_deg + theta_deg,
                    self.tilt_deg + phi_deg,
                    self.f
                )
                cv2.circle(full_img, (int(x), int(y)), radius=10,
                           color=(0, 255, 255), thickness=-1)

    def get_frame(self, full_img):
        src = np.array(self.get_corner_coords(
            self.pan_deg, self.tilt_deg, self.f), dtype=np.uint16)
        frame_w = PerspectiveCamera.FRAME_W
        frame_h = int(frame_w / PerspectiveCamera.FRAME_ASPECT_RATIO)
        dst = np.array([
            [0, 0],
            [0, frame_h-1],
            [frame_w-1, 0],
            [frame_w-1, frame_h-1]
        ], dtype=np.uint16)
        H, _ = cv2.findHomography(src, dst)
        return cv2.warpPerspective(full_img, H, (frame_w, frame_h), flags=cv2.INTER_LINEAR)

    def pan(self, dx):
        pan_deg = self.pan_deg + dx
        if not self.check_ptz_bounds(pan_deg, self.tilt_deg, self.f):
            return
        self.pan_deg = pan_deg

    def tilt(self, dy):
        tilt_deg = self.tilt_deg + dy
        if not self.check_ptz_bounds(self.pan_deg, tilt_deg, self.f):
            return
        self.tilt_deg = tilt_deg

    def zoom(self, dz):
        f = self.f + dz
        if not self.check_ptz_bounds(self.pan_deg, self.tilt_deg, f):
            return
        print(f"f = {f}")
        print(f"fov_horiz = {self.fov_horiz_deg}")
        self.f = f


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
        print(center_x)
        if self.check_bounds(self.get_frame_x(center_x)):
            self.center_x = center_x
