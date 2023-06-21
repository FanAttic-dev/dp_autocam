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
    def __init__(self, full_img, fov_horiz_deg=35, fov_vert_deg=15, width_px=1000):
        self.full_img_h, self.full_img_w, _ = full_img.shape
        self.fov_horiz_deg = fov_horiz_deg
        self.fov_vert_deg = fov_vert_deg
        self.r = width_px / (2 * np.tan(np.deg2rad(self.fov_horiz_deg / 2)))
        self.center_x = self.full_img_w // 2
        self.center_y = self.full_img_h // 2
        self.pan_deg = 0
        self.tilt_deg = 0

    def check_bounds(self, x, y):
        return x >= 0 and x < self.full_img_w and y >= 0 and y < self.full_img_h

    def shift_coords(self, x, y):
        x = x + self.center_x
        y = y + self.center_y
        return int(x), int(y)

    def get_frame(self, full_img):
        for theta in range(-self.fov_horiz_deg // 2, self.fov_horiz_deg // 2):
            for phi in [-self.fov_vert_deg // 2, self.fov_vert_deg // 2]:

                theta_rad = np.deg2rad(theta + self.pan_deg)
                x = np.tan(theta_rad) * self.r

                phi_rad = np.deg2rad(phi + self.tilt_deg)
                y = np.tan(phi_rad) * np.sqrt(self.r**2 + x**2)

                x, y = self.shift_coords(x, y)
                if self.check_bounds(x, y):
                    cv2.circle(full_img, (x, y), radius=10,
                               color=(0, 255, 255), thickness=-1)

        return full_img

    def pan(self, dx):
        # TODO: check bounds
        self.pan_deg += dx

    def tilt(self, dy):
        self.tilt_deg += dy


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
