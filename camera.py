class Camera:
    def __init__(self):
        ...


class FixedHeightCamera(Camera):
    def __init__(self, full_img):
        self.full_img_h, self.full_img_w, _ = full_img.shape
        self.center_x = self.full_img_w // 2
        self.w = int(self.full_img_h / 9 * 16)

    def check_bounds(self, x):
        return x >= 0 and x+self.w < self.full_img_w

    def get_frame_x(self, center_x):
        return center_x - self.w // 2

    def get_frame(self, full_img):
        if full_img is None:
            return None

        x = self.get_frame_x(self.center_x)
        if not self.check_bounds(x):
            return None

        return full_img[:, x:x+self.w]

    def pan(self, dx):
        self.center_x += dx

    def set_center_x(self, center_x):
        x = self.get_frame_x(center_x)
        if not self.check_bounds(x):
            return False

        self.center_x = center_x
        return True
