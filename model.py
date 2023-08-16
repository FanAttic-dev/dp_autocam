class Model:
    @property
    def pos(self):
        ...

    def set_pos(self, x, y):
        ...

    @property
    def vel(self):
        ...

    def set_vel(self, x, y):
        ...

    def set_decelerating(self, is_decelerating):
        self.is_decelerating = is_decelerating

    def get_stats(self):
        ...

    def print(self):
        ...

    def predict(self):
        ...

    def update(self, x_meas, y_meas):
        ...
