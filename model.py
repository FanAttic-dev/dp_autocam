class Model:
    def __init__(self, decel_rate):
        self.decel_rate = decel_rate
        self.set_decelerating(False)
        self.last_measurement = None

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

    def set_last_measurement(self, x_meas, y_meas):
        self.last_measurement = x_meas, y_meas

    def update(self, x_meas, y_meas):
        ...