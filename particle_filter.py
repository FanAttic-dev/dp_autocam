import numpy as np

from model import Model


class ParticleFilter(Model):
    def __init__(self, dt, std_meas):
        super().__init__(decel_rate=0)
        self.dt = dt
        self.std_meas = std_meas
        self.init_x()

    @property
    def pos(self):
        return np.array([self.x[0], self.x[2]])

    @property
    def vel(self):
        return np.array([self.x[1], self.x[3]])

    def init_x(self):
        self.x = np.array([
            [0],  # x
            [0],  # x'
            [0],  # y
            [0],  # y'
        ])

    def set_pos(self, x, y):
        self.x[0] = x
        self.x[2] = y

    def get_stats(self):
        return {
            "Name": "Particle Filter",
            "dt": self.dt,
            "std_meas": self.std_meas,
        }

    def predict(self):
        # TODO
        ...

    def update(self, x_meas, y_meas):
        self.last_measurement = x_meas, y_meas
        # TODO
        ...
