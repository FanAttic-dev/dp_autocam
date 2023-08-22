import numpy as np

from model import Model


class Dynamics(Model):
    def __init__(self, dt, accel_rate, decel_rate):
        super().__init__(decel_rate)
        self.dt = dt
        self.accel_rate = accel_rate

        self.F = np.array([
            [1, dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dt],
            [0, 0, 0, 1],
        ])

        self.G = np.array([
            [dt, 0],
            [1, 0],
            [0, dt],
            [0, 1],
        ])

        self.x = np.array([
            [0],  # x
            [0],  # x'
            [0],  # y
            [0],  # y'
        ])

        self.set_u(0, 0)

    @property
    def pos(self):
        return np.array([self.x[0], self.x[2]])

    def set_pos(self, x, y):
        self.x[0] = x
        self.x[2] = y

    @property
    def vel(self):
        return np.array([self.x[1], self.x[3]])

    def set_vel(self, x, y):
        self.x[1] = x
        self.x[3] = y

    def set_u(self, dx, dy):
        self.u = np.array([
            dx,
            dy
        ]) * self.accel_rate

    def set_decelerating(self, is_decelerating):
        self.is_decelerating = is_decelerating

    def predict(self):
        self.x = self.F @ self.x

        if self.is_decelerating:
            print("Decelerating")
            self.u = -self.decel_rate * self.vel
            self.x = self.x + self.G @ self.u

    def update(self, x_meas, y_meas):
        self.set_last_measurement(x_meas, y_meas)

        x_pos, y_pos = self.pos
        dx = x_meas - x_pos
        dy = y_meas - y_pos

        self.set_u(dx, dy)
        self.x = self.x + self.G @ self.u

    def get_stats(self):
        stats = {
            "Name": "Dynamics Velocity Model",
            "is_decelerating": self.is_decelerating,
            "Decel. rate": self.decel_rate,
            "Accel. rate": self.accel_rate,
            "dt": self.dt,
            "Pos": [f"{self.x[0].item():.2f}", f"{self.x[2].item():.2f}"],
            "Vel": [f"{self.x[1].item():.2f}", f"{self.x[3].item():.2f}"],
            "U": [f"{self.u[0].item():.2f}", f"{self.u[1].item():.2f}"],
        }
        return stats

    def print(self):
        print(self.get_stats())
