import numpy as np

from model import Model


class Dynamics(Model):
    DECELERATION_RATE = 1.5

    def __init__(self, dt, alpha):
        self.dt = dt
        self.alpha = alpha
        self.is_decelerating = False

        self.F = np.array([
            [1, dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dt],
            [0, 0, 0, 1],
        ])

        self.G = np.array([
            [alpha * dt, 0],
            [alpha, 0],
            [0, alpha * dt],
            [0, alpha],
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
        ])

    def set_decelerating(self, is_decelerating):
        self.is_decelerating = is_decelerating

    def predict(self):
        self.x = self.F @ self.x

        if self.is_decelerating:
            self.u = -Dynamics.DECELERATION_RATE * self.vel
            self.x = self.x + self.G @ self.u

    def update(self, x_meas, y_meas):
        x_pos, y_pos = self.pos
        dx = x_meas - x_pos
        dy = y_meas - y_pos

        self.set_u(dx, dy)
        self.x = self.x + self.G @ self.u

    def get_stats(self):
        stats = {
            "Name": "Dynamics Velocity Model",
            "Decel. rate": Dynamics.DECELERATION_RATE,
            "dt": self.dt,
            "alpha": self.alpha,
            "Pos": [f"{self.x[0].item():.2f}", f"{self.x[2].item():.2f}"],
            "Vel": [f"{self.x[1].item():.2f}", f"{self.x[3].item():.2f}"],
            "U": [f"{self.u[0].item():.2f}", f"{self.u[1].item():.2f}"],
        }
        return stats

    def print(self):
        print(self.get_stats())
