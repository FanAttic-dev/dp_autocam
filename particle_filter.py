import cv2
import numpy as np
from numpy.random import random, randn
from constants import colors

from model import Model


class ParticleFilter(Model):
    def __init__(self, dt, std_meas):
        super().__init__(decel_rate=0)
        self.dt = dt
        self.std_meas = std_meas
        self.N = 1000
        self.init_x()
        self.particles = None
        self.weights = None

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

    def generate_gaussian_particles(self, mean, std):
        N = self.N
        particles = np.empty((N, 2))
        particles[:, 0] = mean[0] + (randn(N) * std)
        particles[:, 1] = mean[1] + (randn(N) * std)
        return particles

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
        self.set_last_measurement(x_meas, y_meas)
        # TODO
        ...

    def set_last_measurement(self, x_meas, y_meas):
        is_initialized = self.last_measurement is not None
        self.last_measurement = x_meas, y_meas

        if not is_initialized:
            self.particles = self.generate_gaussian_particles(
                mean=self.last_measurement, std=50)
            self.weights = np.ones(self.N) / self.N

    def draw_particles_(self, frame, color=colors["red"]):
        for particle in self.particles:
            x, y = particle
            cv2.circle(frame, (int(x), int(y)), radius=1,
                       color=color, thickness=-1)
