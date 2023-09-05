import cv2
import numpy as np
from numpy.random import random, randn
import scipy
from constants import colors
from filterpy.monte_carlo import systematic_resample


class ParticleFilter():
    PARTICLES_STD = 40

    def __init__(self, dt=0.1, std_meas=10, std_u=1, N=10000):
        self.dt = dt
        self.std_meas = std_meas
        self.std_u = std_u
        self.N = N
        self.particles = None
        self.weights = None

    def init(self, mean):
        self.particles = self.generate_gaussian_particles(
            mean, std=ParticleFilter.PARTICLES_STD)
        self.weights = np.ones(self.N) / self.N

    def generate_gaussian_particles(self, mean, std):
        particles = np.empty((self.N, 2))
        particles[:, 0] = mean[0] + (randn(self.N) * std)
        particles[:, 1] = mean[1] + (randn(self.N) * std)
        return particles

    @property
    def estimate(self):
        pos = self.particles[:, 0:2]
        mean = np.average(pos, weights=self.weights, axis=0)
        var = np.average((pos - mean)**2, weights=self.weights, axis=0)
        return mean, var

    @property
    def neff(self):
        return 1. / np.sum(np.square(self.weights))

    def resample_from_index(self, indexes):
        self.particles[:] = self.particles[indexes]
        self.weights.resize(len(self.particles))
        self.weights.fill(1.0 / len(self.weights))

    def predict(self):
        # self.particles += self.vel * self.dt + randn(self.N, 2) * self.std_u
        # TODO use ball dynamics
        ...

    def update(self, zs, ball_centers):
        for i, ball in enumerate(ball_centers):
            dist = np.linalg.norm(self.particles[:, 0:2] - ball, axis=1)
            self.weights *= scipy.stats.norm(dist, self.std_meas).pdf(zs[i])

        self.weights += 1.e-300  # avoid round-off to zero
        self.weights /= sum(self.weights)

    def resample(self):
        if self.neff < self.N / 2:
            print("Resampling")
            indexes = systematic_resample(self.weights)
            self.resample_from_index(indexes)
            assert np.allclose(self.weights, 1/self.N)

    def draw_particles_(self, frame, color=colors["red"]):
        for particle, weight in zip(self.particles, self.weights):
            x, y = particle
            cv2.circle(frame, (int(x), int(y)), radius=0,
                       color=color, thickness=int(weight * 100))

    def get_stats(self):
        return {
            "Name": "Particle Filter",
            "dt": self.dt,
            "std_meas": self.std_meas,
        }
