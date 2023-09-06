import cv2
import numpy as np
from numpy.random import random, randn
import scipy
from constants import colors
from filterpy.monte_carlo import systematic_resample, stratified_resample, residual_resample, multinomial_resample


class ParticleFilter():
    PARTICLES_STD = 50

    def __init__(self, dt=1, std_pos=20, std_meas=80, N=1000):
        self.dt = dt
        self.std_pos = std_pos
        self.std_meas = std_meas
        self.u = np.array([0, 0])
        self.N = N
        self.particles = None
        self.weights = None

    def init(self, mean):
        self.particles = self.generate_gaussian_particles(
            mean, std=ParticleFilter.PARTICLES_STD)
        self.weights = np.ones(self.N) / self.N

    def generate_gaussian_particles(self, mean, std):
        # particles = np.empty((self.N, 2))
        # particles[:, 0] = mean[0] + (randn(self.N) * std)
        # particles[:, 1] = mean[1] + (randn(self.N) * std)
        # return particles
        particles = mean + randn(self.N, 2) * std
        return particles

    @property
    def estimate(self):
        mean = np.average(self.particles, weights=self.weights, axis=0)
        var = np.average((self.particles - mean)**2,
                         weights=self.weights, axis=0)
        return mean, var

    @property
    def neff(self):
        return 1. / np.sum(np.square(self.weights))

    def resample_from_index(self, indexes):
        self.particles = self.particles[indexes]
        self.weights.resize(len(self.particles))
        self.weights.fill(1.0 / len(self.weights))

    def predict(self):
        self.particles += randn(self.N, 2) * self.std_pos

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
            # indexes = stratified_resample(self.weights)
            # indexes = multinomial_resample(self.weights)
            # indexes = residual_resample(self.weights)
            self.resample_from_index(indexes)
            assert np.allclose(self.weights, 1/self.N)

    def draw_particles_(self, frame, color=colors["red"]):
        for particle, weight in zip(self.particles, self.weights):
            x, y = particle
            cv2.circle(frame, (int(x), int(y)), radius=int(weight * self.N),
                       color=color, thickness=-1)

    def get_stats(self):
        return {
            "Name": "Particle Filter",
            "dt": self.dt,
            "std_meas": self.std_meas,
        }
