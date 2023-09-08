import cv2
import numpy as np
from numpy.random import randn
import scipy
from constants import colors
from filterpy.monte_carlo import systematic_resample


class ParticleFilter():
    INIT_STD = 100

    def __init__(self, dt=0.1, std_pos=10, std_meas=90, N=1000):
        self.dt = dt
        self.std_pos = std_pos
        self.std_meas = std_meas
        self.reset_u()
        self.N = N
        self.particles = None
        self.weights = None
        self.mean_last = None

    def init(self, mu):
        self.particles = self.generate_gaussian_particles(
            mu, std=ParticleFilter.INIT_STD)
        self.weights = np.ones(self.N) / self.N

    def reset_u(self):
        self.u = np.array([0, 0])

    def generate_gaussian_particles(self, mu, std):
        particles = mu + randn(self.N, 2) * std
        return particles

    @property
    def mu(self):
        return np.average(self.particles, weights=self.weights, axis=0)

    @property
    def var(self):
        return np.average((self.particles - self.mu)**2,
                          weights=self.weights, axis=0)

    @property
    def neff(self):
        return 1. / np.sum(np.square(self.weights))

    def set_u(self, u):
        self.u = u

    def resample_from_index(self, indexes):
        self.particles = self.particles[indexes]
        self.weights.resize(len(self.particles))
        self.weights.fill(1.0 / len(self.weights))

    def predict(self):
        self.particles += self.u * self.dt + randn(self.N, 2) * self.std_pos

    def update(self, players_center, ball_centers):
        players_ball_alpha = 0.7

        dist_players = np.linalg.norm(self.particles - players_center, axis=1)
        for ball in ball_centers:
            dist_ball = np.linalg.norm(self.particles - ball, axis=1)
            self.weights *= scipy.stats.norm(0, self.std_meas).pdf(
                players_ball_alpha * dist_ball + (1-players_ball_alpha) * dist_players)
            # self.weights *= scipy.stats.norm(0, self.std_meas).pdf(dist_ball)

        self.weights += 1.e-300  # avoid round-off to zero
        self.weights /= sum(self.weights)

    def resample(self):
        if self.neff < self.N / 2:
            print("Resampling")
            indexes = systematic_resample(self.weights)
            # indexes = stratified_resample(self.weights)
            self.resample_from_index(indexes)
            # assert np.allclose(self.weights, 1/self.N)

    def draw_particles_(self, frame, color=colors["red"]):
        for particle, weight in zip(self.particles, self.weights):
            x, y = particle
            cv2.circle(frame, (int(x), int(y)), radius=int(weight * self.N),
                       color=color, thickness=-1)

    def get_stats(self):
        return {
            "Name": "Particle Filter",
            "N": self.N,
            "dt": self.dt,
            "std_pos": self.std_pos,
            "std_meas": self.std_meas,
            "u": self.u,
            "mu": self.mu,
            "var": self.var
        }