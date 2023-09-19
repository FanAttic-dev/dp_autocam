import cv2
import numpy as np
from numpy.random import randn
import scipy
from scipy import stats
from constants import colors, params
from filterpy.monte_carlo import systematic_resample


class ParticleFilter():
    INIT_STD = 100

    def __init__(self, pf_params):
        self.dt = pf_params["dt"]
        self.std_pos = pf_params["std_pos"]
        self.std_meas = pf_params["std_meas"]
        self.reset_u()
        self.N = pf_params["N"]
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
    def estimate(self):
        mu = self.mu
        var = np.average((self.particles - self.mu)**2,
                         weights=self.weights, axis=0)
        return mu, var

    @property
    def mu(self):
        return np.average(self.particles, weights=self.weights, axis=0)

    @property
    def var(self):
        _, var = self.estimate
        return var

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
        players_ball_alpha = params["ball_pf"]["players_ball_alpha"]

        mu_pred = self.mu + self.u
        ball_weights = np.array(list(map(lambda ball: np.linalg.norm(
            mu_pred - ball), ball_centers)))
        ball_weights /= sum(ball_weights)
        ball_weights = 1 / ball_weights  # the closer, the larger weight

        dist_players = np.linalg.norm(self.particles - players_center, axis=1)
        w = np.zeros(len(self.particles))
        for ball, ball_weight in zip(ball_centers, ball_weights):
            dist_ball = np.linalg.norm(self.particles - ball, axis=1)

            x = players_ball_alpha * dist_ball + \
                (1-players_ball_alpha) * dist_players

            w += ball_weight * stats.norm(0, self.std_meas).pdf(x)

        self.weights *= w

        self.weights += 1.e-300  # avoid round-off to zero
        self.weights /= sum(self.weights)

    def resample(self):
        if self.neff < self.N / 2:
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
