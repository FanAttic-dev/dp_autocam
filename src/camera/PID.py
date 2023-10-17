import numpy as np

from utils.protocols import HasStats


class PID(HasStats):
    def __init__(self, kp, ki=0, kd=0):
        self.dt = 0.01
        self.kp = kp
        self.ki = ki  # 0.009
        self.kd = kd  # 0.006

        self.P = 0
        self.I = 0
        self.D = 0

        self.target = 0
        self.e_prev = 0
        self.init(0)

    def init(self, signal):
        self.signal = signal
        self._set_target(signal)

    def _set_target(self, target):
        self.target = target

    def update(self, target=None):
        if target is not None:
            self._set_target(target)

        e = self.target - self.signal

        self.P = self.kp * e
        self.I += self.ki * e * self.dt
        self.D = self.kd * (e - self.e_prev) / self.dt
        self.signal += self.P + self.I + self.D

        self.e_prev = e

    def get_stats(self):
        stats = {
            "Name": "PID Controller",
            # "dt": self.dt,
            "signal": self.signal,
            "target": self.target,
            "Kp": self.kp,
            "P": self.P,
        }
        if self.ki > 0:
            stats["Ki"] = self.ki
            stats["I"] = self.I

        if self.kd > 0:
            stats["Kd"] = self.kd
            stats["D"] = self.D

        return stats
