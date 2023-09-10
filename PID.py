import numpy as np


class PID:
    def __init__(self, kp=0.03, ki=0, kd=0):
        self.dt = 0.01
        self.kp = kp
        self.ki = ki  # 0.009
        self.kd = kd  # 0.006

        self.P = 0
        self.I = 0
        self.D = 0

        self.target = 0
        self.init(0)

    def init(self, signal):
        self.signal = signal
        self.set_target(signal)

    def set_target(self, target):
        if np.isclose(target, self.target):
            return
        self.target = target
        self.e_prev = self.target - self.signal

    def get(self):
        return self.signal

    def update(self, target):
        self.set_target(target)
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
