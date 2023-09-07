import numpy as np


class PID:
    def __init__(self):
        self.dt = 0.01
        self.kp = 0.05
        self.ki = 0.1

        self.th = 50

        self.e_prev = None
        self.I = 0

        self.set(0)

    def set(self, signal):
        self.signal = signal
        self.target = signal

    def get(self):
        return self.signal

    def update(self, target):
        self.target = target
        e = target - self.signal

        e = np.clip(e, -self.th, self.th)

        P = self.kp * e
        self.I += self.ki * e * self.dt
        # D = self.kd * (e - self.e_prev) / self.dt
        self.signal += P + self.I

    def get_stats(self):
        stats = {
            "Name": "PID Controller",
            "dt": self.dt,
            "Kp": self.kp,
            "Ki": self.ki,
            "signal": self.signal,
            "target": self.target
        }
        return stats
