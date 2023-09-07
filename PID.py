class PID:
    def __init__(self, dt):
        self.dt = dt
        self.kp = 0.1
        # self.kd = 0

        self.set(0)

    def set(self, signal):
        self.signal = signal
        self.target = signal

    def get(self):
        return self.signal

    def update(self, target):
        self.target = target
        e = target - self.signal

        self.signal += e * self.kp

    def get_stats(self):
        stats = {
            "Name": "PID Controller",
            "dt": self.dt,
            "kp": self.kp,
            "signal": self.signal,
            "target": self.target
        }
        return stats
