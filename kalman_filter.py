import numpy as np

from model import Model


class KalmanFilterBase(Model):
    def __init__(self, dt, std_acc, std_meas, decel_rate):
        super().__init__(decel_rate)
        self.dt = dt
        self.std_acc = std_acc
        self.std_meas = std_meas
        self.init_x()
        self.init_F()
        self.init_G()
        self.init_H()
        self.init_P()
        self.init_Q()
        self.init_R()

    @property
    def K(self):
        # K = P * H' * inv(H * P * H' + R)
        S = np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        return self.P @ self.H.T @ S

    @property
    def K_x(self):
        return self.K[0][0]

    def init_x(self):
        self.x = None

    def init_F(self):
        self.F = None

    def init_G(self):
        self.G = None

    def init_H(self):
        self.H = None

    def init_P(self):
        self.P = np.eye(self.F.shape[1])

    def init_Q(self):
        ...

    def init_R(self):
        self.R = np.eye(self.H.shape[0]) * self.std_meas**2

    def get_stats(self):
        return {
            "Name": "Base KF",
            "dt": self.dt,
            "std_acc": self.std_acc,
            "std_meas": self.std_meas,
            "is_decelerating": self.is_decelerating,
            "decel_rate": self.decel_rate
        }

    def print(self):
        print(self.get_stats())

    def predict(self):
        self.x = self.F @ self.x

        # P = F * P * F' + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.pos

    def update(self, meas_x, meas_y):
        self.set_last_measurement(meas_x, meas_y)

        z = np.array([
            [np.array(meas_x).item()],
            [np.array(meas_y).item()]
        ])

        K = self.K
        self.x = self.x + K @ (z - self.H @ self.x)

        # P = (I - K * H) * P * (I - K * H)' + K * R * K
        I = np.eye(self.H.shape[1])
        I_KH = I - (K @ self.H)
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T


class KalmanFilterVel(KalmanFilterBase):
    def __init__(self, dt, std_acc, std_meas, decel_rate):
        super().__init__(dt, std_acc, std_meas, decel_rate)
        self.__u = np.array([
            [0],
            [0]
        ])

    @property
    def pos(self):
        return np.array([self.x[0], self.x[2]])

    @property
    def vel(self):
        return np.array([self.x[1], self.x[3]])

    @property
    def u(self):
        if self.is_decelerating:
            return -self.vel * self.decel_rate
        return self.__u

    def set_u(self, x, y):
        self.__u[0] = x
        self.__u[1] = y

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

    def init_F(self):
        dt = self.dt
        self.F = np.array([
            [1, dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dt],
            [0, 0, 0, 1],
        ])

    def init_G(self):
        dt = self.dt
        self.G = np.array([
            [dt**2 / 2, 0],
            [dt, 0],
            [0, dt**2 / 2],
            [0, dt]
        ])

    def init_H(self):
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ])

    def init_Q(self):
        dt = self.dt
        self.Q = np.array([
            [dt**4 / 4, dt**3 / 2, 0, 0],
            [dt**3 / 2, dt**2, 0, 0],
            [0, 0, dt**4 / 4, dt**3 / 2],
            [0, 0, dt**3 / 2, dt**2],
        ]) * self.std_acc**2

    def get_stats(self):
        stats = super().get_stats()
        stats.update({
            "Name": "KF Constant Velocity",
            "Pos": [f"{self.x[0].item():.2f}", f"{self.x[2].item():.2f}"],
            "Vel": [f"{self.x[1].item():.2f}", f"{self.x[3].item():.2f}"],
            "u": [f"{self.u[0].item():.2f}", f"{self.u[1].item():.2f}"],
            "K x": f"{self.K_x.item():.2f}",
        })
        return stats

    def print(self):
        print(self.get_stats())

    def predict(self):
        self.x = self.F @ self.x + self.G @ self.u

        # P = F * P * F' + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.pos


class KalmanFilterAcc(KalmanFilterBase):
    def __init__(self, dt, std_acc, std_meas, decel_rate):
        super().__init__(dt, std_acc, std_meas, decel_rate)

    @property
    def pos(self):
        return np.array([self.x[0], self.x[3]])

    @property
    def vel(self):
        return np.array([self.x[1], self.x[4]])

    def init_x(self):
        self.x = np.array([
            [0],  # x
            [0],  # x'
            [0],  # x''
            [0],  # y
            [0],  # y'
            [0]   # y''
        ])

    def set_pos(self, x, y):
        self.x[0] = x
        self.x[3] = y

    def set_acc(self, acc_x, acc_y):
        self.x[2] = acc_x
        self.x[5] = acc_y

    def init_F(self):
        dt = self.dt
        self.F = np.array([
            [1, dt, dt**2 / 2, 0, 0, 0],
            [0, 1, dt, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, dt, dt**2 / 2],
            [0, 0, 0, 0, 1, dt],
            [0, 0, 0, 0, 0, 1]
        ])

    def init_H(self):
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ])

    def init_Q(self):
        dt = self.dt
        self.Q = np.array([
            [dt**4 / 4, dt**3 / 2, dt**2 / 2, 0, 0, 0],
            [dt**3 / 2, dt**2, dt, 0, 0, 0],
            [dt**2 / 2, dt, 1, 0, 0, 0],
            [0, 0, 0, dt**4 / 4, dt**3 / 2, dt**2 / 2],
            [0, 0, 0, dt**3 / 2, dt**2, dt],
            [0, 0, 0, dt**2 / 2, dt, 1]
        ]) * self.std_acc**2

    def get_stats(self):
        stats = super().get_stats()
        stats.update({
            "Name": "KF Constant Acceleration",
            "Pos": [f"{self.x[0].item():.2f}", f"{self.x[3].item():.2f}"],
            "Vel": [f"{self.x[1].item():.2f}", f"{self.x[4].item():.2f}"],
            "Acc": [f"{self.x[2].item():.2f}", f"{self.x[5].item():.2f}"],
            "K x": f"{self.K_x.item():.2f}",
        })
        return stats

    def predict(self):
        self.x = self.F @ self.x

        if self.is_decelerating:
            self.set_acc(*(-self.vel * self.decel_rate))

        # P = F * P * F' + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.pos
