import numpy as np


class KalmanFilterBase():

    @property
    def pos(self):
        ...

    def set_pos(self, x, y):
        ...

    def print(self):
        ...

    def predict(self):
        self.x = self.A @ self.x

        # P = A * P * A' + Q
        self.P = self.A @ self.P @ self.A.T + self.Q

        return self.pos

    def update(self, x_meas, y_meas):
        # K = P * H' * inv(H * P * H' + R)
        S = np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        K = self.P @ self.H.T @ S
        self.K = K

        z = np.array([
            [x_meas],
            [y_meas]
        ])
        self.x = self.x + K @ (z - self.H @ self.x)

        # P = (I - K * H) * P * (I - K * H)' + K * R * K
        I = np.eye(self.H.shape[1])
        I_KH = I - (K @ self.H)
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        return K


class KalmanFilter(KalmanFilterBase):
    DECELERATION_RATE = 0.3

    def __init__(self, dt, acc_x, acc_y, std_acc, std_measurement):
        self.dt = dt

        self.x = np.array([
            [0],  # x
            [0],  # x'
            [acc_x],  # x''
            [0],  # y
            [0],  # y'
            [acc_y]  # y''
        ])

        self.A = np.array([
            [1, dt, dt**2 / 2, 0, 0, 0],
            [0, 1, dt, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, dt, dt**2 / 2],
            [0, 0, 0, 0, 1, dt],
            [0, 0, 0, 0, 0, 1]
        ])

        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ])

        self.Q = np.array([
            [dt**4 / 4, dt**3 / 2, dt**2 / 2, 0, 0, 0],
            [dt**3 / 2, dt**2, dt, 0, 0, 0],
            [dt**2 / 2, dt, 1, 0, 0, 0],
            [0, 0, 0, dt**4 / 4, dt**3 / 2, dt**2 / 2],
            [0, 0, 0, dt**3 / 2, dt**2, dt],
            [0, 0, 0, dt**2 / 2, dt, 1]
        ]) * std_acc**2

        self.P = np.eye(self.A.shape[1])

        self.R = np.eye(self.H.shape[0]) * std_measurement**2

        self.K = self.x

    @property
    def pos(self):
        return np.array([self.x[0], self.x[3]])

    @property
    def vel(self):
        return np.array([self.x[1], self.x[4]])

    def set_pos(self, x, y):
        self.x[0] = x
        self.x[3] = y

    def set_acc(self, acc_x, acc_y):
        self.x[2] = acc_x
        self.x[5] = acc_y

    def print(self):
        print((f"Pos x: {self.x[0].item():.2f}, "
               f"Vel x: {self.x[1].item():.2f}, "
               f"Acc x: {self.x[2].item():.2f} "
               f"P x: {self.P[0][0].item():.2f}, "
               f"K x: {self.K[0][0].item():.2f}"
               ))

    def predict(self):
        self.x = self.A @ self.x

        # if decelerate:
        #     print("Decelerating")
        #     self.set_acc(*(-self.vel * KalmanFilter.DECELERATION_RATE))

        # P = A * P * A' + Q
        self.P = self.A @ self.P @ self.A.T + self.Q

        return self.pos


class KalmanFilterControl(KalmanFilterBase):
    DECELERATION_RATE = 0.1

    def __init__(self, dt, acc_x, acc_y, std_acc, std_measurement):
        self.dt = dt
        self.acc_x = acc_x
        self.acc_y = acc_y

        self.x = np.array([
            [0],  # x
            [0],  # x'
            [0],  # y
            [0],  # y'
        ])

        self.u_acc = np.array([
            [acc_x],
            [acc_y]
        ])

        self.A = np.array([
            [1, dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dt],
            [0, 0, 0, 1],
        ])

        self.B = np.array([
            [dt**2 / 2, 0],
            [dt, 0],
            [0, dt**2 / 2],
            [0, dt]
        ])

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ])

        self.Q = np.array([
            [dt**4 / 4, dt**3 / 2, 0, 0],
            [dt**3 / 2, dt**2, 0, 0],
            [0, 0, dt**4 / 4, dt**3 / 2],
            [0, 0, dt**3 / 2, dt**2],
        ]) * std_acc**2

        self.P = np.eye(self.A.shape[1])

        self.R = np.eye(self.H.shape[0]) * std_measurement**2

        self.K = self.x

    @property
    def u_dec(self):
        return np.array([
            [-self.x[1].item()],
            [-self.x[3].item()]
        ]) * KalmanFilterControl.DECELERATION_RATE

    @property
    def pos(self):
        return np.array([self.x[0], self.x[3]])

    def set_pos(self, x, y):
        self.x[0] = x
        self.x[2] = y

    def print(self):
        print((f"Pos x: {self.x[0].item():.2f}, "
               f"Vel x: {self.x[1].item():.2f}, "
               f"Acc x: {self.u_acc[0].item():.2f} "
               f"P x: {self.P[0][0].item():.2f}, "
               f"K x: {self.K[0][0].item():.2f}"
               ))

    def predict(self, decelerate=False):
        u = self.u_dec if decelerate else self.u_acc
        if decelerate:
            print("Decelerating")
        # u = self.u_acc
        self.x = self.A @ self.x + self.B @ u

        # P = A * P * A' + Q
        self.P = self.A @ self.P @ self.A.T + self.Q

        return self.pos
