import numpy as np


class KalmanFilter():
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

    @property
    def u_dec(self):
        return np.array([
            [-self.x[1].item()],
            [-self.x[3].item()]
        ])

    def set_pos(self, x, y):
        self.x[0] = x
        self.x[2] = y

    def predict(self, decelerate=False):
        u = self.u_dec if decelerate else self.u_acc
        self.x = self.A @ self.x + self.B @ u

        # P = A * P * A' + Q
        self.P = self.A @ self.P @ self.A.T + self.Q

        return self.x[0].item(), self.x[2].item()

    def update(self, x_meas, y_meas):
        # K = P * H' * inv(H * P * H' + R)
        S = np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        K = self.P @ self.H.T @ S

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
