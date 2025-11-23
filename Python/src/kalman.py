# kalman.py
"""
Angular Kalman Filter for smoothing DOA estimates.

Class:
- AngularKalman: simple 2-state linear Kalman on [angle, angular_velocity]

Usage:
kf = AngularKalman(dt)
kf.step(measurement_angle_rad) -> returns current state [angle, omega]
"""

import numpy as np

class AngularKalman:
    def __init__(self, dt=0.02, q_angle=1e-4, q_omega=1e-4, r_meas=1e-2):
        self.dt = dt
        self.A = np.array([[1.0, dt],
                           [0.0, 1.0]])
        self.Q = np.diag([q_angle, q_omega])
        self.H = np.array([[1.0, 0.0]])
        self.R = np.array([[r_meas]])
        self.x = np.zeros((2,1))  # [angle, angular velocity]
        self.P = np.eye(2) * 1.0
        self.init = False

    def _wrap(self, ang):
        return (ang + np.pi) % (2*np.pi) - np.pi

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, z):
        if z is None or np.isnan(z):
            return
        if not self.init:
            self.x[0,0] = z
            self.x[1,0] = 0.0
            self.init = True
            return
        z = np.array([[z]])
        y = z - (self.H @ self.x)
        # wrap innovation angle
        y[0,0] = self._wrap(y[0,0])
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P

    def step(self, meas_angle_rad):
        # meas_angle_rad may be None or nan
        self.predict()
        self.update(meas_angle_rad)
        self.x[0,0] = self._wrap(self.x[0,0])
        return self.x.flatten()
