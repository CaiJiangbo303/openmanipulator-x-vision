"""Adaptive constant-velocity Kalman filter.

All internal units are SI:
position: m
velocity: m/s
process noise intensity: m^2/s^3
"""

import numpy as np


class AdaptiveCVFilter:

    def __init__(
        self,
        position,
        q_low=1e-4,
        q_high=1e-3,
        measurement_std=2e-4,
        nis_threshold=11.345,
        boost_duration=0.20,
    ):

        position = np.asarray(
            position, dtype=np.float64
        )

        if (
            position.shape != (3,)
            or not np.all(np.isfinite(position))
        ):
            raise ValueError("Invalid initial position")

        parameters = np.array([
            q_low,
            q_high,
            measurement_std,
            nis_threshold,
            boost_duration,
        ], dtype=np.float64)

        if not np.all(np.isfinite(parameters)):
            raise ValueError("Non-finite parameters")

        if not (
            0 < q_low <= q_high
            and measurement_std > 0
            and nis_threshold > 0
            and boost_duration > 0
        ):
            raise ValueError("Invalid filter parameters")

        self.q_low = float(q_low)
        self.q_high = float(q_high)
        self.nis_threshold = float(nis_threshold)
        self.boost_duration = float(boost_duration)

        self.I3 = np.eye(3)
        self.I6 = np.eye(6)

        self.H = np.zeros((3, 6))
        self.H[:, :3] = self.I3

        self.R = self.I3 * measurement_std**2

        self.x = np.zeros(6)
        self.x[:3] = position

        self.P = np.diag(
            [0.01**2]*3
            + [0.10**2]*3
        )

        self.boost_remaining = 0.0
        self.last_nis = 0.0
        self.last_q = self.q_low

        self.update_count = 0
        self.boost_count = 0

    @property
    def position(self):
        return self.x[:3].copy()

    @property
    def velocity(self):
        return self.x[3:].copy()

    @property
    def position_std(self):
        return np.sqrt(
            np.maximum(
                np.diag(self.P)[:3], 0.0
            )
        )

    def transition(self, dt):

        F = self.I6.copy()
        F[:3, 3:] = self.I3 * dt

        return F

    def process_noise(self, dt, q):

        I = self.I3

        return q * np.block([
            [I*dt**3/3, I*dt**2/2],
            [I*dt**2/2, I*dt]
        ])

    def step(self, measurement, dt):

        z = np.asarray(
            measurement, dtype=np.float64
        )

        dt = float(dt)

        if (
            z.shape != (3,)
            or not np.all(np.isfinite(z))
        ):
            raise ValueError("Invalid measurement")

        if not (
            np.isfinite(dt)
            and 0.0001 < dt <= 0.25
        ):
            raise ValueError(
                f"Invalid measurement dt: {dt}"
            )

        F = self.transition(dt)

        predicted = F @ self.x

        residual = (
            z - self.H @ predicted
        )

        base_P = F @ self.P @ F.T

        low_P = (
            base_P
            + self.process_noise(
                dt, self.q_low
            )
        )

        low_S = (
            self.H @ low_P @ self.H.T
            + self.R
        )

        nis = float(
            residual @ np.linalg.solve(
                low_S, residual
            )
        )

        boost_remaining = max(
            0.0,
            self.boost_remaining - dt
        )

        if nis > self.nis_threshold:
            boost_remaining = (
                self.boost_duration
            )

        boosted = boost_remaining > 0

        q = (
            self.q_high
            if boosted
            else self.q_low
        )

        Pp = (
            base_P
            + self.process_noise(dt, q)
        )

        S = (
            self.H @ Pp @ self.H.T
            + self.R
        )

        K = np.linalg.solve(
            S, self.H @ Pp
        ).T

        new_x = predicted + K @ residual

        A = self.I6 - K @ self.H

        # Joseph covariance update.
        new_P = (
            A @ Pp @ A.T
            + K @ self.R @ K.T
        )

        new_P = (
            new_P + new_P.T
        ) * 0.5

        if not (
            np.all(np.isfinite(new_x))
            and np.all(np.isfinite(new_P))
            and np.isfinite(nis)
        ):
            raise FloatingPointError(
                "Non-finite filter state"
            )

        self.x = new_x
        self.P = new_P

        self.last_nis = nis
        self.last_q = q
        self.boost_remaining = boost_remaining

        self.update_count += 1
        self.boost_count += int(boosted)

        return self.x.copy()
