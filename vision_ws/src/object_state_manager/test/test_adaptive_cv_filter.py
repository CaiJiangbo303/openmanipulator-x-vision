import unittest
import numpy as np

from object_state_manager.adaptive_cv_filter import (
    AdaptiveCVFilter
)


class TestAdaptiveCVFilter(unittest.TestCase):

    def setUp(self):

        self.origin = np.array([
            0.02, -0.03, 0.20
        ])

        self.f = AdaptiveCVFilter(
            self.origin
        )

    def settle(self, n=100):

        for _ in range(n):
            self.f.step(
                self.origin, 0.033
            )

    def test_initialization(self):

        np.testing.assert_allclose(
            self.f.position,
            self.origin
        )

        np.testing.assert_array_equal(
            self.f.velocity,
            np.zeros(3)
        )

    def test_si_units(self):

        self.assertAlmostEqual(
            self.f.q_low, 1e-4
        )

        self.assertAlmostEqual(
            self.f.q_high, 1e-3
        )

        np.testing.assert_allclose(
            np.diag(self.f.R),
            [4e-8]*3
        )

    def test_stationary(self):

        self.settle()

        np.testing.assert_allclose(
            self.f.position,
            self.origin
        )

        np.testing.assert_allclose(
            self.f.velocity,
            np.zeros(3),
            atol=1e-10
        )

        self.assertEqual(
            self.f.last_q,
            self.f.q_low
        )

    def test_constant_velocity(self):

        velocity = np.array([
            0.05, 0.01, 0.0
        ])

        for i in range(1, 201):

            z = (
                self.origin
                + velocity*i*0.033
            )

            self.f.step(z, 0.033)

        np.testing.assert_allclose(
            self.f.velocity,
            velocity,
            atol=1e-5
        )

        np.testing.assert_allclose(
            self.f.position,
            z,
            atol=1e-5
        )

    def test_adaptive_boost(self):

        self.settle()

        self.f.step(
            self.origin
            + np.array([0.005, 0, 0]),
            0.033
        )

        self.assertGreater(
            self.f.last_nis,
            self.f.nis_threshold
        )

        self.assertEqual(
            self.f.last_q,
            self.f.q_high
        )

    def test_return_to_low_q(self):

        self.settle()

        target = (
            self.origin
            + np.array([0.005, 0, 0])
        )

        self.f.step(target, 0.033)

        for _ in range(100):
            self.f.step(target, 0.033)

        self.assertEqual(
            self.f.last_q,
            self.f.q_low
        )

        self.assertEqual(
            self.f.boost_remaining,
            0.0
        )

    def test_covariance_stability(self):

        rng = np.random.default_rng(42)

        for _ in range(1000):

            dt = rng.uniform(
                0.03, 0.08
            )

            z = (
                self.origin
                + rng.normal(
                    0, 0.0002, 3
                )
            )

            self.f.step(z, dt)

            np.testing.assert_allclose(
                self.f.P,
                self.f.P.T,
                atol=1e-12
            )

            self.assertGreaterEqual(
                np.linalg.eigvalsh(
                    self.f.P
                ).min(),
                -1e-12
            )

            self.assertTrue(
                np.all(
                    np.isfinite(
                        self.f.x
                    )
                )
            )

    def test_invalid_dt_preserves_state(self):

        before = self.f.x.copy()
        covariance = self.f.P.copy()

        for dt in (
            -0.1, 0, 0.0001, 0.5,
            float("nan")
        ):

            with self.assertRaises(ValueError):
                self.f.step(
                    self.origin, dt
                )

        np.testing.assert_array_equal(
            self.f.x, before
        )

        np.testing.assert_array_equal(
            self.f.P, covariance
        )

        self.assertEqual(
            self.f.update_count, 0
        )

    def test_invalid_measurement(self):

        before = self.f.x.copy()

        with self.assertRaises(ValueError):

            self.f.step(
                [float("nan"), 0, 0.2],
                0.033
            )

        np.testing.assert_array_equal(
            self.f.x, before
        )

    def test_independent_lifecycles(self):

        self.settle()

        self.f.step(
            self.origin
            + np.array([0.01, 0, 0]),
            0.033
        )

        new_filter = AdaptiveCVFilter(
            self.origin
        )

        np.testing.assert_array_equal(
            new_filter.velocity,
            np.zeros(3)
        )

        self.assertEqual(
            new_filter.boost_remaining,
            0.0
        )

        self.assertEqual(
            new_filter.update_count,
            0
        )


if __name__ == "__main__":
    unittest.main()
