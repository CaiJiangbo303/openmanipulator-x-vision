import unittest
from unittest.mock import patch

import numpy as np
import rclpy

from vision_msgs.msg import (
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)

from object_state_manager.state_manager_node import (
    ObjectStateManager,
)

from object_state_manager.adaptive_cv_filter import (
    AdaptiveCVFilter,
)


class TestAdaptiveIntegration(unittest.TestCase):

    def setUp(self):

        rclpy.init(args=[
            "--ros-args",
            "-p", "filter_backend:=adaptive_cv",
            "-p", "enable_timestamp_guard:=true",
            "-p", "world_frame:=camera_color_optical_frame",
        ])

        self.addCleanup(rclpy.try_shutdown)

        self.node = ObjectStateManager()
        self.addCleanup(self.node.destroy_node)

        self.stamp = 10_000_000_000

        self.origin = np.array([
            0.03, -0.02, 0.20
        ])

        # Isolate fusion behavior from time-guard startup.
        # These tests do not replace actual guard integration tests.
        for name in (
            "accept_guarded_fast",
            "accept_guarded_precise"
        ):

            p = patch.object(
                self.node,
                name,
                return_value=True
            )

            p.start()
            self.addCleanup(p.stop)

    def message(self, position, stamp):

        msg = Detection3DArray()

        msg.header.frame_id = (
            "camera_color_optical_frame"
        )

        sec, ns = divmod(
            stamp, 1_000_000_000
        )

        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = ns

        d = Detection3D()
        d.id = "track_001"

        result = ObjectHypothesisWithPose()

        result.hypothesis.class_id = "blue block"
        result.hypothesis.score = 0.95

        p = result.pose.pose.position

        p.x, p.y, p.z = map(
            float, position
        )

        d.results.append(result)
        msg.detections.append(d)

        return msg

    def fast(self, position=None):

        if position is None:
            position = self.origin

        self.stamp += 33_000_000

        self.node.fast_measurement_callback(
            self.message(position, self.stamp)
        )

        return self.node.tracks["track_001"]

    def precise(self, position, stamp=None):

        if stamp is None:
            stamp = self.stamp

        self.node.precise_measurement_callback(
            self.message(position, stamp)
        )

        return self.node.tracks["track_001"]

    def test_backend_initialization(self):

        state = self.fast()

        self.assertIsInstance(
            state.adaptive_filter,
            AdaptiveCVFilter
        )

        np.testing.assert_allclose(
            state.position,
            self.origin
        )

        np.testing.assert_allclose(
            state.velocity,
            np.zeros(3)
        )

    def test_mask_does_not_modify_filter(self):

        state = self.fast()

        before_x = (
            state.adaptive_filter.x.copy()
        )

        before_P = (
            state.adaptive_filter.P.copy()
        )

        bias = np.array([
            0.001, -0.001, 0.003
        ])

        self.precise(self.origin + bias)

        np.testing.assert_array_equal(
            state.adaptive_filter.x,
            before_x
        )

        np.testing.assert_array_equal(
            state.adaptive_filter.P,
            before_P
        )

        np.testing.assert_allclose(
            state.position,
            self.origin + bias
        )

        np.testing.assert_allclose(
            state.velocity,
            np.zeros(3)
        )

    def test_bias_not_interpreted_as_motion(self):

        self.fast()

        bias = np.array([
            0.002, 0.0, 0.005
        ])

        self.precise(self.origin + bias)

        for _ in range(20):

            state = self.fast()

            np.testing.assert_allclose(
                state.velocity,
                np.zeros(3),
                atol=1e-10
            )

            np.testing.assert_allclose(
                state.position,
                self.origin + bias
            )

    def test_bias_expiry(self):

        self.fast()

        self.precise(
            self.origin
            + np.array([0, 0, 0.004])
        )

        self.node.precise_bias_timeout = 0.05

        self.fast()
        state = self.fast()

        self.assertFalse(
            state.precise_bias_valid
        )

        np.testing.assert_allclose(
            state.position,
            self.origin
        )

        np.testing.assert_allclose(
            state.velocity,
            np.zeros(3),
            atol=1e-10
        )

    def test_unmatched_precise_rejected(self):

        state = self.fast()

        before = state.position.copy()

        self.precise(
            self.origin
            + np.array([0.01, 0, 0]),
            stamp=self.stamp - 1_000_000
        )

        np.testing.assert_array_equal(
            state.position,
            before
        )

    def test_raw_motion_matches_reference(self):

        state = self.fast()

        reference = AdaptiveCVFilter(
            self.origin
        )

        for i in range(1, 40):

            z = (
                self.origin
                + np.array([i*0.001, 0, 0])
            )

            state = self.fast(z)

            reference.step(z, 0.033)

            self.precise(
                z + np.array([
                    0, 0, 0.003
                ])
            )

            np.testing.assert_allclose(
                state.adaptive_filter.x,
                reference.x,
                atol=1e-10
            )

            np.testing.assert_allclose(
                state.adaptive_filter.P,
                reference.P,
                atol=1e-12
            )

    def test_time_reset_new_lifecycle(self):

        state = self.fast()

        old_generation = state.generation_id
        old_filter = state.adaptive_filter

        self.fast(
            self.origin
            + np.array([0.005, 0, 0])
        )

        decision = (
            self.node.timestamp_guard.invalidate(
                "synthetic integration reset"
            )
        )

        self.node.handle_time_reset(decision)

        self.assertEqual(self.node.tracks, {})
        self.assertEqual(self.node.fast_history, {})

        new_state = self.fast()

        self.assertNotEqual(
            old_generation,
            new_state.generation_id
        )

        self.assertIsNot(
            old_filter,
            new_state.adaptive_filter
        )

        np.testing.assert_allclose(
            new_state.velocity,
            np.zeros(3)
        )


class TestLegacyBackend(unittest.TestCase):

    def test_default_remains_alpha_beta(self):

        rclpy.init(args=[])

        self.addCleanup(rclpy.try_shutdown)

        node = ObjectStateManager()

        self.addCleanup(node.destroy_node)

        self.assertEqual(
            node.filter_backend,
            "alpha_beta"
        )


if __name__ == "__main__":
    unittest.main()
