import unittest

from object_state_manager.timestamp_guard import (
    TimestampGuard,
)


MS = 1_000_000


class TestTimestampGuard(unittest.TestCase):

    def setUp(self):
        self.guard = TimestampGuard()
        self.source = 100000 * MS
        self.steady = 10000 * MS

    def feed(self, source_ms=33, receipt_ms=33):
        self.source += source_ms * MS
        self.steady += receipt_ms * MS

        return self.guard.observe(
            self.source, self.steady
        )

    def warmup(self):
        self.feed()

        for _ in range(10):
            result = self.feed()

        self.assertTrue(result.accepted)

    def test_normal_stream(self):
        self.warmup()

        for _ in range(100):
            self.assertTrue(self.feed().accepted)

    def test_startup_recovery(self):
        self.assertFalse(self.feed().accepted)

        for _ in range(9):
            self.assertFalse(self.feed().accepted)

        self.assertTrue(self.feed().accepted)

    def test_realsense_forward_jump(self):
        self.warmup()
        result = self.feed(1658, 41)

        self.assertTrue(result.reset)
        self.assertFalse(result.accepted)

    def test_six_second_source_jump(self):
        self.warmup()
        self.assertTrue(
            self.feed(6001, 33).reset
        )

    def test_backward_stamp(self):
        self.warmup()
        self.assertTrue(
            self.feed(-1000, 33).reset
        )

    def test_duplicate_stamp(self):
        self.warmup()
        self.assertTrue(
            self.feed(0, 33).reset
        )

    def test_normal_frame_loss(self):
        self.warmup()
        self.assertTrue(
            self.feed(99, 100).accepted
        )

    def test_observation_timeout(self):
        self.warmup()
        self.assertTrue(
            self.feed(300, 300).reset
        )

    def test_watchdog_timeout(self):
        self.warmup()

        result = self.guard.poll(
            self.steady + 251 * MS
        )

        self.assertTrue(result.reset)
        self.assertFalse(result.accepted)

    def test_timeout_not_repeated(self):
        self.warmup()
        now = self.steady + 300 * MS

        self.assertTrue(self.guard.poll(now).reset)
        self.assertFalse(self.guard.poll(now).reset)

    def test_recovery_after_jump(self):
        self.warmup()
        self.feed(6000, 33)

        self.assertFalse(self.feed().accepted)

        for _ in range(9):
            self.assertFalse(self.feed().accepted)

        self.assertTrue(self.feed().accepted)
        self.assertEqual(self.guard.epoch, 1)

    def test_system_clock_forward_jump(self):
        self.warmup()

        self.guard.check_clock(
            self.source, self.steady
        )

        result = self.guard.check_clock(
            self.source + 2868 * MS,
            self.steady + 33 * MS
        )

        self.assertTrue(result.reset)
        self.assertFalse(self.guard.ready)

    def test_system_clock_backward_jump(self):
        self.warmup()

        self.guard.check_clock(
            self.source, self.steady
        )

        result = self.guard.check_clock(
            self.source - 1000 * MS,
            self.steady + 33 * MS
        )

        self.assertTrue(result.reset)

    def test_fixed_clock_offset(self):
        self.warmup()

        self.guard.check_clock(
            self.source + 6000 * MS,
            self.steady
        )

        result = self.guard.check_clock(
            self.source + 6033 * MS,
            self.steady + 33 * MS
        )

        self.assertFalse(result.reset)
        self.assertTrue(result.accepted)


if __name__ == "__main__":
    unittest.main()
