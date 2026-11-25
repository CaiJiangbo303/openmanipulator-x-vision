import unittest

from recovery_manager import (
    Observation,
    Status,
    TrackingRecoveryManager
)


def obs(t=100.0, track="track_001",
        generation="gen_001", x=0.0, mask=None):

    return Observation(
        track,
        generation,
        (x, 0.0, 0.20),
        t,
        t if mask is None else mask
    )


class RecoveryTests(unittest.TestCase):

    def setUp(self):
        self.m = TrackingRecoveryManager(obs(), 0.0)

    def test_normal_tracking(self):
        self.assertEqual(
            self.m.step(0.1, tracking=obs(100.1)),
            Status.TRACKING
        )
        self.assertTrue(self.m.authorized)

    def test_short_hold(self):
        self.assertEqual(
            self.m.step(0.1),
            Status.HOLD
        )
        self.assertFalse(self.m.authorized)

        self.assertEqual(
            self.m.step(0.2, tracking=obs(100.2)),
            Status.TRACKING
        )

    def test_generation_recovery(self):
        self.m.step(0.5)

        for t in (1.0, 1.15):
            self.assertEqual(
                self.m.step(
                    t,
                    recovery=obs(100+t, generation="gen_002")
                ),
                Status.REVALIDATING
            )
            self.assertFalse(self.m.authorized)

        self.assertEqual(
            self.m.step(
                1.30,
                recovery=obs(101.30, generation="gen_002")
            ),
            Status.REAUTHORIZED
        )

        self.assertEqual(
            self.m.identity,
            ("track_001", "gen_002")
        )
        self.assertEqual(self.m.reauth_count, 1)

    def test_timeout(self):
        self.m.step(0.5)

        self.assertEqual(
            self.m.step(3.1),
            Status.EXPIRED
        )

        self.assertEqual(
            self.m.step(3.2, recovery=obs(103.2)),
            Status.EXPIRED
        )
        self.assertFalse(self.m.authorized)

    def test_wrong_track(self):
        self.m.step(0.5)

        for t in (1.0, 1.15, 1.30):
            self.assertEqual(
                self.m.step(
                    t,
                    recovery=obs(100+t, track="track_002")
                ),
                Status.RECOVERY_WAIT
            )

        self.assertFalse(self.m.authorized)

    def test_position_mismatch(self):
        self.m.step(0.5)

        self.assertEqual(
            self.m.step(1.0, recovery=obs(101, x=0.05)),
            Status.RECOVERY_WAIT
        )

    def test_duplicate_source(self):
        self.m.step(0.5)
        evidence = obs(101, generation="gen_002")

        for t in (1.0, 1.15, 1.30, 1.45):
            self.assertEqual(
                self.m.step(t, recovery=evidence),
                Status.REVALIDATING
            )

        self.assertEqual(self.m.pending_count, 1)

    def test_duplicate_mask(self):
        self.m.step(0.5)

        for t in (1.0, 1.15, 1.30):
            self.m.step(
                t,
                recovery=obs(100+t, mask=101)
            )

        self.assertEqual(self.m.pending_count, 1)
        self.assertFalse(self.m.authorized)

    def test_ambiguous_evidence_resets_validation(self):
        self.m.step(0.5)
        self.m.step(1.0, recovery=obs(101))
        self.m.step(1.15, recovery=obs(101.15))

        self.m.step(1.20, recovery=None)

        self.assertEqual(self.m.pending_count, 0)

        self.assertEqual(
            self.m.step(1.30, recovery=obs(101.30)),
            Status.REVALIDATING
        )
        self.assertFalse(self.m.authorized)

    def test_clock_rollback(self):
        self.m.step(0.5)

        self.assertEqual(
            self.m.step(0.4),
            Status.EXPIRED
        )

    def test_generation_change_requires_validation(self):
        new = obs(100.1, generation="gen_002")

        self.assertEqual(
            self.m.step(0.1, tracking=new, recovery=new),
            Status.REVALIDATING
        )
        self.assertFalse(self.m.authorized)


if __name__ == "__main__":
    unittest.main(verbosity=2)
