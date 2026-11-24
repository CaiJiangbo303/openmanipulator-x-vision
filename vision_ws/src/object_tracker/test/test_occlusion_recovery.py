import unittest
from types import SimpleNamespace
from unittest.mock import patch

import rclpy

from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)

import object_tracker.tracker_node as tracker_module

from object_tracker.tracker_node import (
    ObjectTrackerNode,
)


class CapturePublisher:

    def __init__(self):
        self.last = None

    def publish(self, msg):
        self.last = msg


class TestOcclusionRecovery(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.try_shutdown()

    def setUp(self):

        self.now_ns = 10_000_000_000

        self.clock_patch = patch.object(
            tracker_module,
            "time",
            SimpleNamespace(
                monotonic_ns=lambda: self.now_ns
            )
        )

        self.clock_patch.start()
        self.addCleanup(self.clock_patch.stop)

        self.node = ObjectTrackerNode()
        self.addCleanup(self.node.destroy_node)

        self.assertFalse(
            self.node.enable_occlusion_recovery
        )

        self.node.enable_occlusion_recovery = True

        self.publisher = CapturePublisher()
        self.original_pub = self.node.pub
        self.node.pub = self.publisher

        self.addCleanup(
            setattr,
            self.node,
            "pub",
            self.original_pub
        )

    def make_detection(
        self,
        x=200,
        y=200,
        w=100,
        h=100,
        cls="blue block"
    ):
        d = Detection2D()

        d.bbox.center.position.x = float(x)
        d.bbox.center.position.y = float(y)
        d.bbox.size_x = float(w)
        d.bbox.size_y = float(h)

        result = ObjectHypothesisWithPose()
        result.hypothesis.class_id = cls
        result.hypothesis.score = 0.95

        d.results.append(result)
        d.id = "det_00001"

        return d

    def frame(self, detections=None, dt=0.033):

        self.now_ns += int(dt * 1e9)

        msg = Detection2DArray()

        msg.header.frame_id = (
            "camera_color_optical_frame"
        )

        sec, ns = divmod(
            self.now_ns, 1_000_000_000
        )

        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = ns

        msg.detections = (
            detections
            if detections is not None
            else []
        )

        self.node.callback(msg)

        return [
            d.id
            for d in self.publisher.last.detections
        ]

    def establish(self):

        ids = []

        for _ in range(3):
            ids = self.frame([
                self.make_detection()
            ])

        self.assertEqual(ids, ["track_001"])

        return ids[0]

    def occlude(self, seconds):

        for _ in range(round(seconds / 0.033)):
            self.assertEqual(
                self.frame([]), []
            )

    def seed_track(self, x, lost=False):

        bbox = self.node.bbox_xyxy(
            self.make_detection(x=x)
        )

        track_id = self.node.create_track(
            "blue block", bbox
        )

        if lost:
            state = self.node.tracks[track_id]
            state.missed = 10

            state.last_seen_ns = (
                self.now_ns - 1_000_000_000
            )

        return track_id

    def test_normal_tracking(self):

        identity = self.establish()

        for i in range(30):
            self.assertEqual(
                self.frame([
                    self.make_detection(x=200+i)
                ]),
                [identity]
            )

    def test_two_second_occlusion(self):

        identity = self.establish()
        self.occlude(2.0)

        self.assertEqual(
            self.frame([self.make_detection()]),
            [identity]
        )

    def test_displaced_recovery(self):

        identity = self.establish()
        self.occlude(1.0)

        self.assertEqual(
            self.frame([
                self.make_detection(x=240)
            ]),
            [identity]
        )

    def test_expired_identity(self):

        identity = self.establish()
        self.occlude(3.2)

        self.assertNotIn(
            1, self.node.tracks
        )

        ids = []

        for _ in range(3):
            ids = self.frame([
                self.make_detection()
            ])

        self.assertEqual(len(ids), 1)
        self.assertNotEqual(ids[0], identity)

    def test_active_track_priority(self):

        lost = self.seed_track(
            x=200, lost=True
        )

        active = self.seed_track(
            x=240, lost=False
        )

        ids = self.frame([
            self.make_detection(x=200)
        ])

        self.assertEqual(
            ids, [f"track_{active:03d}"]
        )

        self.assertNotEqual(active, lost)

    def test_ambiguous_lost_tracks(self):

        self.seed_track(x=190, lost=True)
        self.seed_track(x=210, lost=True)

        ids = self.frame([
            self.make_detection(x=200)
        ])

        self.assertEqual(ids, [])

    def test_ambiguous_detections(self):

        self.seed_track(x=200, lost=True)

        ids = self.frame([
            self.make_detection(x=180),
            self.make_detection(x=220)
        ])

        self.assertEqual(ids, [])

    def test_no_phantom_detection(self):

        self.establish()
        self.occlude(2.0)

        self.assertIn(1, self.node.tracks)

        self.assertEqual(
            self.publisher.last.detections,
            []
        )

    def test_class_mismatch(self):

        self.establish()
        self.occlude(1.0)

        ids = self.frame([
            self.make_detection(cls="cup")
        ])

        self.assertEqual(ids, [])

    def test_legacy_compatibility(self):

        self.node.enable_occlusion_recovery = False

        self.establish()

        for _ in range(
            self.node.max_missed + 1
        ):
            self.frame([])

        self.assertNotIn(
            1, self.node.tracks
        )

    def test_expiry_without_intermediate_frames(self):

        self.establish()

        ids = self.frame(
            [self.make_detection()],
            dt=3.1
        )

        self.assertEqual(ids, [])
        self.assertNotIn(1, self.node.tracks)

    def test_insufficient_overlap(self):

        self.establish()
        self.occlude(1.0)

        ids = self.frame([
            self.make_detection(x=450)
        ])

        self.assertEqual(ids, [])



    def establish_two(self):

        for _ in range(3):
            ids = self.frame([
                self.make_detection(x=200),
                self.make_detection(x=450)
            ])

        self.assertEqual(len(ids), 2)
        self.assertEqual(len(set(ids)), 2)

        return ids

    def test_dual_target_short_stream_gap(self):

        before = self.establish_two()

        after = self.frame([
            self.make_detection(x=200),
            self.make_detection(x=450)
        ], dt=0.325)

        self.assertEqual(after, before)

    def test_dual_target_long_stream_gap(self):

        self.establish_two()

        after = self.frame([
            self.make_detection(x=200),
            self.make_detection(x=450)
        ], dt=0.60)

        self.assertEqual(after, [])

    def test_observed_missing_still_triggers_lost(self):

        self.establish_two()

        self.frame([], dt=0.12)
        self.frame([], dt=0.12)

        after = self.frame([
            self.make_detection(x=200),
            self.make_detection(x=450)
        ], dt=0.033)

        self.assertEqual(after, [])



    def test_single_side_occlusion(self):

        left, right = self.establish_two()

        for _ in range(60):
            self.assertEqual(
                self.frame([
                    self.make_detection(x=450)
                ]),
                [right]
            )

        for _ in range(3):
            ids = self.frame([
                self.make_detection(x=200),
                self.make_detection(x=450)
            ])

        self.assertEqual(len(ids), 2)
        self.assertNotEqual(ids[0], left)
        self.assertEqual(ids[1], right)

    def test_intermittent_right_detection_loss(self):

        left, right = self.establish_two()

        for i in range(60):

            if i % 12 in (0, 1):
                self.assertEqual(
                    self.frame([]), []
                )
            else:
                self.assertEqual(
                    self.frame([
                        self.make_detection(x=450)
                    ]),
                    [right]
                )

        for _ in range(3):
            ids = self.frame([
                self.make_detection(x=200),
                self.make_detection(x=450)
            ])

        self.assertEqual(len(ids), 2)
        self.assertEqual(ids[1], right)
        self.assertNotEqual(ids[0], left)

    def test_both_targets_lost_conservatively(self):

        old = self.establish_two()

        self.occlude(1.0)

        for _ in range(3):
            ids = self.frame([
                self.make_detection(x=200),
                self.make_detection(x=450)
            ])

        self.assertEqual(len(ids), 2)
        self.assertTrue(
            set(ids).isdisjoint(old)
        )


if __name__ == "__main__":
    unittest.main()
