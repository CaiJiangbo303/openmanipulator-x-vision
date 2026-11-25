import time
import numpy as np
import rclpy

from collections import Counter
from geometry_msgs.msg import TransformStamped
from vision_msgs.msg import Detection2DArray, Detection3DArray
from object_state_msgs.msg import ObjectStateArray
from rclpy.qos import qos_profile_sensor_data

import real_visual_tracker_v3_recovery as v3

original = v3.original


class Harness:

    state_input = v3.TrackerV3.state_input
    mask_input = v3.TrackerV3.mask_input
    raw_input = v3.TrackerV3.raw_input
    raw_guard = v3.TrackerV3.raw_guard
    get_candidates = v3.TrackerV3.get_candidates
    tick = v3.TrackerV3.tick
    hold = v3.TrackerV3.hold

    goal_response_cb = original.FastTracker.goal_response_cb

    def __init__(self, node):

        self.node = node

        self.arm = v3.Arm(
            tcp_offset=original.TCP_OFFSET
        )

        self.q = np.array([
            -0.16, 0.48, -0.16, 0.04
        ])

        tcp = self.arm.fk(self.q)[:3, 3]

        self.fixed_z = float(tcp[2])
        self.fixed_pitch = float(np.sum(self.q[1:]))

        self.joint_time = None
        self.target_world = None
        self.target_seen = None
        self.target_id = None
        self.target_class = None
        self.target_score = 0.0

        self.identity = None
        self.recovery_manager = None
        self.previous_status = None

        self.goal_pending = False
        self.command_count = 0
        self.planned_count = 0

        self.state_msg = None
        self.state_rx = 0.0
        self.mask_cache = {}
        self.raw = {}

        self.last_print = 0.0
        self.reasons = Counter()
        self.stats = Counter()

        self.tf_buffer = self
        self.virtual_tf = None

        self.pending_q = None
        self.max_joint_step = 0.0
        self.max_tcp_step = 0.0

        self.action = v3.ShadowAction(self)

        shadow_send = self.action.send_goal_async

        def capture(goal):

            q_cmd = np.asarray(
                goal.trajectory.points[0].positions,
                dtype=float
            )

            assert self.recovery_manager.authorized
            assert q_cmd.shape == (4,)
            assert np.all(np.isfinite(q_cmd))

            step = float(
                np.max(np.abs(q_cmd-self.q))
            )

            assert step <= original.MAX_JOINT_STEP + 1e-8

            assert np.all(
                q_cmd >= original.LOW + original.IK_MARGIN
            )

            assert np.all(
                q_cmd <= original.HIGH - original.IK_MARGIN
            )

            self.max_joint_step = max(
                self.max_joint_step, step
            )

            self.pending_q = q_cmd.copy()

            return shadow_send(goal)

        self.action.send_goal_async = capture

    def ros_now(self):

        return self.node.get_clock().now().nanoseconds / 1e9

    def status(self, message):

        if message.startswith("HOLD"):
            self.reasons[message] += 1

        now = time.monotonic()

        if now-self.last_print >= 0.8:
            print(message, flush=True)
            self.last_print = now

    def lookup_transform(self, target, source, stamp):

        if (
            self.virtual_tf is None
            or target != "world"
            or source != "camera_color_optical_frame"
        ):
            raise RuntimeError("No virtual transform")

        return self.virtual_tf

    def initialize_tf(self):

        if self.virtual_tf is not None:
            return

        if self.state_msg is None:
            return

        if self.raw_guard() != "PASS":
            return

        if self.state_msg.header.frame_id != \
                "camera_color_optical_frame":
            return

        targets = [
            o for o in self.state_msg.states
            if o.class_id in v3.TRACK_CLASSES
        ]

        if len(targets) != 1:
            return

        o = targets[0]

        if (
            not o.valid_3d
            or o.stale
            or o.missed != 0
            or o.confidence < v3.ACQUIRE_CONF
        ):
            return

        pc = v3.xyz(o.position)

        if not np.all(np.isfinite(pc)):
            return

        tcp = self.arm.fk(self.q)[:3, 3]

        target = tcp + np.array([
            -0.025, 0.0, 0.0
        ])

        translation = target - pc

        tf = TransformStamped()

        tf.header.frame_id = "world"
        tf.child_frame_id = "camera_color_optical_frame"

        tf.transform.rotation.w = 1.0

        tf.transform.translation.x = float(translation[0])
        tf.transform.translation.y = float(translation[1])
        tf.transform.translation.z = float(translation[2])

        self.virtual_tf = tf

        print("VIRTUAL TF INITIALIZED")
        print("Initial virtual TCP:", tcp)
        print("Initial virtual target:", target)

    def cycle(self):

        self.initialize_tf()

        self.joint_time = time.monotonic()

        self.pending_q = None

        self.tick()

        if self.pending_q is not None:

            before = self.arm.fk(self.q)[:3, 3]

            self.q = self.pending_q.copy()

            after = self.arm.fk(self.q)[:3, 3]

            self.max_tcp_step = max(
                self.max_tcp_step,
                float(np.linalg.norm(after-before))
            )

        if self.recovery_manager:
            self.stats[
                self.recovery_manager.status.value
            ] += 1


rclpy.init()

node = rclpy.create_node("v3_vision_integration_test")
h = Harness(node)

subscriptions = [
    node.create_subscription(
        ObjectStateArray,
        "/bench/timestamp_guard/states",
        h.state_input,
        qos_profile_sensor_data
    ),
    node.create_subscription(
        Detection3DArray,
        "/object_localizer/detections_3d_mask",
        h.mask_input,
        qos_profile_sensor_data
    ),
    node.create_subscription(
        Detection2DArray,
        "/object_detector/detections",
        lambda m: h.raw_input("DET", m),
        qos_profile_sensor_data
    ),
    node.create_subscription(
        Detection2DArray,
        "/object_tracker/detections",
        lambda m: h.raw_input("TRACK", m),
        qos_profile_sensor_data
    )
]

print("SHADOW ONLY - NO HARDWARE COMMANDS")
print("Using virtual joints and virtual TF")
print("Waiting for initial target...")

deadline = time.monotonic() + 30
start = None
next_tick = time.monotonic()
moved = False

try:

    while rclpy.ok():

        rclpy.spin_once(node, timeout_sec=0.01)

        now = time.monotonic()

        if start is None and now > deadline:
            print("INITIAL ACQUISITION TIMEOUT")
            break

        if now >= next_tick:

            next_tick = now + 0.1

            h.cycle()

            if start is None and h.recovery_manager:
                start = now

                print(
                    "\a\n>>> [0s] 保持方块静止 <<<",
                    flush=True
                )

        if start is not None:

            elapsed = now-start

            if elapsed >= 5 and not moved:

                print(
                    "\a\n>>> [5s] 缓慢小幅移动方块，不要遮挡 <<<",
                    flush=True
                )

                moved = True

            if elapsed >= 15:
                break

finally:

    print("\n===== INTEGRATION SUMMARY =====")

    print("Plans:", h.planned_count)
    print("Stats:", dict(h.stats))
    print("Hold reasons:", dict(h.reasons))

    if h.recovery_manager:

        print("Identity:", h.recovery_manager.identity)
        print("Final status:", h.recovery_manager.status.value)
        print("Authorized:", h.recovery_manager.authorized)
        print("Reauthorizations:", h.recovery_manager.reauth_count)

    print("Max joint step rad:", round(h.max_joint_step, 6))
    print("Max TCP step mm:", round(h.max_tcp_step*1000, 3))

    print("Final virtual joints:", h.q)
    print("Final virtual TCP:", h.arm.fk(h.q)[:3, 3])

    print("NO HARDWARE MOTION")

    node.destroy_node()

    if rclpy.ok():
        rclpy.shutdown()
