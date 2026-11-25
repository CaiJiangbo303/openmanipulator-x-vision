import time
import os
import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from vision_msgs.msg import Detection3DArray, Detection2DArray
from rclpy.qos import qos_profile_sensor_data
from semantic_tracking.recovery_manager import (
    Observation,
    Status,
    TrackingRecoveryManager
)
from control_msgs.msg import DynamicJointState
from object_state_msgs.msg import ObjectStateArray
from tf2_ros import Buffer, TransformListener

import real_visual_tracker_final as original

from manipulator_grasp.kinematics import Arm


# ==================================================
# CONFIG
# ==================================================

TRACK_CLASS = "blue block"
TRACK_CLASSES = {"blue block", "plastic cube"}

# Retain original 10Hz control architecture.
original.CONTROL_HZ = 10.0

# Conservative planning limits.
original.MAX_CART_STEP = 0.010
original.MAX_JOINT_STEP = 0.020

STATE_MAX_AGE = 0.20
MASK_MAX_AGE = 0.25

ACQUIRE_CONF = 0.50
TRACK_CONF = 0.40
MASK_CONF = 0.40

NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4"
]


def stamp(s):
    return s.sec + s.nanosec / 1e9


def xyz(p):
    return np.array(
        [p.x, p.y, p.z],
        dtype=float
    )


# ==================================================
# SHADOW ACTION
# No real ROS action client.
# ==================================================

class ShadowFuture:

    def add_done_callback(self, callback):
        callback(self)

    def result(self):
        return self

    accepted = True


class ShadowAction:

    def __init__(self, node):
        self.node = node

    def server_is_ready(self):
        return True

    def send_goal_async(self, goal):

        q_cmd = np.array(
            goal.trajectory.points[0].positions
        )

        node = self.node

        node.planned_count += 1

        node.status(
            "ROLLING PLAN | "
            f"id={node.target_id} "
            f"count={node.planned_count} "
            f"q={np.round(q_cmd, 4)}"
        )

        return ShadowFuture()


# ==================================================
# V3 TRACKER
# Reuse original control_cb directly.
# ==================================================

class TrackerV3(original.FastTracker):

    def __init__(self):

        # Do not initialize original motion client.
        Node.__init__(self, "real_visual_tracker_v3_recovery")

        self.arm = Arm(
            tcp_offset=original.TCP_OFFSET
        )

        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        self.action = ShadowAction(self)

        self.q = None
        self.joint_time = None

        self.fixed_z = None
        self.fixed_pitch = None

        self.target_world = None
        self.target_seen = None
        self.target_id = None
        self.target_class = None
        self.target_score = 0.0

        self.identity = None
        self.recovery_manager = None
        self.raw = {}
        self.previous_status = None

        self.goal_pending = False
        self.command_count = 0
        self.planned_count = 0

        self.last_print = 0.0
        self.last_hw_print = 0.0

        self.state_msg = None
        self.state_rx = 0.0

        self.mask_cache = {}

        self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_input,
            10
        )

        self.create_subscription(
            ObjectStateArray,
            "/bench/timestamp_guard/states",
            self.state_input,
            10
        )

        self.create_subscription(
            Detection3DArray,
            "/bench/real_self_filter/detections_3d",
            self.mask_input,
            10
        )

        self.create_subscription(
            DynamicJointState,
            "/dynamic_joint_states",
            self.hardware_input,
            10
        )

        self.create_subscription(
            Detection2DArray,
            "/object_detector/detections",
            lambda msg: self.raw_input("DET", msg),
            qos_profile_sensor_data
        )

        self.create_subscription(
            Detection2DArray,
            "/object_tracker/detections",
            lambda msg: self.raw_input("TRACK", msg),
            qos_profile_sensor_data
        )

        self.create_timer(
            1.0 / original.CONTROL_HZ,
            self.tick
        )

        print("====================================")
        print("REAL VISUAL TRACKER V3 RECOVERY")
        print("Control: original FastTracker")
        print("Vision: State Manager V2")
        print("Self Filter: asynchronous evidence")
        print("Rate: 10 Hz")
        print("Mode: SHADOW / NO MOTION")
        print("Session limit: NONE")
        print("====================================")

    def ros_now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def joint_input(self, msg):

        age = self.ros_now() - stamp(
            msg.header.stamp
        )

        if not 0 <= age <= 0.10:
            return

        values = dict(
            zip(msg.name, msg.position)
        )

        if not all(n in values for n in NAMES):
            return

        q = np.array(
            [values[n] for n in NAMES]
        )

        if not np.all(np.isfinite(q)):
            return

        self.joint_time = time.monotonic()

        # Original joint callback:
        # keep original TCP / Z / pitch calculation.
        original.FastTracker.joint_cb(self, msg)

    def state_input(self, msg):

        self.state_msg = msg
        self.state_rx = time.monotonic()

    def mask_input(self, msg):

        if msg.header.frame_id != \
                "camera_color_optical_frame":
            return

        now = self.ros_now()
        source = stamp(msg.header.stamp)

        if not 0 <= now-source <= MASK_MAX_AGE:
            return

        for d in msg.detections:

            if not d.results:
                continue

            r = d.results[0]

            cls = r.hypothesis.class_id
            score = float(r.hypothesis.score)

            if cls not in TRACK_CLASSES:
                continue

            if not np.isfinite(score):
                continue

            if score < MASK_CONF:
                continue

            p = xyz(r.pose.pose.position)

            if not np.all(np.isfinite(p)):
                continue

            self.mask_cache[d.id] = (
                source,
                p,
                score
            )

        self.mask_cache = {
            k: v
            for k, v in self.mask_cache.items()
            if 0 <= now-v[0] <= MASK_MAX_AGE
        }

    def hardware_input(self, msg):

        now = time.monotonic()

        if now-self.last_hw_print < 2.0:
            return

        self.last_hw_print = now

        for name, item in zip(
            msg.joint_names,
            msg.interface_values
        ):

            if name != "dxl2":
                continue

            values = dict(zip(
                item.interface_names,
                item.values
            ))

            print(
                "ID12 MONITOR |",
                "torque=", values.get("Torque Enable"),
                "current_raw=", values.get("Present Current"),
                "voltage=", values.get("Present Input Voltage")
            )

    def raw_input(self, name, msg):

        targets = [
            d for d in msg.detections
            if d.results
            and d.results[0].hypothesis.class_id
            in TRACK_CLASSES
        ]

        self.raw[name] = (
            time.monotonic(),
            msg,
            targets
        )

        if len(targets) > 1 and self.recovery_manager:
            self.recovery_manager.expire()

    def raw_guard(self):

        now = self.ros_now()

        for name in ("DET", "TRACK"):

            data = self.raw.get(name)

            if data is None:
                return name + " NO DATA"

            received, msg, targets = data

            if not 0 <= time.monotonic()-received <= 0.20:
                return name + " RECEIPT STALE"

            if not 0 <= now-stamp(msg.header.stamp) <= 0.20:
                return name + " TIMESTAMP STALE"

            if len(targets) != 1:
                return name + " MISSING OR AMBIGUOUS"

        return "PASS"

    def get_candidates(self):

        if self.state_msg is None:
            return [], "WAIT STATE"

        now = self.ros_now()

        if time.monotonic()-self.state_rx > STATE_MAX_AGE:
            return [], "STATE RECEIPT STALE"

        msg = self.state_msg

        if not 0 <= now-stamp(msg.header.stamp) <= STATE_MAX_AGE:
            return [], "STATE TIMESTAMP STALE"

        compatible = [
            o for o in msg.states
            if o.class_id in TRACK_CLASSES
        ]

        if len(compatible) > 1:
            if self.recovery_manager:
                self.recovery_manager.expire()
            return [], "STATE AMBIGUOUS"

        guard = self.raw_guard()

        if guard != "PASS":
            return [], guard

        frame = "camera_color_optical_frame"

        if msg.header.frame_id != frame:
            return [], "FRAME MISMATCH"

        try:
            tf = self.tf_buffer.lookup_transform(
                "world",
                frame,
                Time()
            )
        except Exception:
            return [], "WAIT TF"

        r = tf.transform.rotation
        t = tf.transform.translation

        R = original.quat_to_rot(
            [r.x, r.y, r.z, r.w]
        )

        tcp = self.arm.fk(self.q)[:3, 3]

        candidates = []
        rejected = []

        for o in msg.states:

            if o.class_id not in TRACK_CLASSES:
                continue

            ident = (
                o.track_id,
                o.generation_id
            )

            if not all(ident):
                continue

            if sum(
                x.track_id == o.track_id
                for x in msg.states
            ) != 1:
                rejected.append("AMBIGUOUS ID")
                continue

            if not o.valid_3d or o.stale \
                    or o.state != 1 or o.missed != 0:

                rejected.append("TARGET INVALID")
                continue

            source = stamp(o.source_stamp)
            source_age = now-source

            if self.raw["TRACK"][2][0].id != o.track_id:
                rejected.append("TRACK ID MISMATCH")
                continue

            if any(
                abs(stamp(self.raw[name][1].header.stamp)-source)
                > 0.15
                for name in ("DET", "TRACK")
            ):
                rejected.append("UPSTREAM SOURCE MISMATCH")
                continue

            if not 0 <= source_age <= STATE_MAX_AGE:
                rejected.append("SOURCE STALE")
                continue

            if not 0 <= o.measurement_age_ms <= 150:
                rejected.append("MEASUREMENT STALE")
                continue

            threshold = TRACK_CONF

            if (
                not np.isfinite(o.confidence)
                or not o.confidence >= threshold
            ):
                rejected.append("LOW CONFIDENCE")
                continue

            if not np.linalg.norm(
                xyz(o.position_std)
            ) <= 0.03:

                rejected.append("UNCERTAINTY")
                continue

            pc = xyz(o.position)

            if not np.all(np.isfinite(pc)):
                continue

            if not 0.12 <= pc[2] <= 0.48:
                rejected.append("DEPTH LIMIT")
                continue

            # Recent Self Filter evidence,
            # not exact same-frame synchronization.
            evidence = self.mask_cache.get(
                o.track_id
            )

            if evidence is None:
                rejected.append("WAIT MASK EVIDENCE")
                continue

            mask_stamp, mask_pos, mask_score = evidence

            if not 0 <= now-mask_stamp <= MASK_MAX_AGE:
                rejected.append("MASK STALE")
                continue

            if abs(mask_stamp-source) > 0.15:
                rejected.append("MASK SOURCE MISMATCH")
                continue

            if np.linalg.norm(pc-mask_pos) > 0.025:
                rejected.append("MASK POSITION MISMATCH")
                continue

            pw = R @ pc + xyz(t)

            if not np.all(np.isfinite(pw)):
                continue

            if not (
                original.SAFE_X[0]
                <= pw[0]
                <= original.SAFE_X[1]
                and
                original.SAFE_Y[0]
                <= pw[1]
                <= original.SAFE_Y[1]
            ):
                rejected.append("WORKSPACE LIMIT")
                continue

            distance = np.linalg.norm(
                pw[:2]-tcp[:2]
            )

            if distance > 0.18:
                rejected.append("TARGET TOO FAR")
                continue

            candidates.append((
                float(distance),
                -float(o.confidence),
                ident,
                pw,
                source_age,
                source,
                mask_stamp
            ))

        return candidates, (
            ", ".join(sorted(set(rejected)))
            or "NO TARGET"
        )

    def hold(self, reason):

        self.target_world = None
        self.target_seen = None
        self.status("HOLD: " + reason)

    def tick(self):

        manager = self.recovery_manager

        if manager and manager.status == Status.EXPIRED:
            self.hold("RECOVERY EXPIRED - SESSION LOCKED")
            return

        candidates = []
        reason = "WAIT JOINT"

        if self.q is not None and self.joint_time is not None:
            if 0 <= time.monotonic()-self.joint_time <= 0.10:
                candidates, reason = self.get_candidates()
            else:
                reason = "JOINT STALE"

        observation = None
        confidence = 0.0
        candidate = None

        if len(candidates) == 1:

            candidate = candidates[0]

            _, score, ident, pw, age, source, mask_stamp = candidate

            confidence = -score

            observation = Observation(
                track_id=ident[0],
                generation_id=ident[1],
                position=tuple(float(v) for v in pw),
                source_stamp=source,
                mask_stamp=mask_stamp
            )

            if not observation.valid():
                observation = None
                reason = "INVALID OBSERVATION"

            elif manager is not None:

                jump = np.linalg.norm(
                    pw[:2] -
                    np.array(manager.observation.position[:2])
                )

                if jump > original.MAX_TARGET_JUMP:
                    observation = None
                    reason = "TARGET JUMP"

        elif len(candidates) > 1:
            reason = "AMBIGUOUS CANDIDATES"

            if manager:
                manager.expire()

        acquisition = (
            observation
            if observation is not None
            and confidence >= ACQUIRE_CONF
            else None
        )

        now = time.monotonic()

        if manager is None:

            if acquisition is None:
                self.hold("WAIT INITIAL ACQUISITION")
                return

            manager = TrackingRecoveryManager(
                acquisition, now
            )

            self.recovery_manager = manager

            print("INITIAL LOCKED:", manager.identity)

        else:

            manager.step(
                now,
                tracking=observation,
                recovery=acquisition
            )

        self.identity = manager.identity

        if manager.status != self.previous_status:

            print(
                "RECOVERY STATE:",
                manager.status.value,
                "| identity:", manager.identity,
                "| shadow_authorized:", manager.authorized,
                flush=True
            )

            self.previous_status = manager.status

        if (
            not manager.authorized
            or observation is None
            or observation.identity != manager.identity
        ):
            self.hold(manager.status.value + " | " + reason)
            return

        _, score, ident, pw, _, source, _ = candidate

        source_age = self.ros_now()-source

        if not 0 <= source_age <= STATE_MAX_AGE:
            manager.step(time.monotonic())
            self.hold("SOURCE STALE BEFORE PLAN")
            return

        if type(self.action) is not ShadowAction:
            manager.expire()
            raise RuntimeError("SHADOW ACTION REQUIRED")

        self.target_world = pw.copy()
        self.target_seen = time.monotonic()-source_age
        self.target_id = ident[0]
        self.target_class = TRACK_CLASS
        self.target_score = -score

        original.FastTracker.control_cb(self)



def main():

    rclpy.init()

    node = TrackerV3()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        print("USER STOP")

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
