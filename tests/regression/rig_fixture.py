import os
import sys
import time
import uuid
import threading
import subprocess
from pathlib import Path

import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import JointState
from vision_msgs.msg import Detection3D, Detection3DArray
from vision_msgs.msg import ObjectHypothesisWithPose
from geometry_msgs.msg import TransformStamped
from control_msgs.action import FollowJointTrajectory
from tf2_ros import StaticTransformBroadcaster

from real_visual_tracker_final import (
    Arm, TCP_OFFSET, LOW, HIGH, quat_to_rot
)

assert os.environ["ROS_DOMAIN_ID"] == os.environ["TEST_DOMAIN"]
assert os.environ.get("ROS_LOCALHOST_ONLY") == "1"
assert os.environ.get("RMW_IMPLEMENTATION") == "rmw_cyclonedds_cpp"


PREFIX = "/final_bench_" + uuid.uuid4().hex[:10]
FRAME = PREFIX.strip("/") + "_camera"

JOINTS = ["joint1", "joint2", "joint3", "joint4"]

Q0 = np.array([
    -0.16567, 0.48353,
    -0.15680, 0.03682
])

T = np.array([0.240, 0.039, 0.052])

QUAT = np.array([
    0.680, -0.648, 0.226, -0.259
])

QUAT /= np.linalg.norm(QUAT)
R = quat_to_rot(QUAT)

ARM = Arm(tcp_offset=TCP_OFFSET)
TCP0 = ARM.fk(Q0)[:3, 3]

REMAPS = [
    "-r", "/joint_states:=" + PREFIX + "/joints",
    "-r", "/object_localizer/detections_3d_mask:=" + PREFIX + "/detections",
    "-r", "/arm_controller/follow_joint_trajectory:=" + PREFIX + "/trajectory",
    "-r", "/arm_controller/follow_joint_trajectory/_action/send_goal:=" + PREFIX + "/trajectory/_action/send_goal",
    "-r", "/arm_controller/follow_joint_trajectory/_action/get_result:=" + PREFIX + "/trajectory/_action/get_result",
    "-r", "/arm_controller/follow_joint_trajectory/_action/cancel_goal:=" + PREFIX + "/trajectory/_action/cancel_goal",
    "-r", "/arm_controller/follow_joint_trajectory/_action/feedback:=" + PREFIX + "/trajectory/_action/feedback",
    "-r", "/arm_controller/follow_joint_trajectory/_action/status:=" + PREFIX + "/trajectory/_action/status",
    "-r", "/tf:=" + PREFIX + "/tf",
    "-r", "/tf_static:=" + PREFIX + "/tf_static"
]


class VirtualRig(Node):

    def __init__(self):
        super().__init__(
            "virtual_final_bench",
            cli_args=["--ros-args"] + REMAPS
        )

        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.group = ReentrantCallbackGroup()

        self.q = Q0.copy()
        self.target = None
        self.mode = "none"
        self.phase_start = time.monotonic()

        self.segment = None
        self.goal_count = 0
        self.rejected_count = 0

        self.joint_pub = self.create_publisher(
            JointState, PREFIX + "/joints", 10
        )

        self.det_pub = self.create_publisher(
            Detection3DArray, PREFIX + "/detections", 10
        )

        self.tf = StaticTransformBroadcaster(self)

        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = "world"
        tf.child_frame_id = FRAME

        for name, value in zip("xyz", T):
            setattr(tf.transform.translation, name, float(value))

        for name, value in zip("xyzw", QUAT):
            setattr(tf.transform.rotation, name, float(value))

        self.tf.sendTransform(tf)

        self.server = ActionServer(
            self,
            FollowJointTrajectory,
            PREFIX + "/trajectory",
            execute_callback=self.execute,
            goal_callback=self.validate_goal,
            handle_accepted_callback=self.accept,
            callback_group=self.group
        )

        self.create_timer(
            0.02, self.update,
            callback_group=self.group
        )

    def validate_goal(self, goal):
        tr = goal.trajectory

        valid = (
            list(tr.joint_names) == JOINTS
            and len(tr.points) == 1
        )

        if valid:
            pt = tr.points[0]
            q = np.asarray(pt.positions, dtype=float)

            duration = (
                pt.time_from_start.sec
                + pt.time_from_start.nanosec * 1e-9
            )

            valid = (
                q.shape == (4,)
                and np.all(np.isfinite(q))
                and np.all(q >= LOW)
                and np.all(q <= HIGH)
                and 0.01 <= duration <= 1.0
            )

        if not valid:
            with self.lock:
                self.rejected_count += 1
            return GoalResponse.REJECT

        return GoalResponse.ACCEPT

    def advance(self, now):
        if self.segment is None:
            return

        _, start, end, t0, duration = self.segment

        alpha = np.clip(
            (now - t0) / duration, 0.0, 1.0
        )

        self.q = start + alpha * (end - start)

    def accept(self, handle):
        pt = handle.request.trajectory.points[0]

        duration = (
            pt.time_from_start.sec
            + pt.time_from_start.nanosec * 1e-9
        )

        with self.lock:
            now = time.monotonic()
            self.advance(now)

            self.segment = (
                handle,
                self.q.copy(),
                np.array(pt.positions, dtype=float),
                now,
                duration
            )

            self.goal_count += 1

        handle.execute()

    def execute(self, handle):
        result = FollowJointTrajectory.Result()

        while not self.stop_event.is_set():
            with self.lock:
                if (
                    self.segment is None
                    or self.segment[0] is not handle
                ):
                    handle.abort()
                    result.error_code = -1
                    result.error_string = "Superseded"
                    return result

                _, _, _, t0, duration = self.segment

                now = time.monotonic()
                self.advance(now)

                if now >= t0 + duration:
                    handle.succeed()
                    result.error_code = 0
                    return result

            time.sleep(0.01)

        handle.abort()
        result.error_code = -1
        return result

    def set_mode(self, mode):
        with self.lock:
            self.mode = mode
            self.phase_start = time.monotonic()
            self.target = None

    def update(self):
        with self.lock:
            now = time.monotonic()
            self.advance(now)

            stamp = self.get_clock().now().to_msg()

            joints = JointState()
            joints.header.stamp = stamp
            joints.name = JOINTS
            joints.position = self.q.tolist()

            self.joint_pub.publish(joints)

            if self.mode == "none":
                return

            target = TCP0.copy()

            if self.mode == "A":
                target[0] += 0.025

            elif self.mode == "B":
                target[1] -= 0.025

            elif self.mode == "circle":
                theta = (
                    now - self.phase_start
                ) * 2 * np.pi / 6.0

                target[0] += 0.018 * np.cos(theta)
                target[1] += 0.018 * np.sin(theta)

            self.target = target.copy()

            pc = R.T @ (target - T)

            msg = Detection3DArray()
            msg.header.stamp = stamp
            msg.header.frame_id = FRAME

            det = Detection3D()
            det.header = msg.header
            det.id = "virtual_blue_block"

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = "blue block"
            hyp.hypothesis.score = 0.90

            p = hyp.pose.pose.position
            p.x, p.y, p.z = map(float, pc)

            hyp.pose.pose.orientation.w = 1.0

            det.results.append(hyp)
            msg.detections.append(det)

            self.det_pub.publish(msg)

    def snapshot(self):
        with self.lock:
            tcp = ARM.fk(self.q)[:3, 3]

            error = (
                None if self.target is None
                else float(np.linalg.norm(
                    tcp[:2] - self.target[:2]
                ) * 1000)
            )

            return self.goal_count, error, tcp.copy()
