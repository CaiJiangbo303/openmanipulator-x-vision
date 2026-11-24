#!/usr/bin/env python3

import os
import time
import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.time import Time
from rclpy.action import ActionClient

from sensor_msgs.msg import JointState
from vision_msgs.msg import Detection3DArray
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from tf2_ros import Buffer, TransformListener

# 学长原始运动学代码，READ ONLY
from manipulator_grasp.kinematics import Arm, LOW, HIGH, interpolate


# ============================================================
# FAST CONTINUOUS TRACKING
# ============================================================

CONTROL_HZ = 10.0

# 10 Hz 下单周期最多 14 mm，理论指令级约 140 mm/s
MAX_CART_STEP = 0.035

# 单周期每个关节最多变化约 2.3°
MAX_JOINT_STEP = 0.100

# 短时域轨迹，下一周期允许覆盖
HORIZON_SEC = 0.14

XY_DEADBAND = 0.003
TARGET_STALE_SEC = 0.35

MIN_SCORE = 0.28

# 目标离当前 TCP 太远直接拒绝，避免追远处背景物体
MAX_TARGET_TCP_DISTANCE = 0.18

# 目标跳变超过 8 cm，不允许突然换物体
MAX_TARGET_JUMP = 0.08

IK_MARGIN = 0.04

# 两夹爪最前端中心
# end_effector = 126 mm
# physical front center = 146.7 mm
TCP_OFFSET = (0.0207, 0.0, 0.0)

# 第一阶段保持 Z 不动
TRACK_Z = False

# 当前机械臂实际安全工作区
SAFE_X = (0.20, 0.395)
SAFE_Y = (-0.16, 0.16)

# 防止把相机远处误检测当跟踪对象
CAMERA_Z_RANGE = (0.12, 0.48)

# 可选：
# export TRACK_CLASS="cup"
# 留空则自动选择最近非 person 目标
TRACK_CLASS = os.environ.get(
    'TRACK_CLASS',
    ''
).strip()


def quat_to_rot(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)

    if n < 1e-12:
        raise ValueError("bad quaternion")

    x, y, z, w = q / n

    return np.array([
        [
            1-2*(y*y+z*z),
            2*(x*y-z*w),
            2*(x*z+y*w)
        ],
        [
            2*(x*y+z*w),
            1-2*(x*x+z*z),
            2*(y*z-x*w)
        ],
        [
            2*(x*z-y*w),
            2*(y*z+x*w),
            1-2*(x*x+y*y)
        ],
    ], dtype=float)


class FastTracker(Node):

    def __init__(self):
        super().__init__('real_visual_tracker_final')

        # 学长 Arm 解析 IK
        self.arm = Arm(
            tcp_offset=TCP_OFFSET
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        self.action = ActionClient(
            self,
            FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory'
        )

        self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_cb,
            10
        )

        self.create_subscription(
            Detection3DArray,
            '/object_localizer/detections_3d_mask',
            self.detection_cb,
            10
        )

        self.timer = self.create_timer(
            1.0 / CONTROL_HZ,
            self.control_cb
        )

        self.q = None

        self.fixed_z = None
        self.fixed_pitch = None

        self.target_world = None
        self.target_seen = None
        self.target_id = None
        self.target_class = None
        self.target_score = 0.0

        # 只防止 action goal 的“提交请求”叠加，
        # goal 一旦 accepted 就允许下一周期覆盖旧轨迹。
        self.goal_pending = False

        self.command_count = 0
        self.last_print = 0.0

        print()
        print("==========================================")
        print(" FAST REAL VISUAL TRACKING")
        print("==========================================")
        print("Perception : existing Tracker + Mask3D")
        print("IK         : senior Arm.ik()")
        print("Step limit : senior interpolate()")
        print("Control    : 10 Hz receding horizon")
        print(f"Horizon    : {HORIZON_SEC:.2f} s")
        print(
            f"Cart step  : "
            f"{MAX_CART_STEP*1000:.0f} mm/cycle"
        )
        print(
            f"Joint step : "
            f"{MAX_JOINT_STEP:.3f} rad/cycle"
        )
        print(
            "Class lock : "
            + (
                TRACK_CLASS
                if TRACK_CLASS
                else "AUTO nearest non-person"
            )
        )
        print("Z tracking : OFF")
        print("Grasp      : OFF")
        print()
        print("Ctrl+C = STOP TRACKING")
        print("Power Hub = EMERGENCY STOP")
        print()

    def joint_cb(self, msg):

        p = dict(
            zip(msg.name, msg.position)
        )

        names = [
            'joint1',
            'joint2',
            'joint3',
            'joint4'
        ]

        if not all(
            n in p
            for n in names
        ):
            return

        self.q = np.array(
            [p[n] for n in names],
            dtype=float
        )

        if self.fixed_z is None:

            tcp = self.arm.fk(
                self.q
            )[:3, 3]

            self.fixed_z = float(
                tcp[2]
            )

            self.fixed_pitch = float(
                np.sum(
                    self.q[1:]
                )
            )

            print(
                "START TCP = "
                f"[{tcp[0]:+.4f}, "
                f"{tcp[1]:+.4f}, "
                f"{tcp[2]:+.4f}]"
            )

            print(
                f"LOCKED Z = "
                f"{self.fixed_z:+.4f}"
            )

            print(
                f"LOCKED PITCH = "
                f"{self.fixed_pitch:+.4f}"
            )

    def detection_cb(self, msg):

        if self.q is None:
            return

        try:
            tf = self.tf_buffer.lookup_transform(
                'world',
                msg.header.frame_id,
                Time()
            )

        except Exception:
            return

        tr = tf.transform.translation
        qr = tf.transform.rotation

        R = quat_to_rot([
            qr.x,
            qr.y,
            qr.z,
            qr.w
        ])

        t = np.array([
            tr.x,
            tr.y,
            tr.z
        ])

        current_tcp = self.arm.fk(
            self.q
        )[:3, 3]

        candidates = []

        for det in msg.detections:

            if not det.results:
                continue

            r = det.results[0]

            cls = r.hypothesis.class_id
            score = float(
                r.hypothesis.score
            )

            # 人永远不能驱动机械臂
            if cls == 'person':
                continue

            if (
                TRACK_CLASS
                and cls != TRACK_CLASS
            ):
                continue

            if score < MIN_SCORE:
                continue

            p = r.pose.pose.position

            pc = np.array([
                p.x,
                p.y,
                p.z
            ], dtype=float)

            if not np.all(
                np.isfinite(pc)
            ):
                continue

            if not (
                CAMERA_Z_RANGE[0]
                <= pc[2]
                <= CAMERA_Z_RANGE[1]
            ):
                continue

            pw = R @ pc + t

            if not np.all(
                np.isfinite(pw)
            ):
                continue

            dist_tcp = float(
                np.linalg.norm(
                    pw[:2]
                    - current_tcp[:2]
                )
            )

            if (
                dist_tcp
                > MAX_TARGET_TCP_DISTANCE
            ):
                continue

            candidates.append(
                (
                    dist_tcp,
                    -score,
                    det.id,
                    cls,
                    score,
                    pw
                )
            )

        if not candidates:
            return

        # 已有目标：空间连续性优先。
        if self.target_world is not None:

            chosen = min(
                candidates,
                key=lambda x:
                    np.linalg.norm(
                        x[5][:2]
                        - self.target_world[:2]
                    )
            )

            jump = np.linalg.norm(
                chosen[5][:2]
                - self.target_world[:2]
            )

            if jump > MAX_TARGET_JUMP:
                return

        else:

            # 初次自动选择离 TCP 最近的物体
            candidates.sort(
                key=lambda x:
                    (x[0], x[1])
            )

            chosen = candidates[0]

        _, _, track_id, cls, score, pw = chosen

        # 适度平滑，但保持较快响应
        if self.target_world is None:
            filtered = pw.copy()
        else:
            alpha = 0.85

            filtered = (
                alpha * pw
                + (1.0-alpha)
                * self.target_world
            )

        self.target_world = filtered
        self.target_seen = time.monotonic()

        self.target_id = track_id
        self.target_class = cls
        self.target_score = score

    def status(self, msg):

        now = time.monotonic()

        if now - self.last_print > 0.8:
            print(msg)
            self.last_print = now

    def control_cb(self):

        if self.q is None:
            return

        now = time.monotonic()

        if (
            self.target_world is None
            or self.target_seen is None
        ):
            self.status(
                "HOLD: waiting for valid object"
            )
            return

        age = (
            now - self.target_seen
        )

        if age > TARGET_STALE_SEC:
            self.status(
                f"HOLD: target stale "
                f"{age:.2f}s"
            )
            return

        tcp = self.arm.fk(
            self.q
        )[:3, 3]

        target = (
            self.target_world.copy()
        )

        # ========================================
        # FAST XY SERVO
        # ========================================

        error_xy = (
            target[:2]
            - tcp[:2]
        )

        error = float(
            np.linalg.norm(
                error_xy
            )
        )

        if error < XY_DEADBAND:
            self.status(
                f"HOLD: aligned "
                f"{error*1000:.1f} mm"
            )
            return

        # 自适应步长：
        # 目标越远越接近 MAX_CART_STEP
        # Faster adaptive servo:
        # large error -> fast pursuit
        # small error -> automatic slowdown
        if error > 0.060:
            step = MAX_CART_STEP
        elif error > 0.030:
            step = min(MAX_CART_STEP, 0.026)
        elif error > 0.015:
            step = min(MAX_CART_STEP, 0.016)
        else:
            step = min(MAX_CART_STEP, max(0.005, error * 0.80))

        desired_xy = (
            tcp[:2]
            + error_xy / error
            * step
        )

        if not (
            SAFE_X[0]
            <= desired_xy[0]
            <= SAFE_X[1]
        ):
            self.status(
                "HOLD: X workspace limit"
            )
            return

        if not (
            SAFE_Y[0]
            <= desired_xy[1]
            <= SAFE_Y[1]
        ):
            self.status(
                "HOLD: Y workspace limit"
            )
            return

        desired = np.array([
            desired_xy[0],
            desired_xy[1],
            (
                target[2]
                if TRACK_Z
                else self.fixed_z
            )
        ])

        # ========================================
        # SENIOR ANALYTIC IK
        # ========================================

        solutions = self.arm.ik(
            desired,
            self.fixed_pitch,
            margin=IK_MARGIN
        )

        if not solutions:
            self.status(
                "HOLD: IK unavailable"
            )
            return

        # 离当前真实关节最近的 IK branch
        q_goal = min(
            solutions,
            key=lambda x:
                np.linalg.norm(
                    x - self.q
                )
        )

        # 防止 branch 突然跳解
        if (
            np.max(
                np.abs(
                    q_goal
                    - self.q
                )
            )
            > 0.40
        ):
            self.status(
                "HOLD: IK branch jump"
            )
            return

        # ========================================
        # SENIOR BOUNDED INTERPOLATION
        # ========================================

        path = interpolate(
            self.q,
            q_goal,
            MAX_JOINT_STEP
        )

        if len(path) < 2:
            return

        q_cmd = np.asarray(
            path[1],
            dtype=float
        )

        if not (
            np.all(
                q_cmd >= LOW + IK_MARGIN
            )
            and np.all(
                q_cmd <= HIGH - IK_MARGIN
            )
        ):
            self.status(
                "HOLD: joint limit"
            )
            return

        if self.goal_pending:
            return

        if not self.action.server_is_ready():
            self.status(
                "HOLD: action server"
            )
            return

        goal = FollowJointTrajectory.Goal()

        goal.trajectory.joint_names = [
            'joint1',
            'joint2',
            'joint3',
            'joint4'
        ]

        pt = JointTrajectoryPoint()

        pt.positions = [
            float(v)
            for v in q_cmd
        ]

        sec = int(HORIZON_SEC)

        nsec = int(
            (
                HORIZON_SEC
                - sec
            )
            * 1e9
        )

        pt.time_from_start = Duration(
            sec=sec,
            nanosec=nsec
        )

        goal.trajectory.points = [pt]

        # 只等 goal 被 controller 接受，
        # 不等 trajectory 执行结束。
        self.goal_pending = True

        f = self.action.send_goal_async(
            goal
        )

        f.add_done_callback(
            self.goal_response_cb
        )

        self.command_count += 1

        if self.command_count % 10 == 0:

            print(
                f"TRACK "
                f"{self.target_class}/"
                f"{self.target_id} "
                f"score={self.target_score:.2f} "
                f"err={error*1000:.0f}mm "
                f"TCP=["
                f"{tcp[0]:+.3f},"
                f"{tcp[1]:+.3f}] "
                f"OBJ=["
                f"{target[0]:+.3f},"
                f"{target[1]:+.3f}]"
            )

    def goal_response_cb(
        self,
        future
    ):

        try:

            gh = future.result()

            if (
                gh is None
                or not gh.accepted
            ):
                print(
                    "GOAL ACCEPTED = FAIL"
                )

            # Receding horizon:
            # 一旦 controller 接受就允许下一帧更新
            self.goal_pending = False

        except Exception as exc:

            self.goal_pending = False

            print(
                "GOAL SEND ERROR:",
                exc
            )


def main():

    rclpy.init()

    node = FastTracker()

    if not node.action.wait_for_server(
        timeout_sec=5.0
    ):
        print(
            "ARM ACTION SERVER = FAIL"
        )

        raise SystemExit(1)

    print(
        "ARM ACTION SERVER = PASS"
    )

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        print()
        print(
            "TRACKING STOPPED BY USER"
        )

    finally:

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
