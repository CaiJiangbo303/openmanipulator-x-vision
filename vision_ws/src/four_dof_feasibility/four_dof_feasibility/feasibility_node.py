#!/usr/bin/env python3

import math
from collections import deque
import numpy as np

import rclpy

from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration

from geometry_msgs.msg import PointStamped

from tf2_ros import (
    Buffer,
    TransformListener,
)

from tf2_geometry_msgs import (
    do_transform_point,
)

from object_state_msgs.msg import (
    ObjectPredictionArray,
)

from four_dof_feasibility_msgs.msg import (
    FourDofFeasibility,
    FourDofFeasibilityArray,
)


LOWER = np.array([
    -math.pi,
    -1.5,
    -1.5,
    -1.7,
], dtype=np.float64)

UPPER = np.array([
    math.pi,
    1.5,
    1.4,
    1.97,
], dtype=np.float64)

# Conservative geometric upper bound from arm_link1
# to arm_end_effector_link.
#
# Derived from the real URDF chain and intentionally
# slightly padded. Targets beyond this radius cannot
# be reached regardless of joint configuration.
MAX_GEOMETRIC_REACH = 0.47

# -------------------------------------------------
# Real URDF geometry for the inverted 4-DOF arm.
#
# After q1, all remaining joints rotate around Y,
# so the arm becomes a planar 3-pitch chain.
# -------------------------------------------------

SHOULDER_X = 0.012
SHOULDER_Z = 0.017 + 0.0595

FIRST_LINK_X = 0.024
FIRST_LINK_Z = 0.128

FIRST_LINK_LENGTH = math.hypot(
    FIRST_LINK_X,
    FIRST_LINK_Z
)

FIRST_LINK_ANGLE = math.atan2(
    FIRST_LINK_Z,
    FIRST_LINK_X
)

SECOND_LINK_LENGTH = 0.124
TOOL_LINK_LENGTH = 0.126

# q2 + q3 + q4 feasible orientation interval.
PHI_MIN = float(
    LOWER[1]
    +
    LOWER[2]
    +
    LOWER[3]
)

PHI_MAX = float(
    UPPER[1]
    +
    UPPER[2]
    +
    UPPER[3]
)

# About 0.76 degree spacing.
# This is cheap because all samples are evaluated
# vectorially with NumPy rather than iterative IK.
PHI_SAMPLES = np.linspace(
    PHI_MIN,
    PHI_MAX,
    721
)


def trans(x, y, z):

    T = np.eye(4)

    T[:3, 3] = [
        x,
        y,
        z,
    ]

    return T


def rz(q):

    c = math.cos(q)
    s = math.sin(q)

    T = np.eye(4)

    T[:3, :3] = [
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ]

    return T


def ry(q):

    c = math.cos(q)
    s = math.sin(q)

    T = np.eye(4)

    T[:3, :3] = [
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c],
    ]

    return T


def fk(q):

    q1, q2, q3, q4 = q

    T = np.eye(4)

    T = T @ trans(
        0.012,
        0.0,
        0.017
    )

    T = T @ rz(q1)

    T = T @ trans(
        0.0,
        0.0,
        0.0595
    )

    T = T @ ry(q2)

    T = T @ trans(
        0.024,
        0.0,
        0.128
    )

    T = T @ ry(q3)

    T = T @ trans(
        0.124,
        0.0,
        0.0
    )

    T = T @ ry(q4)

    T = T @ trans(
        0.126,
        0.0,
        0.0
    )

    return T


def position(q):

    return fk(q)[:3, 3]


def numerical_jacobian(q):

    eps = 1e-6

    J = np.zeros(
        (3, 4),
        dtype=np.float64
    )

    p0 = position(q)

    for i in range(4):

        q2 = q.copy()

        q2[i] += eps

        J[:, i] = (
            position(q2)
            -
            p0
        ) / eps

    return J


def joint_margins(q):

    return np.minimum(
        q - LOWER,
        UPPER - q
    )


def solve_from_seed(
    target,
    seed,
    max_iterations=250
):

    q = np.clip(
        np.array(
            seed,
            dtype=np.float64
        ),
        LOWER,
        UPPER
    )

    damping = 0.03

    for iteration in range(
        max_iterations
    ):

        p = position(q)

        error = (
            target
            -
            p
        )

        residual = np.linalg.norm(
            error
        )

        if residual < 1e-5:

            return (
                q,
                residual,
                iteration + 1
            )

        J = numerical_jacobian(
            q
        )

        A = (
            J @ J.T
            +
            damping ** 2
            *
            np.eye(3)
        )

        try:

            dq = (
                J.T
                @
                np.linalg.solve(
                    A,
                    error
                )
            )

        except np.linalg.LinAlgError:

            break

        step_norm = np.linalg.norm(
            dq
        )

        max_step = 0.12

        if step_norm > max_step:

            dq *= (
                max_step
                /
                step_norm
            )

        q += dq

        q = np.clip(
            q,
            LOWER,
            UPPER
        )

    residual = np.linalg.norm(
        target
        -
        position(q)
    )

    return (
        q,
        residual,
        max_iterations
    )


def wrapped_angle(angle):

    return math.atan2(
        math.sin(angle),
        math.cos(angle)
    )


def solve_ik(
    target,
    marginal_tolerance
):
    # -------------------------------------------------
    # Structure-aware IK for:
    #
    # q1 : Z yaw
    # q2 : Y pitch
    # q3 : Y pitch
    # q4 : Y pitch
    #
    # Once q1 is selected, the remaining chain is
    # exactly planar.
    #
    # We scan only:
    #
    #   phi = q2 + q3 + q4
    #
    # and solve q2/q3 analytically as a two-link
    # planar problem. This avoids iterative Jacobian IK.
    # -------------------------------------------------

    del marginal_tolerance

    x = float(target[0])
    y = float(target[1])
    z = float(target[2])

    dx = (
        x
        -
        SHOULDER_X
    )

    dy = y

    planar_z = (
        z
        -
        SHOULDER_Z
    )

    radial_abs = math.hypot(
        dx,
        dy
    )

    if radial_abs > 1e-12:

        yaw = math.atan2(
            dy,
            dx
        )

    else:

        yaw = 0.0

    yaw_front = wrapped_angle(
        yaw
    )

    yaw_back = wrapped_angle(
        yaw
        +
        math.pi
    )

    # Two equivalent radial descriptions:
    #
    # front:
    #   positive planar X
    #
    # back:
    #   q1 rotated by pi and planar X negative
    branches = [
        (
            yaw_front,
            radial_abs
        ),
        (
            yaw_back,
            -radial_abs
        ),
    ]

    best = None

    for q1, radial in branches:

        phi = PHI_SAMPLES

        # Remove the final 126 mm tool vector.
        #
        # Ry(phi) * [L, 0, 0]:
        #
        # x = +L cos(phi)
        # z = -L sin(phi)
        wrist_x = (
            radial
            -
            TOOL_LINK_LENGTH
            *
            np.cos(phi)
        )

        wrist_z = (
            planar_z
            +
            TOOL_LINK_LENGTH
            *
            np.sin(phi)
        )

        cos_delta = (
            wrist_x * wrist_x
            +
            wrist_z * wrist_z
            -
            FIRST_LINK_LENGTH
            *
            FIRST_LINK_LENGTH
            -
            SECOND_LINK_LENGTH
            *
            SECOND_LINK_LENGTH
        ) / (
            2.0
            *
            FIRST_LINK_LENGTH
            *
            SECOND_LINK_LENGTH
        )

        valid = (
            (cos_delta >= -1.000000001)
            &
            (cos_delta <= +1.000000001)
        )

        if not np.any(valid):
            continue

        valid_phi = phi[valid]

        valid_x = wrist_x[valid]
        valid_z = wrist_z[valid]

        valid_cos = np.clip(
            cos_delta[valid],
            -1.0,
            +1.0
        )

        elbow_angle = np.arccos(
            valid_cos
        )

        # Both elbow branches.
        for delta in (
            elbow_angle,
            -elbow_angle
        ):

            alpha1 = (
                np.arctan2(
                    valid_z,
                    valid_x
                )
                -
                np.arctan2(
                    SECOND_LINK_LENGTH
                    *
                    np.sin(delta),

                    FIRST_LINK_LENGTH
                    +
                    SECOND_LINK_LENGTH
                    *
                    np.cos(delta)
                )
            )

            alpha2 = (
                alpha1
                +
                delta
            )

            # First physical link has an intrinsic
            # offset angle because its URDF vector is
            # (0.024, 0, 0.128), not a pure X link.
            q2 = (
                FIRST_LINK_ANGLE
                -
                alpha1
            )

            q3 = (
                -alpha2
                -
                q2
            )

            q4 = (
                valid_phi
                -
                q2
                -
                q3
            )

            q1_array = np.full_like(
                q2,
                q1
            )

            Q = np.column_stack(
                (
                    q1_array,
                    q2,
                    q3,
                    q4
                )
            )

            margins = np.minimum(
                Q - LOWER,
                UPPER - Q
            )

            minimum_margins = np.min(
                margins,
                axis=1
            )

            # Prefer the configuration furthest from
            # all joint limits.
            index = int(
                np.argmax(
                    minimum_margins
                )
            )

            candidate_q = (
                Q[index].copy()
            )

            candidate_margins = (
                margins[index].copy()
            )

            candidate_min_margin = float(
                minimum_margins[index]
            )

            if (
                best is None
                or
                candidate_min_margin
                >
                best['min_margin']
            ):

                best = {
                    'q':
                        candidate_q,

                    'margins':
                        candidate_margins,

                    'min_margin':
                        candidate_min_margin,
                }

    if best is None:

        # No planar geometric solution exists.
        #
        # Use neutral configuration only to provide
        # a finite diagnostic residual.
        q = np.zeros(
            4,
            dtype=np.float64
        )

        residual = float(
            np.linalg.norm(
                target
                -
                position(q)
            )
        )

        margins = joint_margins(
            q
        )

        return {
            'q':
                q,

            'residual':
                residual,

            'margins':
                margins,

            'min_margin':
                -1.0,

            'iterations':
                0,
        }

    # The analytic candidate should reconstruct the
    # target essentially exactly. Verify using the
    # already validated FK.
    residual = float(
        np.linalg.norm(
            target
            -
            position(
                best['q']
            )
        )
    )

    best['residual'] = residual

    best['iterations'] = 0

    return best


class FourDofFeasibilityNode(Node):

    def __init__(self):

        super().__init__(
            'four_dof_feasibility'
        )

        self.declare_parameter(
            'input_topic',
            '/object_motion_predictor/predictions'
        )

        self.declare_parameter(
            'output_topic',
            '/four_dof_feasibility/results'
        )

        self.declare_parameter(
            'arm_base_frame',
            'arm_link1'
        )

        self.declare_parameter(
            'reachable_tolerance',
            0.003
        )

        self.declare_parameter(
            'marginal_tolerance',
            0.008
        )

        self.declare_parameter(
            'joint_margin_warning',
            0.15
        )

        self.input_topic = (
            self.get_parameter(
                'input_topic'
            ).value
        )

        self.output_topic = (
            self.get_parameter(
                'output_topic'
            ).value
        )

        self.arm_base_frame = (
            self.get_parameter(
                'arm_base_frame'
            ).value
        )

        self.reachable_tolerance = float(
            self.get_parameter(
                'reachable_tolerance'
            ).value
        )

        self.marginal_tolerance = float(
            self.get_parameter(
                'marginal_tolerance'
            ).value
        )

        self.joint_margin_warning = float(
            self.get_parameter(
                'joint_margin_warning'
            ).value
        )

        self.tf_buffer = Buffer()

        self.tf_listener = (
            TransformListener(
                self.tf_buffer,
                self
            )
        )

        self.sub = self.create_subscription(
            ObjectPredictionArray,
            self.input_topic,
            self.prediction_callback,
            1
        )

        self.pub = self.create_publisher(
            FourDofFeasibilityArray,
            self.output_topic,
            10
        )

        # -------------------------------------------------
        # Exact-stamp TF synchronization.
        #
        # Prediction messages can arrive a few milliseconds
        # before the corresponding dynamic TF. Never fall
        # back to "latest" TF because the F550 base can move.
        #
        # Keep a short queue and retry from a timer so the
        # TF listener itself is never blocked by this callback.
        # -------------------------------------------------

        # One message is actively waiting for its
        # exact-stamp TF. While it waits, retain only
        # the newest incoming prediction.
        #
        # This gives TF time to catch up without ever
        # building an unbounded stale backlog.
        self.pending_message = None
        self.latest_waiting_message = None

        self.tf_wait_timeout = 0.50

        self.process_timer = self.create_timer(
            0.01,
            self.process_pending
        )

        self.counter = 0

        self.get_logger().info(
            'Four-DOF Feasibility started'
        )

        self.get_logger().info(
            f'Input: {self.input_topic}'
        )

        self.get_logger().info(
            f'Output: {self.output_topic}'
        )

        self.get_logger().info(
            f'Arm base: {self.arm_base_frame}'
        )

        self.get_logger().warning(
            'Feasibility v1 uses the arm-base pose '
            'at prediction source time. Future F550 '
            'base motion is not yet predicted.'
        )

    def classify(
        self,
        residual,
        min_margin
    ):

        if (
            min_margin
            <
            0.0
        ):

            return (
                FourDofFeasibility.UNREACHABLE,
                (
                    'joint limits violated '
                    f'{min_margin:.3f} rad'
                )
            )

        if (
            residual
            >
            self.marginal_tolerance
        ):

            return (
                FourDofFeasibility.UNREACHABLE,
                (
                    'IK residual '
                    f'{residual*1000:.1f} mm'
                )
            )

        if (
            residual
            >
            self.reachable_tolerance
        ):

            return (
                FourDofFeasibility.MARGINAL,
                (
                    'position residual '
                    f'{residual*1000:.1f} mm'
                )
            )

        if (
            min_margin
            <
            self.joint_margin_warning
        ):

            return (
                FourDofFeasibility.MARGINAL,
                (
                    'near joint limit '
                    f'{min_margin:.3f} rad'
                )
            )

        return (
            FourDofFeasibility.REACHABLE,
            'position reachable'
        )

    def prediction_callback(
        self,
        msg
    ):

        # -------------------------------------------------
        # Two-slot freshness policy.
        #
        # pending_message:
        #   protected while waiting for its exact TF
        #
        # latest_waiting_message:
        #   always replaced by the newest prediction
        #
        # Intermediate frames are intentionally dropped.
        # -------------------------------------------------

        if self.pending_message is None:

            self.pending_message = msg

        else:

            self.latest_waiting_message = msg

    def promote_latest(
        self
    ):

        self.pending_message = (
            self.latest_waiting_message
        )

        self.latest_waiting_message = None

    def process_pending(
        self
    ):

        if self.pending_message is None:
            return

        msg = self.pending_message

        if not msg.header.frame_id:

            self.pending_message = None

            self.promote_latest()

            return

        stamp = Time.from_msg(
            msg.header.stamp
        )

        try:

            transform = (
                self.tf_buffer.lookup_transform(
                    self.arm_base_frame,
                    msg.header.frame_id,
                    stamp,
                    timeout=Duration(
                        seconds=0.0
                    )
                )
            )

        except Exception:

            now = self.get_clock().now()

            age = (
                now.nanoseconds
                -
                stamp.nanoseconds
            ) * 1e-9

            # TF may simply be a few milliseconds behind.
            # Keep this exact timestamp protected while
            # waiting.
            if age <= self.tf_wait_timeout:
                return

            self.get_logger().warning(
                'Drop prediction: exact TF unavailable '
                f'after {age:.3f}s | '
                f'stamp={stamp.nanoseconds * 1e-9:.3f}'
            )

            self.pending_message = None

            self.promote_latest()

            return

        # Exact timestamp TF is available.
        #
        # Clear the protected slot BEFORE processing so
        # incoming callbacks cannot create a stale queue.

        self.pending_message = None

        self.process_prediction(
            msg,
            transform
        )

        self.promote_latest()

    def process_prediction(
        self,
        msg,
        transform
    ):

        output = (
            FourDofFeasibilityArray()
        )

        output.header = msg.header

        for prediction in (
            msg.predictions
        ):

            for horizon, world_point in zip(
                prediction.horizons,
                prediction.positions
            ):

                point_world = (
                    PointStamped()
                )

                point_world.header = (
                    msg.header
                )

                point_world.point = (
                    world_point
                )

                point_arm = (
                    do_transform_point(
                        point_world,
                        transform
                    )
                )

                target = np.array(
                    [
                        point_arm.point.x,
                        point_arm.point.y,
                        point_arm.point.z,
                    ],
                    dtype=np.float64
                )

                target_distance = float(
                    np.linalg.norm(
                        target
                    )
                )

                # -------------------------------------------------
                # Fast geometric rejection.
                # -------------------------------------------------

                if (
                    target_distance
                    >
                    MAX_GEOMETRIC_REACH
                ):

                    result = (
                        FourDofFeasibility()
                    )

                    result.track_id = (
                        prediction.track_id
                    )

                    result.class_id = (
                        prediction.class_id
                    )

                    result.horizon = float(
                        horizon
                    )

                    result.target_world = (
                        world_point
                    )

                    result.target_arm = (
                        point_arm.point
                    )

                    result.joint_positions = [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]

                    result.joint_margins = [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]

                    result.position_residual = float(
                        target_distance
                        -
                        MAX_GEOMETRIC_REACH
                    )

                    result.min_joint_margin = 0.0

                    result.status = int(
                        FourDofFeasibility.UNREACHABLE
                    )

                    result.reason = (
                        'outside geometric reach: '
                        f'{target_distance:.3f} m > '
                        f'{MAX_GEOMETRIC_REACH:.3f} m'
                    )

                    result.frozen_base_assumption = (
                        True
                    )

                    output.candidates.append(
                        result
                    )

                    continue

                # -------------------------------------------------
                # Precise numerical IK.
                # -------------------------------------------------

                solution = solve_ik(
                    target,
                    self.marginal_tolerance
                )

                status, reason = (
                    self.classify(
                        solution[
                            'residual'
                        ],
                        solution[
                            'min_margin'
                        ]
                    )
                )

                result = (
                    FourDofFeasibility()
                )

                result.track_id = (
                    prediction.track_id
                )

                result.class_id = (
                    prediction.class_id
                )

                result.horizon = float(
                    horizon
                )

                result.target_world = (
                    world_point
                )

                result.target_arm = (
                    point_arm.point
                )

                result.joint_positions = [
                    float(v)
                    for v in solution['q']
                ]

                result.joint_margins = [
                    float(v)
                    for v in solution[
                        'margins'
                    ]
                ]

                result.position_residual = float(
                    solution[
                        'residual'
                    ]
                )

                result.min_joint_margin = float(
                    solution[
                        'min_margin'
                    ]
                )

                result.status = int(
                    status
                )

                result.reason = (
                    reason
                )

                result.frozen_base_assumption = (
                    True
                )

                output.candidates.append(
                    result
                )

        self.pub.publish(
            output
        )

        self.counter += 1

        if self.counter % 20 == 0:

            self.log_summary(
                output
            )

    def log_summary(
        self,
        msg
    ):

        reachable = 0
        marginal = 0
        unreachable = 0

        for item in msg.candidates:

            if (
                item.status
                ==
                FourDofFeasibility.REACHABLE
            ):
                reachable += 1

            elif (
                item.status
                ==
                FourDofFeasibility.MARGINAL
            ):
                marginal += 1

            else:
                unreachable += 1

        self.get_logger().info(
            'Feasibility: '
            f'R={reachable} '
            f'M={marginal} '
            f'U={unreachable}'
        )

        for item in msg.candidates:

            status_name = {
                FourDofFeasibility.REACHABLE:
                    'REACHABLE',

                FourDofFeasibility.MARGINAL:
                    'MARGINAL',

                FourDofFeasibility.UNREACHABLE:
                    'UNREACHABLE',

            }.get(
                item.status,
                'UNKNOWN'
            )

            self.get_logger().info(
                f'  '
                f'{item.track_id:<10} '
                f'{item.class_id:<8} '
                f't+{item.horizon:.1f}s '
                f'{status_name:<11} '
                f'residual='
                f'{item.position_residual*1000:.1f}mm '
                f'margin='
                f'{item.min_joint_margin:.3f}rad'
            )


def main(args=None):

    rclpy.init(
        args=args
    )

    node = (
        FourDofFeasibilityNode()
    )

    try:

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.try_shutdown()


if __name__ == '__main__':

    main()
