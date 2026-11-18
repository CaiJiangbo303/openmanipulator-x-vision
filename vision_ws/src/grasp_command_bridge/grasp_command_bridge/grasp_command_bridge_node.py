import copy
import math

import rclpy
from rclpy.node import Node
from rclpy.clock import Clock, ClockType

from std_msgs.msg import String
from target_selector_msgs.msg import TargetSelection


class GraspCommandBridge(Node):

    def __init__(self):
        super().__init__('grasp_command_bridge')

        # --------------------------------------------------
        # Topics / mode
        # --------------------------------------------------

        self.declare_parameter(
            'selection_topic',
            '/target_selector/selection'
        )

        self.declare_parameter(
            'status_topic',
            '/grasp_command_bridge/status'
        )

        self.declare_parameter(
            'world_frame',
            'world'
        )

        self.declare_parameter(
            'dry_run',
            True
        )

        # --------------------------------------------------
        # Safety gates
        # --------------------------------------------------

        self.declare_parameter(
            'max_message_age',
            0.50
        )

        self.declare_parameter(
            'max_future_skew',
            0.05
        )

        self.declare_parameter(
            'max_horizon',
            0.50
        )

        self.declare_parameter(
            'min_confidence',
            0.50
        )

        self.declare_parameter(
            'max_uncertainty',
            0.030
        )

        # Acquisition remains strict at max_uncertainty.
        # A locked target may briefly enter this small
        # hysteresis band before the lock is released.
        self.declare_parameter(
            'max_uncertainty_hard',
            0.035
        )

        self.declare_parameter(
            'quality_grace_time',
            0.30
        )

        self.declare_parameter(
            'min_joint_margin',
            0.10
        )

        self.declare_parameter(
            'max_position_residual',
            0.005
        )

        self.declare_parameter(
            'max_target_distance',
            0.470
        )

        # --------------------------------------------------
        # Target-lock hysteresis
        # --------------------------------------------------

        self.declare_parameter(
            'lock_dwell_time',
            0.20
        )

        self.declare_parameter(
            'switch_dwell_time',
            0.60
        )

        # Short selector dropouts are tolerated while a
        # target is already locked.
        self.declare_parameter(
            'invalid_grace_time',
            0.50
        )

        # Independent upstream receive watchdog.
        # Uses STEADY_TIME rather than simulation time so
        # an upstream crash or frozen /clock cannot leave
        # an old target locked forever.
        self.declare_parameter(
            'selection_watchdog_timeout',
            0.50
        )

        self.declare_parameter(
            'watchdog_period',
            0.05
        )

        # --------------------------------------------------
        # Read parameters
        # --------------------------------------------------

        self.selection_topic = str(
            self.get_parameter(
                'selection_topic'
            ).value
        )

        self.status_topic = str(
            self.get_parameter(
                'status_topic'
            ).value
        )

        self.world_frame = str(
            self.get_parameter(
                'world_frame'
            ).value
        )

        self.dry_run = bool(
            self.get_parameter(
                'dry_run'
            ).value
        )

        self.max_message_age = float(
            self.get_parameter(
                'max_message_age'
            ).value
        )

        self.max_future_skew = float(
            self.get_parameter(
                'max_future_skew'
            ).value
        )

        self.max_horizon = float(
            self.get_parameter(
                'max_horizon'
            ).value
        )

        self.min_confidence = float(
            self.get_parameter(
                'min_confidence'
            ).value
        )

        self.max_uncertainty = float(
            self.get_parameter(
                'max_uncertainty'
            ).value
        )

        self.max_uncertainty_hard = float(
            self.get_parameter(
                'max_uncertainty_hard'
            ).value
        )

        self.quality_grace_time = float(
            self.get_parameter(
                'quality_grace_time'
            ).value
        )

        self.min_joint_margin = float(
            self.get_parameter(
                'min_joint_margin'
            ).value
        )

        self.max_position_residual = float(
            self.get_parameter(
                'max_position_residual'
            ).value
        )

        self.max_target_distance = float(
            self.get_parameter(
                'max_target_distance'
            ).value
        )

        self.lock_dwell_time = float(
            self.get_parameter(
                'lock_dwell_time'
            ).value
        )

        self.switch_dwell_time = float(
            self.get_parameter(
                'switch_dwell_time'
            ).value
        )

        self.invalid_grace_time = float(
            self.get_parameter(
                'invalid_grace_time'
            ).value
        )

        self.selection_watchdog_timeout = float(
            self.get_parameter(
                'selection_watchdog_timeout'
            ).value
        )

        self.watchdog_period = float(
            self.get_parameter(
                'watchdog_period'
            ).value
        )

        # --------------------------------------------------
        # Verified OpenManipulator-X joint limits
        # --------------------------------------------------

        self.joint_limits = [
            (-math.pi, math.pi),
            (-1.5, 1.5),
            (-1.5, 1.4),
            (-1.7, 1.97),
        ]

        # --------------------------------------------------
        # Lock state
        # --------------------------------------------------

        self.locked_identity = None
        self.locked_message = None

        self.pending_identity = None
        self.pending_since_ns = None

        self.gap_since_ns = None
        self.quality_gap_since_ns = None

        self.last_now_ns = None

        # Receive watchdog uses steady wall time, not /clock.
        self.watchdog_clock = Clock(
            clock_type=ClockType.STEADY_TIME
        )

        self.last_selection_rx_ns = None
        self.watchdog_tripped = False

        # Logging throttle only.
        self.last_log_ns = 0

        # --------------------------------------------------
        # ROS interfaces
        # --------------------------------------------------

        self.status_publisher = (
            self.create_publisher(
                String,
                self.status_topic,
                10
            )
        )

        self.selection_subscription = (
            self.create_subscription(
                TargetSelection,
                self.selection_topic,
                self.selection_callback,
                10
            )
        )

        self.watchdog_timer = (
            self.create_timer(
                self.watchdog_period,
                self.watchdog_callback,
                clock=self.watchdog_clock
            )
        )

        self.get_logger().info(
            'Grasp Command Bridge started'
        )

        self.get_logger().info(
            f'Input: {self.selection_topic}'
        )

        self.get_logger().info(
            f'Status: {self.status_topic}'
        )

        if self.dry_run:
            self.get_logger().info(
                'Mode: DRY-RUN ONLY'
            )
        else:
            self.get_logger().error(
                'dry_run=false requested, but this '
                'version intentionally has NO robot '
                'command publisher'
            )

        self.get_logger().info(
            'No robot command publisher exists '
            'in this version'
        )

        self.get_logger().info(
            'Target lock: '
            f'initial={self.lock_dwell_time:.2f}s, '
            f'switch={self.switch_dwell_time:.2f}s, '
            f'invalid_grace={self.invalid_grace_time:.2f}s'
        )

        self.get_logger().info(
            'Selection watchdog: '
            f'{self.selection_watchdog_timeout:.2f}s '
            '(STEADY_TIME)'
        )

    # ======================================================
    # Helpers
    # ======================================================

    @staticmethod
    def is_finite(value):
        return math.isfinite(
            float(value)
        )

    @staticmethod
    def selection_identity(msg):
        return (
            msg.track_id,
            msg.class_id,
        )

    @staticmethod
    def identity_string(identity):

        if identity is None:
            return '<none>'

        return (
            f'{identity[0]}/'
            f'{identity[1]}'
        )

    def clear_target_lock(self):

        self.locked_identity = None
        self.locked_message = None

        self.pending_identity = None
        self.pending_since_ns = None

        self.gap_since_ns = None
        self.quality_gap_since_ns = None

    def start_pending_target(
        self,
        identity,
        now_ns
    ):

        self.pending_identity = identity
        self.pending_since_ns = now_ns

    def reset_pending_target(self):

        self.pending_identity = None
        self.pending_since_ns = None

    # ======================================================
    # Safety validation
    # ======================================================

    def validate_selection(
        self,
        msg,
        uncertainty_limit=None
    ):

        if uncertainty_limit is None:
            uncertainty_limit = self.max_uncertainty

        if not msg.valid:
            return (
                False,
                'selector marked candidate invalid'
            )

        if not msg.track_id:
            return (
                False,
                'empty track_id'
            )

        if not msg.class_id:
            return (
                False,
                'empty class_id'
            )

        if msg.header.frame_id != self.world_frame:
            return (
                False,
                'unexpected frame '
                f'{msg.header.frame_id}; '
                f'expected {self.world_frame}'
            )

        stamp_ns = (
            int(msg.header.stamp.sec)
            * 1_000_000_000
            +
            int(msg.header.stamp.nanosec)
        )

        if stamp_ns <= 0:
            return (
                False,
                'invalid timestamp'
            )

        now_ns = (
            self.get_clock()
            .now()
            .nanoseconds
        )

        age = (
            now_ns - stamp_ns
        ) * 1e-9

        if age > self.max_message_age:
            return (
                False,
                f'message age={age:.3f}s > '
                f'{self.max_message_age:.3f}s'
            )

        if age < -self.max_future_skew:
            return (
                False,
                f'message is {-age:.3f}s '
                'in the future'
            )

        if (
            not self.is_finite(
                msg.horizon
            )
            or
            msg.horizon < 0.0
            or
            msg.horizon > self.max_horizon
        ):
            return (
                False,
                f'invalid horizon={msg.horizon:.3f}s'
            )

        xyz = [
            msg.target_world.x,
            msg.target_world.y,
            msg.target_world.z,
            msg.target_arm.x,
            msg.target_arm.y,
            msg.target_arm.z,
        ]

        if not all(
            self.is_finite(v)
            for v in xyz
        ):
            return (
                False,
                'non-finite target position'
            )

        arm_distance = math.sqrt(
            msg.target_arm.x ** 2
            +
            msg.target_arm.y ** 2
            +
            msg.target_arm.z ** 2
        )

        if (
            arm_distance
            >
            self.max_target_distance
        ):
            return (
                False,
                f'target distance='
                f'{arm_distance:.4f}m > '
                f'{self.max_target_distance:.4f}m'
            )

        if len(msg.joint_positions) != 4:
            return (
                False,
                'expected exactly 4 '
                'joint positions'
            )

        for i, q in enumerate(
            msg.joint_positions
        ):

            if not self.is_finite(q):
                return (
                    False,
                    f'joint {i + 1} is non-finite'
                )

            lower, upper = (
                self.joint_limits[i]
            )

            if q < lower or q > upper:
                return (
                    False,
                    f'joint {i + 1}='
                    f'{q:.4f} outside '
                    f'[{lower:.4f}, {upper:.4f}]'
                )

        metrics = [
            msg.score,
            msg.confidence,
            msg.uncertainty,
            msg.min_joint_margin,
            msg.position_residual,
        ]

        if not all(
            self.is_finite(v)
            for v in metrics
        ):
            return (
                False,
                'non-finite quality metric'
            )

        if (
            msg.confidence
            <
            self.min_confidence
        ):
            return (
                False,
                f'confidence='
                f'{msg.confidence:.3f} < '
                f'{self.min_confidence:.3f}'
            )

        if (
            msg.uncertainty
            >
            uncertainty_limit
        ):
            return (
                False,
                f'uncertainty='
                f'{msg.uncertainty:.4f}m > '
                f'{uncertainty_limit:.4f}m'
            )

        if (
            msg.min_joint_margin
            <
            self.min_joint_margin
        ):
            return (
                False,
                f'joint margin='
                f'{msg.min_joint_margin:.4f} < '
                f'{self.min_joint_margin:.4f}'
            )

        if (
            msg.position_residual
            >
            self.max_position_residual
        ):
            return (
                False,
                f'position residual='
                f'{msg.position_residual:.6f}m > '
                f'{self.max_position_residual:.6f}m'
            )

        return (
            True,
            'all safety gates passed'
        )

    # ======================================================
    # HOLD / release
    # ======================================================

    def hold_or_release(
        self,
        now_ns,
        reason
    ):

        # A gap must break challenger continuity.
        self.reset_pending_target()

        if self.gap_since_ns is None:

            self.gap_since_ns = now_ns

        # Handle simulation-time reset safely.
        if now_ns < self.gap_since_ns:

            self.gap_since_ns = now_ns

        gap = (
            now_ns
            -
            self.gap_since_ns
        ) * 1e-9

        if (
            self.locked_message is not None
            and
            gap
            <
            self.invalid_grace_time
        ):

            return (
                'HOLD',
                self.locked_message,
                f'{reason}; '
                f'holding '
                f'{self.identity_string(self.locked_identity)} '
                f'{gap:.3f}/'
                f'{self.invalid_grace_time:.3f}s'
            )

        old_identity = (
            self.locked_identity
        )

        self.clear_target_lock()

        return (
            'REJECT',
            None,
            f'lock released '
            f'{self.identity_string(old_identity)}; '
            f'{reason}; '
            f'grace expired'
        )

    # ======================================================
    # Target-lock state machine
    # ======================================================

    def process_selection(
        self,
        msg
    ):

        now_ns = (
            self.get_clock()
            .now()
            .nanoseconds
        )

        # Simulation time moved backwards.
        # Never carry a lock across that discontinuity.
        if (
            self.last_now_ns is not None
            and
            now_ns < self.last_now_ns
        ):

            self.clear_target_lock()

        self.last_now_ns = now_ns

        # --------------------------------------------------
        # UNLOCKED
        # --------------------------------------------------

        if self.locked_identity is None:

            if not msg.valid:

                self.reset_pending_target()

                return (
                    'REJECT',
                    None,
                    'selector marked candidate invalid'
                )

            safe, reason = (
                self.validate_selection(msg)
            )

            if not safe:

                self.reset_pending_target()

                return (
                    'REJECT',
                    msg,
                    reason
                )

            identity = (
                self.selection_identity(msg)
            )

            if (
                self.pending_identity
                !=
                identity
                or
                self.pending_since_ns is None
                or
                now_ns
                <
                self.pending_since_ns
            ):

                self.start_pending_target(
                    identity,
                    now_ns
                )

                return (
                    'REJECT',
                    msg,
                    'target lock pending '
                    f'{self.identity_string(identity)} '
                    f'0.000/'
                    f'{self.lock_dwell_time:.3f}s'
                )

            dwell = (
                now_ns
                -
                self.pending_since_ns
            ) * 1e-9

            if dwell < self.lock_dwell_time:

                return (
                    'REJECT',
                    msg,
                    'target lock pending '
                    f'{self.identity_string(identity)} '
                    f'{dwell:.3f}/'
                    f'{self.lock_dwell_time:.3f}s'
                )

            self.locked_identity = identity

            self.locked_message = (
                copy.deepcopy(msg)
            )

            self.reset_pending_target()
            self.gap_since_ns = None

            return (
                'ACCEPT',
                self.locked_message,
                'target locked; '
                'all safety gates passed'
            )

        # --------------------------------------------------
        # Already LOCKED
        # --------------------------------------------------

        # Selector itself temporarily has no reachable target.
        if not msg.valid:

            return self.hold_or_release(
                now_ns,
                'selector temporarily invalid'
            )

        identity = (
            self.selection_identity(msg)
        )

        # --------------------------------------------------
        # Current selector agrees with lock
        # --------------------------------------------------

        if identity == self.locked_identity:

            safe, reason = (
                self.validate_selection(msg)
            )

            # Normal strict-quality refresh.
            if safe:

                self.gap_since_ns = None
                self.quality_gap_since_ns = None

                self.locked_message = (
                    copy.deepcopy(msg)
                )

                self.reset_pending_target()

                return (
                    'ACCEPT',
                    self.locked_message,
                    'locked target refreshed; '
                    'all safety gates passed'
                )

            # Re-check using ONLY the small uncertainty
            # hysteresis band. Every other safety gate is
            # still identical and therefore remains hard.
            hard_safe, hard_reason = (
                self.validate_selection(
                    msg,
                    uncertainty_limit=
                    self.max_uncertainty_hard
                )
            )

            if (
                hard_safe
                and
                msg.uncertainty
                >
                self.max_uncertainty
            ):

                if self.quality_gap_since_ns is None:
                    self.quality_gap_since_ns = now_ns

                if now_ns < self.quality_gap_since_ns:
                    self.quality_gap_since_ns = now_ns

                quality_gap = (
                    now_ns
                    -
                    self.quality_gap_since_ns
                ) * 1e-9

                self.reset_pending_target()
                self.gap_since_ns = None

                if (
                    quality_gap
                    <
                    self.quality_grace_time
                ):

                    return (
                        'HOLD',
                        self.locked_message,
                        'locked target uncertainty grace '
                        f'current={msg.uncertainty:.4f}m '
                        f'strict={self.max_uncertainty:.4f}m '
                        f'hard={self.max_uncertainty_hard:.4f}m '
                        f'{quality_gap:.3f}/'
                        f'{self.quality_grace_time:.3f}s'
                    )

                old_identity = (
                    self.locked_identity
                )

                self.clear_target_lock()

                return (
                    'REJECT',
                    None,
                    'lock released '
                    f'{self.identity_string(old_identity)}; '
                    'uncertainty remained above strict '
                    f'limit for {quality_gap:.3f}s'
                )

            # Hard failure: timestamp, reach, joints,
            # residual, confidence, or uncertainty beyond
            # max_uncertainty_hard.
            old_identity = (
                self.locked_identity
            )

            self.clear_target_lock()

            return (
                'REJECT',
                None,
                'locked target failed hard safety gate '
                f'{self.identity_string(old_identity)}; '
                f'{hard_reason}'
            )

        # --------------------------------------------------
        # Challenger
        # --------------------------------------------------

        self.quality_gap_since_ns = None

        safe, reason = (
            self.validate_selection(msg)
        )

        if not safe:

            return self.hold_or_release(
                now_ns,
                'challenger failed safety gate; '
                f'{reason}'
            )

        # A safe selector result means the perception /
        # decision chain is alive.
        self.gap_since_ns = None

        # --------------------------------------------------
        # Safe challenger
        # --------------------------------------------------

        if (
            self.pending_identity
            !=
            identity
            or
            self.pending_since_ns is None
            or
            now_ns
            <
            self.pending_since_ns
        ):

            self.start_pending_target(
                identity,
                now_ns
            )

            return (
                'HOLD',
                self.locked_message,
                'target switch pending '
                f'locked='
                f'{self.identity_string(self.locked_identity)} '
                f'candidate='
                f'{self.identity_string(identity)} '
                f'0.000/'
                f'{self.switch_dwell_time:.3f}s'
            )

        dwell = (
            now_ns
            -
            self.pending_since_ns
        ) * 1e-9

        if dwell < self.switch_dwell_time:

            return (
                'HOLD',
                self.locked_message,
                'target switch pending '
                f'locked='
                f'{self.identity_string(self.locked_identity)} '
                f'candidate='
                f'{self.identity_string(identity)} '
                f'{dwell:.3f}/'
                f'{self.switch_dwell_time:.3f}s'
            )

        old_identity = (
            self.locked_identity
        )

        self.locked_identity = identity

        self.locked_message = (
            copy.deepcopy(msg)
        )

        self.reset_pending_target()
        self.gap_since_ns = None

        return (
            'ACCEPT',
            self.locked_message,
            'target lock switched '
            f'{self.identity_string(old_identity)}'
            ' -> '
            f'{self.identity_string(identity)}'
        )

    # ======================================================
    # Status / logging
    # ======================================================

    def publish_decision(
        self,
        decision,
        msg,
        reason
    ):

        if msg is None:

            track_id = ''
            class_id = ''
            horizon = 0.0

        else:

            track_id = msg.track_id
            class_id = msg.class_id
            horizon = float(
                msg.horizon
            )

        status = String()

        status.data = (
            f'{decision} '
            f'track={track_id} '
            f'class={class_id} '
            f'horizon={horizon:.3f} '
            f'reason={reason}'
        )

        self.status_publisher.publish(
            status
        )

        # Avoid flooding logs while still publishing every
        # status message on the ROS topic.
        now_ns = (
            self.get_clock()
            .now()
            .nanoseconds
        )

        major_event = (
            'target locked;' in reason
            or
            'target lock switched' in reason
            or
            'lock released' in reason
            or
            'failed safety gate' in reason
        )

        should_log = (
            major_event
            or
            now_ns < self.last_log_ns
            or
            (
                now_ns
                -
                self.last_log_ns
            )
            >=
            500_000_000
        )

        if not should_log:
            return

        self.last_log_ns = now_ns

        if msg is None:

            detail = (
                '<none> <none>'
            )

        else:

            q = list(
                msg.joint_positions
            )

            if len(q) == 4:

                detail = (
                    f'{track_id} {class_id} '
                    f'| h={horizon:.2f}s '
                    f'| q=['
                    f'{q[0]:+.4f}, '
                    f'{q[1]:+.4f}, '
                    f'{q[2]:+.4f}, '
                    f'{q[3]:+.4f}] '
                    f'| margin='
                    f'{msg.min_joint_margin:.3f} '
                    f'| residual='
                    f'{msg.position_residual * 1000.0:.3f}mm'
                )

            else:

                detail = (
                    f'{track_id} {class_id}'
                )

        line = (
            f'DRY-RUN {decision} '
            f'| {detail} '
            f'| {reason}'
        )

        if decision == 'REJECT':

            self.get_logger().warn(
                line
            )

        else:

            self.get_logger().info(
                line
            )

    # ======================================================
    # Upstream receive watchdog
    # ======================================================

    def watchdog_callback(self):

        if self.last_selection_rx_ns is None:
            return

        now_ns = (
            self.watchdog_clock
            .now()
            .nanoseconds
        )

        # Steady time should not move backwards, but reset
        # defensively if that ever occurs.
        if now_ns < self.last_selection_rx_ns:

            self.last_selection_rx_ns = now_ns
            self.watchdog_tripped = False
            return

        silence = (
            now_ns
            -
            self.last_selection_rx_ns
        ) * 1e-9

        if (
            silence
            <=
            self.selection_watchdog_timeout
        ):
            return

        if self.watchdog_tripped:
            return

        self.watchdog_tripped = True

        if self.locked_identity is None:
            return

        old_identity = (
            self.locked_identity
        )

        self.clear_target_lock()

        self.publish_decision(
            'REJECT',
            None,
            'selection stream timeout; '
            f'released '
            f'{self.identity_string(old_identity)} '
            f'after {silence:.3f}s without input'
        )

    # ======================================================
    # Subscription callback
    # ======================================================

    def selection_callback(
        self,
        msg
    ):

        self.last_selection_rx_ns = (
            self.watchdog_clock
            .now()
            .nanoseconds
        )

        self.watchdog_tripped = False

        decision, output_msg, reason = (
            self.process_selection(
                msg
            )
        )

        self.publish_decision(
            decision,
            output_msg,
            reason
        )


def main(args=None):

    rclpy.init(args=args)

    node = GraspCommandBridge()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
