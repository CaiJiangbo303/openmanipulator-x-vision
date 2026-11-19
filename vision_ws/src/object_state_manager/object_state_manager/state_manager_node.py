#!/usr/bin/env python3

from collections import deque
from dataclasses import dataclass, field

import numpy as np

import rclpy

from rclpy.node import Node
from rclpy.time import Time

from vision_msgs.msg import Detection3DArray

from object_state_msgs.msg import (
    ObjectState,
    ObjectStateArray,
)


@dataclass
class TrackState:

    track_id: str
    class_id: str

    position: np.ndarray
    velocity: np.ndarray

    variance: np.ndarray

    confidence: float

    last_stamp: Time

    age: int = 1
    missed: int = 0

    measurement_source: str = 'unknown'
    measurement_source_type: int = 0

    # Slowly varying correction from precise mask depth
    # relative to Fast3D bbox depth.
    precise_bias: np.ndarray = field(
        default_factory=lambda: np.zeros(
            3,
            dtype=np.float64
        )
    )

    precise_bias_valid: bool = False
    last_precise_stamp_ns: int = -1


class ObjectStateManager(Node):

    def __init__(self):

        super().__init__(
            'object_state_manager'
        )

        # -------------------------------------------------
        # Parameters
        # -------------------------------------------------

        self.declare_parameter(
            'fast_measurement_topic',
            '/object_localizer/detections_3d_fast'
        )

        self.declare_parameter(
            'precise_measurement_topic',
            '/object_self_filter/detections_3d'
        )

        self.declare_parameter(
            'fast_history_size',
            32
        )

        self.declare_parameter(
            'precise_match_tolerance_ms',
            20.0
        )

        self.declare_parameter(
            'precise_bias_alpha',
            0.50
        )

        self.declare_parameter(
            'precise_bias_timeout',
            1.50
        )

        self.declare_parameter(
            'stale_timeout',
            0.50
        )

        self.declare_parameter(
            'output_topic',
            '/object_state_manager/states'
        )

        self.declare_parameter(
            'world_frame',
            'world'
        )

        self.declare_parameter(
            'publish_rate_hz',
            20.0
        )

        self.declare_parameter(
            'alpha',
            0.85
        )

        self.declare_parameter(
            'beta',
            0.35
        )

        self.declare_parameter(
            'uncertainty_alpha',
            0.15
        )

        self.declare_parameter(
            'initial_position_std',
            0.010
        )

        self.declare_parameter(
            'minimum_position_std',
            0.002
        )

        self.declare_parameter(
            'uncertainty_growth_rate',
            0.010
        )

        self.declare_parameter(
            'lost_timeout',
            1.5
        )

        self.declare_parameter(
            'remove_timeout',
            3.0
        )

        self.declare_parameter(
            'max_prediction_time',
            1.00
        )

        self.fast_measurement_topic = (
            self.get_parameter(
                'fast_measurement_topic'
            ).value
        )

        self.precise_measurement_topic = (
            self.get_parameter(
                'precise_measurement_topic'
            ).value
        )

        self.fast_history_size = max(
            4,
            int(
                self.get_parameter(
                    'fast_history_size'
                ).value
            )
        )

        self.precise_match_tolerance_ns = int(
            max(
                0.0,
                float(
                    self.get_parameter(
                        'precise_match_tolerance_ms'
                    ).value
                )
            )
            * 1_000_000.0
        )

        self.precise_bias_alpha = max(
            0.0,
            min(
                1.0,
                float(
                    self.get_parameter(
                        'precise_bias_alpha'
                    ).value
                )
            )
        )

        self.precise_bias_timeout = max(
            0.0,
            float(
                self.get_parameter(
                    'precise_bias_timeout'
                ).value
            )
        )

        self.stale_timeout = max(
            0.0,
            float(
                self.get_parameter(
                    'stale_timeout'
                ).value
            )
        )

        self.output_topic = (
            self.get_parameter(
                'output_topic'
            ).value
        )

        self.world_frame = (
            self.get_parameter(
                'world_frame'
            ).value
        )

        self.publish_rate = float(
            self.get_parameter(
                'publish_rate_hz'
            ).value
        )

        self.alpha = float(
            self.get_parameter(
                'alpha'
            ).value
        )

        self.beta = float(
            self.get_parameter(
                'beta'
            ).value
        )

        self.uncertainty_alpha = float(
            self.get_parameter(
                'uncertainty_alpha'
            ).value
        )

        self.initial_std = float(
            self.get_parameter(
                'initial_position_std'
            ).value
        )

        self.minimum_std = float(
            self.get_parameter(
                'minimum_position_std'
            ).value
        )

        self.uncertainty_growth_rate = float(
            self.get_parameter(
                'uncertainty_growth_rate'
            ).value
        )

        self.lost_timeout = float(
            self.get_parameter(
                'lost_timeout'
            ).value
        )

        self.remove_timeout = float(
            self.get_parameter(
                'remove_timeout'
            ).value
        )

        self.max_prediction_time = float(
            self.get_parameter(
                'max_prediction_time'
            ).value
        )

        # -------------------------------------------------
        # State database
        # -------------------------------------------------

        self.tracks = {}

        # Raw Fast3D measurements keyed by tracker ID.
        # These are used only to time-align delayed precise
        # measurements. They are bounded per track.
        self.fast_history = {}

        # -------------------------------------------------
        # ROS interfaces
        # -------------------------------------------------

        self.fast_measurement_sub = (
            self.create_subscription(
                Detection3DArray,
                self.fast_measurement_topic,
                self.fast_measurement_callback,
                10
            )
        )

        self.precise_measurement_sub = (
            self.create_subscription(
                Detection3DArray,
                self.precise_measurement_topic,
                self.precise_measurement_callback,
                10
            )
        )

        self.state_pub = (
            self.create_publisher(
                ObjectStateArray,
                self.output_topic,
                10
            )
        )

        timer_period = (
            1.0 / self.publish_rate
        )

        self.timer = (
            self.create_timer(
                timer_period,
                self.publish_states
            )
        )

        self.publish_counter = 0

        self.get_logger().info(
            'Object State Manager started'
        )

        self.get_logger().info(
            f'Fast measurement: '
            f'{self.fast_measurement_topic}'
        )

        self.get_logger().info(
            f'Precise measurement: '
            f'{self.precise_measurement_topic}'
        )

        self.get_logger().info(
            f'Output: '
            f'{self.output_topic}'
        )

        self.get_logger().info(
            f'Publish rate: '
            f'{self.publish_rate:.1f} Hz'
        )

    # -----------------------------------------------------
    # Measurement parsing / fusion
    # -----------------------------------------------------

    def parse_detection(
        self,
        detection
    ):

        if not detection.id:
            return None

        if not detection.results:
            return None

        result = detection.results[0]

        p = result.pose.pose.position

        measurement = np.array(
            [
                p.x,
                p.y,
                p.z,
            ],
            dtype=np.float64
        )

        if not np.all(
            np.isfinite(
                measurement
            )
        ):
            return None

        return (
            detection.id,
            result.hypothesis.class_id,
            float(
                result.hypothesis.score
            ),
            measurement,
        )

    # -----------------------------------------------------

    def create_track(
        self,
        track_id,
        class_id,
        measurement,
        confidence,
        stamp,
        source_name,
        source_type
    ):

        variance = np.full(
            3,
            self.initial_std ** 2,
            dtype=np.float64
        )

        state = TrackState(
            track_id=track_id,
            class_id=class_id,
            position=measurement.copy(),
            velocity=np.zeros(
                3,
                dtype=np.float64
            ),
            variance=variance,
            confidence=confidence,
            last_stamp=stamp,
            measurement_source=source_name,
            measurement_source_type=source_type,
        )

        self.tracks[
            track_id
        ] = state

        self.get_logger().info(
            f'NEW {track_id} '
            f'{class_id} '
            f'source={source_name} '
            f'position=('
            f'{measurement[0]:+.4f},'
            f'{measurement[1]:+.4f},'
            f'{measurement[2]:+.4f})'
        )

        return state

    # -----------------------------------------------------

    def update_track(
        self,
        state,
        class_id,
        measurement,
        confidence,
        stamp,
        source_name,
        source_type
    ):

        dt = (
            stamp.nanoseconds
            -
            state.last_stamp.nanoseconds
        ) * 1e-9

        # Freshness > completeness.
        # Never rewind the filter with an old direct
        # measurement.
        if dt < -1e-6:
            return False

        if dt <= 1e-4:

            residual = (
                measurement
                -
                state.position
            )

            state.position = (
                state.position
                +
                self.alpha
                *
                residual
            )

        else:

            predicted = (
                state.position
                +
                state.velocity
                *
                dt
            )

            residual = (
                measurement
                -
                predicted
            )

            state.position = (
                predicted
                +
                self.alpha
                *
                residual
            )

            state.velocity = (
                state.velocity
                +
                (
                    self.beta
                    /
                    dt
                )
                *
                residual
            )

        residual_sq = (
            residual ** 2
        )

        state.variance = (
            (
                1.0
                -
                self.uncertainty_alpha
            )
            *
            state.variance
            +
            self.uncertainty_alpha
            *
            residual_sq
        )

        state.variance = np.maximum(
            state.variance,
            self.minimum_std ** 2
        )

        state.class_id = class_id

        state.confidence = (
            0.8
            *
            state.confidence
            +
            0.2
            *
            confidence
        )

        state.last_stamp = stamp
        state.age += 1
        state.missed = 0

        state.measurement_source = (
            source_name
        )

        state.measurement_source_type = int(
            source_type
        )

        return True

    # -----------------------------------------------------

    def remember_fast_measurement(
        self,
        track_id,
        stamp,
        measurement
    ):

        history = self.fast_history.get(
            track_id
        )

        if history is None:

            history = deque(
                maxlen=self.fast_history_size
            )

            self.fast_history[
                track_id
            ] = history

        history.append(
            (
                stamp.nanoseconds,
                measurement.copy(),
            )
        )

    # -----------------------------------------------------

    def find_matching_fast(
        self,
        track_id,
        precise_stamp
    ):

        history = self.fast_history.get(
            track_id
        )

        if not history:
            return None

        target_ns = (
            precise_stamp.nanoseconds
        )

        best = None
        best_error = None

        for stamp_ns, measurement in history:

            error = abs(
                stamp_ns
                -
                target_ns
            )

            if (
                best_error is None
                or
                error < best_error
            ):

                best_error = error

                best = (
                    stamp_ns,
                    measurement
                )

                if error == 0:
                    break

        if (
            best is None
            or
            best_error
            >
            self.precise_match_tolerance_ns
        ):
            return None

        return best

    # -----------------------------------------------------
    # Fast path
    # -----------------------------------------------------

    def fast_measurement_callback(
        self,
        msg
    ):

        stamp = Time.from_msg(
            msg.header.stamp
        )

        observed_ids = set()

        for detection in msg.detections:

            parsed = self.parse_detection(
                detection
            )

            if parsed is None:
                continue

            (
                track_id,
                class_id,
                confidence,
                raw_measurement,
            ) = parsed

            observed_ids.add(
                track_id
            )

            self.remember_fast_measurement(
                track_id,
                stamp,
                raw_measurement
            )

            state = self.tracks.get(
                track_id
            )

            if state is None:

                self.create_track(
                    track_id,
                    class_id,
                    raw_measurement,
                    confidence,
                    stamp,
                    'fast_bbox_depth',
                    ObjectState.SOURCE_FAST_BBOX_DEPTH
                )

                continue

            # If a precise observation was already accepted
            # for exactly this source timestamp, do not let
            # the lower-precision Fast3D observation replace
            # it merely because callback order was reversed.
            dt_ns = (
                stamp.nanoseconds
                -
                state.last_stamp.nanoseconds
            )

            if (
                abs(dt_ns) <= 100_000
                and
                state.measurement_source_type
                ==
                ObjectState.SOURCE_PRECISE_MASK_DEPTH
            ):

                state.missed = 0
                continue

            bias_active = (
                state.precise_bias_valid
            )

            if (
                bias_active
                and
                state.last_precise_stamp_ns >= 0
            ):

                bias_age = (
                    stamp.nanoseconds
                    -
                    state.last_precise_stamp_ns
                ) * 1e-9

                if (
                    bias_age
                    >
                    self.precise_bias_timeout
                ):

                    state.precise_bias_valid = False
                    bias_active = False

            if bias_active:

                measurement = (
                    raw_measurement
                    +
                    state.precise_bias
                )

                source_name = (
                    'fused_fast_precise'
                )

                source_type = (
                    ObjectState.SOURCE_FUSED
                )

            else:

                measurement = (
                    raw_measurement
                )

                source_name = (
                    'fast_bbox_depth'
                )

                source_type = (
                    ObjectState.SOURCE_FAST_BBOX_DEPTH
                )

            self.update_track(
                state,
                class_id,
                measurement,
                confidence,
                stamp,
                source_name,
                source_type
            )

        # Missing-count bookkeeping follows the high-rate
        # primary path only. Slow precise callbacks must not
        # artificially inflate missed counts.
        for track_id, state in (
            self.tracks.items()
        ):

            if track_id not in observed_ids:
                state.missed += 1

    # -----------------------------------------------------
    # Precise path
    # -----------------------------------------------------

    def precise_measurement_callback(
        self,
        msg
    ):

        stamp = Time.from_msg(
            msg.header.stamp
        )

        for detection in msg.detections:

            parsed = self.parse_detection(
                detection
            )

            if parsed is None:
                continue

            (
                track_id,
                class_id,
                confidence,
                precise_measurement,
            ) = parsed

            state = self.tracks.get(
                track_id
            )

            # Precise-only degradation mode.
            if state is None:

                state = self.create_track(
                    track_id,
                    class_id,
                    precise_measurement,
                    confidence,
                    stamp,
                    'precise_mask_depth',
                    ObjectState.SOURCE_PRECISE_MASK_DEPTH
                )

                state.last_precise_stamp_ns = (
                    stamp.nanoseconds
                )

                continue

            matched = self.find_matching_fast(
                track_id,
                stamp
            )

            if matched is not None:

                (
                    _,
                    fast_measurement
                ) = matched

                # If the precise frame is extremely delayed
                # relative to the current state, discard it.
                # A numerically correct old correction can
                # still be unsafe on a moving platform.
                lag = (
                    state.last_stamp.nanoseconds
                    -
                    stamp.nanoseconds
                ) * 1e-9

                if (
                    lag
                    >
                    self.precise_bias_timeout
                ):
                    continue

                correction = (
                    precise_measurement
                    -
                    fast_measurement
                )

                old_bias = (
                    state.precise_bias.copy()
                )

                if state.precise_bias_valid:

                    new_bias = (
                        (
                            1.0
                            -
                            self.precise_bias_alpha
                        )
                        *
                        old_bias
                        +
                        self.precise_bias_alpha
                        *
                        correction
                    )

                else:

                    new_bias = (
                        correction.copy()
                    )

                state.precise_bias = (
                    new_bias
                )

                state.precise_bias_valid = True

                state.last_precise_stamp_ns = (
                    stamp.nanoseconds
                )

                # Apply only the change in calibration to
                # the current filter state. Do not rewind
                # state.last_stamp to the delayed precise
                # frame.
                state.position = (
                    state.position
                    +
                    (
                        new_bias
                        -
                        old_bias
                    )
                )

                state.class_id = class_id

                state.confidence = (
                    0.8
                    *
                    state.confidence
                    +
                    0.2
                    *
                    confidence
                )

                state.age += 1
                state.missed = 0

                state.measurement_source = (
                    'fused_fast_precise'
                )

                state.measurement_source_type = (
                    ObjectState.SOURCE_FUSED
                )

                continue

            # No corresponding Fast frame exists. A precise
            # measurement may still update the state if it
            # is newer. Old unmatched precise frames are
            # intentionally rejected.
            accepted = self.update_track(
                state,
                class_id,
                precise_measurement,
                confidence,
                stamp,
                'precise_mask_depth',
                ObjectState.SOURCE_PRECISE_MASK_DEPTH
            )

            if accepted:

                state.last_precise_stamp_ns = (
                    stamp.nanoseconds
                )

    # -----------------------------------------------------
    # State publication / prediction
    # -----------------------------------------------------

    def publish_states(self):

        now = self.get_clock().now()

        output = ObjectStateArray()

        output.header.stamp = (
            now.to_msg()
        )

        output.header.frame_id = (
            self.world_frame
        )

        remove_ids = []

        for track_id, state in (
            sorted(
                self.tracks.items()
            )
        ):

            since_seen = (
                now.nanoseconds
                -
                state.last_stamp.nanoseconds
            ) * 1e-9

            if since_seen < 0.0:
                since_seen = 0.0

            # ---------------------------------------------
            # Lifecycle
            # ---------------------------------------------

            if since_seen > self.remove_timeout:

                remove_ids.append(
                    track_id
                )

                continue

            if since_seen > self.lost_timeout:

                tracking_state = (
                    ObjectState.LOST
                )

            else:

                tracking_state = (
                    ObjectState.TRACKED
                )

            # ---------------------------------------------
            # Predict between low-rate SAM2 measurements.
            # ---------------------------------------------

            prediction_time = min(
                since_seen,
                self.max_prediction_time
            )

            predicted_position = (
                state.position
                +
                state.velocity
                *
                prediction_time
            )

            # Uncertainty increases while no new
            # measurement arrives.

            base_std = np.sqrt(
                state.variance
            )

            predicted_std = (
                base_std
                +
                self.uncertainty_growth_rate
                *
                prediction_time
            )

            msg = ObjectState()

            msg.track_id = (
                state.track_id
            )

            msg.class_id = (
                state.class_id
            )

            msg.position.x = float(
                predicted_position[0]
            )

            msg.position.y = float(
                predicted_position[1]
            )

            msg.position.z = float(
                predicted_position[2]
            )

            msg.velocity.x = float(
                state.velocity[0]
            )

            msg.velocity.y = float(
                state.velocity[1]
            )

            msg.velocity.z = float(
                state.velocity[2]
            )

            # We currently observe position only.
            # Do NOT pretend orientation is measured.

            msg.orientation.x = 0.0
            msg.orientation.y = 0.0
            msg.orientation.z = 0.0
            msg.orientation.w = 1.0

            msg.orientation_observable = False

            msg.position_std.x = float(
                predicted_std[0]
            )

            msg.position_std.y = float(
                predicted_std[1]
            )

            msg.position_std.z = float(
                predicted_std[2]
            )

            msg.confidence = float(
                state.confidence
            )

            msg.last_seen = (
                state.last_stamp.to_msg()
            )

            msg.source_stamp = (
                state.last_stamp.to_msg()
            )

            msg.measurement_age_ms = float(
                since_seen
                *
                1000.0
            )

            msg.measurement_source_type = int(
                state.measurement_source_type
            )

            msg.position_predicted = bool(
                prediction_time
                >
                1e-6
            )

            msg.valid_3d = bool(
                np.all(
                    np.isfinite(
                        predicted_position
                    )
                )
            )

            msg.stale = bool(
                since_seen
                >
                self.stale_timeout
            )

            msg.age = int(
                state.age
            )

            msg.missed = int(
                state.missed
            )

            msg.state = int(
                tracking_state
            )

            msg.measurement_source = (
                state.measurement_source
            )

            output.states.append(
                msg
            )

        for track_id in remove_ids:

            self.get_logger().info(
                f'REMOVE {track_id}'
            )

            del self.tracks[
                track_id
            ]

            self.fast_history.pop(
                track_id,
                None
            )

        self.state_pub.publish(
            output
        )

        self.publish_counter += 1

        # Roughly once per second.
        if (
            self.publish_counter
            %
            max(
                1,
                int(
                    self.publish_rate
                )
            )
            ==
            0
        ):

            self.log_state_summary(
                output
            )

    # -----------------------------------------------------
    # Debug summary
    # -----------------------------------------------------

    def log_state_summary(
        self,
        msg
    ):

        self.get_logger().info(
            f'Object states: '
            f'{len(msg.states)}'
        )

        for state in msg.states:

            speed = (
                state.velocity.x ** 2
                +
                state.velocity.y ** 2
                +
                state.velocity.z ** 2
            ) ** 0.5

            state_name = (
                'TRACKED'
                if state.state
                ==
                ObjectState.TRACKED
                else
                'LOST'
            )

            self.get_logger().info(
                f'  {state.track_id:<10} '
                f'{state.class_id:<10} '
                f'{state_name:<7} '
                f'p=('
                f'{state.position.x:+.4f},'
                f'{state.position.y:+.4f},'
                f'{state.position.z:+.4f}) '
                f'v={speed:.4f}m/s '
                f'std=('
                f'{state.position_std.x:.4f},'
                f'{state.position_std.y:.4f},'
                f'{state.position_std.z:.4f})'
            )


def main(args=None):

    rclpy.init(
        args=args
    )

    node = (
        ObjectStateManager()
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
