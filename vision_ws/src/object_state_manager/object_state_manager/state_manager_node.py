#!/usr/bin/env python3

from dataclasses import dataclass

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

    measurement_source: str = 'mask_rgbd'


class ObjectStateManager(Node):

    def __init__(self):

        super().__init__(
            'object_state_manager'
        )

        # -------------------------------------------------
        # Parameters
        # -------------------------------------------------

        self.declare_parameter(
            'measurement_topic',
            '/object_localizer/detections_3d_mask'
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
            0.50
        )

        self.measurement_topic = (
            self.get_parameter(
                'measurement_topic'
            ).value
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

        # -------------------------------------------------
        # ROS interfaces
        # -------------------------------------------------

        self.measurement_sub = (
            self.create_subscription(
                Detection3DArray,
                self.measurement_topic,
                self.measurement_callback,
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
            f'Measurement: '
            f'{self.measurement_topic}'
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
    # Measurement update
    # -----------------------------------------------------

    def measurement_callback(
        self,
        msg
    ):

        stamp = Time.from_msg(
            msg.header.stamp
        )

        observed_ids = set()

        for detection in msg.detections:

            if not detection.id:
                continue

            if not detection.results:
                continue

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
                continue

            track_id = detection.id

            class_id = (
                result.hypothesis.class_id
            )

            confidence = float(
                result.hypothesis.score
            )

            observed_ids.add(
                track_id
            )

            # ---------------------------------------------
            # New object
            # ---------------------------------------------

            if track_id not in self.tracks:

                variance = np.full(
                    3,
                    self.initial_std ** 2,
                    dtype=np.float64
                )

                self.tracks[
                    track_id
                ] = TrackState(
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
                )

                self.get_logger().info(
                    f'NEW {track_id} '
                    f'{class_id} '
                    f'position=('
                    f'{measurement[0]:+.4f},'
                    f'{measurement[1]:+.4f},'
                    f'{measurement[2]:+.4f})'
                )

                continue

            # ---------------------------------------------
            # Existing object
            # ---------------------------------------------

            state = self.tracks[
                track_id
            ]

            dt = (
                stamp.nanoseconds
                -
                state.last_stamp.nanoseconds
            ) * 1e-9

            # Ignore old measurements.
            if dt < -1e-6:
                continue

            # Same timestamp:
            # accept position correction but do not
            # estimate velocity.
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

                # -----------------------------------------
                # Alpha-beta constant velocity filter
                # -----------------------------------------

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

            # ---------------------------------------------
            # Heuristic uncertainty estimate.
            #
            # This is NOT yet a calibrated covariance.
            # It measures recent position innovation.
            # ---------------------------------------------

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

            min_variance = (
                self.minimum_std ** 2
            )

            state.variance = np.maximum(
                state.variance,
                min_variance
            )

            state.class_id = (
                class_id
            )

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

        # ---------------------------------------------
        # Tracks that were absent from this measurement
        # ---------------------------------------------

        for track_id, state in (
            self.tracks.items()
        ):

            if track_id not in observed_ids:
                state.missed += 1

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
