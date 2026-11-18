#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, Vector3

from object_state_msgs.msg import (
    ObjectState,
    ObjectStateArray,
    ObjectPrediction,
    ObjectPredictionArray,
)


class ObjectMotionPredictor(Node):

    def __init__(self):
        super().__init__('object_motion_predictor')

        self.declare_parameter(
            'input_topic',
            '/object_state_manager/states'
        )

        self.declare_parameter(
            'output_topic',
            '/object_motion_predictor/predictions'
        )

        self.declare_parameter(
            'horizons',
            [0.2, 0.5, 1.0]
        )

        self.declare_parameter(
            'uncertainty_growth_per_sec',
            0.015
        )

        self.declare_parameter(
            'max_speed',
            1.0
        )

        self.input_topic = self.get_parameter(
            'input_topic'
        ).value

        self.output_topic = self.get_parameter(
            'output_topic'
        ).value

        self.horizons = [
            float(v)
            for v in self.get_parameter(
                'horizons'
            ).value
        ]

        self.uncertainty_growth = float(
            self.get_parameter(
                'uncertainty_growth_per_sec'
            ).value
        )

        self.max_speed = float(
            self.get_parameter(
                'max_speed'
            ).value
        )

        self.sub = self.create_subscription(
            ObjectStateArray,
            self.input_topic,
            self.state_callback,
            10
        )

        self.pub = self.create_publisher(
            ObjectPredictionArray,
            self.output_topic,
            10
        )

        self.counter = 0

        self.get_logger().info(
            'Object Motion Predictor started'
        )

        self.get_logger().info(
            f'Input: {self.input_topic}'
        )

        self.get_logger().info(
            f'Output: {self.output_topic}'
        )

        self.get_logger().info(
            'Horizons: '
            + ', '.join(
                f'{h:.1f}s'
                for h in self.horizons
            )
        )

    def state_callback(self, msg):

        output = ObjectPredictionArray()

        output.header = msg.header

        for state in msg.states:

            if state.state != ObjectState.TRACKED:
                continue

            vx = float(
                state.velocity.x
            )

            vy = float(
                state.velocity.y
            )

            vz = float(
                state.velocity.z
            )

            speed = math.sqrt(
                vx * vx
                + vy * vy
                + vz * vz
            )

            if speed > self.max_speed:

                self.get_logger().warning(
                    f'Skip {state.track_id}: '
                    f'unreasonable speed '
                    f'{speed:.3f} m/s'
                )

                continue

            prediction = ObjectPrediction()

            prediction.track_id = state.track_id
            prediction.class_id = state.class_id

            prediction.velocity = state.velocity

            prediction.confidence = (
                state.confidence
            )

            prediction.source_stamp = (
                msg.header.stamp
            )

            prediction.last_seen = (
                state.last_seen
            )

            for horizon in self.horizons:

                position = Point()

                position.x = (
                    state.position.x
                    + vx * horizon
                )

                position.y = (
                    state.position.y
                    + vy * horizon
                )

                position.z = (
                    state.position.z
                    + vz * horizon
                )

                prediction.horizons.append(
                    float(horizon)
                )

                prediction.positions.append(
                    position
                )

                uncertainty = Vector3()

                growth = (
                    self.uncertainty_growth
                    * horizon
                )

                uncertainty.x = math.sqrt(
                    state.position_std.x ** 2
                    + growth ** 2
                )

                uncertainty.y = math.sqrt(
                    state.position_std.y ** 2
                    + growth ** 2
                )

                uncertainty.z = math.sqrt(
                    state.position_std.z ** 2
                    + growth ** 2
                )

                prediction.position_std.append(
                    uncertainty
                )

            output.predictions.append(
                prediction
            )

        self.pub.publish(output)

        self.counter += 1

        if self.counter % 20 == 0:
            self.log_summary(output)

    def log_summary(self, msg):

        self.get_logger().info(
            f'Predictions: '
            f'{len(msg.predictions)}'
        )

        for prediction in msg.predictions:

            if not prediction.positions:
                continue

            i = len(
                prediction.positions
            ) - 1

            horizon = (
                prediction.horizons[i]
            )

            position = (
                prediction.positions[i]
            )

            self.get_logger().info(
                f'  {prediction.track_id:<10} '
                f'{prediction.class_id:<10} '
                f't+{horizon:.1f}s -> '
                f'('
                f'{position.x:+.4f},'
                f'{position.y:+.4f},'
                f'{position.z:+.4f})'
            )


def main(args=None):

    rclpy.init(args=args)

    node = ObjectMotionPredictor()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
