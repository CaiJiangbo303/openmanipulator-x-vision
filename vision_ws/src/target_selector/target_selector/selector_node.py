#!/usr/bin/env python3

import math

import rclpy

from rclpy.node import Node

from object_state_msgs.msg import (
    ObjectPredictionArray,
)

from four_dof_feasibility_msgs.msg import (
    FourDofFeasibility,
    FourDofFeasibilityArray,
)

from target_selector_msgs.msg import (
    TargetSelection,
)


def stamp_ns(stamp):
    return (
        int(stamp.sec) * 1_000_000_000
        +
        int(stamp.nanosec)
    )


def clamp01(value):
    return max(
        0.0,
        min(
            1.0,
            float(value)
        )
    )


class TargetSelectorNode(Node):

    def __init__(self):

        super().__init__(
            'target_selector'
        )

        self.declare_parameter(
            'prediction_topic',
            '/object_motion_predictor/predictions'
        )

        self.declare_parameter(
            'feasibility_topic',
            '/four_dof_feasibility/results'
        )

        self.declare_parameter(
            'output_topic',
            '/target_selector/selection'
        )

        self.declare_parameter(
            'desired_horizon',
            0.5
        )

        self.declare_parameter(
            'horizon_scale',
            0.35
        )

        self.declare_parameter(
            'margin_scale',
            0.8
        )

        self.declare_parameter(
            'uncertainty_scale',
            0.03
        )

        self.declare_parameter(
            'residual_scale',
            0.003
        )

        self.declare_parameter(
            'weight_margin',
            0.30
        )

        self.declare_parameter(
            'weight_uncertainty',
            0.25
        )

        self.declare_parameter(
            'weight_confidence',
            0.25
        )

        self.declare_parameter(
            'weight_horizon',
            0.15
        )

        self.declare_parameter(
            'weight_residual',
            0.05
        )

        self.prediction_topic = (
            self.get_parameter(
                'prediction_topic'
            ).value
        )

        self.feasibility_topic = (
            self.get_parameter(
                'feasibility_topic'
            ).value
        )

        self.output_topic = (
            self.get_parameter(
                'output_topic'
            ).value
        )

        self.desired_horizon = float(
            self.get_parameter(
                'desired_horizon'
            ).value
        )

        self.horizon_scale = float(
            self.get_parameter(
                'horizon_scale'
            ).value
        )

        self.margin_scale = float(
            self.get_parameter(
                'margin_scale'
            ).value
        )

        self.uncertainty_scale = float(
            self.get_parameter(
                'uncertainty_scale'
            ).value
        )

        self.residual_scale = float(
            self.get_parameter(
                'residual_scale'
            ).value
        )

        self.weight_margin = float(
            self.get_parameter(
                'weight_margin'
            ).value
        )

        self.weight_uncertainty = float(
            self.get_parameter(
                'weight_uncertainty'
            ).value
        )

        self.weight_confidence = float(
            self.get_parameter(
                'weight_confidence'
            ).value
        )

        self.weight_horizon = float(
            self.get_parameter(
                'weight_horizon'
            ).value
        )

        self.weight_residual = float(
            self.get_parameter(
                'weight_residual'
            ).value
        )

        self.prediction_cache = {}
        self.feasibility_cache = {}

        self.latest_processed_ns = -1

        self.counter = 0

        self.pred_sub = self.create_subscription(
            ObjectPredictionArray,
            self.prediction_topic,
            self.prediction_callback,
            1
        )

        self.feas_sub = self.create_subscription(
            FourDofFeasibilityArray,
            self.feasibility_topic,
            self.feasibility_callback,
            1
        )

        self.pub = self.create_publisher(
            TargetSelection,
            self.output_topic,
            1
        )

        self.get_logger().info(
            'Target Selector started'
        )

        self.get_logger().info(
            f'Predictions: {self.prediction_topic}'
        )

        self.get_logger().info(
            f'Feasibility: {self.feasibility_topic}'
        )

        self.get_logger().info(
            f'Output: {self.output_topic}'
        )

        self.get_logger().info(
            f'Desired horizon: '
            f'{self.desired_horizon:.2f}s'
        )

    def prediction_callback(
        self,
        msg
    ):

        key = stamp_ns(
            msg.header.stamp
        )

        self.prediction_cache[key] = (
            msg
        )

        self.try_process(
            key
        )

        self.trim_caches()

    def feasibility_callback(
        self,
        msg
    ):

        key = stamp_ns(
            msg.header.stamp
        )

        self.feasibility_cache[key] = (
            msg
        )

        self.try_process(
            key
        )

        self.trim_caches()

    def trim_caches(
        self
    ):

        all_keys = sorted(
            set(
                self.prediction_cache.keys()
            )
            |
            set(
                self.feasibility_cache.keys()
            )
        )

        while len(all_keys) > 8:

            oldest = all_keys.pop(0)

            self.prediction_cache.pop(
                oldest,
                None
            )

            self.feasibility_cache.pop(
                oldest,
                None
            )

    def find_prediction(
        self,
        predictions,
        track_id
    ):

        for prediction in (
            predictions.predictions
        ):

            if (
                prediction.track_id
                ==
                track_id
            ):

                return prediction

        return None

    def find_horizon_index(
        self,
        prediction,
        horizon
    ):

        best_index = None
        best_error = None

        for index, value in enumerate(
            prediction.horizons
        ):

            error = abs(
                float(value)
                -
                float(horizon)
            )

            if (
                best_error is None
                or
                error < best_error
            ):

                best_error = error
                best_index = index

        if (
            best_index is None
            or
            best_error > 1e-3
        ):

            return None

        return best_index

    def uncertainty_at(
        self,
        prediction,
        index
    ):

        if (
            index
            >=
            len(
                prediction.position_std
            )
        ):

            return float('inf')

        std = prediction.position_std[
            index
        ]

        return math.sqrt(
            std.x * std.x
            +
            std.y * std.y
            +
            std.z * std.z
        )

    def calculate_score(
        self,
        candidate,
        prediction,
        uncertainty
    ):

        margin_score = clamp01(
            candidate.min_joint_margin
            /
            self.margin_scale
        )

        uncertainty_score = math.exp(
            -
            uncertainty
            /
            self.uncertainty_scale
        )

        confidence_score = clamp01(
            prediction.confidence
        )

        horizon_score = math.exp(
            -
            abs(
                float(candidate.horizon)
                -
                self.desired_horizon
            )
            /
            self.horizon_scale
        )

        residual_score = math.exp(
            -
            float(
                candidate.position_residual
            )
            /
            self.residual_scale
        )

        score = (
            self.weight_margin
            *
            margin_score
            +
            self.weight_uncertainty
            *
            uncertainty_score
            +
            self.weight_confidence
            *
            confidence_score
            +
            self.weight_horizon
            *
            horizon_score
            +
            self.weight_residual
            *
            residual_score
        )

        return float(
            score
        )

    def try_process(
        self,
        key
    ):

        if (
            key
            <=
            self.latest_processed_ns
        ):

            self.prediction_cache.pop(
                key,
                None
            )

            self.feasibility_cache.pop(
                key,
                None
            )

            return

        if (
            key
            not in self.prediction_cache
            or
            key
            not in self.feasibility_cache
        ):

            return

        predictions = (
            self.prediction_cache.pop(
                key
            )
        )

        feasibility = (
            self.feasibility_cache.pop(
                key
            )
        )

        self.latest_processed_ns = (
            key
        )

        self.select_target(
            predictions,
            feasibility
        )

    def select_target(
        self,
        predictions,
        feasibility
    ):

        scored = []

        for candidate in (
            feasibility.candidates
        ):

            # Safety policy v1:
            # only REACHABLE candidates may be selected.
            if (
                candidate.status
                !=
                FourDofFeasibility.REACHABLE
            ):

                continue

            prediction = (
                self.find_prediction(
                    predictions,
                    candidate.track_id
                )
            )

            if prediction is None:
                continue

            index = (
                self.find_horizon_index(
                    prediction,
                    candidate.horizon
                )
            )

            if index is None:
                continue

            uncertainty = (
                self.uncertainty_at(
                    prediction,
                    index
                )
            )

            if not math.isfinite(
                uncertainty
            ):

                continue

            score = (
                self.calculate_score(
                    candidate,
                    prediction,
                    uncertainty
                )
            )

            scored.append(
                (
                    score,
                    uncertainty,
                    candidate,
                    prediction,
                )
            )

        output = TargetSelection()

        output.header = (
            feasibility.header
        )

        if not scored:

            output.valid = False

            output.reason = (
                'no reachable candidate'
            )

            self.pub.publish(
                output
            )

            self.counter += 1

            if self.counter % 20 == 0:

                self.get_logger().info(
                    'Selection: NONE '
                    '(no reachable candidate)'
                )

            return

        # Deterministic ordering:
        #
        # 1. highest total score
        # 2. lower uncertainty
        # 3. larger joint margin
        # 4. horizon closer to desired horizon
        # 5. track id
        scored.sort(
            key=lambda item: (
                -item[0],
                item[1],
                -item[2].min_joint_margin,
                abs(
                    item[2].horizon
                    -
                    self.desired_horizon
                ),
                item[2].track_id,
            )
        )

        (
            score,
            uncertainty,
            candidate,
            prediction,
        ) = scored[0]

        output.valid = True

        output.track_id = (
            candidate.track_id
        )

        output.class_id = (
            candidate.class_id
        )

        output.horizon = float(
            candidate.horizon
        )

        output.target_world = (
            candidate.target_world
        )

        output.target_arm = (
            candidate.target_arm
        )

        output.joint_positions = list(
            candidate.joint_positions
        )

        output.score = float(
            score
        )

        output.confidence = float(
            prediction.confidence
        )

        output.uncertainty = float(
            uncertainty
        )

        output.min_joint_margin = float(
            candidate.min_joint_margin
        )

        output.position_residual = float(
            candidate.position_residual
        )

        output.reason = (
            'best reachable candidate'
        )

        self.pub.publish(
            output
        )

        self.counter += 1

        if self.counter % 20 == 0:

            self.get_logger().info(
                'Selection: '
                f'{output.track_id} '
                f'{output.class_id} '
                f't+{output.horizon:.1f}s '
                f'score={output.score:.3f} '
                f'conf={output.confidence:.3f} '
                f'sigma={output.uncertainty*1000:.1f}mm '
                f'margin={output.min_joint_margin:.3f}rad'
            )


def main(args=None):

    rclpy.init(
        args=args
    )

    node = TargetSelectorNode()

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
