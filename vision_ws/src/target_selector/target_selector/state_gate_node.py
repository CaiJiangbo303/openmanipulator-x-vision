import json
import time

import rclpy

from rclpy.node import Node
from rclpy.clock import Clock, ClockType

from std_msgs.msg import String
from std_srvs.srv import Trigger
from object_state_msgs.msg import ObjectStateArray

from .state_gate import StateGate


class StateGateNode(Node):

    def __init__(self, **kwargs):

        super().__init__('state_gate', **kwargs)

        self.declare_parameter(
            'input_topic',
            '/object_state_manager/states'
        )

        self.declare_parameter(
            'status_topic',
            '/state_gate/status'
        )

        self.declare_parameter(
            'requested_track_id',
            ''
        )

        self.declare_parameter(
            'frame',
            'camera_color_optical_frame'
        )

        self.declare_parameter(
            'allowed_classes',
            'blue block,cell phone,smartphone,mobile phone'
        )

        self.declare_parameter('min_confidence', 0.50)
        self.declare_parameter('max_uncertainty', 0.030)
        self.declare_parameter('max_age_ms', 350.0)
        self.declare_parameter('max_prediction_age_ms', 250.0)
        self.declare_parameter('watchdog_ms', 250.0)

        self.requested_track_id = str(
            self.get_parameter('requested_track_id').value
        ).strip()

        classes = self.get_parameter(
            'allowed_classes'
        ).value

        self.gate = StateGate(
            allowed_classes=[
                c.strip()
                for c in classes.split(',')
                if c.strip()
            ],

            # Camera-frame synthetic bench only.
            # These are NOT verified robot workspace limits.
            workspace=(
                (-0.15, 0.15),
                (-0.15, 0.15),
                (0.12, 0.48),
            ),

            frame=self.get_parameter('frame').value,

            min_confidence=float(
                self.get_parameter('min_confidence').value
            ),

            max_uncertainty=float(
                self.get_parameter('max_uncertainty').value
            ),

            max_age_ms=float(
                self.get_parameter('max_age_ms').value
            ),

            max_prediction_age_ms=float(
                self.get_parameter(
                    'max_prediction_age_ms'
                ).value
            ),

            watchdog_ms=float(
                self.get_parameter('watchdog_ms').value
            ),
        )

        self.latest_msg = None
        self.latest_rx = None

        self.input_count = 0
        self.reason = 'not authorized'

        self.pub = self.create_publisher(
            String,
            self.get_parameter('status_topic').value,
            10
        )

        self.sub = self.create_subscription(
            ObjectStateArray,
            self.get_parameter('input_topic').value,
            self.state_callback,
            1
        )

        self.authorize_service = self.create_service(
            Trigger,
            '/state_gate/authorize',
            self.authorize_callback
        )

        self.revoke_service = self.create_service(
            Trigger,
            '/state_gate/revoke',
            self.revoke_callback
        )

        self.steady_clock = Clock(
            clock_type=ClockType.STEADY_TIME
        )

        self.timer = self.create_timer(
            0.05,
            self.watchdog_callback,
            clock=self.steady_clock
        )

        self.get_logger().info(
            'State Gate started | BENCH ONLY'
        )

        self.get_logger().info(
            'No robot command publisher'
        )

        self.get_logger().info(
            f'Requested target: {self.requested_track_id}'
        )

    def times(self):

        return (
            self.get_clock().now().nanoseconds,
            time.monotonic_ns()
        )

    def clear_cache(self):

        self.latest_msg = None
        self.latest_rx = None

    def publish_status(self):

        authorized = self.gate.locked_id is not None

        msg = String()

        msg.data = json.dumps({
            'authorized': authorized,
            'decision': 'ALLOW' if authorized else 'HOLD',
            'locked_id': self.gate.locked_id,
            'requested_track_id': self.requested_track_id,
            'reason': self.reason,
            'input_count': self.input_count,
            'bench_only': True,
        })

        self.pub.publish(msg)

    def state_callback(self, msg):

        now_ns, steady_ns = self.times()

        self.input_count += 1

        self.latest_msg = msg
        self.latest_rx = steady_ns

        if self.gate.locked_id is not None:

            ok, reason = self.gate.update(
                msg,
                now_ns,
                steady_ns
            )

            self.reason = reason

            if not ok:

                self.clear_cache()

                self.get_logger().warning(
                    f'Authorization revoked: {reason}'
                )

        self.publish_status()

    def authorize_callback(self, request, response):

        now_ns, steady_ns = self.times()

        if self.gate.locked_id is not None:

            response.success = False
            response.message = 'release existing authorization first'

            return response

        if not self.requested_track_id:

            response.success = False
            response.message = 'requested_track_id is empty'

            return response

        if (
            self.latest_msg is None
            or self.latest_rx is None
        ):

            response.success = False
            response.message = 'no fresh state available'

            return response

        cache_age_ms = (
            steady_ns - self.latest_rx
        ) / 1e6

        if not 0 <= cache_age_ms <= self.gate.watchdog_ms:

            self.clear_cache()

            response.success = False
            response.message = 'cached state expired'

            return response

        ok, reason = self.gate.authorize(
            self.latest_msg,
            self.requested_track_id,
            now_ns,
            steady_ns
        )

        self.reason = reason

        response.success = ok
        response.message = reason

        self.publish_status()

        return response

    def revoke_callback(self, request, response):

        self.gate.revoke('manual release')
        self.clear_cache()

        self.reason = 'manual release'

        response.success = True
        response.message = self.reason

        self.publish_status()

        return response

    def watchdog_callback(self):

        if self.gate.locked_id is not None:

            now_ns, steady_ns = self.times()

            ok, reason = self.gate.poll(
                now_ns,
                steady_ns
            )

            if not ok:

                self.reason = reason
                self.clear_cache()

                self.get_logger().warning(
                    f'Watchdog HOLD: {reason}'
                )

        self.publish_status()


def main(args=None):

    rclpy.init(args=args)

    node = StateGateNode()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
