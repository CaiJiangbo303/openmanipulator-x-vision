import json
import time

import rclpy

from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter

from std_msgs.msg import String
from std_srvs.srv import Trigger

from object_state_msgs.msg import (
    ObjectState,
    ObjectStateArray,
)

from target_selector.state_gate_node import StateGateNode


def stamp(value, ns):
    value.sec = ns // 1_000_000_000
    value.nanosec = ns % 1_000_000_000


def main():

    rclpy.init(args=[])

    gate = StateGateNode(
        parameter_overrides=[
            Parameter(
                'requested_track_id',
                value='track_001'
            ),
            Parameter(
                'input_topic',
                value='/test_gate/states'
            ),
        ]
    )

    node = rclpy.create_node('state_gate_ros_test')

    executor = SingleThreadedExecutor()

    executor.add_node(gate)
    executor.add_node(node)

    pub = node.create_publisher(
        ObjectStateArray,
        '/test_gate/states',
        10
    )

    authorize = node.create_client(
        Trigger,
        '/state_gate/authorize'
    )

    revoke = node.create_client(
        Trigger,
        '/state_gate/revoke'
    )

    latest = {}
    passed = 0

    def status_callback(msg):

        latest.clear()
        latest.update(json.loads(msg.data))

    sub = node.create_subscription(
        String,
        '/state_gate/status',
        status_callback,
        10
    )

    def wait_until(predicate, timeout=2.0):

        deadline = time.monotonic() + timeout

        while not predicate():

            if time.monotonic() >= deadline:

                raise RuntimeError(
                    f'Timeout; latest status={latest}'
                )

            executor.spin_once(timeout_sec=0.005)

    def call(client):

        future = client.call_async(
            Trigger.Request()
        )

        wait_until(future.done)

        return future.result()

    def report(name):

        nonlocal passed

        passed += 1

        print(f'{name}: PASS')

    def send_frame(
        ids=('track_001',),
        cls='blue block'
    ):

        previous = latest.get('input_count', 0)

        now = node.get_clock().now().nanoseconds

        msg = ObjectStateArray()

        msg.header.frame_id = (
            'camera_color_optical_frame'
        )

        stamp(msg.header.stamp, now)

        for track_id in ids:

            s = ObjectState()

            s.track_id = track_id
            s.class_id = cls

            s.position.x = 0.02
            s.position.y = 0.0
            s.position.z = 0.30

            s.position_std.x = 0.003
            s.position_std.y = 0.003
            s.position_std.z = 0.003

            s.confidence = 0.95
            s.valid_3d = True
            s.stale = False

            s.state = ObjectState.TRACKED

            s.measurement_source_type = (
                ObjectState.SOURCE_FUSED
            )

            s.position_predicted = False
            s.measurement_age_ms = 20.0

            s.age = 10
            s.missed = 0

            stamp(
                s.last_seen,
                now - 20_000_000
            )

            # Deliberate source-clock offset.
            stamp(
                s.source_stamp,
                now - 8_000_000_000
            )

            msg.states.append(s)

        pub.publish(msg)

        wait_until(
            lambda: latest.get('input_count', 0)
            > previous
        )

    try:

        print('===== ROS CONNECTIVITY =====')

        wait_until(
            lambda:
            pub.get_subscription_count() > 0
            and authorize.service_is_ready()
            and revoke.service_is_ready()
            and bool(latest),
            timeout=10.0
        )

        report('ROS connectivity')

        assert latest['decision'] == 'HOLD'
        assert not latest['authorized']

        report('Startup unauthorized')

        response = call(authorize)

        assert not response.success

        report('Authorization without data rejected')

        send_frame()

        assert not latest['authorized']

        report('Incoming data does not auto-authorize')

        response = call(authorize)

        assert response.success, response.message

        wait_until(
            lambda: latest.get('authorized') is True
        )

        assert latest['locked_id'] == 'track_001'

        report('Explicit ROS service authorization')

        send_frame(
            ('track_001', 'track_002')
        )

        assert latest['authorized']
        assert latest['locked_id'] == 'track_001'

        report('Challenger cannot change target')

        send_frame(('track_002',))

        assert not latest['authorized']
        assert latest['locked_id'] is None

        report('Missing target revokes authorization')

        send_frame(('track_001',))

        assert not latest['authorized']

        report('Reappearing target remains unauthorized')

        response = call(authorize)

        assert response.success, response.message

        wait_until(
            lambda: latest.get('authorized') is True
        )

        report('Explicit reauthorization')

        # Stop publishing and let the steady timer
        # invalidate the previous authorization.
        wait_until(
            lambda: latest.get('authorized') is False,
            timeout=1.0
        )

        assert latest['decision'] == 'HOLD'

        print('Watchdog reason:', latest['reason'])

        report('Automatic watchdog HOLD')

        response = call(authorize)

        assert not response.success

        report('Expired cache cannot reauthorize')

        send_frame()

        response = call(authorize)

        assert response.success, response.message

        wait_until(
            lambda: latest.get('authorized') is True
        )

        response = call(revoke)

        assert response.success

        wait_until(
            lambda: latest.get('authorized') is False
        )

        report('Manual revoke service')

        response = call(authorize)

        assert not response.success

        report('Revoked cache cannot reauthorize')

        send_frame(cls='person')

        response = call(authorize)

        assert not response.success

        report('Unsafe class authorization rejected')

        print()
        print(f'Total ROS checks passed: {passed}')
        print('ALL STATE GATE ROS TESTS PASSED')

    finally:

        executor.remove_node(gate)
        executor.remove_node(node)

        gate.destroy_node()
        node.destroy_node()

        executor.shutdown()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
