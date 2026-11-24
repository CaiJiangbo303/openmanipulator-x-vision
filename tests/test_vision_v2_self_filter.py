import copy
from types import SimpleNamespace

import rclpy

from rclpy.parameter import Parameter
from geometry_msgs.msg import TransformStamped

from vision_msgs.msg import (
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)

from object_self_filter.self_filter_node import (
    ObjectSelfFilter,
    point_to_segment_distance,
)


FRAME = 'camera_color_optical_frame'


def make_node(name, timeout):

    return ObjectSelfFilter(
        cli_args=[
            '--ros-args',
            '-r',
            f'__node:={name}',
        ],
        parameter_overrides=[
            Parameter('world_frame', value=FRAME),
            Parameter(
                'input_topic',
                value=f'/test/{name}/input'
            ),
            Parameter(
                'output_topic',
                value=f'/test/{name}/output'
            ),
            Parameter(
                'tf_wait_timeout',
                value=timeout
            ),
        ]
    )


def install_robot_tf(node, skip=None):

    frames = sorted({
        frame
        for segment in node.segment_definitions
        for frame in segment[:2]
    })

    for i, frame in enumerate(frames):

        if frame == skip:
            continue

        tf = TransformStamped()

        tf.header.frame_id = FRAME
        tf.child_frame_id = frame

        tf.header.stamp = (
            node.get_clock().now().to_msg()
        )

        tf.transform.translation.x = 0.25
        tf.transform.translation.y = i * 0.01
        tf.transform.translation.z = 0.30

        tf.transform.rotation.w = 1.0

        node.tf_buffer.set_transform_static(
            tf,
            'synthetic_robot'
        )


def detection(track_id, xyz):

    d = Detection3D()
    d.id = track_id

    h = ObjectHypothesisWithPose()

    h.hypothesis.class_id = 'blue block'
    h.hypothesis.score = 0.95

    h.pose.pose.position.x = float(xyz[0])
    h.pose.pose.position.y = float(xyz[1])
    h.pose.pose.position.z = float(xyz[2])

    h.pose.pose.orientation.w = 1.0

    d.results.append(h)

    return d


def make_frame(node):

    msg = Detection3DArray()

    msg.header.frame_id = FRAME

    msg.header.stamp = (
        node.get_clock().now().to_msg()
    )

    msg.detections = [
        detection(
            'track_001',
            (0.02, 0.0, 0.30)
        ),
        detection(
            'track_002',
            (0.25, 0.03, 0.30)
        ),
    ]

    return msg


def test_branch(node, name):

    received = []
    original_pub = node.pub

    node.pub = SimpleNamespace(
        publish=received.append
    )

    try:

        install_robot_tf(node)

        msg = make_frame(node)

        node.measurement_callback(msg)
        node.process_pending()

        assert len(received) == 1

        output = received[-1]

        ids = [d.id for d in output.detections]

        assert ids == ['track_001'], ids

        assert output.header == msg.header

        print(f'{name} geometry filtering: PASS')
        print('  Kept:', ids)

        # Zero source timestamp must not request latest TF.

        bad = copy.deepcopy(msg)
        bad.header.stamp.sec = 0
        bad.header.stamp.nanosec = 0

        node.measurement_callback(bad)
        node.process_pending()

        assert len(received) == 1
        assert node.pending_message is None

        print(f'{name} zero timestamp rejection: PASS')

        # Reject mismatched coordinate systems.

        bad = copy.deepcopy(msg)
        bad.header.frame_id = 'wrong_frame'

        node.measurement_callback(bad)
        node.process_pending()

        assert len(received) == 1
        assert node.pending_message is None

        print(f'{name} frame mismatch rejection: PASS')

        # Reject non-finite positions.

        bad = make_frame(node)

        bad.detections.append(
            detection(
                'track_nan',
                (float('nan'), 0.0, 0.30)
            )
        )

        node.measurement_callback(bad)
        node.process_pending()

        assert len(received) == 2

        assert [
            d.id
            for d in received[-1].detections
        ] == ['track_001']

        print(f'{name} non-finite rejection: PASS')

        # Simulate a queued observation past its deadline,
        # even though all robot TF is available.

        node.measurement_callback(
            make_frame(node)
        )

        node.pending_since_ns -= int(1e9)

        node.process_pending()

        assert len(received) == 2
        assert node.pending_message is None

        print(f'{name} expired queue rejection: PASS')

        # Clear TF and leave one required frame missing.

        from tf2_ros import Buffer
        node.tf_buffer = Buffer()

        missing = node.segment_definitions[0][0]

        install_robot_tf(
            node,
            skip=missing
        )

        node.measurement_callback(
            make_frame(node)
        )

        node.process_pending()

        assert len(received) == 2
        assert node.pending_message is not None

        node.pending_since_ns -= int(1e9)

        node.process_pending()

        assert len(received) == 2
        assert node.pending_message is None

        print(f'{name} missing TF fail-closed: PASS')

    finally:

        node.pub = original_pub


def main():

    rclpy.init(args=[])

    fast = make_node(
        'self_filter_fast_test',
        0.12
    )

    precise = make_node(
        'self_filter_precise_test',
        0.20
    )

    try:

        print('===== GEOMETRY CHECK =====')

        distance = point_to_segment_distance(
            (0.5, 0.03, 0.0),
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0)
        )

        assert abs(distance - 0.03) < 1e-9

        print('Point-to-segment distance: PASS')

        print('\n===== FAST FILTER =====')

        test_branch(fast, 'FAST')

        print('\n===== PRECISE FILTER =====')

        test_branch(precise, 'PRECISE')

        print('\nALL 13 SELF FILTER TESTS PASSED')

    finally:

        fast.destroy_node()
        precise.destroy_node()

        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
