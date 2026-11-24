import os
import signal
import subprocess
import tempfile
import time

from pathlib import Path

import rclpy

from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster

from vision_msgs.msg import (
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)

from object_state_msgs.msg import ObjectStateArray


ROOT = Path(__file__).resolve().parents[1]
FRAME = 'camera_color_optical_frame'

assert os.environ.get('ROS_DOMAIN_ID') == '198'
assert os.environ.get('ROS_LOCALHOST_ONLY') == '1'

rclpy.init(args=[])

node = rclpy.create_node('filtered_fusion_test')

broadcaster = StaticTransformBroadcaster(node)

frames = [
    'arm_link1',
    'arm_link2',
    'arm_link3',
    'arm_link4',
    'arm_link5',
    'arm_end_effector_link',
    'arm_gripper_left_link',
    'arm_gripper_right_link',
]

transforms = []

for i, frame in enumerate(frames):

    tf = TransformStamped()

    tf.header.frame_id = FRAME
    tf.header.stamp = node.get_clock().now().to_msg()
    tf.child_frame_id = frame

    tf.transform.translation.x = 0.25
    tf.transform.translation.y = i * 0.01
    tf.transform.translation.z = 0.30
    tf.transform.rotation.w = 1.0

    transforms.append(tf)

broadcaster.sendTransform(transforms)

print('Synthetic robot TF: published')

received = {
    'fast': [],
    'mask': [],
}

state_history = []

subs = []

for branch in received:

    subs.append(
        node.create_subscription(
            Detection3DArray,
            f'/object_self_filter/detections_3d_{branch}',
            lambda msg, b=branch: received[b].append(
                [d.id for d in msg.detections]
            ),
            10
        )
    )

subs.append(
    node.create_subscription(
        ObjectStateArray,
        '/object_state_manager/states',
        lambda msg: state_history.append(
            [s.track_id for s in msg.states]
        ),
        10
    )
)

pubs = [
    node.create_publisher(
        Detection3DArray,
        f'/object_localizer/detections_3d_{branch}',
        10
    )
    for branch in received
]


def spin_for(seconds):

    end = time.monotonic() + seconds

    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.01)


def detection(track_id, x, y, z):

    d = Detection3D()
    d.id = track_id

    h = ObjectHypothesisWithPose()

    h.hypothesis.class_id = 'blue block'
    h.hypothesis.score = 0.95

    h.pose.pose.position.x = float(x)
    h.pose.pose.position.y = float(y)
    h.pose.pose.position.z = float(z)

    h.pose.pose.orientation.w = 1.0

    d.results.append(h)

    return d


log = tempfile.NamedTemporaryFile(
    prefix='vision_v2_filtered_',
    suffix='.log',
    delete=False
)

launch = None
test = None
rc = 1

try:

    launch = subprocess.Popen(
        [
            'ros2', 'launch',
            str(ROOT / 'run_nodes/vision_v2_fusion.launch.py'),
            'enable_self_filter:=true',
            'enable_state_gate:=true',
            'requested_track_id:=track_001',
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    spin_for(2.0)

    assert launch.poll() is None

    env = os.environ.copy()
    env['VISION_V2_TEST_GATE'] = '1'

    test = subprocess.Popen(
        [
            'python3', '-u',
            str(ROOT / 'tests/test_vision_v2_fusion.py'),
        ],
        env=env,
    )

    deadline = time.monotonic() + 30.0

    while test.poll() is None:

        assert time.monotonic() < deadline, 'Test timeout'
        assert launch.poll() is None, 'Launch exited early'

        rclpy.spin_once(node, timeout_sec=0.01)

    assert test.returncode == 0

    spin_for(0.3)

    print('\n===== FILTERED DELIVERY =====')

    for branch, expected in [('fast', 40), ('mask', 10)]:

        data = received[branch]

        print(branch, 'received:', len(data))

        assert len(data) == expected
        assert all(ids == ['track_001'] for ids in data)

        print(f'{branch} filtered delivery: PASS')

    # Inject two same-class 3D observations into both filters.
    # One external object and one robot-self object.

    assert all(
        pub.get_subscription_count() > 0
        for pub in pubs
    )

    for data in received.values():
        data.clear()

    state_history.clear()

    for _ in range(5):

        msg = Detection3DArray()

        msg.header.frame_id = FRAME
        msg.header.stamp = node.get_clock().now().to_msg()

        msg.detections = [
            detection('track_901', 0.02, 0.0, 0.30),
            detection('track_902', 0.25, 0.03, 0.30),
        ]

        for pub in pubs:
            pub.publish(msg)

        spin_for(0.1)

    spin_for(0.3)

    print('\n===== SELF REJECTION PROBE =====')

    for branch, data in received.items():

        print(branch, 'outputs:', data)

        assert data
        assert all(ids == ['track_901'] for ids in data)

        print(f'{branch} robot-self rejection: PASS')

    assert any(
        'track_901' in ids
        for ids in state_history
    )

    print('External object reaches fusion: PASS')

    assert all(
        'track_902' not in ids
        for ids in state_history
    )

    print('Robot-self object excluded from fusion: PASS')

    print('\nALL FILTERED FUSION TESTS PASSED')
    rc = 0

finally:

    if test is not None and test.poll() is None:
        test.kill()
        test.wait()

    if launch is not None and launch.poll() is None:

        launch.send_signal(signal.SIGINT)

        try:
            launch.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(launch.pid, signal.SIGKILL)
            launch.wait()

    log.close()

    print('\n===== LAUNCH LOG =====')

    print('\n'.join(
        Path(log.name).read_text(
            errors='replace'
        ).splitlines()[-35:]
    ))

    print('Full log:', log.name)
    print('Test return code:', rc)

    if launch is not None:
        print('Launch return code:', launch.returncode)

    node.destroy_node()
    rclpy.try_shutdown()

raise SystemExit(
    rc if rc != 0 else int(launch.returncode != 0)
)
