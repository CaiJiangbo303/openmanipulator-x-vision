import time
from collections import Counter

import numpy as np
import rclpy

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster

from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    Detection3DArray,
    ObjectHypothesisWithPose,
)

from object_state_msgs.msg import ObjectStateArray


FRAME = 'camera_color_optical_frame'

W = 640
H = 480

FX = 600.0
FY = 600.0
CX = 320.0
CY = 240.0

OBJ_X = 360
OBJ_Y = 240
OBJ_Z = 0.300

EXPECTED = np.array([
    (OBJ_X - CX) * OBJ_Z / FX,
    (OBJ_Y - CY) * OBJ_Z / FY,
    OBJ_Z,
])

rclpy.init()
node = rclpy.create_node('vision_v2_synthetic_test')

depth_pub = node.create_publisher(
    Image,
    '/camera/camera/aligned_depth_to_color/image_raw',
    10
)

info_pub = node.create_publisher(
    CameraInfo,
    '/camera/camera/color/camera_info',
    10
)

det_pub = node.create_publisher(
    Detection2DArray,
    '/object_tracker/detections',
    10
)

mask_pub = node.create_publisher(
    Image,
    '/object_segmenter/instance_mask',
    10
)

stats = {
    'FAST': [],
    'MASK': [],
}

sources = Counter()
state_errors = []
latest = {}
phase = 'RUNNING'

input_times = []
input_ros_times = []
state_times = []
no_input_times = []
no_input_states = []
running_interruptions = []


def xyz_error(p):
    xyz = np.array([p.x, p.y, p.z])
    return float(
        np.linalg.norm(xyz - EXPECTED) * 1000
    )


def detection_cb(msg, name):
    for d in msg.detections:

        if d.id != 'track_001' or not d.results:
            continue

        p = d.results[0].pose.pose.position
        error = xyz_error(p)

        stats[name].append(error)

        if len(stats[name]) == 1:
            print(
                f'{name} FIRST: '
                f'xyz=({p.x:.4f}, '
                f'{p.y:.4f}, {p.z:.4f}) '
                f'error={error:.2f} mm',
                flush=True
            )


def state_cb(msg):
    found = False
    for field, field_type in (
        msg.get_fields_and_field_types().items()
    ):
        if 'ObjectState' not in field_type:
            continue

        for s in getattr(msg, field):

            if s.track_id != 'track_001':
                continue

            found = True

            if phase == 'RUNNING':

                state_times.append(time.monotonic())

                if s.stale or int(s.state) == 2:
                    running_interruptions.append(
                        ('invalid_state', s.track_id)
                    )

                sources[int(s.measurement_source_type)] += 1

                if s.valid_3d and not s.stale:
                    state_errors.append(
                        xyz_error(s.position)
                    )

            latest.update({
                'id': s.track_id,
                'source': s.measurement_source,
                'x': round(s.position.x, 5),
                'y': round(s.position.y, 5),
                'z': round(s.position.z, 5),
                'age_ms': round(s.measurement_age_ms, 1),
                'valid': s.valid_3d,
                'stale': s.stale,
                'state': int(s.state),
            })


            if phase == 'NO_INPUT':
                no_input_times.append(time.monotonic())
                no_input_states.append(dict(latest))

    if phase == 'RUNNING' and latest and not found:
        running_interruptions.append(
            ('missing_target', 'track_001')
        )


fast_sub = node.create_subscription(
    Detection3DArray,
    '/object_localizer/detections_3d_fast',
    lambda msg: detection_cb(msg, 'FAST'),
    10
)

mask_sub = node.create_subscription(
    Detection3DArray,
    '/object_localizer/detections_3d_mask',
    lambda msg: detection_cb(msg, 'MASK'),
    10
)

state_sub = node.create_subscription(
    ObjectStateArray,
    '/object_state_manager/states',
    state_cb,
    10
)


def spin_for(seconds):
    end = time.monotonic() + seconds

    while time.monotonic() < end:
        rclpy.spin_once(
            node,
            timeout_sec=0.005
        )


# Synthetic depth image:
# background = 800 mm
# object = 300 mm
depth = np.full(
    (H, W), 800, dtype='<u2'
)

depth[210:271, 330:391] = 300

# Instance mask:
# background = 0
# track_001 = 1
mask = np.zeros(
    (H, W), dtype='<u2'
)

mask[210:271, 330:391] = 1


def image_message(array, encoding, stamp):
    msg = Image()

    msg.header.stamp = stamp
    msg.header.frame_id = FRAME

    msg.height = H
    msg.width = W
    msg.encoding = encoding
    msg.is_bigendian = False
    msg.step = W * 2
    msg.data = array.tobytes()

    return msg


def camera_info(stamp):
    msg = CameraInfo()

    msg.header.stamp = stamp
    msg.header.frame_id = FRAME

    msg.width = W
    msg.height = H

    msg.distortion_model = 'plumb_bob'
    msg.d = [0.0] * 5

    msg.k = [
        FX, 0.0, CX,
        0.0, FY, CY,
        0.0, 0.0, 1.0
    ]

    msg.r = [
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    ]

    msg.p = [
        FX, 0.0, CX, 0.0,
        0.0, FY, CY, 0.0,
        0.0, 0.0, 1.0, 0.0
    ]

    return msg


def detection(stamp):
    msg = Detection2DArray()

    msg.header.stamp = stamp
    msg.header.frame_id = FRAME

    d = Detection2D()
    d.header = msg.header
    d.id = 'track_001'

    d.bbox.center.position.x = float(OBJ_X)
    d.bbox.center.position.y = float(OBJ_Y)

    d.bbox.size_x = 61.0
    d.bbox.size_y = 61.0

    h = ObjectHypothesisWithPose()

    h.hypothesis.class_id = 'blue block'
    h.hypothesis.score = 0.95

    d.results.append(h)
    msg.detections.append(d)

    return msg


def connected():
    return (
        depth_pub.get_subscription_count() >= 2
        and info_pub.get_subscription_count() >= 2
        and det_pub.get_subscription_count() >= 2
        and mask_pub.get_subscription_count() >= 1
        and node.count_publishers(
            '/object_localizer/detections_3d_fast'
        ) >= 1
        and node.count_publishers(
            '/object_localizer/detections_3d_mask'
        ) >= 1
        and node.count_publishers(
            '/object_state_manager/states'
        ) >= 1
    )


try:
    print('===== SYNTHETIC RGB-D TEST =====')
    print('Expected XYZ:', EXPECTED.tolist())

    broadcaster = StaticTransformBroadcaster(node)

    tf = TransformStamped()

    tf.header.stamp = node.get_clock().now().to_msg()
    tf.header.frame_id = 'test_fixture_world'
    tf.child_frame_id = FRAME
    tf.transform.rotation.w = 1.0

    broadcaster.sendTransform(tf)

    deadline = time.monotonic() + 10

    while not connected():
        if time.monotonic() > deadline:
            print('Depth subscribers:',
                  depth_pub.get_subscription_count())
            print('Detection subscribers:',
                  det_pub.get_subscription_count())
            print('Mask subscribers:',
                  mask_pub.get_subscription_count())

            raise RuntimeError(
                'Fusion nodes not connected'
            )

        spin_for(0.1)

    print('ROS connectivity: PASS')

    spin_for(0.5)

    print('\n===== PUBLISH SYNTHETIC INPUT =====')

    start = time.monotonic()

    for i in range(40):

        input_times.append(time.monotonic())

        stamp = node.get_clock().now().to_msg()

        input_ros_times.append(
            stamp.sec + stamp.nanosec / 1e9
        )

        info_pub.publish(camera_info(stamp))

        depth_pub.publish(
            image_message(depth, '16UC1', stamp)
        )

        det_pub.publish(
            detection(stamp)
        )

        if i % 4 == 0:
            spin_for(0.04)

            mask_pub.publish(
                image_message(mask, 'mono16', stamp)
            )

        next_tick = start + (i + 1) * 0.1

        spin_for(
            max(0, next_tick - time.monotonic())
        )

    spin_for(0.15)

    print('\n===== LOCALIZATION SUMMARY =====')

    for name, errors in stats.items():

        print(f'{name} received: {len(errors)}')

        if errors:
            print(
                f'{name} mean error: '
                f'{np.mean(errors):.3f} mm'
            )

            print(
                f'{name} max error: '
                f'{np.max(errors):.3f} mm'
            )

    print('\n===== FUSION SUMMARY =====')

    print('Source counts:', dict(sources))
    print('Latest state:', latest)

    if state_errors:
        print(
            'Fused/state mean error:',
            round(float(np.mean(state_errors)), 3),
            'mm'
        )

    phase = 'NO_INPUT'

    print('\n===== STALE TEST =====')

    stop_mono = time.monotonic()
    stop_ros = node.get_clock().now().nanoseconds

    stale_state = {}
    lost_state = {}

    deadline = stop_mono + 2.5

    while time.monotonic() < deadline:

        spin_for(0.02)

        for sample in no_input_states:

            if sample.get('stale') is True:
                if not stale_state:
                    stale_state = dict(sample)

            if sample.get('state') == 2:
                lost_state = dict(sample)

        if stale_state and lost_state:
            break

    elapsed_mono = time.monotonic() - stop_mono
    elapsed_ros = (
        node.get_clock().now().nanoseconds - stop_ros
    ) / 1e9

    print('Stale state:', stale_state)
    print('Lost state:', lost_state)
    print('Last received state:', latest)

    print(
        'Elapsed monotonic:',
        round(elapsed_mono, 3), 's'
    )

    print(
        'Elapsed ROS clock:',
        round(elapsed_ros, 3), 's'
    )

    print(
        'No-input state messages:',
        len(no_input_states)
    )

    print('\n===== ACCEPTANCE =====')

    def gaps_ms(values):
        return [
            (b - a) * 1000
            for a, b in zip(values, values[1:])
        ]

    def timing_ok(values):
        gaps = gaps_ms(values)

        return bool(gaps) and all(
            0 < dt <= 250.0 for dt in gaps
        )

    print('===== TIMING SUMMARY =====')

    for name, values in [
        ('Input monotonic', input_times),
        ('Input ROS clock', input_ros_times),
        ('State output', state_times),
        ('No-input output', no_input_times),
    ]:
        gaps = gaps_ms(values)

        if gaps:
            print(
                f'{name}: '
                f'min={min(gaps):.1f} ms, '
                f'max={max(gaps):.1f} ms, '
                f'mean={np.mean(gaps):.1f} ms'
            )
        else:
            print(f'{name}: insufficient samples')

    print(
        'Running interruptions:',
        running_interruptions
    )

    checks = {
        'Input timing continuity':
            timing_ok(input_times),

        'ROS timestamp continuity':
            timing_ok(input_ros_times),

        'State output continuity':
            timing_ok(state_times),

        'No-input output continuity':
            timing_ok(state_times[-1:] + no_input_times),

        'Target lifecycle continuity':
            not running_interruptions,

        'FAST complete delivery':
            len(stats['FAST']) == 40,

        'MASK complete delivery':
            len(stats['MASK']) == 10,

        'FAST3D output':
            len(stats['FAST']) > 0,

        'MASK3D output':
            len(stats['MASK']) > 0,

        'FAST3D position':
            bool(stats['FAST'])
            and max(stats['FAST']) < 10.0,

        'MASK3D position':
            bool(stats['MASK'])
            and max(stats['MASK']) < 10.0,

        'Fused state':
            sources[3] > 0,

        'State position':
            bool(state_errors)
            and max(state_errors) < 10.0,

        'Stale protection':
            stale_state.get('stale') is True,

        'Lost transition':
            lost_state.get('state') == 2,
    }

    for name, passed in checks.items():
        print(
            f'{name}: '
            f'{"PASS" if passed else "FAIL"}'
        )

    assert all(checks.values()), \
        'One or more acceptance checks failed'

    print('\nALL VISION V2 FUSION TESTS PASSED')

finally:
    node.destroy_node()
    rclpy.shutdown()
