#!/usr/bin/env python3

import csv
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

from sensor_msgs.msg import Image, CameraInfo
from tf2_ros import Buffer, TransformListener


RGB_TOPIC = '/camera/camera/color/image_raw'
DEPTH_TOPIC = '/camera/camera/aligned_depth_to_color/image_raw'
INFO_TOPIC = '/camera/camera/color/camera_info'

WORLD_FRAME = 'world'

# Physical calibration reference:
# midpoint of the front-most edge of gripper_left_palm.stl
# STL coordinates: X=65 mm, Y=5.1 mm, Z spans +/-7.856608 mm.
REF_LINK = 'link5'
REF_POINT_LOCAL = np.array(
    [0.1467, 0.0, 0.0],
    dtype=np.float64
)

SAVE_FILE = Path.home() / 'openmanipulator-x-vision' / \
    'camera_world_calibration_points.csv'


def rotation_to_quaternion(R):
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]

    tr = m00 + m11 + m22

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif m00 > m11 and m00 > m22:
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= np.linalg.norm(q)

    return q


def rigid_fit(camera_points, world_points):
    A = np.asarray(camera_points, dtype=np.float64)
    B = np.asarray(world_points, dtype=np.float64)

    ca = A.mean(axis=0)
    cb = B.mean(axis=0)

    AA = A - ca
    BB = B - cb

    H = AA.T @ BB
    U, _, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = cb - R @ ca

    pred = (R @ A.T).T + t

    err = np.linalg.norm(pred - B, axis=1)

    return R, t, err


class CalibrationNode(Node):

    def __init__(self):
        super().__init__('camera_world_calibration')

        self.bridge = CvBridge()

        self.rgb = None
        self.depth = None

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.candidate = None
        self.tip_a = None
        self.tip_b = None
        self.tip_a_uv = None
        self.tip_b_uv = None
        self.candidate_uv = None

        self.camera_points = []
        self.world_points = []

        self.load_existing_samples()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        self.create_subscription(
            Image,
            RGB_TOPIC,
            self.rgb_cb,
            qos_profile_sensor_data
        )

        self.create_subscription(
            Image,
            DEPTH_TOPIC,
            self.depth_cb,
            qos_profile_sensor_data
        )

        self.create_subscription(
            CameraInfo,
            INFO_TOPIC,
            self.info_cb,
            qos_profile_sensor_data
        )

        cv2.namedWindow('Camera-World Calibration')

        cv2.setMouseCallback(
            'Camera-World Calibration',
            self.mouse_cb
        )

        self.timer = self.create_timer(
            1.0 / 30.0,
            self.render
        )

        print()
        print('==========================================')
        print(' CAMERA <-> WORLD CALIBRATION')
        print('==========================================')
        print('LEFT CLICK : select left-gripper reference point')
        print('S          : save pair')
        print('U          : undo last pair')
        print('F          : fit transform')
        print('Q / ESC    : quit')
        print()
        print('IMPORTANT:')
        print('Robot must be completely stationary')
        print('when clicking and saving a point.')
        print()

    def rgb_cb(self, msg):
        self.rgb = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='rgb8'
        )

    def depth_cb(self, msg):
        self.depth = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='passthrough'
        )

    def info_cb(self, msg):
        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])

    def depth_at(self, u, v):
        if self.depth is None:
            return None

        h, w = self.depth.shape[:2]

        r = 4

        x1 = max(0, u - r)
        x2 = min(w, u + r + 1)

        y1 = max(0, v - r)
        y2 = min(h, v + r + 1)

        patch = np.asarray(
            self.depth[y1:y2, x1:x2]
        ).astype(np.float64)

        valid = patch[
            np.isfinite(patch) &
            (patch > 0)
        ]

        if valid.size < 8:
            return None

        z_raw = float(np.median(valid))

        # RealSense aligned depth 16UC1 is millimetres.
        if self.depth.dtype == np.uint16:
            return z_raw * 0.001

        return z_raw

    def deproject(self, u, v, z):
        if None in (
            self.fx,
            self.fy,
            self.cx,
            self.cy
        ):
            return None

        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy

        return np.array(
            [x, y, z],
            dtype=np.float64
        )

    def mouse_cb(
        self,
        event,
        x,
        y,
        flags,
        param
    ):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # ---------------------------
        # First click = TIP A
        # Must have real depth.
        # ---------------------------
        if self.tip_a is None or self.tip_b is not None:

            z = self.depth_at(x, y)

            if z is None:
                print(
                    f'TIP A CLICK ({x},{y}) -> NO VALID DEPTH'
                )
                return

            p = self.deproject(x, y, z)

            if p is None:
                print('CameraInfo not ready.')
                return

            self.tip_a = p
            self.tip_b = None

            self.tip_a_uv = (x, y)
            self.tip_b_uv = None

            self.candidate = None

            print()
            print(
                'TIP A CAMERA XYZ = '
                f'[{p[0]:+.5f}, '
                f'{p[1]:+.5f}, '
                f'{p[2]:+.5f}] m'
            )

            print(
                'Now click the OTHER fingertip.'
            )
            return

        # ---------------------------
        # Second click = TIP B
        #
        # IMPORTANT:
        # Ignore D435 depth completely.
        # Both physical front tips lie on
        # the same front plane, so use
        # TIP A depth.
        # ---------------------------

        z = float(self.tip_a[2])

        p = self.deproject(x, y, z)

        if p is None:
            print('CameraInfo not ready.')
            return

        self.tip_b = p
        self.tip_b_uv = (x, y)

        print()
        print(
            'TIP B CAMERA XYZ = '
            f'[{p[0]:+.5f}, '
            f'{p[1]:+.5f}, '
            f'{p[2]:+.5f}] m '
            '(Z forced from TIP A)'
        )

        sep = float(
            np.linalg.norm(
                self.tip_b - self.tip_a
            )
        )

        print(
            f'TIP SEPARATION = '
            f'{sep*1000:.1f} mm'
        )

        if not (0.040 <= sep <= 0.100):
            print(
                'TIP PAIR = FAIL '
                '(wrong pixel / wrong fingertip)'
            )

            self.tip_b = None
            self.tip_b_uv = None
            self.candidate = None
            return

        self.candidate = (
            self.tip_a + self.tip_b
        ) / 2.0

        print()
        print(
            'CANDIDATE CENTER XYZ = '
            f'[{self.candidate[0]:+.5f}, '
            f'{self.candidate[1]:+.5f}, '
            f'{self.candidate[2]:+.5f}] m'
        )

        print(
            "Press 's' to save."
        )

    def get_world_reference(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                WORLD_FRAME,
                REF_LINK,
                Time()
            )
        except Exception as exc:
            print(
                'TF LOOKUP FAILED:',
                str(exc)
            )
            return None

        tr = tf.transform.translation
        qr = tf.transform.rotation

        q = np.array(
            [qr.x, qr.y, qr.z, qr.w],
            dtype=np.float64
        )

        n = np.linalg.norm(q)

        if n < 1e-12:
            print('INVALID TF QUATERNION')
            return None

        q /= n
        x, y, z, w = q

        R = np.array([
            [
                1 - 2*(y*y + z*z),
                2*(x*y - z*w),
                2*(x*z + y*w)
            ],
            [
                2*(x*y + z*w),
                1 - 2*(x*x + z*z),
                2*(y*z - x*w)
            ],
            [
                2*(x*z - y*w),
                2*(y*z + x*w),
                1 - 2*(x*x + y*y)
            ]
        ], dtype=np.float64)

        t = np.array(
            [tr.x, tr.y, tr.z],
            dtype=np.float64
        )

        return t + R @ REF_POINT_LOCAL

    def load_existing_samples(self):
        """
        Restore calibration samples from CSV so restarting the
        calibration GUI does not reset sample numbering or erase
        previously collected points.
        """
        if not SAVE_FILE.exists():
            print('CALIBRATION CSV: no existing file')
            return

        camera = []
        world = []

        try:
            with SAVE_FILE.open('r', newline='') as f:
                reader = csv.DictReader(f)

                required = {
                    'cam_x', 'cam_y', 'cam_z',
                    'world_x', 'world_y', 'world_z'
                }

                fields = set(reader.fieldnames or [])

                if not required.issubset(fields):
                    print(
                        'CALIBRATION CSV = FAIL '
                        '(unexpected header)'
                    )
                    print(
                        'fields =',
                        reader.fieldnames
                    )
                    return

                for row in reader:
                    pc = np.array([
                        float(row['cam_x']),
                        float(row['cam_y']),
                        float(row['cam_z'])
                    ], dtype=np.float64)

                    pw = np.array([
                        float(row['world_x']),
                        float(row['world_y']),
                        float(row['world_z'])
                    ], dtype=np.float64)

                    if (
                        np.all(np.isfinite(pc))
                        and np.all(np.isfinite(pw))
                    ):
                        camera.append(pc)
                        world.append(pw)

        except Exception as exc:
            print(
                'CALIBRATION CSV LOAD = FAIL:',
                exc
            )
            return

        self.camera_points.extend(camera)
        self.world_points.extend(world)

        print()
        print(
            f'LOADED EXISTING SAMPLES = '
            f'{len(camera)}'
        )

        for i, (pc, pw) in enumerate(
            zip(camera, world),
            1
        ):
            print(
                f'#{i}: '
                f'Camera='
                f'[{pc[0]:+.5f}, '
                f'{pc[1]:+.5f}, '
                f'{pc[2]:+.5f}] '
                f'World='
                f'[{pw[0]:+.5f}, '
                f'{pw[1]:+.5f}, '
                f'{pw[2]:+.5f}]'
            )

        print(
            'CALIBRATION_HISTORY_LOAD = PASS'
        )
        print()

    def save_pair(self):
        if self.candidate is None:
            print('No clicked candidate.')
            return

        pw = self.get_world_reference()

        if pw is None:
            return

        pc = self.candidate.copy()

        self.camera_points.append(pc)
        self.world_points.append(pw)

        print()
        print(
            f'SAVED #{len(self.camera_points)}'
        )

        print(
            'Camera = '
            f'[{pc[0]:+.5f}, '
            f'{pc[1]:+.5f}, '
            f'{pc[2]:+.5f}]'
        )

        print(
            'World  = '
            f'[{pw[0]:+.5f}, '
            f'{pw[1]:+.5f}, '
            f'{pw[2]:+.5f}]'
        )

        self.write_csv()

    def write_csv(self):
        with SAVE_FILE.open(
            'w',
            newline=''
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                'cam_x',
                'cam_y',
                'cam_z',
                'world_x',
                'world_y',
                'world_z'
            ])

            for pc, pw in zip(
                self.camera_points,
                self.world_points
            ):
                writer.writerow(
                    list(pc) + list(pw)
                )

    def fit(self):
        n = len(self.camera_points)

        if n < 4:
            print(
                f'Need >= 4 points, currently {n}.'
            )
            return

        R, t, err = rigid_fit(
            self.camera_points,
            self.world_points
        )

        q = rotation_to_quaternion(R)

        rmse = float(
            np.sqrt(
                np.mean(err ** 2)
            )
        )

        max_err = float(
            np.max(err)
        )

        print()
        print('==========================================')
        print(' CALIBRATION RESULT')
        print('==========================================')

        print()
        print('R(world <- camera_optical) =')
        print(R)

        print()
        print(
            't = '
            f'[{t[0]:+.6f}, '
            f'{t[1]:+.6f}, '
            f'{t[2]:+.6f}] m'
        )

        print(
            'q(x y z w) = '
            f'[{q[0]:+.7f}, '
            f'{q[1]:+.7f}, '
            f'{q[2]:+.7f}, '
            f'{q[3]:+.7f}]'
        )

        print()
        for i, e in enumerate(err, 1):
            print(
                f'point {i:02d}: '
                f'{e * 1000.0:.2f} mm'
            )

        print()
        print(
            f'RMSE    = {rmse * 1000.0:.2f} mm'
        )

        print(
            f'MAX ERR = {max_err * 1000.0:.2f} mm'
        )

        print()

        if (
            rmse <= 0.012
            and
            max_err <= 0.025
        ):
            print('CALIBRATION_FIT = PASS')
        else:
            print(
                'CALIBRATION_FIT = '
                'FAIL_OR_NEEDS_MORE_POINTS'
            )

        print()
        print('DO NOT RUN THIS STATIC TF YET.')
        print('Candidate command only:')
        print()

        print(
            'ros2 run tf2_ros static_transform_publisher '
            f'--x {t[0]:.9f} '
            f'--y {t[1]:.9f} '
            f'--z {t[2]:.9f} '
            f'--qx {q[0]:.9f} '
            f'--qy {q[1]:.9f} '
            f'--qz {q[2]:.9f} '
            f'--qw {q[3]:.9f} '
            '--frame-id world '
            '--child-frame-id camera_color_optical_frame'
        )

        print()
        print('==========================================')

    def render(self):
        if self.rgb is None:
            return

        frame = cv2.cvtColor(
            self.rgb,
            cv2.COLOR_RGB2BGR
        )

        if self.tip_a_uv is not None:
            cv2.drawMarker(
                frame,
                self.tip_a_uv,
                (0, 255, 0),
                cv2.MARKER_CROSS,
                22,
                2
            )

        if self.tip_b_uv is not None:
            cv2.drawMarker(
                frame,
                self.tip_b_uv,
                (255, 0, 255),
                cv2.MARKER_CROSS,
                22,
                2
            )

        cv2.rectangle(
            frame,
            (0, 0),
            (frame.shape[1], 62),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            f'Calibration samples: '
            f'{len(self.camera_points)}',
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            'Click REF | S save | U undo | F fit | Q quit',
            (12, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1
        )

        cv2.imshow(
            'Camera-World Calibration',
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            rclpy.shutdown()

        elif key == ord('s'):
            self.save_pair()

        elif key == ord('u'):
            if self.camera_points:
                self.camera_points.pop()
                self.world_points.pop()
                self.write_csv()
                print(
                    'UNDO ->',
                    len(self.camera_points),
                    'samples'
                )

        elif key == ord('f'):
            self.fit()


def main():
    rclpy.init()

    node = CalibrationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
