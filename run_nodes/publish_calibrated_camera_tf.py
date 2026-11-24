#!/usr/bin/env python3

import time
import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import TransformStamped
from tf2_ros import (
    Buffer,
    TransformListener,
    StaticTransformBroadcaster,
)

# Calibrated:
# world <- camera_color_optical_frame
R_WO = np.array([
    [ 0.05748848, -0.76377511,  0.64291730],
    [-0.99797608, -0.02643141,  0.05783711],
    [-0.02718134, -0.64494105, -0.76374879],
], dtype=np.float64)

t_WO = np.array([
    0.240490733,
    0.038758274,
    0.051950796,
], dtype=np.float64)


def quat_to_rot(q):
    x, y, z, w = q
    n = np.linalg.norm(q)
    x, y, z, w = q / n

    return np.array([
        [
            1 - 2*(y*y + z*z),
            2*(x*y - z*w),
            2*(x*z + y*w),
        ],
        [
            2*(x*y + z*w),
            1 - 2*(x*x + z*z),
            2*(y*z - x*w),
        ],
        [
            2*(x*z - y*w),
            2*(y*z + x*w),
            1 - 2*(x*x + y*y),
        ],
    ], dtype=np.float64)


def rot_to_quat(R):
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]

    tr = m00 + m11 + m22

    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s

    elif m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s

    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s

    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    return q / np.linalg.norm(q)


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


class Publisher(Node):
    def __init__(self):
        super().__init__('camera_world_static_tf')

        self.buffer = Buffer()
        self.listener = TransformListener(
            self.buffer,
            self
        )

        self.broadcaster = StaticTransformBroadcaster(
            self
        )


rclpy.init()
node = Publisher()

# Give RealSense static TF time to arrive
deadline = time.monotonic() + 5.0
tf = None

while time.monotonic() < deadline:
    rclpy.spin_once(node, timeout_sec=0.1)

    try:
        # camera_link <- camera_color_optical_frame
        tf = node.buffer.lookup_transform(
            'camera_link',
            'camera_color_optical_frame',
            Time()
        )
        break

    except Exception:
        pass

if tf is None:
    print("CAMERA INTERNAL TF = FAIL")
    node.destroy_node()
    rclpy.shutdown()
    raise SystemExit(1)

tr = tf.transform.translation
qr = tf.transform.rotation

t_LO = np.array(
    [tr.x, tr.y, tr.z],
    dtype=np.float64
)

R_LO = quat_to_rot(
    np.array(
        [qr.x, qr.y, qr.z, qr.w],
        dtype=np.float64
    )
)

T_WO = make_T(R_WO, t_WO)
T_LO = make_T(R_LO, t_LO)

# We know:
#   world <- optical
#   link  <- optical
#
# Therefore:
#   world <- link
# = (world <- optical) * (optical <- link)
T_WL = T_WO @ np.linalg.inv(T_LO)

R_WL = T_WL[:3, :3]
t_WL = T_WL[:3, 3]
q_WL = rot_to_quat(R_WL)

print("===== DERIVED STATIC TF =====")
print("world -> camera_link")
print(
    "t = "
    f"[{t_WL[0]:+.9f}, "
    f"{t_WL[1]:+.9f}, "
    f"{t_WL[2]:+.9f}]"
)
print(
    "q = "
    f"[{q_WL[0]:+.9f}, "
    f"{q_WL[1]:+.9f}, "
    f"{q_WL[2]:+.9f}, "
    f"{q_WL[3]:+.9f}]"
)

msg = TransformStamped()

msg.header.stamp = node.get_clock().now().to_msg()
msg.header.frame_id = 'world'
msg.child_frame_id = 'camera_link'

msg.transform.translation.x = float(t_WL[0])
msg.transform.translation.y = float(t_WL[1])
msg.transform.translation.z = float(t_WL[2])

msg.transform.rotation.x = float(q_WL[0])
msg.transform.rotation.y = float(q_WL[1])
msg.transform.rotation.z = float(q_WL[2])
msg.transform.rotation.w = float(q_WL[3])

node.broadcaster.sendTransform(msg)

print("STATIC TF BROADCAST = PASS")
print("Keeping node alive...")

try:
    rclpy.spin(node)
except KeyboardInterrupt:
    pass

node.destroy_node()
rclpy.shutdown()
