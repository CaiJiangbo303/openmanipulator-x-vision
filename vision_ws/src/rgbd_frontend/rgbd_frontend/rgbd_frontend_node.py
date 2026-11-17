#!/usr/bin/env python3

import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.duration import Duration
from rclpy.time import Time

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped

from message_filters import Subscriber, ApproximateTimeSynchronizer

from tf2_ros import Buffer, TransformListener, TransformException
from tf2_geometry_msgs import do_transform_point


class RGBDFrontendNode(Node):

    def __init__(self):
        super().__init__('rgbd_frontend')

        self.rgb_topic = '/camera/image'
        self.depth_topic = '/camera/depth_image'
        self.info_topic = '/camera/camera_info'

        self.optical_frame = 'realsense_camera_optical_frame'
        self.base_frame = 'dji_f550/base_link'
        self.world_frame = 'world'

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.synced_count = 0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        self.rgb_sub = Subscriber(
            self,
            Image,
            self.rgb_topic,
            qos_profile=qos_profile_sensor_data
        )

        self.depth_sub = Subscriber(
            self,
            Image,
            self.depth_topic,
            qos_profile=qos_profile_sensor_data
        )

        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=20,
            slop=0.02
        )

        self.sync.registerCallback(self.rgbd_callback)

        self.info_sub = self.create_subscription(
            CameraInfo,
            self.info_topic,
            self.info_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info('RGB-D frontend started.')
        self.get_logger().info(
            f'TF chain: {self.optical_frame} '
            f'-> {self.base_frame} '
            f'-> {self.world_frame}'
        )

    def info_callback(self, msg):

        if self.fx is not None:
            return

        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

        self.get_logger().info(
            f'CameraInfo | '
            f'fx={self.fx:.2f} '
            f'fy={self.fy:.2f} '
            f'cx={self.cx:.2f} '
            f'cy={self.cy:.2f}'
        )

    def rgbd_callback(self, rgb_msg, depth_msg):

        if self.fx is None:
            return

        self.synced_count += 1

        if self.synced_count % 30 != 0:
            return

        if depth_msg.encoding != '32FC1':
            self.get_logger().warning(
                f'Unsupported depth encoding: {depth_msg.encoding}'
            )
            return

        if rgb_msg.encoding != 'rgb8':
            self.get_logger().warning(
                f'Unsupported RGB encoding: {rgb_msg.encoding}'
            )
            return

        rgb = np.ndarray(
            shape=(rgb_msg.height, rgb_msg.width, 3),
            dtype=np.uint8,
            buffer=rgb_msg.data,
            strides=(rgb_msg.step, 3, 1)
        )

        dtype = np.dtype(
            '>f4' if depth_msg.is_bigendian else '<f4'
        )

        depth = np.ndarray(
            shape=(depth_msg.height, depth_msg.width),
            dtype=dtype,
            buffer=depth_msg.data,
            strides=(depth_msg.step, 4)
        )

        r = rgb[:, :, 0].astype(np.int16)
        g = rgb[:, :, 1].astype(np.int16)
        b = rgb[:, :, 2].astype(np.int16)

        color_masks = {
            'RED': (
                (r > 100) &
                (r > g * 1.4) &
                (r > b * 1.4)
            ),

            'GREEN': (
                (g > 100) &
                (g > r * 1.4) &
                (g > b * 1.4)
            ),

            'BLUE': (
                (b > 100) &
                (b > r * 1.4) &
                (b > g * 1.4)
            ),
        }

        # 当前仿真验证阶段：
        # 排除 F550 / 机械臂自身的近距离颜色区域
        depth_valid = (
            np.isfinite(depth) &
            (depth > 0.5) &
            (depth < 2.0)
        )

        stamp = Time.from_msg(rgb_msg.header.stamp)

        try:

            tf_base = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.optical_frame,
                stamp,
                timeout=Duration(seconds=0.2)
            )

            tf_world = self.tf_buffer.lookup_transform(
                self.world_frame,
                self.optical_frame,
                stamp,
                timeout=Duration(seconds=0.2)
            )

        except TransformException as exc:

            self.get_logger().warning(
                f'TF lookup failed: {exc}'
            )
            return

        self.get_logger().info(
            f'===== RGB-D targets | pairs={self.synced_count} ====='
        )

        for name, color_mask in color_masks.items():

            mask = color_mask & depth_valid

            ys, xs = np.where(mask)

            if len(xs) < 20:
                self.get_logger().info(
                    f'{name:<5} | NOT FOUND'
                )
                continue

            # 当前用目标区域像素中位数作为二维代表点
            u = int(np.median(xs))
            v = int(np.median(ys))

            # 当前用目标区域所有有效深度的中位数
            z = float(np.median(depth[mask]))

            # 像素 + 深度 -> optical frame
            x = (u - self.cx) * z / self.fx
            y = (v - self.cy) * z / self.fy

            point_optical = PointStamped()

            point_optical.header.stamp = rgb_msg.header.stamp
            point_optical.header.frame_id = self.optical_frame

            point_optical.point.x = float(x)
            point_optical.point.y = float(y)
            point_optical.point.z = float(z)

            # optical -> F550 base_link
            point_base = do_transform_point(
                point_optical,
                tf_base
            )

            # optical -> world
            point_world = do_transform_point(
                point_optical,
                tf_world
            )

            self.get_logger().info(
                f'{name:<5} | '
                f'pixel=({u:3d},{v:3d}) | '
                f'optical=('
                f'{x:+.3f}, '
                f'{y:+.3f}, '
                f'{z:+.3f}) | '
                f'base=('
                f'{point_base.point.x:+.3f}, '
                f'{point_base.point.y:+.3f}, '
                f'{point_base.point.z:+.3f}) | '
                f'world=('
                f'{point_world.point.x:+.3f}, '
                f'{point_world.point.y:+.3f}, '
                f'{point_world.point.z:+.3f}) m'
            )


def main(args=None):

    rclpy.init(args=args)

    node = RGBDFrontendNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
