#!/usr/bin/env python3

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.qos import qos_profile_sensor_data

import message_filters

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped

from vision_msgs.msg import (
    Detection2DArray,
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)

from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point


class ObjectLocalizerNode(Node):

    def __init__(self):
        super().__init__('object_localizer')

        self.declare_parameter(
            'detections_topic',
            '/object_detector/detections'
        )
        self.declare_parameter(
            'depth_topic',
            '/camera/depth_image'
        )
        self.declare_parameter(
            'camera_info_topic',
            '/camera/camera_info'
        )
        self.declare_parameter(
            'output_topic',
            '/object_localizer/detections_3d'
        )
        self.declare_parameter(
            'optical_frame',
            'realsense_camera_optical_frame'
        )
        self.declare_parameter(
            'world_frame',
            'world'
        )
        self.declare_parameter(
            'use_depth_frame_id',
            False
        )
        self.declare_parameter(
            'roi_scale',
            0.60
        )
        self.declare_parameter(
            'min_depth',
            0.05
        )
        self.declare_parameter(
            'max_depth',
            3.0
        )
        self.declare_parameter(
            'min_valid_pixels',
            20
        )

        self.detections_topic = self.get_parameter(
            'detections_topic'
        ).value

        self.depth_topic = self.get_parameter(
            'depth_topic'
        ).value

        self.camera_info_topic = self.get_parameter(
            'camera_info_topic'
        ).value

        self.output_topic = self.get_parameter(
            'output_topic'
        ).value

        self.optical_frame = self.get_parameter(
            'optical_frame'
        ).value

        self.world_frame = self.get_parameter(
            'world_frame'
        ).value

        self.use_depth_frame_id = bool(
            self.get_parameter(
                'use_depth_frame_id'
            ).value
        )

        self.roi_scale = float(
            self.get_parameter('roi_scale').value
        )

        self.min_depth = float(
            self.get_parameter('min_depth').value
        )

        self.max_depth = float(
            self.get_parameter('max_depth').value
        )

        self.min_valid_pixels = int(
            self.get_parameter(
                'min_valid_pixels'
            ).value
        )

        # Camera intrinsics
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            qos_profile_sensor_data
        )

        # TF
        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        # Detection2D keeps the original RGB timestamp.
        # Depth comes from the same RGB-D sensor.
        self.detection_sub = message_filters.Subscriber(
            self,
            Detection2DArray,
            self.detections_topic,
            qos_profile=10
        )

        self.depth_sub = message_filters.Subscriber(
            self,
            Image,
            self.depth_topic,
            qos_profile=qos_profile_sensor_data
        )

        self.sync = (
            message_filters.ApproximateTimeSynchronizer(
                [
                    self.detection_sub,
                    self.depth_sub
                ],
                queue_size=20,
                slop=0.005,
                allow_headerless=False
            )
        )

        self.sync.registerCallback(
            self.synced_callback
        )

        self.output_pub = self.create_publisher(
            Detection3DArray,
            self.output_topic,
            10
        )

        self.frame_count = 0

        self.get_logger().info(
            'Object localizer started'
        )

        self.get_logger().info(
            f'Input detections: '
            f'{self.detections_topic}'
        )

        self.get_logger().info(
            f'Input depth: {self.depth_topic}'
        )

        self.get_logger().info(
            f'Output: {self.output_topic}'
        )

        self.get_logger().info(
            f'TF fallback: {self.optical_frame} '
            f'-> {self.world_frame}'
        )

        self.get_logger().info(
            'Depth frame source: '
            + (
                'message header'
                if self.use_depth_frame_id
                else 'configured optical_frame'
            )
        )

    def camera_info_callback(self, msg):

        if self.fx is not None:
            return

        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])

        self.get_logger().info(
            'Camera intrinsics loaded | '
            f'fx={self.fx:.2f} '
            f'fy={self.fy:.2f} '
            f'cx={self.cx:.2f} '
            f'cy={self.cy:.2f}'
        )

    def depth_to_numpy(self, msg):

        # Gazebo / some RGB-D pipelines:
        #   32FC1, metres.
        #
        # RealSense raw/aligned depth:
        #   16UC1, millimetres.
        #
        # Internally the fast localizer always works
        # with float32 metres.

        if msg.encoding == '32FC1':

            dtype = (
                '>f4'
                if msg.is_bigendian
                else '<f4'
            )

            row_width = msg.step // 4

            depth = np.frombuffer(
                msg.data,
                dtype=dtype
            ).reshape(
                msg.height,
                row_width
            )

            return depth[
                :,
                :msg.width
            ].astype(
                np.float32,
                copy=False
            )

        if msg.encoding == '16UC1':

            dtype = (
                '>u2'
                if msg.is_bigendian
                else '<u2'
            )

            row_width = msg.step // 2

            depth_mm = np.frombuffer(
                msg.data,
                dtype=dtype
            ).reshape(
                msg.height,
                row_width
            )

            depth_mm = depth_mm[
                :,
                :msg.width
            ]

            return (
                depth_mm.astype(
                    np.float32
                )
                * 0.001
            )

        self.get_logger().warning(
            'Unsupported depth encoding: '
            f'{msg.encoding}'
        )

        return None

    def estimate_depth(
        self,
        depth,
        detection
    ):

        bbox = detection.bbox

        center_u = float(
            bbox.center.position.x
        )

        center_v = float(
            bbox.center.position.y
        )

        width = float(
            bbox.size_x
        )

        height = float(
            bbox.size_y
        )

        if width <= 1.0 or height <= 1.0:
            return None

        roi_width = (
            width * self.roi_scale
        )

        roi_height = (
            height * self.roi_scale
        )

        x1 = int(round(
            center_u - roi_width / 2.0
        ))

        x2 = int(round(
            center_u + roi_width / 2.0
        ))

        y1 = int(round(
            center_v - roi_height / 2.0
        ))

        y2 = int(round(
            center_v + roi_height / 2.0
        ))

        x1 = max(
            0,
            min(depth.shape[1] - 1, x1)
        )

        x2 = max(
            x1 + 1,
            min(depth.shape[1], x2)
        )

        y1 = max(
            0,
            min(depth.shape[0] - 1, y1)
        )

        y2 = max(
            y1 + 1,
            min(depth.shape[0], y2)
        )

        roi = depth[
            y1:y2,
            x1:x2
        ]

        valid_mask = (
            np.isfinite(roi)
            &
            (roi > self.min_depth)
            &
            (roi < self.max_depth)
        )

        valid_count = int(
            np.count_nonzero(
                valid_mask
            )
        )

        if valid_count < self.min_valid_pixels:
            return None

        local_y, local_x = (
            np.nonzero(valid_mask)
        )

        z_values = (
            roi[valid_mask]
            .astype(np.float64)
        )

        z_median = float(
            np.median(z_values)
        )

        absolute_deviation = np.abs(
            z_values - z_median
        )

        mad = float(
            np.median(
                absolute_deviation
            )
        )

        if mad > 1e-6:

            robust_sigma = (
                1.4826 * mad
            )

            inlier_mask = (
                absolute_deviation
                <=
                3.0 * robust_sigma
            )

            if (
                np.count_nonzero(inlier_mask)
                >=
                self.min_valid_pixels
            ):

                z_values = (
                    z_values[
                        inlier_mask
                    ]
                )

                local_x = (
                    local_x[
                        inlier_mask
                    ]
                )

                local_y = (
                    local_y[
                        inlier_mask
                    ]
                )

        z = float(
            np.median(z_values)
        )

        u = float(
            np.median(
                local_x + x1
            )
        )

        v = float(
            np.median(
                local_y + y1
            )
        )

        return (
            u,
            v,
            z,
            mad,
            len(z_values)
        )

    def project_to_optical(
        self,
        u,
        v,
        z,
        stamp,
        frame_id
    ):

        x = (
            (u - self.cx)
            * z
            / self.fx
        )

        y = (
            (v - self.cy)
            * z
            / self.fy
        )

        point = PointStamped()

        point.header.stamp = stamp
        point.header.frame_id = frame_id

        point.point.x = float(x)
        point.point.y = float(y)
        point.point.z = float(z)

        return point

    def transform_to_world(
        self,
        optical_point
    ):

        stamp = Time.from_msg(
            optical_point.header.stamp
        )

        transform = (
            self.tf_buffer.lookup_transform(
                self.world_frame,
                optical_point.header.frame_id,
                stamp,
                timeout=Duration(
                    seconds=0.05
                )
            )
        )

        return do_transform_point(
            optical_point,
            transform
        )

    @staticmethod
    def timestamp_ns(stamp):

        return (
            int(stamp.sec)
            * 1_000_000_000
            +
            int(stamp.nanosec)
        )

    def synced_callback(
        self,
        detections_msg,
        depth_msg
    ):

        if self.fx is None:
            return

        depth = self.depth_to_numpy(
            depth_msg
        )

        if depth is None:
            return

        self.frame_count += 1

        detection_time = (
            self.timestamp_ns(
                detections_msg.header.stamp
            )
        )

        depth_time = (
            self.timestamp_ns(
                depth_msg.header.stamp
            )
        )

        sync_ms = abs(
            detection_time - depth_time
        ) / 1e6

        output = Detection3DArray()

        output.header.stamp = (
            depth_msg.header.stamp
        )

        output.header.frame_id = (
            self.world_frame
        )

        debug_lines = []

        for detection2d in (
            detections_msg.detections
        ):

            if not detection2d.results:
                continue

            depth_result = (
                self.estimate_depth(
                    depth,
                    detection2d
                )
            )

            if depth_result is None:
                continue

            (
                u,
                v,
                z,
                mad,
                valid_pixels
            ) = depth_result

            source_frame = (
                depth_msg.header.frame_id
                if (
                    self.use_depth_frame_id
                    and
                    depth_msg.header.frame_id
                )
                else
                self.optical_frame
            )

            optical_point = (
                self.project_to_optical(
                    u,
                    v,
                    z,
                    depth_msg.header.stamp,
                    source_frame
                )
            )

            try:
                world_point = (
                    self.transform_to_world(
                        optical_point
                    )
                )

            except Exception as exc:
                self.get_logger().warning(
                    'TF lookup failed: '
                    f'{exc}'
                )
                continue

            source_result = (
                detection2d.results[0]
            )

            detection3d = (
                Detection3D()
            )

            detection3d.header = (
                output.header
            )

            # Keep ID untouched.
            # It is empty now; tracker will own it later.
            detection3d.id = (
                detection2d.id
            )

            result3d = (
                ObjectHypothesisWithPose()
            )

            result3d.hypothesis.class_id = (
                source_result
                .hypothesis
                .class_id
            )

            result3d.hypothesis.score = (
                source_result
                .hypothesis
                .score
            )

            result3d.pose.pose.position.x = (
                world_point.point.x
            )

            result3d.pose.pose.position.y = (
                world_point.point.y
            )

            result3d.pose.pose.position.z = (
                world_point.point.z
            )

            # Orientation is not observable yet.
            # Identity is only a neutral quaternion.
            result3d.pose.pose.orientation.w = 1.0

            detection3d.results.append(
                result3d
            )

            # For now this is a localized 3D point,
            # not a true reconstructed 3D bounding box.
            detection3d.bbox.center.position.x = (
                world_point.point.x
            )

            detection3d.bbox.center.position.y = (
                world_point.point.y
            )

            detection3d.bbox.center.position.z = (
                world_point.point.z
            )

            detection3d.bbox.center.orientation.w = 1.0

            # Size intentionally stays zero.
            # Later mask / point-cloud geometry will estimate it.
            detection3d.bbox.size.x = 0.0
            detection3d.bbox.size.y = 0.0
            detection3d.bbox.size.z = 0.0

            output.detections.append(
                detection3d
            )

            debug_lines.append(
                (
                    source_result
                    .hypothesis
                    .class_id,
                    source_result
                    .hypothesis
                    .score,
                    u,
                    v,
                    z,
                    world_point.point.x,
                    world_point.point.y,
                    world_point.point.z,
                    valid_pixels,
                    mad,
                )
            )

        self.output_pub.publish(
            output
        )

        if (
            self.frame_count % 20 == 0
        ):

            self.get_logger().info(
                f'frame={self.frame_count} | '
                f'2D={len(detections_msg.detections)} | '
                f'3D={len(output.detections)} | '
                f'sync={sync_ms:.3f} ms'
            )

            for item in debug_lines[:5]:

                (
                    class_name,
                    score,
                    u,
                    v,
                    z,
                    world_x,
                    world_y,
                    world_z,
                    valid_pixels,
                    mad,
                ) = item

                self.get_logger().info(
                    f'  {class_name:<15} '
                    f'conf={score:.3f} | '
                    f'pixel=({u:.1f},{v:.1f}) '
                    f'depth={z:.3f}m | '
                    f'world=('
                    f'{world_x:+.3f},'
                    f'{world_y:+.3f},'
                    f'{world_z:+.3f})m | '
                    f'pixels={valid_pixels} '
                    f'mad={mad:.4f}'
                )


def main(args=None):

    rclpy.init(args=args)

    node = ObjectLocalizerNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
