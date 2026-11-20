#!/usr/bin/env python3

import numpy as np

import rclpy
import message_filters

from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
    qos_profile_sensor_data,
)

from sensor_msgs.msg import (
    Image,
    CameraInfo,
)

from geometry_msgs.msg import PointStamped

from vision_msgs.msg import (
    Detection2DArray,
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)

from tf2_ros import (
    Buffer,
    TransformListener,
)

from tf2_geometry_msgs import (
    do_transform_point,
)


class MaskLocalizerNode(Node):

    def __init__(self):
        super().__init__('mask_localizer')

        self.declare_parameter(
            'detections_topic',
            '/object_tracker/detections'
        )

        self.declare_parameter(
            'depth_topic',
            '/camera/depth_image'
        )

        self.declare_parameter(
            'mask_topic',
            '/object_segmenter/instance_mask'
        )

        self.declare_parameter(
            'camera_info_topic',
            '/camera/camera_info'
        )

        self.declare_parameter(
            'output_topic',
            '/object_localizer/detections_3d_mask'
        )

        self.declare_parameter(
            'optical_frame',
            'realsense_camera_optical_frame'
        )

        self.declare_parameter(
            'world_frame',
            'world'
        )

        # In real RealSense mode, the aligned depth image
        # already carries the correct optical frame.
        # Keep False by default so existing simulation
        # behaviour is unchanged.
        self.declare_parameter(
            'use_depth_frame_id',
            False
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
            10
        )

        self.detections_topic = (
            self.get_parameter(
                'detections_topic'
            ).value
        )

        self.depth_topic = (
            self.get_parameter(
                'depth_topic'
            ).value
        )

        self.mask_topic = (
            self.get_parameter(
                'mask_topic'
            ).value
        )

        self.camera_info_topic = (
            self.get_parameter(
                'camera_info_topic'
            ).value
        )

        self.output_topic = (
            self.get_parameter(
                'output_topic'
            ).value
        )

        self.optical_frame = (
            self.get_parameter(
                'optical_frame'
            ).value
        )

        self.world_frame = (
            self.get_parameter(
                'world_frame'
            ).value
        )

        self.use_depth_frame_id = bool(
            self.get_parameter(
                'use_depth_frame_id'
            ).value
        )

        self.min_depth = float(
            self.get_parameter(
                'min_depth'
            ).value
        )

        self.max_depth = float(
            self.get_parameter(
                'max_depth'
            ).value
        )

        self.min_valid_pixels = int(
            self.get_parameter(
                'min_valid_pixels'
            ).value
        )

        # ----------------------------------------------
        # Camera intrinsics
        # ----------------------------------------------

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.camera_info_sub = (
            self.create_subscription(
                CameraInfo,
                self.camera_info_topic,
                self.camera_info_callback,
                qos_profile_sensor_data
            )
        )

        # ----------------------------------------------
        # TF
        # ----------------------------------------------

        self.tf_buffer = Buffer()

        self.tf_listener = (
            TransformListener(
                self.tf_buffer,
                self
            )
        )

        # ----------------------------------------------
        # Synchronization
        #
        # Important:
        # SAM2 may publish 100+ ms later in wall time,
        # but the header stamp still belongs to the
        # original camera frame.
        #
        # message_filters therefore matches by stamp,
        # not by arrival time.
        # ----------------------------------------------

        self.detection_sub = (
            message_filters.Subscriber(
                self,
                Detection2DArray,
                self.detections_topic,
                qos_profile=10
            )
        )

        depth_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.depth_sub = (
            message_filters.Subscriber(
                self,
                Image,
                self.depth_topic,
                qos_profile=depth_qos
            )
        )

        mask_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.mask_sub = (
            message_filters.Subscriber(
                self,
                Image,
                self.mask_topic,
                qos_profile=mask_qos
            )
        )

        self.sync = (
            message_filters.ApproximateTimeSynchronizer(
                [
                    self.detection_sub,
                    self.depth_sub,
                    self.mask_sub,
                ],
                queue_size=40,
                slop=0.005,
                allow_headerless=False
            )
        )

        self.sync.registerCallback(
            self.synced_callback
        )

        self.output_pub = (
            self.create_publisher(
                Detection3DArray,
                self.output_topic,
                10
            )
        )

        self.frame_count = 0

        self.get_logger().info(
            'Mask localizer started'
        )

        self.get_logger().info(
            f'Detections: {self.detections_topic}'
        )

        self.get_logger().info(
            f'Depth: {self.depth_topic}'
        )

        self.get_logger().info(
            f'Mask: {self.mask_topic}'
        )

        self.get_logger().info(
            f'Output: {self.output_topic}'
        )

    def camera_info_callback(
        self,
        msg
    ):

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

    @staticmethod
    def track_label(track_id):

        try:

            label = int(
                track_id.rsplit(
                    '_',
                    1
                )[-1]
            )

            if 1 <= label <= 65535:
                return label

        except Exception:
            pass

        return None

    @staticmethod
    def class_name(detection):

        if not detection.results:
            return ''

        return (
            detection.results[0]
            .hypothesis
            .class_id
        )

    def depth_to_numpy(
        self,
        msg
    ):

        # Gazebo / some pipelines:
        #   32FC1, meters.
        #
        # RealSense raw/aligned depth:
        #   16UC1, millimeters.
        #
        # Internally this node always returns float32 meters.

        if msg.encoding == '32FC1':

            dtype = (
                '>f4'
                if msg.is_bigendian
                else '<f4'
            )

            row_width = (
                msg.step // 4
            )

            depth = np.frombuffer(
                msg.data,
                dtype=dtype
            ).reshape(
                msg.height,
                row_width
            )

            depth = depth[
                :,
                :msg.width
            ].astype(
                np.float32,
                copy=False
            )

            return depth

        if msg.encoding == '16UC1':

            dtype = (
                '>u2'
                if msg.is_bigendian
                else '<u2'
            )

            row_width = (
                msg.step // 2
            )

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

            # RealSense depth units are represented here
            # as millimetres in the ROS image.
            depth_m = (
                depth_mm.astype(
                    np.float32
                )
                * 0.001
            )

            return depth_m

        self.get_logger().warning(
            'Unsupported depth encoding: '
            f'{msg.encoding}'
        )

        return None

    def mask_to_numpy(
        self,
        msg
    ):

        if msg.encoding != 'mono16':

            self.get_logger().warning(
                'Unsupported mask encoding: '
                f'{msg.encoding}'
            )

            return None

        row_width = (
            msg.step // 2
        )

        mask = np.frombuffer(
            msg.data,
            dtype=np.uint16
        ).reshape(
            msg.height,
            row_width
        )

        return mask[
            :,
            :msg.width
        ]

    def estimate_3d_from_mask(
        self,
        depth,
        label_map,
        label
    ):

        instance_mask = (
            label_map == label
        )

        mask_pixels = int(
            np.count_nonzero(
                instance_mask
            )
        )

        if (
            mask_pixels
            <
            self.min_valid_pixels
        ):
            return None

        valid = (
            instance_mask
            &
            np.isfinite(depth)
            &
            (depth > self.min_depth)
            &
            (depth < self.max_depth)
        )

        valid_count = int(
            np.count_nonzero(valid)
        )

        if (
            valid_count
            <
            self.min_valid_pixels
        ):
            return None

        v_pixels, u_pixels = (
            np.nonzero(valid)
        )

        z_values = (
            depth[valid]
            .astype(np.float64)
        )

        # ----------------------------------------------
        # Robust depth outlier rejection
        # ----------------------------------------------

        z_median = float(
            np.median(z_values)
        )

        deviation = np.abs(
            z_values - z_median
        )

        mad = float(
            np.median(deviation)
        )

        if mad > 1e-6:

            sigma = (
                1.4826 * mad
            )

            inliers = (
                deviation
                <=
                3.0 * sigma
            )

            if (
                np.count_nonzero(inliers)
                >=
                self.min_valid_pixels
            ):

                z_values = (
                    z_values[inliers]
                )

                u_pixels = (
                    u_pixels[inliers]
                )

                v_pixels = (
                    v_pixels[inliers]
                )

        # ----------------------------------------------
        # Back-project every valid mask pixel.
        #
        # This is better than:
        # "mask centroid + one median depth"
        #
        # because each RGB-D pixel contributes its own
        # measured depth.
        # ----------------------------------------------

        u = (
            u_pixels.astype(
                np.float64
            )
        )

        v = (
            v_pixels.astype(
                np.float64
            )
        )

        z = z_values

        x = (
            (u - self.cx)
            *
            z
            /
            self.fx
        )

        y = (
            (v - self.cy)
            *
            z
            /
            self.fy
        )

        # Robust 3D centroid in optical coordinates.
        optical_x = float(
            np.median(x)
        )

        optical_y = float(
            np.median(y)
        )

        optical_z = float(
            np.median(z)
        )

        return {
            'x': optical_x,
            'y': optical_y,
            'z': optical_z,
            'mask_pixels': mask_pixels,
            'valid_pixels': len(z),
            'mad': mad,
        }

    def transform_to_world(
        self,
        point
    ):

        stamp = Time.from_msg(
            point.header.stamp
        )

        transform = (
            self.tf_buffer.lookup_transform(
                self.world_frame,
                point.header.frame_id,
                stamp,
                timeout=Duration(
                    seconds=0.05
                )
            )
        )

        return do_transform_point(
            point,
            transform
        )

    def synced_callback(
        self,
        detections_msg,
        depth_msg,
        mask_msg
    ):

        if self.fx is None:
            return

        depth = self.depth_to_numpy(
            depth_msg
        )

        label_map = (
            self.mask_to_numpy(
                mask_msg
            )
        )

        if (
            depth is None
            or
            label_map is None
        ):
            return

        self.frame_count += 1

        output = Detection3DArray()

        output.header.stamp = (
            depth_msg.header.stamp
        )

        output.header.frame_id = (
            self.world_frame
        )

        logs = []

        for detection in (
            detections_msg.detections
        ):

            label = self.track_label(
                detection.id
            )

            if label is None:
                continue

            estimate = (
                self.estimate_3d_from_mask(
                    depth,
                    label_map,
                    label
                )
            )

            if estimate is None:
                continue

            point = PointStamped()

            point.header.stamp = (
                depth_msg.header.stamp
            )

            if (
                self.use_depth_frame_id
                and
                depth_msg.header.frame_id
            ):

                point.header.frame_id = (
                    depth_msg.header.frame_id
                )

            else:

                point.header.frame_id = (
                    self.optical_frame
                )

            point.point.x = estimate['x']
            point.point.y = estimate['y']
            point.point.z = estimate['z']

            try:

                world = (
                    self.transform_to_world(
                        point
                    )
                )

            except Exception as exc:

                self.get_logger().warning(
                    'TF lookup failed: '
                    f'{exc}'
                )

                continue

            detection3d = (
                Detection3D()
            )

            detection3d.header = (
                output.header
            )

            detection3d.id = (
                detection.id
            )

            result = (
                ObjectHypothesisWithPose()
            )

            if detection.results:

                result.hypothesis.class_id = (
                    detection.results[0]
                    .hypothesis
                    .class_id
                )

                result.hypothesis.score = (
                    detection.results[0]
                    .hypothesis
                    .score
                )

            result.pose.pose.position.x = (
                world.point.x
            )

            result.pose.pose.position.y = (
                world.point.y
            )

            result.pose.pose.position.z = (
                world.point.z
            )

            result.pose.pose.orientation.w = 1.0

            detection3d.results.append(
                result
            )

            detection3d.bbox.center.position.x = (
                world.point.x
            )

            detection3d.bbox.center.position.y = (
                world.point.y
            )

            detection3d.bbox.center.position.z = (
                world.point.z
            )

            detection3d.bbox.center.orientation.w = 1.0

            output.detections.append(
                detection3d
            )

            logs.append(
                (
                    detection.id,
                    self.class_name(
                        detection
                    ),
                    world.point.x,
                    world.point.y,
                    world.point.z,
                    estimate['mask_pixels'],
                    estimate['valid_pixels'],
                    estimate['mad'],
                )
            )

        self.output_pub.publish(
            output
        )

        self.get_logger().info(
            f'frame={self.frame_count} | '
            f'3D={len(output.detections)}'
        )

        for item in logs:

            (
                track_id,
                class_id,
                x,
                y,
                z,
                mask_pixels,
                valid_pixels,
                mad,
            ) = item

            self.get_logger().info(
                f'  {track_id:<10} '
                f'{class_id:<10} '
                f'world=('
                f'{x:+.4f},'
                f'{y:+.4f},'
                f'{z:+.4f})m '
                f'mask={mask_pixels} '
                f'depth={valid_pixels} '
                f'mad={mad:.5f}'
            )


def main(args=None):

    rclpy.init(args=args)

    node = MaskLocalizerNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
