#!/usr/bin/env python3

from collections import deque
from time import monotonic_ns

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.qos import (
    qos_profile_sensor_data,
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)


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
        self.declare_parameter(
            'depth_cache_size',
            16
        )
        self.declare_parameter(
            'detection_cache_size',
            16
        )
        self.declare_parameter(
            'detection_cache_ttl_sec',
            0.50
        )
        self.declare_parameter(
            'max_depth_time_diff_ms',
            0.0
        )
        self.declare_parameter(
            'tf_pending_queue_size',
            16
        )
        self.declare_parameter(
            'tf_pending_ttl_sec',
            0.50
        )
        self.declare_parameter(
            'tf_poll_period_sec',
            0.02
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

        self.depth_cache_size = max(
            2,
            int(
                self.get_parameter(
                    'depth_cache_size'
                ).value
            )
        )

        self.detection_cache_size = max(
            2,
            int(
                self.get_parameter(
                    'detection_cache_size'
                ).value
            )
        )

        self.detection_cache_ttl_sec = max(
            0.05,
            float(
                self.get_parameter(
                    'detection_cache_ttl_sec'
                ).value
            )
        )

        self.max_depth_time_diff_ms = float(
            self.get_parameter(
                'max_depth_time_diff_ms'
            ).value
        )

        self.tf_pending_queue_size = max(
            2,
            int(
                self.get_parameter(
                    'tf_pending_queue_size'
                ).value
            )
        )

        self.tf_pending_ttl_sec = max(
            0.05,
            float(
                self.get_parameter(
                    'tf_pending_ttl_sec'
                ).value
            )
        )

        self.tf_poll_period_sec = max(
            0.005,
            float(
                self.get_parameter(
                    'tf_poll_period_sec'
                ).value
            )
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

        # Source-stamp cache:
        # depth arrival timing no longer has to coincide with detections.
        self.depth_cache = deque(
            maxlen=self.depth_cache_size
        )

        # Keep only a small bounded set of unmatched detections.
        # This prevents arrival-order jitter from destroying exact
        # source-stamp matches while still bounding latency/memory.
        self.pending_detections = deque(
            maxlen=self.detection_cache_size
        )

        self.detection_sub = self.create_subscription(
            Detection2DArray,
            self.detections_topic,
            self.detection_callback,
            10
        )

        # Fast3D depth stream:
        # reliable delivery is required because RGB-D matching below
        # currently requires an exact source timestamp.
        depth_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            depth_qos
        )

        self.output_pub = self.create_publisher(
            Detection3DArray,
            self.output_topic,
            10
        )

        self.frame_count = 0

        # Matched RGB-D frames waiting for their exact-stamp TF.
        #
        # The queue is bounded. When TF catches up we process the
        # newest frame that is already transformable and discard
        # older ready frames. This keeps freshness above completeness.
        self.pending_tf_pairs = deque(
            maxlen=self.tf_pending_queue_size
        )

        self.tf_gate_timer = self.create_timer(
            self.tf_poll_period_sec,
            self.retry_tf_pending
        )

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

    @staticmethod
    def timestamp_ns(stamp):

        return (
            int(stamp.sec)
            * 1_000_000_000
            +
            int(stamp.nanosec)
        )

    def find_matching_depth(
        self,
        detections_msg
    ):

        if not self.depth_cache:
            return None

        detection_time = self.timestamp_ns(
            detections_msg.header.stamp
        )

        best_depth = None
        best_diff_ns = None

        # Exact source stamp wins immediately.
        # Otherwise keep the nearest cached depth frame.
        for depth_msg in reversed(
            self.depth_cache
        ):

            depth_time = self.timestamp_ns(
                depth_msg.header.stamp
            )

            diff_ns = abs(
                detection_time - depth_time
            )

            if diff_ns == 0:
                return depth_msg

            if (
                best_diff_ns is None
                or
                diff_ns < best_diff_ns
            ):
                best_diff_ns = diff_ns
                best_depth = depth_msg

        if best_depth is None:
            return None

        diff_ms = best_diff_ns / 1e6

        # Safety gate: never force-pair measurements
        # whose source timestamps are too far apart.
        if (
            diff_ms
            > self.max_depth_time_diff_ms
        ):
            return None

        return best_depth

    def detection_callback(
        self,
        detections_msg
    ):

        self.prune_pending_detections()

        depth_msg = self.find_matching_depth(
            detections_msg
        )

        if depth_msg is not None:

            self.process_pair(
                detections_msg,
                depth_msg
            )

            return

        self.pending_detections.append(
            (
                monotonic_ns(),
                detections_msg
            )
        )

    def find_matching_detection(
        self,
        depth_msg
    ):

        if not self.pending_detections:
            return None

        depth_time = self.timestamp_ns(
            depth_msg.header.stamp
        )

        best_index = None
        best_diff_ns = None

        # Newest-first so ties prefer fresher detections.
        for index in range(
            len(self.pending_detections) - 1,
            -1,
            -1
        ):

            _, detections_msg = (
                self.pending_detections[index]
            )

            detection_time = self.timestamp_ns(
                detections_msg.header.stamp
            )

            diff_ns = abs(
                detection_time - depth_time
            )

            if diff_ns == 0:
                return index

            if (
                best_diff_ns is None
                or
                diff_ns < best_diff_ns
            ):
                best_diff_ns = diff_ns
                best_index = index

        if best_index is None:
            return None

        if (
            best_diff_ns / 1e6
            >
            self.max_depth_time_diff_ms
        ):
            return None

        return best_index

    def prune_pending_detections(
        self
    ):

        # IMPORTANT:
        # prune by local receipt age, never by source timestamp.
        # GZ / DDS callback arrival order is not guaranteed to
        # be monotonic in source time.
        if not self.pending_detections:
            return

        now_ns = monotonic_ns()

        ttl_ns = int(
            self.detection_cache_ttl_sec
            * 1_000_000_000
        )

        while self.pending_detections:

            receipt_ns, _ = (
                self.pending_detections[0]
            )

            if (
                now_ns - receipt_ns
                <=
                ttl_ns
            ):
                break

            self.pending_detections.popleft()

    def depth_callback(
        self,
        depth_msg
    ):

        self.prune_pending_detections()

        self.depth_cache.append(
            depth_msg
        )

        match_index = (
            self.find_matching_detection(
                depth_msg
            )
        )

        if match_index is not None:

            _, detections_msg = (
                self.pending_detections[
                    match_index
                ]
            )

            del self.pending_detections[
                match_index
            ]

            self.process_pair(
                detections_msg,
                depth_msg
            )

    def depth_source_frame(
        self,
        depth_msg
    ):

        return (
            depth_msg.header.frame_id
            if (
                self.use_depth_frame_id
                and
                depth_msg.header.frame_id
            )
            else
            self.optical_frame
        )

    def lookup_exact_transform(
        self,
        depth_msg
    ):

        source_frame = self.depth_source_frame(
            depth_msg
        )

        stamp = Time.from_msg(
            depth_msg.header.stamp
        )

        try:
            return self.tf_buffer.lookup_transform(
                self.world_frame,
                source_frame,
                stamp,
                timeout=Duration(
                    seconds=0.0
                )
            )

        except Exception:
            # TF not ready yet.
            # Do not block and do not spam warnings.
            return None

    def enqueue_tf_pair(
        self,
        detections_msg,
        depth_msg
    ):

        stamp_ns = self.timestamp_ns(
            depth_msg.header.stamp
        )

        # Avoid inserting the same matched frame repeatedly.
        for _, _, pending_depth in self.pending_tf_pairs:

            if (
                self.timestamp_ns(
                    pending_depth.header.stamp
                )
                ==
                stamp_ns
            ):
                return

        self.pending_tf_pairs.append(
            (
                monotonic_ns(),
                detections_msg,
                depth_msg
            )
        )

    def retry_tf_pending(
        self
    ):

        if (
            self.fx is None
            or
            not self.pending_tf_pairs
        ):
            return

        now_ns = monotonic_ns()

        ttl_ns = int(
            self.tf_pending_ttl_sec
            * 1_000_000_000
        )

        while self.pending_tf_pairs:

            receipt_ns, _, _ = (
                self.pending_tf_pairs[0]
            )

            if now_ns - receipt_ns <= ttl_ns:
                break

            self.pending_tf_pairs.popleft()

        if not self.pending_tf_pairs:
            return

        selected_index = None

        # Newest-first:
        # choose the freshest frame whose exact-stamp
        # transform has become available.
        for index in range(
            len(self.pending_tf_pairs) - 1,
            -1,
            -1
        ):

            _, _, depth_msg = (
                self.pending_tf_pairs[index]
            )

            transform = (
                self.lookup_exact_transform(
                    depth_msg
                )
            )

            if transform is not None:
                selected_index = index
                break

        if selected_index is None:
            return

        _, detections_msg, depth_msg = (
            self.pending_tf_pairs[
                selected_index
            ]
        )

        # Everything older than the selected ready frame
        # is stale. Drop it instead of replaying backlog.
        for _ in range(
            selected_index + 1
        ):
            self.pending_tf_pairs.popleft()

        # process_pair() will perform one final exact lookup.
        self.process_pair(
            detections_msg,
            depth_msg
        )

    def process_pair(
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

        transform = self.lookup_exact_transform(
            depth_msg
        )

        if transform is None:

            self.enqueue_tf_pair(
                detections_msg,
                depth_msg
            )

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
                self.depth_source_frame(
                    depth_msg
                )
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

            world_point = (
                do_transform_point(
                    optical_point,
                    transform
                )
            )

            source_result = (
                detection2d.results[0]
            )

            detection3d = (
                Detection3D()
            )

            detection3d.header = (
                output.header
            )

            # Preserve the tracker-owned persistent ID.
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
