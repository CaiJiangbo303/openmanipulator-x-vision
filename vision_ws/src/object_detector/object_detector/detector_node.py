#!/usr/bin/env python3

from time import perf_counter

import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)

from sensor_msgs.msg import Image

from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)

from .backends.yolo_backend import YOLOBackend
from .backends.yoloe_backend import YOLOEBackend
from .backends.color_debug_backend import ColorDebugBackend


class ObjectDetectorNode(Node):

    def __init__(self):
        super().__init__('object_detector')

        self.declare_parameter(
            'backend',
            'yolo'
        )

        self.declare_parameter(
            'image_topic',
            '/camera/image'
        )

        self.declare_parameter(
            'model',
            'yolo11s.pt'
        )

        self.declare_parameter(
            'confidence',
            0.25
        )

        self.declare_parameter(
            'iou',
            0.45
        )

        self.declare_parameter(
            'imgsz',
            640
        )

        self.declare_parameter(
            'device',
            '0'
        )

        self.declare_parameter(
            'quantize',
            '16'
        )

        # Optional post-YOLO geometric gate.
        # 0 keeps the original behaviour.
        self.declare_parameter(
            'min_bbox_area',
            0.0
        )

        self.declare_parameter(
            'prompt_mode',
            'prompt_free'
        )

        self.declare_parameter(
            'prompt_classes',
            ''
        )

        self.backend_name = (
            self.get_parameter(
                'backend'
            ).value
        )

        self.image_topic = (
            self.get_parameter(
                'image_topic'
            ).value
        )

        self.min_bbox_area = float(
            self.get_parameter(
                'min_bbox_area'
            ).value
        )

        # --------------------------------------------------
        # Backend selection
        # --------------------------------------------------

        if self.backend_name == 'yolo':

            self.backend = YOLOBackend(
                model_path=self.get_parameter(
                    'model'
                ).value,

                confidence=float(
                    self.get_parameter(
                        'confidence'
                    ).value
                ),

                iou=float(
                    self.get_parameter(
                        'iou'
                    ).value
                ),

                image_size=int(
                    self.get_parameter(
                        'imgsz'
                    ).value
                ),

                device=self.get_parameter(
                    'device'
                ).value,

                quantize=self.get_parameter(
                    'quantize'
                ).value,
            )

        elif self.backend_name == 'yoloe':

            self.backend = YOLOEBackend(
                model_path=self.get_parameter('model').value,
                confidence=float(
                    self.get_parameter('confidence').value
                ),
                iou=float(
                    self.get_parameter('iou').value
                ),
                image_size=int(
                    self.get_parameter('imgsz').value
                ),
                device=self.get_parameter('device').value,
                quantize=self.get_parameter('quantize').value,
                prompt_mode=self.get_parameter('prompt_mode').value,
                prompt_classes=self.get_parameter('prompt_classes').value,
            )

        elif self.backend_name == 'color_debug':

            self.backend = ColorDebugBackend()

        else:

            raise RuntimeError(
                'Unknown detector backend: '
                f'{self.backend_name}'
            )

        # --------------------------------------------------
        # Latest-frame QoS
        # --------------------------------------------------

        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.image_sub = (
            self.create_subscription(
                Image,
                self.image_topic,
                self.image_callback,
                image_qos
            )
        )

        self.detection_pub = (
            self.create_publisher(
                Detection2DArray,
                '/object_detector/detections',
                10
            )
        )

        self.debug_pub = (
            self.create_publisher(
                Image,
                '/object_detector/debug_image',
                image_qos
            )
        )

        self.frame_count = 0

        self.get_logger().info(
            'Object detector started | '
            f'backend={self.backend_name} | '
            f'image={self.image_topic}'
        )

        self.get_logger().info(
            'Publishing detections: '
            '/object_detector/detections'
        )

    def image_callback(self, msg):

        if msg.encoding != 'rgb8':

            self.get_logger().warning(
                'Unsupported encoding: '
                f'{msg.encoding}'
            )

            return

        self.frame_count += 1

        rgb = np.ndarray(
            shape=(
                msg.height,
                msg.width,
                3
            ),
            dtype=np.uint8,
            buffer=msg.data,
            strides=(
                msg.step,
                3,
                1
            )
        )

        bgr = np.ascontiguousarray(
            rgb[:, :, ::-1]
        )

        start = perf_counter()

        detections, debug_bgr = (
            self.backend.predict(
                bgr
            )
        )

        # Keep the backend generic.  Geometric filtering is
        # applied here before detections reach the tracker
        # and SAM2.
        filtered_by_area = 0

        if self.min_bbox_area > 0.0:

            kept = []

            for det in detections:

                x1, y1, x2, y2 = (
                    det.bbox_xyxy
                )

                width = max(
                    0.0,
                    x2 - x1
                )

                height = max(
                    0.0,
                    y2 - y1
                )

                area = width * height

                if area >= self.min_bbox_area:

                    kept.append(det)

                else:

                    filtered_by_area += 1

            detections = kept

        total_ms = (
            perf_counter() - start
        ) * 1000.0

        self.publish_detections(
            msg,
            detections
        )

        if debug_bgr is not None:

            self.publish_debug_image(
                msg,
                debug_bgr
            )

        if self.frame_count % 20 == 0:

            self.get_logger().info(
                f'frame={self.frame_count} | '
                f'backend={self.backend_name} | '
                f'detections={len(detections)} | '
                f'area_filtered={filtered_by_area} | '
                f'total={total_ms:.1f} ms'
            )

            for det in detections[:10]:

                x1, y1, x2, y2 = (
                    det.bbox_xyxy
                )

                self.get_logger().info(
                    f'  {det.class_name:<10} '
                    f'conf={det.confidence:.3f} '
                    f'bbox=('
                    f'{x1:.0f},'
                    f'{y1:.0f},'
                    f'{x2:.0f},'
                    f'{y2:.0f})'
                )

    def publish_detections(
        self,
        source_msg,
        detections
    ):

        array_msg = (
            Detection2DArray()
        )

        # Preserve exact RGB timestamp.
        array_msg.header = (
            source_msg.header
        )

        for det in detections:

            detection_msg = (
                Detection2D()
            )

            detection_msg.header = (
                source_msg.header
            )

            x1, y1, x2, y2 = (
                det.bbox_xyxy
            )

            width = x2 - x1
            height = y2 - y1

            detection_msg.bbox.center.position.x = (
                float(
                    (x1 + x2) / 2.0
                )
            )

            detection_msg.bbox.center.position.y = (
                float(
                    (y1 + y2) / 2.0
                )
            )

            detection_msg.bbox.center.theta = 0.0

            detection_msg.bbox.size_x = (
                float(width)
            )

            detection_msg.bbox.size_y = (
                float(height)
            )

            result = (
                ObjectHypothesisWithPose()
            )

            result.hypothesis.class_id = (
                det.class_name
            )

            result.hypothesis.score = (
                float(
                    det.confidence
                )
            )

            detection_msg.results.append(
                result
            )

            # Detector does not own persistent IDs.
            detection_msg.id = ''

            array_msg.detections.append(
                detection_msg
            )

        self.detection_pub.publish(
            array_msg
        )

    def publish_debug_image(
        self,
        source_msg,
        debug_bgr
    ):

        debug_rgb = (
            np.ascontiguousarray(
                debug_bgr[:, :, ::-1]
            )
        )

        out = Image()

        out.header = source_msg.header
        out.height = debug_rgb.shape[0]
        out.width = debug_rgb.shape[1]
        out.encoding = 'rgb8'
        out.is_bigendian = False
        out.step = (
            debug_rgb.shape[1] * 3
        )
        out.data = debug_rgb.tobytes()

        self.debug_pub.publish(
            out
        )


def main(args=None):

    rclpy.init(args=args)

    node = ObjectDetectorNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
