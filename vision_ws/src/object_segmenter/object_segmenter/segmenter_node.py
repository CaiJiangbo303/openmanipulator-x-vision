#!/usr/bin/env python3

from pathlib import Path
import time

import numpy as np
import torch
import rclpy
import message_filters

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class ObjectSegmenterNode(Node):

    def __init__(self):
        super().__init__('object_segmenter')

        project_root = (
            Path.home()
            / 'openmanipulator-x-vision'
        )

        self.declare_parameter(
            'image_topic',
            '/camera/image'
        )

        self.declare_parameter(
            'detections_topic',
            '/object_tracker/detections'
        )

        self.declare_parameter(
            'instance_mask_topic',
            '/object_segmenter/instance_mask'
        )

        self.declare_parameter(
            'debug_topic',
            '/object_segmenter/debug_image'
        )

        self.declare_parameter(
            'checkpoint',
            str(
                project_root
                / 'third_party/sam2/checkpoints/'
                'sam2.1_hiera_small.pt'
            )
        )

        self.declare_parameter(
            'model_cfg',
            'configs/sam2.1/sam2.1_hiera_s.yaml'
        )

        # ImagePredictor v1 does not need to run at camera rate.
        # Temporal SAM2 will later replace this behavior.
        self.declare_parameter(
            'process_every_n_frames',
            4
        )

        # Detailed profiling is sampled sparsely because
        # CUDA synchronization itself perturbs timing.
        self.declare_parameter(
            'profile_every_n_processed',
            0
        )

        self.image_topic = self.get_parameter(
            'image_topic'
        ).value

        self.detections_topic = self.get_parameter(
            'detections_topic'
        ).value

        self.instance_mask_topic = self.get_parameter(
            'instance_mask_topic'
        ).value

        self.debug_topic = self.get_parameter(
            'debug_topic'
        ).value

        checkpoint = self.get_parameter(
            'checkpoint'
        ).value

        model_cfg = self.get_parameter(
            'model_cfg'
        ).value

        self.process_every_n_frames = int(
            self.get_parameter(
                'process_every_n_frames'
            ).value
        )

        self.profile_every_n_processed = int(
            self.get_parameter(
                'profile_every_n_processed'
            ).value
        )

        if not torch.cuda.is_available():
            raise RuntimeError(
                'SAM2 requires CUDA in this configuration.'
            )

        self.get_logger().info(
            'Loading SAM2.1 Hiera Small...'
        )

        self.model = build_sam2(
            model_cfg,
            checkpoint,
            device='cuda'
        )

        self.predictor = SAM2ImagePredictor(
            self.model
        )

        self.get_logger().info(
            'SAM2 model loaded on CUDA'
        )

        self.image_sub = message_filters.Subscriber(
            self,
            Image,
            self.image_topic,
            qos_profile=qos_profile_sensor_data
        )

        self.detection_sub = message_filters.Subscriber(
            self,
            Detection2DArray,
            self.detections_topic,
            qos_profile=10
        )

        self.sync = (
            message_filters.ApproximateTimeSynchronizer(
                [
                    self.image_sub,
                    self.detection_sub
                ],
                queue_size=10,
                slop=0.005,
                allow_headerless=False
            )
        )

        self.sync.registerCallback(
            self.synced_callback
        )

        # uint16 label image:
        #
        # 0 = background
        # 1 = track_001
        # 2 = track_002
        # ...
        #
        # This gives us all instance masks in one ROS image.
        self.mask_pub = self.create_publisher(
            Image,
            self.instance_mask_topic,
            qos_profile_sensor_data
        )

        self.debug_pub = self.create_publisher(
            Image,
            self.debug_topic,
            qos_profile_sensor_data
        )

        self.frame_count = 0
        self.processed_count = 0

        self.get_logger().info(
            'Object segmenter started'
        )

        self.get_logger().info(
            f'Image: {self.image_topic}'
        )

        self.get_logger().info(
            f'Detections: {self.detections_topic}'
        )

        self.get_logger().info(
            f'Instance mask: {self.instance_mask_topic}'
        )

    @staticmethod
    def image_to_numpy(msg):

        if msg.encoding != 'rgb8':
            raise ValueError(
                f'Unsupported RGB encoding: {msg.encoding}'
            )

        image = np.ndarray(
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

        return np.ascontiguousarray(
            image
        )

    @staticmethod
    def bbox_xyxy(detection):

        cx = float(
            detection.bbox.center.position.x
        )

        cy = float(
            detection.bbox.center.position.y
        )

        width = float(
            detection.bbox.size_x
        )

        height = float(
            detection.bbox.size_y
        )

        return np.array(
            [
                cx - width / 2.0,
                cy - height / 2.0,
                cx + width / 2.0,
                cy + height / 2.0,
            ],
            dtype=np.float32
        )

    @staticmethod
    def class_name(detection):

        if not detection.results:
            return 'UNKNOWN'

        return (
            detection.results[0]
            .hypothesis
            .class_id
        )

    @staticmethod
    def track_label(track_id):

        # track_001 -> uint16 1
        try:
            value = int(
                track_id.rsplit(
                    '_',
                    1
                )[-1]
            )

            if 1 <= value <= 65535:
                return value

        except Exception:
            pass

        return 0

    def publish_mask(
        self,
        source_msg,
        label_map
    ):

        out = Image()

        out.header = source_msg.header
        out.height = label_map.shape[0]
        out.width = label_map.shape[1]

        out.encoding = 'mono16'
        out.is_bigendian = False

        out.step = (
            label_map.shape[1] * 2
        )

        if (
            label_map.dtype != np.uint16
            or
            not label_map.flags.c_contiguous
        ):
            label_map = np.ascontiguousarray(
                label_map,
                dtype=np.uint16
            )

        out.data = label_map.tobytes()

        self.mask_pub.publish(
            out
        )

    def publish_debug(
        self,
        source_msg,
        rgb,
        label_map
    ):

        debug = rgb.copy()

        unique_labels = np.unique(
            label_map
        )

        for label in unique_labels:

            if label == 0:
                continue

            mask = (
                label_map == label
            )

            # Deterministic per-track tint.
            tint = np.array(
                [
                    (53 * label) % 255,
                    (97 * label) % 255,
                    (193 * label) % 255,
                ],
                dtype=np.float32
            )

            original = (
                debug[mask]
                .astype(np.float32)
            )

            debug[mask] = np.clip(
                original * 0.55
                +
                tint * 0.45,
                0,
                255
            ).astype(np.uint8)

        out = Image()

        out.header = source_msg.header
        out.height = debug.shape[0]
        out.width = debug.shape[1]
        out.encoding = 'rgb8'
        out.is_bigendian = False
        out.step = debug.shape[1] * 3

        out.data = (
            np.ascontiguousarray(
                debug
            ).tobytes()
        )

        self.debug_pub.publish(
            out
        )

    def synced_callback(
        self,
        image_msg,
        detections_msg
    ):

        self.frame_count += 1

        if (
            self.frame_count
            %
            self.process_every_n_frames
            != 0
        ):
            return

        if not detections_msg.detections:
            return

        self.processed_count += 1

        profile_this = (
            self.profile_every_n_processed > 0
            and
            self.processed_count
            %
            self.profile_every_n_processed
            == 0
        )

        if profile_this:
            callback_start = time.perf_counter()
            image_start = callback_start

        try:

            rgb = self.image_to_numpy(
                image_msg
            )

        except Exception as exc:

            self.get_logger().warning(
                str(exc)
            )

            return

        if profile_this:
            image_ms = (
                time.perf_counter()
                - image_start
            ) * 1000.0

        start = time.perf_counter()

        set_image_ms = 0.0
        predict_ms = 0.0
        merge_ms = 0.0

        # SAM2 encodes the image once.
        with torch.inference_mode(), torch.autocast(
            'cuda',
            dtype=torch.bfloat16
        ):

            if profile_this:
                torch.cuda.synchronize()
                stage_start = time.perf_counter()

            self.predictor.set_image(
                rgb
            )

            if profile_this:
                torch.cuda.synchronize()
                set_image_ms = (
                    time.perf_counter()
                    - stage_start
                ) * 1000.0

            label_map = np.zeros(
                (
                    image_msg.height,
                    image_msg.width
                ),
                dtype=np.uint16
            )

            confidence_map = np.full(
                (
                    image_msg.height,
                    image_msg.width
                ),
                -np.inf,
                dtype=np.float32
            )

            summaries = []

            for detection in (
                detections_msg.detections
            ):

                label = self.track_label(
                    detection.id
                )

                if label == 0:
                    continue

                box = self.bbox_xyxy(
                    detection
                )

                if profile_this:
                    torch.cuda.synchronize()
                    stage_start = time.perf_counter()

                masks, scores, _ = (
                    self.predictor.predict(
                        box=box,
                        multimask_output=False
                    )
                )

                if profile_this:
                    torch.cuda.synchronize()
                    predict_ms += (
                        time.perf_counter()
                        - stage_start
                    ) * 1000.0

                    merge_start = time.perf_counter()

                mask = (
                    masks[0]
                    .astype(bool)
                )

                score = float(
                    scores[0]
                )

                # Resolve rare mask overlap by keeping
                # the higher-SAM2-score instance.
                replace = (
                    mask
                    &
                    (
                        score
                        >
                        confidence_map
                    )
                )

                label_map[
                    replace
                ] = label

                confidence_map[
                    replace
                ] = score

                summaries.append(
                    (
                        detection.id,
                        self.class_name(
                            detection
                        ),
                        score,
                        int(
                            np.count_nonzero(
                                mask
                            )
                        )
                    )
                )

                if profile_this:
                    merge_ms += (
                        time.perf_counter()
                        - merge_start
                    ) * 1000.0

        torch.cuda.synchronize()

        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        if profile_this:
            publish_start = time.perf_counter()

        self.publish_mask(
            image_msg,
            label_map
        )

        if profile_this:
            publish_mask_ms = (
                time.perf_counter()
                - publish_start
            ) * 1000.0

        if self.debug_pub.get_subscription_count() > 0:
            self.publish_debug(
                image_msg,
                rgb,
                label_map
            )

        if profile_this:
            callback_ms = (
                time.perf_counter()
                - callback_start
            ) * 1000.0

            self.get_logger().info(
                '[PROFILE] '
                f'processed={self.processed_count} | '
                f'image={image_ms:.1f} ms | '
                f'set_image={set_image_ms:.1f} ms | '
                f'predict={predict_ms:.1f} ms | '
                f'merge={merge_ms:.1f} ms | '
                f'publish_mask={publish_mask_ms:.1f} ms | '
                f'sam2={elapsed_ms:.1f} ms | '
                f'callback={callback_ms:.1f} ms'
            )

        self.get_logger().info(
            f'frame={self.frame_count} | '
            f'instances={len(summaries)} | '
            f'total={elapsed_ms:.1f} ms'
        )

        for (
            track_id,
            class_name,
            score,
            mask_pixels
        ) in summaries:

            self.get_logger().info(
                f'  {track_id:<10} '
                f'{class_name:<10} '
                f'score={score:.3f} '
                f'mask_pixels={mask_pixels}'
            )


def main(args=None):

    rclpy.init(args=args)

    node = ObjectSegmenterNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
