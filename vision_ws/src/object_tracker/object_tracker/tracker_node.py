#!/usr/bin/env python3

from dataclasses import dataclass

import rclpy
from rclpy.node import Node

from vision_msgs.msg import Detection2DArray


@dataclass
class Track:
    track_id: int
    class_id: str
    bbox: tuple
    missed: int = 0
    age: int = 1


class ObjectTrackerNode(Node):

    def __init__(self):
        super().__init__('object_tracker')

        self.declare_parameter(
            'input_topic',
            '/object_detector/detections'
        )

        self.declare_parameter(
            'output_topic',
            '/object_tracker/detections'
        )

        self.declare_parameter(
            'iou_threshold',
            0.20
        )

        self.declare_parameter(
            'max_missed',
            5
        )

        self.input_topic = self.get_parameter(
            'input_topic'
        ).value

        self.output_topic = self.get_parameter(
            'output_topic'
        ).value

        self.iou_threshold = float(
            self.get_parameter(
                'iou_threshold'
            ).value
        )

        self.max_missed = int(
            self.get_parameter(
                'max_missed'
            ).value
        )

        self.next_track_id = 1
        self.tracks = {}

        self.sub = self.create_subscription(
            Detection2DArray,
            self.input_topic,
            self.callback,
            10
        )

        self.pub = self.create_publisher(
            Detection2DArray,
            self.output_topic,
            10
        )

        self.frame_count = 0

        self.get_logger().info(
            'Object tracker started'
        )

        self.get_logger().info(
            f'Input: {self.input_topic}'
        )

        self.get_logger().info(
            f'Output: {self.output_topic}'
        )

    @staticmethod
    def bbox_xyxy(detection):

        cx = float(
            detection.bbox.center.position.x
        )

        cy = float(
            detection.bbox.center.position.y
        )

        w = float(
            detection.bbox.size_x
        )

        h = float(
            detection.bbox.size_y
        )

        return (
            cx - w / 2.0,
            cy - h / 2.0,
            cx + w / 2.0,
            cy + h / 2.0,
        )

    @staticmethod
    def iou(a, b):

        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)

        intersection = iw * ih

        area_a = max(
            0.0,
            ax2 - ax1
        ) * max(
            0.0,
            ay2 - ay1
        )

        area_b = max(
            0.0,
            bx2 - bx1
        ) * max(
            0.0,
            by2 - by1
        )

        union = (
            area_a
            + area_b
            - intersection
        )

        if union <= 0.0:
            return 0.0

        return intersection / union

    @staticmethod
    def get_class_id(detection):

        if not detection.results:
            return ''

        return (
            detection.results[0]
            .hypothesis
            .class_id
        )

    def create_track(
        self,
        class_id,
        bbox
    ):

        track_id = self.next_track_id
        self.next_track_id += 1

        self.tracks[track_id] = Track(
            track_id=track_id,
            class_id=class_id,
            bbox=bbox,
        )

        return track_id

    def callback(self, msg):

        self.frame_count += 1

        detections = list(
            msg.detections
        )

        detection_info = []

        for index, detection in enumerate(
            detections
        ):

            detection_info.append(
                {
                    'index': index,
                    'class_id': self.get_class_id(
                        detection
                    ),
                    'bbox': self.bbox_xyxy(
                        detection
                    ),
                }
            )

        # ----------------------------------------------
        # Build all valid class-aware association pairs.
        # ----------------------------------------------

        candidates = []

        for track_id, track in (
            self.tracks.items()
        ):

            for info in detection_info:

                if (
                    track.class_id
                    !=
                    info['class_id']
                ):
                    continue

                score = self.iou(
                    track.bbox,
                    info['bbox']
                )

                if score >= self.iou_threshold:

                    candidates.append(
                        (
                            score,
                            track_id,
                            info['index'],
                        )
                    )

        # Highest IoU wins first.
        candidates.sort(
            key=lambda item: item[0],
            reverse=True
        )

        matched_tracks = set()
        matched_detections = set()
        assignments = {}

        for (
            score,
            track_id,
            detection_index
        ) in candidates:

            if track_id in matched_tracks:
                continue

            if (
                detection_index
                in
                matched_detections
            ):
                continue

            assignments[
                detection_index
            ] = track_id

            matched_tracks.add(
                track_id
            )

            matched_detections.add(
                detection_index
            )

        # ----------------------------------------------
        # Update matched tracks.
        # ----------------------------------------------

        for info in detection_info:

            detection_index = (
                info['index']
            )

            if (
                detection_index
                not in assignments
            ):
                continue

            track_id = assignments[
                detection_index
            ]

            track = self.tracks[
                track_id
            ]

            track.bbox = info['bbox']
            track.missed = 0
            track.age += 1

        # ----------------------------------------------
        # Age unmatched tracks.
        # ----------------------------------------------

        for track_id, track in list(
            self.tracks.items()
        ):

            if track_id not in matched_tracks:
                track.missed += 1

            if track.missed > self.max_missed:

                del self.tracks[
                    track_id
                ]

        # ----------------------------------------------
        # Create new tracks for unmatched detections.
        # ----------------------------------------------

        for info in detection_info:

            detection_index = (
                info['index']
            )

            if (
                detection_index
                in assignments
            ):
                continue

            track_id = self.create_track(
                info['class_id'],
                info['bbox']
            )

            assignments[
                detection_index
            ] = track_id

        # ----------------------------------------------
        # Publish current detections with persistent IDs.
        # ----------------------------------------------

        output = Detection2DArray()
        output.header = msg.header

        for index, detection in enumerate(
            detections
        ):

            if index not in assignments:
                continue

            track_id = assignments[index]

            detection.id = (
                f'track_{track_id:03d}'
            )

            output.detections.append(
                detection
            )

        self.pub.publish(
            output
        )

        if self.frame_count % 20 == 0:

            summary = []

            for detection in (
                output.detections
            ):

                class_id = (
                    self.get_class_id(
                        detection
                    )
                )

                summary.append(
                    f'{detection.id}:{class_id}'
                )

            self.get_logger().info(
                f'frame={self.frame_count} | '
                f'active_tracks={len(self.tracks)} | '
                +
                ', '.join(summary)
            )


def main(args=None):

    rclpy.init(args=args)

    node = ObjectTrackerNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
