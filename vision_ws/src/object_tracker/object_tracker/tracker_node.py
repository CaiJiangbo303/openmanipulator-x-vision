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


@dataclass
class TentativeTrack:
    tentative_id: int
    class_id: str
    bbox: tuple
    hits: int = 1
    missed: int = 0


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

        # Fallback association used only when IoU matching fails.
        # Distance is normalized by the average bbox size.
        self.declare_parameter(
            'center_distance_threshold',
            0.75
        )

        # Reject fallback matches with a large bbox-size change.
        self.declare_parameter(
            'max_area_ratio',
            2.0
        )

        # Fallback is only allowed for recently missed tracks.
        self.declare_parameter(
            'fallback_max_missed',
            2
        )

        # New detections do not immediately become
        # persistent tracks. They must survive several
        # consecutive frames first.
        self.declare_parameter(
            'new_track_confirm_hits',
            3
        )

        self.declare_parameter(
            'tentative_max_missed',
            1
        )

        # Suppress tentative tracks that are very likely
        # partial/duplicate views of an already-confirmed
        # same-class track visible in the current frame.
        self.declare_parameter(
            'duplicate_containment_threshold',
            0.70
        )

        self.declare_parameter(
            'duplicate_min_area_ratio',
            1.40
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

        self.center_distance_threshold = float(
            self.get_parameter(
                'center_distance_threshold'
            ).value
        )

        self.max_area_ratio = float(
            self.get_parameter(
                'max_area_ratio'
            ).value
        )

        self.fallback_max_missed = int(
            self.get_parameter(
                'fallback_max_missed'
            ).value
        )

        self.new_track_confirm_hits = int(
            self.get_parameter(
                'new_track_confirm_hits'
            ).value
        )

        self.tentative_max_missed = int(
            self.get_parameter(
                'tentative_max_missed'
            ).value
        )

        self.duplicate_containment_threshold = float(
            self.get_parameter(
                'duplicate_containment_threshold'
            ).value
        )

        self.duplicate_min_area_ratio = float(
            self.get_parameter(
                'duplicate_min_area_ratio'
            ).value
        )

        self.declare_parameter(
            'class_alias_groups',
            ''
        )

        self.class_alias_map = self.parse_class_alias_groups(
            self.get_parameter('class_alias_groups').value
        )

        self.next_track_id = 1
        self.tracks = {}

        self.next_tentative_id = 1
        self.tentative_tracks = {}

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
    def parse_class_alias_groups(value):
        aliases = {}

        for group in value.split(';'):
            names = [
                name.strip().lower()
                for name in group.split('|')
                if name.strip()
            ]

            if not names:
                continue

            canonical = names[0]

            for name in names:
                if name in aliases and aliases[name] != canonical:
                    raise ValueError(
                        f'Conflicting class alias: {name}'
                    )

                aliases[name] = canonical

        return aliases

    def class_key(self, class_id):
        name = str(class_id)
        return self.class_alias_map.get(
            name.strip().lower(),
            name
        )

    def same_class(self, a, b):
        return self.class_key(a) == self.class_key(b)

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
    def bbox_geometry(bbox):

        x1, y1, x2, y2 = bbox

        width = max(
            0.0,
            x2 - x1
        )

        height = max(
            0.0,
            y2 - y1
        )

        center_x = (
            x1 + x2
        ) / 2.0

        center_y = (
            y1 + y2
        ) / 2.0

        area = width * height

        return (
            center_x,
            center_y,
            width,
            height,
            area,
        )

    @classmethod
    def normalized_center_distance(
        cls,
        a,
        b
    ):

        (
            acx,
            acy,
            aw,
            ah,
            _
        ) = cls.bbox_geometry(a)

        (
            bcx,
            bcy,
            bw,
            bh,
            _
        ) = cls.bbox_geometry(b)

        scale_x = max(
            1.0,
            (aw + bw) / 2.0
        )

        scale_y = max(
            1.0,
            (ah + bh) / 2.0
        )

        dx = (
            bcx - acx
        ) / scale_x

        dy = (
            bcy - acy
        ) / scale_y

        return (
            dx * dx
            +
            dy * dy
        ) ** 0.5

    @classmethod
    def area_ratio(
        cls,
        a,
        b
    ):

        area_a = cls.bbox_geometry(
            a
        )[4]

        area_b = cls.bbox_geometry(
            b
        )[4]

        smaller = min(
            area_a,
            area_b
        )

        larger = max(
            area_a,
            area_b
        )

        if smaller <= 0.0:
            return float('inf')

        return larger / smaller

    @staticmethod
    def containment_ratio(a, b):

        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(
            0.0,
            ix2 - ix1
        )

        ih = max(
            0.0,
            iy2 - iy1
        )

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

        smaller = min(
            area_a,
            area_b
        )

        if smaller <= 0.0:
            return 0.0

        return intersection / smaller

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

    def create_tentative_track(
        self,
        class_id,
        bbox
    ):

        tentative_id = (
            self.next_tentative_id
        )

        self.next_tentative_id += 1

        self.tentative_tracks[
            tentative_id
        ] = TentativeTrack(
            tentative_id=tentative_id,
            class_id=class_id,
            bbox=bbox,
        )

        return tentative_id

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
                    not self.same_class(track.class_id, info['class_id'])
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
        # Fallback association:
        #
        # If IoU matching failed, try reconnecting a
        # recently seen track using normalized center
        # distance and bbox-size consistency.
        #
        # This is intentionally second-stage only:
        # IoU remains the preferred association method.
        # ----------------------------------------------

        fallback_candidates = []

        for track_id, track in (
            self.tracks.items()
        ):

            if track_id in matched_tracks:
                continue

            if (
                track.missed
                >
                self.fallback_max_missed
            ):
                continue

            for info in detection_info:

                detection_index = (
                    info['index']
                )

                if (
                    detection_index
                    in
                    matched_detections
                ):
                    continue

                if (
                    not self.same_class(track.class_id, info['class_id'])
                ):
                    continue

                distance = (
                    self.normalized_center_distance(
                        track.bbox,
                        info['bbox']
                    )
                )

                ratio = self.area_ratio(
                    track.bbox,
                    info['bbox']
                )

                if (
                    distance
                    <=
                    self.center_distance_threshold
                    and
                    ratio
                    <=
                    self.max_area_ratio
                ):

                    fallback_candidates.append(
                        (
                            distance,
                            ratio,
                            track_id,
                            detection_index,
                        )
                    )

        # Closest geometrically compatible detection wins.
        fallback_candidates.sort(
            key=lambda item: (
                item[0],
                item[1]
            )
        )

        for (
            distance,
            ratio,
            track_id,
            detection_index
        ) in fallback_candidates:

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
        # Associate unmatched detections with tentative
        # tracks.
        #
        # A one-frame false detection must NOT immediately
        # receive a persistent track_xxx ID.
        # ----------------------------------------------

        tentative_candidates = []

        for (
            tentative_id,
            tentative
        ) in self.tentative_tracks.items():

            for info in detection_info:

                detection_index = (
                    info['index']
                )

                if (
                    detection_index
                    in assignments
                ):
                    continue

                if (
                    not self.same_class(tentative.class_id, info['class_id'])
                ):
                    continue

                overlap = self.iou(
                    tentative.bbox,
                    info['bbox']
                )

                distance = (
                    self.normalized_center_distance(
                        tentative.bbox,
                        info['bbox']
                    )
                )

                ratio = self.area_ratio(
                    tentative.bbox,
                    info['bbox']
                )

                iou_valid = (
                    overlap
                    >=
                    self.iou_threshold
                )

                geometry_valid = (
                    distance
                    <=
                    self.center_distance_threshold
                    and
                    ratio
                    <=
                    self.max_area_ratio
                )

                if not (
                    iou_valid
                    or
                    geometry_valid
                ):
                    continue

                # Prefer IoU-valid associations.
                priority = (
                    1
                    if iou_valid
                    else 0
                )

                tentative_candidates.append(
                    (
                        priority,
                        overlap,
                        -distance,
                        tentative_id,
                        detection_index,
                    )
                )

        tentative_candidates.sort(
            reverse=True
        )

        matched_tentatives = set()
        tentative_assignments = {}

        for (
            priority,
            overlap,
            negative_distance,
            tentative_id,
            detection_index
        ) in tentative_candidates:

            if (
                tentative_id
                in matched_tentatives
            ):
                continue

            if (
                detection_index
                in assignments
                or
                detection_index
                in tentative_assignments
            ):
                continue

            tentative_assignments[
                detection_index
            ] = tentative_id

            matched_tentatives.add(
                tentative_id
            )

        # ----------------------------------------------
        # Update matched tentative tracks.
        # ----------------------------------------------

        info_by_index = {
            info['index']: info
            for info in detection_info
        }

        for (
            detection_index,
            tentative_id
        ) in tentative_assignments.items():

            tentative = (
                self.tentative_tracks[
                    tentative_id
                ]
            )

            info = info_by_index[
                detection_index
            ]

            tentative.bbox = (
                info['bbox']
            )

            tentative.hits += 1
            tentative.missed = 0

        # ----------------------------------------------
        # Age unmatched tentative tracks.
        # ----------------------------------------------

        for (
            tentative_id,
            tentative
        ) in list(
            self.tentative_tracks.items()
        ):

            if (
                tentative_id
                not in matched_tentatives
            ):
                tentative.missed += 1

            if (
                tentative.missed
                >
                self.tentative_max_missed
            ):

                del self.tentative_tracks[
                    tentative_id
                ]

        # ----------------------------------------------
        # Create tentative tracks for completely new
        # detections.
        # ----------------------------------------------

        for info in detection_info:

            detection_index = (
                info['index']
            )

            if (
                detection_index
                in assignments
                or
                detection_index
                in tentative_assignments
            ):
                continue

            self.create_tentative_track(
                info['class_id'],
                info['bbox']
            )

        # ----------------------------------------------
        # Promote confirmed tentative tracks.
        #
        # Before promotion, reject a tentative detection
        # if it is strongly contained inside (or contains)
        # an already matched same-class confirmed track.
        #
        # This targets persistent partial duplicate boxes
        # without imposing a one-object-per-class rule.
        # ----------------------------------------------

        duplicate_promotions_suppressed = 0

        for (
            detection_index,
            tentative_id
        ) in list(
            tentative_assignments.items()
        ):

            tentative = (
                self.tentative_tracks.get(
                    tentative_id
                )
            )

            if tentative is None:
                continue

            if (
                tentative.hits
                <
                self.new_track_confirm_hits
            ):
                continue

            duplicate = False

            for (
                confirmed_id,
                confirmed
            ) in self.tracks.items():

                # Only compare with confirmed tracks that
                # were actually observed in this frame.
                if (
                    confirmed_id
                    not in matched_tracks
                ):
                    continue

                if (
                    not self.same_class(confirmed.class_id, tentative.class_id)
                ):
                    continue

                containment = (
                    self.containment_ratio(
                        confirmed.bbox,
                        tentative.bbox
                    )
                )

                ratio = self.area_ratio(
                    confirmed.bbox,
                    tentative.bbox
                )

                if (
                    containment
                    >=
                    self.duplicate_containment_threshold
                    and
                    ratio
                    >=
                    self.duplicate_min_area_ratio
                ):

                    duplicate = True
                    break

            if duplicate:

                duplicate_promotions_suppressed += 1

                del self.tentative_tracks[
                    tentative_id
                ]

                continue

            track_id = self.create_track(
                tentative.class_id,
                tentative.bbox
            )

            assignments[
                detection_index
            ] = track_id

            del self.tentative_tracks[
                tentative_id
            ]

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
                f'confirmed_alive={len(self.tracks)} | '
                f'tentative={len(self.tentative_tracks)} | '
                f'dup_suppressed={duplicate_promotions_suppressed} | '
                f'output_tracks={len(output.detections)} | '
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
