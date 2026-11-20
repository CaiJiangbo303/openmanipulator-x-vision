import time

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from vision_msgs.msg import (
    Detection2DArray,
    Detection3DArray,
)


class VisionDebugViewer(Node):

    def __init__(self):
        super().__init__('vision_debug_viewer')

        self.bridge = CvBridge()

        self.latest_image = None
        self.latest_detections = None
        self.latest_mask = None

        # Display-only 3D caches.
        # These do not affect perception or control.
        self.latest_fast3d = {}
        self.latest_mask3d = {}

        # Keep exact Fast3D source-stamp samples so a later
        # Mask3D result can be compared with the same frame.
        self.fast3d_history = {}
        self.latest_delta3d = {}

        # Display-only filtering.
        # Raw detector/tracker topics are NOT changed.
        self.min_display_confidence = 0.50

        self.camera_fps = 0.0
        self.mask_fps = 0.0
        self.viewer_fps = 0.0

        self.camera_count = 0
        self.mask_count = 0

        now = time.perf_counter()
        self.camera_t0 = now
        self.mask_t0 = now
        self.viewer_last = now

        self.create_subscription(
            Image,
            '/camera/image',
            self.image_callback,
            qos_profile_sensor_data
        )

        self.create_subscription(
            Detection2DArray,
            '/object_tracker/detections',
            self.detection_callback,
            10
        )

        self.create_subscription(
            Image,
            '/object_segmenter/instance_mask',
            self.mask_callback,
            qos_profile_sensor_data
        )

        self.create_subscription(
            Detection3DArray,
            '/object_localizer/detections_3d_fast',
            self.fast3d_callback,
            10
        )

        self.create_subscription(
            Detection3DArray,
            '/object_localizer/detections_3d_mask',
            self.mask3d_callback,
            10
        )

        self.timer = self.create_timer(
            1.0 / 30.0,
            self.render
        )

        self.get_logger().info(
            'Vision Debug Viewer started'
        )

        self.get_logger().info(
            'Display confidence threshold = 0.50'
        )

    def image_callback(self, msg):
        try:
            self.latest_image = (
                self.bridge.imgmsg_to_cv2(
                    msg,
                    desired_encoding='rgb8'
                )
            )

            self.camera_count += 1

            now = time.perf_counter()
            dt = now - self.camera_t0

            if dt >= 1.0:
                self.camera_fps = (
                    self.camera_count / dt
                )

                self.camera_count = 0
                self.camera_t0 = now

        except Exception as exc:
            self.get_logger().warn(
                f'Image conversion failed: {exc}'
            )

    def detection_callback(self, msg):
        self.latest_detections = msg

    def mask_callback(self, msg):
        try:
            self.latest_mask = (
                self.bridge.imgmsg_to_cv2(
                    msg,
                    desired_encoding='passthrough'
                )
            )

            self.mask_count += 1

            now = time.perf_counter()
            dt = now - self.mask_t0

            if dt >= 1.0:
                self.mask_fps = (
                    self.mask_count / dt
                )

                self.mask_count = 0
                self.mask_t0 = now

        except Exception as exc:
            self.get_logger().warn(
                f'Mask conversion failed: {exc}'
            )

    @staticmethod
    def stamp_ns(msg):
        return (
            int(msg.header.stamp.sec)
            * 1_000_000_000
            +
            int(msg.header.stamp.nanosec)
        )

    @staticmethod
    def detection_xyz(det):
        if not det.id or not det.results:
            return None

        position = (
            det.results[0]
            .pose.pose.position
        )

        return (
            det.id,
            (
                float(position.x),
                float(position.y),
                float(position.z),
            )
        )

    def fast3d_callback(self, msg):
        stamp = self.stamp_ns(msg)
        now = time.perf_counter()

        for det in msg.detections:
            parsed = self.detection_xyz(det)

            if parsed is None:
                continue

            track_id, xyz = parsed

            self.latest_fast3d[track_id] = (
                xyz,
                now
            )

            self.fast3d_history[
                (stamp, track_id)
            ] = xyz

        # About 50 seconds at 5 Hz. This is only a
        # small display-side cache.
        while len(self.fast3d_history) > 256:
            first_key = next(
                iter(self.fast3d_history)
            )
            self.fast3d_history.pop(first_key)

    def mask3d_callback(self, msg):
        stamp = self.stamp_ns(msg)
        now = time.perf_counter()

        for det in msg.detections:
            parsed = self.detection_xyz(det)

            if parsed is None:
                continue

            track_id, xyz = parsed

            self.latest_mask3d[track_id] = (
                xyz,
                now
            )

            fast_xyz = self.fast3d_history.get(
                (stamp, track_id)
            )

            if fast_xyz is None:
                continue

            delta = float(
                np.linalg.norm(
                    np.asarray(
                        fast_xyz,
                        dtype=np.float64
                    )
                    -
                    np.asarray(
                        xyz,
                        dtype=np.float64
                    )
                )
            )

            self.latest_delta3d[track_id] = (
                delta,
                now
            )

    def draw_mask(self, frame):
        if self.latest_mask is None:
            return frame

        mask = self.latest_mask

        if mask.shape[:2] != frame.shape[:2]:
            return frame

        instance_ids = np.unique(mask)
        instance_ids = instance_ids[
            instance_ids != 0
        ]

        if len(instance_ids) == 0:
            return frame

        # BGR palette for visualization only.
        palette = [
            (40, 220, 80),
            (255, 150, 40),
            (220, 60, 220),
            (40, 220, 240),
            (240, 100, 100),
        ]

        result = frame.copy()

        for index, instance_id in enumerate(
            instance_ids
        ):
            active = (
                mask == instance_id
            )

            if not np.any(active):
                continue

            color = palette[
                index % len(palette)
            ]

            overlay = result.copy()
            overlay[active] = color

            result = cv2.addWeighted(
                result,
                0.62,
                overlay,
                0.38,
                0
            )

            binary = (
                active.astype(np.uint8) * 255
            )

            contours, _ = cv2.findContours(
                binary,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            cv2.drawContours(
                result,
                contours,
                -1,
                color,
                2,
                cv2.LINE_AA
            )

        return result

    def draw_detections(self, frame):
        if self.latest_detections is None:
            return

        for det in (
            self.latest_detections.detections
        ):
            if not det.results:
                continue

            result = det.results[0]

            class_id = (
                result.hypothesis.class_id
            )

            score = float(
                result.hypothesis.score
            )

            # Viewer-only filter.
            if (
                score
                <
                self.min_display_confidence
            ):
                continue

            cx = float(
                det.bbox.center.position.x
            )

            cy = float(
                det.bbox.center.position.y
            )

            w = float(det.bbox.size_x)
            h = float(det.bbox.size_y)

            x1 = int(cx - w / 2.0)
            y1 = int(cy - h / 2.0)
            x2 = int(cx + w / 2.0)
            y2 = int(cy + h / 2.0)

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(
                frame.shape[1] - 1,
                x2
            )

            y2 = min(
                frame.shape[0] - 1,
                y2
            )

            track_id = (
                det.id
                if det.id
                else 'untracked'
            )

            label = (
                f'{class_id} '
                f'{score:.2f} '
                f'{track_id}'
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2
            )

            # Black background improves readability.
            (tw, th), _ = (
                cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    2
                )
            )

            text_y = max(
                th + 8,
                y1
            )

            cv2.rectangle(
                frame,
                (
                    x1,
                    text_y - th - 8
                ),
                (
                    min(
                        frame.shape[1] - 1,
                        x1 + tw + 8
                    ),
                    text_y + 4
                ),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                frame,
                label,
                (
                    x1 + 4,
                    text_y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            # -------------------------------------------------
            # Display-only 3D information
            # -------------------------------------------------

            now = time.perf_counter()
            info_lines = []

            fast_item = self.latest_fast3d.get(
                track_id
            )

            if (
                fast_item is not None
                and
                now - fast_item[1] <= 0.75
            ):
                x, y, z = fast_item[0]

                info_lines.append(
                    (
                        'FAST',
                        (
                            f'Fast  '
                            f'X={x:+.3f} '
                            f'Y={y:+.3f} '
                            f'Z={z:+.3f} m'
                        ),
                        (80, 255, 80)
                    )
                )

            mask_item = self.latest_mask3d.get(
                track_id
            )

            if (
                mask_item is not None
                and
                now - mask_item[1] <= 2.0
            ):
                x, y, z = mask_item[0]

                info_lines.append(
                    (
                        'MASK',
                        (
                            f'Mask  '
                            f'X={x:+.3f} '
                            f'Y={y:+.3f} '
                            f'Z={z:+.3f} m'
                        ),
                        (80, 220, 255)
                    )
                )

            delta_item = self.latest_delta3d.get(
                track_id
            )

            if (
                delta_item is not None
                and
                now - delta_item[1] <= 2.0
            ):
                delta_mm = (
                    delta_item[0] * 1000.0
                )

                info_lines.append(
                    (
                        'DELTA',
                        f'Delta3D = {delta_mm:.1f} mm',
                        (255, 200, 80)
                    )
                )

            if info_lines:
                line_height = 18

                start_y = (
                    y2 + line_height
                )

                bottom = (
                    start_y
                    +
                    line_height
                    * (len(info_lines) - 1)
                )

                if bottom >= frame.shape[0] - 4:
                    start_y = max(
                        90,
                        y1
                        -
                        line_height
                        * len(info_lines)
                        -
                        4
                    )

                for index, (
                    _,
                    info,
                    color
                ) in enumerate(info_lines):

                    info_y = (
                        start_y
                        +
                        index * line_height
                    )

                    (iw, ih), _ = (
                        cv2.getTextSize(
                            info,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.43,
                            1
                        )
                    )

                    cv2.rectangle(
                        frame,
                        (
                            x1,
                            info_y - ih - 4
                        ),
                        (
                            min(
                                frame.shape[1] - 1,
                                x1 + iw + 7
                            ),
                            info_y + 3
                        ),
                        (0, 0, 0),
                        -1
                    )

                    cv2.putText(
                        frame,
                        info,
                        (
                            x1 + 3,
                            info_y
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.43,
                        color,
                        1,
                        cv2.LINE_AA
                    )

    def draw_status(self, frame):
        line1 = (
            'OpenManipulator-X '
            'RGB Perception Demo'
        )

        line2 = (
            f'Camera {self.camera_fps:.1f} FPS'
            f'  |  SAM2 {self.mask_fps:.1f} FPS'
            f'  |  Viewer {self.viewer_fps:.1f} FPS'
        )

        line3 = (
            'YOLO11s + Tracker + Fast3D + '
            'SAM2/Mask3D  |  Display conf >= 0.50'
        )

        # HUD background
        cv2.rectangle(
            frame,
            (0, 0),
            (frame.shape[1], 78),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            line1,
            (12, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.57,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            line2,
            (12, 47),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            line3,
            (12, 69),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )

    def render(self):
        if self.latest_image is None:
            return

        frame = cv2.cvtColor(
            self.latest_image,
            cv2.COLOR_RGB2BGR
        )

        # SAM2 first, bounding boxes afterwards.
        frame = self.draw_mask(frame)

        self.draw_detections(frame)

        now = time.perf_counter()
        dt = now - self.viewer_last

        if dt > 0:
            current = 1.0 / dt

            if self.viewer_fps <= 0:
                self.viewer_fps = current
            else:
                self.viewer_fps = (
                    0.90 * self.viewer_fps
                    +
                    0.10 * current
                )

        self.viewer_last = now

        self.draw_status(frame)

        cv2.imshow(
            'OpenManipulator-X Vision Demo',
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    node = VisionDebugViewer()

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
