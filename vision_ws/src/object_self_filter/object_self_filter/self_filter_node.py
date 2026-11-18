from collections import deque
#!/usr/bin/env python3

import math

import rclpy

from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration

from vision_msgs.msg import Detection3DArray

from tf2_ros import (
    Buffer,
    TransformListener,
)


def point_to_segment_distance(
    point,
    start,
    end,
):
    px, py, pz = point
    ax, ay, az = start
    bx, by, bz = end

    abx = bx - ax
    aby = by - ay
    abz = bz - az

    apx = px - ax
    apy = py - ay
    apz = pz - az

    length_sq = (
        abx * abx
        +
        aby * aby
        +
        abz * abz
    )

    if length_sq < 1e-12:
        dx = px - ax
        dy = py - ay
        dz = pz - az

        return math.sqrt(
            dx * dx
            +
            dy * dy
            +
            dz * dz
        )

    t = (
        apx * abx
        +
        apy * aby
        +
        apz * abz
    ) / length_sq

    t = max(
        0.0,
        min(
            1.0,
            t
        )
    )

    cx = ax + t * abx
    cy = ay + t * aby
    cz = az + t * abz

    dx = px - cx
    dy = py - cy
    dz = pz - cz

    return math.sqrt(
        dx * dx
        +
        dy * dy
        +
        dz * dz
    )


class ObjectSelfFilter(Node):

    def __init__(self):

        super().__init__(
            'object_self_filter'
        )

        self.declare_parameter(
            'input_topic',
            '/object_localizer/detections_3d_mask'
        )

        self.declare_parameter(
            'output_topic',
            '/object_self_filter/detections_3d'
        )

        self.declare_parameter(
            'world_frame',
            'world'
        )

        self.declare_parameter(
            'arm_radius',
            0.040
        )

        self.declare_parameter(
            'wrist_radius',
            0.050
        )

        self.declare_parameter(
            'gripper_radius',
            0.045
        )

        self.declare_parameter(
            'tf_wait_timeout',
            0.50
        )

        self.input_topic = (
            self.get_parameter(
                'input_topic'
            ).value
        )

        self.output_topic = (
            self.get_parameter(
                'output_topic'
            ).value
        )

        self.world_frame = (
            self.get_parameter(
                'world_frame'
            ).value
        )

        self.arm_radius = float(
            self.get_parameter(
                'arm_radius'
            ).value
        )

        self.wrist_radius = float(
            self.get_parameter(
                'wrist_radius'
            ).value
        )

        self.gripper_radius = float(
            self.get_parameter(
                'gripper_radius'
            ).value
        )

        self.tf_wait_timeout = float(
            self.get_parameter(
                'tf_wait_timeout'
            ).value
        )

        # Each tuple:
        #
        # start frame
        # end frame
        # radius
        # label
        self.segment_definitions = [
            (
                'arm_link1',
                'arm_link2',
                self.arm_radius,
                'link1-link2',
            ),
            (
                'arm_link2',
                'arm_link3',
                self.arm_radius,
                'link2-link3',
            ),
            (
                'arm_link3',
                'arm_link4',
                self.arm_radius,
                'link3-link4',
            ),
            (
                'arm_link4',
                'arm_link5',
                self.arm_radius,
                'link4-link5',
            ),
            (
                'arm_link5',
                'arm_end_effector_link',
                self.wrist_radius,
                'wrist-end-effector',
            ),
            (
                'arm_link5',
                'arm_gripper_left_link',
                self.gripper_radius,
                'left-gripper',
            ),
            (
                'arm_link5',
                'arm_gripper_right_link',
                self.gripper_radius,
                'right-gripper',
            ),
        ]

        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        self.sub = self.create_subscription(
            Detection3DArray,
            self.input_topic,
            self.measurement_callback,
            1
        )

        self.pub = self.create_publisher(
            Detection3DArray,
            self.output_topic,
            10
        )

                # Exact-stamp TF synchronization.
        #
        # Keep one protected pending measurement plus
        # a small bounded FIFO for following measurements.
        self.pending_message = None
        self.pending_since_ns = None

        self.waiting_messages = deque(
            maxlen=8
        )

        self.queue_overflow_count = 0
        self.tf_timeout_drop_count = 0

        self.process_timer = self.create_timer(
            0.01,
            self.process_pending
        )

        self.frame_count = 0
        self.total_kept = 0
        self.total_rejected = 0

        self.get_logger().info(
            'Object Self-Filter started'
        )

        self.get_logger().info(
            f'Input: {self.input_topic}'
        )

        self.get_logger().info(
            f'Output: {self.output_topic}'
        )

        self.get_logger().info(
            'Capsule radii: '
            f'arm={self.arm_radius:.3f}m '
            f'wrist={self.wrist_radius:.3f}m '
            f'gripper={self.gripper_radius:.3f}m'
        )

    def measurement_callback(
        self,
        msg
    ):

        arrival_ns = (
            self.get_clock()
            .now()
            .nanoseconds
        )

        if self.pending_message is None:

            self.pending_message = msg
            self.pending_since_ns = arrival_ns

            return

        if (
            len(self.waiting_messages)
            ==
            self.waiting_messages.maxlen
        ):

            self.waiting_messages.popleft()

            self.queue_overflow_count += 1

            self.get_logger().warning(
                'Self-filter queue overflow: '
                'dropped oldest waiting measurement '
                f'(total={self.queue_overflow_count})'
            )

        self.waiting_messages.append(
            (
                msg,
                arrival_ns,
            )
        )

    def promote_next(
        self
    ):

        if self.waiting_messages:

            (
                self.pending_message,
                self.pending_since_ns,
            ) = self.waiting_messages.popleft()

        else:

            self.pending_message = None
            self.pending_since_ns = None

    def lookup_frame_position(
        self,
        frame,
        stamp
    ):

        transform = (
            self.tf_buffer.lookup_transform(
                self.world_frame,
                frame,
                stamp,
                timeout=Duration(
                    seconds=0.0
                )
            )
        )

        translation = (
            transform.transform.translation
        )

        return (
            float(translation.x),
            float(translation.y),
            float(translation.z),
        )

    def build_segments(
        self,
        stamp
    ):

        frame_names = set()

        for (
            start_frame,
            end_frame,
            _,
            _,
        ) in self.segment_definitions:

            frame_names.add(
                start_frame
            )

            frame_names.add(
                end_frame
            )

        positions = {}

        for frame in frame_names:

            positions[frame] = (
                self.lookup_frame_position(
                    frame,
                    stamp
                )
            )

        segments = []

        for (
            start_frame,
            end_frame,
            radius,
            label,
        ) in self.segment_definitions:

            segments.append(
                (
                    positions[
                        start_frame
                    ],

                    positions[
                        end_frame
                    ],

                    radius,

                    label,
                )
            )

        return segments

    def process_pending(
        self
    ):

        if self.pending_message is None:
            return

        msg = self.pending_message

        if not msg.header.frame_id:

            self.promote_next()

            return

        if (
            msg.header.frame_id
            !=
            self.world_frame
        ):

            self.get_logger().warning(
                'Drop measurement: expected '
                f'frame={self.world_frame}, '
                f'got={msg.header.frame_id}'
            )

            self.promote_next()

            return

        stamp = Time.from_msg(
            msg.header.stamp
        )

        try:

            segments = self.build_segments(
                stamp
            )

        except Exception:

            now_ns = (
                self.get_clock()
                .now()
                .nanoseconds
            )

            if self.pending_since_ns is None:

                self.pending_since_ns = now_ns

            waited = (
                now_ns
                -
                self.pending_since_ns
            ) * 1e-9

            measurement_age = (
                now_ns
                -
                stamp.nanoseconds
            ) * 1e-9

            if (
                waited
                <=
                self.tf_wait_timeout
            ):

                return

            self.tf_timeout_drop_count += 1

            self.get_logger().warning(
                'Drop measurement: exact robot TF '
                'unavailable '
                f'after waiting {waited:.3f}s '
                f'(measurement age='
                f'{measurement_age:.3f}s, '
                f'total={self.tf_timeout_drop_count})'
            )

            self.promote_next()

            return

        self.filter_message(
            msg,
            segments
        )

        self.promote_next()

    def is_robot_self(
        self,
        point,
        segments
    ):

        nearest_distance = float(
            'inf'
        )

        nearest_label = ''

        nearest_radius = 0.0

        for (
            start,
            end,
            radius,
            label,
        ) in segments:

            distance = (
                point_to_segment_distance(
                    point,
                    start,
                    end
                )
            )

            if (
                distance
                <
                nearest_distance
            ):

                nearest_distance = (
                    distance
                )

                nearest_label = label
                nearest_radius = radius

            if distance <= radius:

                return (
                    True,
                    label,
                    distance,
                    radius,
                )

        return (
            False,
            nearest_label,
            nearest_distance,
            nearest_radius,
        )

    def filter_message(
        self,
        msg,
        segments
    ):

        output = Detection3DArray()

        output.header = msg.header

        kept = 0
        rejected = 0

        for detection in (
            msg.detections
        ):

            if not detection.results:

                continue

            position = (
                detection.results[
                    0
                ].pose.pose.position
            )

            point = (
                float(position.x),
                float(position.y),
                float(position.z),
            )

            (
                is_self,
                nearest_label,
                distance,
                radius,
            ) = self.is_robot_self(
                point,
                segments
            )

            if is_self:

                rejected += 1

                class_id = (
                    detection.results[
                        0
                    ].hypothesis.class_id
                )

                self.get_logger().warning(
                    'SELF REJECT '
                    f'id={detection.id} '
                    f'class={class_id} '
                    f'near={nearest_label} '
                    f'd={distance*1000:.1f}mm '
                    f'limit={radius*1000:.1f}mm'
                )

                continue

            output.detections.append(
                detection
            )

            kept += 1

        self.pub.publish(
            output
        )

        self.frame_count += 1

        self.total_kept += kept
        self.total_rejected += rejected

        if (
            self.frame_count
            %
            20
            ==
            0
        ):

            self.get_logger().info(
                'Self-filter: '
                f'input={len(msg.detections)} '
                f'kept={kept} '
                f'rejected={rejected}'
            )


def main(args=None):

    rclpy.init(
        args=args
    )

    node = ObjectSelfFilter()

    try:

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.try_shutdown()


if __name__ == '__main__':

    main()
