#!/usr/bin/env python3

import rclpy

from rclpy.node import Node

from rosgraph_msgs.msg import Clock
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


TARGET_FRAMES = {
    'arm_link2',
    'arm_link3',
    'arm_link4',
    'arm_link5',
    'arm_gripper_left_link',
    'arm_gripper_right_link',
}


def quat_multiply(a, b):

    ax, ay, az, aw = a
    bx, by, bz, bw = b

    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def rotate_vector(q, v):

    qx, qy, qz, qw = q
    vx, vy, vz = v

    uvx = qy * vz - qz * vy
    uvy = qz * vx - qx * vz
    uvz = qx * vy - qy * vx

    uuvx = qy * uvz - qz * uvy
    uuvy = qz * uvx - qx * uvz
    uuvz = qx * uvy - qy * uvx

    return (
        vx + 2.0 * (qw * uvx + uuvx),
        vy + 2.0 * (qw * uvy + uuvy),
        vz + 2.0 * (qw * uvz + uuvz),
    )


class SimulationArmTf(Node):

    def __init__(self):

        super().__init__(
            'simulation_arm_tf'
        )

        self.latest_clock = None

        self.broadcaster = (
            TransformBroadcaster(
                self
            )
        )

        self.create_subscription(
            Clock,
            '/clock',
            self.clock_callback,
            10
        )

        self.create_subscription(
            TFMessage,
            '/simulation/poses',
            self.pose_callback,
            10
        )

        self.frame_count = 0

        self.get_logger().info(
            'Simulation arm TF bridge started'
        )

        self.get_logger().info(
            'Source: /simulation/poses'
        )

        self.get_logger().info(
            'Publishing world TF for '
            'movable arm links'
        )

    def clock_callback(
        self,
        msg
    ):

        self.latest_clock = msg.clock

    def pose_callback(
        self,
        msg
    ):

        if self.latest_clock is None:
            return

        root = None
        local = {}

        for item in msg.transforms:

            name = item.child_frame_id

            if name == 'f550_high':

                root = item

            elif name in TARGET_FRAMES:

                local[name] = item

        if root is None:
            return

        root_t = root.transform.translation
        root_r = root.transform.rotation

        root_position = (
            float(root_t.x),
            float(root_t.y),
            float(root_t.z),
        )

        root_quaternion = (
            float(root_r.x),
            float(root_r.y),
            float(root_r.z),
            float(root_r.w),
        )

        output = []

        for name in sorted(
            TARGET_FRAMES
        ):

            if name not in local:
                continue

            item = local[name]

            local_t = (
                float(
                    item.transform.translation.x
                ),
                float(
                    item.transform.translation.y
                ),
                float(
                    item.transform.translation.z
                ),
            )

            local_q = (
                float(
                    item.transform.rotation.x
                ),
                float(
                    item.transform.rotation.y
                ),
                float(
                    item.transform.rotation.z
                ),
                float(
                    item.transform.rotation.w
                ),
            )

            rotated = rotate_vector(
                root_quaternion,
                local_t
            )

            world_position = (
                root_position[0]
                +
                rotated[0],

                root_position[1]
                +
                rotated[1],

                root_position[2]
                +
                rotated[2],
            )

            world_quaternion = (
                quat_multiply(
                    root_quaternion,
                    local_q
                )
            )

            tf = TransformStamped()

            tf.header.stamp = (
                self.latest_clock
            )

            tf.header.frame_id = 'world'

            tf.child_frame_id = name

            tf.transform.translation.x = (
                world_position[0]
            )

            tf.transform.translation.y = (
                world_position[1]
            )

            tf.transform.translation.z = (
                world_position[2]
            )

            tf.transform.rotation.x = (
                world_quaternion[0]
            )

            tf.transform.rotation.y = (
                world_quaternion[1]
            )

            tf.transform.rotation.z = (
                world_quaternion[2]
            )

            tf.transform.rotation.w = (
                world_quaternion[3]
            )

            output.append(
                tf
            )

        if output:

            self.broadcaster.sendTransform(
                output
            )

        self.frame_count += 1

        if self.frame_count % 100 == 0:

            self.get_logger().info(
                'Published arm TF frames: '
                f'{len(output)}'
            )


def main(args=None):

    rclpy.init(
        args=args
    )

    node = SimulationArmTf()

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
