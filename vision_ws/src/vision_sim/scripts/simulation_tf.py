#!/usr/bin/env python3

import rclpy

from rclpy.node import Node
from rclpy.parameter import Parameter

from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from tf2_ros import TransformBroadcaster


class SimulationTF(Node):

    def __init__(self):

        super().__init__(
            'vision_sim_localization',
            parameter_overrides=[
                Parameter('use_sim_time', value=True)
            ]
        )

        self.broadcaster = TransformBroadcaster(self)

        self.create_subscription(
            TFMessage,
            '/simulation/poses',
            self.pose_callback,
            10
        )

        self.get_logger().info(
            'Simulation TF started: world -> dji_f550/base_link'
        )

    def pose_callback(self, msg):

        for item in msg.transforms:

            if item.child_frame_id != 'f550_high':
                continue

            stamp = item.header.stamp

            # Gazebo Fortress Pose_V 经 bridge 后时间戳可能为 0
            if stamp.sec == 0 and stamp.nanosec == 0:
                stamp = self.get_clock().now().to_msg()

            tf = TransformStamped()

            tf.header.stamp = stamp
            tf.header.frame_id = 'world'
            tf.child_frame_id = 'dji_f550/base_link'

            tf.transform = item.transform

            self.broadcaster.sendTransform(tf)


def main(args=None):

    rclpy.init(args=args)

    node = SimulationTF()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
