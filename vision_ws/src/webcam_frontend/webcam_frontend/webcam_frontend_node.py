import socket
import struct
import time

import cv2
import numpy as np

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class WebcamFrontend(Node):

    def __init__(self):
        super().__init__('webcam_frontend')

        self.declare_parameter('host', '172.29.128.1')
        self.declare_parameter('port', 5001)
        self.declare_parameter('image_topic', '/camera/image')
        self.declare_parameter('frame_id', 'webcam_optical_frame')
        self.declare_parameter('reconnect_delay', 1.0)

        self.host = str(
            self.get_parameter('host').value
        )

        self.port = int(
            self.get_parameter('port').value
        )

        self.image_topic = str(
            self.get_parameter('image_topic').value
        )

        self.frame_id = str(
            self.get_parameter('frame_id').value
        )

        self.reconnect_delay = float(
            self.get_parameter('reconnect_delay').value
        )

        self.bridge = CvBridge()

        self.publisher = self.create_publisher(
            Image,
            self.image_topic,
            qos_profile_sensor_data
        )

        self.sock = None
        self.buffer = bytearray()

        self.frame_count = 0
        self.last_report_ns = (
            self.get_clock().now().nanoseconds
        )
        self.report_frame_count = 0

        self.timer = self.create_timer(
            0.001,
            self.poll
        )

        self.get_logger().info(
            'Webcam Frontend started'
        )

        self.get_logger().info(
            f'Windows stream: {self.host}:{self.port}'
        )

        self.get_logger().info(
            f'Publishing: {self.image_topic}'
        )

    def connect(self):

        if self.sock is not None:
            return True

        try:
            sock = socket.socket(
                socket.AF_INET,
                socket.SOCK_STREAM
            )

            sock.settimeout(2.0)

            sock.connect(
                (self.host, self.port)
            )

            sock.settimeout(0.01)

            self.sock = sock
            self.buffer.clear()

            self.get_logger().info(
                f'Connected to '
                f'{self.host}:{self.port}'
            )

            return True

        except OSError as exc:

            self.get_logger().warn(
                f'Camera connection failed: {exc}',
                throttle_duration_sec=2.0
            )

            try:
                sock.close()
            except Exception:
                pass

            time.sleep(
                self.reconnect_delay
            )

            return False

    def disconnect(self):

        if self.sock is not None:

            try:
                self.sock.close()
            except Exception:
                pass

        self.sock = None
        self.buffer.clear()

    def recv_more(self):

        try:
            chunk = self.sock.recv(65536)

            if not chunk:
                raise ConnectionError(
                    'camera server closed connection'
                )

            self.buffer.extend(chunk)

            return True

        except socket.timeout:
            return False

        except (
            OSError,
            ConnectionError
        ) as exc:

            self.get_logger().warn(
                f'Camera stream disconnected: {exc}'
            )

            self.disconnect()

            return False

    def pop_frame(self):

        if len(self.buffer) < 4:
            return None

        frame_size = struct.unpack(
            '!I',
            self.buffer[:4]
        )[0]

        if (
            frame_size <= 0
            or
            frame_size > 20_000_000
        ):
            self.get_logger().error(
                f'Invalid JPEG size: {frame_size}'
            )

            self.disconnect()

            return None

        total_size = 4 + frame_size

        if len(self.buffer) < total_size:
            return None

        jpeg = bytes(
            self.buffer[4:total_size]
        )

        del self.buffer[:total_size]

        encoded = np.frombuffer(
            jpeg,
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            encoded,
            cv2.IMREAD_COLOR
        )

        return frame

    def publish_frame(self, frame):

        if frame is None:
            return

        # Windows/OpenCV provides BGR, while the existing
        # perception pipeline expects RGB8 just like the
        # RealSense / Gazebo frontend.
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        msg = self.bridge.cv2_to_imgmsg(
            rgb_frame,
            encoding='rgb8'
        )

        msg.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        msg.header.frame_id = (
            self.frame_id
        )

        self.publisher.publish(msg)

        self.frame_count += 1
        self.report_frame_count += 1

        now_ns = (
            self.get_clock()
            .now()
            .nanoseconds
        )

        elapsed = (
            now_ns
            -
            self.last_report_ns
        ) * 1e-9

        if elapsed >= 5.0:

            fps = (
                self.report_frame_count
                /
                elapsed
            )

            h, w = frame.shape[:2]

            self.get_logger().info(
                f'RGB stream '
                f'{w}x{h} '
                f'{fps:.1f} FPS'
            )

            self.last_report_ns = now_ns
            self.report_frame_count = 0

    def poll(self):

        if not self.connect():
            return

        self.recv_more()

        # Drain all currently complete frames.
        # Keeping only the newest frame prevents latency
        # from accumulating if downstream becomes slower.
        newest = None

        while True:

            frame = self.pop_frame()

            if frame is None:
                break

            newest = frame

        if newest is not None:
            self.publish_frame(newest)


def main(args=None):

    rclpy.init(args=args)

    node = WebcamFrontend()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.disconnect()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
