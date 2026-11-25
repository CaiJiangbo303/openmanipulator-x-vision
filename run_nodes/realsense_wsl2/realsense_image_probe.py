import time
import statistics
import rclpy

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

rclpy.init()
node = Node("realsense_image_probe")

data = {"color": [], "aligned_depth": []}

def callback(msg, stream):
    now = time.monotonic_ns()
    wall = time.time_ns()
    stamp = msg.header.stamp
    stamp_ns = stamp.sec * 10**9 + stamp.nanosec

    data[stream].append((
        now,
        stamp_ns,
        (wall - stamp_ns) / 1e6
    ))

topics = {
    "color": "/camera/camera/color/image_raw",
    "aligned_depth":
        "/camera/camera/aligned_depth_to_color/image_raw"
}

for stream, topic in topics.items():
    node.create_subscription(
        Image, topic,
        lambda msg, s=stream: callback(msg, s),
        qos_profile_sensor_data
    )

print("Warmup: 2 seconds", flush=True)
end = time.monotonic() + 2

while time.monotonic() < end:
    rclpy.spin_once(node, timeout_sec=0.05)

for rows in data.values():
    rows.clear()

print("Measuring: 15 seconds", flush=True)

start = time.monotonic()
while time.monotonic() - start < 15:
    rclpy.spin_once(node, timeout_sec=0.05)

for stream, rows in data.items():
    print(f"\n===== {stream.upper()} =====")
    print("Received images:", len(rows))

    if len(rows) < 2:
        continue

    gaps = []
    for a, b in zip(rows, rows[1:]):
        gap = (b[0] - a[0]) / 1e6
        stamp_gap = (b[1] - a[1]) / 1e6
        gaps.append(gap)

        if gap > 100 or stamp_gap <= 0 or stamp_gap > 100:
            print(
                f"EVENT t={b[0]/1e9-start:.2f}s "
                f"recv_gap={gap:.2f}ms "
                f"stamp_gap={stamp_gap:.2f}ms"
            )

    ages = sorted(r[2] for r in rows)

    print("Receive Hz:", round(
        (len(rows)-1)*1e9 /
        (rows[-1][0]-rows[0][0]), 2))
    print("Max gap ms:", round(max(gaps), 2))
    print("Gap std ms:", round(statistics.pstdev(gaps), 2))
    print("Gaps over 100ms:", sum(g > 100 for g in gaps))
    print("Min age ms:", round(ages[0], 2))
    print("P95 age ms:", round(ages[int(0.95*(len(ages)-1))], 2))
    print("Max age ms:", round(ages[-1], 2))

node.destroy_node()
rclpy.shutdown()
