import json
import time
import rclpy

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from realsense2_camera_msgs.msg import Metadata
from diagnostic_msgs.msg import DiagnosticArray

rclpy.init()
node = Node("realsense_probe")

data = {"color": [], "depth": []}
diagnostics = {}

def callback(msg, stream):
    now = time.monotonic()
    wall = time.time_ns()
    m = json.loads(msg.json_data)
    stamp = msg.header.stamp
    stamp_ns = stamp.sec * 10**9 + stamp.nanosec

    data[stream].append((
        now,
        m.get("frame_counter", m.get("frame_number")),
        m.get("hw_timestamp"),
        stamp_ns / 1e9,
        (wall - stamp_ns) / 1e6
    ))

def diag_callback(msg):
    for s in msg.status:
        v = {x.key: x.value for x in s.values}
        if "Actual frequency (Hz)" in v:
            diagnostics[s.name] = v["Actual frequency (Hz)"]

for stream in data:
    node.create_subscription(
        Metadata,
        f"/camera/camera/{stream}/metadata",
        lambda msg, s=stream: callback(msg, s),
        qos_profile_sensor_data
    )

node.create_subscription(
    DiagnosticArray, "/diagnostics",
    diag_callback, qos_profile_sensor_data
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
    print("Received frames:", len(rows))

    if len(rows) < 2:
        continue

    gaps = []
    skipped = 0
    resets = 0

    for a, b in zip(rows, rows[1:]):
        dt = (b[0] - a[0]) * 1000
        dc = b[1] - a[1]
        ds = (b[3] - a[3]) * 1000
        dh = None if None in (a[2], b[2]) else b[2] - a[2]

        skipped += max(0, dc - 1)
        resets += int(dc <= 0)

        gaps.append(dt)

        if dt > 100 or dc != 1 or ds <= 0 or ds > 100:
            print(
                f"EVENT t={b[0]-start:.2f}s "
                f"recv_gap={dt:.1f}ms "
                f"counter_delta={dc} "
                f"stamp_gap={ds:.1f}ms "
                f"hw_delta_raw={dh}"
            )

    print("Receive Hz:", round(
        (len(rows)-1)/(rows[-1][0]-rows[0][0]), 2))
    print("Max receive gap ms:", round(max(gaps), 2))
    print("Missing counter values:", skipped)
    print("Counter resets/repeats:", resets)
    print("Min age ms:", round(min(r[4] for r in rows), 2))
    print("Max age ms:", round(max(r[4] for r in rows), 2))

print("\n===== DRIVER DIAGNOSTICS =====")
print(diagnostics)

node.destroy_node()
rclpy.shutdown()
