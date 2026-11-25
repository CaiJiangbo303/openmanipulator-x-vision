import time
import math
from collections import Counter

import rclpy
from rosidl_runtime_py.utilities import get_message
from rclpy.qos import qos_profile_sensor_data
from vision_msgs.msg import Detection3DArray

rclpy.init()
node = rclpy.create_node("visual_gate_probe")

topic = "/bench/timestamp_guard/states"

deadline = time.monotonic() + 5
while topic not in dict(node.get_topic_names_and_types()):
    if time.monotonic() > deadline:
        raise RuntimeError("State topic unavailable")
    rclpy.spin_once(node, timeout_sec=0.1)

msg_type = get_message(
    dict(node.get_topic_names_and_types())[topic][0]
)

latest = None
receipt = 0.0
masks = {}

identity = None
counts = {
    0.45: Counter(),
    0.40: Counter()
}

def stamp(s):
    return s.sec + s.nanosec / 1e9

def xyz(p):
    return (p.x, p.y, p.z)

def norm(p):
    return math.sqrt(sum(x*x for x in p))

def state_cb(msg):
    global latest, receipt
    latest = msg
    receipt = time.monotonic()

def mask_cb(msg):
    for d in msg.detections:
        if not d.results:
            continue

        r = d.results[0]

        masks[d.id] = (
            stamp(msg.header.stamp),
            xyz(r.pose.pose.position),
            r.hypothesis.score,
            msg.header.frame_id
        )

node.create_subscription(
    msg_type, topic, state_cb,
    qos_profile_sensor_data
)

node.create_subscription(
    Detection3DArray,
    "/object_localizer/detections_3d_mask",
    mask_cb,
    qos_profile_sensor_data
)


from vision_msgs.msg import Detection2DArray

upstream = {}

observed_ids = {}

def capture(msg, name):
    upstream[name] = msg

    if name not in ("DET", "TRACK"):
        return

    if "start" not in globals():
        return

    targets = [
        d for d in msg.detections
        if d.results and
        d.results[0].hypothesis.class_id
        in {"blue block", "plastic cube"}
    ]

    current = tuple(sorted(
        d.id for d in targets
    )) if name == "TRACK" else len(targets)

    if observed_ids.get(name) != current:
        observed_ids[name] = current
        elapsed = time.monotonic() - start

        print(
            f"EVENT [{elapsed:.2f}s] "
            f"{name}: {current}",
            flush=True
        )

for name, topic_name, typ in [
    ("DET", "/object_detector/detections", Detection2DArray),
    ("TRACK", "/object_tracker/detections", Detection2DArray),
    ("FAST", "/object_localizer/detections_3d_fast", Detection3DArray)
]:
    node.create_subscription(
        typ, topic_name,
        lambda msg, n=name: capture(msg, n),
        qos_profile_sensor_data
    )

def describe(msg):
    if msg is None:
        return "NO MESSAGE"

    items = []

    for d in msg.detections:
        if not d.results:
            continue

        h = d.results[0].hypothesis

        items.append(
            f"{d.id or '-'}:{h.class_id}:{h.score:.3f}"
        )

    age = (
        node.get_clock().now().nanoseconds / 1e9
        - stamp(msg.header.stamp)
    ) * 1000

    return f"age={age:.1f}ms " + str(items)

last_trace = -1.0

def evaluate(threshold):
    if latest is None:
        return "NO STATE", None

    now = node.get_clock().now().nanoseconds / 1e9

    if time.monotonic() - receipt > 0.20:
        return "STATE RECEIPT STALE", None

    targets = [
        s for s in latest.states
        if s.class_id in {"blue block", "plastic cube"}
    ]

    if len(targets) != 1:
        return "MISSING OR AMBIGUOUS", None

    s = targets[0]
    ident = (s.track_id, s.generation_id)

    if not all(ident):
        return "INVALID IDENTITY", None

    if identity is not None and ident != identity:
        return "IDENTITY CHANGED", None

    if sum(x.track_id == s.track_id
           for x in latest.states) != 1:
        return "AMBIGUOUS ID", None

    if not s.valid_3d or s.stale \
            or s.missed != 0 or s.state != 1:
        return "TARGET INVALID", None

    source = stamp(s.source_stamp)

    if not 0 <= now-source <= 0.20:
        return "SOURCE STALE", None

    if not 0 <= s.measurement_age_ms <= 150:
        return "MEASUREMENT STALE", None

    if not s.confidence >= threshold:
        return "LOW CONFIDENCE", None

    if not norm(xyz(s.position_std)) <= 0.03:
        return "UNCERTAINTY", None

    pc = xyz(s.position)

    if not all(math.isfinite(x) for x in pc):
        return "INVALID POSITION", None

    if not 0.12 <= pc[2] <= 0.48:
        return "DEPTH LIMIT", None

    evidence = masks.get(s.track_id)

    if evidence is None:
        return "WAIT MASK EVIDENCE", None

    ms, mp, score, frame = evidence

    if frame != latest.header.frame_id:
        return "FRAME MISMATCH", None

    if not score >= 0.40:
        return "MASK LOW CONFIDENCE", None

    if not 0 <= now-ms <= 0.25:
        return "MASK STALE", None

    if abs(ms-source) > 0.15:
        return "MASK SOURCE MISMATCH", None

    delta = norm([a-b for a,b in zip(pc, mp)])

    if not delta <= 0.025:
        return "MASK POSITION MISMATCH", None

    return "PASS", ident



from recovery_manager import (
    Observation,
    Status,
    TrackingRecoveryManager
)

def raw_candidate_guard():
    now = node.get_clock().now().nanoseconds / 1e9

    for name in ("DET", "TRACK"):
        msg = upstream.get(name)

        if msg is None:
            return name + " NO DATA"

        age = now - stamp(msg.header.stamp)

        if not 0 <= age <= 0.20:
            return name + " STALE"

        targets = [
            d for d in msg.detections
            if d.results
            and d.results[0].hypothesis.class_id
            in {"blue block", "plastic cube"}
        ]

        if len(targets) != 1:
            return name + " MISSING OR AMBIGUOUS"

    return "PASS"


def read_evidence(threshold, acquire=False):
    guard_reason = raw_candidate_guard()

    if guard_reason != "PASS":
        return guard_reason, None

    global identity

    saved = identity

    if acquire:
        identity = None

    try:
        reason, ident = evaluate(threshold)
    finally:
        identity = saved

    if reason != "PASS" or ident is None:
        return reason, None

    now_ros = node.get_clock().now().nanoseconds / 1e9
    header_age = now_ros - stamp(latest.header.stamp)

    if not 0 <= header_age <= 0.20:
        return "STATE HEADER STALE", None

    o = next(
        s for s in latest.states
        if (s.track_id, s.generation_id) == ident
    )

    evidence = Observation(
        track_id=o.track_id,
        generation_id=o.generation_id,
        position=xyz(o.position),
        source_stamp=stamp(o.source_stamp),
        mask_stamp=masks[o.track_id][0]
    )

    if not evidence.valid():
        return "INVALID EVIDENCE", None

    return "PASS", evidence


print("等待初始授权，请保持方块静止", flush=True)

deadline = time.monotonic() + 30

while True:
    rclpy.spin_once(node, timeout_sec=0.02)

    reason, initial = read_evidence(0.50, acquire=True)

    if initial is not None:
        break

    if time.monotonic() > deadline:
        print("INITIAL ACQUIRE FAILED:", reason)
        node.destroy_node()
        rclpy.shutdown()
        raise SystemExit(1)

manager = TrackingRecoveryManager(
    initial,
    time.monotonic()
)

identity = manager.identity
initial_identity = identity

print("INITIAL LOCKED:", identity, flush=True)

prompts = [
    (0, "保持方块静止"),
    (4, "现在完全遮挡方块！"),
    (6, "现在移开遮挡物！保持方块位置不变"),
    (9, "如果已重新授权，开始正常移动方块")
]


import os

scenario = os.environ.get("RECOVERY_SCENARIO", "normal")

if scenario == "timeout":
    prompts = [
        (0, "保持方块静止"),
        (4, "完全遮挡方块！"),
        (9, "移开遮挡物！"),
        (12, "保持静止，检查超时保护")
    ]

elif scenario == "ambiguity":
    prompts = [
        (0, "单个方块保持静止"),
        (4, "完全遮挡原方块！"),
        (6, "同时露出两个方块！"),
        (9, "保持两个方块静止")
    ]

elif scenario != "normal":
    raise ValueError("Unknown recovery scenario")

print("TEST SCENARIO:", scenario, flush=True)

prompt_index = 0

stats = Counter()
previous_status = manager.status

start = time.monotonic()
next_tick = start

while time.monotonic() - start < 15:

    rclpy.spin_once(node, timeout_sec=0.01)

    now = time.monotonic()
    elapsed = now - start

    if prompt_index < len(prompts):
        t, message = prompts[prompt_index]

        if elapsed >= t:
            print(chr(7) + chr(10) +
                  f">>> [{elapsed:.1f}s] {message} <<<",
                  flush=True)
            prompt_index += 1

    if now < next_tick:
        continue

    next_tick = now + 0.1

    identity = manager.identity

    tracking_reason, tracking = read_evidence(0.40)

    recovery_reason, recovery = read_evidence(
        0.50, acquire=True
    )

    old_identity = manager.identity
    old_position = manager.observation.position
    old_last_good = manager.last_good

    status = manager.step(
        now,
        tracking=tracking,
        recovery=recovery
    )

    identity = manager.identity
    stats[status.value] += 1

    if status != previous_status:
        print(
            f"STATE [{elapsed:.2f}s]: "
            f"{previous_status.value} -> {status.value} "
            f"| authorized={manager.authorized} "
            f"| tracking={tracking_reason} "
            f"| recovery={recovery_reason}",
            flush=True
        )
        previous_status = status

    if status == Status.REAUTHORIZED:
        delta = norm([
            a-b for a,b in zip(
                manager.observation.position,
                old_position
            )
        ])

        print("SHADOW REAUTHORIZED:",
              old_identity, "->", identity)
        print("Position delta mm:", round(delta*1000, 2))
        print("Recovery gap seconds:",
              round(now-old_last_good, 3))

print(chr(10) + "===== MODULE SUMMARY =====")
print("Initial identity:", initial_identity)
print("Final identity:", manager.identity)
print("Final status:", manager.status.value)
print("Authorized:", manager.authorized)
print("Reauthorizations:", manager.reauth_count)
print("Stats:", dict(stats))

node.destroy_node()
rclpy.shutdown()
