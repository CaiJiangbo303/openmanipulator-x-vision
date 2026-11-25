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
    )) if name == "TRACK" else bool(targets)

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

print("Warmup: 2 seconds", flush=True)
end = time.monotonic() + 2

while time.monotonic() < end:
    rclpy.spin_once(node, timeout_sec=0.02)

print("Move blue block for 15 seconds", flush=True)


seen_classes = Counter()
previous_class = None
class_switches = 0
hold_run = 0
max_hold_run = 0


prompts = [
    (0, "保持方块静止"),
    (4, "现在完全遮挡方块！"),
    (6, "现在移开遮挡物！"),
    (9, "开始正常移动方块")
]

prompt_index = 0

print("等待初始授权，请保持方块静止", flush=True)

pre_acquire = Counter()
deadline = time.monotonic() + 30

while identity is None:
    rclpy.spin_once(node, timeout_sec=0.02)

    reason, ident = evaluate(0.50)
    pre_acquire[reason] += 1

    if ident is not None:
        identity = ident
        break

    if time.monotonic() >= deadline:
        print("初始授权超时:", dict(pre_acquire))
        node.destroy_node()
        rclpy.shutdown()
        raise SystemExit(1)

print("INITIAL LOCKED:", identity, flush=True)

previous_generation = identity[1]

start = time.monotonic()


next_tick = start
acquire = Counter()

while time.monotonic() - start < 15:
    rclpy.spin_once(node, timeout_sec=0.01)

    now = time.monotonic()

    elapsed = now - start

    if prompt_index < len(prompts):
        seconds, message = prompts[prompt_index]

        if elapsed >= seconds:
            print(f"\a", end="")
            print(f"\n>>> [{elapsed:.1f}s] {message} <<<\n",
                  flush=True)
            prompt_index += 1

    if now < next_tick:

        continue

    next_tick = now + 0.1

    if identity is None:
        reason, ident = evaluate(0.50)
        acquire[reason] += 1

        if ident is None:
            continue

        identity = ident
        print("LOCKED:", identity, flush=True)

    for threshold in counts:
        reason, _ = evaluate(threshold)
        counts[threshold][reason] += 1

        if threshold == 0.40 and latest:
            for o in latest.states:
                if o.track_id != identity[0]:
                    continue

                if not o.valid_3d or o.stale or o.missed != 0:
                    continue

                if o.generation_id != previous_generation:
                    print(
                        "GENERATION CHANGED:",
                        previous_generation,
                        "->", o.generation_id,
                        flush=True
                    )

                    previous_generation = o.generation_id


        if threshold == 0.40:
            hold_run = hold_run + 1 if reason != "PASS" else 0
            max_hold_run = max(max_hold_run, hold_run)

            for o in latest.states if latest else []:
                if (o.track_id, o.generation_id) != identity:
                    continue

                if o.stale or o.missed != 0:
                    continue

                seen_classes[o.class_id] += 1

                if previous_class is not None and previous_class != o.class_id:
                    class_switches += 1
                    print("CLASS SWITCH, ID KEPT:",
                          previous_class, "->", o.class_id,
                          identity, flush=True)

                previous_class = o.class_id


        if threshold == 0.40 and reason != "PASS":
            t = time.monotonic() - start

            if t - last_trace >= 0.3:
                last_trace = t

                print(f"\nTRACE t={t:.2f}s {reason}")

                for name in ("DET", "TRACK", "FAST"):
                    print(name, describe(upstream.get(name)))

                if latest:
                    print("STATE", [
                        (
                            o.track_id,
                            o.class_id,
                            round(o.confidence, 3),
                            o.missed,
                            o.stale
                        )
                        for o in latest.states
                    ])


print("\n===== ACQUIRE =====")
print(dict(acquire))
print("Locked identity:", identity)
print("Observed mask IDs:", list(masks.keys()))

for threshold, c in counts.items():
    total = sum(c.values())
    print(f"\n===== TRACK CONF {threshold} =====")
    print("Total ticks:", total)
    print("Passed:", c["PASS"])
    print("Pass rate:", round(
        100*c["PASS"]/max(1,total), 2), "%")
    print("Reasons:", dict(c))


print("\n===== CONTINUITY =====")
print("Observed classes:", dict(seen_classes))
print("Same identity class switches:", class_switches)
print("Longest consecutive rejected ticks:", max_hold_run)

node.destroy_node()

rclpy.shutdown()
