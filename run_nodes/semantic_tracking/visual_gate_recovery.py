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


def get_state(ident):
    return next(
        o for o in latest.states
        if (o.track_id, o.generation_id) == ident
    )

def candidate():
    global identity
    saved = identity
    identity = None
    try:
        return evaluate(0.50)
    finally:
        identity = saved

print("等待初始锁定，请保持方块静止", flush=True)

deadline = time.monotonic() + 30

while identity is None:
    rclpy.spin_once(node, timeout_sec=0.02)
    reason, ident = evaluate(0.50)

    if ident is not None:
        identity = ident
        break

    if time.monotonic() > deadline:
        print("INITIAL ACQUIRE FAILED:", reason)
        node.destroy_node()
        rclpy.shutdown()
        raise SystemExit(1)

initial_identity = identity
locked_track = identity[0]

print("INITIAL LOCKED:", identity, flush=True)

o = get_state(identity)
last_pos = xyz(o.position)
last_source = stamp(o.source_stamp)
last_mask = masks[locked_track][0]
last_good = time.monotonic()

hold_since = None
reauth_required = False
expired = False

pending = None
pending_stamps = set()
pending_since = None

reauth_count = 0
stats = Counter()

prompts = [
    (0, "保持方块静止"),
    (4, "现在完全遮挡方块！"),
    (6, "现在移开遮挡物！保持方块位置不变"),
    (9, "若已恢复授权，开始正常移动方块")
]

prompt_index = 0
start = time.monotonic()
next_tick = start

while time.monotonic() - start < 15:
    rclpy.spin_once(node, timeout_sec=0.01)

    now = time.monotonic()
    elapsed = now - start

    if prompt_index < len(prompts):
        t, message = prompts[prompt_index]
        if elapsed >= t:
            print(chr(7) + f"\\n>>> {message} <<<",
                  flush=True)
            prompt_index += 1

    if now < next_tick:
        continue

    next_tick = now + 0.1
    reason, ident = evaluate(0.40)

    if not expired and not reauth_required and reason == "PASS" and now-last_good <= 0.35:
        if hold_since is not None:
            print("SHORT HOLD CLEARED", flush=True)

        hold_since = None
        o = get_state(identity)
        last_pos = xyz(o.position)
        last_source = stamp(o.source_stamp)
        last_mask = masks[locked_track][0]
        last_good = now
        stats["TRACKING"] += 1
        continue

    if hold_since is None:
        hold_since = now
        print("HOLD:", reason, flush=True)

    if reason == "IDENTITY CHANGED" or now-last_good > 0.35:
        reauth_required = True

    if now-last_good > 3.0:
        if not expired:
            print("RECOVERY EXPIRED", flush=True)
        expired = True

    if expired:
        stats["EXPIRED"] += 1
        continue

    if not reauth_required:
        stats["HOLD"] += 1
        continue

    reason, ident = candidate()

    good = False

    if reason == "PASS" and ident[0] == locked_track:
        o = get_state(ident)
        source = stamp(o.source_stamp)
        mp_stamp = masks[locked_track][0]

        delta = norm([
            a-b for a,b in zip(xyz(o.position), last_pos)
        ])

        good = (
            delta <= 0.030
            and source > last_source
            and mp_stamp > last_mask
        )

    if not good:
        pending = None
        pending_stamps.clear()
        pending_since = None
        stats["RECOVERY_WAIT"] += 1
        continue

    if ident != pending:
        pending = ident
        pending_stamps.clear()
        pending_since = now

    pending_stamps.add(source)

    if len(pending_stamps) < 3 or now-pending_since < 0.20:
        stats["REVALIDATING"] += 1
        continue

    print("SHADOW REAUTHORIZED:", identity, "->", ident,
          flush=True)
    print("Position delta mm:", round(delta*1000, 2))
    print("Recovery gap seconds:", round(now-last_good, 3))

    identity = ident
    reauth_count += 1
    reauth_required = False
    hold_since = None
    pending = None
    pending_stamps.clear()
    pending_since = None

    last_pos = xyz(o.position)
    last_source = source
    last_mask = mp_stamp
    last_good = now

    stats["REAUTHORIZED"] += 1

print("\\n===== RECOVERY SUMMARY =====")
print("Initial identity:", initial_identity)
print("Final identity:", identity)
print("Reauthorizations:", reauth_count)
print("Stats:", dict(stats))

node.destroy_node()
rclpy.shutdown()
