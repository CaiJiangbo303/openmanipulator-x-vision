import rclpy
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose
)
from object_tracker.tracker_node import ObjectTrackerNode

BASE = (
    "cell phone|smartphone|mobile phone;"
    "toy brick|building brick;"
    "screwdriver|screw driver"
)

ALIASES = BASE + ";blue block|plastic cube"

class Capture:
    def __init__(self):
        self.msg = None

    def publish(self, msg):
        self.msg = msg

def frame(node, objects):
    msg = Detection2DArray()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = "synthetic_camera"

    for i, (cls, x, y) in enumerate(objects):
        d = Detection2D()
        d.header = msg.header
        d.id = f"det_{i}"

        d.bbox.center.position.x = float(x)
        d.bbox.center.position.y = float(y)
        d.bbox.size_x = 80.0
        d.bbox.size_y = 80.0

        h = ObjectHypothesisWithPose()
        h.hypothesis.class_id = cls
        h.hypothesis.score = 0.90
        d.results.append(h)

        msg.detections.append(d)

    node.pub.msg = None
    node.callback(msg)

    return {
        round(d.bbox.center.position.x): d.id
        for d in node.pub.msg.detections
    }

def repeat(node, objects, n=4):
    result = {}

    for _ in range(n):
        result = frame(node, objects)

    return result

def run(name, aliases, test):
    node = ObjectTrackerNode()
    node.class_alias_map = (
        node.parse_class_alias_groups(aliases)
    )

    # Capture output locally instead of publishing.
    node.pub = Capture()

    try:
        return test(node)
    finally:
        node.destroy_node()

def class_switch(node):
    initial = repeat(node, [("blue block", 200, 200)])
    old = initial.get(200)

    ids = []

    for i in range(24):
        cls = (
            "plastic cube"
            if (i // 4) % 2 == 0
            else "blue block"
        )

        x = 200 + i
        out = frame(node, [(cls, x, 200)])
        ids.append(out.get(x))

    passed = (
        old is not None
        and all(i == old for i in ids)
    )

    print("Initial:", old)
    print("Sequence:", ids)

    return passed

def two_objects(node):
    initial = repeat(node, [
        ("blue block", 150, 200),
        ("plastic cube", 450, 200)
    ])

    final = repeat(node, [
        ("plastic cube", 150, 200),
        ("blue block", 450, 200)
    ])

    print("Before:", initial)
    print("After: ", final)

    return (
        len(initial) == 2
        and len(set(initial.values())) == 2
        and initial == final
    )

def hand_isolation(node):
    initial = repeat(node, [
        ("blue block", 200, 200)
    ])

    old = initial.get(200)

    hand_ids = []

    for _ in range(4):
        out = frame(node, [("hand", 200, 200)])
        hand_ids.extend(out.values())

    print("Block ID:", old)
    print("Hand IDs:", hand_ids)

    return (
        old is not None
        and bool(hand_ids)
        and old not in hand_ids
    )

def short_gap(node):
    initial = repeat(node, [
        ("blue block", 200, 200)
    ])

    old = initial.get(200)

    empty = [frame(node, []) for _ in range(2)]

    final = frame(node, [
        ("plastic cube", 205, 200)
    ])

    print("Before:", old)
    print("After: ", final)

    return (
        old is not None
        and not any(empty)
        and final.get(205) == old
    )

def expired_track(node):
    initial = repeat(node, [
        ("blue block", 200, 200)
    ])

    old = initial.get(200)

    empty = [
        frame(node, [])
        for _ in range(node.max_missed + 1)
    ]

    final = repeat(node, [
        ("blue block", 200, 200)
    ])

    new = final.get(200)

    print("Expired ID:", old)
    print("New ID:", new)

    return (
        old is not None
        and new is not None
        and not any(empty)
        and old != new
    )

rclpy.init(args=[
    "--ros-args",
    "-p", "input_topic:=/synthetic/input",
    "-p", "output_topic:=/synthetic/output",
    "-p", "enable_native_mask:=false",
    "-p", "enable_occlusion_recovery:=false"
])

tests = [
    ("CLASS SWITCH", class_switch),
    ("TWO OBJECTS", two_objects),
    ("HAND ISOLATION", hand_isolation),
    ("SHORT GAP", short_gap),
    ("EXPIRED TRACK", expired_track)
]

results = {}

for name, test in tests:
    print(f"\n===== {name} =====")

    for group, aliases in [
        ("A", BASE),
        ("B", ALIASES)
    ]:
        print(f"\nTracker {group}")
        passed = run(name, aliases, test)
        results[(name, group)] = passed
        print("RESULT:", "PASS" if passed else "FAIL")

print("\n========== SUMMARY ==========")

for name, _ in tests:
    a = results[(name, "A")]
    b = results[(name, "B")]

    print(
        f"{name:20s} "
        f"A={'PASS' if a else 'FAIL':4s} "
        f"B={'PASS' if b else 'FAIL':4s}"
    )

rclpy.shutdown()
