import ast
import sys
import numpy as np
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path("run_nodes").resolve()))

from semantic_tracking.recovery_manager import (
    Observation,
    Status,
    TrackingRecoveryManager
)

clock = SimpleNamespace(t=0.0)
time = SimpleNamespace(monotonic=lambda: clock.t)

class ShadowAction:
    pass

def control_cb(self):
    self.plans += 1

original = SimpleNamespace(
    MAX_TARGET_JUMP=0.08,
    FastTracker=SimpleNamespace(control_cb=control_cb)
)

TRACK_CLASS = "blue block"
TRACK_CLASSES = {"blue block", "plastic cube"}
ACQUIRE_CONF = 0.50
STATE_MAX_AGE = 0.20

path = Path("run_nodes/real_visual_tracker_v3_recovery.py")
tree = ast.parse(path.read_text())

cls = next(
    n for n in tree.body
    if isinstance(n, ast.ClassDef)
    and n.name == "TrackerV3"
)

methods = [
    n for n in cls.body
    if isinstance(n, ast.FunctionDef)
    and n.name in {"hold", "tick", "raw_input"}
]

assert len(methods) == 3

# Execute the actual controller methods without ROS or hardware.
module = ast.Module(body=methods, type_ignores=[])

exec(compile(module, str(path), "exec"), globals())

class Harness:
    tick = tick
    hold = hold
    raw_input = raw_input

    def __init__(self):
        clock.t = 0.0
        self.recovery_manager = None
        self.identity = None
        self.previous_status = None
        self.target_world = None
        self.target_seen = None
        self.target_id = None
        self.target_class = None
        self.target_score = 0.0
        self.raw = {}
        self.q = np.zeros(4)
        self.joint_time = None
        self.action = ShadowAction()
        self.plans = 0
        self.candidates = []

    def status(self, message):
        pass

    def ros_now(self):
        return 100.0 + clock.t

    def get_candidates(self):
        return self.candidates, "TEST EVIDENCE"

    def run(self, t, track="A", generation="G1",
            visible=True, joints=True):

        clock.t = t
        self.joint_time = t if joints else None

        source = 100.0 + t - 0.02

        self.candidates = [(
            0.10,
            -0.80,
            (track, generation),
            np.array([0.30, 0.0, 0.20]),
            0.02,
            source,
            source
        )] if visible else []

        before = self.plans
        self.tick()

        return self.plans - before

def check(name, condition):
    assert condition, name
    print("PASS:", name)

h = Harness()

check(
    "No joints: no planning",
    h.run(0.0, joints=False) == 0
)

check(
    "Initial acquisition",
    h.run(0.1) == 1
)

check(
    "Target loss blocks planning",
    h.run(0.2, visible=False) == 0
    and not h.recovery_manager.authorized
)

h.run(0.5, visible=False)

r1 = h.run(0.8, generation="G2")
r2 = h.run(0.92, generation="G2")

check(
    "Generation validation blocks planning",
    r1 == 0 and r2 == 0
)

check(
    "Validated recovery resumes planning",
    h.run(1.04, generation="G2") == 1
    and h.identity == ("A", "G2")
)

check(
    "Wrong track cannot replace target",
    h.run(1.2, track="B") == 0
    and h.identity == ("A", "G2")
)

check(
    "Timeout expires authorization",
    h.run(4.2, visible=False) == 0
    and h.recovery_manager.status == Status.EXPIRED
)

check(
    "Expired session cannot restart",
    h.run(4.3, generation="G2") == 0
)

h = Harness()
h.run(0.1)

detection = SimpleNamespace(
    results=[
        SimpleNamespace(
            hypothesis=SimpleNamespace(
                class_id="blue block"
            )
        )
    ]
)

h.raw_input(
    "DET",
    SimpleNamespace(detections=[detection, detection])
)

check(
    "Raw ambiguity immediately expires session",
    h.recovery_manager.status == Status.EXPIRED
    and h.run(0.2) == 0
)

print("\n===== V3 CONTROL TEST SUMMARY =====")
print("9/9 PASS")
print("No ROS action client or hardware motion used")
