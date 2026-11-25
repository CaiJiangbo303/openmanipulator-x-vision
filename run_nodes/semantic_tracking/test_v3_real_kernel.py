import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path("run_nodes").resolve()))

import real_visual_tracker_v3_recovery as v3

original = v3.original

assert original.MAX_CART_STEP == 0.010
assert original.MAX_JOINT_STEP == 0.020
assert original.TRACK_Z is False

Q0 = np.array([
    -0.16,
    0.48,
    -0.16,
    0.04
], dtype=float)


class Harness:

    def __init__(self):

        self.arm = v3.Arm(
            tcp_offset=original.TCP_OFFSET
        )

        self.q = Q0.copy()

        tcp = self.arm.fk(self.q)[:3, 3]

        self.fixed_z = float(tcp[2])
        self.fixed_pitch = float(np.sum(self.q[1:]))

        self.target_world = None
        self.target_seen = None
        self.target_id = "offline_test"
        self.target_class = "blue block"
        self.target_score = 0.90

        self.goal_pending = False
        self.command_count = 0
        self.planned_count = 0

        self.last_reason = ""
        self.commands = []

        self.action = v3.ShadowAction(self)

        shadow_send = self.action.send_goal_async

        def capture(goal):

            point = goal.trajectory.points[0]

            q_cmd = np.asarray(
                point.positions,
                dtype=float
            )

            assert q_cmd.shape == (4,)
            assert np.all(np.isfinite(q_cmd))

            delta = np.max(
                np.abs(q_cmd-self.q)
            )

            assert delta <= (
                original.MAX_JOINT_STEP + 1e-8
            ), "JOINT STEP EXCEEDED"

            assert np.all(
                q_cmd >= original.LOW
                + original.IK_MARGIN
            )

            assert np.all(
                q_cmd <= original.HIGH
                - original.IK_MARGIN
            )

            self.commands.append(q_cmd.copy())

            return shadow_send(goal)

        self.action.send_goal_async = capture

    def status(self, msg):
        self.last_reason = msg

    def goal_response_cb(self, future):
        original.FastTracker.goal_response_cb(
            self, future
        )

    def tcp(self):
        return self.arm.fk(self.q)[:3, 3]

    def step(self):

        before = len(self.commands)

        original.FastTracker.control_cb(self)

        if len(self.commands) == before:
            return False

        return True


def check(name, condition):
    assert condition, name
    print("PASS:", name)


print("===== REAL CONTROL KERNEL TEST =====")

h = Harness()

print("Initial joints:", h.q)
print("Initial TCP:", np.round(h.tcp(), 5))

check(
    "Initial joint limits",
    np.all(h.q >= original.LOW + original.IK_MARGIN)
    and
    np.all(h.q <= original.HIGH - original.IK_MARGIN)
)

check(
    "Missing target blocks planning",
    not h.step()
)

h.target_world = h.tcp().copy()
h.target_world[0] -= 0.025
h.target_seen = time.monotonic() - 1.0

check(
    "Stale target blocks planning",
    not h.step()
)

h.target_world = h.tcp().copy()
h.target_seen = time.monotonic()

check(
    "Aligned target blocks unnecessary planning",
    not h.step()
)

print("\n===== CLOSED LOOP IDEAL FEEDBACK =====")

cases = [
    ("X_NEG", -0.025, 0.0),
    ("X_POS",  0.025, 0.0),
    ("Y_NEG",  0.0, -0.025),
    ("Y_POS",  0.0,  0.025),
    ("DIAGONAL", -0.025, 0.025)
]

passed = 0

for name, dx, dy in cases:

    h = Harness()

    start = h.tcp().copy()

    target = start + np.array([dx, dy, 0.0])

    h.target_world = target

    initial_error = np.linalg.norm(
        target[:2]-start[:2]
    )

    errors = []
    max_joint_step = 0.0
    max_cart_step = 0.0
    max_z_error = 0.0
    max_pitch_error = 0.0

    print("\nCASE:", name)

    for i in range(80):

        before_tcp = h.tcp().copy()

        error = np.linalg.norm(
            target[:2]-before_tcp[:2]
        )

        errors.append(float(error))

        if error < original.XY_DEADBAND:
            break

        h.target_seen = time.monotonic()

        if not h.step():
            print("Planning stopped:", h.last_reason)
            break

        q_cmd = h.commands[-1]

        max_joint_step = max(
            max_joint_step,
            np.max(np.abs(q_cmd-h.q))
        )

        # Ideal simulated feedback only.
        h.q = q_cmd.copy()

        after_tcp = h.tcp()

        max_cart_step = max(
            max_cart_step,
            np.linalg.norm(after_tcp-before_tcp)
        )

        max_z_error = max(
            max_z_error,
            abs(after_tcp[2]-h.fixed_z)
        )

        max_pitch_error = max(
            max_pitch_error,
            abs(np.sum(h.q[1:])-h.fixed_pitch)
        )

    final_error = np.linalg.norm(
        target[:2]-h.tcp()[:2]
    )

    ok = final_error < original.XY_DEADBAND

    passed += int(ok)

    print("Initial error mm:", round(initial_error*1000, 3))
    print("Final error mm:", round(final_error*1000, 3))
    print("Plans:", len(h.commands))
    print("Max joint step rad:", round(max_joint_step, 6))
    print("Max TCP step mm:", round(max_cart_step*1000, 3))
    print("Max Z deviation mm:", round(max_z_error*1000, 3))
    print("Max pitch deviation rad:", round(max_pitch_error, 6))
    print("Result:", "PASS" if ok else "FAIL")

print("\n===== FINAL SUMMARY =====")
print("Convergence:", f"{passed}/{len(cases)}")
print("No ROS node or real action client created")
print("No hardware motion")
