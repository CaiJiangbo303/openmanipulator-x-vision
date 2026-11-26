import time
import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.time import Time
from rclpy.action import ActionClient
from rclpy.signals import SignalHandlerOptions
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import JointState
from vision_msgs.msg import Detection3DArray
from object_state_msgs.msg import ObjectStateArray
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTrajectoryControllerState
from action_msgs.msg import GoalStatusArray
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from tf2_ros import Buffer, TransformListener

from manipulator_grasp.kinematics import Arm, LOW, HIGH, interpolate
from real_visual_tracker_final import quat_to_rot

from itertools import islice
from rclpy.logging import LoggingSeverity
from v2_input_guard import ReadyEnter, stamp_ns

rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)

node = Node("real_visual_tracker_fused_v2")
arm = Arm(tcp_offset=(0.0207, 0, 0))
buffer = Buffer()
listener = TransformListener(buffer, node)

client = ActionClient(
    node, FollowJointTrajectory,
    "/arm_controller/follow_joint_trajectory"
)

names = ["joint1", "joint2", "joint3", "joint4"]
data = {}
subs = []
identity = None
clock_base = node.get_clock().now().nanoseconds / 1e9
mono_base = time.monotonic()

def subscribe(key, cls, topic, qos=10):
    def cb(msg):
        data[key] = (msg, time.monotonic())
    subs.append(node.create_subscription(cls, topic, cb, qos))

subscribe("joint", JointState, "/joint_states")
subscribe("state", ObjectStateArray,
          node.declare_parameter("state_topic", "/bench/timestamp_guard/states").value)
subscribe("mask", Detection3DArray,
          node.declare_parameter("mask_topic", "/bench/real_self_filter/detections_3d").value)
subscribe("controller", JointTrajectoryControllerState,
          "/arm_controller/controller_state")

subscribe(
    "status", GoalStatusArray,
    "/arm_controller/follow_joint_trajectory/_action/status",
    QoSProfile(depth=10,
               durability=DurabilityPolicy.TRANSIENT_LOCAL)
)

def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)

def stamp(s):
    return s.sec + s.nanosec / 1e9

def xyz(p):
    return np.array([p.x, p.y, p.z], dtype=float)

def get(key, limit):
    require(key in data, "WAIT " + key)
    msg, received = data[key]
    require(time.monotonic()-received <= limit,
            key + " receipt stale")
    return msg

def observe():
    global identity

    now = node.get_clock().now().nanoseconds / 1e9

    require(abs((now-clock_base) -
                (time.monotonic()-mono_base)) < 0.05,
            "CLOCK JUMP")

    j = get("joint", 0.10)
    s = get("state", 0.15)
    m = get("mask", 0.15)

    require(0 <= now-stamp(j.header.stamp) <= 0.10,
            "JOINT TIMESTAMP: %.1f ms" % ((now-stamp(j.header.stamp))*1000))
    require(0 <= now-stamp(s.header.stamp) <= 0.10,
            "STATE TIMESTAMP")
    require(0 <= now-stamp(m.header.stamp) <= 0.15,
            "MASK TIMESTAMP: %.1f ms" % ((now-stamp(m.header.stamp))*1000))

    q = np.array([j.position[j.name.index(n)]
                  for n in names])
    v = np.array([j.velocity[j.name.index(n)]
                  for n in names])

    require(np.all(np.isfinite(q)) and
            np.all(np.isfinite(v)), "JOINT INVALID")

    frame = "camera_color_optical_frame"
    require(s.header.frame_id == m.header.frame_id == frame,
            "FRAME MISMATCH")

    candidates = [
        o for o in s.states
        if o.class_id == "blue block"
    ]

    if identity is not None:
        candidates = [
            o for o in candidates
            if (o.track_id, o.generation_id) == identity
        ]

    require(len(candidates) > 0, "NO TARGET OR LOCK LOST")

    tf = buffer.lookup_transform("world", frame, Time())
    r = tf.transform.rotation
    t = tf.transform.translation
    R = quat_to_rot([r.x, r.y, r.z, r.w])

    tcp = arm.fk(q)[:3, 3]

    state_threshold = 0.50 if identity is None else 0.45
    mask_threshold = 0.50 if identity is None else 0.40

    valid = []
    rejected = []

    for o in candidates:
        try:
            require(
                o.valid_3d and not o.stale
                and o.state == 1 and o.missed == 0,
                "TARGET INVALID"
            )

            require(
                o.confidence >= state_threshold
                and 0 <= o.measurement_age_ms <= 100
                and 0 <= now-stamp(o.source_stamp) <= 0.15,
                "VISION STALE"
            )

            require(
                np.linalg.norm(xyz(o.position_std)) <= 0.03,
                "TARGET UNCERTAINTY"
            )

            ident = (o.track_id, o.generation_id)
            require(all(ident), "IDENTITY INVALID")

            matches = []

            for d in m.detections:
                if d.id != o.track_id or not d.results:
                    continue

                res = d.results[0]

                if (
                    res.hypothesis.class_id == o.class_id
                    and res.hypothesis.score >= mask_threshold
                ):
                    matches.append(
                        xyz(res.pose.pose.position)
                    )

            same_id = [
                d for d in m.detections
                if d.id == o.track_id
            ]

            require(
                len(same_id) > 0,
                "MASK ID MISSING | available="
                + str([d.id for d in m.detections])
            )

            require(
                len(matches) == 1,
                "MASK MATCH FAILED | "
                + str([
                    (
                        d.id,
                        d.results[0].hypothesis.class_id,
                        round(d.results[0].hypothesis.score, 4)
                    )
                    for d in same_id if d.results
                ])
                + " | threshold=" + str(mask_threshold)
                + " | mask_age_ms="
                + str(round((now-stamp(m.header.stamp))*1000, 1))
                + " | source_delta_ms="
                + str(round(
                    (stamp(m.header.stamp)
                    -stamp(o.source_stamp))*1000, 1
                ))
            )

            pc = xyz(o.position)

            require(
                np.all(np.isfinite(pc))
                and np.all(np.isfinite(matches[0]))
                and np.linalg.norm(pc-matches[0]) <= 0.025,
                "MASK STATE MISMATCH"
            )

            require(0.12 <= pc[2] <= 0.48,
                    "DEPTH LIMIT")

            pw = R @ pc + xyz(t)

            require(
                np.all(np.isfinite(pw)),
                "TARGET INVALID"
            )

            require(
                0.20 <= pw[0] <= 0.395
                and abs(pw[1]) <= 0.16,
                "WORKSPACE LIMIT"
            )

            distance = float(
                np.linalg.norm(pw[:2]-tcp[:2])
            )

            require(distance <= 0.18,
                    "TARGET TOO FAR")

            valid.append((
                distance, -float(o.confidence),
                ident, pw
            ))

        except RuntimeError as e:
            rejected.append(f"{o.track_id}: {e}")

    require(
        len(valid) > 0,
        "NO VALID TARGET: " + "; ".join(rejected)
    )

    # 同一 Track ID 存在多个有效身份时拒绝歧义
    valid = [
        item for item in valid
        if sum(
            other[2][0] == item[2][0]
            for other in valid
        ) == 1
    ]

    require(len(valid) > 0, "AMBIGUOUS TRACK ID")

    # 多目标排序：距离优先，置信度次优先
    valid.sort(key=lambda x: (x[0], x[1], x[2]))

    distance, _, ident, target = valid[0]

    if identity is None:
        print("CANDIDATES:", len(candidates))
        print("VALID TARGETS:", len(valid))

        for dist, score, tid, pos in valid:
            print(
                "  ", tid[0],
                "distance_mm=", round(dist*1000, 1),
                "confidence=", round(-score, 3)
            )

        identity = ident
        print("LOCKED IDENTITY:", identity)

    require(identity == ident, "IDENTITY CHANGED")

    return q, v, target

def spin():
    rclpy.spin_once(node, timeout_sec=0.01)

def wait(f, seconds):
    end = time.monotonic()+seconds
    while not f.done() and time.monotonic() < end:
        spin()
    require(f.done(), "ACTION RESPONSE TIMEOUT")
    return f.result()


import sys

from real_visual_tracker_final import (
    CONTROL_HZ,
    HORIZON_SEC,
    MAX_CART_STEP,
    MAX_JOINT_STEP,
    XY_DEADBAND,
    IK_MARGIN,
    MAX_TARGET_JUMP,
    SAFE_X,
    SAFE_Y
)

EXECUTE = "--execute" in sys.argv

# 首轮融合真机测试限速
CART_CAP = min(MAX_CART_STEP, 0.010)
JOINT_CAP = min(MAX_JOINT_STEP, 0.020)


sent = None
handle = None
result = None
completed = True

owned_goals = set()

def inside_workspace(p):
    return (
        SAFE_X[0] <= p[0] <= SAFE_X[1]
        and SAFE_Y[0] <= p[1] <= SAFE_Y[1]
    )

def controller_ref(q):
    c = get("controller", 0.15)

    ref = np.array([
        c.output.positions[c.joint_names.index(n)]
        for n in names
    ])

    require(
        np.all(np.isfinite(ref)),
        "CONTROLLER REFERENCE INVALID"
    )

    require(
        np.max(np.abs(ref-q)) <= JOINT_CAP,
        "CONTROLLER TRACKING LAG"
    )

    return ref

def check_other_goals():
    require("status" in data, "WAIT ACTION STATUS")

    for item in data["status"][0].status_list:
        uid = bytes(item.goal_info.goal_id.uuid)

        require(
            item.status not in (1,2,3)
            or uid in owned_goals,
            "OTHER GOAL ACTIVE"
        )

def plan(q, target, fixed_z, fixed_pitch):

    tcp = arm.fk(q)[:3,3]

    require(inside_workspace(tcp),
            "TCP OUTSIDE WORKSPACE")

    require(
        np.all(q >= np.asarray(LOW)+IK_MARGIN)
        and np.all(q <= np.asarray(HIGH)-IK_MARGIN),
        "CURRENT JOINT LIMIT"
    )

    delta = target[:2]-tcp[:2]
    error = float(np.linalg.norm(delta))

    if error <= XY_DEADBAND:
        return None, error, tcp

    require(error <= 0.18, "TARGET TOO FAR")

    # 昨晚的自适应步长算法
    if error > 0.060:
        step = MAX_CART_STEP
    elif error > 0.030:
        step = min(MAX_CART_STEP, 0.026)
    elif error > 0.015:
        step = min(MAX_CART_STEP, 0.016)
    else:
        step = min(
            MAX_CART_STEP,
            max(0.005, error*0.80)
        )

    step = min(step, CART_CAP, error)

    desired = tcp.copy()
    desired[:2] += delta/error*step
    desired[2] = fixed_z

    require(inside_workspace(desired),
            "WORKSPACE LIMIT")

    solutions = arm.ik(
        desired,
        fixed_pitch,
        margin=IK_MARGIN
    )

    require(len(solutions)>0, "IK UNAVAILABLE")

    q_goal = min(
        solutions,
        key=lambda x: np.linalg.norm(x-q)
    )

    require(
        np.max(np.abs(q_goal-q)) <= 0.40,
        "IK BRANCH JUMP"
    )

    path = interpolate(q,q_goal,JOINT_CAP)

    require(len(path)>=2, "EMPTY PATH")

    q_cmd = np.asarray(path[1],dtype=float)
    ref = controller_ref(q)

    require(
        np.all(np.isfinite(q_cmd))
        and np.max(np.abs(q_cmd-q)) <= JOINT_CAP+1e-8
        and np.max(np.abs(q_cmd-ref)) <= JOINT_CAP+1e-8,
        "JOINT STEP LIMIT"
    )

    for origin in (q,ref):
        for u in np.linspace(0,1,11):
            qi = origin+u*(q_cmd-origin)
            p = arm.fk(qi)[:3,3]

            require(
                np.all(qi >= np.asarray(LOW)+IK_MARGIN)
                and np.all(qi <= np.asarray(HIGH)-IK_MARGIN),
                "PATH JOINT LIMIT"
            )

            require(
                inside_workspace(p)
                and abs(p[2]-fixed_z)<=0.005,
                "PATH WORKSPACE LIMIT"
            )

    return q_cmd,error,tcp



import select
from control_msgs.msg import DynamicJointState

subscribe("hardware", DynamicJointState,
          "/dynamic_joint_states")

# Continuous tracking, conservative first-test horizon
TRACK_HORIZON = 0.30

q_sent = None
q_start = None
settle_deadline = None
stall_count = 0

armed = False
enter_ticket = ReadyEnter(1.0 / CONTROL_HZ)
ready_since = None
last_target = None
last_print = 0.0
next_control = 0.0
commands = 0


def report(*args):
    global last_print

    now = time.monotonic()

    if now-last_print >= 0.8:
        print(*args)
        last_print = now


def feedback():

    now = node.get_clock().now().nanoseconds/1e9

    require(
        abs((now-clock_base) -
            (time.monotonic()-mono_base)) < 0.05,
        "CLOCK JUMP"
    )

    h = get("hardware", 0.15)

    require(
        0 <= now-stamp(h.header.stamp) <= 0.15,
        "HARDWARE TIMESTAMP"
    )

    for motor in ("dxl1", "dxl2", "dxl3", "dxl4", "dxl5"):

        require(
            motor in h.joint_names,
            "HARDWARE MISSING: " + motor
        )

        item = h.interface_values[
            h.joint_names.index(motor)
        ]

        values = dict(zip(
            item.interface_names,
            item.values
        ))

        for key in (
            "Torque Enable",
            "Present Position",
            "Present Velocity",
            "Present Current",
            "Present Input Voltage"
        ):
            require(
                key in values
                and np.isfinite(values[key]),
                "HARDWARE INVALID: " + motor + " " + key
            )

        require(
            values["Torque Enable"] == 1.0,
            "TORQUE LOST: " + motor
        )

    j = get("joint", 0.10)

    require(
        0 <= now-stamp(j.header.stamp) <= 0.10,
        "JOINT TIMESTAMP: %.1f ms" % ((now-stamp(j.header.stamp))*1000)
    )

    q = np.array([
        j.position[j.name.index(n)]
        for n in names
    ])

    v = np.array([
        j.velocity[j.name.index(n)]
        for n in names
    ])

    require(
        np.all(np.isfinite(q))
        and np.all(np.isfinite(v)),
        "JOINT INVALID"
    )

    require(
        np.all(q >= np.asarray(LOW)+IK_MARGIN)
        and np.all(q <= np.asarray(HIGH)-IK_MARGIN),
        "CURRENT JOINT LIMIT"
    )

    return q,v


def cancel_current():

    global sent,handle,result

    if sent is None:
        return

    if handle is None:
        handle = wait(sent,2)

    require(handle.accepted,"GOAL REJECTED")

    owned_goals.add(bytes(handle.goal_id.uuid))

    if result is None:
        result = handle.get_result_async()

    if not result.done():

        print("REQUESTING CANCEL")

        response = wait(
            handle.cancel_goal_async(),2
        )

        print("CANCEL CODE:",response.return_code)

    r = wait(result,3)

    require(
        r.status in (4,5,6),
        "ACTION STILL ACTIVE"
    )

    print("FINAL ACTION STATUS:",r.status)

    sent = None
    handle = None
    result = None

    require(r.status != 6,"ACTION ABORTED")


def verify_stop():

    stationary_since = None
    deadline = time.monotonic()+3

    while time.monotonic()<deadline:

        spin()

        q,v = feedback()

        check_other_goals()

        if np.max(np.abs(v)) <= 0.01:

            if stationary_since is None:
                stationary_since = time.monotonic()

            if time.monotonic()-stationary_since >= 0.5:

                print("STOP VERIFIED")
                return

        else:
            stationary_since = None

    raise RuntimeError("STOP NOT VERIFIED")


def input_diagnostics():
    inputs = {}
    for key in ("joint", "state", "mask"):
        if key in data:
            msg, receipt = data[key]
            inputs[key] = (stamp_ns(msg.header.stamp), receipt)
    if "state" in data:
        inputs["state_source_ns"] = list(islice((
            (o.track_id, o.generation_id, stamp_ns(o.source_stamp))
            for o in data["state"][0].states if o.class_id == "blue block"
        ), 128))
    return inputs


def pause_tracking(reason):

    global armed,identity,ready_since
    global last_target,settle_deadline
    global next_control

    enter_ticket.clear()
    if node.get_logger().is_enabled_for(LoggingSeverity.DEBUG):
        node.get_logger().debug(f"V2 HOLD {reason} identity={identity} "
                                f"inputs={input_diagnostics()} now_mono={time.monotonic():.9f}")
    if armed and EXECUTE:

        print("TRACKING PAUSED:",reason)

        cancel_current()
        verify_stop()

        print("WAITING FOR VALID TARGET")

    else:
        report("HOLD:",reason)

    armed = False
    identity = None
    ready_since = None
    last_target = None
    settle_deadline = None
    next_control = 0.0


def recoverable(reason):

    return reason.startswith((
        "NO VALID TARGET",
        "NO TARGET OR LOCK LOST",
        "AMBIGUOUS TRACK ID",
        "MASK TIMESTAMP",
        "STATE TIMESTAMP",
        "mask receipt stale",
        "state receipt stale",
        "WAIT state",
        "WAIT mask",
        "TARGET JUMP",
        "TARGET TOO FAR",
        "TARGET CHANGED BEFORE SEND",
        "JOINT CHANGED BEFORE SEND",
        "IK UNAVAILABLE",
        "IK BRANCH JUMP",
        "WORKSPACE LIMIT",
        "PATH WORKSPACE LIMIT",
        "PATH JOINT LIMIT",
        "EMPTY PATH"
    ))


def enter_pressed():

    if not EXECUTE:
        return False

    if select.select([sys.stdin],[],[],0)[0]:
        return sys.stdin.readline() == "\n"

    return False


try:

    print("==================================")
    print("FUSED CONTINUOUS TRACKING V2")
    print("Vision: State Manager V2 + Self Filter")
    print("Planning rate:",CONTROL_HZ,"Hz")
    print("Horizon:",TRACK_HORIZON,"s")
    print("Cart cap:",CART_CAP,"m")
    print("Joint cap:",JOINT_CAP,"rad")
    print("Session limit: NONE")
    print("Torque monitoring: ON")
    print("EXECUTE:",EXECUTE)
    print("==================================")

    require(
        client.wait_for_server(timeout_sec=3),
        "ACTION SERVER UNAVAILABLE"
    )

    startup_ready = None

    while rclpy.ok():

        spin()

        try:

            q,v = feedback()

            require(
                np.max(np.abs(v))<=0.01,
                "ARM NOT STATIONARY"
            )

            check_other_goals()
            controller_ref(q)

            if startup_ready is None:
                startup_ready = time.monotonic()

            if time.monotonic()-startup_ready >= 0.25:
                print("HARDWARE READY")
                break

        except RuntimeError as e:

            startup_ready = None
            reason = str(e)

            if not reason.startswith((
                "WAIT ",
                "JOINT TIMESTAMP",
                "HARDWARE TIMESTAMP",
                "joint receipt stale",
                "hardware receipt stale",
                "controller receipt stale",
                "ARM NOT STATIONARY"
            )):
                raise

            report("WAITING FOR FEEDBACK:",reason)

    tcp = arm.fk(q)[:3,3]

    fixed_z = float(tcp[2])
    fixed_pitch = float(np.sum(q[1:]))

    print("START TCP:",tcp)
    print("LOCKED Z:",fixed_z)
    print("LOCKED PITCH:",fixed_pitch)

    # Continuous loop: no 30-second session limit.
    while rclpy.ok():

        spin()

        now = time.monotonic()

        if enter_pressed() and not armed:
            enter_ticket.capture(identity, ready_since, now)

        # Independent hardware guard runs even
        # when no visual target is available.
        try:
            q,v = feedback()
            controller_ref(q)

        except RuntimeError as e:

            if not EXECUTE and str(e).startswith((
                "JOINT TIMESTAMP",
                "HARDWARE TIMESTAMP",
                "joint receipt stale",
                "hardware receipt stale",
                "controller receipt stale"
            )):
                report("WAITING FOR FEEDBACK:",e)
                continue

            raise

        # Monitor the currently submitted trajectory.
        if sent is not None:

            require(
                np.max(np.abs(q-q_start))<=0.035,
                "UNEXPECTED JOINT MOVEMENT"
            )

            if handle is None and sent.done():

                handle = sent.result()

                require(
                    handle.accepted,
                    "GOAL REJECTED"
                )

                owned_goals.add(
                    bytes(handle.goal_id.uuid)
                )

                result = handle.get_result_async()

            if result is not None and result.done():

                r = result.result()

                require(
                    r.status == 4
                    and r.result.error_code == 0,
                    "ACTION FAILED"
                )

                position_error = float(
                    np.max(np.abs(q-q_sent))
                )

                if settle_deadline is None:
                    settle_deadline = now+0.4

                if position_error <= 0.008:

                    planned = np.max(
                        np.abs(q_sent-q_start)
                    )

                    actual = np.max(
                        np.abs(q-q_start)
                    )

                    if planned >= 0.008 and actual < 0.002:
                        stall_count += 1
                    else:
                        stall_count = 0

                    require(
                        stall_count < 3,
                        "REPEATED MOTION STALL"
                    )

                    sent = None
                    handle = None
                    result = None
                    settle_deadline = None

                else:
                    require(
                        now<settle_deadline,
                        "POSITION NOT REACHED"
                    )

            else:
                require(
                    now<deadline,
                    "ACTION TIMEOUT"
                )

        if sent is None or handle is not None:
            check_other_goals()

        try:

            # Updated vision pipeline.
            q,v,target = observe()

            if last_target is not None:

                require(
                    np.linalg.norm(
                        target-last_target
                    )<=MAX_TARGET_JUMP,
                    "TARGET JUMP"
                )

            last_target = target.copy()

            if now < next_control:
                continue

            next_control = now+1.0/CONTROL_HZ

            # One active trajectory at a time.
            if sent is not None:
                continue

            q_cmd,error,tcp = plan(
                q,target,fixed_z,fixed_pitch
            )

            # Dry run remains active indefinitely.
            if not EXECUTE:

                report(
                    "DRY TRACK",
                    identity[0],
                    "error_mm=",round(error*1000,1),
                    "target=",np.round(target,3)
                )

                continue

            # Initial authorization and recovery.
            if not armed:

                require(
                    np.max(np.abs(v))<=0.01,
                    "ARM NOT STATIONARY"
                )

                if ready_since is None:

                    ready_since = now

                    print(
                        "TARGET READY:",
                        identity,
                        "press Enter to enable tracking"
                    )

                if now-ready_since < 0.15:
                    continue

                if not enter_ticket.consume(identity, ready_since, time.monotonic()):
                    continue

                armed = True

                print("CONTINUOUS TRACKING ENABLED")

            # Inside deadband: hold and keep tracking.
            if q_cmd is None:

                report(
                    "ALIGNED:",
                    round(error*1000,1),"mm"
                )

                continue

            q_new,v_new,target_new = observe()

            require(
                np.max(np.abs(q_new-q))<=0.003,
                "JOINT CHANGED BEFORE SEND"
            )

            require(
                np.linalg.norm(target_new-target)<=0.010,
                "TARGET CHANGED BEFORE SEND"
            )

            feedback()
            controller_ref(q_new)
            check_other_goals()

            goal = FollowJointTrajectory.Goal()
            goal.trajectory.joint_names = names

            point = JointTrajectoryPoint()

            point.positions = q_cmd.tolist()
            point.velocities = [0.0]*4

            point.time_from_start = Duration(
                sec=int(TRACK_HORIZON),
                nanosec=int(
                    (TRACK_HORIZON-int(TRACK_HORIZON))*1e9
                )
            )

            goal.trajectory.points = [point]

            q_start = q_new.copy()
            q_sent = q_cmd.copy()

            handle = None
            result = None
            settle_deadline = None

            deadline = time.monotonic()+1.5

            sent = client.send_goal_async(goal)

            commands += 1

            report(
                "TRACK",
                identity[0],
                "error_mm=",round(error*1000,1),
                "TCP=",np.round(tcp,3),
                "OBJ=",np.round(target,3),
                "commands=",commands
            )

        except RuntimeError as e:

            reason = str(e)

            if not recoverable(reason) and not (
                not EXECUTE and reason.startswith((
                    "JOINT TIMESTAMP",
                    "joint receipt stale"
                ))
            ):
                raise

            pause_tracking(reason)

except KeyboardInterrupt:

    print("USER STOP")

except Exception as e:

    print("FAULT LATCHED:",e)
    print("AUTOMATIC MOTION DISABLED")

finally:

    try:

        if EXECUTE:

            cancel_current()
            verify_stop()

    except Exception as e:

        print("STOP NOT VERIFIED:",e)
        print("USE INDEPENDENT EMERGENCY STOP IF NEEDED")

    print("TRACKING SESSION CLOSED")
    print("NO AUTOMATIC RESTART")

    node.destroy_node()
    rclpy.try_shutdown()
