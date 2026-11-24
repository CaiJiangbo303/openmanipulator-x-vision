import os

if (
    os.environ.get("ROS_DOMAIN_ID") != "199"
    or os.environ.get("ROS_LOCALHOST_ONLY") != "1"
):
    raise RuntimeError(
        "BENCH ONLY: requires ROS_DOMAIN_ID=199 "
        "and ROS_LOCALHOST_ONLY=1"
    )

import time
import json
import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.executors import SingleThreadedExecutor

from std_msgs.msg import String
from std_srvs.srv import Trigger

from vision_msgs.msg import (
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose
)

from object_state_manager.state_manager_node import (
    ObjectStateManager
)

from target_selector.state_gate_node import (
    StateGateNode
)

FRAME = "camera_color_optical_frame"
TRACK = "track_001"

rclpy.init(args=[
    "--ros-args",
    "-p", "filter_backend:=adaptive_cv",
    "-p", "enable_timestamp_guard:=true",
    "-p", "world_frame:=camera_color_optical_frame",
    "-p", "fast_measurement_topic:=/test/fast",
    "-p", "precise_measurement_topic:=/test/mask",
    "-p", "output_topic:=/test/states",
    "-p", "publish_rate_hz:=30.0"
])

manager = ObjectStateManager()

gate = StateGateNode(
    use_global_arguments=False,
    parameter_overrides=[
        Parameter("input_topic", value="/test/states"),
        Parameter("status_topic", value="/test/gate_status"),
        Parameter("requested_track_id", value=TRACK),
        Parameter("frame", value=FRAME)
    ]
)

driver = Node(
    "adaptive_gate_test_driver",
    use_global_arguments=False
)

executor = SingleThreadedExecutor()

for node in (manager, gate, driver):
    executor.add_node(node)

fast_pub = driver.create_publisher(
    Detection3DArray, "/test/fast", 10
)

mask_pub = driver.create_publisher(
    Detection3DArray, "/test/mask", 10
)

statuses = []

status_sub = driver.create_subscription(
    String,
    "/test/gate_status",
    lambda msg: statuses.append(json.loads(msg.data)),
    10
)

authorize = driver.create_client(
    Trigger, "/state_gate/authorize"
)

revoke = driver.create_client(
    Trigger, "/state_gate/revoke"
)

origin = np.array([0.03, -0.02, 0.20])
bias = np.array([0.0, 0.0, 0.003])

clock_start = time.monotonic_ns()

source_start = (
    driver.get_clock().now().nanoseconds
)

source_offset = 0
last_stamp = None
passed = 0


def pump(seconds):

    deadline = time.monotonic() + seconds

    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.002)


def check(name, condition):

    global passed

    if not condition:
        raise AssertionError(
            f"{name} FAILED; "
            f"latest status={statuses[-1:]}"
        )

    passed += 1

    print(
        f"[PASS {passed}] {name}",
        flush=True
    )


def make_message(position, stamp):

    msg = Detection3DArray()

    msg.header.frame_id = FRAME

    sec, ns = divmod(
        stamp, 1_000_000_000
    )

    msg.header.stamp.sec = sec
    msg.header.stamp.nanosec = ns

    d = Detection3D()
    d.id = TRACK

    result = ObjectHypothesisWithPose()

    result.hypothesis.class_id = "blue block"
    result.hypothesis.score = 0.95

    p = result.pose.pose.position

    p.x, p.y, p.z = map(float, position)

    d.results.append(result)
    msg.detections.append(d)

    return msg


def frame(position=origin):

    global last_stamp

    stamp = (
        source_start
        + time.monotonic_ns()
        - clock_start
        + source_offset
    )

    last_stamp = stamp

    fast_pub.publish(
        make_message(position, stamp)
    )

    pump(0.010)

    mask_pub.publish(
        make_message(position+bias, stamp)
    )

    pump(0.023)


def frames(n):

    for _ in range(n):
        frame()


def call(client):

    future = client.call_async(
        Trigger.Request()
    )

    deadline = time.monotonic() + 1.0

    while (
        not future.done()
        and time.monotonic() < deadline
    ):
        executor.spin_once(timeout_sec=0.002)

    if not future.done():
        raise RuntimeError("Service timeout")

    result = future.result()

    print(
        "SERVICE:",
        result.success,
        result.message,
        flush=True
    )

    pump(0.025)

    return result.success


def hold():

    return (
        bool(statuses)
        and statuses[-1]["decision"] == "HOLD"
        and gate.gate.locked_id is None
    )


def allowed():

    return (
        bool(statuses)
        and statuses[-1]["decision"] == "ALLOW"
        and gate.gate.locked_id == TRACK
    )


def current_state():

    return manager.tracks.get(TRACK)


try:

    pump(1.0)

    assert authorize.service_is_ready()
    assert revoke.service_is_ready()
    assert fast_pub.get_subscription_count() >= 1
    assert mask_pub.get_subscription_count() >= 1

    check(
        "Startup HOLD and authorization rejected",
        hold() and not call(authorize)
    )

    frames(25)

    state = current_state()

    check(
        "Adaptive filter recovered",
        state is not None
        and state.adaptive_filter is not None
        and manager.timestamp_guard.ready
    )

    old_generation = state.generation_id
    old_filter = state.adaptive_filter
    old_stamp = last_stamp

    check(
        "Explicit authorization succeeds",
        call(authorize) and allowed()
    )

    frames(5)

    check(
        "Normal stream maintains authorization",
        allowed()
    )

    # Simulated camera source-clock forward jump.
    source_offset += 6_000_000_000

    frame()

    check(
        "Six-second source jump revokes authorization",
        hold()
        and current_state() is None
    )

    frames(25)

    state = current_state()

    check(
        "Recovery creates new generation and filter",
        state is not None
        and state.generation_id != old_generation
        and state.adaptive_filter is not old_filter
    )

    check(
        "No automatic reauthorization",
        hold()
    )

    # Delayed precise observation from old lifecycle.
    before_x = state.adaptive_filter.x.copy()
    before_bias = state.precise_bias.copy()
    before_stamp = state.last_precise_stamp_ns

    mask_pub.publish(
        make_message(
            origin + np.array([0.02, 0, 0]),
            old_stamp
        )
    )

    pump(0.04)

    state = current_state()

    check(
        "Old Mask3D measurement rejected",
        state is not None
        and np.array_equal(
            state.adaptive_filter.x,
            before_x
        )
        and np.array_equal(
            state.precise_bias,
            before_bias
        )
        and state.last_precise_stamp_ns == before_stamp
    )

    frames(5)

    check(
        "New generation can be explicitly authorized",
        call(authorize) and allowed()
    )

    # Simulated backward jump.
    source_offset -= 7_000_000_000

    frame()

    check(
        "Backward source jump revokes authorization",
        hold()
        and current_state() is None
    )

    frames(25)

    check(
        "Recovered stream requires explicit authorization",
        current_state() is not None
        and hold()
        and call(authorize)
        and allowed()
    )

    # No more Fast3D or Mask3D measurements.
    pump(0.65)

    check(
        "Observation timeout forces HOLD",
        hold()
        and current_state() is None
    )

    frames(25)

    check(
        "Timeout recovery does not rearm gate",
        current_state() is not None
        and hold()
    )

    check(
        "Explicit authorization after timeout",
        call(authorize) and allowed()
    )

    check(
        "Manual revoke forces HOLD",
        call(revoke) and hold()
    )

    print("\n===== FINAL RESULT =====")
    print(f"Passed: {passed}/15")
    print("ADAPTIVE GATE E2E: PASS")

finally:

    for node in (driver, gate, manager):
        executor.remove_node(node)
        node.destroy_node()

    executor.shutdown()
    rclpy.try_shutdown()
