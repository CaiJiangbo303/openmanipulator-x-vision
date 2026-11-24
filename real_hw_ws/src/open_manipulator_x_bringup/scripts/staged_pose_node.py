#!/usr/bin/env python3
"""Run a collision-aware two-stage OpenMANIPULATOR-X pose.

The controller requires complete joint trajectories, so the first point keeps
all non-base joints at their measured positions while joint1 moves.  A second
point is sent after a configurable pause and moves the remaining joints to the
requested zero or home pose.  The node is passive unless a service is called
or ``startup_pose`` is explicitly set to ``zero``/``home``.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Optional

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectoryPoint


ZERO = [math.pi - (2.0 * math.pi / 4096.0), 0.0, 0.0, 0.0]
HOME = [math.pi - (2.0 * math.pi / 4096.0), -1.0, 0.7, 0.3]


def duration(seconds: float) -> Duration:
    whole = int(seconds)
    return Duration(sec=whole, nanosec=int(round((seconds - whole) * 1e9)))


def build_staged_trajectory(
    joint_names: list[str], current: list[float], target: list[float],
    joint1_duration: float, delay: float, remainder_duration: float,
) -> FollowJointTrajectory.Goal:
    """Build a trajectory with joint1 first, a hold, then the remaining joints."""
    if len(joint_names) != 4 or len(current) != 4 or len(target) != 4:
        raise ValueError('staged pose requires four arm joints')
    if min(joint1_duration, delay, remainder_duration) <= 0:
        raise ValueError('pose durations must be positive')
    if not all(math.isfinite(value) for value in [*current, *target]):
        raise ValueError('pose positions must be finite')
    trajectory = FollowJointTrajectory.Goal()
    trajectory.trajectory.joint_names = list(joint_names)
    first = JointTrajectoryPoint()
    first.positions = [target[0], *current[1:]]
    first.time_from_start = duration(joint1_duration)
    hold = JointTrajectoryPoint()
    hold.positions = list(first.positions)
    hold.time_from_start = duration(joint1_duration + delay)
    final = JointTrajectoryPoint()
    final.positions = list(target)
    final.time_from_start = duration(joint1_duration + delay + remainder_duration)
    trajectory.trajectory.points = [first, hold, final]
    return trajectory


class StagedPoseNode(Node):
    def __init__(self) -> None:
        super().__init__('arm_pose_sequence')
        self.declare_parameter('controller_action', '/arm_controller/follow_joint_trajectory')
        self.declare_parameter('joint_state_topic', '/joint_states')
        self.declare_parameter('joint_names', ['joint1', 'joint2', 'joint3', 'joint4'])
        self.declare_parameter('startup_pose', 'none')
        self.declare_parameter('joint1_delay_sec', 1.0)
        self.declare_parameter('joint1_duration_sec', 2.0)
        self.declare_parameter('remainder_duration_sec', 4.0)
        self.declare_parameter('state_max_age_sec', 0.5)

        self.joint_names = [str(v) for v in self.get_parameter('joint_names').value]
        if len(self.joint_names) != 4:
            raise ValueError('joint_names must contain joint1 through joint4')
        self.action = ActionClient(
            self, FollowJointTrajectory,
            str(self.get_parameter('controller_action').value),
        )
        self.state_lock = threading.Lock()
        self.latest_state: Optional[JointState] = None
        self.latest_state_wall = 0.0
        self.busy_lock = threading.Lock()
        self.busy = False
        self.status_pub = self.create_publisher(String, '~/status', 10)
        self.create_subscription(
            JointState, str(self.get_parameter('joint_state_topic').value),
            self._joint_state, 50,
        )
        self.create_service(Trigger, '~/zero', self._zero)
        self.create_service(Trigger, '~/home', self._home)
        self.startup_pose = str(self.get_parameter('startup_pose').value).lower()
        self.startup_timer = self.create_timer(0.5, self._maybe_startup)
        self.startup_started = False

    def _joint_state(self, message: JointState) -> None:
        with self.state_lock:
            self.latest_state = message
            self.latest_state_wall = time.monotonic()

    def _publish_status(self, value: str) -> None:
        message = String()
        message.data = value
        self.status_pub.publish(message)
        self.get_logger().info(value)

    def _maybe_startup(self) -> None:
        if self.startup_started:
            return
        if self.startup_pose not in ('zero', 'home'):
            self.startup_timer.cancel()
            return
        if not self._has_fresh_state():
            return
        self.startup_started = True
        self.startup_timer.cancel()
        self._accept_pose(self.startup_pose)

    def _has_fresh_state(self) -> bool:
        with self.state_lock:
            return self.latest_state is not None and (
                time.monotonic() - self.latest_state_wall
                <= float(self.get_parameter('state_max_age_sec').value)
            )

    def _zero(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        return self._service_pose('zero', response)

    def _home(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        return self._service_pose('home', response)

    def _service_pose(self, pose: str, response: Trigger.Response) -> Trigger.Response:
        if not self._has_fresh_state():
            response.message = '没有收到新鲜的 /joint_states，拒绝发送姿态命令'
            return response
        if not self.busy_lock.acquire(blocking=False):
            response.message = '已有 zero/home 动作进行中'
            return response
        self.busy = True
        threading.Thread(target=self._run_pose, args=(pose,), daemon=True).start()
        response.success = True
        response.message = f'{pose} 已接受，先移动 joint1，随后等待并移动其余关节'
        return response

    def _accept_pose(self, pose: str) -> None:
        if not self.busy_lock.acquire(blocking=False):
            self._publish_status('rejected: another pose is active')
            return
        self.busy = True
        threading.Thread(target=self._run_pose, args=(pose,), daemon=True).start()

    def _current_positions(self) -> list[float]:
        with self.state_lock:
            state = self.latest_state
            state_wall = self.latest_state_wall
        if state is None or time.monotonic() - state_wall > float(self.get_parameter('state_max_age_sec').value):
            raise RuntimeError('joint_states is stale')
        values = dict(zip(state.name, state.position))
        missing = [name for name in self.joint_names if name not in values]
        if missing:
            raise RuntimeError(f'joint_states 缺少关节: {missing}')
        current = [float(values[name]) for name in self.joint_names]
        if not all(math.isfinite(value) for value in current):
            raise RuntimeError('joint_states contains non-finite positions')
        return current

    def _run_pose(self, pose: str) -> None:
        try:
            target = ZERO if pose == 'zero' else HOME
            current = self._current_positions()
            joint1_duration = float(self.get_parameter('joint1_duration_sec').value)
            delay = float(self.get_parameter('joint1_delay_sec').value)
            remainder_duration = float(self.get_parameter('remainder_duration_sec').value)
            trajectory = build_staged_trajectory(
                self.joint_names, current, target,
                joint1_duration, delay, remainder_duration,
            )

            self._publish_status(f'{pose}: waiting for controller action')
            if not self.action.wait_for_server(timeout_sec=10.0):
                raise RuntimeError('arm_controller/follow_joint_trajectory action unavailable')
            self._publish_status(f'{pose}: moving joint1 first')
            goal_future = self.action.send_goal_async(trajectory)
            goal_event = threading.Event()
            goal_holder = {}

            def goal_done(future) -> None:
                try:
                    goal_holder['goal'] = future.result()
                except Exception as error:  # pragma: no cover - middleware failure
                    goal_holder['error'] = error
                goal_event.set()

            goal_future.add_done_callback(goal_done)
            if not goal_event.wait(timeout=15.0):
                raise RuntimeError('timed out waiting for trajectory goal acceptance')
            if 'error' in goal_holder:
                raise RuntimeError(str(goal_holder['error']))
            goal_handle = goal_holder.get('goal')
            if goal_handle is None or not goal_handle.accepted:
                raise RuntimeError('trajectory goal was rejected')
            time.sleep(joint1_duration)
            self._publish_status(f'{pose}: waiting {delay:.1f}s before remaining joints')
            time.sleep(delay)
            self._publish_status(f'{pose}: moving joints 2-4')
            result_future = goal_handle.get_result_async()
            result_event = threading.Event()
            result_holder = {}

            def result_done(future) -> None:
                try:
                    result_holder['result'] = future.result()
                except Exception as error:  # pragma: no cover - middleware failure
                    result_holder['error'] = error
                result_event.set()

            result_future.add_done_callback(result_done)
            if not result_event.wait(timeout=joint1_duration + delay + remainder_duration + 15.0):
                raise RuntimeError('timed out waiting for staged pose result')
            if 'error' in result_holder:
                raise RuntimeError(str(result_holder['error']))
            result = result_holder.get('result')
            if result is None or result.result.error_code != 0:
                code = 'unknown' if result is None else str(result.result.error_code)
                raise RuntimeError(f'trajectory failed with error code {code}')
            self._publish_status(f'{pose}: completed')
        except Exception as error:
            self._publish_status(f'{pose}: failed: {error}')
        finally:
            self.busy = False
            self.busy_lock.release()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = StagedPoseNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
