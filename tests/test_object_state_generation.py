import uuid
from types import SimpleNamespace

import numpy as np
import rclpy

from rclpy.time import Time

from object_state_msgs.msg import ObjectState

from object_state_manager.state_manager_node import (
    ObjectStateManager,
)


def main():

    rclpy.init(args=[])

    manager = ObjectStateManager()

    captured = []

    original_pub = manager.state_pub

    manager.state_pub = SimpleNamespace(
        publish=captured.append
    )

    def create(node):

        now = node.get_clock().now()

        return node.create_track(
            'track_001',
            'cell phone',
            np.array([0.02, 0.0, 0.30]),
            0.95,
            now,
            now,
            'fast_bbox_depth',
            ObjectState.SOURCE_FAST_BBOX_DEPTH,
        )

    def snapshot():

        manager.publish_states()

        return captured[-1]

    try:

        state = create(manager)

        generation_a = state.generation_id

        assert uuid.UUID(generation_a).version == 4

        print('UUID generation: PASS')
        print('Original generation:', generation_a)

        msg = snapshot()

        assert len(msg.states) == 1

        assert (
            msg.states[0].generation_id
            == generation_a
        )

        print('UUID publication: PASS')

        now = manager.get_clock().now()

        stamp = Time(
            nanoseconds=state.last_stamp.nanoseconds
            + 100_000_000,
            clock_type=state.last_stamp.clock_type
        )

        ok = manager.update_track(
            state,
            'smartphone',
            np.array([0.021, 0.0, 0.30]),
            0.96,
            stamp,
            now,
            'fused_fast_precise',
            ObjectState.SOURCE_FUSED,
        )

        assert ok

        assert state.generation_id == generation_a

        assert (
            snapshot().states[0].generation_id
            == generation_a
        )

        print('Update preserves generation: PASS')

        now = manager.get_clock().now()

        state.last_receipt_stamp = Time(
            nanoseconds=(
                now.nanoseconds
                - int(
                    (manager.remove_timeout + 1.0)
                    * 1e9
                )
            ),
            clock_type=now.clock_type
        )

        msg = snapshot()

        assert not msg.states
        assert 'track_001' not in manager.tracks

        print('Expired lifecycle removed: PASS')

        state_b = create(manager)

        generation_b = state_b.generation_id

        assert generation_b != generation_a

        assert (
            snapshot().states[0].generation_id
            == generation_b
        )

        print('Recreated track gets new UUID: PASS')
        print('Recreated generation:', generation_b)

    finally:

        manager.state_pub = original_pub
        manager.destroy_node()

    manager = ObjectStateManager()

    try:

        state_c = create(manager)

        assert state_c.generation_id not in (
            generation_a,
            generation_b,
        )

        print('New manager gets new UUID: PASS')

        print('New manager generation:',
              state_c.generation_id)

        print()
        print('ALL 6 GENERATION LIFECYCLE TESTS PASSED')

    finally:

        manager.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
