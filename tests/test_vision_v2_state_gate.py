import copy

from object_state_msgs.msg import (
    ObjectState,
    ObjectStateArray,
)

from target_selector.state_gate import StateGate


NOW = 10_000_000_000
STEADY = 5_000_000_000


def set_stamp(stamp, ns):
    stamp.sec = ns // 1_000_000_000
    stamp.nanosec = ns % 1_000_000_000


def frame(dt=0, cls='cell phone', track='track_001'):

    msg = ObjectStateArray()
    msg.header.frame_id = 'world'

    set_stamp(msg.header.stamp, NOW + dt)

    s = ObjectState()

    s.track_id = track
    s.class_id = cls

    s.position.x = 0.30
    s.position.y = 0.00
    s.position.z = 0.15

    s.position_std.x = 0.003
    s.position_std.y = 0.003
    s.position_std.z = 0.003

    s.confidence = 0.95
    s.valid_3d = True
    s.stale = False

    s.state = ObjectState.TRACKED

    s.measurement_source_type = (
        ObjectState.SOURCE_FUSED
    )

    s.position_predicted = True
    s.measurement_age_ms = 20.0

    s.age = 5
    s.missed = 0

    set_stamp(
        s.last_seen,
        NOW + dt - 20_000_000
    )

    set_stamp(
        s.source_stamp,
        2_000_000_000 + dt
    )

    msg.states.append(s)

    return msg


def make_gate():
    return StateGate(
        allowed_classes={
            'cell phone',
            'smartphone',
            'mobile phone',
        },

        # Synthetic test workspace only.
        workspace=(
            (0.20, 0.395),
            (-0.16, 0.16),
            (0.05, 0.40),
        )
    )


def armed_gate():

    gate = make_gate()

    ok, reason = gate.authorize(
        frame(),
        'track_001',
        NOW,
        STEADY,
    )

    assert ok, reason

    return gate


DT = 50_000_000
passed = 0


def report(name):
    global passed
    passed += 1
    print(f'{name}: PASS')


gate = make_gate()

assert not gate.update(
    frame(), NOW, STEADY
)[0]

report('Default unauthorized')


gate = armed_gate()
report('Explicit target authorization')


msg = frame(DT, cls='smartphone')

assert gate.update(
    msg,
    NOW + DT,
    STEADY + DT,
)[0]

assert gate.locked_id == 'track_001'

report('Semantic alias preserves identity')


gate = armed_gate()

msg = frame(DT)

challenger = copy.deepcopy(msg.states[0])
challenger.track_id = 'track_002'
challenger.confidence = 0.99

msg.states.append(challenger)

assert gate.update(
    msg,
    NOW + DT,
    STEADY + DT,
)[0]

assert gate.locked_id == 'track_001'

report('No automatic target switching')


gate = armed_gate()

msg = frame(DT, track='track_002')

assert not gate.update(
    msg,
    NOW + DT,
    STEADY + DT,
)[0]

assert gate.locked_id is None

report('Missing locked target revokes authorization')


assert not gate.update(
    frame(2 * DT),
    NOW + 2 * DT,
    STEADY + 2 * DT,
)[0]

report('No automatic reauthorization')


def rejected_case(name, mutate):

    gate = armed_gate()

    msg = frame(DT)

    mutate(msg, msg.states[0])

    ok, reason = gate.update(
        msg,
        NOW + DT,
        STEADY + DT,
    )

    assert not ok, name
    assert gate.locked_id is None

    report(name)
    print('  reason:', reason)


rejected_case(
    'Person rejection',
    lambda m, s: setattr(s, 'class_id', 'person')
)

rejected_case(
    'Hand rejection',
    lambda m, s: setattr(s, 'class_id', 'hand')
)

rejected_case(
    'Robot self class rejection',
    lambda m, s: setattr(s, 'class_id', 'robot arm')
)

rejected_case(
    'Invalid 3D rejection',
    lambda m, s: setattr(s, 'valid_3d', False)
)

rejected_case(
    'Stale rejection',
    lambda m, s: setattr(s, 'stale', True)
)

rejected_case(
    'Lost state rejection',
    lambda m, s: setattr(s, 'state', ObjectState.LOST)
)

rejected_case(
    'Missed observation rejection',
    lambda m, s: setattr(s, 'missed', 1)
)

rejected_case(
    'Prediction age rejection',
    lambda m, s: setattr(
        s, 'measurement_age_ms', 300.0
    )
)

rejected_case(
    'Low confidence rejection',
    lambda m, s: setattr(s, 'confidence', 0.10)
)

rejected_case(
    'High uncertainty rejection',
    lambda m, s: setattr(s.position_std, 'x', 0.10)
)

rejected_case(
    'Non-finite position rejection',
    lambda m, s: setattr(s.position, 'x', float('nan'))
)

rejected_case(
    'Workspace rejection',
    lambda m, s: setattr(s.position, 'x', 0.80)
)

rejected_case(
    'Frame mismatch rejection',
    lambda m, s: setattr(
        m.header,
        'frame_id',
        'camera_color_optical_frame'
    )
)

rejected_case(
    'Lifecycle restart rejection',
    lambda m, s: setattr(s, 'age', 1)
)

rejected_case(
    'Source timestamp regression rejection',
    lambda m, s: set_stamp(
        s.source_stamp, 1_000_000_000
    )
)

rejected_case(
    'Duplicate state rejection',
    lambda m, s: (
        set_stamp(m.header.stamp, NOW),
        set_stamp(s.last_seen, NOW - 20_000_000)
    )
)


gate = armed_gate()

assert not gate.poll(
    NOW + 300_000_000,
    STEADY + 300_000_000,
)[0]

assert gate.locked_id is None

report('Independent watchdog timeout')


gate = armed_gate()

for i in range(1, 7):

    dt = i * DT
    msg = frame(dt)

    # Replayed source timestamp with fresh receipt time.
    set_stamp(
        msg.states[0].source_stamp,
        2_000_000_000
    )

    ok, reason = gate.update(
        msg,
        NOW + dt,
        STEADY + dt,
    )

    if not ok:
        break

assert not ok
assert gate.locked_id is None

report('Frozen source timestamp rejection')


gate = armed_gate()

assert not gate.poll(
    NOW - DT,
    STEADY + DT,
)[0]

report('Clock regression rejection')


print()
print(f'Total checks passed: {passed}')
print('ALL STATE GATE TESTS PASSED')
