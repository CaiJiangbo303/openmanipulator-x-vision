import time


def run(r, base, case, publish, emit, start, run_dir):
    if case in ('async', 'repeated'):
        # State publishes every 50 ms. MASK arrives 18 ms after each 100 ms acquisition.
        for i in range(160):
            t = i * .05
            time.sleep(max(0, start + t - time.monotonic()))
            now = r.get_clock().now().nanoseconds
            if i % 2 == 0 or case == 'async':
                source = now
            publish(r, (t, 0, 'state', source, t, case), base)
            if i % 2 == 0:
                time.sleep(max(0, start + t + .018 - time.monotonic()))
                publish(r, (t + .018, 0, 'mask', source, t, case), base)
        return

    fault_time = None
    key_t = None
    for i in range(181):
        t = i * .02
        time.sleep(max(0, start + t - time.monotonic()))
        if key_t is None and (run_dir / 'key').exists():
            key_t = t
        age = -1 if key_t is None else t - key_t
        if case in ('joint','track_identity') and fault_time is None:
            with r.lock:
                if r.segment and r.goal_count>=2 and time.monotonic()-r.segment[3]>.04:
                    fault_time=t
                    emit('inject',case=case,goal_id=bytes(r.segment[0].goal_id.uuid).hex())
        generation = 'fixture-life-2' if (case == 'identity' and age >= 0) or (case=='track_identity' and fault_time is not None) else 'fixture-life-1'
        hide = (case == 'vision' and 0 <= age < .28) or (case == 'consume' and .70 <= age < 1.04)
        phase = 'short_hide' if hide else 'sync'
        r.inject_joint = case == 'joint' and fault_time is not None and t-fault_time < .34
        source = r.get_clock().now().nanoseconds
        publish(r, (t, 0, 'state', source, t, phase), base, generation)
        publish(r, (t, 0, 'mask', source, t, phase), base, generation)
