import math
import uuid


def stamp_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class StateGate:
    """Perception-only tracking authorization policy.

    No ROS publishers, action clients or robot commands.
    """

    def __init__(
        self,
        allowed_classes,
        workspace,
        frame='world',
        min_confidence=0.50,
        max_uncertainty=0.030,
        max_age_ms=350.0,
        max_prediction_age_ms=250.0,
        watchdog_ms=250.0,
    ):
        self.allowed_classes = {
            str(c).strip().lower()
            for c in allowed_classes
        }

        self.workspace = workspace
        self.frame = frame
        self.min_confidence = min_confidence
        self.max_uncertainty = max_uncertainty

        self.max_age_ms = max_age_ms
        self.max_prediction_age_ms = max_prediction_age_ms
        self.watchdog_ms = watchdog_ms

        self.locked_id = None
        self.locked_generation_id = None
        self.reason = 'not authorized'

        self.last_header = None
        self.last_source = None
        self.last_seen = None
        self.last_age = None

        self.last_rx = None
        self.last_source_progress = None
        self.last_limit = None
        self.last_now = None

    def revoke(self, reason):
        self.locked_id = None
        self.locked_generation_id = None
        self.reason = reason

        return False, reason

    def validate(self, msg, state, now_ns):

        if msg.header.frame_id != self.frame:
            return False, 'frame mismatch'

        if not state.track_id:
            return False, 'empty track id'

        generation = state.generation_id

        try:
            parsed = uuid.UUID(generation)
        except (ValueError, TypeError, AttributeError):
            return False, 'invalid generation id'

        if (
            parsed.version != 4
            or str(parsed) != generation
        ):
            return False, 'invalid generation id'

        cls = state.class_id.strip().lower()

        if cls not in self.allowed_classes:
            return False, 'class not allowed'

        if not state.valid_3d:
            return False, 'invalid 3d'

        if state.stale:
            return False, 'stale measurement'

        if state.state != state.TRACKED:
            return False, 'target not tracked'

        if state.missed > 0:
            return False, 'target measurement missed'

        if state.age < 1:
            return False, 'invalid lifecycle age'

        if state.measurement_source_type not in (1, 2, 3):
            return False, 'unknown measurement source'

        position = (
            state.position.x,
            state.position.y,
            state.position.z,
        )

        velocity = (
            state.velocity.x,
            state.velocity.y,
            state.velocity.z,
        )

        std = (
            state.position_std.x,
            state.position_std.y,
            state.position_std.z,
        )

        values = (
            *position,
            *velocity,
            *std,
            state.confidence,
            state.measurement_age_ms,
        )

        if not all(math.isfinite(v) for v in values):
            return False, 'non-finite state'

        if not (
            self.min_confidence
            <= state.confidence <= 1.0
        ):
            return False, 'invalid confidence'

        if any(v < 0 for v in std):
            return False, 'negative uncertainty'

        uncertainty = math.sqrt(
            sum(v * v for v in std)
        )

        if uncertainty > self.max_uncertainty:
            return False, 'uncertainty too high'

        for value, bounds in zip(
            position, self.workspace
        ):
            if not bounds[0] <= value <= bounds[1]:
                return False, 'outside workspace'

        header = stamp_ns(msg.header.stamp)
        seen = stamp_ns(state.last_seen)
        source = stamp_ns(state.source_stamp)

        if not 0 < seen <= header <= now_ns:
            return False, 'invalid local timestamp'

        if source <= 0:
            return False, 'invalid source timestamp'

        header_age = (now_ns - header) / 1e6
        receipt_age = (now_ns - seen) / 1e6

        if header_age > self.watchdog_ms:
            return False, 'old state message'

        limit = (
            self.max_prediction_age_ms
            if state.position_predicted
            else self.max_age_ms
        )

        reported_age = float(
            state.measurement_age_ms
        )

        if reported_age < 0:
            return False, 'negative measurement age'

        effective_age = max(
            receipt_age,
            reported_age + header_age,
        )

        if effective_age > limit:
            return False, 'measurement too old'

        return True, 'validated'

    def get_target(self, msg, track_id):
        matches = [
            s for s in msg.states
            if s.track_id == track_id
        ]

        if len(matches) != 1:
            return None

        return matches[0]

    def record(self, msg, state, now_ns, steady_ns):

        source = stamp_ns(state.source_stamp)

        if (
            self.last_source is None
            or source > self.last_source
        ):
            self.last_source_progress = steady_ns

        self.last_header = stamp_ns(msg.header.stamp)
        self.last_source = source
        self.last_seen = stamp_ns(state.last_seen)
        self.last_age = state.age

        self.last_limit = (
            self.max_prediction_age_ms
            if state.position_predicted
            else self.max_age_ms
        )

        self.last_rx = steady_ns
        self.last_now = now_ns

    def authorize(
        self, msg, track_id, now_ns, steady_ns
    ):
        if self.locked_id is not None:
            return self.revoke(
                'release existing authorization first'
            )

        state = self.get_target(msg, track_id)

        if state is None:
            return self.revoke(
                'requested target missing or ambiguous'
            )

        ok, reason = self.validate(
            msg, state, now_ns
        )

        if not ok:
            return self.revoke(reason)

        self.locked_id = track_id
        self.locked_generation_id = state.generation_id
        self.last_source = None

        self.record(
            msg, state, now_ns, steady_ns
        )

        self.reason = 'target authorized'

        return True, self.reason

    def poll(self, now_ns, steady_ns):

        if self.locked_id is None:
            return False, 'not authorized'

        if (
            now_ns < self.last_now
            or steady_ns < self.last_rx
        ):
            return self.revoke('clock moved backwards')

        if (
            steady_ns - self.last_rx
        ) / 1e6 > self.watchdog_ms:
            return self.revoke('state stream timeout')

        if (
            steady_ns - self.last_source_progress
        ) / 1e6 > self.max_prediction_age_ms:
            return self.revoke(
                'source measurements stopped'
            )

        if (
            now_ns - self.last_seen
        ) / 1e6 > self.last_limit:
            return self.revoke('measurement expired')

        self.last_now = now_ns

        return True, 'authorization current'

    def update(self, msg, now_ns, steady_ns):

        ok, reason = self.poll(now_ns, steady_ns)

        if not ok:
            return False, reason

        state = self.get_target(
            msg, self.locked_id
        )

        if state is None:
            return self.revoke(
                'locked target missing or ambiguous'
            )

        ok, reason = self.validate(
            msg, state, now_ns
        )

        if not ok:
            return self.revoke(reason)

        if state.generation_id != self.locked_generation_id:
            return self.revoke('target generation changed')

        if stamp_ns(msg.header.stamp) <= self.last_header:
            return self.revoke(
                'duplicate or out-of-order state'
            )

        if stamp_ns(state.source_stamp) < self.last_source:
            return self.revoke(
                'source timestamp moved backwards'
            )

        if stamp_ns(state.last_seen) < self.last_seen:
            return self.revoke(
                'receipt timestamp moved backwards'
            )

        if state.age < self.last_age:
            return self.revoke(
                'target lifecycle restarted'
            )

        self.record(
            msg, state, now_ns, steady_ns
        )

        return True, 'locked target validated'
