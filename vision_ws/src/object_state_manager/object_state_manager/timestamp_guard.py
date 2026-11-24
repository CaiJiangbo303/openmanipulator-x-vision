"""Timestamp continuity guard independent of ROS."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GuardDecision:
    accepted: bool
    reset: bool
    reason: str


class TimestampGuard:

    def __init__(
        self,
        jump_threshold_ms=200.0,
        timeout_ms=250.0,
        recovery_frames=10,
        clock_jump_ms=50.0
    ):
        self.jump_ns = int(jump_threshold_ms * 1e6)
        self.timeout_ns = int(timeout_ms * 1e6)
        self.clock_jump_ns = int(clock_jump_ms * 1e6)
        self.recovery_frames = int(recovery_frames)

        if min(
            self.jump_ns,
            self.timeout_ns,
            self.clock_jump_ns,
            self.recovery_frames
        ) <= 0:
            raise ValueError("Invalid guard configuration")

        self.last_source_ns = None
        self.last_receipt_ns = None

        self.last_clock_ns = None
        self.last_clock_steady_ns = None

        self.good_frames = 0
        self.ready = False
        self.epoch = 0

    def invalidate(self, reason):
        self.last_source_ns = None
        self.last_receipt_ns = None
        self.good_frames = 0
        self.ready = False
        self.epoch += 1

        return GuardDecision(False, True, reason)

    def check_clock(self, ros_ns, steady_ns):
        old_ros = self.last_clock_ns
        old_steady = self.last_clock_steady_ns

        self.last_clock_ns = ros_ns
        self.last_clock_steady_ns = steady_ns

        if old_ros is not None:
            dr = ros_ns - old_ros
            dm = steady_ns - old_steady

            if dm < 0 or dr < 0:
                return self.invalidate(
                    "clock moved backwards"
                )

            if abs(dr - dm) > self.clock_jump_ns:
                return self.invalidate(
                    "system clock discontinuity"
                )

        return GuardDecision(
            self.ready, False, "clock continuous"
        )

    def observe(self, source_ns, steady_ns):

        if source_ns <= 0 or steady_ns < 0:
            return self.invalidate(
                "invalid timestamp"
            )

        if self.last_source_ns is None:
            self.last_source_ns = source_ns
            self.last_receipt_ns = steady_ns

            return GuardDecision(
                False, False, "baseline established"
            )

        ds = source_ns - self.last_source_ns
        dr = steady_ns - self.last_receipt_ns

        if ds <= 0:
            return self.invalidate(
                "non-increasing source timestamp"
            )

        if dr < 0 or dr > self.timeout_ns:
            return self.invalidate(
                "observation timeout"
            )

        if abs(ds - dr) > self.jump_ns:
            return self.invalidate(
                "source timestamp discontinuity"
            )

        self.last_source_ns = source_ns
        self.last_receipt_ns = steady_ns

        self.good_frames = min(
            self.good_frames + 1,
            self.recovery_frames
        )

        self.ready = (
            self.good_frames >= self.recovery_frames
        )

        return GuardDecision(
            self.ready,
            False,
            "accepted" if self.ready else "recovering"
        )

    def poll(self, steady_ns):

        if self.last_receipt_ns is not None:
            elapsed = (
                steady_ns - self.last_receipt_ns
            )

            if elapsed < 0 or elapsed > self.timeout_ns:
                return self.invalidate(
                    "observation stream timeout"
                )

        return GuardDecision(
            self.ready, False, "stream status"
        )
