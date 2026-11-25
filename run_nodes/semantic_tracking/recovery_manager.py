"""Pure tracking recovery state machine. No ROS or motion commands."""

import math
from dataclasses import dataclass
from enum import Enum


class Status(str, Enum):
    TRACKING = "TRACKING"
    HOLD = "HOLD"
    RECOVERY_WAIT = "RECOVERY_WAIT"
    REVALIDATING = "REVALIDATING"
    REAUTHORIZED = "REAUTHORIZED"
    EXPIRED = "EXPIRED"


@dataclass(frozen=True)
class Observation:
    track_id: str
    generation_id: str
    position: tuple
    source_stamp: float
    mask_stamp: float

    @property
    def identity(self):
        return self.track_id, self.generation_id

    def valid(self):
        return (
            bool(self.track_id)
            and bool(self.generation_id)
            and len(self.position) == 3
            and all(math.isfinite(x) for x in self.position)
            and math.isfinite(self.source_stamp)
            and math.isfinite(self.mask_stamp)
            and self.source_stamp > 0
            and self.mask_stamp > 0
        )


@dataclass(frozen=True)
class RecoveryConfig:
    short_hold_sec: float = 0.35
    recovery_timeout_sec: float = 3.0
    max_position_delta: float = 0.030
    required_updates: int = 3
    min_validation_sec: float = 0.20

    def __post_init__(self):
        values = (
            self.short_hold_sec,
            self.recovery_timeout_sec,
            self.max_position_delta,
            self.min_validation_sec
        )
        if (
            not all(math.isfinite(x) for x in values)
            or not 0 < self.short_hold_sec < self.recovery_timeout_sec
            or self.max_position_delta <= 0
            or self.required_updates < 3
            or not 0 < self.min_validation_sec < self.recovery_timeout_sec
        ):
            raise ValueError("Invalid recovery configuration")


class TrackingRecoveryManager:

    def __init__(self, initial, now, config=None):
        if not initial.valid() or not math.isfinite(now):
            raise ValueError("Invalid initial observation")

        self.config = config or RecoveryConfig()
        self.observation = initial
        self.last_good = now
        self.last_tick = now
        self.status = Status.TRACKING
        self.reauth_required = False
        self.reauth_count = 0

        self.clear_pending()

    @property
    def identity(self):
        return self.observation.identity

    @property
    def authorized(self):
        return self.status in (
            Status.TRACKING,
            Status.REAUTHORIZED
        )

    def clear_pending(self):
        self.pending_identity = None
        self.pending_since = None
        self.pending_source = None
        self.pending_mask = None
        self.pending_count = 0

    def expire(self):
        self.clear_pending()
        self.status = Status.EXPIRED
        return self.status

    def accept(self, obs, now, reauthorized=False):
        self.observation = obs
        self.last_good = now
        self.reauth_required = False
        self.clear_pending()

        if reauthorized:
            self.reauth_count += 1
            self.status = Status.REAUTHORIZED
        else:
            self.status = Status.TRACKING

        return self.status

    def step(self, now, tracking=None, recovery=None):
        """
        Inputs must already pass external visual safety checks.

        tracking: observation passing tracking threshold.
        recovery: observation passing acquisition threshold.

        None means missing, ambiguous or rejected evidence.
        """

        if self.status == Status.EXPIRED:
            return self.status

        if not math.isfinite(now) or now < self.last_tick:
            return self.expire()

        self.last_tick = now
        gap = now - self.last_good

        if gap >= self.config.recovery_timeout_sec:
            return self.expire()

        old = self.observation

        if tracking is not None and tracking.valid():
            if tracking.identity != old.identity:
                self.reauth_required = True

            elif (
                not self.reauth_required
                and gap <= self.config.short_hold_sec
                and tracking.source_stamp >= old.source_stamp
                and tracking.mask_stamp >= old.mask_stamp
                and (
                    self.status != Status.HOLD
                    or tracking.source_stamp > old.source_stamp
                )
            ):
                return self.accept(tracking, now)

        if (
            recovery is not None
            and recovery.valid()
            and recovery.identity != old.identity
        ):
            self.reauth_required = True

        if gap > self.config.short_hold_sec:
            self.reauth_required = True

        if not self.reauth_required:
            self.status = Status.HOLD
            return self.status

        if (
            recovery is None
            or not recovery.valid()
            or recovery.track_id != old.track_id
            or recovery.source_stamp <= old.source_stamp
            or recovery.mask_stamp <= old.mask_stamp
            or math.dist(
                recovery.position,
                old.position
            ) > self.config.max_position_delta
        ):
            self.clear_pending()
            self.status = Status.RECOVERY_WAIT
            return self.status

        if recovery.identity != self.pending_identity:
            self.clear_pending()
            self.pending_identity = recovery.identity
            self.pending_since = now

        if (
            self.pending_source is not None
            and (
                recovery.source_stamp < self.pending_source
                or recovery.mask_stamp < self.pending_mask
            )
        ):
            self.clear_pending()
            self.status = Status.RECOVERY_WAIT
            return self.status

        if (
            self.pending_source is None
            or (
                recovery.source_stamp > self.pending_source
                and recovery.mask_stamp > self.pending_mask
            )
        ):
            self.pending_count += 1
            self.pending_source = recovery.source_stamp
            self.pending_mask = recovery.mask_stamp

        if (
            self.pending_count >= self.config.required_updates
            and now - self.pending_since
            >= self.config.min_validation_sec
        ):
            return self.accept(recovery, now, True)

        self.status = Status.REVALIDATING
        return self.status
