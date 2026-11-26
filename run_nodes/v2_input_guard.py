"""Input bookkeeping for the V2 controller. All receipt times use the monotonic clock."""


def stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


class ReadyEnter:
    def __init__(self, lifetime):
        self.lifetime = lifetime
        self.clear()

    def clear(self):
        self.pending = None

    def capture(self, identity, ready_since, now):
        if identity is not None and ready_since is not None and now - ready_since >= 0.15:
            self.pending = (identity, ready_since, now)

    def consume(self, identity, ready_since, now):
        ticket = self.pending
        self.clear()
        return (ticket is not None and ticket[:2] == (identity, ready_since)
                and 0 <= now - ticket[2] <= self.lifetime)
