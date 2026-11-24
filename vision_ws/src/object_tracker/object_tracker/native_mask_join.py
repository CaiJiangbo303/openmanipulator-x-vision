"""Exact-stamp native mask and tracker assignment join."""

import time
import numpy as np


class NativeMaskJoin:

    def __init__(self, capacity=8, ttl=0.20):
        self.capacity = capacity
        self.ttl = ttl

        self.masks = {}
        self.mappings = {}

        self.watermarks = {
            "mask": -1,
            "mapping": -1
        }

        self.last_output_stamp = -1

    @staticmethod
    def key(header):
        stamp = header.stamp
        ns = stamp.sec * 1000000000 + stamp.nanosec

        if not header.frame_id or ns <= 0:
            raise ValueError("Invalid frame or stamp")

        return header.frame_id, ns

    def prune(self, now):
        for cache in (self.masks, self.mappings):
            for key, (arrival, _) in list(cache.items()):
                if now - arrival > self.ttl:
                    del cache[key]

            while len(cache) > self.capacity:
                cache.pop(next(iter(cache)))

    def push(self, stream, key, value):
        now = time.monotonic()
        self.prune(now)

        if key[1] <= self.watermarks[stream]:
            return None

        self.watermarks[stream] = key[1]

        cache = (
            self.masks if stream == "mask"
            else self.mappings
        )

        cache[key] = (now, value)
        self.prune(now)

        if key not in self.masks or key not in self.mappings:
            return None

        _, msg = self.masks.pop(key)
        _, lut = self.mappings.pop(key)

        if key[1] <= self.last_output_stamp:
            return None

        labels = np.frombuffer(
            msg.data, dtype="<u2"
        ).reshape(msg.height, msg.width)

        if int(labels.max()) >= len(lut):
            raise ValueError("Unknown detection label")

        output = lut[labels]

        self.last_output_stamp = key[1]

        return msg, output

    def add_mask(self, msg):
        if (
            msg.encoding != "mono16"
            or msg.is_bigendian
            or msg.height <= 0
            or msg.width <= 0
            or msg.step != msg.width * 2
            or len(msg.data) != msg.height * msg.step
        ):
            raise ValueError("Invalid native mask format")

        return self.push(
            "mask",
            self.key(msg.header),
            msg
        )

    def add_assignments(self, header, detections, assignments):
        count = len(detections)

        if count > 65535:
            raise ValueError("Too many detections")

        for index, det in enumerate(detections, 1):
            if det.id != f"det_{index:05d}":
                raise ValueError("Invalid detector instance ID")

        lut = np.zeros(count + 1, dtype="<u2")
        used = set()

        for index, track_id in assignments.items():
            if not 0 <= index < count:
                raise ValueError("Invalid detection index")

            if not 1 <= track_id <= 65535:
                raise ValueError("Track ID exceeds mono16 range")

            if track_id in used:
                raise ValueError("Duplicate track assignment")

            lut[index + 1] = track_id
            used.add(track_id)

        return self.push(
            "mapping",
            self.key(header),
            lut
        )
