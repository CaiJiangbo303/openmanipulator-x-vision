"""Build frame-local instance labels."""

import numpy as np


def build_label_map(detections, height, width):
    if height <= 0 or width <= 0:
        raise ValueError("Invalid image dimensions")

    if len(detections) > 65535:
        raise ValueError("Too many instances")

    labels = np.zeros((height, width), dtype="<u2")
    occupied = np.zeros((height, width), dtype=bool)
    ambiguous = np.zeros((height, width), dtype=bool)

    for index, detection in enumerate(detections, start=1):
        if detection.mask is None:
            raise ValueError("Missing native instance mask")

        mask = np.asarray(detection.mask)

        if mask.shape != labels.shape:
            raise ValueError("Native mask shape mismatch")

        if mask.dtype != np.bool_:
            raise ValueError("Expected binary boolean mask")

        ambiguous |= occupied & mask
        labels[mask] = index
        occupied |= mask

    # Do not assign overlapping ambiguous pixels.
    labels[ambiguous] = 0

    return labels
