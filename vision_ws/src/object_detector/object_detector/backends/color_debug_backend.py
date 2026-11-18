import cv2
import numpy as np

from .base import Detection, DetectorBackend


class ColorDebugBackend(DetectorBackend):
    """
    Simulation-only color detector.

    This backend exists only for deterministic Gazebo
    perception validation.

    It is NOT used by the production YOLO perception path.
    """

    def __init__(
        self,
        min_area: int = 20,
        max_area: int = 5000,
    ):
        self.min_area = min_area
        self.max_area = max_area

        print(
            '[ColorDebugBackend] simulation validation mode'
        )

        print(
            '[ColorDebugBackend] component area='
            f'{self.min_area}..{self.max_area}'
        )

        print(
            '[ColorDebugBackend] no fixed image-Y ROI'
        )

    def _find_component(
        self,
        mask,
        class_id,
        class_name,
    ):
        mask = (
            mask.astype(np.uint8)
            *
            255
        )

        (
            count,
            labels,
            stats,
            centroids,
        ) = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )

        candidates = []

        for component_id in range(
            1,
            count
        ):
            x = int(
                stats[
                    component_id,
                    cv2.CC_STAT_LEFT
                ]
            )

            y = int(
                stats[
                    component_id,
                    cv2.CC_STAT_TOP
                ]
            )

            w = int(
                stats[
                    component_id,
                    cv2.CC_STAT_WIDTH
                ]
            )

            h = int(
                stats[
                    component_id,
                    cv2.CC_STAT_HEIGHT
                ]
            )

            area = int(
                stats[
                    component_id,
                    cv2.CC_STAT_AREA
                ]
            )

            if area < self.min_area:
                continue

            if area > self.max_area:
                continue

            center_x = float(
                centroids[
                    component_id,
                    0
                ]
            )

            center_y = float(
                centroids[
                    component_id,
                    1
                ]
            )

            detection = Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=1.0,
                bbox_xyxy=(
                    float(x),
                    float(y),
                    float(x + w),
                    float(y + h),
                ),
            )

            candidates.append(
                {
                    'area': area,
                    'center_x': center_x,
                    'center_y': center_y,
                    'detection': detection,
                }
            )

        if not candidates:
            return None

        # -------------------------------------------------
        # Simulation-only selection policy.
        #
        # The Gazebo model contains a red component on the
        # robot itself. In the current validation world,
        # the red cylinder is the rightmost legitimate red
        # connected component.
        #
        # Unlike the previous implementation, there is NO
        # fixed center_y cutoff. The target is therefore
        # still detectable when the F550 changes position
        # and the object moves lower in the image.
        #
        # Production perception does not use this rule.
        # -------------------------------------------------

        if class_name == 'RED':

            candidates.sort(
                key=lambda item: (
                    item['center_x'],
                    item['area'],
                ),
                reverse=True
            )

        else:

            candidates.sort(
                key=lambda item:
                    item['area'],
                reverse=True
            )

        return candidates[0][
            'detection'
        ]

    def predict(
        self,
        bgr_image
    ):
        b = (
            bgr_image[
                :,
                :,
                0
            ].astype(
                np.int16
            )
        )

        g = (
            bgr_image[
                :,
                :,
                1
            ].astype(
                np.int16
            )
        )

        r = (
            bgr_image[
                :,
                :,
                2
            ].astype(
                np.int16
            )
        )

        red_mask = (
            (r > 170)
            &
            (r > g + 50)
            &
            (r > b + 50)
        )

        green_mask = (
            (g > 150)
            &
            (g > r + 40)
            &
            (g > b + 40)
        )

        blue_mask = (
            (b > 150)
            &
            (b > r + 40)
            &
            (b > g + 40)
        )

        definitions = [
            (
                red_mask,
                0,
                'RED'
            ),
            (
                green_mask,
                1,
                'GREEN'
            ),
            (
                blue_mask,
                2,
                'BLUE'
            ),
        ]

        detections = []

        for (
            mask,
            class_id,
            class_name,
        ) in definitions:

            detection = (
                self._find_component(
                    mask,
                    class_id,
                    class_name,
                )
            )

            if detection is not None:

                detections.append(
                    detection
                )

        return detections, None
