import cv2
import numpy as np

from .base import Detection, DetectorBackend


class ColorDebugBackend(DetectorBackend):
    """
    Simulation-only detector.

    Purpose:
        Validate RGB -> Detection2D -> Depth -> TF -> world XYZ
        against known Gazebo object positions.

    This backend is NOT part of the production perception path.
    """

    def __init__(
        self,
        min_area: int = 20,
        max_area: int = 350,
        roi_y_max: int = 340,
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.roi_y_max = roi_y_max

        print('[ColorDebugBackend] simulation validation mode')
        print(
            f'[ColorDebugBackend] area='
            f'{self.min_area}..{self.max_area}'
        )
        print(
            f'[ColorDebugBackend] roi_y_max='
            f'{self.roi_y_max}'
        )

    def _find_component(
        self,
        mask,
        class_id,
        class_name,
    ):
        mask = mask.astype(np.uint8) * 255

        count, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                mask,
                connectivity=8
            )
        )

        candidates = []

        for component_id in range(1, count):

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

            center_y = (
                y + h / 2.0
            )

            if area < self.min_area:
                continue

            if area > self.max_area:
                continue

            # Current Gazebo validation objects are in the
            # upper / middle portion of the RGB image.
            # This rejects the large lower-frame red robot
            # self-occlusion seen in our test scene.
            if center_y > self.roi_y_max:
                continue

            candidates.append(
                (
                    area,
                    Detection(
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
                )
            )

        if not candidates:
            return None

        # One Gazebo validation target per color.
        #
        # RED has an additional robot-self red component near
        # the image center in this simulation. The actual red
        # validation cylinder is the rightmost valid red component.
        #
        # This rule is simulation/debug only and is never used by
        # the production YOLO backend.
        if class_name == 'RED':

            candidates.sort(
                key=lambda item: (
                    item[1].bbox_xyxy[0]
                    +
                    item[1].bbox_xyxy[2]
                ) / 2.0,
                reverse=True
            )

        else:

            candidates.sort(
                key=lambda item: item[0],
                reverse=True
            )

        return candidates[0][1]

    def predict(self, bgr_image):

        b = bgr_image[:, :, 0].astype(np.int16)
        g = bgr_image[:, :, 1].astype(np.int16)
        r = bgr_image[:, :, 2].astype(np.int16)

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

        detections = []

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

        for (
            mask,
            class_id,
            class_name
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

        # No visualization in validation mode.
        # Keep machine pipeline as lightweight as possible.
        return detections, None
