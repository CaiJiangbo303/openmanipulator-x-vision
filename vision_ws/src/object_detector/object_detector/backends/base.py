from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]


class DetectorBackend(ABC):

    @abstractmethod
    def predict(self, bgr_image: np.ndarray) -> Tuple[List[Detection], np.ndarray]:
        """
        Returns:
            detections:
                unified detector output

            debug_bgr:
                visualization image in BGR format
        """
        raise NotImplementedError
