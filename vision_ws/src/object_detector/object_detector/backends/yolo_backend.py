from typing import List, Tuple

import numpy as np
import torch
from ultralytics import YOLO

from .base import Detection, DetectorBackend


class YOLOBackend(DetectorBackend):

    def __init__(
        self,
        model_path: str = 'yolo11s.pt',
        confidence: float = 0.25,
        iou: float = 0.45,
        image_size: int = 640,
        device: str = '0',
        quantize: str = '16',
    ):
        self.model_path = model_path
        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size

        if device != 'cpu' and torch.cuda.is_available():
            self.device = device
            self.quantize = quantize
        else:
            self.device = 'cpu'
            self.quantize = '32'

        print(f'[YOLOBackend] model={self.model_path}')
        print(f'[YOLOBackend] device={self.device}')
        print(f'[YOLOBackend] quantize={self.quantize}')

        self.model = YOLO(self.model_path)

    def predict(
        self,
        bgr_image: np.ndarray
    ) -> Tuple[List[Detection], np.ndarray]:

        results = self.model.predict(
            source=bgr_image,
            imgsz=self.image_size,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            quantize=self.quantize,
            verbose=False,
        )

        result = results[0]

        detections = []

        if result.boxes is not None:
            for box in result.boxes:

                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())

                x1, y1, x2, y2 = (
                    box.xyxy[0]
                    .detach()
                    .cpu()
                    .tolist()
                )

                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=result.names[class_id],
                        confidence=confidence,
                        bbox_xyxy=(
                            float(x1),
                            float(y1),
                            float(x2),
                            float(y2),
                        ),
                    )
                )

        debug_bgr = None

        return detections, debug_bgr
