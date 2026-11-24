from typing import List, Tuple

import numpy as np
import torch

from .base import Detection, DetectorBackend


class YOLOEBackend(DetectorBackend):

    def __init__(
        self,
        model_path: str = 'yoloe-26s-seg-pf.pt',
        confidence: float = 0.25,
        iou: float = 0.45,
        image_size: int = 640,
        device: str = '0',
        quantize: str = '16',
        prompt_mode: str = 'prompt_free',
        prompt_classes: str = '',
    ):
        from ultralytics import YOLOE

        self.model_path = model_path
        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size
        self.prompt_mode = prompt_mode

        self.classes = list(dict.fromkeys(
            name.strip()
            for name in prompt_classes.split(',')
            if name.strip()
        ))

        if device != 'cpu' and torch.cuda.is_available():
            self.device = device
            self.quantize = quantize
        else:
            self.device = 'cpu'
            self.quantize = '32'

        if prompt_mode not in ('prompt_free', 'text'):
            raise ValueError(
                f'Unsupported YOLOE mode: {prompt_mode}'
            )

        is_pf = model_path.endswith('-pf.pt')

        if prompt_mode == 'prompt_free':
            if not is_pf:
                raise ValueError(
                    'Prompt-free mode requires a -pf.pt model'
                )

        if prompt_mode == 'text':
            if is_pf:
                raise ValueError(
                    'Text mode requires a non-PF model'
                )

            if not self.classes:
                raise ValueError(
                    'Text mode requires prompt_classes'
                )

        print(f'[YOLOE] model={self.model_path}')
        print(f'[YOLOE] mode={self.prompt_mode}')
        print(f'[YOLOE] device={self.device}')
        print(f'[YOLOE] quantize={self.quantize}')

        self.model = YOLOE(self.model_path)

        if self.prompt_mode == 'text':
            self.model.set_classes(self.classes)
            print(f'[YOLOE] classes={self.classes}')

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

        if result.boxes is None or len(result.boxes) == 0:
            return detections, None

        boxes = result.boxes.xyxy.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()

        for bbox, cls, score in zip(
            boxes, classes, scores
        ):
            class_id = int(cls)

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=str(result.names[class_id]),
                    confidence=float(score),
                    bbox_xyxy=tuple(
                        float(v) for v in bbox
                    ),
                )
            )

        return detections, None
