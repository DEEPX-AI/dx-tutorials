"""PP-OCRv6 tiny DET-to-REC inference pipeline for DEEPX DXNN models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectionConfig:
    pixel_threshold: float = 0.7
    box_threshold: float = 0.6
    unclip_ratio: float = 1.4
    max_candidates: int = 50
    max_boxes: int = 50
    min_box_side: float = 4.0
    max_tilt_degrees: float = 30.0


@dataclass(frozen=True)
class RecognitionConfig:
    confidence_threshold: float = 0.5
    min_crop_side: int = 4


@dataclass
class OcrText:
    box_index: int
    box: np.ndarray
    text: str
    score: float


@dataclass
class OcrResult:
    boxes: list[np.ndarray]
    texts: list[OcrText]
    detection_ms: float
    recognition_ms: float
    total_ms: float


@dataclass(frozen=True)
class RecognitionModel:
    name: str
    engine: Any
    height: int
    width: int

    @property
    def capacity(self) -> float:
        return self.width / self.height


REC_MODEL_NAMES = {
    2.5: "rec_fixed_ratio_2_5",
    5.0: "rec_fixed_ratio_5",
    10.0: "rec_fixed_ratio_10",
    15.0: "rec_fixed_ratio_15",
    25.0: "rec_fixed_ratio_25",
}


class PaddleOcrV6:
    """Run PP-OCRv6 tiny text detection followed by text recognition."""

    def __init__(
        self,
        model_dir: Path,
        dictionary_path: Path,
        detection: DetectionConfig | None = None,
        recognition: RecognitionConfig | None = None,
        use_ort: bool = True,
    ) -> None:
        try:
            from dx_engine import InferenceEngine, InferenceOption
        except ImportError as error:
            raise RuntimeError(
                "dx_engine is not installed in this Python environment. "
                "Install the SDK's dx_rt/python_package into the tutorial .venv."
            ) from error

        self.model_dir = Path(model_dir).expanduser().resolve()
        self.dictionary_path = Path(dictionary_path).expanduser().resolve()
        self.detection_config = detection or DetectionConfig()
        self.recognition_config = recognition or RecognitionConfig()
        self.characters = self._load_characters(self.dictionary_path)

        option = InferenceOption().set_use_ort(use_ort)
        det_path = self._model_path("det_fixed")
        self.det_engine = InferenceEngine(str(det_path), option)
        self.det_height, self.det_width = self._input_hw(self.det_engine, "DET")

        self.rec_models: list[RecognitionModel] = []
        for _, model_name in REC_MODEL_NAMES.items():
            path = self._model_path(model_name)
            engine = InferenceEngine(str(path), option)
            height, width = self._input_hw(engine, model_name)
            self.rec_models.append(RecognitionModel(model_name, engine, height, width))

        self.rec_models.sort(key=lambda model: model.capacity)
        self._validate_model_contracts()

    def _model_path(self, model_name: str) -> Path:
        path = self.model_dir / model_name / f"{model_name}.dxnn"
        if not path.is_file():
            raise FileNotFoundError(f"Required DXNN model was not found: {path}")
        return path

    @staticmethod
    def _input_hw(engine: Any, stage: str) -> tuple[int, int]:
        inputs = engine.get_input_tensors_info()
        if len(inputs) != 1:
            raise RuntimeError(f"{stage} must have exactly one input tensor; got {len(inputs)}")
        shape = list(inputs[0]["shape"])
        if len(shape) != 4 or shape[0] != 1 or shape[3] != 3:
            raise RuntimeError(f"{stage} requires NHWC [1,H,W,3] input; got {shape}")
        if inputs[0]["dtype"] is not np.uint8:
            raise RuntimeError(f"{stage} requires uint8 input; got {inputs[0]['dtype']}")
        return int(shape[1]), int(shape[2])

    def _validate_model_contracts(self) -> None:
        if (self.det_height, self.det_width) != (640, 640):
            raise RuntimeError(
                "This camera application expects a 640x640 detector, but the model input is "
                f"{self.det_width}x{self.det_height}."
            )
        if not self.rec_models:
            raise RuntimeError("No recognition models were loaded")
        for model in self.rec_models:
            if model.height != 48:
                raise RuntimeError(
                    f"{model.name} must use recognition height 48; got {model.height}"
                )

    @staticmethod
    def _load_characters(path: Path) -> list[str]:
        if not path.is_file():
            raise FileNotFoundError(f"PP-OCRv6 tiny dictionary was not found: {path}")
        dictionary = path.read_text(encoding="utf-8").splitlines()
        if not dictionary:
            raise RuntimeError(f"Recognition dictionary is empty: {path}")
        return ["blank", *dictionary, " "]

    @staticmethod
    def _run(engine: Any, image: np.ndarray) -> list[np.ndarray]:
        tensor = np.ascontiguousarray(image[np.newaxis, ...], dtype=np.uint8)
        outputs = engine.run([tensor])
        if not outputs:
            raise RuntimeError("DXNN inference returned no output tensors")
        return outputs

    def run(self, frame_bgr: np.ndarray) -> OcrResult:
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Input frame is empty")

        started = perf_counter()
        boxes, detection_ms = self._detect(frame_bgr)
        crops: list[np.ndarray] = []
        valid_boxes: list[np.ndarray] = []
        for box in boxes[: self.detection_config.max_boxes]:
            crop = self._perspective_crop(frame_bgr, box)
            if crop.size == 0:
                continue
            if min(crop.shape[:2]) < self.recognition_config.min_crop_side:
                continue
            if crop.shape[0] > crop.shape[1] * 2:
                crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
            valid_boxes.append(box)
            crops.append(crop)

        texts, recognition_ms = self._recognize(valid_boxes, crops)
        return OcrResult(
            boxes=valid_boxes,
            texts=texts,
            detection_ms=detection_ms,
            recognition_ms=recognition_ms,
            total_ms=(perf_counter() - started) * 1000.0,
        )

    def validate_models(self) -> list[str]:
        """Run one synthetic inference through DET and every REC model."""
        messages: list[str] = []

        det_input = np.full((self.det_height, self.det_width, 3), 114, dtype=np.uint8)
        det_output = np.asarray(self._run(self.det_engine, det_input)[0])
        if det_output.squeeze().ndim != 2:
            raise RuntimeError(f"Unexpected DET output shape: {det_output.shape}")
        messages.append(
            f"PASS DET  det_fixed input=1x{self.det_height}x{self.det_width}x3 "
            f"output={list(det_output.shape)}"
        )

        for model in self.rec_models:
            rec_input = np.full((model.height, model.width, 3), 114, dtype=np.uint8)
            rec_output = np.asarray(self._run(model.engine, rec_input)[0])
            self._decode_ctc(rec_output)
            messages.append(
                f"PASS REC  {model.name} input=1x{model.height}x{model.width}x3 "
                f"output={list(rec_output.shape)}"
            )
        return messages

    def _detect(self, frame_bgr: np.ndarray) -> tuple[list[np.ndarray], float]:
        resized = cv2.resize(
            frame_bgr,
            (self.det_width, self.det_height),
            interpolation=cv2.INTER_LINEAR,
        )
        input_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        started = perf_counter()
        output = self._run(self.det_engine, input_rgb)[0]
        elapsed_ms = (perf_counter() - started) * 1000.0

        probability = np.asarray(output, dtype=np.float32).squeeze()
        if probability.ndim != 2:
            raise RuntimeError(f"DET output must reduce to a 2D map; got {output.shape}")
        boxes = self._decode_detection(probability, frame_bgr.shape[:2])
        return boxes, elapsed_ms

    def _decode_detection(
        self,
        probability: np.ndarray,
        image_shape: tuple[int, int],
    ) -> list[np.ndarray]:
        config = self.detection_config
        mask = (probability > config.pixel_threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        image_height, image_width = image_shape
        scale_x = image_width / probability.shape[1]
        scale_y = image_height / probability.shape[0]
        boxes: list[np.ndarray] = []

        for contour in contours[: config.max_candidates]:
            box, min_side = self._mini_box(contour)
            if min_side < 3.0:
                continue
            if self._box_score(probability, box) < config.box_threshold:
                continue

            expanded = self._unclip_approx(contour, config.unclip_ratio)
            if expanded is None:
                continue
            box, min_side = self._mini_box(expanded)
            if min_side < 5.0:
                continue

            box[:, 0] *= scale_x
            box[:, 1] *= scale_y
            box = self._order_clockwise(box)
            box[:, 0] = np.clip(box[:, 0], 0, image_width - 1)
            box[:, 1] = np.clip(box[:, 1], 0, image_height - 1)

            width = max(np.linalg.norm(box[1] - box[0]), np.linalg.norm(box[2] - box[3]))
            height = max(np.linalg.norm(box[3] - box[0]), np.linalg.norm(box[2] - box[1]))
            if min(width, height) < config.min_box_side:
                continue
            if not self._tilt_is_supported(box, config.max_tilt_degrees):
                continue
            boxes.append(box.astype(np.float32))

        boxes.sort(key=lambda box: (int(np.mean(box[:, 1]) // 10), float(np.mean(box[:, 0]))))
        return boxes

    @staticmethod
    def _mini_box(points: np.ndarray) -> tuple[np.ndarray, float]:
        contour = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        rectangle = cv2.minAreaRect(contour)
        box = PaddleOcrV6._order_clockwise(cv2.boxPoints(rectangle))
        return box, float(min(rectangle[1]))

    @staticmethod
    def _order_clockwise(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32).reshape(4, 2)
        ordered = np.empty((4, 2), dtype=np.float32)
        sums = points.sum(axis=1)
        differences = np.diff(points, axis=1).reshape(-1)
        ordered[0] = points[np.argmin(sums)]
        ordered[2] = points[np.argmax(sums)]
        ordered[1] = points[np.argmin(differences)]
        ordered[3] = points[np.argmax(differences)]
        return ordered

    @staticmethod
    def _box_score(probability: np.ndarray, box: np.ndarray) -> float:
        height, width = probability.shape
        xmin = int(np.clip(np.floor(box[:, 0].min()), 0, width - 1))
        xmax = int(np.clip(np.ceil(box[:, 0].max()), 0, width - 1))
        ymin = int(np.clip(np.floor(box[:, 1].min()), 0, height - 1))
        ymax = int(np.clip(np.ceil(box[:, 1].max()), 0, height - 1))
        if xmax < xmin or ymax < ymin:
            return 0.0

        local_box = np.rint(box - np.array([xmin, ymin], dtype=np.float32)).astype(np.int32)
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        cv2.fillPoly(mask, [local_box], 1)
        return float(cv2.mean(probability[ymin : ymax + 1, xmin : xmax + 1], mask)[0])

    @staticmethod
    def _unclip_approx(contour: np.ndarray, unclip_ratio: float) -> np.ndarray | None:
        area = abs(float(cv2.contourArea(contour)))
        perimeter = float(cv2.arcLength(contour, True))
        if area <= 1e-6 or perimeter <= 1e-6:
            return None
        distance = area * unclip_ratio / perimeter
        center, (width, height), angle = cv2.minAreaRect(contour)
        expanded = (center, (width + 2.0 * distance, height + 2.0 * distance), angle)
        return cv2.boxPoints(expanded)

    @staticmethod
    def _tilt_is_supported(box: np.ndarray, max_degrees: float) -> bool:
        horizontal = box[1] - box[0]
        vertical = box[3] - box[0]
        direction = vertical if np.linalg.norm(vertical) > np.linalg.norm(horizontal) else horizontal
        if np.linalg.norm(direction) <= 1e-6:
            return False
        angle = float(np.degrees(np.arctan2(direction[1], direction[0])))
        while angle <= -90.0:
            angle += 180.0
        while angle > 90.0:
            angle -= 180.0
        return abs(angle) < max_degrees

    @staticmethod
    def _perspective_crop(image: np.ndarray, box: np.ndarray) -> np.ndarray:
        width = int(round(max(np.linalg.norm(box[1] - box[0]), np.linalg.norm(box[2] - box[3]))))
        height = int(round(max(np.linalg.norm(box[3] - box[0]), np.linalg.norm(box[2] - box[1]))))
        if width < 1 or height < 1:
            return np.empty((0, 0, 3), dtype=np.uint8)
        destination = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(box.astype(np.float32), destination)
        return cv2.warpPerspective(
            image,
            transform,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def _select_recognition_model(self, aspect_ratio: float) -> RecognitionModel:
        for model in self.rec_models:
            if aspect_ratio <= model.capacity + 1e-6:
                return model
        return self.rec_models[-1]

    @staticmethod
    def _prepare_recognition(crop_bgr: np.ndarray, model: RecognitionModel) -> np.ndarray:
        ratio = crop_bgr.shape[1] / max(crop_bgr.shape[0], 1)
        resized_width = min(model.width, max(1, int(np.ceil(model.height * ratio))))
        resized = cv2.resize(crop_bgr, (resized_width, model.height), interpolation=cv2.INTER_LINEAR)
        padded = np.full((model.height, model.width, 3), 114, dtype=np.uint8)
        padded[:, :resized_width] = resized
        return cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

    def _recognize(
        self,
        boxes: list[np.ndarray],
        crops: list[np.ndarray],
    ) -> tuple[list[OcrText], float]:
        if not crops:
            return [], 0.0

        started = perf_counter()
        pending: list[tuple[int, np.ndarray, RecognitionModel, int, np.ndarray]] = []
        for index, (box, crop) in enumerate(zip(boxes, crops)):
            aspect_ratio = crop.shape[1] / max(crop.shape[0], 1)
            model = self._select_recognition_model(aspect_ratio)
            prepared = self._prepare_recognition(crop, model)
            tensor = np.ascontiguousarray(prepared[np.newaxis, ...], dtype=np.uint8)
            job_id = model.engine.run_async([tensor])
            pending.append((index, box, model, job_id, tensor))

        results: list[OcrText] = []
        for index, box, model, job_id, _input_owner in pending:
            outputs = model.engine.wait(job_id)
            if not outputs:
                continue
            text, score = self._decode_ctc(np.asarray(outputs[0]))
            if text and score >= self.recognition_config.confidence_threshold:
                results.append(OcrText(index, box, text, score))

        return results, (perf_counter() - started) * 1000.0

    def _decode_ctc(self, output: np.ndarray) -> tuple[str, float]:
        logits = np.asarray(output, dtype=np.float32)
        if logits.ndim == 3:
            logits = logits[0]
        if logits.ndim != 2:
            raise RuntimeError(f"REC output must be [1,T,C] or [T,C]; got {output.shape}")
        if logits.shape[1] != len(self.characters):
            raise RuntimeError(
                "REC class count does not match the PP-OCRv6 tiny dictionary: "
                f"model={logits.shape[1]}, dictionary={len(self.characters)}"
            )

        indices = logits.argmax(axis=1)
        scores = logits.max(axis=1)
        decoded: list[str] = []
        confidence: list[float] = []
        previous = -1
        for index, score in zip(indices.tolist(), scores.tolist()):
            duplicate = index == previous
            previous = index
            if index == 0 or duplicate:
                continue
            decoded.append(self.characters[index])
            confidence.append(float(score))

        if not decoded:
            return "", 0.0
        return "".join(decoded), float(np.mean(confidence))

    def close(self) -> None:
        for engine in [self.det_engine, *(model.engine for model in self.rec_models)]:
            dispose = getattr(engine, "dispose", None)
            if callable(dispose):
                dispose()
