"""OpenCV wrappers for YuNet detection and SFace embedding extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

from face_pipeline.config import (
    DEFAULT_DETECTION_THRESHOLD,
    DEFAULT_NMS_THRESHOLD,
)
from face_pipeline.matching import normalize_embedding


@dataclass(frozen=True)
class Detection:
    """One face returned by YuNet."""

    raw: NDArray[np.float32]

    @property
    def box(self) -> tuple[int, int, int, int]:
        x, y, width, height = self.raw[:4]
        return int(x), int(y), int(width), int(height)

    @property
    def landmarks(self) -> NDArray[np.float32]:
        return self.raw[4:14].reshape(5, 2)

    @property
    def confidence(self) -> float:
        return float(self.raw[14])


class OpenCVFaceModels:
    """Expose detection and embedding as separate, replaceable operations."""

    def __init__(
        self,
        detector_path: Path,
        recognizer_path: Path,
        detection_threshold: float = DEFAULT_DETECTION_THRESHOLD,
        nms_threshold: float = DEFAULT_NMS_THRESHOLD,
    ) -> None:
        for path in (detector_path, recognizer_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing model: {path}. Run `uv run sce-face download-models` first."
                )

        self.detector = cv.FaceDetectorYN.create(
            str(detector_path),
            "",
            (320, 320),
            detection_threshold,
            nms_threshold,
            5_000,
        )
        self.recognizer = cv.FaceRecognizerSF.create(str(recognizer_path), "")

    def detect(self, frame: NDArray[np.uint8]) -> list[Detection]:
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty")

        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(frame)
        if faces is None:
            return []
        return [Detection(np.asarray(row, dtype=np.float32).copy()) for row in faces]

    def align_and_embed(
        self,
        frame: NDArray[np.uint8],
        detection: Detection,
    ) -> NDArray[np.float32]:
        aligned_face = self.recognizer.alignCrop(frame, detection.raw)
        feature = self.recognizer.feature(aligned_face)
        return normalize_embedding(feature)
