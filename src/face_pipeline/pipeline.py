"""End-to-end orchestration and drawing for the recognition pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

from face_pipeline.gallery import Gallery, Match
from face_pipeline.vision import Detection, OpenCVFaceModels


@dataclass(frozen=True)
class FaceResult:
    detection: Detection
    match: Match


@dataclass(frozen=True)
class Timings:
    detection_ms: float
    embedding_ms: float
    matching_ms: float

    @property
    def total_ms(self) -> float:
        return self.detection_ms + self.embedding_ms + self.matching_ms

    @property
    def fps(self) -> float:
        return 1000.0 / self.total_ms if self.total_ms > 0 else 0.0


@dataclass(frozen=True)
class FrameAnalysis:
    faces: tuple[FaceResult, ...]
    timings: Timings


class RecognitionPipeline:
    def __init__(
        self,
        models: OpenCVFaceModels,
        gallery: Gallery,
        similarity_threshold: float,
    ) -> None:
        self.models = models
        self.gallery = gallery
        self.similarity_threshold = similarity_threshold

    def analyze(self, frame: NDArray[np.uint8]) -> FrameAnalysis:
        start = perf_counter()
        detections = self.models.detect(frame)
        after_detection = perf_counter()

        embeddings = [
            self.models.align_and_embed(frame, detection) for detection in detections
        ]
        after_embedding = perf_counter()

        matches = [
            self.gallery.match(embedding, self.similarity_threshold)
            for embedding in embeddings
        ]
        after_matching = perf_counter()

        return FrameAnalysis(
            faces=tuple(
                FaceResult(detection=detection, match=match)
                for detection, match in zip(detections, matches)
            ),
            timings=Timings(
                detection_ms=(after_detection - start) * 1000,
                embedding_ms=(after_embedding - after_detection) * 1000,
                matching_ms=(after_matching - after_embedding) * 1000,
            ),
        )

    @staticmethod
    def draw(frame: NDArray[np.uint8], analysis: FrameAnalysis) -> NDArray[np.uint8]:
        output = frame.copy()
        height, width = output.shape[:2]

        for result in analysis.faces:
            x, y, box_width, box_height = result.detection.box
            x1 = max(0, min(width - 1, x))
            y1 = max(0, min(height - 1, y))
            x2 = max(0, min(width - 1, x + box_width))
            y2 = max(0, min(height - 1, y + box_height))
            color = (60, 200, 60) if result.match.is_known else (50, 80, 230)

            cv.rectangle(output, (x1, y1), (x2, y2), color, 2)
            for landmark_x, landmark_y in result.detection.landmarks:
                cv.circle(output, (int(landmark_x), int(landmark_y)), 2, (255, 200, 0), -1)

            score = "n/a" if result.match.score < -0.5 else f"{result.match.score:.3f}"
            label = f"{result.match.label}  cosine={score}"
            text_y = max(22, y1 - 8)
            cv.putText(
                output,
                label,
                (x1, text_y),
                cv.FONT_HERSHEY_SIMPLEX,
                0.58,
                color,
                2,
                cv.LINE_AA,
            )

        timing_text = (
            f"faces={len(analysis.faces)}  detect={analysis.timings.detection_ms:.1f}ms  "
            f"embed={analysis.timings.embedding_ms:.1f}ms  "
            f"match={analysis.timings.matching_ms:.2f}ms  "
            f"pipeline_fps={analysis.timings.fps:.1f}"
        )
        cv.rectangle(output, (0, 0), (min(width, 860), 31), (20, 20, 20), -1)
        cv.putText(
            output,
            timing_text,
            (8, 21),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (245, 245, 245),
            1,
            cv.LINE_AA,
        )
        return output

