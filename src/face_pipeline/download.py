"""Download the two versioned OpenCV Zoo model files used by the project."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from face_pipeline.config import DEFAULT_DETECTOR_PATH, DEFAULT_RECOGNIZER_PATH


@dataclass(frozen=True)
class ModelDownload:
    destination: Path
    url: str
    minimum_bytes: int


MODEL_DOWNLOADS = (
    ModelDownload(
        destination=DEFAULT_DETECTOR_PATH,
        url=(
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_detection_yunet/face_detection_yunet_2023mar.onnx"
        ),
        minimum_bytes=100_000,
    ),
    ModelDownload(
        destination=DEFAULT_RECOGNIZER_PATH,
        url=(
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_recognition_sface/face_recognition_sface_2021dec.onnx"
        ),
        minimum_bytes=10_000_000,
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_models(force: bool = False) -> list[tuple[Path, str]]:
    """Download missing models and return each path with its SHA-256 hash."""

    downloaded: list[tuple[Path, str]] = []
    for spec in MODEL_DOWNLOADS:
        path = spec.destination
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists() and path.stat().st_size >= spec.minimum_bytes and not force:
            downloaded.append((path, sha256(path)))
            continue

        temporary = path.with_suffix(path.suffix + ".part")
        request = urllib.request.Request(
            spec.url,
            headers={"User-Agent": "sce-face-pipeline/0.1"},
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                content_type = response.headers.get("Content-Type", "")
                if "text/html" in content_type:
                    raise RuntimeError(f"Model URL returned HTML instead of a model: {spec.url}")
                with temporary.open("wb") as output:
                    shutil.copyfileobj(response, output)

            if temporary.stat().st_size < spec.minimum_bytes:
                raise RuntimeError(
                    f"Downloaded file is unexpectedly small: {temporary.stat().st_size} bytes"
                )
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()

        downloaded.append((path, sha256(path)))
    return downloaded

