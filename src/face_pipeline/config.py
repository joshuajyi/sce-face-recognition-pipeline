"""Shared defaults for paths and model settings."""

from pathlib import Path

DEFAULT_MODELS_DIR = Path("models")
DEFAULT_DETECTOR_PATH = DEFAULT_MODELS_DIR / "face_detection_yunet_2023mar.onnx"
DEFAULT_RECOGNIZER_PATH = DEFAULT_MODELS_DIR / "face_recognition_sface_2021dec.onnx"
DEFAULT_GALLERY_PATH = Path("data/gallery.npz")

DEFAULT_DETECTION_THRESHOLD = 0.85
DEFAULT_NMS_THRESHOLD = 0.30
DEFAULT_COSINE_THRESHOLD = 0.363
