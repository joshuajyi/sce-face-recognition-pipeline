"""Command-line interface for downloading, enrolling, and recognizing faces."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

import cv2 as cv
import numpy as np

from face_pipeline.config import (
    DEFAULT_COSINE_THRESHOLD,
    DEFAULT_DETECTOR_PATH,
    DEFAULT_GALLERY_PATH,
    DEFAULT_RECOGNIZER_PATH,
)
from face_pipeline.download import download_models, sha256
from face_pipeline.gallery import Gallery
from face_pipeline.pipeline import RecognitionPipeline
from face_pipeline.vision import OpenCVFaceModels


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR_PATH)
    parser.add_argument("--recognizer", type=Path, default=DEFAULT_RECOGNIZER_PATH)


def add_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
    add_model_arguments(parser)
    parser.add_argument("--gallery", type=Path, default=DEFAULT_GALLERY_PATH)
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_COSINE_THRESHOLD,
        help="Cosine threshold for a known identity (default: %(default)s)",
    )


def build_models(args: argparse.Namespace) -> OpenCVFaceModels:
    return OpenCVFaceModels(args.detector, args.recognizer)


def read_image(path: Path) -> np.ndarray:
    image = cv.imread(str(path))
    if image is None:
        raise ValueError(
            f"Could not read {path}. Use a JPG or PNG file; HEIC support varies by OpenCV build."
        )
    return image


def command_download(args: argparse.Namespace) -> int:
    for path, digest in download_models(force=args.force):
        print(f"{path}  sha256={digest}")
    return 0


def command_enroll(args: argparse.Namespace) -> int:
    models = build_models(args)
    embeddings: list[np.ndarray] = []

    for image_path in args.images:
        image = read_image(image_path)
        detections = models.detect(image)
        if not detections:
            raise ValueError(f"No face detected in {image_path}")

        selected = max(
            detections,
            key=lambda detection: detection.box[2] * detection.box[3],
        )
        if len(detections) > 1:
            print(
                f"warning: {image_path} contains {len(detections)} faces; "
                "using the largest face"
            )
        embeddings.append(models.align_and_embed(image, selected))
        print(f"accepted {image_path}")

    gallery = Gallery.load(args.gallery)
    gallery.enroll(args.name, embeddings)
    gallery.save(args.gallery)
    print(
        f"enrolled {args.name!r} from {len(embeddings)} image(s); "
        f"gallery now contains {len(gallery)} profile(s)"
    )
    print(f"private embedding data saved to {args.gallery} (excluded from Git)")
    return 0


def print_analysis(analysis: object) -> None:
    faces = analysis.faces
    if not faces:
        print("no faces detected")
    for index, face in enumerate(faces, start=1):
        print(
            f"face {index}: label={face.match.label!r} "
            f"cosine={face.match.score:.3f} detection={face.detection.confidence:.3f}"
        )
    timing = analysis.timings
    print(
        f"detect={timing.detection_ms:.1f}ms embed={timing.embedding_ms:.1f}ms "
        f"match={timing.matching_ms:.2f}ms pipeline_fps={timing.fps:.1f}"
    )


def recognize_image(args: argparse.Namespace, pipeline: RecognitionPipeline) -> int:
    image = read_image(args.image)
    analysis = pipeline.analyze(image)
    annotated = pipeline.draw(image, analysis)
    output_path = args.output or Path("outputs") / f"recognized_{args.image.stem}.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv.imwrite(str(output_path), annotated):
        raise RuntimeError(f"Could not save {output_path}")
    print_analysis(analysis)
    print(f"annotated image saved to {output_path}")
    if args.show:
        cv.imshow("SCE Face Recognition", annotated)
        cv.waitKey(0)
        cv.destroyAllWindows()
    return 0


def recognize_camera(args: argparse.Namespace, pipeline: RecognitionPipeline) -> int:
    camera_index = 0 if args.camera is None else args.camera
    capture = cv.VideoCapture(camera_index)
    capture.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    if not capture.isOpened():
        raise RuntimeError(
            "Could not open the camera. Check the camera index and macOS camera permission."
        )

    print("camera running; press q or Escape to stop")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Camera stopped returning frames")
            analysis = pipeline.analyze(frame)
            annotated = pipeline.draw(frame, analysis)
            cv.imshow("SCE Face Recognition", annotated)
            key = cv.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv.destroyAllWindows()
    return 0


def command_recognize(args: argparse.Namespace) -> int:
    gallery = Gallery.load(args.gallery)
    if not gallery.names:
        print(
            "warning: gallery is empty; faces will be labeled Unknown. "
            "Run the enroll command first."
        )
    pipeline = RecognitionPipeline(build_models(args), gallery, args.threshold)
    if args.image is not None:
        return recognize_image(args, pipeline)
    return recognize_camera(args, pipeline)


def command_profiles(args: argparse.Namespace) -> int:
    gallery = Gallery.load(args.gallery)
    if args.delete is not None:
        if not gallery.delete(args.delete):
            raise ValueError(f"No profile named {args.delete!r}")
        gallery.save(args.gallery)
        print(f"deleted profile {args.delete!r}")
    elif args.clear:
        gallery.clear()
        gallery.save(args.gallery)
        print("deleted all profiles")
    else:
        if gallery.names:
            print("\n".join(gallery.names))
        else:
            print("gallery is empty")
    return 0


def percentile(values: list[float], percentile_value: float) -> float:
    return float(np.percentile(np.asarray(values), percentile_value))


def command_benchmark(args: argparse.Namespace) -> int:
    gallery = Gallery.load(args.gallery)
    pipeline = RecognitionPipeline(build_models(args), gallery, args.threshold)
    images = [(path, read_image(path)) for path in args.images]

    # One unmeasured pass warms model and memory allocations.
    for _, image in images:
        pipeline.analyze(image)

    detection: list[float] = []
    embedding: list[float] = []
    matching: list[float] = []
    totals: list[float] = []
    face_counts: list[int] = []
    for _ in range(args.repeat):
        for _, image in images:
            analysis = pipeline.analyze(image)
            detection.append(analysis.timings.detection_ms)
            embedding.append(analysis.timings.embedding_ms)
            matching.append(analysis.timings.matching_ms)
            totals.append(analysis.timings.total_ms)
            face_counts.append(len(analysis.faces))

    summary = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "opencv": cv.__version__,
            "numpy": np.__version__,
        },
        "configuration": {
            "images": [str(path) for path, _ in images],
            "repeat": args.repeat,
            "similarity_threshold": args.threshold,
            "detector": str(args.detector),
            "detector_sha256": sha256(args.detector),
            "recognizer": str(args.recognizer),
            "recognizer_sha256": sha256(args.recognizer),
            "gallery_profiles": list(gallery.names),
        },
        "results": {
            "runs": len(totals),
            "median_faces": statistics.median(face_counts),
            "detection_median_ms": statistics.median(detection),
            "detection_p95_ms": percentile(detection, 95),
            "embedding_median_ms": statistics.median(embedding),
            "embedding_p95_ms": percentile(embedding, 95),
            "matching_median_ms": statistics.median(matching),
            "matching_p95_ms": percentile(matching, 95),
            "total_median_ms": statistics.median(totals),
            "total_p95_ms": percentile(totals, 95),
            "pipeline_fps_from_median": 1000.0 / statistics.median(totals),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["results"], indent=2))
    print(f"benchmark details saved to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sce-face",
        description="Local YuNet + SFace facial-recognition pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download-models")
    download_parser.add_argument("--force", action="store_true")
    download_parser.set_defaults(handler=command_download)

    enroll_parser = subparsers.add_parser("enroll")
    enroll_parser.add_argument("name", help="Name stored with the profile")
    enroll_parser.add_argument("images", nargs="+", type=Path)
    add_model_arguments(enroll_parser)
    enroll_parser.add_argument("--gallery", type=Path, default=DEFAULT_GALLERY_PATH)
    enroll_parser.set_defaults(handler=command_enroll)

    recognize_parser = subparsers.add_parser("recognize")
    input_group = recognize_parser.add_mutually_exclusive_group()
    input_group.add_argument("--image", type=Path)
    input_group.add_argument("--camera", type=int)
    recognize_parser.add_argument("--output", type=Path)
    recognize_parser.add_argument("--show", action="store_true")
    add_pipeline_arguments(recognize_parser)
    recognize_parser.set_defaults(handler=command_recognize)

    profiles_parser = subparsers.add_parser("profiles")
    profiles_parser.add_argument("--gallery", type=Path, default=DEFAULT_GALLERY_PATH)
    profile_action = profiles_parser.add_mutually_exclusive_group()
    profile_action.add_argument("--delete", metavar="NAME")
    profile_action.add_argument("--clear", action="store_true")
    profiles_parser.set_defaults(handler=command_profiles)

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("images", nargs="+", type=Path)
    benchmark_parser.add_argument("--repeat", type=int, default=10)
    benchmark_parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/benchmark.json"),
    )
    add_pipeline_arguments(benchmark_parser)
    benchmark_parser.set_defaults(handler=command_benchmark)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "repeat", 1) < 1:
        parser.error("--repeat must be at least 1")
    try:
        return int(args.handler(args))
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

