# AI-assisted development

The assignment explicitly encouraged using AI while building the project. I
used ChatGPT/Codex as a development assistant instead of treating its output as
automatically correct.

## What AI helped with

- Breaking the assignment into detection, alignment, embedding, matching, and
  evaluation stages
- Comparing DeepFace, InsightFace, and OpenCV's YuNet/SFace pipeline
- Finding the relevant official model documentation and license information
- Creating an initial Python package and command-line structure
- Suggesting edge cases and unit tests
- Debugging formatting, type, and command-line issues
- Reviewing the README for missing setup and reproduction steps

## How I checked the suggestions

- I used the official OpenCV tutorial and OpenCV Zoo model pages as the main
  technical references.
- Downloaded model files are checked against expected SHA-256 hashes.
- Ruff checks the Python source for common problems.
- pytest checks vector normalization, cosine matching, gallery storage, and
  `Unknown` rejection.
- I manually test enrollment, a known face, an unknown face, multiple faces,
  no-face input, and the webcam before submitting.
- I record measured MacBook results instead of repeating published benchmarks
  as if they were my own.

## What I rejected or limited

- I did not build a React dashboard because the assignment says appearance is
  not the focus.
- I did not use a one-call DeepFace verification pipeline because it would hide
  stages I need to understand.
- I did not claim that ONNX automatically makes a model FPGA-compatible.
- I did not add Vitis, Docker, Tailscale, or cloud storage to the laptop
  take-home.
- I do not send private face photos or generated embeddings to an AI service.

I reviewed the submitted files and can explain how data moves through each
part of the pipeline.

