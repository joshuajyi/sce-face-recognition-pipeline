# Final evaluation checklist

Use separate enrollment and test images. Only use photos of people who agreed
to participate.

## Functional checks

- [ ] One known face is accepted
- [ ] A second known face is accepted, if enrolled
- [ ] An unenrolled face is labeled `Unknown`
- [ ] No-face image exits without crashing
- [ ] Multiple faces receive separate boxes and labels
- [ ] Slight left and right head angles work
- [ ] Dimmer lighting is tested
- [ ] Webcam opens and closes with `q`
- [ ] Profile deletion works

## Measurements

- [ ] Record detection median and p95 latency
- [ ] Record embedding median and p95 latency
- [ ] Record total median and p95 latency
- [ ] Record pipeline FPS
- [ ] Record the threshold used
- [ ] Count known test photos accepted
- [ ] Count unknown test photos rejected

## Repository checks

- [ ] `uv sync --dev` works from a clean checkout
- [ ] `uv run sce-face download-models` succeeds
- [ ] `uv run pytest` passes
- [ ] `uv run ruff check .` passes
- [ ] No face photos or `gallery.npz` are tracked by Git
- [ ] README results contain real measurements
- [ ] GitHub repository is accessible to the reviewers

