# Home Security Hub

Starter project scaffold for a local-first home security pipeline with face/object perception, rule-based decisions, and pluggable outputs.

## Quick start (Conda)

```bash
conda env create -f environment.yml
conda activate hshub
cp .env.example .env
python -m hshub.app
```

## Update environment after dependency changes

```bash
conda env update -f environment.yml --prune
```

## Tune face knobs with graphs

The app now logs face/frame/flush telemetry to `artifacts/tuning_events.csv` by default.

Run the pipeline:

```bash
python -m hshub.app
```

Override any knob per run (example):

```bash
set HSHUB_MATCH_THRESHOLD=0.55
set HSHUB_MARGIN=0.05
set HSHUB_MIN_BLUR_LAPLACIAN_VAR=90
set HSHUB_BLUR_GAUSSIAN_KERNEL=5
set HSHUB_BLUR_GAUSSIAN_SIGMA=0
python -m hshub.app
```

Generate graphs:

```bash
python scripts/plot_tuning_graphs.py --log artifacts/tuning_events.csv --out-dir artifacts/tuning_plots
```

Generated files:
- `artifacts/tuning_plots/detection_knobs.png`
- `artifacts/tuning_plots/matching_knobs.png`
- `artifacts/tuning_plots/runtime_knobs.png`

Useful env knobs for one-by-one tuning:
- `HSHUB_MATCH_THRESHOLD`
- `HSHUB_MARGIN`
- `HSHUB_COMMIT_EVERY_N`
- `HSHUB_COMMIT_EVERY_SEC`
- `HSHUB_DETECT_EVERY_N`
- `HSHUB_DETECT_FRAME_SCALE`
- `HSHUB_DET_SIZE` (format: `1280x1280`)
- `HSHUB_DETECTION_SCORE_THRESHOLD`
- `HSHUB_MIN_FACE_SIZE_PX`
- `HSHUB_MAX_FACE_ASPECT_RATIO`
- `HSHUB_MIN_FACE_AREA_RATIO`
- `HSHUB_CENTROID_ALPHA`
- `HSHUB_NEW_PERSON_SAVE_DELAY_SEC`
- `HSHUB_PENDING_NEW_SIM_THRESHOLD`
- `HSHUB_PENDING_NEW_STALE_SEC`
- `HSHUB_MIN_BLUR_LAPLACIAN_VAR`
- `HSHUB_BLUR_GAUSSIAN_KERNEL`
- `HSHUB_BLUR_GAUSSIAN_SIGMA`
- `HSHUB_TUNING_LOG_PATH` (set to `none` to disable logging)

## Layout

See the directory structure in this repository for module boundaries.
