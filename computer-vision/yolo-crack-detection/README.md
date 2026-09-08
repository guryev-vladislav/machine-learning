# YOLO Crack Detection

A compact computer vision pipeline for classifying images and video frames as `crack` or `no_crack` with Ultralytics YOLO classification models.

## Highlights

- Reproducible dataset preparation and class balancing
- YAML-based experiment configuration
- CPU smoke test with a synthetic dataset
- Training metrics, confusion matrices, reports, and run manifests
- Video inference with annotated output and frame-level reports
- Structured console and JSON logging through [python-logger](https://github.com/guryev-vladislav/python-logger)

## Project Layout

```text
configs/              Experiment configurations
src/data_preparation  Dataset and video preparation
src/models            YOLO model adapter
src/inference         Video inference
src/utils              Configuration, metrics, and visualization
scripts/              Dataset utilities and smoke-test tools
tests/                Unit tests
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Quick Start

Run a complete local smoke test with one command:

```bash
python main.py \
  --config configs/quickstart.yaml \
  --smoke-test \
  --run-name quickstart
```

The command creates a small synthetic dataset, trains `yolov8n-cls` for one epoch on CPU, and stores results under `outputs/quickstart_<timestamp>/`.

## Training

For a real experiment, place the DeepCrack and SDNET datasets at the paths configured in `configs/train_config.yaml`, then run:

```bash
python main.py \
  --config configs/train_config.yaml \
  --run-name baseline
```

CLI options override values from YAML:

```bash
python main.py \
  --config configs/train_config.yaml \
  --epochs 20 \
  --batch-size 8 \
  --device cpu \
  --run-name baseline-cpu
```

Important settings include `epochs`, `batch_size`, `imgsz`, `device`, `workers`, dataset paths, and confidence thresholds.

## Video Inference

```bash
python -m src.inference.video_processor \
  --video data/videos/input.mp4 \
  --model outputs/<run_name>/best_model.pt \
  --output outputs/result.mp4
```

The processor writes an annotated video and a text report with frame-level statistics.

## Logging

Console logging is enabled by default. JSON logging and debug output can be enabled with environment variables:

```bash
LOG_FILE=outputs/run.json \
LOGGER_MIN_LEVEL=DEBUG \
python main.py --config configs/quickstart.yaml --smoke-test
```

OpenTelemetry tracing is optional:

```bash
LOGGER_TRACE_ENABLED=true \
OTEL_EXPORTER_OTLP_ENDPOINT=localhost:4317 \
python main.py --config configs/quickstart.yaml --smoke-test
```

## Validation

```bash
python -m unittest discover -s tests -p 'test_*.py'
python -m compileall -q src main.py train.py scripts tests
```

## Scope

This project performs image-level classification. It does not localize cracks with bounding boxes or produce pixel-level segmentation masks. Model quality must be validated on representative data from the target surface and environment.
