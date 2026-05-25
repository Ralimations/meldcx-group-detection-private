# GroupDetection

Person and group detection workflows using OpenVINO.

## Documentation

For a detailed technical overview of the models, algorithms, and social grouping logic, see [model_documentation.md](model_documentation.md).
For a full dashboard guide and implementation/progress overview, see [docs/dashboard_documentation.md](docs/dashboard_documentation.md).
For the current PAR merge state and next-step validation, see [docs/par_runtime_status.md](docs/par_runtime_status.md) and [docs/par_runtime_smoke_test.md](docs/par_runtime_smoke_test.md).

## Main Entry Point

Use `run_openvino_yolo.py` (project root) as the primary runner.
This runner is purely config-driven via `config.py` (no CLI overrides).

## Project Structure

```text
7_GroupDetection/
  config.py
  models/
    person/
      best.xml
      best.bin
      labels.txt
  run_openvino_yolo.py
  run_tests.ps1
  scripts/
    person_detector.py
    group_detector.py
    perspective_detector.py
  media/
  output/
  model_documentation.md
```

## Setup

```powershell
python -m pip install -U openvino ultralytics opencv-python numpy
```

Run commands from project root (`7_GroupDetection`).

## Run

1. Edit settings in `config.py`:
- Input mode (`input_mode: "source", "image", "camera", "batch"`)
- Perspective (`perspective: "AUTO", "LEVELED", "TOP-DOWN"`)
- Detection/tracking/group parameters
- Rendering flags (`show_person_boxes`, `show_person_confidence`)
- Output behavior (`out`, `auto_output_name`)

2. Start:

```powershell
python .\run_openvino_yolo.py
```

## Dashboard Validation

For the rebuilt dashboard, use this quick validation flow:

1. Smoke-check the backend bootstrap:

```powershell
py .\scripts\smoke_dashboard_backend.py
```

2. Start the dashboard launcher:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_dashboard.ps1
```

3. Verify manually:
- startup progress appears while backend/frontend boot
- sensor list loads without hanging
- selecting a sensor opens the workspace
- engine start/stop/reset/pause show pending states
- exiting a scene returns to the sensor list and stops the engine

### Batch Mode & Session Folders
When setting `input_mode = "batch"` and populating the `batch_sources` list in `config.py`, the system organizes the run cleanly. It automatically provisions an isolated timestamped folder under `output/[YYYYMMDD_HHMMSS] batch_session` containing all individual output `.mp4` videos along with a comprehensive markdown summary report showing total tracking metrics.

### AUTO Perspective
When `perspective = "AUTO"`, the system loads the universal model but intelligently captures the camera view by measuring the aspect-ratio heuristics of the bounding boxes during the initial frames of the video or camera sequence. Once enough history is collected, it permanently locks the estimated perspective for the remainder of the session to ensure stability.

## Quality / Balanced / Speed

In `config.py`, set both `imgsz` and `skip_frames`:
- Quality: `imgsz=640`, `skip_frames=1`
- Balanced: `imgsz=416`, `skip_frames=2`
- Speed: `imgsz=320`, `skip_frames=3`

## OOP Configuration Engine

All runtime defaults are injected via the `OpenVinoDefaults` data-class defined in `config.py`.
`run_openvino_yolo.py`, `scripts/person_detector.py`, `scripts/group_detector.py`, and `scripts/perspective_detector.py` initialize natively via this unified object without hardcoded math or sprawling arguments.

## Python Cache

Python bytecode cache is centralized outside the repo tree to prevent clutter:
- Windows: `<repo-drive>\MeldCX\pycache\7_GroupDetection` (example: `D:\MeldCX\pycache\7_GroupDetection`)

`run_openvino_yolo.py` sets this automatically, and `run_tests.ps1` exports `PYTHONPYCACHEPREFIX`.
