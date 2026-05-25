#!/usr/bin/env python3
"""Initialize source-specific runtime, ROI, and undistortion config rows in settings.db."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config import OPENVINO_DEFAULTS
from scripts.configuration.source_runtime_defaults import SourceRuntimeDefaults
from scripts.configuration.image_undistorter import save_source_undistort_settings
from scripts.configuration.runtime_config import init_runtime_db, source_key_for_runtime

DB_PATH = PROJECT_ROOT / "settings.db"
FULL_FRAME_ROI = (
    (0.0, 0.0),
    (1.0, 0.0),
    (1.0, 1.0),
    (0.0, 1.0),
)
def _init_db() -> None:
    init_runtime_db()


def _default_runtime_payload() -> dict[str, object]:
    payload: dict[str, object] = {}
    for field_def in fields(SourceRuntimeDefaults):
        name = field_def.name
        if name in {"roi_polygon", "height_roi_polygon", "undistort_coeffs"}:
            continue
        if name.startswith("undistort_"):
            continue
        payload[name] = getattr(OPENVINO_DEFAULTS, name)
    return payload


def initialize_source(source: str, *, mode: str, camera_index: int, force: bool) -> str:
    _init_db()
    source_key = source_key_for_runtime(source, mode=mode, camera_index=camera_index)
    con = sqlite3.connect(str(DB_PATH))

    runtime_exists = con.execute(
        "SELECT 1 FROM source_runtime_settings WHERE source_key = ?",
        (source_key,),
    ).fetchone() is not None
    roi_exists = con.execute(
        "SELECT 1 FROM source_roi_settings WHERE source_key = ?",
        (source_key,),
    ).fetchone() is not None

    if force or not runtime_exists:
        con.execute(
            "INSERT INTO source_runtime_settings (source_key, config_json, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP) "
            "ON CONFLICT(source_key) DO UPDATE SET config_json = excluded.config_json, updated_at = CURRENT_TIMESTAMP",
            (source_key, json.dumps(_default_runtime_payload())),
        )

    if force or not roi_exists:
        con.execute(
            "INSERT INTO source_roi_settings (source_key, roi_polygon_json, height_roi_polygon_json, updated_at) "
            "VALUES (?, ?, ?, CURRENT_TIMESTAMP) "
            "ON CONFLICT(source_key) DO UPDATE SET "
            "roi_polygon_json = excluded.roi_polygon_json, "
            "height_roi_polygon_json = excluded.height_roi_polygon_json, "
            "updated_at = CURRENT_TIMESTAMP",
            (
                source_key,
                json.dumps(FULL_FRAME_ROI, ensure_ascii=True),
                json.dumps(FULL_FRAME_ROI, ensure_ascii=True),
            ),
        )

    con.commit()
    con.close()

    save_source_undistort_settings(
        source=source,
        enable=False,
        model="fisheye",
        coeffs=(),
        fx=0.0,
        fy=0.0,
        balance=0.0,
        alpha=0.0,
        mode=mode,
        camera_index=camera_index,
    )
    return source_key


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize a source config in settings.db")
    parser.add_argument("--source", default="", help="Video file path or RTSP/HTTP stream URL")
    parser.add_argument("--mode", choices=("source", "camera"), default="source", help="Source mode")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index when --mode camera")
    parser.add_argument("--force", action="store_true", help="Overwrite existing source rows with starter values")
    args = parser.parse_args()

    if args.mode != "camera" and not str(args.source).strip():
        raise SystemExit("Provide --source for source mode.")

    source_value = args.source if args.mode != "camera" else f"camera:{args.camera_index}"
    source_key = initialize_source(
        source=source_value if args.mode == "camera" else args.source,
        mode=args.mode,
        camera_index=args.camera_index,
        force=bool(args.force),
    )

    print(f"Initialized source config for {source_key}")
    print("Next steps:")
    print("  1. Run .\\run.ps1 detect and adjust ROI/height ROI in the preview UI.")
    print("  2. Run .\\run.ps1 undistort --source <your-source> to tune undistortion.")
    print("  3. Run .\\run.ps1 detect again for regular detection.")


if __name__ == "__main__":
    main()
