#!/usr/bin/env python3
"""Copy DB-backed source configuration from one source to another."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.configuration.image_undistorter import copy_source_undistort_settings
from scripts.configuration.runtime_config import (
    copy_source_roi_settings,
    copy_source_runtime_settings,
    get_source_initialization_status,
    source_key_for_runtime,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy one source config to another in settings.db")
    parser.add_argument("--from-source", required=True, help="Existing source path/URL to copy from")
    parser.add_argument("--to-source", required=True, help="Destination source path/URL to copy to")
    parser.add_argument("--from-mode", choices=("source", "camera"), default="source", help="Input mode for source to copy from")
    parser.add_argument("--to-mode", choices=("source", "camera"), default="source", help="Input mode for source to copy to")
    parser.add_argument("--from-camera-index", type=int, default=0, help="Camera index when --from-mode camera")
    parser.add_argument("--to-camera-index", type=int, default=0, help="Camera index when --to-mode camera")
    args = parser.parse_args()

    from_status = get_source_initialization_status(
        args.from_source,
        mode=args.from_mode,
        camera_index=args.from_camera_index,
    )
    if not from_status["ready"]:
        missing = ", ".join(from_status["missing"])
        raise SystemExit(
            f"Source '{from_status['source_key']}' is not fully initialized in settings.db.\n"
            f"Missing: {missing}."
        )

    from_source_key = source_key_for_runtime(
        args.from_source,
        mode=args.from_mode,
        camera_index=args.from_camera_index,
    )
    to_source_key = source_key_for_runtime(
        args.to_source,
        mode=args.to_mode,
        camera_index=args.to_camera_index,
    )

    copy_source_runtime_settings(from_source_key, to_source_key)
    copy_source_roi_settings(from_source_key, to_source_key)
    copy_source_undistort_settings(from_source_key, to_source_key)

    print(f"Copied configuration from {from_source_key}")
    print(f"                        to {to_source_key}")
    print("Copied sections: runtime, roi, undistort")


if __name__ == "__main__":
    main()
