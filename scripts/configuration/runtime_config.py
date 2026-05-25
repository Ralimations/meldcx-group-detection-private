#!/usr/bin/env python3
"""Source-specific runtime and ROI configuration helpers."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import fields
from pathlib import Path

from config import OpenVinoDefaults


PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "settings.db"

GLOBAL_ONLY_FIELDS: set[str] = {
    "show_body",
    "show_head",
    "show_merged_warnings",
    "device",
    "detect_body",
    "detect_head",
    "person_detection_conf",
    "head_detection_conf",
    "person_overlap_threshold",
    "head_overlap_threshold",
    "save_output_video",
    "save_session_artifacts",
    "output_video_fps",
    "show_perf",
    "verbose_logging",
    "mimic_live",
    "imgsz",
    "downscale_to_imgsz",
    "skip_frames",
    "output_processed_only",
    "show_body_confidence",
    "show_head_confidence",
    "show_face",
    "show_face_confidence",
    "show_ui_overlay",
    "label_font_scale",
    "label_thickness",
    "label_bg_alpha",
    "label_max_width_chars",
    "label_default_color",
    "label_segment_colors",
    "show_gender",
    "show_gender_confidence",
    "show_attributes",
    "show_body_par_raw_labels",
    "detect_gender",
    "gender_use_face_override",
    "demographics_min_conf",
    "gender_face_override_min_conf",
    "show_age",
    "show_age_confidence",
    "show_age_height_stats",
    "show_age_height_unlocked",
    "show_face_analysis_unlocked",
    "show_height_measurement_line",
    "show_pose_keypoints",
    "show_pose_indices",
    "pose_force_run",
    "pose_full_frame",
    "detect_age",
    "detect_face",
    "enable_face_analysis",
    "face_age_override",
    "face_age_senior_threshold",
    "face_detection_conf",
    "face_overlap_threshold",
    "height_min_samples_in_roi",
    "stable",
    "show_unconfirmed",
    "track_overlap_threshold",
    "min_hits",
    "max_misses",
    "track_max_draw_misses",
    "smooth",
    "track_min_conf",
    "track_reid_window",
    "track_reid_similarity_thresh",
    "track_reid_update_interval",
    "track_reid_backend",
    "track_reid_model",
    "track_reid_embedding_similarity_thresh",
    "track_reid_require_not_touching_frame_edge",
    "track_reid_min_quality_frames",
    "track_reid_identity_confirm_frames",
    "track_reid_use_spatial_gate",
    "track_reid_spatial_gate_scale",
    "use_doorway_monitor",
    "show_doorway_status",
    "room_presence_alert_seconds",
    "doorway_commit_frames",
    "doorway_missing_frames",
    "use_pose_visibility_gate",
    "pose_conf",
    "pose_overlap_threshold",
    "pose_keypoint_conf",
    "pose_min_visible_keypoints",
    "pose_min_torso_keypoints",
    "pose_min_lower_body_keypoints",
    "pose_min_lowest_keypoint_ratio",
    "group_detect",
    "group_max_relative_dist_check",
    "group_max_relative_dist_lock",
    "group_min_history",
    "group_idle_speed_thresh",
    "group_lock_threshold",
    "group_max_score",
    "show_carry_overlap_debug",
    "show_carry_status",
    "carry_overlap_score_threshold",
    "carry_overlap_lock_frames",
    "carry_overlap_release_frames",
    "testing_output_root",
    "nvr_record_enable",
    "nvr_segment_minutes",
    "nvr_retention_hours",
    "nvr_output_root",
    "group_log_enable",
    "show_roi_mask",
    "roi_transparency",
    "show_height_roi_mask",
    "height_roi_transparency",
    "no_show",
    "preview_width",
    "preview_height",
    "preview_window_mode",
    "turbo_mode",
    "input_mode",
    "source",
    "camera_index",
}


def _file_source_key(source_str: str) -> str:
    path = Path(source_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    normalized = Path(os.path.normpath(str(path)))
    project_root_norm = Path(os.path.normpath(str(PROJECT_ROOT)))
    try:
        relative = normalized.relative_to(project_root_norm)
    except ValueError:
        return f"file:{normalized}"
    return f"file:{relative.as_posix()}"


def init_runtime_db() -> None:
    con = sqlite3.connect(str(DB_PATH))
    con.execute(
        "CREATE TABLE IF NOT EXISTS source_runtime_settings ("
        "source_key TEXT PRIMARY KEY, "
        "config_json TEXT NOT NULL, "
        "updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
    )
    con.execute(
        "CREATE TABLE IF NOT EXISTS source_roi_settings ("
        "source_key TEXT PRIMARY KEY, "
        "roi_polygon_json TEXT NOT NULL, "
        "height_roi_polygon_json TEXT NOT NULL, "
        "updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
    )
    con.commit()
    con.close()


def source_key_for_runtime(source: str, *, mode: str = "source", camera_index: int = 0) -> str:
    source_str = str(source).strip().strip("'\"")
    if mode == "camera":
        return f"camera:{int(camera_index)}"
    if source_str.lower().startswith(("rtsp://", "http://", "https://")):
        return f"stream:{source_str}"
    return _file_source_key(source_str)


def apply_cfg_overrides(cfg: OpenVinoDefaults, overrides: dict) -> OpenVinoDefaults:
    overrides = {k: v for k, v in overrides.items() if k not in GLOBAL_ONLY_FIELDS}
    current = {f.name: getattr(cfg, f.name) for f in fields(OpenVinoDefaults)}
    current.update(overrides)
    kwargs = {}
    for field_def in fields(OpenVinoDefaults):
        name = field_def.name
        value = current.get(name, getattr(cfg, name))
        try:
            if name == "undistort_coeffs":
                if isinstance(value, str):
                    kwargs[name] = tuple(float(part.strip()) for part in value.split(",") if part.strip())
                else:
                    kwargs[name] = tuple(float(v) for v in value)
            elif name in {"roi_polygon", "height_roi_polygon", "doorway_roi_polygon"} and isinstance(value, (list, tuple)):
                kwargs[name] = tuple(tuple(float(coord) for coord in point) for point in value)
            elif field_def.type is bool:
                kwargs[name] = bool(value)
            elif field_def.type is int:
                kwargs[name] = int(value)
            elif field_def.type is float:
                kwargs[name] = float(value)
            else:
                kwargs[name] = value
        except (TypeError, ValueError):
            kwargs[name] = value
    return OpenVinoDefaults(**kwargs)


def apply_source_runtime_overrides(
    cfg: OpenVinoDefaults,
    *,
    source: str | None = None,
    mode: str | None = None,
    camera_index: int | None = None,
) -> OpenVinoDefaults:
    init_runtime_db()
    source_key = source_key_for_runtime(
        source if source is not None else getattr(cfg, "source", ""),
        mode=mode if mode is not None else str(getattr(cfg, "input_mode", "source")).strip().lower(),
        camera_index=int(camera_index if camera_index is not None else getattr(cfg, "camera_index", 0)),
    )
    con = sqlite3.connect(str(DB_PATH))
    row = con.execute("SELECT config_json FROM source_runtime_settings WHERE source_key = ?", (source_key,)).fetchone()
    con.close()
    if row is None:
        return cfg
    try:
        overrides = json.loads(row[0])
    except Exception:
        return cfg
    return apply_cfg_overrides(cfg, overrides)


def save_source_runtime_overrides(source_key: str, overrides: dict[str, object]) -> None:
    init_runtime_db()
    overrides = {k: v for k, v in overrides.items() if k not in GLOBAL_ONLY_FIELDS}
    con = sqlite3.connect(str(DB_PATH))
    row = con.execute(
        "SELECT config_json FROM source_runtime_settings WHERE source_key = ?",
        (source_key,),
    ).fetchone()
    current: dict[str, object] = {}
    if row is not None:
        try:
            current = json.loads(row[0])
        except Exception:
            current = {}
    current.update(overrides)
    con.execute(
        "INSERT INTO source_runtime_settings (source_key, config_json, updated_at) "
        "VALUES (?, ?, CURRENT_TIMESTAMP) "
        "ON CONFLICT(source_key) DO UPDATE SET "
        "config_json = excluded.config_json, "
        "updated_at = CURRENT_TIMESTAMP",
        (source_key, json.dumps(current, ensure_ascii=True)),
    )
    con.commit()
    con.close()


def get_source_initialization_status(source: str, *, mode: str = "source", camera_index: int = 0) -> dict[str, object]:
    source_key = source_key_for_runtime(source, mode=mode, camera_index=camera_index)
    con = sqlite3.connect(str(DB_PATH))
    runtime_ready = con.execute(
        "SELECT 1 FROM source_runtime_settings WHERE source_key = ?",
        (source_key,),
    ).fetchone() is not None
    roi_ready = con.execute(
        "SELECT 1 FROM source_roi_settings WHERE source_key = ?",
        (source_key,),
    ).fetchone() is not None
    undistort_ready = con.execute(
        "SELECT 1 FROM source_undistort_settings WHERE source_key = ?",
        (source_key,),
    ).fetchone() is not None
    con.close()
    missing = []
    if not runtime_ready:
        missing.append("runtime")
    if not roi_ready:
        missing.append("roi")
    if not undistort_ready:
        missing.append("undistort")
    return {"source_key": source_key, "ready": runtime_ready and roi_ready and undistort_ready, "missing": missing}


def ensure_source_initialized(source: str, *, mode: str = "source", camera_index: int = 0) -> None:
    source_str = str(source).strip().strip("'\"")
    if mode != "camera" and not source_str:
        raise SystemExit(
            "No source configured.\n"
            "What to do:\n"
            "  1. Choose the source you want to use.\n"
            "  2. Initialize it with:\n"
            "     .\\run.ps1 init-source --source <your-source>\n"
            "  3. Save that source's ROI and height ROI.\n"
            "  4. Save that source's undistortion values.\n"
            "  5. Run detection again."
        )
    status = get_source_initialization_status(source, mode=mode, camera_index=camera_index)
    if status["ready"]:
        return
    missing = ", ".join(status["missing"])
    raise SystemExit(
        f"Source '{status['source_key']}' is not initialized in settings.db.\n"
        f"Missing: {missing}.\n"
        "What to do before regular run:\n"
        f"  1. Set the source to '{source_str or status['source_key']}'.\n"
        "  2. Initialize missing rows with:\n"
        "     .\\run.ps1 init-source --source <your-source>\n"
        "  3. Start the preview and save ROI / height ROI for that source.\n"
        "  4. Tune and save undistortion with:\n"
        "     .\\run.ps1 undistort --source <your-source>\n"
        "  5. Run run_openvino_yolo.py again after setup is saved."
    )


def save_source_roi_settings(source_key: str, roi_polygon, height_roi_polygon) -> None:
    init_runtime_db()
    roi_tuple = tuple((float(p[0]), float(p[1])) for p in roi_polygon)
    height_roi_tuple = tuple((float(p[0]), float(p[1])) for p in height_roi_polygon)
    con = sqlite3.connect(str(DB_PATH))
    con.execute(
        "INSERT INTO source_roi_settings (source_key, roi_polygon_json, height_roi_polygon_json, updated_at) "
        "VALUES (?, ?, ?, CURRENT_TIMESTAMP) "
        "ON CONFLICT(source_key) DO UPDATE SET "
        "roi_polygon_json = excluded.roi_polygon_json, "
        "height_roi_polygon_json = excluded.height_roi_polygon_json, "
        "updated_at = CURRENT_TIMESTAMP",
        (
            source_key,
            json.dumps(roi_tuple, ensure_ascii=True),
            json.dumps(height_roi_tuple, ensure_ascii=True),
        ),
    )
    con.commit()
    con.close()


def load_source_roi_settings(source_key: str):
    init_runtime_db()
    con = sqlite3.connect(str(DB_PATH))
    row = con.execute(
        "SELECT roi_polygon_json, height_roi_polygon_json FROM source_roi_settings WHERE source_key = ?",
        (source_key,),
    ).fetchone()
    con.close()
    if row is None:
        return None
    return (
        tuple(tuple(float(coord) for coord in point) for point in json.loads(row[0])),
        tuple(tuple(float(coord) for coord in point) for point in json.loads(row[1])),
    )


def copy_source_runtime_settings(from_source_key: str, to_source_key: str) -> None:
    init_runtime_db()
    con = sqlite3.connect(str(DB_PATH))
    row = con.execute(
        "SELECT config_json FROM source_runtime_settings WHERE source_key = ?",
        (from_source_key,),
    ).fetchone()
    if row is None:
        con.close()
        raise ValueError(f"Missing runtime config for {from_source_key}")
    con.execute(
        "INSERT INTO source_runtime_settings (source_key, config_json, updated_at) "
        "VALUES (?, ?, CURRENT_TIMESTAMP) "
        "ON CONFLICT(source_key) DO UPDATE SET "
        "config_json = excluded.config_json, "
        "updated_at = CURRENT_TIMESTAMP",
        (to_source_key, row[0]),
    )
    con.commit()
    con.close()


def copy_source_roi_settings(from_source_key: str, to_source_key: str) -> None:
    init_runtime_db()
    con = sqlite3.connect(str(DB_PATH))
    row = con.execute(
        "SELECT roi_polygon_json, height_roi_polygon_json FROM source_roi_settings WHERE source_key = ?",
        (from_source_key,),
    ).fetchone()
    if row is None:
        con.close()
        raise ValueError(f"Missing ROI config for {from_source_key}")
    con.execute(
        "INSERT INTO source_roi_settings (source_key, roi_polygon_json, height_roi_polygon_json, updated_at) "
        "VALUES (?, ?, ?, CURRENT_TIMESTAMP) "
        "ON CONFLICT(source_key) DO UPDATE SET "
        "roi_polygon_json = excluded.roi_polygon_json, "
        "height_roi_polygon_json = excluded.height_roi_polygon_json, "
        "updated_at = CURRENT_TIMESTAMP",
        (to_source_key, row[0], row[1]),
    )
    con.commit()
    con.close()

init_runtime_db()
