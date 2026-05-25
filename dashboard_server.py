#!/usr/bin/env python3
"""
dashboard_server.py
Flask-based backend for the Group Detection dashboard.

Endpoints (all under http://localhost:8765):
  GET  /video_feed              – MJPEG stream of annotated frames
  GET  /api/config              – current config values + param metadata
  POST /api/config              – update one or more config values
  GET  /api/engine/status       – {running, paused, people, groups, fps}
  POST /api/engine/start        – start the engine thread
  POST /api/engine/stop         – stop the engine thread
  POST /api/engine/pause        – toggle pause; returns {paused}
  POST /api/engine/run_default  – alias for start
  GET  /api/logs                – {logs: [{time, msg}]}
  GET  /api/videos              – {videos:[...], current:"..."}
  POST /api/switch              – {mode, video|camera} – hot-swap input
  POST /api/reset               – restart engine with current config
  POST /api/defaults            – reset config to OPENVINO_DEFAULTS; returns {values}
  GET  /api/presets             – {presets:[...]}
  POST /api/save                – {name} – save current config as preset
  POST /api/load                – {name} – load preset; returns {values}
"""

from __future__ import annotations

import sys
from pathlib import Path

from flask import Flask
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Bootstrap: locate the project root so imports work regardless of CWD
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard_backend.config_store import DEFAULTS_DICT
from dashboard_backend.engine_manager import EngineManager
from dashboard_backend.persistence import init_db
from dashboard_backend.routes import register_routes

# ---------------------------------------------------------------------------
# Parameter metadata – drives the Tuning UI sliders / toggles
# ---------------------------------------------------------------------------
PARAM_META: dict[str, dict] = {
    # ── Detection ──
    "person_detection_conf": {"cat": "Detection", "type": "float", "min": 0.1, "max": 1.0, "step": 0.01,  "desc": "Person detection confidence threshold"},
    "person_overlap_threshold": {"cat": "Detection", "type": "float", "min": 0.1, "max": 1.0, "step": 0.01,  "desc": "Person detector NMS IoU threshold"},
    "imgsz":              {"cat": "Detection", "type": "int",   "min": 320, "max": 1280,"step": 32,    "desc": "Inference image size (0=native)", "needs_restart": True},
    "skip_frames":        {"cat": "Detection", "type": "int",   "min": 0,   "max": 10,  "step": 1,     "desc": "Process every Nth frame (0=all)"},
    "downscale_to_imgsz": {"cat": "Detection", "type": "bool",                                          "desc": "Downscale input to imgsz before detection"},
    "detect_body":        {"cat": "Detection", "type": "bool",                                          "desc": "Detect body class", "needs_restart": True},
    "detect_head":        {"cat": "Detection", "type": "bool",                                          "desc": "Detect head class",  "needs_restart": True},
    "head_detection_conf": {"cat": "Detection", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01, "desc": "Head detection confidence threshold"},
    "head_overlap_threshold": {"cat": "Detection", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01, "desc": "Head detector NMS IoU threshold"},
    "show_merged_warnings": {"cat": "Detection", "type": "bool", "advanced": True,                      "desc": "Show merged-group warnings in overlays"},

    # ── Tracking ──
    "stable":             {"cat": "Tracking",  "type": "bool",                                          "desc": "Enable stable person tracker"},
    "track_overlap_threshold": {"cat": "Tracking",  "type": "float", "min": 0.1, "max": 1.0, "step": 0.01,  "desc": "Tracker IoU match threshold"},
    "min_hits":           {"cat": "Tracking",  "type": "int",   "min": 1,   "max": 20,  "step": 1,     "desc": "Min hits to confirm track"},
    "max_misses":         {"cat": "Tracking",  "type": "int",   "min": 1,   "max": 120, "step": 1,     "desc": "Max misses before track is dropped"},
    "smooth":             {"cat": "Tracking",  "type": "float", "min": 0.0, "max": 1.0, "step": 0.05,  "desc": "Box smoothing EMA alpha"},
    "show_unconfirmed":   {"cat": "Tracking",  "type": "bool",                                          "desc": "Show unconfirmed tracks"},
    "track_min_conf":     {"cat": "Tracking",  "type": "float", "min": 0.0, "max": 1.0, "step": 0.01,  "desc": "Minimum confidence for a track to be kept", "advanced": True},
    "track_max_draw_misses": {"cat": "Tracking","type": "int",  "min": 0,   "max": 120, "step": 1,     "desc": "How long to keep drawing a missed track", "advanced": True},
    "track_reid_window":  {"cat": "Tracking",  "type": "int",   "min": 1,   "max": 1000,"step": 1,     "desc": "Track memory window for ReID matching", "advanced": True},
    "track_reid_similarity_thresh": {"cat": "Tracking", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01, "desc": "Similarity threshold for ReID recovery", "advanced": True},
    "track_reid_update_interval": {"cat": "Tracking", "type": "int", "min": 1, "max": 60, "step": 1,   "desc": "Frames between ReID feature updates", "advanced": True},
    "track_reid_backend": {"cat": "Tracking", "type": "select", "options": "off,histogram,openvino,hybrid", "desc": "ReID backend used for identity recovery", "advanced": True},
    "track_reid_model": {"cat": "Tracking", "type": "str",                                          "desc": "Path to the OpenVINO ReID model", "advanced": True},
    "track_reid_embedding_similarity_thresh": {"cat": "Tracking", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01, "desc": "Embedding similarity threshold for OpenVINO ReID", "advanced": True},
    "track_reid_require_not_touching_frame_edge": {"cat": "Tracking", "type": "bool",                "desc": "Require a clean in-frame crop before collecting ReID identity evidence", "advanced": True},
    "track_reid_min_quality_frames": {"cat": "Tracking", "type": "int", "min": 1, "max": 30, "step": 1, "desc": "Quality frames required before OpenVINO ReID starts confirming identity", "advanced": True},
    "track_reid_identity_confirm_frames": {"cat": "Tracking", "type": "int", "min": 1, "max": 30, "step": 1, "desc": "Frames required before a ReID identity candidate is accepted", "advanced": True},
    "track_reid_use_spatial_gate": {"cat": "Tracking", "type": "bool",                               "desc": "Restrict ReID identity recovery using a spatial gate", "advanced": True},
    "track_reid_spatial_gate_scale": {"cat": "Tracking", "type": "float", "min": 0.5, "max": 10.0, "step": 0.1, "desc": "Size multiplier for the ReID spatial gate", "advanced": True},

    # ── Display ──
    "show_body":          {"cat": "Display",   "type": "bool",                                          "desc": "Draw body bounding boxes"},
    "show_body_confidence": {"cat": "Display", "type": "bool",                                          "desc": "Show body detection confidence"},
    "show_head":          {"cat": "Display",   "type": "bool",                                          "desc": "Draw head bounding boxes"},
    "show_head_confidence": {"cat": "Display", "type": "bool",                                          "desc": "Show head detection confidence"},
    "show_face":          {"cat": "Display",   "type": "bool",                                          "desc": "Draw detected face boxes"},
    "show_face_confidence": {"cat": "Display", "type": "bool",                                          "desc": "Show face detection confidence"},
    "show_ui_overlay":    {"cat": "Display",   "type": "bool",                                          "desc": "Show HUD overlay (counts, FPS)"},
    "no_show":            {"cat": "Display",   "type": "bool",                                          "desc": "Suppress CV2 preview window"},
    "preview_width":      {"cat": "Display",   "type": "int",   "min": 320, "max": 1920, "step": 8,     "desc": "Stream width (px)"},
    "preview_height":     {"cat": "Display",   "type": "int",   "min": 180, "max": 1080, "step": 8,     "desc": "Stream height (px)"},
    "label_font_scale":   {"cat": "Display",   "type": "float", "min": 0.8, "max": 2.5, "step": 0.05,   "desc": "Detection label text size multiplier"},
    "label_thickness":    {"cat": "Display",   "type": "int",   "min": 1, "max": 5, "step": 1,          "desc": "Detection label outline thickness"},
    "label_bg_alpha":     {"cat": "Display",   "type": "float", "min": 0.0, "max": 1.0, "step": 0.05,   "desc": "Detection label background opacity"},
    "label_max_width_chars": {"cat": "Display", "type": "int", "min": 16, "max": 48, "step": 1,         "desc": "Max characters per detection label line before wrapping"},
    "label_default_color": {"cat": "Display",  "type": "str",                                          "desc": "Default label text color in #RRGGBB"},
    "label_segment_colors": {"cat": "Display", "type": "str",                                          "desc": "Comma-separated label prefix colors, e.g. Male=#4DA6FF"},

    # ── Detection ROI ──
    "use_roi":            {"cat": "Detection ROI", "type": "bool",                                      "desc": "Enable region-of-interest filtering"},
    "roi_polygon":        {"cat": "Detection ROI", "type": "str",                                       "desc": "Detection ROI polygon as JSON points"},
    "show_roi_mask":      {"cat": "Detection ROI", "type": "bool",                                      "desc": "Show detection ROI overlay"},
    "roi_transparency":   {"cat": "Detection ROI", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01, "desc": "Detection ROI overlay transparency"},

    # ── Height ROI ──
    "height_roi_polygon": {"cat": "Height ROI","type": "str",                                           "desc": "Height ROI polygon as JSON points"},
    "show_height_roi_mask": {"cat": "Height ROI", "type": "bool",                                       "desc": "Show height ROI overlay"},
    "height_roi_transparency": {"cat": "Height ROI", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01, "desc": "Height ROI overlay transparency"},
    "age_adult_threshold_px": {"cat": "Height ROI", "type": "float", "min": 50.0, "max": 400.0, "step": 0.1, "desc": "Minor/adult height cutoff in 640px-normalized pixels"},
    "height_min_samples_in_roi": {"cat": "Height ROI", "type": "int", "min": 1, "max": 60, "step": 1, "desc": "Minimum valid in-ROI samples before height can finalize"},

    # ── Doorway ROI ──
    "doorway_roi_polygon": {"cat": "Doorway ROI", "type": "str",                                       "desc": "Doorway ROI polygon as JSON points"},

    # ── Group Detection ──
    "group_detect":       {"cat": "Groups",    "type": "bool",                                          "desc": "Enable group detection"},
    "group_min_history":  {"cat": "Groups",    "type": "int",   "min": 1,   "max": 60,  "step": 1,     "desc": "Min frames before group forms"},
    "group_max_relative_dist_check": {"cat": "Groups", "type": "float", "min": 0.1, "max": 3.0, "step": 0.05, "desc": "Max relative dist (check)"},
    "group_max_relative_dist_lock":  {"cat": "Groups", "type": "float", "min": 0.05,"max": 1.0, "step": 0.05, "desc": "Max relative dist (lock)"},
    "group_history_len":  {"cat": "Groups",    "type": "int",   "min": 1,   "max": 120, "step": 1,     "desc": "History length used by group scoring", "advanced": True},
    "group_idle_speed_thresh": {"cat": "Groups", "type": "float", "min": 0.0, "max": 10.0, "step": 0.1, "desc": "Speed threshold below which a person counts as idle", "advanced": True},
    "group_lock_threshold": {"cat": "Groups",  "type": "float", "min": 0.0, "max": 5.0, "step": 0.05,  "desc": "Score required before a group locks", "advanced": True},
    "group_max_score":    {"cat": "Groups",    "type": "float", "min": 0.1, "max": 5.0, "step": 0.05,  "desc": "Maximum allowed persistent group score", "advanced": True},
    "group_pair_angle_history_len": {"cat": "Groups", "type": "int", "min": 1, "max": 60, "step": 1,   "desc": "Angle history length for pair stability", "advanced": True},
    "group_merged_warn_interval": {"cat": "Groups", "type": "int", "min": 1, "max": 120, "step": 1,    "desc": "Frames between merged-group warnings", "advanced": True},
    "group_origin_gap_threshold": {"cat": "Groups", "type": "float", "min": 0.0, "max": 2000.0, "step": 1.0, "desc": "Origin-gap threshold for breaking weak groups", "advanced": True},

    # ── Demographics ──
    "detect_gender":      {"cat": "Demographics", "type": "bool",                                       "desc": "Enable gender labels from body PAR with optional face override", "needs_restart": True},
    "detect_age":         {"cat": "Demographics", "type": "bool",                                       "desc": "Enable the runtime age-label pipeline",  "needs_restart": True},
    "show_gender":        {"cat": "Demographics", "type": "bool",                                       "desc": "Show gender label on boxes"},
    "show_age":           {"cat": "Demographics", "type": "bool",                                       "desc": "Show age label on boxes"},
    "show_attributes":    {"cat": "Demographics", "type": "bool",                                       "desc": "Show body attribute tags such as Hat/Bag/Jacket"},
    "show_body_par_raw_labels": {"cat": "Demographics", "type": "bool",                                 "desc": "Show raw body PAR output scores in labels"},
    "demographics_interval": {"cat": "Demographics", "type": "int", "min": 0, "max": 30, "step": 1,    "desc": "Frames between body PAR updates (0=every frame)"},
    "demographics_min_conf": {"cat": "Demographics", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01, "desc": "Minimum confidence for gender labels"},
    "detect_face": {"cat": "Demographics", "type": "bool",                                               "desc": "Enable face detection for live overlays and overrides", "needs_restart": True},
    "gender_use_face_override": {"cat": "Demographics", "type": "bool",                                 "desc": "Allow face gender results to override body PAR gender"},
    "gender_face_override_min_conf": {"cat": "Demographics", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01, "desc": "Minimum face confidence required before overriding gender"},
    "show_gender_confidence": {"cat": "Demographics", "type": "bool",                                   "desc": "Show confidence for gender labels"},
    "show_age_confidence": {"cat": "Demographics", "type": "bool",                                      "desc": "Show confidence for age labels"},
    "enable_face_analysis": {"cat": "Demographics", "type": "bool",                                     "desc": "Run face age/gender inference for overrides", "needs_restart": True},
    "face_age_override": {"cat": "Demographics", "type": "bool",                                        "desc": "Allow face age results to override the live age label"},
    "face_age_senior_threshold": {"cat": "Demographics", "type": "int", "min": 40, "max": 80, "step": 1, "desc": "Face-age threshold above which a result becomes Senior"},
    "face_detection_conf": {"cat": "Demographics", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01, "desc": "Minimum confidence for face detection"},
    "face_overlap_threshold": {"cat": "Demographics", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01, "desc": "Face detector NMS IoU threshold"},
    "show_face_analysis_unlocked": {"cat": "Demographics", "type": "bool",                              "desc": "Keep showing live face analysis before it locks"},
    "use_pose_visibility_gate": {"cat": "Demographics", "type": "bool",                                 "desc": "Require pose visibility checks before accepting height samples"},
    "show_pose_keypoints": {"cat": "Demographics", "type": "bool",                                      "desc": "Show pose keypoints used by the height age pipeline"},
    "show_pose_indices": {"cat": "Demographics", "type": "bool",                                        "desc": "Draw pose keypoint indices for debugging", "advanced": True},
    "pose_interval":      {"cat": "Demographics", "type": "int", "min": 0, "max": 30, "step": 1,      "desc": "Frames between pose updates (0=every frame)"},
    "pose_max_tracks_per_frame": {"cat": "Demographics", "type": "int", "min": 0, "max": 50, "step": 1, "desc": "Max tracked people sent to pose per frame (0 = no limit)"},
    "pose_force_run": {"cat": "Demographics", "type": "bool",                                           "desc": "Run pose even when the height-age engine would normally skip a track", "advanced": True},
    "pose_full_frame": {"cat": "Demographics", "type": "bool",                                          "desc": "Infer pose from the full frame instead of per-person crops", "advanced": True},
    "pose_conf": {"cat": "Demographics", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01,       "desc": "Pose detector confidence threshold"},
    "pose_overlap_threshold": {"cat": "Demographics", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01, "desc": "Pose detector NMS IoU threshold"},
    "pose_crop_padding": {"cat": "Demographics", "type": "float", "min": 0.0, "max": 0.75, "step": 0.01, "desc": "Extra crop padding around each person before pose inference"},
    "pose_keypoint_conf": {"cat": "Demographics", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01, "desc": "Minimum keypoint confidence for pose-based height checks"},
    "pose_min_visible_keypoints": {"cat": "Demographics", "type": "int", "min": 0, "max": 17, "step": 1, "desc": "Minimum visible keypoints required for a pose sample"},
    "pose_min_torso_keypoints": {"cat": "Demographics", "type": "int", "min": 0, "max": 8, "step": 1, "desc": "Minimum torso keypoints required for a pose sample"},
    "pose_min_lower_body_keypoints": {"cat": "Demographics", "type": "int", "min": 0, "max": 6, "step": 1, "desc": "Minimum lower-body keypoints required for a pose sample"},
    "pose_min_lowest_keypoint_ratio": {"cat": "Demographics", "type": "float", "min": 0.0, "max": 1.0, "step": 0.01, "desc": "Required lowest-keypoint height ratio within the person box"},
    "show_age_height_stats": {"cat": "Demographics", "type": "bool",                                    "desc": "Show live height-derived age stats text"},
    "show_age_height_unlocked": {"cat": "Demographics", "type": "bool",                                 "desc": "Keep showing live height stats before lock"},
    "show_height_measurement_line": {"cat": "Demographics", "type": "bool",                             "desc": "Draw the head-to-feet line used for height age measurement"},
    "height_average_stability_px": {"cat": "Demographics", "type": "float", "min": 0.0, "max": 20.0, "step": 0.1, "desc": "640px-reference stability tolerance for height locking"},
    "height_display_freeze_samples": {"cat": "Demographics", "type": "int", "min": 0, "max": 120, "step": 1, "desc": "Samples required before freezing a trusted display baseline"},
    "height_peak_stability_frames": {"cat": "Demographics", "type": "int", "min": 0, "max": 120, "step": 1, "desc": "Stable frames required before freezing the display baseline"},
    "height_lock_min_samples": {"cat": "Demographics", "type": "int", "min": 0, "max": 120, "step": 1, "desc": "Minimum samples required before age can lock"},
    "height_lock_stability_frames": {"cat": "Demographics", "type": "int", "min": 0, "max": 120, "step": 1, "desc": "Stable frames required before age can lock"},
    "height_lock_requires_trusted_baseline": {"cat": "Demographics", "type": "bool",                    "desc": "Require a frozen trusted baseline before height age locks"},

    # ── Alerts ──
    "use_doorway_monitor": {"cat": "Alerts", "type": "bool",                                            "desc": "Track minor-adult groups entering a doorway ROI"},
    "show_doorway_status": {"cat": "Alerts", "type": "bool",                                            "desc": "Show doorway room-timer statuses in the overlay"},
    "room_presence_alert_seconds": {"cat": "Alerts", "type": "float", "min": 1.0, "max": 60.0, "step": 0.5, "desc": "Seconds in room before a doorway alert triggers"},
    "doorway_commit_frames": {"cat": "Alerts", "type": "int", "min": 1, "max": 30, "step": 1,         "advanced": True, "desc": "Frames required before the doorway monitor commits an entry event"},
    "doorway_missing_frames": {"cat": "Alerts", "type": "int", "min": 1, "max": 120, "step": 1,       "advanced": True, "desc": "Missing frames tolerated by the doorway monitor before exit logic advances"},
    "show_carry_status": {"cat": "Alerts", "type": "bool",                                              "desc": "Show carry-alert statuses for minor-on-adult interactions"},
    "show_carry_overlap_debug": {"cat": "Alerts", "type": "bool", "advanced": True,                     "desc": "Show carry overlap debug diagnostics"},
    "carry_overlap_score_threshold": {"cat": "Alerts", "type": "int", "min": 1, "max": 10, "step": 1, "advanced": True, "desc": "Carry overlap score required before the alert path engages"},
    "carry_overlap_lock_frames": {"cat": "Alerts", "type": "int", "min": 1, "max": 30, "step": 1,     "advanced": True, "desc": "Frames required before a carry alert locks"},
    "carry_overlap_release_frames": {"cat": "Alerts", "type": "int", "min": 1, "max": 30, "step": 1,  "advanced": True, "desc": "Frames required before a carry alert releases"},

    # ── Archive / Output ──
    "out": {"cat": "Archive", "type": "str",                                                            "desc": "Legacy/manual output path used by file and image runs", "advanced": True},
    "auto_output_name": {"cat": "Archive", "type": "bool",                                              "desc": "Auto-name runtime output files from the current source", "advanced": True},
    "testing_output_root": {"cat": "Archive", "type": "str",                                            "desc": "Root folder for testing outputs", "advanced": True},
    "nvr_record_enable": {"cat": "Archive", "type": "bool",                                             "desc": "Enable rolling archive video recording"},
    "nvr_segment_minutes": {"cat": "Archive", "type": "int", "min": 1, "max": 60, "step": 1,          "desc": "Minutes per archive segment", "advanced": True},
    "nvr_retention_hours": {"cat": "Archive", "type": "float", "min": 1.0, "max": 168.0, "step": 1.0, "desc": "How long archive segments are retained", "advanced": True},
    "nvr_output_root": {"cat": "Archive", "type": "str",                                                "desc": "Root folder for archive/NVR output", "advanced": True},
    "group_log_enable": {"cat": "Archive", "type": "bool",                                              "desc": "Write group-event logs alongside archive output"},

    # ── Input ──
    "input_mode":         {"cat": "Input",     "type": "select", "options": "source,camera",           "desc": "Input mode"},
    "source":             {"cat": "Input",     "type": "str",                                          "desc": "Source (URL or media/path)"},
    "camera_index":       {"cat": "Input",     "type": "int",   "min": 0,   "max": 10,  "step": 1,     "desc": "Camera device index"},
    "device":             {"cat": "Input",     "type": "select", "options": "CPU,GPU,AUTO",            "desc": "Inference device", "needs_restart": True},
    "perspective":        {"cat": "Input",     "type": "select", "options": "TOP-DOWN,LEVELED,AUTO",   "desc": "Camera perspective hint",                 "needs_restart": True},

    # ── Performance ──
    "turbo_mode":         {"cat": "Performance","type": "bool",                                         "desc": "Turbo mode (no preview, no output save)", "needs_restart": True},
    "mimic_live":         {"cat": "Performance","type": "bool",                                         "desc": "Process video files like a live stream"},
    "save_output_video": {"cat": "Performance", "type": "bool",                                         "desc": "Save the runtime preview video output", "advanced": True},
    "output_video_fps": {"cat": "Performance", "type": "float", "min": 0.0, "max": 120.0, "step": 1.0, "desc": "Override FPS for saved output video (0 uses source FPS)", "advanced": True},
    "output_processed_only": {"cat": "Performance", "type": "bool",                                     "desc": "Skip drawing/output on frames where inference is skipped", "advanced": True},
    "save_session_artifacts": {"cat": "Performance", "type": "bool",                                    "desc": "Save annotated video and JSON artifacts in Run Summaries"},
    "show_perf":          {"cat": "Performance","type": "bool",                                         "desc": "Show live performance stats in the runtime overlay", "advanced": True},
    "verbose_logging":    {"cat": "Performance","type": "bool",                                         "desc": "Emit extra runtime logs for debugging", "advanced": True},
    "preview_window_mode": {"cat": "Performance", "type": "select", "options": "normal,fullscreen,maximized", "desc": "Native preview window mode when CV2 preview is enabled", "advanced": True},

    # ── Lens Undistortion ──
    "undistort_enable":   {"cat": "Lens",      "type": "bool",                                          "desc": "Enable lens undistortion"},
    "undistort_model":    {"cat": "Lens",      "type": "select", "options": "fisheye,standard",        "desc": "Undistortion model"},
    "undistort_fx":       {"cat": "Lens",      "type": "float", "min": 1.0, "max": 4000.0, "step": 1.0, "desc": "Focal length X (fx)"},
    "undistort_fy":       {"cat": "Lens",      "type": "float", "min": 1.0, "max": 4000.0, "step": 1.0, "desc": "Focal length Y (fy)"},
    "undistort_balance":  {"cat": "Lens",      "type": "float", "min": 0.0, "max": 1.0,  "step": 0.01, "desc": "Fisheye balance (0=crop, 1=FOV)"},
    "undistort_alpha":    {"cat": "Lens",      "type": "float", "min": 0.0, "max": 1.0,  "step": 0.01, "desc": "Standard alpha (0=crop, 1=FOV)"},
    "undistort_coeffs":   {"cat": "Lens",      "type": "str",                                          "desc": "Distortion coeffs (k1, k2, p1, p2, k3, ...)"},
}

# ---------------------------------------------------------------------------
# Persistent state: live config and persistence
# ---------------------------------------------------------------------------

DB_PATH = PROJECT_ROOT / "settings.db"
init_db(DB_PATH)

# ---------------------------------------------------------------------------
# Global engine instance
# ---------------------------------------------------------------------------
engine = EngineManager(PROJECT_ROOT, DB_PATH)

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app_flask = Flask(__name__)
CORS(app_flask)
register_routes(
    app_flask,
    engine=engine,
    param_meta=PARAM_META,
    defaults_dict=DEFAULTS_DICT,
    project_root=PROJECT_ROOT,
    db_path=DB_PATH,
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Group Detection Dashboard Server")
    print("  API  -> http://localhost:8765/api/...")
    print("  Feed -> http://localhost:8765/video_feed")
    print("=" * 60)
    print("[Server] Engine is idle on startup. Start it from the dashboard or /api/engine/start.")
    try:
        app_flask.run(host="0.0.0.0", port=8765, threaded=True, debug=False)
    finally:
        print("\n[Server] Shutting down...")
        engine.stop()
