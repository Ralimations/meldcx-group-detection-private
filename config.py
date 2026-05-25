#!/usr/bin/env python3
"""Global/operator defaults plus merged runtime config type."""

from __future__ import annotations

from dataclasses import dataclass

from scripts.configuration.source_runtime_defaults import SourceRuntimeDefaults


@dataclass
class GlobalConfigDefaults:
    # ------------------------------------------------------------------
    # Input Sources
    # ------------------------------------------------------------------
    input_mode: str = "source"  # "source", "camera", "image", or "batch"
    source: str = ""
    camera_index: int = 0
    batch_sources: tuple[str, ...] = ()

    # ------------------------------------------------------------------
    # Output Paths
    # ------------------------------------------------------------------
    out: str = "output/result.jpg"  # Legacy/manual output path.
    auto_output_name: bool = True
    testing_output_root: str = "output/testing"
    nvr_output_root: str = "output/archive"

    # ------------------------------------------------------------------
    # Recording / Archive Policy
    # ------------------------------------------------------------------
    save_output_video: bool = True
    save_session_artifacts: bool = False
    output_video_fps: float = 0.0  # 0 uses source FPS; set a positive value to override.
    nvr_record_enable: bool = False
    nvr_segment_minutes: int = 1
    nvr_retention_hours: float = 24.0
    group_log_enable: bool = True

    # ------------------------------------------------------------------
    # Runtime / Performance
    # ------------------------------------------------------------------
    device: str = "CPU"  # "CPU", "GPU", etc.
    mimic_live: bool = True
    turbo_mode: bool = False
    show_perf: bool = False
    verbose_logging: bool = False
    imgsz: int = 640
    downscale_to_imgsz: bool = True
    skip_frames: int = 0
    output_processed_only: bool = True
    
    # ------------------------------------------------------------------
    # On-Screen Display
    # ------------------------------------------------------------------

    show_ui_overlay: bool = True
    label_font_scale: float = 1.35
    label_thickness: int = 2
    label_bg_alpha: float = 0.82
    label_max_width_chars: int = 26
    label_default_color: str = "#FFFF00"  # Default label text color as #RRGGBB.
    label_segment_colors: str = (
        "Height [waiting ROI]=#FFA500,"
        "Height [invalid sample]=#FF4D4F,"
        "Height [pending pose]=#FF4D4F,"
        "Height [PX:=#FFFF00,"
        "(B)=#FFFF00,"
        "(F)=#FFFF00,"
        "(H)=#FFFF00,"
        "Male=#4DA6FF,"
        "Female=#FF69B4,"
        "Minor=#FFB347,"
        "Adult=#7DDA58,"
        "Senior=#FF8C42"
    )    

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------
    stable: bool = True
    show_unconfirmed: bool = False
    track_overlap_threshold: float = 0.2
    min_hits: int = 3
    max_misses: int = 30
    track_max_draw_misses: int = 5
    smooth: float = 0.3
    track_min_conf: float = 0.20
    track_reid_window: int = 10000
    track_reid_similarity_thresh: float = 0.3
    track_reid_update_interval: int = 5
    track_reid_backend: str = "openvino"  # "off", "histogram", "openvino", or "hybrid"
    track_reid_model: str = "models/reid/person-reidentification-retail-0277/person-reidentification-retail-0277.xml"
    track_reid_embedding_similarity_thresh: float = 0.3
    track_reid_require_not_touching_frame_edge: bool = True
    track_reid_min_quality_frames: int = 5
    track_reid_identity_confirm_frames: int = 3
    track_reid_use_spatial_gate: bool = False
    track_reid_spatial_gate_scale: float = 2.5

    # ------------------------------------------------------------------
    # Room Trip Monitor
    # ------------------------------------------------------------------
    use_doorway_monitor: bool = True
    show_doorway_status: bool = True
    room_presence_alert_seconds: float = 5.0
    doorway_commit_frames: int = 2
    doorway_missing_frames: int = 15

    # ------------------------------------------------------------------
    # Overlay / ROI display
    # ------------------------------------------------------------------
    show_roi_mask: bool = True
    roi_transparency: float = 0
    show_height_roi_mask: bool = True
    height_roi_transparency: float = 0

    # ------------------------------------------------------------------
    # Person/Head Detection
    # ------------------------------------------------------------------
    detect_body: bool = True
    show_body: bool = True
    show_body_confidence: bool = False
    person_detection_conf: float = 0.4
    person_overlap_threshold: float = 0.6

    detect_head: bool = True
    show_head: bool = True
    show_head_confidence: bool = True
    head_detection_conf: float = 0.7
    head_overlap_threshold: float = 0.3

    show_merged_warnings: bool = False

    # ------------------------------------------------------------------
    # Face Detection
    # ------------------------------------------------------------------
    detect_face: bool = False  # Enable face detection model.
    show_face: bool = True
    show_face_confidence: bool = True
    face_detection_conf: float = 0.5
    face_overlap_threshold: float = 0.4
    
    enable_face_analysis: bool = False  # Enable face age/gender analysis model on face crops.
    show_face_analysis_unlocked: bool = True  # Debug: keep showing live face analysis and disable face analysis freeze/lock.

    # ------------------------------------------------------------------
    # Gender (face vs par)
    # ------------------------------------------------------------------
    detect_gender: bool = True
    show_gender: bool = True
    show_gender_confidence: bool = False

    gender_use_face_override: bool = True
    gender_face_override_min_conf: float = 0.6
    
    demographics_min_conf: float = 0.5
    show_attributes: bool = True
    show_body_par_raw_labels: bool = False
    
    # ------------------------------------------------------------------
    # Age (face vs height)
    # ------------------------------------------------------------------
    detect_age: bool = True
    show_age: bool = True
    show_age_confidence: bool = True
    
    face_age_override: bool = True
    face_age_senior_threshold: int = 50

    height_min_samples_in_roi: int = 6
    show_age_height_stats: bool = False
    show_age_height_unlocked: bool = False  # Debug: keep showing live height stats and disable age freeze/lock.
    
    show_height_measurement_line: bool = False
    show_pose_keypoints: bool = True
    show_pose_indices: bool = False

    
    # ------------------------------------------------------------------
    # Pose
    # ------------------------------------------------------------------
    pose_force_run: bool = False
    pose_full_frame: bool = False
    use_pose_visibility_gate: bool = True
    pose_conf: float = 0.4
    pose_overlap_threshold: float = 0.6
    pose_keypoint_conf: float = 0.3
    pose_min_visible_keypoints: int = 8
    pose_min_torso_keypoints: int = 3
    pose_min_lower_body_keypoints: int = 2
    pose_min_lowest_keypoint_ratio: float = 0.82

    # ------------------------------------------------------------------
    # Group Detection
    # ------------------------------------------------------------------
    group_detect: bool = True
    group_max_relative_dist_check: float = 0.8
    group_max_relative_dist_lock: float = 0.2
    group_min_history: int = 7
    group_idle_speed_thresh: float = 1.0
    group_lock_threshold: float = 1.0
    group_max_score: float = 1.2
    show_carry_overlap_debug: bool = True
    show_carry_status: bool = True
    carry_overlap_score_threshold: int = 2
    carry_overlap_lock_frames: int = 3
    carry_overlap_release_frames: int = 3

    # ------------------------------------------------------------------
    # Preview Window
    # ------------------------------------------------------------------
    no_show: bool = False
    preview_width: int = 800
    preview_height: int = 450
    preview_window_mode: str = "normal"  # "normal", "fullscreen", or "maximized"


@dataclass
class OpenVinoDefaults(SourceRuntimeDefaults, GlobalConfigDefaults):
    """Merged runtime config type: global display/input + source runtime fields."""


OPENVINO_DEFAULTS = OpenVinoDefaults()
