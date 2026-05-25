#!/usr/bin/env python3
"""Source-specific runtime defaults for the detection pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SourceRuntimeDefaults:
    # ------------------------------------------------------------------
    # Models and Device Selection
    # ------------------------------------------------------------------
    model: str = "models/person/best.xml"
    model_angled: str = "models/person/best.xml"
    labels: str = "models/person/labels.txt"
    perspective: str = "AUTO"  # "LEVELED", "TOP-DOWN", or "AUTO"

    # ------------------------------------------------------------------
    # Lens Undistortion
    # ------------------------------------------------------------------
    undistort_enable: bool = False
    undistort_model: str = "fisheye"  # "fisheye" or "standard"
    undistort_coeffs: tuple[float, ...] = ()
    undistort_fx: float = 0.0
    undistort_fy: float = 0.0
    undistort_ref_width: int = 1080
    undistort_ref_height: int = 1080
    undistort_balance: float = 0.000
    undistort_alpha: float = 0.0

    # ------------------------------------------------------------------
    # ROI
    # ------------------------------------------------------------------
    use_roi: bool = True
    roi_polygon: tuple[tuple[float, float], ...] = ()
    roi_padding: float = 0.0
    height_roi_polygon: tuple[tuple[float, float], ...] = ()
    doorway_roi_polygon: tuple[tuple[float, float], ...] = ()

    # ------------------------------------------------------------------
    # Demographics: Body Attributes
    # ------------------------------------------------------------------
    person_attributes_model: str = "models/attributes/person-attributes-recognition-crossroad-0234.xml"
    person_attributes_labels: str = "models/attributes/person-attributes-recognition-crossroad-0234.labels.txt"
    demographics_history_len: int = 30
    demographics_interval: int = 0

    # ------------------------------------------------------------------
    # Age Classification
    # ------------------------------------------------------------------
    # Height / size measurement
    topdown_height_metric: str = "max_side"
    age_adult_threshold_px: float = 240
    age_history_len: int = 120
    age_smoothing: float = 0.15
    height_average_stability_px: float = 3.0
    height_display_freeze_samples: int = 18
    height_peak_stability_frames: int = 10
    height_lock_min_samples: int = 12
    height_lock_stability_frames: int = 10
    height_lock_requires_trusted_baseline: bool = True

    # Pose model parameters
    pose_model: str = "models/pose/yolo11n-pose.xml"
    pose_interval: int = 0
    pose_max_tracks_per_frame: int = 0
    pose_crop_padding: float = 0.15

    # Face analysis models: face detection + age/gender override
    face_detection_model: str = (
        "models/face_analysis/face-detection-retail-0004/face-detection-retail-0004.xml"
    )
    face_detection_labels: str = (
        "models/face_analysis/face-detection-retail-0004/labels.txt"
    )
    face_age_gender_model: str = (
        "models/face_analysis/age-gender-recognition-retail-0013/"
        "age-gender-recognition-retail-0013.xml"
    )

    # ------------------------------------------------------------------
    # Group Detection tuning
    # ------------------------------------------------------------------
    group_history_len: int = 30
    group_pair_angle_history_len: int = 10
    group_merged_warn_interval: int = 15
    group_origin_gap_threshold: float = 400.0
    leveled_min_size_ratio: float = 0.55
    leveled_max_foot_y_relative: float = 0.4
    leveled_combo_size_ratio: float = 0.75
    leveled_combo_foot_y_relative: float = 0.25
    leveled_z_velocity_blocking: bool = True
    topdown_min_size_ratio: float = 0.20
    topdown_max_foot_y_relative: float = 1.0
    topdown_combo_size_ratio: float = 0.30
    topdown_combo_foot_y_relative: float = 0.8
    topdown_z_velocity_blocking: bool = False
