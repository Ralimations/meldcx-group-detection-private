#!/usr/bin/env python3
"""Detector and classifier engine construction helpers."""

from __future__ import annotations

from pathlib import Path

from scripts.group_detector import GroupDetector
from scripts.perspective_detector import PerspectiveDetector
from scripts.person_detector import PersonDetector, PersonTracker
from scripts.demographics_classifier import DemographicsClassifier
from scripts.face_age_classifier import FaceAgeClassifier
from scripts.height_age_classifier import HeightAgeClassifier
from scripts.pose_estimator import PoseEstimator


def load_labels(path: Path) -> list[str]:
    labels: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            labels.append(line.split(":", 1)[1].strip().strip("'\""))
        else:
            labels.append(line)
    return labels


def initialize_detection_engines(cfg, model_path: Path, resolve_existing_path):
    detector = PersonDetector(model_path=model_path, cfg=cfg)

    demographics_engine = None
    demo_configs = {}
    needs_body_attributes = (
        bool(getattr(cfg, "show_attributes", False))
        or bool(getattr(cfg, "detect_gender", False))
    )
    if needs_body_attributes:
        demo_configs["gender"] = {
            "model": resolve_existing_path(cfg.person_attributes_model),
            "labels": resolve_existing_path(cfg.person_attributes_labels),
            "history": cfg.demographics_history_len,
        }
    if demo_configs:
        print(f"[Demographics] Initializing with {list(demo_configs.keys())}")
        demographics_engine = DemographicsClassifier(
            configs=demo_configs,
            device=cfg.device,
            interval=cfg.demographics_interval,
            stale_ttl=cfg.max_misses,
        )
        print("[Demographics] Initialized.")

    height_age_engine = None
    if bool(getattr(cfg, "detect_age", False)):
        try:
            print("[HeightAge] Initializing height-based age classifier...")
            height_age_engine = HeightAgeClassifier(cfg)
            print("[HeightAge] Initialized.")
        except Exception as e:
            print(f"[Warning] Height age classifier not available: {e}")
            print("         Height-based age disabled.")

    pose_engine = None
    if (
        (cfg.detect_age and getattr(cfg, "use_pose_visibility_gate", False))
        or bool(getattr(cfg, "pose_force_run", False))
        or bool(getattr(cfg, "show_pose_keypoints", False))
    ):
        try:
            print("[Pose] Initializing pose estimator...")
            pose_engine = PoseEstimator(cfg)
            print("[Pose] Initialized.")
        except Exception as e:
            print(f"[Warning] Pose model not available: {e}")
            print("         Pose visibility gate disabled.")

    face_age_engine = None
    needs_face_demographics = bool(
        getattr(cfg, "detect_face", False)
        and (
            (
                cfg.detect_age
                and getattr(cfg, "enable_face_analysis", False)
                and getattr(cfg, "face_age_override", False)
            )
            or (cfg.detect_gender and getattr(cfg, "gender_use_face_override", False))
        )
    )
    if needs_face_demographics:
        try:
            print("[Face] Initializing face age/gender models...")
            face_age_engine = FaceAgeClassifier(cfg)
            print("[Face] Initialized.")
        except Exception as e:
            print(f"[Warning] Face age/gender model not available: {e}")
            print("         Face-based demographics disabled.")

    return detector, demographics_engine, height_age_engine, pose_engine, face_age_engine


def create_stream_processors(cfg):
    use_stable_tracking = cfg.stable or cfg.group_detect or cfg.detect_gender or cfg.detect_age
    perspective_detector = PerspectiveDetector(cfg=cfg)
    person_tracker = PersonTracker(cfg=cfg) if use_stable_tracking else None
    group_detector = GroupDetector(cfg=cfg) if cfg.group_detect else None
    return perspective_detector, person_tracker, group_detector
