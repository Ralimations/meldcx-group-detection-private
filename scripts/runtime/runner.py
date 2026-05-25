#!/usr/bin/env python3
"""Runtime runner orchestration."""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace
import sys
import time

PROJECT_ROOT = Path(__file__).parent.parent.parent
GLOBAL_PYCACHE_DIR = PROJECT_ROOT / ".cache" / "pycache"

# Centralize bytecode cache for this project.
if sys.pycache_prefix is None:
    GLOBAL_PYCACHE_DIR.mkdir(parents=True, exist_ok=True)
    sys.pycache_prefix = str(GLOBAL_PYCACHE_DIR)

import cv2
import numpy as np

from config import OpenVinoDefaults
from scripts.configuration.image_undistorter import ImageUndistorter, apply_source_undistort_overrides
from scripts.configuration.runtime_config import (
    apply_source_runtime_overrides,
    ensure_source_initialized,
    save_source_runtime_overrides,
)
from scripts.runtime.preview_ui import (
    load_roi_for_source,
    roi_source_key,
    save_roi_for_source,
)
from scripts.runtime.archive_recorder import RollingVideoRecorder
from scripts.runtime.group_event_logger import GroupEventLogger
from scripts.runtime.engine_factory import initialize_detection_engines, load_labels
from scripts.runtime.room_trip_monitor import RoomTripMonitor

def resolve_existing_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path
    rooted = PROJECT_ROOT / path_str
    if rooted.exists():
        return rooted
    return rooted


def resolve_output_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


class DetectionApp:
    """Config-driven runner for image/video/camera inference."""

    def __init__(self, cfg: OpenVinoDefaults) -> None:
        self.cfg = cfg
        
        # --- Turbo Mode Overrides ---
        if getattr(self.cfg, "turbo_mode", False):
            self.cfg = replace(
                self.cfg,
                mimic_live=False,
                no_show=True,
                show_perf=False
            )
            print("[TURBO] Mode enabled: mimic_live=OFF, no_show=ON, show_perf=OFF")

        self.skip_frames = max(1, int(cfg.skip_frames))
        self.mode = str(cfg.input_mode).strip().lower()
        if self.mode in ("network", "file"):
            self.mode = "source"
        self.perspective = getattr(cfg, "perspective", "AUTO").upper()
        if self.perspective not in ("AUTO", "LEVELED", "TOP-DOWN"):
            self.perspective = "AUTO"
        
        if self.perspective == "TOP-DOWN":
            self.model_path = resolve_existing_path(cfg.model_angled)
            print(f"Loading angled nano model for perspective: {self.perspective}", flush=True)
        else:
            self.model_path = resolve_existing_path(cfg.model)
            if self.perspective == "AUTO":
                print("Perspective AUTO: Loading standard leveled model by default.", flush=True)
            else:
                print(f"Loading standard model for perspective: {self.perspective}", flush=True)

        print(f"[Engine] Internal labels path: {cfg.labels}", flush=True)
        # For RTSP/HTTP streams, keep the URL as a raw string (NOT Path — Windows breaks slashes)
        source_str = str(cfg.source).strip().strip("'\"")
        self.source_str = source_str
        self._stream_url: str | None = None
        if self.mode in ("source", "batch") and source_str.lower().startswith(("rtsp://", "http://", "https://")):
            self._stream_url = source_str
            self.source_path = None
            self._is_stream = True
            print(f"[Engine] Input is live stream: {source_str}", flush=True)
        else:
            print(f"[Engine] Resolving source path: {cfg.source}...", flush=True)
            self.source_path = resolve_existing_path(cfg.source) if self.mode in ("source", "batch") else None
            self._is_stream = False
            print(f"[Engine] Source path resolved: {self.source_path}", flush=True)
        if self.mode != "image":
            ensure_source_initialized(
                self.source_str,
                mode=self.mode,
                camera_index=int(getattr(self.cfg, "camera_index", 0)),
            )
        self.cfg = apply_source_runtime_overrides(
            self.cfg,
            source=self.source_str,
            mode=self.mode,
            camera_index=int(getattr(self.cfg, "camera_index", 0)),
        )
        self.cfg = apply_source_undistort_overrides(
            self.cfg,
            source=self.source_str,
            mode=self.mode,
            camera_index=int(getattr(self.cfg, "camera_index", 0)),
        )
        self.camera_index = int(self.cfg.camera_index)
        self._apply_source_roi_overrides()
        self.skip_frames = max(1, int(self.cfg.skip_frames))
        self.perspective = getattr(self.cfg, "perspective", "AUTO").upper()
        if self.perspective not in ("AUTO", "LEVELED", "TOP-DOWN"):
            self.perspective = "AUTO"
        if self.perspective == "TOP-DOWN":
            self.model_path = resolve_existing_path(self.cfg.model_angled)
        else:
            self.model_path = resolve_existing_path(self.cfg.model)
        self.undistorter = ImageUndistorter(self.cfg) if getattr(self.cfg, "undistort_enable", False) else None
        self.batch_sources = getattr(self.cfg, "batch_sources", ())
        self.classes_to_keep = set()
        if getattr(self.cfg, "detect_body", True): self.classes_to_keep.add(0)
        if getattr(self.cfg, "detect_head", False): self.classes_to_keep.add(1)
        if not self.classes_to_keep:
            self.classes_to_keep = None # Fallback to all if both disabled (though unlikely intended)
        
        print(f"[Engine] Resolving labels path: {self.cfg.labels}...", flush=True)
        self.labels_path = resolve_existing_path(self.cfg.labels)
        print(f"[Engine] Loading labels from: {self.labels_path}...", flush=True)
        self.labels = load_labels(self.labels_path) if self.labels_path.exists() else []
        print(f"[Engine] Labels loaded: {len(self.labels)} classes", flush=True)
        self.output_path = self._build_output_path()
        print(f"[Engine] Creating output dir: {self.output_path.parent}...", flush=True)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[Engine] Output dir ready.", flush=True)

        print(f"[Engine] Validating inputs...", flush=True)
        self._validate_inputs()
        print(f"[Engine] Inputs validated.", flush=True)

        print(f"[Detector] Loading model file: {self.model_path.name}...", flush=True)
        (
            self.detector,
            self.demographics_engine,
            self.height_age_engine,
            self.pose_engine,
            self.face_age_engine,
        ) = initialize_detection_engines(self.cfg, self.model_path, resolve_existing_path)
        print(f"[Detector] Model loaded on {getattr(self.detector, 'device', 'unknown')}.", flush=True)

        # Keep remembered demographics keyed by persistent person_id.
        self._gender_cache: dict[int, tuple[str, str, float]] = {}
        self._gender_cache_stale: dict[int, int] = {}
        self._age_cache: dict[int, tuple[str, str, float]] = {}
        self._age_cache_stale: dict[int, int] = {}
        self.room_trip_monitor = RoomTripMonitor(self.cfg)

    def _validate_inputs(self) -> None:
        if self.mode not in {"source", "camera", "image", "batch"}:
            raise SystemExit(f"config.py `input_mode` must be one of: source, camera, image, batch. Got: '{self.mode}'")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        if self.mode == "image":
            if self.source_path is None or not self.source_path.exists():
                raise FileNotFoundError(f"Image not found: {self.source_path}")
        elif self.mode in ("source", "batch"):
            if not self._is_stream and (self.source_path is None or not self.source_path.exists()):
                raise FileNotFoundError(f"Video not found: {self.source_path}")

        if (
            getattr(self.cfg, "save_output_video", True)
            and self.mode in {"source", "camera", "batch"}
            and self.output_path.suffix.lower() != ".mp4"
        ):
            raise SystemExit("Video/camera output must use .mp4 extension.")

    def _build_output_path(self) -> Path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        detection_mode = "group" if self.cfg.group_detect else "person"
        testing_root = resolve_output_path(getattr(self.cfg, "testing_output_root", "output/testing"))
        
        if self._is_stream:
            stem = "stream"
            name = f"[{timestamp}] {detection_mode}_{stem}"
            filename = f"{stem}.mp4"
        elif self.mode in ("source", "batch") and self.source_path is not None:
            stem = "stream" if self._is_stream else self.source_path.stem
            name = f"[{timestamp}] {detection_mode}_{stem}"
            filename = f"{stem}.mp4"
        elif self.mode == "image" and self.source_path is not None:
            name = f"[{timestamp}] {detection_mode}_{self.source_path.stem}"
            filename = f"{self.source_path.stem}.jpg"
        elif self.mode == "camera":
            name = f"[{timestamp}] {detection_mode}_camera"
            filename = "camera_output.mp4"
        else:
            name = f"[{timestamp}] {detection_mode}_inference"
            filename = self.cfg.out.split("/")[-1] if "/" in self.cfg.out else self.cfg.out

        session_dir = testing_root / name
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir / filename

    def run(self) -> None:
        if self.mode == "image":
            self._run_image()
        elif self.mode == "batch":
            self._run_batch()
        else:
            metrics, _ = self._run_stream()
            video_name = self.source_path.name if self.source_path else "camera"
            self._write_batch_summary(
                [{"video": video_name, "people": metrics["people"], "groups": metrics["groups"]}],
                self.output_path.parent
            )

        self._cleanup_old_outputs()

    def _cleanup_old_outputs(self) -> None:
        out_dir = resolve_output_path(getattr(self.cfg, "testing_output_root", "output/testing"))
        if not out_dir.exists():
            return
            
        session_dirs = []
        for p in out_dir.iterdir():
            if p.is_dir():
                # Simple check for session-like names (starting with [202...)
                if p.name.startswith("["):
                    session_dirs.append(p)
                
        session_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        
        max_dirs = 5
        to_delete = session_dirs[max_dirs:]
        
        import shutil
        for d in to_delete:
            try:
                shutil.rmtree(d)
                print(f"Cleaned up old session: {d.name}")
            except OSError as e:
                print(f"Failed to delete {d.name}: {e}")

    def _run_image(self) -> None:
        frame = cv2.imread(str(self.source_path))
        if frame is None:
            raise RuntimeError(f"Failed to read image: {self.source_path}")
        if self.undistorter is not None:
            frame = self.undistorter.apply(frame)

        boxes, confidence_scores, class_ids = self.detector.detect_frame(
            frame,
            confidence_threshold=self.cfg.person_detection_conf,
            overlap_threshold=self.cfg.person_overlap_threshold,
        )
        boxes, confidence_scores, class_ids = filter_to_requested_classes(
            boxes, confidence_scores, class_ids, self.classes_to_keep
        )
        if self.cfg.show_body or self.cfg.show_head:
            draw_detection_boxes(
                frame,
                boxes,
                confidence_scores,
                class_ids,
                show_body_confidence=self.cfg.show_body_confidence,
                show_head_confidence=self.cfg.show_head_confidence,
                show_body=self.cfg.show_body,
                show_head=self.cfg.show_head,
                label_default_color=getattr(self.cfg, "label_default_color", "#FFFF00"),
                label_segment_colors=getattr(self.cfg, "label_segment_colors", ""),
            )

        for box, confidence, class_id in zip(boxes, confidence_scores, class_ids):
            x1, y1, x2, y2 = box.astype(int)
            class_id_int = int(class_id)
            label = self.labels[class_id_int] if 0 <= class_id_int < len(self.labels) else f"class_{class_id_int}"
            print(f"{label}: {float(confidence):.3f} box=({x1},{y1},{x2},{y2})")

        if len(boxes) == 0:
            print("No detections above confidence threshold.")

        cv2.imwrite(str(self.output_path), frame)
        print(f"Saved: {self.output_path}")
        self._write_batch_summary(
            [{"video": self.source_path.name, "people": len(boxes), "groups": 0}],
            self.output_path.parent
        )

    def _run_batch(self) -> None:
        all_metrics = []
        batch_timestamp = time.strftime("%Y%m%d_%H%M%S")
        testing_root = resolve_output_path(getattr(self.cfg, "testing_output_root", "output/testing"))
        batch_dir = testing_root / f"[{batch_timestamp}] batch_session"
        batch_dir.mkdir(parents=True, exist_ok=True)

        for src in self.batch_sources:
            ensure_source_initialized(src, mode="source", camera_index=int(getattr(self.cfg, "camera_index", 0)))
            path = resolve_existing_path(src)
            if not path.exists():
                print(f"Skipping not found: {src}")
                continue
            
            self.source_path = path
            self._is_stream = False
            self.source_str = str(path)
            self.cfg = apply_source_runtime_overrides(
                self.cfg,
                source=self.source_str,
                mode="source",
                camera_index=int(getattr(self.cfg, "camera_index", 0)),
            )
            self.cfg = apply_source_undistort_overrides(
                self.cfg,
                source=self.source_str,
                mode="source",
                camera_index=int(getattr(self.cfg, "camera_index", 0)),
            )
            self._apply_source_roi_overrides()
            self.undistorter = ImageUndistorter(self.cfg) if getattr(self.cfg, "undistort_enable", False) else None
            detection_mode = "group" if self.cfg.group_detect else "person"
            self.output_path = batch_dir / f"{detection_mode}_{path.stem}.mp4"
            print(f"\n--- Batch sequence: {path.name} ---")
            
            metrics, end_all = self._run_stream(is_batch=True)
            all_metrics.append({"video": path.name, "people": metrics["people"], "groups": metrics["groups"]})
            
            if end_all:
                print("Batch sequence ended early by user.")
                break
                
        self._write_batch_summary(all_metrics, batch_dir)

    def _write_batch_summary(self, metrics_list: list[dict], batch_dir: Path) -> None:
        if not metrics_list:
            return
            
        summary_path = batch_dir / "session_summary.md"
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "# Detection Session Summary",
            f"**Timestamp:** {timestamp}",
            "",
            "| Video | People | Groups |",
            "| :--- | :--- | :--- |"
        ]
        
        total_p = 0
        total_g = 0
        for m in metrics_list:
            lines.append(f"| {m['video']} | {m['people']} | {m['groups']} |")
            total_p += m['people']
            total_g += m['groups']
            
        lines.append("")
        lines.append("---")
        lines.append(f"**Grand Total People:** {total_p}  ")
        lines.append(f"**Grand Total Groups:** {total_g}")
        summary_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"\nSummary saved to: {summary_path}")

    def _run_stream(self, is_batch: bool = False) -> tuple[dict[str, int], bool]:
        from scripts.runtime.stream_runner import run_stream

        return run_stream(self, is_batch=is_batch)

    def _save_editor_config(
        self,
        roi_polygon,
        height_roi_polygon,
        age_adult_threshold_px=None,
        doorway_roi_polygon=None,
    ):
        """Save live-edited ROI settings for the current source into settings.db."""
        source_key = self._roi_source_key()
        roi_tuple = tuple((float(p[0]), float(p[1])) for p in roi_polygon)
        height_roi_tuple = tuple((float(p[0]), float(p[1])) for p in height_roi_polygon)
        save_roi_for_source(source_key, roi_tuple, height_roi_tuple)
        runtime_updates = {}
        if age_adult_threshold_px is not None:
            threshold_value = float(age_adult_threshold_px)
            runtime_updates["age_adult_threshold_px"] = threshold_value
            self.cfg.age_adult_threshold_px = threshold_value
        if doorway_roi_polygon is not None:
            doorway_roi_tuple = tuple((float(p[0]), float(p[1])) for p in doorway_roi_polygon)
            runtime_updates["doorway_roi_polygon"] = list(doorway_roi_tuple)
            self.cfg.doorway_roi_polygon = doorway_roi_tuple
        if runtime_updates:
            save_source_runtime_overrides(source_key, runtime_updates)

        self.cfg.roi_polygon = roi_tuple
        self.cfg.height_roi_polygon = height_roi_tuple
        print(f"\n[Config] Saved ROI settings for source '{source_key}' to settings.db")

    def _roi_source_key(self) -> str:
        return roi_source_key(
            mode=self.mode,
            camera_index=self.camera_index,
            is_stream=self._is_stream,
            source_str=self.source_str,
            source_path=self.source_path,
        )

    def _apply_source_roi_overrides(self) -> None:
        source_key = self._roi_source_key()
        roi_settings = load_roi_for_source(source_key)
        if roi_settings is None:
            return

        roi_polygon, height_roi_polygon = roi_settings
        self.cfg.roi_polygon = roi_polygon
        self.cfg.height_roi_polygon = height_roi_polygon
        print(f"[Config] Loaded source-specific ROI for '{source_key}'", flush=True)

    def _build_video_writer(self, cap: cv2.VideoCapture, frame: np.ndarray) -> cv2.VideoWriter | None:
        h, w = frame.shape[:2]

        fps_src = cap.get(cv2.CAP_PROP_FPS)
        fps_src = fps_src if fps_src and fps_src > 0 else 30.0
        fps_out = float(getattr(self.cfg, "output_video_fps", 0.0) or 0.0)
        fps_out = fps_out if fps_out > 0 else fps_src
        # Use 'mp4v' for maximum compatibility
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self.output_path), fourcc, fps_out, (w, h))
        if not writer.isOpened():
            print(f"[Error] Failed to initialize VideoWriter at {w}x{h}")
            return None
        return writer

    def _nvr_source_name(self) -> str:
        if self.mode == "camera":
            return f"camera_{self.cfg.camera_index}"
        if self._is_stream:
            return self.source_str if self.source_str else "stream"
        if self.source_path is not None:
            return self.source_path.stem
        return "source"

    def _build_rolling_recorder(
        self,
        cap: cv2.VideoCapture,
        frame: np.ndarray,
        group_event_logger: GroupEventLogger | None = None,
    ) -> RollingVideoRecorder:
        h, w = frame.shape[:2]
        fps_src = cap.get(cv2.CAP_PROP_FPS)
        fps_src = fps_src if fps_src and fps_src > 0 else 30.0
        root_dir = resolve_output_path(getattr(self.cfg, "nvr_output_root", "output/archive"))
        root_dir.mkdir(parents=True, exist_ok=True)
        return RollingVideoRecorder(
            root_dir=root_dir,
            source_name=self._nvr_source_name(),
            fps=fps_src,
            frame_size=(w, h),
            segment_minutes=int(getattr(self.cfg, "nvr_segment_minutes", 1)),
            retention_hours=float(getattr(self.cfg, "nvr_retention_hours", 24.0)),
            on_segment_open=group_event_logger.on_segment_open if group_event_logger is not None else None,
            on_segment_close=group_event_logger.on_segment_close if group_event_logger is not None else None,
        )


