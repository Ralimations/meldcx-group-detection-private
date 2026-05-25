#!/usr/bin/env python3
"""Thin entrypoint for the detection app runtime."""

from __future__ import annotations

import argparse
from dataclasses import replace
from collections import defaultdict
from pathlib import Path
from typing import Any
import re
import sys
import time
import threading
import ctypes

PROJECT_ROOT = Path(__file__).resolve().parent
GLOBAL_PYCACHE_DIR = PROJECT_ROOT / ".cache" / "pycache"

# Centralize bytecode cache for this project.
if sys.pycache_prefix is None:
    GLOBAL_PYCACHE_DIR.mkdir(parents=True, exist_ok=True)
    sys.pycache_prefix = str(GLOBAL_PYCACHE_DIR)

import cv2
import numpy as np

from config import OPENVINO_DEFAULTS, OpenVinoDefaults
from scripts.group_detector import GroupDetector
from scripts.perspective_detector import PerspectiveDetector
from scripts.person_detector import (
    PersonDetector,
    PersonTracker,
    draw_detection_boxes,
    filter_to_requested_classes,
)
from scripts.demographics_classifier import DemographicsClassifier
from scripts.height_age_classifier import HeightAgeClassifier
from scripts.face_age_classifier import FaceAgeClassifier
try:
    from scripts.depth_calibrator import DepthCalibrator
except ModuleNotFoundError:
    DepthCalibrator = None  # type: ignore[assignment]
from scripts.pose_estimator import PoseEstimator
from scripts.configuration.image_undistorter import ImageUndistorter

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


class DetectionApp:
    """Config-driven runner for image/video/camera inference."""

    def __init__(self, cfg: OpenVinoDefaults) -> None:
        self.cfg = cfg
        self.undistorter = ImageUndistorter(cfg) if getattr(cfg, "undistort_enable", False) else None
        
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
        self.camera_index = int(cfg.camera_index)
        self.batch_sources = getattr(cfg, "batch_sources", ())
        self.classes_to_keep = set()
        if getattr(cfg, "detect_body", True): self.classes_to_keep.add(0)
        if getattr(cfg, "detect_head", False): self.classes_to_keep.add(1)
        if not self.classes_to_keep:
            self.classes_to_keep = None # Fallback to all if both disabled (though unlikely intended)
        
        print(f"[Engine] Resolving labels path: {cfg.labels}...", flush=True)
        self.labels_path = resolve_existing_path(cfg.labels)
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
        self.detector = PersonDetector(
            model_path=self.model_path,
            cfg=cfg,
        )
        print(f"[Detector] Model loaded on {getattr(self.detector, 'device', 'unknown')}.", flush=True)
        
        # Demographics (Body-based PAR)
        self.demographics_engine = None
        demo_configs = {}
        if self.cfg.detect_gender:
            demo_configs['gender'] = {
                'model': resolve_existing_path(cfg.person_attributes_model),
                'labels': resolve_existing_path(cfg.person_attributes_labels),
                'history': cfg.demographics_history_len
            }
            
        if demo_configs:
            print(f"[Demographics] Initializing with {list(demo_configs.keys())}")
            self.demographics_engine = DemographicsClassifier(
                configs=demo_configs,
                device=self.cfg.device,
                interval=self.cfg.demographics_interval
            )
            print("[Demographics] Initialized.")

        # MiDaS Depth Estimation
        self.depth_engine = None
        self._snapshot_depth_engine: DepthCalibrator | None = None
        self._depth_overlay_alpha_override: float | None = None
        self._depth_model_path: Path | None = None
        self._depth_saved_map_path: Path | None = None
        self._depth_init_failed = False
        if getattr(self.cfg, "use_depth", False):
            saved_map_cfg = str(getattr(self.cfg, "depth_saved_map", "")).strip()
            if getattr(self.cfg, "depth_use_saved_map", False) and saved_map_cfg:
                saved_map_path = resolve_existing_path(saved_map_cfg)
                if saved_map_path.exists():
                    try:
                        self.depth_engine = DepthCalibrator.from_saved_map(saved_map_path)
                        self._depth_saved_map_path = saved_map_path
                        print("[Depth] Regular run is using the saved calibration map; MiDaS model init skipped.")
                    except Exception as e:
                        print(f"[Depth] Failed to load saved depth map: {e}")
                else:
                    print(f"[Depth] Saved depth map requested but not found: {saved_map_path}")

            if self.depth_engine is None:
                print(f"[Depth] Initializing MiDaS model: {self.cfg.depth_model}")
                self._depth_model_path = self._resolve_depth_model_path()

        # Height-based age classifier (no ML model, pure perspective math)
        self.height_age_engine = HeightAgeClassifier(cfg) if self.cfg.detect_age else None
        self.pose_engine = None
        if self.cfg.detect_age and getattr(self.cfg, "use_pose_visibility_gate", False):
            try:
                print("[Pose] Initializing pose estimator...")
                self.pose_engine = PoseEstimator(cfg)
                print("[Pose] Initialized.")
            except Exception as e:
                print(f"[Warning] Pose model not available: {e}")
                print("         Pose visibility gate disabled. Run: python scripts/download_pose_model.py")


        # Face-based demographics classifier (OpenVINO: age + gender)
        self.face_age_engine = None
        needs_face_demographics = (
            (
                self.cfg.detect_age
                and getattr(self.cfg, "detect_face_age", False)
                and getattr(self.cfg, "age_use_face_override", False)
            )
            or (self.cfg.detect_gender and getattr(self.cfg, "gender_use_face_override", False))
        )
        if needs_face_demographics:
            try:
                print("[Face] Initializing face age/gender models...")
                self.face_age_engine = FaceAgeClassifier(cfg)
                print("[Face] Initialized.")
            except Exception as e:
                print(f"[Warning] Face age/gender model not available: {e}")
                print("         Face-based demographics disabled. Run: python scripts/download_face_age_models.py")

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

    def _resolve_depth_model_path(self) -> Path:
        """Select the configured MiDaS model, falling back safely when needed."""
        base_path = resolve_existing_path(self.cfg.depth_model)
        if not getattr(self.cfg, "depth_use_heavy_model", False):
            return base_path

        heavy_path = resolve_existing_path(
            getattr(self.cfg, "depth_heavy_model", self.cfg.depth_model)
        )
        if heavy_path.exists():
            return heavy_path

        print(
            f"[Depth] Heavy model requested but not found: {heavy_path}. "
            f"Falling back to {base_path.name}."
        )
        return base_path

    def _build_output_path(self) -> Path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        detection_mode = "group" if self.cfg.group_detect else "person"
        
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

        session_dir = resolve_output_path(f"output/{name}")
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
        out_dir = PROJECT_ROOT / "output"
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
            confidence_threshold=self.cfg.conf,
            overlap_threshold=self.cfg.iou,
        )
        boxes, confidence_scores, class_ids = filter_to_requested_classes(
            boxes, confidence_scores, class_ids, self.classes_to_keep
        )
        if self.cfg.show_person_boxes:
            draw_detection_boxes(
                frame,
                boxes,
                confidence_scores,
                class_ids,
                show_confidence=self.cfg.show_person_confidence,
                show_body=self.cfg.show_body,
                show_head=self.cfg.show_head,
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
        batch_dir = PROJECT_ROOT / "output" / f"[{batch_timestamp}] batch_session"
        batch_dir.mkdir(parents=True, exist_ok=True)

        for src in self.batch_sources:
            path = resolve_existing_path(src)
            if not path.exists():
                print(f"Skipping not found: {src}")
                continue
            
            self.source_path = path
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
        if self._is_stream:
            cap_input = self._stream_url
        elif self.mode in ("source", "batch"):
            cap_input = str(self.source_path)
        else:
            cap_input = self.camera_index
        cap = cv2.VideoCapture(cap_input)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(cap_input, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open input: {cap_input}")

        use_stable_tracking = self.cfg.stable or self.cfg.group_detect or self.cfg.detect_gender or self.cfg.detect_age
        
        perspective_detector = PerspectiveDetector(cfg=self.cfg)

        person_tracker = None
        if use_stable_tracking:
            person_tracker = PersonTracker(cfg=self.cfg)

        group_detector = None
        if self.cfg.group_detect:
            group_detector = GroupDetector(cfg=self.cfg)

        # We now initialize the VideoWriter lazily on the first processed frame
        # to ensure the resolution matches exactly after potential downscaling.
        writer = None

        mode_label = "video" if self.mode in ("source", "batch") else "camera"
        if is_batch:
            print(f"Running {mode_label} inference. Press 'q' to skip video, 'e' or ESC to stop all.")
        else:
            print(f"Running {mode_label} inference. Press 'q' or ESC to quit.")
        if not getattr(self.cfg, "save_output_video", True):
            print("[PERF] Output video saving disabled for faster interactive preview.")
        if getattr(self.cfg, "use_depth", False):
            print("[Depth] Controls: '[' more transparent, ']' less transparent, Space = snapshot.")
            if getattr(self.cfg, "depth_freeze_after_init", False):
                init_frames = max(1, int(getattr(self.cfg, "depth_init_frames", 1)))
                print(f"[Depth] Static map mode: freezing after {init_frames} calibration frame(s).")

        window_name = "OpenVINO YOLO Inference"
        if not self.cfg.no_show:
            mode = getattr(self.cfg, "preview_window_mode", "normal").lower()
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            
            if mode == 'fullscreen':
                # Borderless / Exclusive Fullscreen
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            elif mode == 'maximized':
                # Maximized with Title Bar (Windows Native)
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                if sys.platform.startswith("win"):
                    hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
                    if hwnd:
                        ctypes.windll.user32.ShowWindow(hwnd, 3) # SW_MAXIMIZE
            else:
                # Normal windowed mode
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, self.cfg.preview_width, self.cfg.preview_height)
            
            # --- Live Editor State ---
            editor_state = {
                "mode": "none",  # "none" or "roi"
                "dragging_idx": -1,
                "mouse_pos": (0, 0),
                "frame_w": self.cfg.preview_width,
                "frame_h": self.cfg.preview_height,
                "roi": list(getattr(self.cfg, "roi_polygon", [])),
            }
            
            def _norm_to_px(norm_pt, w, h):
                return (int(norm_pt[0] * w), int(norm_pt[1] * h))
                
            def mouse_callback(event, x, y, flags, param):
                editor_state["mouse_pos"] = (x, y)
                if editor_state["mode"] == "none": return
                
                h = editor_state["frame_h"]
                w = editor_state["frame_w"]
                if h <= 0 or w <= 0: return

                if editor_state["mode"] != "roi":
                    return
                current_shape = editor_state["roi"]

                if event == cv2.EVENT_LBUTTONDOWN:
                    min_dist = 60 # Hit hitbox generosity in pixels
                    best_idx = -1
                    for i, p_norm in enumerate(current_shape):
                        px, py = _norm_to_px(p_norm, w, h)
                        dist = ((px - x)**2 + (py - y)**2)**0.5
                        if dist < min_dist:
                            min_dist = dist
                            best_idx = i
                    if best_idx != -1:
                        editor_state["dragging_idx"] = best_idx
                        
                elif event == cv2.EVENT_LBUTTONUP:
                    editor_state["dragging_idx"] = -1
                    
                elif event == cv2.EVENT_MOUSEMOVE:
                    if editor_state["dragging_idx"] != -1:
                        nx = max(0.0, min(1.0, float(x) / w))
                        ny = max(0.0, min(1.0, float(y) / h))
                        current_shape[editor_state["dragging_idx"]] = (nx, ny)

            cv2.setMouseCallback(window_name, mouse_callback)

        last_ts = time.perf_counter()
        ema_fps = 0.0
        frame_count = 0
        last_boxes = np.empty((0, 4), dtype=np.float32)
        last_confidence_scores = np.empty((0,), dtype=np.float32)
        last_class_ids = np.empty((0,), dtype=np.int64)
        last_raw_boxes = np.empty((0, 4), dtype=np.float32)
        last_raw_class_ids = np.empty((0,), dtype=np.int64)
        processed_frame_count = 0
        snapshot_requested = False
        snapshot_overlay_frame: np.ndarray | None = None
        snapshot_overlay_remaining = 0
        
        end_all = False

        # ── Threaded frame grabber for live streams ──
        # For RTSP/camera: a background thread continuously reads frames so we
        # always get the LATEST frame, not a stale buffered one.
        import threading

        is_live = self._is_stream or self.mode == "camera"
        should_mimic_live = getattr(self.cfg, "mimic_live", False)

        class _FrameGrabber:
            def __init__(self, cap, target_fps=0):
                self._cap = cap
                self._interval = 1.0 / target_fps if target_fps > 0 else 0
                self._frame = None
                self._ok = False
                self._lock = threading.Lock()
                self._stopped = False
                self._thread = threading.Thread(target=self._run, daemon=True)
                self._thread.start()

            def _run(self):
                while not self._stopped:
                    t0 = time.perf_counter()
                    try:
                        ok, frame = self._cap.read()
                    except Exception:
                        break
                    with self._lock:
                        self._ok = ok
                        self._frame = frame
                    if not ok:
                        break
                    
                    if self._interval > 0:
                        elapsed = time.perf_counter() - t0
                        sleep_time = self._interval - elapsed
                        if sleep_time > 0:
                            time.sleep(sleep_time)

            def read(self):
                with self._lock:
                    return self._ok, self._frame

            def stop(self):
                self._stopped = True

        # For files in 'mimic_live' mode, we throttle reading to the file's FPS.
        # For true live streams (RTSP/Camera), we read as fast as possible to stay at the "head".
        target_fps = 0
        if should_mimic_live and not is_live:
            target_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        grabber = _FrameGrabber(cap, target_fps=target_fps) if (is_live or should_mimic_live) else None

        try:
            while True:
                # ── Handle Editing Pause ──
                if not self.cfg.no_show and editor_state["mode"] != "none":
                    # When editing, we do not read new frames, we just spin and redraw the current frame
                    if frame_count == 0:
                        continue # wait for at least one frame
                else:
                    if grabber:
                        ok, frame_read = grabber.read()
                        if frame_read is None:
                            continue  # Grabber hasn't read first frame yet
                    else:
                        ok, frame_read = cap.read()
                    if not ok:
                        print("Input ended; stopping.")
                        break
                    frame = self.undistorter.apply(frame_read) if self.undistorter is not None else frame_read
                # ── Handle Native Resolution (imgsz=0) ──
                if self.cfg.imgsz == 0 and frame_count == 0:
                    h_src, w_src = frame.shape[:2]
                    # Use the larger dimension for the square AI input
                    native_size = max(h_src, w_src)
                    self.detector.reshape(native_size)

                frame_count += 1

                # ── Global Optimization: Downscale frame if larger than imgsz ──
                # This drastically speeds up tracking, drawing, and encoding.
                # We do this early so ROI and writer both see the same coordinate system.
                if getattr(self.cfg, "downscale_to_imgsz", False) and self.cfg.imgsz > 0:
                    h0, w0 = frame.shape[:2]
                    if max(h0, w0) > self.cfg.imgsz:
                        scale = self.cfg.imgsz / max(h0, w0)
                        frame = cv2.resize(frame, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR)

                # ── ROI Handling ──────────────────────────────────────────────
                roi_meta: dict | None = None
                display_frame = frame.copy()
                if not self.cfg.no_show:
                    editor_state["frame_h"], editor_state["frame_w"] = display_frame.shape[:2]
                
                active_frame = display_frame
                
                # Override configured ROI with active edited ROI
                current_roi = editor_state["roi"] if not self.cfg.no_show else self.cfg.roi_polygon
                
                if self.cfg.use_roi and current_roi and len(current_roi) >= 3:
                    h_roi, w_roi = display_frame.shape[:2]
                    # Convert normalized to pixel coordinates
                    pts = np.array([(int(p[0] * w_roi), int(p[1] * h_roi)) for p in current_roi], np.int32)
                    rx, ry, rw, rh = cv2.boundingRect(pts)
                    
                    # Expand ROI by a percentage (padding) to give the detector context
                    # This prevents the "bug" where a person on the edge is only partially detected.
                    padding = getattr(self.cfg, "roi_padding", 0.20)
                    px_margin = int(max(rw, rh) * padding)
                    
                    rx_pad = max(0, rx - px_margin)
                    ry_pad = max(0, ry - px_margin)
                    rw_pad = min(w_roi - rx_pad, rw + 2 * px_margin)
                    rh_pad = min(h_roi - ry_pad, rh + 2 * px_margin)
                    
                    if rw_pad > 10 and rh_pad > 10:
                        active_frame = display_frame[ry_pad : ry_pad + rh_pad, rx_pad : rx_pad + rw_pad]
                        roi_meta = {"x": rx_pad, "y": ry_pad, "w": rw_pad, "h": rh_pad, "pts": pts}

                should_infer = (frame_count == 1) or (frame_count % self.skip_frames == 0)

                # Skip display/output for non-processed frames if requested
                if not should_infer and getattr(self.cfg, "output_processed_only", False):
                    continue

                if should_infer:
                    processed_frame_count += 1
                    t0 = time.perf_counter()
                    boxes, confidence_scores, class_ids = self.detector.detect_frame(
                        active_frame,
                        confidence_threshold=self.cfg.conf,
                        overlap_threshold=self.cfg.iou,
                    )
                    t_inf = (time.perf_counter() - t0) * 1000
                    
                    if getattr(self.cfg, "show_perf", True):
                        mode_str = f"ROI {active_frame.shape[1]}x{active_frame.shape[0]}" if roi_meta else f"Full {active_frame.shape[1]}x{active_frame.shape[0]}"
                        print(f"[PERF] Inference: {t_inf:.1f}ms | {mode_str}")
                    
                    # ── Map ROI coordinates back to full-frame ──
                    if roi_meta:
                        boxes[:, [0, 2]] += roi_meta["x"]
                        boxes[:, [1, 3]] += roi_meta["y"]

                    # ── Point-in-Polygon Filter ──
                    if roi_meta and len(boxes) > 0:
                        keep_roi = []
                        roi_poly = roi_meta["pts"]
                        for i, box in enumerate(boxes):
                            # Use foot base center as the point to check
                            px = (box[0] + box[2]) / 2
                            py = box[3]
                            if cv2.pointPolygonTest(roi_poly, (float(px), float(py)), False) >= 0:
                                keep_roi.append(i)
                        
                        if keep_roi:
                            boxes = boxes[keep_roi]
                            confidence_scores = confidence_scores[keep_roi]
                            class_ids = class_ids[keep_roi]
                        else:
                            boxes = np.empty((0, 4), dtype=np.float32)
                            confidence_scores = np.empty((0,), dtype=np.float32)
                            class_ids = np.empty((0,), dtype=np.int64)

                    raw_boxes, raw_class_ids = boxes.copy(), class_ids.copy()
                    boxes, confidence_scores, class_ids = filter_to_requested_classes(
                        boxes, confidence_scores, class_ids, self.classes_to_keep
                    )
                    last_boxes, last_confidence_scores, last_class_ids = boxes.copy(), confidence_scores.copy(), class_ids.copy()
                    last_raw_boxes, last_raw_class_ids = raw_boxes.copy(), raw_class_ids.copy()
                else:
                    boxes, confidence_scores, class_ids = last_boxes, last_confidence_scores, last_class_ids
                    raw_boxes, raw_class_ids = last_raw_boxes, last_raw_class_ids

                person_ids = None
                group_count = 0
                if person_tracker is not None:
                    person_tracker.update(display_frame, boxes, confidence_scores, class_ids)
                    boxes, confidence_scores, class_ids, person_ids = person_tracker.get_visible_people(
                        self.cfg.show_unconfirmed
                    )

                # ── MiDaS Depth Update ──
                if getattr(self.cfg, "use_depth", False):
                    depth_engine = self.depth_engine
                    depth_interval = max(1, int(getattr(self.cfg, "depth_interval", 1)))
                    freeze_after_init = bool(getattr(self.cfg, "depth_freeze_after_init", False))
                    depth_init_frames = max(1, int(getattr(self.cfg, "depth_init_frames", 1)))
                    if depth_engine is None:
                        should_init_depth = (
                            freeze_after_init
                            or len(boxes) > 0
                            or getattr(self.cfg, "show_depth_heatmap", False)
                        )
                        if should_init_depth:
                            depth_engine = self._ensure_live_depth_engine()
                    if depth_engine:
                        depth_engine.ensure_frame_shape(frame.shape[:2])
                        init_incomplete = depth_engine.update_count < depth_init_frames
                        depth_update_due = (
                            should_infer
                            and (
                                not depth_engine.is_calibrated
                                or (freeze_after_init and init_incomplete)
                                or (
                                    not freeze_after_init
                                    and (
                                        processed_frame_count == 1
                                        or processed_frame_count % depth_interval == 0
                                    )
                                )
                            )
                        )
                        needs_scene_calibration = freeze_after_init and init_incomplete
                        if depth_update_due and (
                            needs_scene_calibration
                            or len(boxes) > 0
                            or getattr(self.cfg, "show_depth_heatmap", False)
                        ):
                            # Reuse the last depth map between updates; floor depth changes slowly.
                            prev_updates = depth_engine.update_count
                            depth_engine.add_frame(frame, boxes if person_ids is not None else None)
                            if (
                                freeze_after_init
                                and prev_updates < depth_init_frames
                                and depth_engine.update_count >= depth_init_frames
                            ):
                                print(
                                    f"[Depth] Static map frozen after "
                                    f"{depth_engine.update_count} calibration frame(s)."
                                )
                                depth_engine.release_model()
                        if getattr(self.cfg, "show_depth_heatmap", False):
                            display_frame = depth_engine.blend_overlay(
                                display_frame,
                                alpha=self._depth_overlay_alpha(),
                            )

                if len(boxes) > 0:
                    current_view = perspective_detector.update(raw_boxes)
                else:
                    current_view = perspective_detector.get_current_view()

                # --- NEW: Demographics Computation BEFORE Group Detection ---
                demographics_labels = {}
                track_demographics = defaultdict(lambda: {"gender": None, "age": None, "attrs": []})

                # ── Height-based age classification ──
                age_labels: dict[int, str] = {}
                if person_ids is not None and len(boxes) > 0 and self.height_age_engine:
                    pose_observations = {}
                    if self.pose_engine is not None:
                        pose_observations = self.pose_engine.update(
                            display_frame,
                            boxes,
                            person_ids,
                            skip_track_ids=self.height_age_engine.pose_skip_track_ids,
                        )
                    age_labels = self.height_age_engine.update(
                        boxes, person_ids, display_frame.shape[0], display_frame.shape[1],
                        depth_engine=self.depth_engine,
                        pose_observations=pose_observations,
                    )
                    if self.pose_engine is not None and getattr(self.cfg, "show_pose_keypoints", False):
                        self.pose_engine.draw(display_frame, pose_observations)
                    self.height_age_engine.draw_measurements(display_frame)

                # ── Gender / Attributes (ML model) ──
                if person_ids is not None and len(boxes) > 0 and self.demographics_engine:
                    valid_indices = [i for i, b in enumerate(boxes) if (b[3]-b[1]) >= self.cfg.demographics_min_height]
                    if valid_indices:
                        f_boxes = boxes[valid_indices]
                        f_ids = person_ids[valid_indices]
                        demo_results = self.demographics_engine.update(frame, f_boxes, f_ids)
                        for tid_int, model_data in demo_results.items():
                            if 'gender' in model_data:
                                attrs = model_data['gender']
                                # Gender from Body (B)
                                g_label = None
                                g_conf = 0.0
                                if "gender_male" in attrs:
                                    g_conf = attrs.get("gender_male", 0.5)
                                    g_label = "Male" if g_conf > 0.5 else "Female"
                                    g_conf = g_conf if g_conf > 0.5 else (1.0 - g_conf)
                                elif "is_male" in attrs:
                                    g_conf = attrs.get("is_male", 0.5)
                                    g_label = "Male" if g_conf > 0.5 else "Female"
                                    g_conf = g_conf if g_conf > 0.5 else (1.0 - g_conf)
                                
                                if g_label and self.cfg.show_gender and g_conf >= self.cfg.demographics_min_conf:
                                    track_demographics[tid_int]["gender"] = ("B", g_label, g_conf)

                                # Attributes
                                if self.cfg.show_attributes and "has_hat" in attrs:
                                    if attrs.get("has_hat", 0.0) > self.cfg.attr_hat_threshold:
                                        track_demographics[tid_int]["attrs"].append("Hat")
                                    if attrs.get("has_bag", 0.0) > self.cfg.attr_bag_threshold:
                                        track_demographics[tid_int]["attrs"].append("Bag")
                                    if attrs.get("has_coat_jacket", 0.0) > self.cfg.attr_jacket_threshold:
                                        track_demographics[tid_int]["attrs"].append("Jacket")

                # ── Face-based age + gender classification ──
                face_age_labels: dict[int, str] = {}
                face_gender_labels: dict[int, str] = {}
                new_adult_samples: dict[int, tuple[float, float]] = {}
                if person_ids is not None and len(boxes) > 0 and self.face_age_engine:
                    face_age_labels, face_gender_labels, new_adult_samples = self.face_age_engine.update(
                        frame, boxes, person_ids
                    )

                # ── Face Overrides ──
                if face_gender_labels and self.cfg.show_gender and self.cfg.gender_use_face_override:
                    override_min_conf = float(
                        getattr(
                            self.cfg,
                            "gender_face_override_min_conf",
                            getattr(self.cfg, "face_min_conf", 0.7),
                        )
                    )
                    for tid_int, (face_gender, face_conf) in face_gender_labels.items():
                        if float(face_conf) < override_min_conf:
                            continue
                        # Face gender Priority: overwrite Body gender
                        track_demographics[tid_int]["gender"] = ("F", face_gender, face_conf)

                # ── Age and Label Construction ──
                if self.cfg.show_age:
                    use_face_age_override = bool(
                        self.cfg.detect_age
                        and getattr(self.cfg, "detect_face_age", False)
                        and getattr(self.cfg, "age_use_face_override", False)
                    )
                    # Priority 1: Face Age
                    # Priority 2: Height Age
                    face_age_ids = list(face_age_labels.keys()) if use_face_age_override else []
                    for tid_int in set(list(age_labels.keys()) + face_age_ids):
                        face_res = face_age_labels.get(tid_int) if use_face_age_override else None
                        height_res = age_labels.get(tid_int)     # (label, conf)
                        
                        age_out = None
                        if face_res:
                            age_out = ("F", face_res[0], face_res[1])
                        elif height_res:
                            age_out = ("H", height_res[0], height_res[1])
                        
                        if age_out:
                            track_demographics[tid_int]["age"] = age_out

                # ── Final Label Construction ──
                for tid_int, data in track_demographics.items():
                    parts = []
                    if data["gender"]:
                        src, val, conf = data["gender"]
                        parts.append(f"({src}) {val} {conf:.2f}")
                    if data["age"]:
                        src, val, conf = data["age"]
                        parts.append(f"({src}) {val} {conf:.2f}")
                    if data["attrs"]:
                        parts.append(f"[{', '.join(data['attrs'])}]")
                    if parts:
                        demographics_labels[tid_int] = " ".join(parts)

                group_progress = {}
                if group_detector is not None and person_ids is not None and len(boxes) > 0:
                    display_frame, group_count, group_progress = group_detector.draw_groups(
                        display_frame,
                        boxes,
                        person_ids,
                        current_view=current_view,
                        cls_ids=class_ids,
                        labels=self.labels,
                        raw_boxes=raw_boxes,
                        raw_cls_ids=raw_class_ids,
                        track_demographics=track_demographics, # Pass semantics
                    )

                if self.cfg.show_person_boxes or demographics_labels:
                    draw_detection_boxes(
                        display_frame,
                        boxes,
                        confidence_scores,
                        class_ids,
                        show_confidence=self.cfg.show_person_confidence,
                        show_boxes=self.cfg.show_person_boxes,
                        show_body=self.cfg.show_body,
                        show_head=self.cfg.show_head,
                        person_ids=person_ids,
                        demographics_labels=demographics_labels,
                        group_progress=group_progress,
                    )

                # ── Draw ROI Mask ──
                if self.cfg.show_roi_mask and current_roi and len(current_roi) >= 3:
                    h0, w0 = display_frame.shape[:2]
                    pts = np.array([(int(p[0] * w0), int(p[1] * h0)) for p in current_roi], np.int32)
                    
                    mask = np.zeros_like(display_frame)
                    cv2.fillPoly(mask, [pts], (0, 255, 0))
                    # Draw a border
                    cv2.polylines(display_frame, [pts], True, (0, 255, 0), 2)
                    # Blend the shaded mask (inverted to shade EXCLUDED areas or direct for INCLUDED)
                    # Let's shade the included area lightly
                    alpha = getattr(self.cfg, "roi_transparency", 0.15)
                    cv2.addWeighted(mask, alpha, display_frame, 1.0, 0, display_frame)
                    
                    if editor_state["mode"] == "roi":
                        for i, p_norm in enumerate(current_roi):
                            pt = _norm_to_px(p_norm, w0, h0)
                            color = (0, 255, 255) if i == editor_state["dragging_idx"] else (0, 0, 255)
                            dist = ((pt[0] - editor_state["mouse_pos"][0])**2 + (pt[1] - editor_state["mouse_pos"][1])**2)**0.5
                            radius = 16 if (dist < 60 or i == editor_state["dragging_idx"]) else 8
                            cv2.circle(display_frame, pt, radius, color, -1)
                            cv2.circle(display_frame, pt, radius, (255, 255, 255), 2)

                # ── Draw Tripline(s) ──
                if editor_state["mode"] != "none":
                    _ed_h = display_frame.shape[0]
                    _ed_fs = max(0.40, _ed_h / 1400)      # Slightly larger min scale
                    _ed_fs_sm = max(0.32, _ed_h / 1800)   # Slightly larger min scale
                    _ed_y1 = int(55 * (_ed_fs / 0.5))
                    _ed_y2 = _ed_y1 + int(25 * (_ed_fs / 0.5))
                    
                    mode_text = "ROI"
                    
                    # Construct text lines
                    line1 = f"EDITING {mode_text.upper()} (PAUSED) - Drag points. 's'/Enter=Save"
                    line2 = "Drag points. 's'/Enter=Save"

                    # Draw semi-transparent background for readability
                    def _draw_status_text(frame, text, x, y, scale, color, thick):
                        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
                        cv2.rectangle(frame, (x-5, y-th-5), (x+tw+5, y+baseline+5), (0,0,0), -1) # Black box
                        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

                    # We draw the boxes directly on display_frame (simple opaque boxes for maximum "understandability")
                    _draw_status_text(display_frame, line1, 12, _ed_y1, _ed_fs, (0, 0, 255), 1)
                    _draw_status_text(display_frame, line2, 12, _ed_y2, _ed_fs_sm, (0, 180, 255), 1)

                now = time.perf_counter()
                fps_raw = 1.0 / max(1e-6, now - last_ts)
                last_ts = now
                ema_fps = fps_raw if ema_fps == 0 else ema_fps * 0.9 + fps_raw * 0.1
                fps = ema_fps

                h_scale = display_frame.shape[0]
                fs = 0.3 if h_scale < 400 else 0.5 if h_scale < 720 else 0.6
                th = 1
                
                if getattr(self.cfg, "show_ui_overlay", True) or editor_state["mode"] != "none":
                    cv2.putText(
                        display_frame,
                        f"Detections: {len(boxes)}  Groups: {group_count} (Total: {len(group_detector.seen_frozen_groups) if group_detector else 0})  FPS: {fps:.1f}  View: {current_view}",
                        (12, int(35 * (fs/0.8))),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fs,
                        (0, 255, 255),
                        th,
                    )

                if snapshot_requested:
                    snapshot_requested = False
                    snapshot_boxes = boxes if person_ids is not None else None
                    overlay = self._capture_depth_snapshot(frame, snapshot_boxes, display_frame, frame_count)
                    if overlay is not None:
                        snapshot_overlay_frame = overlay
                        fps_display = getattr(self, "_fps_out", cap.get(cv2.CAP_PROP_FPS) or 30.0) or 30.0
                        overlay_frames = max(15, int(fps_display * 0.5))
                        snapshot_overlay_remaining = overlay_frames
                        print(f"[Depth Snapshot] Frame {frame_count} captured; overlay shows for {overlay_frames} frames.")
                    else:
                        snapshot_overlay_frame = None
                        snapshot_overlay_remaining = 0

                end_all = False
                if not self.cfg.no_show:
                    display_for_show = display_frame
                    if snapshot_overlay_remaining > 0 and snapshot_overlay_frame is not None:
                        display_for_show = snapshot_overlay_frame
                        snapshot_overlay_remaining -= 1
                        if snapshot_overlay_remaining <= 0:
                            snapshot_overlay_frame = None

                    cv2.imshow(window_name, display_for_show)
                    raw_key = cv2.waitKeyEx(1)
                    key = raw_key & 0xFF
                    
                    if raw_key != -1 and editor_state["mode"] != "none":
                       print(f"[DEBUG] Key pressed: {raw_key} (masked: {key})")

                    # --- Unified Key Handler (if/elif chain prevents collisions) ---
                    # 1. Global Session Control
                    if key == ord("e"):
                        end_all = True
                        break
                    elif key == ord("q") or (key == 27 and editor_state["mode"] == "none"):
                        break
                    
                    # 2. Mode Toggles
                    elif key == ord("r"):
                        print(f"[DEBUG] 'r' pressed. Mode: {editor_state['mode']} -> {'roi' if editor_state['mode'] != 'roi' else 'none'}")
                        editor_state["mode"] = "none" if editor_state["mode"] == "roi" else "roi"
                    elif key == ord("["):
                        new_alpha = self._adjust_depth_overlay_alpha(-0.05)
                        print(f"[Depth] Heatmap alpha {new_alpha:.2f} ({(1.0 - new_alpha) * 100:.0f}% transparent)")
                    elif key == ord("]"):
                        new_alpha = self._adjust_depth_overlay_alpha(0.05)
                        print(f"[Depth] Heatmap alpha {new_alpha:.2f} ({(1.0 - new_alpha) * 100:.0f}% transparent)")
                    elif key == ord(" "):
                        if editor_state["mode"] == "none":
                            snapshot_requested = True
                        else:
                            print("[Depth Snapshot] Finish editing before running a MiDaS snapshot.")
                    
                    # 3. Editor Commands (only active in editor modes)
                    elif editor_state["mode"] != "none":
                        # Save: 's', Enter (13), or 'v'
                        if key in (ord('s'), 13, ord('\r'), ord('v')):
                            self._save_editor_config(editor_state["roi"])
                            print("[Config] ROI saved to config.py")
                            editor_state["mode"] = "none"
                        # Cancel: ESC
                        elif key == 27:
                            editor_state["mode"] = "none"
                            editor_state["roi"] = list(getattr(self.cfg, "roi_polygon", []))
                            print("[Config] ROI edit cancelled.")
                        
                if editor_state["mode"] != "none":
                    continue # Do not write paused frames to video

                if writer is None and getattr(self.cfg, "save_output_video", True):
                    writer = self._build_video_writer(cap, display_frame)
                    # Track video timing for padding
                    self._frames_written = 0
                    self._start_time = time.perf_counter()
                    self._fps_out = cap.get(cv2.CAP_PROP_FPS) or 30.0

                if writer is not None:
                    if getattr(self.cfg, "mimic_live", False):
                        # --- Frame Padding Logic ---
                        # In mimic_live mode, we skip frames to stay real-time.
                        # To prevent the output MP4 from being "sped up", we must
                        # pad it with duplicates to match the elapsed real time.
                        elapsed = time.perf_counter() - self._start_time
                        expected_frames = int(elapsed * self._fps_out)
                        
                        # Never drop below 1 frame write
                        to_write = max(1, expected_frames - self._frames_written)
                        
                        for _ in range(to_write):
                            writer.write(display_frame)
                            self._frames_written += 1
                    else:
                        writer.write(display_frame)

            total_people = person_tracker.next_person_id - 1 if person_tracker else 0
            total_groups = group_detector.next_group_id - 1 if group_detector else 0
            print(f"\n[METRICS] {{\"people\": {total_people}, \"groups\": {total_groups}}}")
            
        finally:
            if grabber:
                grabber.stop()
            cap.release()
            if writer is not None:
                writer.release()
                print(f"Saved: {self.output_path}")
            if not is_batch:
                cv2.destroyAllWindows()
                
        return {"people": total_people, "groups": total_groups}, end_all

    def _ensure_snapshot_depth_engine(self) -> DepthCalibrator | None:
        """Lazily load MiDaS for manual snapshots (ignores `use_depth`)."""
        if self._snapshot_depth_engine:
            return self._snapshot_depth_engine

        depth_model_path = self._depth_model_path or self._resolve_depth_model_path()
        try:
            engine = DepthCalibrator(
                model_path=depth_model_path,
                device=self.cfg.device,
                ema_alpha=self.cfg.depth_ema_alpha,
            )
        except Exception as exc:
            print(f"[Depth Snapshot] Failed to load MiDaS: {exc}")
            return None

        self._snapshot_depth_engine = engine
        return engine

    def _ensure_live_depth_engine(self) -> DepthCalibrator | None:
        """Lazily load MiDaS for the live pipeline."""
        if self.depth_engine is not None:
            return self.depth_engine
        if self._depth_init_failed or not getattr(self.cfg, "use_depth", False):
            return None

        depth_model_path = self._depth_model_path or self._resolve_depth_model_path()
        try:
            engine = DepthCalibrator(
                model_path=depth_model_path,
                device=self.cfg.device,
                ema_alpha=self.cfg.depth_ema_alpha,
            )
        except BaseException as e:
            print(f"[Depth] Failed to load MiDaS: {type(e).__name__}: {e}")
            return None

        self._depth_model_path = depth_model_path
        self.depth_engine = engine
        return engine

    def _depth_overlay_alpha(self) -> float:
        """Return the active MiDaS heatmap opacity."""
        if self._depth_overlay_alpha_override is not None:
            return self._depth_overlay_alpha_override

        alpha = float(getattr(self.cfg, "depth_heatmap_alpha", 0.30))
        if getattr(self.cfg, "depth_heatmap_more_transparent", False):
            alpha = float(
                getattr(
                    self.cfg,
                    "depth_heatmap_alpha_more_transparent",
                    min(alpha, 0.15),
                )
            )
        return max(0.0, min(1.0, alpha))

    def _adjust_depth_overlay_alpha(self, delta: float) -> float:
        """Adjust MiDaS heatmap opacity at runtime."""
        alpha = max(0.0, min(1.0, self._depth_overlay_alpha() + float(delta)))
        self._depth_overlay_alpha_override = alpha
        return alpha

    def _capture_depth_snapshot(
        self,
        frame: np.ndarray,
        boxes: np.ndarray | None,
        display_frame: np.ndarray,
        frame_count: int,
    ) -> np.ndarray | None:
        """Run MiDaS on `frame` and prepare an overlay for UI feedback."""
        if self.depth_engine is not None and self.depth_engine.is_calibrated:
            overlay = self.depth_engine.blend_overlay(
                display_frame.copy(),
                alpha=self._depth_overlay_alpha(),
            )
        else:
            engine = self._ensure_snapshot_depth_engine()
            if engine is None:
                print("[Depth Snapshot] Depth model unavailable.")
                return None

            engine.add_frame(frame, boxes)
            overlay = engine.blend_overlay(
                display_frame.copy(),
                alpha=self._depth_overlay_alpha(),
            )
        label = f"MiDaS snapshot (frame {frame_count})"
        cv2.putText(
            overlay,
            label,
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return overlay

    def _save_editor_config(self, roi_polygon):
        """Rewrites config.py to save the live-edited ROI."""
        config_path = PROJECT_ROOT / "config.py"
        if not config_path.exists():
            print("\n❌ Could not find config.py to save editor settings.")
            return

        content = config_path.read_text(encoding="utf-8")
        
        import re
        
        # ── Save ROI ──
        roi_str = "(" + ", ".join([f"({p[0]:.4f}, {p[1]:.4f})" for p in roi_polygon]) + ")"
        roi_pattern = r'(roi_polygon:\s*tuple\[tuple\[float,\s*float\],\s*\.\.\.\]\s*=\s*).*'
        if re.search(roi_pattern, content):
            content = re.sub(roi_pattern, rf'\g<1>{roi_str}', content)
            self.cfg.roi_polygon = tuple(roi_polygon)
        

        config_path.write_text(content, encoding="utf-8")
        print("\n? Saved ROI to config.py!")

    def _build_video_writer(self, cap: cv2.VideoCapture, frame: np.ndarray) -> cv2.VideoWriter | None:
        h, w = frame.shape[:2]
        
        fps_src = cap.get(cv2.CAP_PROP_FPS)
        fps_src = fps_src if fps_src and fps_src > 0 else 30.0
        # Use 'mp4v' for maximum compatibility
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self.output_path), fourcc, fps_src, (w, h))
        if not writer.isOpened():
            print(f"[Error] Failed to initialize VideoWriter at {w}x{h}")
            return None
        return writer


def main() -> None:
    app = DetectionApp(OPENVINO_DEFAULTS)
    app.run()


if __name__ == "__main__":
    main()
