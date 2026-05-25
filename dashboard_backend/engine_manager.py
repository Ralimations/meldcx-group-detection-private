from __future__ import annotations

import dataclasses
import json
import sqlite3
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from config import OPENVINO_DEFAULTS, OpenVinoDefaults
from .config_store import DEFAULTS_DICT, dict_to_cfg
from .persistence import save_summary


class EngineManager:
    """Manages the lifecycle of DetectionApp and exposes a shared frame buffer."""

    def __init__(self, project_root: Path, db_path: Path):
        self._project_root = project_root
        self._db_path = db_path

        self._lock = threading.Lock()
        self._config_values: dict = dict(DEFAULTS_DICT)
        self._config_dirty: bool = False
        self._active_sensor_id: int | None = None

        self._frame: np.ndarray | None = None
        self._raw_frame: np.ndarray | None = None
        self._frame_jpeg: bytes | None = None
        self._frame_version: int = 0
        self._frame_lock = threading.Lock()
        self._raw_frame_version: int = 0

        self._thread: threading.Thread | None = None
        self._current_stop_event: threading.Event | None = None
        self._pause_event = threading.Event()
        self._playback_lock = threading.Lock()
        self._playback_is_file = False
        self._playback_playing = True
        self._playback_speed = 1.0
        self._playback_loop = True
        self._playback_seek_frame: int | None = None
        self._playback_current_frame = 0
        self._playback_total_frames = 0
        self._playback_fps = 0.0

        self._people: int = 0
        self._groups: int = 0
        self._total_groups: int = 0
        self._fps: float = 0.0

        self._session_start_time: float = 0.0
        self._session_max_people: int = 0
        self._session_max_groups: int = 0
        self._session_fps_acc: float = 0.0
        self._session_fps_count: int = 0
        self._session_source: str = ""
        self._session_output_dir: Path | None = None
        self._session_annotated_video_path: Path | None = None
        self._session_detection_json_path: Path | None = None
        self._session_detection_records: list[dict] = []

        self._logs: deque = deque(maxlen=200)
        self._tuning_frame: np.ndarray | None = None

        self._setup_logging_redirection()
        self._log("[Server] Dashboard server ready.")

    def _setup_logging_redirection(self):
        class StdoutRedirector:
            def __init__(self, manager, original_stream):
                self.manager = manager
                self.original_stream = original_stream

            def write(self, data):
                if data.strip():
                    self.manager._log_internal(data.strip())
                self.original_stream.write(data)

            def flush(self):
                self.original_stream.flush()

        sys.stdout = StdoutRedirector(self, sys.stdout)
        sys.stderr = StdoutRedirector(self, sys.stderr)

    def _log_internal(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self._logs.append({"time": ts, "msg": msg})

    def _log(self, msg: str):
        print(msg)

    def get_config(self) -> dict:
        with self._lock:
            return dict(self._config_values)

    def get_status_snapshot(self) -> dict:
        return {
            "running": self.is_running,
            "paused": self.is_paused,
            "people": self._people,
            "groups": self._groups,
            "total_groups": self._total_groups,
            "fps": self._fps,
        }

    def get_logs(self) -> list[dict]:
        return list(self._logs)

    def clear_logs(self) -> None:
        self._logs.clear()
        self._log("[Server] Logs cleared.")

    def get_active_sensor_id(self) -> int | None:
        return self._active_sensor_id

    def set_config(self, updates: dict):
        with self._lock:
            self._config_values.update(updates)
            self._config_dirty = True
            active_id = self._active_sensor_id

        if active_id is not None:
            self._save_sensor_config(active_id)

        self._log(f"[Config] Updated: {list(updates.keys())}")

    def _save_sensor_config(self, sensor_id: int):
        with self._lock:
            config_json = json.dumps(self._config_values)
        try:
            con = sqlite3.connect(str(self._db_path))
            con.execute("UPDATE sensors SET config_json = ? WHERE id = ?", (config_json, sensor_id))
            con.commit()
            con.close()
            self._log(f"[Sensor] Auto-saved config for sensor {sensor_id}.")
        except Exception as e:
            self._log(f"[Sensor] Auto-save failed: {e}")

    def reset_to_defaults(self) -> dict:
        with self._lock:
            self._config_values = dict(DEFAULTS_DICT)
            self._config_dirty = True
            active_id = self._active_sensor_id
        if active_id is not None:
            self._save_sensor_config(active_id)
        self._log("[Config] Reset to defaults.")
        return dict(self._config_values)

    def _build_cfg(self) -> OpenVinoDefaults:
        with self._lock:
            values = dict(self._config_values)
        return dict_to_cfg(OPENVINO_DEFAULTS, values)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_paused(self) -> bool:
        return self._pause_event.is_set()

    def start(self):
        with self._lock:
            if self.is_running:
                self._log("[Engine] Already running – ignoring start.")
                return

            self._current_stop_event = threading.Event()
            self._pause_event.clear()
            self._thread = threading.Thread(target=self._run_engine, args=(self._current_stop_event,), daemon=True)

        self._session_start_time = time.time()
        self._session_max_people = 0
        self._session_max_groups = 0
        self._session_fps_acc = 0.0
        self._session_fps_count = 0
        self._session_source = self.get_config().get("source", "Camera") if self.get_config().get("input_mode") == "source" else "Camera"
        if bool(self.get_config().get("save_session_artifacts", False)):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            source_name = str(self._session_source).strip()
            safe_source = Path(source_name).stem if source_name else "camera"
            safe_source = safe_source or "camera"
            self._session_output_dir = self._project_root / "output" / f"[{timestamp}] dashboard_{safe_source}"
            self._session_output_dir.mkdir(parents=True, exist_ok=True)
            self._session_annotated_video_path = self._session_output_dir / "annotated.mp4"
            self._session_detection_json_path = self._session_output_dir / "detections.json"
        else:
            self._session_output_dir = None
            self._session_annotated_video_path = None
            self._session_detection_json_path = None
        self._session_detection_records = []
        self._thread.start()
        self._log("[Engine] Started.")

    def stop(self):
        with self._lock:
            if not self._thread or not self._current_stop_event:
                self._reset_live_state()
                return True

            stop_ev = self._current_stop_event
            thread = self._thread
            self._log("[Engine] Stopping thread...")
            stop_ev.set()
            self._pause_event.clear()

        thread.join(timeout=5)

        with self._lock:
            if self._thread == thread and not thread.is_alive():
                self._thread = None
                self._current_stop_event = None
                self._reset_live_state()
                self._log("[Engine] Stopped.")
                return True

        self._log("[Engine] Stop requested, but thread is still shutting down.")
        return False

    def _reset_live_state(self):
        self._people = 0
        self._groups = 0
        self._total_groups = 0
        self._fps = 0.0
        with self._frame_lock:
            self._frame = None
            self._raw_frame = None
            self._frame_jpeg = None
            self._frame_version += 1
            self._raw_frame_version += 1
        with self._playback_lock:
            self._playback_is_file = False
            self._playback_playing = True
            self._playback_seek_frame = None
            self._playback_current_frame = 0
            self._playback_total_frames = 0
            self._playback_fps = 0.0

    def _update_stream_frame(self, frame: np.ndarray, cfg: OpenVinoDefaults):
        preview_w = max(1, int(getattr(cfg, "preview_width", 960) or 960))
        preview_h = max(1, int(getattr(cfg, "preview_height", 540) or 540))
        stream_frame = frame
        if frame.shape[1] != preview_w or frame.shape[0] != preview_h:
            stream_frame = cv2.resize(frame, (preview_w, preview_h), interpolation=cv2.INTER_LINEAR)

        ok, buf = cv2.imencode(".jpg", stream_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            return

        with self._frame_lock:
            self._frame = stream_frame
            self._frame_jpeg = buf.tobytes()
            self._frame_version += 1

    def get_frame_snapshot_jpeg(self) -> tuple[bytes | None, int]:
        with self._frame_lock:
            return self._frame_jpeg, self._frame_version

    def _save_session_summary(self):
        duration = time.time() - self._session_start_time
        avg_fps = self._session_fps_acc / self._session_fps_count if self._session_fps_count > 0 else 0.0
        if duration < 1.0:
            return

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            if self._session_detection_json_path is not None:
                payload = {
                    "timestamp": ts,
                    "source": self._session_source,
                    "max_people": self._session_max_people,
                    "max_groups": self._session_max_groups,
                    "avg_fps": avg_fps,
                    "duration": duration,
                    "records": self._session_detection_records,
                }
                self._session_detection_json_path.write_text(
                    json.dumps(payload, indent=2),
                    encoding="utf-8",
                )

            save_summary(
                self._db_path,
                timestamp=ts,
                source=self._session_source,
                max_people=self._session_max_people,
                max_groups=self._session_max_groups,
                avg_fps=avg_fps,
                duration=duration,
                annotated_video_path=(
                    str(self._session_annotated_video_path)
                    if self._session_annotated_video_path is not None and self._session_annotated_video_path.exists()
                    else None
                ),
                detection_json_path=(
                    str(self._session_detection_json_path)
                    if self._session_detection_json_path is not None and self._session_detection_json_path.exists()
                    else None
                ),
            )
            self._log(f"[Summary] Saved session: {self._session_source} ({self._session_max_people}p, {self._session_max_groups}g)")
        except Exception as e:
            self._log(f"[Summary] Save failed: {e}")

    def pause(self) -> bool:
        if self._pause_event.is_set():
            self._pause_event.clear()
            self._log("[Engine] Resumed.")
            return False
        self._pause_event.set()
        self._log("[Engine] Paused.")
        return True

    def reset(self):
        self._log("[Engine] Resetting…")
        stopped = self.stop()
        if not stopped and self.is_running:
            self._log("[Engine] Reset aborted because the previous thread is still running.")
            return
        time.sleep(0.5)
        self.start()

    def switch_input(self, mode: str, data):
        updates: dict = {"input_mode": mode}
        if mode == "camera":
            updates["camera_index"] = int(data)
        else:
            updates["source"] = str(data)
            updates["input_mode"] = "source"
        self.set_config(updates)
        if self.is_running:
            self.reset()

    def get_frame(self) -> np.ndarray | None:
        with self._frame_lock:
            return self._frame.copy() if self._frame is not None else None

    def get_raw_frame(self) -> np.ndarray | None:
        with self._frame_lock:
            return self._raw_frame.copy() if self._raw_frame is not None else None

    def get_frame_jpeg(self) -> tuple[bytes | None, int]:
        with self._frame_lock:
            return self._frame_jpeg, self._frame_version

    def get_playback_status(self) -> dict:
        with self._playback_lock:
            total_frames = self._playback_total_frames
            current_frame = self._playback_current_frame
            fps = self._playback_fps
            duration = (total_frames / fps) if fps > 0 and total_frames > 0 else 0.0
            position = (current_frame / fps) if fps > 0 and current_frame > 0 else 0.0
            return {
                "is_file": self._playback_is_file,
                "playing": self._playback_playing,
                "speed": self._playback_speed,
                "loop": self._playback_loop,
                "current_frame": current_frame,
                "total_frames": total_frames,
                "fps": fps,
                "position_seconds": position,
                "duration_seconds": duration,
            }

    def control_playback(
        self,
        *,
        playing: bool | None = None,
        seek_frame: int | None = None,
        speed: float | None = None,
        loop: bool | None = None,
        restart: bool = False,
    ) -> dict:
        with self._playback_lock:
            is_file = self._playback_is_file
            if not is_file:
                status = {
                    "is_file": False,
                    "playing": True,
                    "speed": 1.0,
                    "loop": False,
                    "current_frame": 0,
                    "total_frames": 0,
                    "fps": 0.0,
                    "position_seconds": 0.0,
                    "duration_seconds": 0.0,
                }
                return status
            if playing is not None:
                self._playback_playing = bool(playing)
            if speed is not None:
                self._playback_speed = max(0.25, min(float(speed), 4.0))
            if loop is not None:
                self._playback_loop = bool(loop)
            if restart:
                self._playback_seek_frame = 0
                self._playback_playing = True
                self._playback_current_frame = 0
            elif seek_frame is not None:
                max_frame = max(0, self._playback_total_frames - 1)
                clamped_frame = max(0, min(int(seek_frame), max_frame))
                self._playback_seek_frame = clamped_frame
                self._playback_current_frame = clamped_frame
        return self.get_playback_status()

    def capture_tuning_snapshot(self):
        with self._frame_lock:
            if self._raw_frame is not None:
                self._tuning_frame = self._raw_frame.copy()
                self._log("[Engine] Captured tuning snapshot (raw frame).")
            elif self._frame is not None:
                self._tuning_frame = self._frame.copy()
                self._log("[Engine] Captured tuning snapshot (corrected fallback).")
            else:
                self._log("[Engine] Failed to capture snapshot (no frame available).")

    def get_workspace_snapshot(self) -> np.ndarray | None:
        if self._tuning_frame is not None:
            return self._tuning_frame.copy()

        with self._frame_lock:
            if self._raw_frame is not None:
                return self._raw_frame.copy()
            if self._frame is not None:
                return self._frame.copy()
        return None

    def get_live_tuning_comparison(self, side: str = "both") -> np.ndarray | None:
        with self._frame_lock:
            if self._raw_frame is None:
                return None
            frame = self._raw_frame.copy()

        from scripts.configuration.image_undistorter import ImageUndistorter

        cfg = self._build_cfg()
        cfg = dataclasses.replace(cfg, undistort_enable=True)
        undistorter = ImageUndistorter(cfg=cfg)
        original = frame
        if side == "original":
            return original
        corrected = undistorter.apply(original)
        if side == "corrected":
            return corrected
        cv2.putText(original, "ORIGINAL", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(corrected, "CORRECTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        return np.hstack([original, corrected])

    def get_tuning_comparison(self, side: str = "both") -> np.ndarray | None:
        if self._tuning_frame is None:
            return None

        from scripts.configuration.image_undistorter import ImageUndistorter

        cfg = self._build_cfg()
        cfg = dataclasses.replace(cfg, undistort_enable=True)
        undistorter = ImageUndistorter(cfg=cfg)
        original = self._tuning_frame.copy()
        if side == "original":
            return original
        corrected = undistorter.apply(original)
        if side == "corrected":
            return corrected
        cv2.putText(original, "ORIGINAL", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(corrected, "CORRECTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        return np.hstack([original, corrected])

    def _run_engine(self, stop_event: threading.Event):
        try:
            cfg = self._build_cfg()
            cfg = dataclasses.replace(cfg, no_show=True, save_output_video=False)
        except Exception as e:
            self._log(f"[Engine] Config error: {e}")
            return

        if stop_event.is_set():
            return

        self._log("[Engine] Launching DetectionApp…")
        try:
            from run_openvino_yolo import DetectionApp

            app = DetectionApp(cfg)
        except BaseException as e:
            self._log(f"[Engine] Init failed: {type(e).__name__}: {e}")
            return

        self._log("[Engine] DetectionApp ready – starting stream loop.")
        try:
            self._stream_loop(app, cfg, stop_event)
        except Exception as e:
            self._log(f"[Engine] Runtime error: {e}")
            self._log(traceback.format_exc())
        except SystemExit as e:
            self._log(f"[Engine] Thread exit requested (SystemExit): {e}")
        finally:
            self._session_source = "None" if self._session_source == "" else self._session_source
            self._save_session_summary()
            with self._lock:
                if self._thread is threading.current_thread():
                    self._thread = None
                    self._current_stop_event = None
                self._reset_live_state()
            self._log("[Engine] Thread exiting.")

    def _stream_loop(self, app, cfg: OpenVinoDefaults, stop_event: threading.Event):
        from scripts.group_detector import GroupDetector
        from scripts.configuration.image_undistorter import ImageUndistorter
        from scripts.perspective_detector import PerspectiveDetector
        from scripts.person_detector import PersonTracker, draw_detection_boxes, filter_to_requested_classes

        mode = str(cfg.input_mode).strip().lower()
        source_str = str(cfg.source).strip().strip("'\"")
        if mode == "camera":
            cap_input = int(cfg.camera_index)
        elif source_str.lower().startswith(("rtsp://", "http://", "https://")):
            cap_input = source_str
        else:
            cap_input = str(self._project_root / source_str) if not Path(source_str).is_absolute() else source_str

        cap = cv2.VideoCapture(cap_input)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(cap_input, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self._log(f"[Engine] Cannot open source: {cap_input}")
            return

        is_live_source = mode == "camera" or source_str.lower().startswith(("rtsp://", "http://", "https://"))
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        with self._playback_lock:
            self._playback_is_file = not is_live_source
            self._playback_playing = True
            self._playback_speed = 1.0
            self._playback_loop = True
            self._playback_seek_frame = None
            self._playback_current_frame = 0
            self._playback_total_frames = total_frames if not is_live_source else 0
            self._playback_fps = source_fps if not is_live_source else 0.0

        use_tracking = cfg.stable or cfg.group_detect or cfg.detect_gender or cfg.detect_age
        perspective_detector = PerspectiveDetector(cfg=cfg)
        undistorter = ImageUndistorter(cfg=cfg)
        person_tracker = PersonTracker(cfg=cfg) if use_tracking else None
        group_detector = GroupDetector(cfg=cfg) if cfg.group_detect else None

        def point_in_normalized_roi(
            x: float,
            y: float,
            frame_w: int,
            frame_h: int,
            roi_polygon: tuple[tuple[float, float], ...] | list[tuple[float, float]] | None,
        ) -> bool:
            if not roi_polygon or len(roi_polygon) < 3 or frame_w <= 0 or frame_h <= 0:
                return True
            pts = np.array(
                [(int(round(px * frame_w)), int(round(py * frame_h))) for px, py in roi_polygon],
                dtype=np.int32,
            )
            return cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0

        def reset_runtime_processors(current_cfg: OpenVinoDefaults):
            tracking_enabled = current_cfg.stable or current_cfg.group_detect or current_cfg.detect_gender or current_cfg.detect_age
            tracker = PersonTracker(cfg=current_cfg) if tracking_enabled else None
            grouping = GroupDetector(cfg=current_cfg) if current_cfg.group_detect else None
            perspective = PerspectiveDetector(cfg=current_cfg)
            return perspective, tracker, grouping

        skip_frames = max(1, int(cfg.skip_frames)) if cfg.skip_frames > 0 else 1
        last_boxes = np.empty((0, 4), dtype=np.float32)
        last_conf = np.empty((0,), dtype=np.float32)
        last_cls = np.empty((0,), dtype=np.int64)
        last_raw_boxes = np.empty((0, 4), dtype=np.float32)
        last_raw_cls = np.empty((0,), dtype=np.int64)
        frame_count = 0
        last_ts = time.perf_counter()
        ema_fps = 0.0
        last_file_tick = time.perf_counter()
        session_video_writer: cv2.VideoWriter | None = None

        def ensure_session_video_writer(frame_bgr: np.ndarray) -> cv2.VideoWriter | None:
            nonlocal session_video_writer
            if session_video_writer is not None:
                return session_video_writer
            if self._session_annotated_video_path is None:
                return None
            h, w = frame_bgr.shape[:2]
            fps_out = source_fps if source_fps > 0 else 30.0
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(self._session_annotated_video_path), fourcc, fps_out, (w, h))
            if not writer.isOpened():
                self._log(f"[Summary] Failed to initialize annotated video writer at {w}x{h}")
                return None
            session_video_writer = writer
            return session_video_writer

        try:
            while not stop_event.is_set():
                while self._pause_event.is_set() and not stop_event.is_set():
                    time.sleep(0.1)
                if stop_event.is_set():
                    break

                pending_seek_frame = None
                file_playing = True
                file_speed = 1.0
                file_loop = True
                if not is_live_source:
                    with self._playback_lock:
                        pending_seek_frame = self._playback_seek_frame
                        if pending_seek_frame is not None:
                            self._playback_seek_frame = None
                        file_playing = self._playback_playing
                        file_speed = self._playback_speed
                        file_loop = self._playback_loop

                    if pending_seek_frame is not None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, float(pending_seek_frame))
                        last_boxes = np.empty((0, 4), dtype=np.float32)
                        last_conf = np.empty((0,), dtype=np.float32)
                        last_cls = np.empty((0,), dtype=np.int64)
                        last_raw_boxes = np.empty((0, 4), dtype=np.float32)
                        last_raw_cls = np.empty((0,), dtype=np.int64)
                        perspective_detector, person_tracker, group_detector = reset_runtime_processors(cfg)
                        frame_count = 0

                    if not file_playing and pending_seek_frame is None:
                        time.sleep(0.05)
                        continue

                if self._config_dirty:
                    with self._lock:
                        new_dict = dict(self._config_values)
                        self._config_dirty = False

                    old_cfg = cfg
                    cfg = dict_to_cfg(OPENVINO_DEFAULTS, new_dict)
                    app.detector.cfg = cfg
                    if app.height_age_engine is not None:
                        app.height_age_engine.cfg = cfg
                    if app.pose_engine is not None:
                        app.pose_engine.cfg = cfg
                    if person_tracker:
                        person_tracker.cfg = cfg
                    if group_detector:
                        group_detector.cfg = cfg

                    runtime_processor_keys = ("stable", "group_detect", "detect_gender", "detect_age", "perspective")
                    processors_changed = any(getattr(old_cfg, key) != getattr(cfg, key) for key in runtime_processor_keys)
                    if processors_changed:
                        perspective_detector, person_tracker, group_detector = reset_runtime_processors(cfg)
                        self._log("[Engine] Sync: Tracking/group processors rebuilt.")

                    lens_keys = [key for key in new_dict.keys() if key.startswith("undistort_")]
                    lens_changed = any(getattr(old_cfg, key) != getattr(cfg, key) for key in lens_keys)
                    if lens_changed:
                        undistorter = ImageUndistorter(cfg=cfg)
                        self._log("[Engine] Sync: Lens parameters updated.")

                    skip_frames = max(1, int(cfg.skip_frames)) if cfg.skip_frames > 0 else 1
                    self._log(f"[Engine] Sync: {len(new_dict)} parameters updated live.")

                ok, frame = cap.read()
                if not ok:
                    self._log("[Engine] Source ended – looping.")
                    if not is_live_source and file_loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        perspective_detector, person_tracker, group_detector = reset_runtime_processors(cfg)
                        last_boxes = np.empty((0, 4), dtype=np.float32)
                        last_conf = np.empty((0,), dtype=np.float32)
                        last_cls = np.empty((0,), dtype=np.int64)
                        last_raw_boxes = np.empty((0, 4), dtype=np.float32)
                        last_raw_cls = np.empty((0,), dtype=np.int64)
                        frame_count = 0
                        ok, frame = cap.read()
                        if not ok:
                            break
                    else:
                        break

                frame_count += 1
                with self._frame_lock:
                    self._raw_frame = frame.copy()
                    self._raw_frame_version += 1
                if not is_live_source:
                    with self._playback_lock:
                        pos_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
                        self._playback_current_frame = max(0, pos_frame - 1)
                if undistorter.enabled:
                    frame = undistorter.apply(frame)

                display_frame = frame.copy()
                roi_meta: dict | None = None
                active_frame = display_frame
                current_roi = getattr(cfg, "roi_polygon", ())
                current_height_roi = getattr(cfg, "height_roi_polygon", ())

                if getattr(cfg, "use_roi", False) and current_roi and len(current_roi) >= 3:
                    h_roi, w_roi = display_frame.shape[:2]
                    pts = np.array([(int(point[0] * w_roi), int(point[1] * h_roi)) for point in current_roi], np.int32)
                    rx, ry, rw, rh = cv2.boundingRect(pts)
                    padding = float(getattr(cfg, "roi_padding", 0.20))
                    px_margin = int(max(rw, rh) * padding)

                    rx_pad = max(0, rx - px_margin)
                    ry_pad = max(0, ry - px_margin)
                    rw_pad = min(w_roi - rx_pad, rw + 2 * px_margin)
                    rh_pad = min(h_roi - ry_pad, rh + 2 * px_margin)

                    if rw_pad > 10 and rh_pad > 10:
                        active_frame = display_frame[ry_pad: ry_pad + rh_pad, rx_pad: rx_pad + rw_pad]
                        roi_meta = {"x": rx_pad, "y": ry_pad, "pts": pts}

                should_infer = (frame_count == 1) or (frame_count % skip_frames == 0)

                if should_infer:
                    boxes, scores, cls_ids = app.detector.detect_frame(
                        active_frame,
                        confidence_threshold=cfg.person_detection_conf,
                        overlap_threshold=cfg.person_overlap_threshold,
                    )
                    if roi_meta and len(boxes) > 0:
                        boxes[:, [0, 2]] += roi_meta["x"]
                        boxes[:, [1, 3]] += roi_meta["y"]

                        keep_roi: list[int] = []
                        roi_poly = roi_meta["pts"]
                        for i, box in enumerate(boxes):
                            px = (box[0] + box[2]) / 2
                            py = box[3]
                            if cv2.pointPolygonTest(roi_poly, (float(px), float(py)), False) >= 0:
                                keep_roi.append(i)

                        if keep_roi:
                            boxes = boxes[keep_roi]
                            scores = scores[keep_roi]
                            cls_ids = cls_ids[keep_roi]
                        else:
                            boxes = np.empty((0, 4), dtype=np.float32)
                            scores = np.empty((0,), dtype=np.float32)
                            cls_ids = np.empty((0,), dtype=np.int64)

                    boxes, scores, cls_ids = filter_to_requested_classes(boxes, scores, cls_ids, app.classes_to_keep)
                    raw_boxes, raw_cls = boxes.copy(), cls_ids.copy()
                    last_boxes, last_conf, last_cls = boxes.copy(), scores.copy(), cls_ids.copy()
                    last_raw_boxes, last_raw_cls = raw_boxes.copy(), raw_cls.copy()
                else:
                    boxes, scores, cls_ids = last_boxes, last_conf, last_cls
                    raw_boxes, raw_cls = last_raw_boxes, last_raw_cls

                person_ids = None
                group_count = 0
                group_progress = {}
                demographics_labels: dict[int, str] = {}

                if person_tracker is not None:
                    person_tracker.update(display_frame, boxes, scores, cls_ids)
                    boxes, scores, cls_ids, person_ids = person_tracker.get_visible_people(cfg.show_unconfirmed)

                current_view = perspective_detector.update(raw_boxes) if len(boxes) > 0 else perspective_detector.get_current_view()

                age_processing_needed = (
                    bool(getattr(cfg, "detect_age", False))
                    and bool(getattr(cfg, "show_age", True))
                    and app.height_age_engine is not None
                )
                if person_ids is not None and len(boxes) > 0 and age_processing_needed:
                    pose_observations = {}
                    if app.pose_engine is not None:
                        frame_h, frame_w = display_frame.shape[:2]
                        roi_skip_ids: set[int] = set()
                        for box, tid_raw in zip(boxes, person_ids):
                            foot_x = float(box[0] + box[2]) * 0.5
                            foot_y = float(box[3])
                            if not point_in_normalized_roi(foot_x, foot_y, frame_w, frame_h, current_height_roi):
                                roi_skip_ids.add(int(tid_raw))
                        pose_observations = app.pose_engine.update(
                            display_frame,
                            boxes,
                            person_ids,
                            skip_track_ids=app.height_age_engine.pose_skip_track_ids | roi_skip_ids,
                        )
                    age_labels = app.height_age_engine.update(
                        boxes,
                        person_ids,
                        display_frame.shape[0],
                        display_frame.shape[1],
                        pose_observations=pose_observations,
                        height_roi_polygon=current_height_roi,
                    )
                    if app.pose_engine is not None and getattr(cfg, "show_pose_keypoints", False):
                        app.pose_engine.draw(display_frame, pose_observations)
                    app.height_age_engine.draw_measurements(display_frame)
                    demographics_labels = {
                        int(track_id): label_text
                        for track_id, (label_text, _conf) in age_labels.items()
                    }

                if group_detector is not None and person_ids is not None and len(boxes) > 0:
                    display_frame, group_count, group_progress = group_detector.draw_groups(
                        display_frame,
                        boxes,
                        person_ids,
                        current_view=current_view,
                        cls_ids=cls_ids,
                        labels=app.labels,
                        raw_boxes=raw_boxes,
                        raw_cls_ids=raw_cls,
                        track_demographics={},
                    )

                if cfg.show_body or cfg.show_head or demographics_labels:
                    draw_detection_boxes(
                        display_frame,
                        boxes,
                        scores,
                        cls_ids,
                        show_body_confidence=cfg.show_body_confidence,
                        show_head_confidence=cfg.show_head_confidence,
                        show_body=cfg.show_body,
                        show_head=cfg.show_head,
                        person_ids=person_ids,
                        demographics_labels=demographics_labels,
                        group_progress=group_progress,
                        label_default_color=getattr(cfg, "label_default_color", "#FFFF00"),
                        label_segment_colors=getattr(cfg, "label_segment_colors", ""),
                    )

                if getattr(cfg, "show_roi_mask", False) and current_roi and len(current_roi) >= 3:
                    h0, w0 = display_frame.shape[:2]
                    pts = np.array([(int(point[0] * w0), int(point[1] * h0)) for point in current_roi], np.int32)
                    mask = np.zeros_like(display_frame)
                    cv2.fillPoly(mask, [pts], (0, 255, 0))
                    cv2.polylines(display_frame, [pts], True, (0, 255, 0), 2)
                    alpha = float(getattr(cfg, "roi_transparency", 0.15))
                    cv2.addWeighted(mask, alpha, display_frame, 1.0, 0, display_frame)

                if getattr(cfg, "show_height_roi_mask", True) and current_height_roi and len(current_height_roi) >= 3:
                    h0, w0 = display_frame.shape[:2]
                    pts = np.array([(int(point[0] * w0), int(point[1] * h0)) for point in current_height_roi], np.int32)
                    mask = np.zeros_like(display_frame)
                    cv2.fillPoly(mask, [pts], (255, 191, 0))
                    cv2.polylines(display_frame, [pts], True, (255, 191, 0), 2)
                    alpha = float(getattr(cfg, "height_roi_transparency", 0.03))
                    cv2.addWeighted(mask, alpha, display_frame, 1.0, 0, display_frame)

                now = time.perf_counter()
                fps_raw = 1.0 / max(1e-6, now - last_ts)
                last_ts = now
                ema_fps = fps_raw if ema_fps == 0 else ema_fps * 0.9 + fps_raw * 0.1

                if cfg.show_ui_overlay:
                    cv2.putText(
                        display_frame,
                        f"People: {len(boxes)}  Groups: {group_count}  FPS: {ema_fps:.1f}  [{current_view}]",
                        (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        1,
                    )

                self._people = len(boxes)
                self._groups = group_count
                self._total_groups = len(group_detector.seen_frozen_groups) if group_detector is not None else 0
                self._fps = round(ema_fps, 1)
                self._session_max_people = max(self._session_max_people, len(boxes))
                self._session_max_groups = max(self._session_max_groups, group_count)
                self._session_fps_acc += ema_fps
                self._session_fps_count += 1

                detection_items: list[dict] = []
                for index, box in enumerate(boxes):
                    track_id = int(person_ids[index]) if person_ids is not None and index < len(person_ids) else None
                    detection_items.append(
                        {
                            "track_id": track_id,
                            "bbox": [float(coord) for coord in box.tolist()],
                            "score": float(scores[index]) if index < len(scores) else None,
                            "class_id": int(cls_ids[index]) if index < len(cls_ids) else None,
                            "label": demographics_labels.get(track_id) if track_id is not None else None,
                        }
                    )
                if self._session_detection_json_path is not None:
                    self._session_detection_records.append(
                        {
                            "frame_index": int(frame_count),
                            "timestamp_sec": float(
                                (self._playback_current_frame / self._playback_fps)
                                if self._playback_fps > 0 and self._playback_current_frame >= 0
                                else 0.0
                            ),
                            "people": int(len(boxes)),
                            "groups": int(group_count),
                            "view": str(current_view),
                            "detections": detection_items,
                        }
                    )

                writer = ensure_session_video_writer(display_frame)
                if writer is not None:
                    writer.write(display_frame)

                self._update_stream_frame(display_frame, cfg)

                if not is_live_source and source_fps > 0:
                    target_delay = max(0.0, (1.0 / source_fps) / max(file_speed, 0.25))
                    elapsed = time.perf_counter() - last_file_tick
                    if elapsed < target_delay:
                        time.sleep(target_delay - elapsed)
                    last_file_tick = time.perf_counter()
        finally:
            if session_video_writer is not None:
                session_video_writer.release()
            cap.release()
            self._reset_live_state()
