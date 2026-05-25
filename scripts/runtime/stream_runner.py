#!/usr/bin/env python3
"""Stream execution for the runtime pipeline."""

from __future__ import annotations

import ctypes
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from scripts.person_detector import draw_detection_boxes, filter_to_requested_classes
from scripts.runtime.archive_recorder import RollingVideoRecorder
from scripts.runtime.group_event_logger import GroupEventLogger
from scripts.configuration.runtime_config import DB_PATH
from scripts.runtime.overlay_renderer import draw_runtime_overlays
from scripts.runtime.preview_ui import (
    bind_editor_mouse,
    build_editor_state,
    configure_preview_window,
    handle_preview_key,
    norm_to_px,
)
from scripts.runtime.source_io import FrameGrabber, open_video_capture
from scripts.runtime.engine_factory import create_stream_processors
from scripts.runtime.demographics_labels import body_gender_from_attrs


PROJECT_ROOT = Path(__file__).parent.parent.parent


def resolve_output_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def draw_face_boxes(frame: np.ndarray, faces: list[tuple[int, int, int, int, float]], *, show_confidence: bool) -> None:
    if not faces:
        return
    h_scale = frame.shape[0]
    fs = 0.3 if h_scale < 400 else 0.4 if h_scale < 720 else 0.5
    thickness = 1
    color = (255, 255, 0)
    for x1, y1, x2, y2, conf in faces:
        x1_i, y1_i, x2_i, y2_i = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), color, thickness)
        label = f"Face {float(conf):.2f}" if show_confidence else "Face"
        text_org = (max(0, x1_i), max(12, y1_i - 6))
        cv2.putText(
            frame,
            label,
            text_org,
            cv2.FONT_HERSHEY_SIMPLEX,
            fs,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            label,
            text_org,
            cv2.FONT_HERSHEY_SIMPLEX,
            fs,
            color,
            thickness,
            cv2.LINE_AA,
        )


def run_stream(app, is_batch: bool = False) -> tuple[dict[str, int], bool]:
    self = app
    if self._is_stream:
        cap_input = self._stream_url
    elif self.mode in ("source", "batch"):
        cap_input = str(self.source_path)
    else:
        cap_input = self.camera_index
    cap = open_video_capture(cap_input)

    perspective_detector, person_tracker, group_detector = create_stream_processors(self.cfg)

    # We now initialize the VideoWriter lazily on the first processed frame
    # to ensure the resolution matches exactly after potential downscaling.
    writer = None
    writer_last_source_index = 0
    writer_last_frame = None

    mode_label = "video" if self.mode in ("source", "batch") else "camera"
    if is_batch:
        print(f"Running {mode_label} inference. Press 'q' to skip video, 'e' or ESC to stop all.")
    else:
        print(f"Running {mode_label} inference. Press 'q' or ESC to quit.")
    if not getattr(self.cfg, "save_output_video", True):
        print("[PERF] Output video saving disabled for faster interactive preview.")

    window_name = "OpenVINO YOLO Inference"
    editor_state = build_editor_state(self.cfg)

    if not self.cfg.no_show:
        configure_preview_window(window_name, self.cfg, sys_module=sys, ctypes_module=ctypes)
        bind_editor_mouse(window_name, editor_state)

    ema_fps = 0.0
    frame_count = 0
    last_boxes = np.empty((0, 4), dtype=np.float32)
    last_confidence_scores = np.empty((0,), dtype=np.float32)
    last_class_ids = np.empty((0,), dtype=np.int64)
    last_raw_boxes = np.empty((0, 4), dtype=np.float32)
    last_raw_class_ids = np.empty((0,), dtype=np.int64)
    processed_frame_count = 0
    nvr_recorder: RollingVideoRecorder | None = None
    group_event_logger: GroupEventLogger | None = None

    end_all = False

    is_live = self._is_stream or self.mode == "camera"
    should_mimic_live = getattr(self.cfg, "mimic_live", False)
    is_finite_file = not is_live
    total_input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if is_finite_file else 0

    # Live sources always use the latest-frame grabber. File sources also
    # use it in mimic_live mode so processing stays near the newest frame
    # instead of falling behind on high-FPS clips.
    target_fps = 0.0
    if should_mimic_live and is_finite_file:
        target_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    grabber = FrameGrabber(cap, target_fps=target_fps) if (is_live or should_mimic_live) else None
    source_fps_display = float(target_fps) if target_fps > 0 else float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    # When a throttled grabber is driving file playback, do not also sleep
    # in the preview loop or the file will be paced twice.
    mimic_live_interval = 0.0
    if should_mimic_live and is_finite_file and grabber is None:
        _src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        mimic_live_interval = 1.0 / _src_fps
    mimic_live_last = time.perf_counter()

    try:
        while True:
            # ── Handle Editing Pause ──
            if not self.cfg.no_show and editor_state["mode"] != "none":
                # When editing, we do not read new frames, we just spin and redraw the current frame
                if frame_count == 0:
                    continue # wait for at least one frame
            else:
                if grabber:
                    ok, frame_read, grabber_eof = grabber.read()
                    if frame_read is None:
                        if grabber_eof:
                            print("Input ended; stopping.")
                            break
                        continue  # Grabber hasn't read first frame yet
                else:
                    ok, frame_read = cap.read()
                if not ok:
                    print("Input ended; stopping.")
                    break
                frame = self.undistorter.apply(frame_read) if self.undistorter is not None else frame_read
                processing_started_at = time.perf_counter()
            source_frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
            is_last_file_frame = bool(
                is_finite_file
                and total_input_frames > 0
                and source_frame_index >= total_input_frames
            )
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
            current_height_roi = (
                editor_state["height_roi"]
                if not self.cfg.no_show
                else getattr(self.cfg, "height_roi_polygon", self.cfg.roi_polygon)
            )
            current_doorway_roi = (
                editor_state["doorway_roi"] if not self.cfg.no_show else getattr(self.cfg, "doorway_roi_polygon", ())
            )

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
                    confidence_threshold=self.cfg.person_detection_conf,
                    overlap_threshold=self.cfg.person_overlap_threshold,
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

            # ── Perspective / height-age update ──
            if len(boxes) > 0:
                current_view = perspective_detector.update(raw_boxes)
            else:
                current_view = perspective_detector.get_current_view()

            # --- NEW: Demographics Computation BEFORE Group Detection ---
            demographics_labels = {}
            track_demographics = defaultdict(
                lambda: {"gender": None, "age": None, "attrs": [], "raw_par": [], "raw_face": []}
            )
            pose_observations = {}
            if person_ids is not None and len(boxes) > 0 and self.pose_engine is not None:
                body_mask = np.array([int(class_id) == 0 for class_id in class_ids], dtype=bool)
                pose_boxes = boxes[body_mask]
                pose_person_ids = person_ids[body_mask]
                force_pose = bool(getattr(self.cfg, "pose_force_run", False))
                if len(pose_boxes) > 0 and self.height_age_engine is not None and not force_pose:
                    pose_candidate_ids = self.height_age_engine.pose_candidate_track_ids(
                        pose_boxes,
                        pose_person_ids,
                        display_frame.shape[0],
                        display_frame.shape[1],
                        roi_polygon=current_height_roi,
                    )
                    pose_skip_ids = self.height_age_engine.pose_skip_track_ids | (
                        set(int(tid) for tid in pose_person_ids) - pose_candidate_ids
                    )
                else:
                    pose_skip_ids = set()
                if len(pose_boxes) > 0:
                    pose_observations = self.pose_engine.update(
                        frame,
                        pose_boxes,
                        pose_person_ids,
                        skip_track_ids=pose_skip_ids,
                    )
                    if getattr(self.cfg, "show_pose_keypoints", False):
                        self.pose_engine.draw(display_frame, pose_observations)

            # ── Height-based age classification ──
            age_labels: dict[int, str] = {}
            if person_ids is not None and len(boxes) > 0 and self.height_age_engine:
                body_mask = np.array([int(class_id) == 0 for class_id in class_ids], dtype=bool)
                age_boxes = boxes[body_mask]
                age_person_ids = person_ids[body_mask]
                age_labels = self.height_age_engine.update(
                    age_boxes, age_person_ids, display_frame.shape[0], display_frame.shape[1],
                    pose_observations=pose_observations,
                    roi_polygon=current_height_roi,
                    frame_index=frame_count,
                )
                self.height_age_engine.draw_measurements(display_frame)

            # ── Gender / Attributes (ML model) ──
            if person_ids is not None and len(boxes) > 0 and self.demographics_engine:
                body_mask = np.array([int(class_id) == 0 for class_id in class_ids], dtype=bool)
                f_boxes = boxes[body_mask]
                f_ids = person_ids[body_mask]
                if len(f_boxes) > 0:
                    demo_results = self.demographics_engine.update(frame, f_boxes, f_ids)
                    for tid_int, model_data in demo_results.items():
                        if 'gender' in model_data:
                            attrs = model_data['gender']
                            if getattr(self.cfg, "show_body_par_raw_labels", False):
                                raw_positive = [
                                    f"{attr_name}:{float(attr_score):.2f}"
                                    for attr_name, attr_score in attrs.items()
                                    if (
                                        float(attr_score) >= self.cfg.demographics_min_conf
                                        or str(attr_name).strip().lower() == "has_longhair"
                                    )
                                ]
                                track_demographics[tid_int]["raw_par"] = raw_positive
                            # Gender from Body (B)
                            body_gender = body_gender_from_attrs(
                                attrs,
                                min_conf=self.cfg.demographics_min_conf,
                            )

                            if body_gender is not None:
                                g_label, g_conf = body_gender
                                if self.cfg.show_gender and g_conf >= self.cfg.demographics_min_conf:
                                    track_demographics[tid_int]["gender"] = ("B", g_label, g_conf)


            # ── Face-based age + gender classification ──
            face_age_labels: dict[int, str] = {}
            face_gender_labels: dict[int, str] = {}
            new_adult_samples: dict[int, tuple[float, float]] = {}
            if person_ids is not None and len(boxes) > 0 and self.face_age_engine:
                face_mask = np.array([int(class_id) in (0, 1) for class_id in class_ids], dtype=bool)
                face_boxes = boxes[face_mask]
                face_person_ids = person_ids[face_mask]
                face_age_labels, face_gender_labels, new_adult_samples = self.face_age_engine.update(
                    frame,
                    face_boxes,
                    face_person_ids,
                )
                if getattr(self.cfg, "show_face_analysis_unlocked", False):
                    for tid_int, raw_face in getattr(self.face_age_engine, "last_raw_results", {}).items():
                        raw_parts: list[str] = []
                        if self.cfg.show_gender:
                            raw_parts.append(
                                f"(F*) {raw_face['gender']} {float(raw_face['gender_conf']):.2f}"
                            )
                        if self.cfg.show_age:
                            raw_parts.append(f"(F*) {float(raw_face['age']):.1f}")
                        track_demographics[tid_int]["raw_face"] = raw_parts

            # ── Face Overrides ──
            if face_gender_labels and self.cfg.show_gender and self.cfg.gender_use_face_override:
                override_min_conf = float(
                    getattr(
                        self.cfg,
                        "gender_face_override_min_conf",
                        0.7,
                    )
                )
                for tid_int, (face_gender, face_conf) in face_gender_labels.items():
                    if float(face_conf) < override_min_conf:
                        continue
                    # Face gender Priority: overwrite Body gender
                    track_demographics[tid_int]["gender"] = ("F", face_gender, face_conf)

            if self.face_age_engine is not None and getattr(self.cfg, "show_face", False):
                draw_face_boxes(
                    display_frame,
                    getattr(self.face_age_engine, "last_detected_faces", []),
                    show_confidence=bool(getattr(self.cfg, "show_face_confidence", False)),
                )

            active_track_ids = set(int(tid) for tid in person_ids) if person_ids is not None else set()
            demographics_memory_ttl = max(1, int(getattr(self.cfg, "track_reid_window", getattr(self.cfg, "max_misses", 30))))
            stale_gender_ids = set(self._gender_cache.keys()) - active_track_ids
            for stale_id in stale_gender_ids:
                self._gender_cache_stale[stale_id] = self._gender_cache_stale.get(stale_id, 0) + 1
                if self._gender_cache_stale[stale_id] > demographics_memory_ttl:
                    self._gender_cache.pop(stale_id, None)
                    self._gender_cache_stale.pop(stale_id, None)
            for active_id in active_track_ids:
                self._gender_cache_stale[active_id] = 0

            if self.cfg.show_gender:
                for tid_int in active_track_ids:
                    gender_data = track_demographics[tid_int]["gender"]
                    if gender_data is not None:
                        self._gender_cache[tid_int] = gender_data
                        self._gender_cache_stale[tid_int] = 0
                    elif tid_int in self._gender_cache:
                        track_demographics[tid_int]["gender"] = self._gender_cache[tid_int]

            # ── Age and Label Construction ──
            if self.cfg.show_age:
                use_face_age_override = bool(
                    self.cfg.detect_age
                    and getattr(self.cfg, "enable_face_analysis", False)
                    and getattr(self.cfg, "face_age_override", False)
                )
                # Priority 1: Face Age
                # Priority 2: Body PAR Age
                # Priority 3: Height Age
                face_age_ids = list(face_age_labels.keys()) if use_face_age_override else []
                body_age_ids = [tid_int for tid_int, data in track_demographics.items() if data["age"] is not None]
                age_candidate_ids = (
                    set(int(tid) for tid in active_track_ids)
                    | set(int(tid) for tid in age_labels.keys())
                    | set(int(tid) for tid in face_age_ids)
                    | set(int(tid) for tid in body_age_ids)
                )
                for tid_int in age_candidate_ids:
                    face_res = face_age_labels.get(tid_int) if use_face_age_override else None
                    height_res = age_labels.get(tid_int)     # (label, conf)
                    height_placeholder = bool(
                        height_res
                        and str(height_res[0]).startswith("Height [")
                    )

                    age_out = None
                    if face_res:
                        age_out = ("F", face_res[0], face_res[1])
                    elif track_demographics[tid_int]["age"] is not None:
                        age_out = track_demographics[tid_int]["age"]
                    elif height_res and not height_placeholder:
                        age_out = ("H", height_res[0], height_res[1])
                    elif tid_int in self._age_cache:
                        age_out = self._age_cache[tid_int]
                    elif height_res:
                        age_out = ("H", height_res[0], height_res[1])

                    if age_out:
                        track_demographics[tid_int]["age"] = age_out
                        if not (age_out[0] == "H" and str(age_out[1]).startswith("Height [")):
                            self._age_cache[tid_int] = age_out
                            self._age_cache_stale[tid_int] = 0
                    elif tid_int in self._age_cache:
                        track_demographics[tid_int]["age"] = self._age_cache[tid_int]

            stale_age_ids = set(self._age_cache.keys()) - active_track_ids
            for stale_id in stale_age_ids:
                self._age_cache_stale[stale_id] = self._age_cache_stale.get(stale_id, 0) + 1
                if self._age_cache_stale[stale_id] > demographics_memory_ttl:
                    self._age_cache.pop(stale_id, None)
                    self._age_cache_stale.pop(stale_id, None)
            for active_id in active_track_ids:
                self._age_cache_stale[active_id] = 0

            # ── Final Label Construction ──
            for tid_int, data in track_demographics.items():
                parts = []
                if data["gender"]:
                    src, val, conf = data["gender"]
                    if self.cfg.show_gender_confidence:
                        parts.append(f"({src}) {val} {conf:.2f}")
                    else:
                        parts.append(f"({src}) {val}")
                if data["age"]:
                    src, val, conf = data["age"]
                    if self.cfg.show_age_confidence:
                        parts.append(f"({src}) {val} {conf:.2f}")
                    else:
                        parts.append(f"({src}) {val}")
                if data["attrs"]:
                    parts.append(f"[{', '.join(data['attrs'])}]")
                if data["raw_par"]:
                    parts.append("{" + ", ".join(data["raw_par"]) + "}")
                if data["raw_face"]:
                    parts.append("<" + ", ".join(data["raw_face"]) + ">")
                if parts:
                    demographics_labels[tid_int] = " ".join(parts)

            group_progress = {}
            group_snapshots = {}
            room_trip_statuses = []
            carry_alert_statuses = []
            interaction_now_ts = time.time()
            boxes_by_person_id = {
                int(tid): tuple(float(v) for v in box)
                for box, tid in zip(boxes, person_ids)
            } if person_ids is not None and len(boxes) > 0 else {}
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
                    pose_observations=pose_observations,
                    now_ts=interaction_now_ts,
                )
                group_snapshots = group_detector.get_group_snapshots()
                if getattr(self.cfg, "show_carry_status", False):
                    for snapshot in group_snapshots.values():
                        if not bool(snapshot.get("interaction_locked", False)):
                            continue
                        member_ids = tuple(int(tid) for tid in snapshot.get("member_ids", ()))
                        group_id = int(snapshot.get("group_id", 0))
                        elapsed_seconds = max(0.0, float(snapshot.get("interaction_elapsed_seconds", 0.0)))
                        if group_id > 0:
                            label = (
                                f"Carry Alert: Group {group_id} Minor on Adult Lap "
                                f"{elapsed_seconds:.1f}s"
                            )
                        else:
                            member_text = ", ".join(str(tid) for tid in member_ids) if member_ids else "unknown"
                            label = (
                                f"Carry Alert: Pair [{member_text}] Minor on Adult Lap "
                                f"{elapsed_seconds:.1f}s"
                            )
                        carry_alert_statuses.append(
                            {
                                "text": label,
                                "color": (0, 140, 255),
                            }
                        )
            if getattr(self.cfg, "use_doorway_monitor", False):
                room_trip_statuses = self.room_trip_monitor.update(
                    group_snapshots,
                    interaction_now_ts,
                    boxes_by_person_id=boxes_by_person_id,
                    frame_shape=(display_frame.shape[0], display_frame.shape[1]),
                )

            if self.cfg.show_body or self.cfg.show_head or demographics_labels:
                draw_detection_boxes(
                    display_frame,
                    boxes,
                    confidence_scores,
                    class_ids,
                    show_body_confidence=self.cfg.show_body_confidence,
                    show_head_confidence=self.cfg.show_head_confidence,
                    show_body=self.cfg.show_body,
                    show_head=self.cfg.show_head,
                    person_ids=person_ids,
                    demographics_labels=demographics_labels,
                    group_progress=group_progress,
                    label_default_color=getattr(self.cfg, "label_default_color", "#FFFF00"),
                    label_segment_colors=getattr(self.cfg, "label_segment_colors", ""),
                )

            now = time.perf_counter()
            fps_raw = 1.0 / max(1e-6, now - processing_started_at)
            ema_fps = fps_raw if ema_fps == 0 else ema_fps * 0.9 + fps_raw * 0.1
            proc_fps = ema_fps
            display_frame = draw_runtime_overlays(
                display_frame,
                cfg=self.cfg,
                editor_state=editor_state,
                current_roi=current_roi,
                current_height_roi=current_height_roi,
                current_doorway_roi=current_doorway_roi,
                group_detector=group_detector,
                room_trip_statuses=room_trip_statuses,
                carry_alert_statuses=carry_alert_statuses,
                current_view=current_view,
                detection_count=len(boxes),
                group_count=group_count,
                proc_fps=proc_fps,
                source_fps=source_fps_display,
                is_last_file_frame=is_last_file_frame,
                norm_to_px=norm_to_px,
            )

            end_all = False
            if not self.cfg.no_show:
                cv2.imshow(window_name, display_frame)
                # mimic_live: sleep to hold preview at source FPS pace.
                if mimic_live_interval > 0:
                    elapsed_since_last = time.perf_counter() - mimic_live_last
                    wait_ms = max(1, int((mimic_live_interval - elapsed_since_last) * 1000))
                else:
                    wait_ms = 1
                raw_key = cv2.waitKeyEx(wait_ms)
                mimic_live_last = time.perf_counter()
                end_all, should_break = handle_preview_key(raw_key, editor_state=editor_state, app=self)
                if should_break:
                    break

            if editor_state["mode"] != "none":
                continue  # Do not write paused frames to video

            if writer is None and getattr(self.cfg, "save_output_video", True):
                writer = self._build_video_writer(cap, display_frame)
            archive_required = bool(getattr(self.cfg, "nvr_record_enable", False))
            if group_event_logger is None and getattr(self.cfg, "group_log_enable", False):
                root_dir = resolve_output_path(getattr(self.cfg, "nvr_output_root", "output/archive"))
                root_dir.mkdir(parents=True, exist_ok=True)
                group_event_logger = GroupEventLogger(
                    db_path=DB_PATH,
                    root_dir=root_dir,
                    source_name=self._nvr_source_name(),
                )
            if nvr_recorder is None and archive_required:
                try:
                    nvr_recorder = self._build_rolling_recorder(cap, display_frame, group_event_logger)
                    print(f"[NVR] Rolling recording enabled for {self._nvr_source_name()}")
                except Exception as e:
                    print(f"[NVR] Failed to initialize rolling recorder: {e}")
                    nvr_recorder = None
            if writer is not None:
                if is_finite_file and grabber is None:
                    # Gap-fill is only reliable when reading directly from cap
                    # (no background grabber thread racing CAP_PROP_POS_FRAMES).
                    current_source_index = max(1, source_frame_index)
                    if writer_last_frame is None:
                        for _ in range(max(1, current_source_index - writer_last_source_index)):
                            writer.write(display_frame)
                        writer_last_source_index = current_source_index
                    else:
                        gap = max(1, current_source_index - writer_last_source_index)
                        for _ in range(max(0, gap - 1)):
                            writer.write(writer_last_frame)
                        writer.write(display_frame)
                        writer_last_source_index = current_source_index
                    writer_last_frame = display_frame.copy()
                else:
                    # Live stream, camera, or mimic_live with grabber:
                    # write each delivered frame once — the grabber already
                    # throttles delivery to target_fps.
                    writer.write(display_frame)
                if nvr_recorder is not None:
                    nvr_recorder.write(display_frame)
            elif nvr_recorder is not None:
                nvr_recorder.write(display_frame)

            current_segment_path = nvr_recorder.current_segment_path if nvr_recorder is not None else None
            if group_event_logger is not None:
                group_event_logger.sync(group_snapshots, time.time(), segment_path=current_segment_path)

            if is_last_file_frame:
                print("Reached last frame; stopping.")
                if not self.cfg.no_show:
                    cv2.waitKey(400)
                break

        total_people = person_tracker.unique_confirmed_count if person_tracker else 0
        total_groups = group_detector.next_group_id - 1 if group_detector else 0
        print(f"\n[METRICS] {{\"people\": {total_people}, \"groups\": {total_groups}}}")

    finally:
        if grabber:
            grabber.stop()
        cap.release()
        if writer is not None:
            if is_finite_file and grabber is None and writer_last_frame is not None and total_input_frames > writer_last_source_index:
                for _ in range(total_input_frames - writer_last_source_index):
                    writer.write(writer_last_frame)
            writer.release()
            print(f"Saved: {self.output_path}")
        if group_event_logger is not None:
            group_event_logger.close(
                time.time(),
                nvr_recorder.current_segment_path if nvr_recorder is not None else None,
            )
        if nvr_recorder is not None:
            nvr_recorder.close()
        if not is_batch:
            cv2.destroyAllWindows()

    return {"people": total_people, "groups": total_groups}, end_all
