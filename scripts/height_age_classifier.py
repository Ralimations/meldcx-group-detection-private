#!/usr/bin/env python3
"""Height-based age classification using live corrected size measurements."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from config import OpenVinoDefaults

if TYPE_CHECKING:
    from scripts.pose_estimator import PoseObservation


class HeightAgeClassifier:
    """Track normalized body size per person and lock Minor/Adult once it stabilizes."""

    _HEAD_IDX = (0, 1, 2, 3, 4)   # nose, eyes, ears
    _ANKLE_IDX = (15, 16)
    _MIN_HEAD_POINTS = 2

    def __init__(self, cfg: OpenVinoDefaults) -> None:
        self.cfg = cfg
        self._default_threshold = getattr(cfg, "age_adult_threshold_px", 350.0)
        self._stale_ttl = max(1, int(getattr(cfg, "max_misses", 30)))
        self._frame_h = 0
        self._frame_w = 0

        self._locked_label: dict[int, str] = {}
        self._locked_text: dict[int, str] = {}
        self._locked_conf: dict[int, float] = {}
        self._last_live_height: dict[int, float] = {}
        self._stable_frames: dict[int, int] = {}
        self._track_samples: dict[int, int] = {}
        self._frozen_display_height: dict[int, float] = {}
        self._last_valid_debug: dict[int, tuple[float, float]] = {}
        self._measurement_segments: dict[int, tuple[tuple[int, int], tuple[int, int], bool]] = {}

    @property
    def pose_skip_track_ids(self) -> set[int]:
        if getattr(self.cfg, "show_age_height_unlocked", False):
            return set()
        return set(self._locked_label.keys()) | set(self._frozen_display_height.keys())

    def _measurement_from_box(self, box: np.ndarray) -> float:
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)

        if getattr(self.cfg, "perspective", "LEVELED").upper() == "TOP-DOWN":
            metric = str(getattr(self.cfg, "topdown_height_metric", "max_side")).strip().lower()
            if metric == "sqrt_area":
                return float(np.sqrt(box_w * box_h))
            if metric == "height":
                return box_h
            return max(box_w, box_h)

        return box_h

    def _measurement_from_pose(
        self,
        pose_obs: PoseObservation | None,
    ) -> tuple[float, tuple[float, float], tuple[float, float]] | None:
        if pose_obs is None or pose_obs.keypoints is None:
            return None

        keypoints = np.asarray(pose_obs.keypoints, dtype=np.float32)
        if keypoints.ndim != 2 or keypoints.shape[0] < 2:
            return None

        if keypoints.shape[0] >= 17:
            head_idx = (0, 1, 2, 3, 4)
            ankle_idx = (15, 16)
            min_head_points = self._MIN_HEAD_POINTS
        elif keypoints.shape[0] >= 14:
            head_idx = (0,)
            ankle_idx = (12, 13)
            min_head_points = 1
        else:
            return None

        conf_thresh = float(getattr(self.cfg, "pose_keypoint_conf", 0.35))
        head_points = keypoints[list(head_idx)]
        ankle_points = keypoints[list(ankle_idx)]

        head_mask = head_points[:, 2] >= conf_thresh
        ankle_mask = ankle_points[:, 2] >= conf_thresh
        if int(np.count_nonzero(head_mask)) < min_head_points:
            return None
        if int(np.count_nonzero(ankle_mask)) < len(ankle_idx):
            return None

        head_x = float(np.median(head_points[head_mask, 0]))
        head_y = float(np.median(head_points[head_mask, 1]))
        foot_xy = np.mean(ankle_points[:, :2], axis=0)
        foot_x = float(foot_xy[0])
        foot_y = float(foot_xy[1])
        measurement = foot_y - head_y
        if not np.isfinite(measurement) or measurement <= 1.0:
            return None
        return float(measurement), (head_x, head_y), (foot_x, foot_y)

    def draw_measurements(self, frame: np.ndarray) -> None:
        if not bool(getattr(self.cfg, "show_height_measurement_line", False)):
            return
        if frame is None or frame.size == 0:
            return

        for tid, (head_pt, foot_pt, is_stale) in self._measurement_segments.items():
            color = (0, 165, 255) if is_stale else (0, 255, 255)
            head = (int(head_pt[0]), int(head_pt[1]))
            foot = (int(foot_pt[0]), int(foot_pt[1]))

            cv2.line(frame, head, foot, color, 2, cv2.LINE_AA)
            cv2.circle(frame, head, 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, foot, 5, color, -1, cv2.LINE_AA)
            cv2.putText(frame, f"H{tid}", (head[0] + 6, head[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    def _height_stability_tolerance(self) -> float:
        """Return the allowed 640px-reference corrected-height drift in frame pixels."""
        tolerance_px = float(
            getattr(
                self.cfg,
                "height_average_stability_px",
                getattr(self.cfg, "height_median_stability_px", 3.0),
            )
        )
        tolerance_px = max(0.0, tolerance_px)
        return tolerance_px * (self._frame_h / 640.0)

    @staticmethod
    def _format_height_debug(
        raw_display_px: float,
        corrected_display_px: float,
        depth_val: float,
        multiplier: float,
    ) -> str:
        """Format the live height diagnostics in the requested order."""
        return (
            f"PX:{raw_display_px:.0f}px"
            f" | C:{corrected_display_px:.0f}px"
            f" | D:{depth_val:.2f}"
            f" | X:{multiplier:.3f}"
        )

    def _lock_track(
        self,
        tid: int,
        normalized_height: float,
        threshold: float,
        confidence: float,
    ) -> tuple[str, float]:
        ratio = threshold / 640.0
        scaled_threshold = ratio * self._frame_h
        display_bh = normalized_height * (640.0 / self._frame_h)

        label = "Minor" if normalized_height < scaled_threshold else "Adult"
        self._locked_label[tid] = label
        self._locked_conf[tid] = float(max(0.0, min(1.0, confidence)))

        if getattr(self.cfg, "show_age_height_stats", False):
            multiplier = 1.0
            if depth_engine and depth_val is not None and getattr(self.cfg, "use_depth", False):
                multiplier = self._depth_multiplier_from_calibration(depth_val)
            raw_height = normalized_height / max(multiplier, 1e-6)
            raw_display_bh = raw_height * (640.0 / self._frame_h)
            depth_display = float(depth_val) if depth_val is not None else 1.0
            debug_text = self._format_height_debug(
                raw_display_px=raw_display_bh,
                corrected_display_px=display_bh,
                depth_val=depth_display,
                multiplier=multiplier,
            )
            self._locked_text[tid] = f"{label} [{debug_text}]"
        else:
            self._locked_text[tid] = label

        return self._locked_text[tid], 1.0

    def _trusted_normalized_height(self, tid: int, fallback_value: float) -> float:
        return self._frozen_display_height.get(tid, fallback_value)

    def _trusted_depth_value(self, tid: int, fallback_value: float) -> float:
        """Return the best trusted depth value available for a track."""
        return self._frozen_display_depth.get(tid, fallback_value)

    def update(
        self,
        boxes: np.ndarray,
        person_ids: np.ndarray,
        frame_height: int,
        frame_width: int,
        pose_observations: dict[int, PoseObservation] | None = None,
    ) -> dict[int, tuple[str, float]]:
        if len(boxes) == 0:
            return {}

        self._frame_h = frame_height
        self._frame_w = frame_width
        active_roi = height_roi_polygon if height_roi_polygon is not None else roi_polygon
        results: dict[int, tuple[str, float]] = {}
        active_ids: set[int] = set()
        live_threshold = self._default_threshold
        tolerance = self._height_stability_tolerance()
        show_unlocked = bool(getattr(self.cfg, "show_age_height_unlocked", False))
        show_height_text = bool(getattr(self.cfg, "show_age_height_stats", False) or show_unlocked)
        show_pose_debug = bool(getattr(self.cfg, "show_age_height_stats", False) or show_unlocked)

        for box, pid in zip(boxes, person_ids):
            tid = int(pid)
            active_ids.add(tid)
            self._stale_counts[tid] = 0

            if tid in self._locked_label and not show_unlocked:
                results[tid] = (
                    self._locked_text.get(tid, self._locked_label[tid]),
                    self._locked_conf.get(tid, 1.0),
                )
                continue

            pose_obs = pose_observations.get(tid) if pose_observations else None
            pose_measurement = self._measurement_from_pose(pose_obs)

            depth_val_live = 1.0
            multiplier_live = 1.0
            depth_sample_valid = True
            current_segment: tuple[tuple[int, int], tuple[int, int], bool] | None = None
            pose_blocks_baseline = (
                pose_measurement is None
                or (
                    bool(getattr(self.cfg, "use_pose_visibility_gate", False))
                    and pose_obs is not None
                    and not pose_obs.full_body_visible
                )
            )
            valid_height_sample = not pose_blocks_baseline

            if valid_height_sample:
                measurement = self._measurement_from_box(box)
                current_segment = (
                    (int(round(head_x)), int(round(head_y))),
                    (int(round(foot_x)), int(round(foot_y))),
                    False,
                )
                if depth_engine and getattr(self.cfg, "use_depth", False):
                    search_radius = max(0, int(getattr(self.cfg, "depth_mask_search_radius", 0)))
                    depth_value, _ = depth_engine.sample_depth_at_point(
                        int(round(foot_x)),
                        int(round(foot_y)),
                        search_radius=search_radius,
                    )
                    if depth_value is None:
                        depth_sample_valid = False
                    else:
                        depth_val_live = depth_value
                        multiplier_live = self._depth_multiplier_from_calibration(depth_val_live)
                if not depth_sample_valid:
                    valid_height_sample = False

            if valid_height_sample:
                normalized_bh_live = measurement * multiplier_live
                prev_live = self._last_live_height.get(tid)
                if prev_live is None:
                    stable_frames = 0
                elif abs(normalized_bh_live - prev_live) <= tolerance:
                    stable_frames = self._stable_frames.get(tid, 0) + 1
                else:
                    stable_frames = 0
                self._stable_frames[tid] = stable_frames
                self._last_live_height[tid] = normalized_bh_live
                self._track_samples[tid] = self._track_samples.get(tid, 0) + 1
                self._last_valid_debug[tid] = (
                    measurement,
                    normalized_bh_live,
                    depth_val_live,
                    multiplier_live,
                )
            else:
                stable_frames = 0
                self._stable_frames[tid] = 0
                self._last_live_height.pop(tid, None)

            if current_segment is not None:
                self._measurement_segments[tid] = current_segment
            else:
                self._measurement_segments.pop(tid, None)

            freeze_samples = max(0, int(getattr(self.cfg, "height_display_freeze_samples", 18)))
            stability_frames = max(0, int(getattr(self.cfg, "height_peak_stability_frames", 10)))
            lock_samples = max(0, int(getattr(self.cfg, "height_lock_min_samples", 12)))
            lock_stability_frames = max(0, int(getattr(self.cfg, "height_lock_stability_frames", 6)))
            lock_requires_trusted_baseline = bool(
                getattr(self.cfg, "height_lock_requires_trusted_baseline", True)
            )

            display_baseline_ready = (
                not show_unlocked
                and (
                    tid in self._frozen_display_height
                    or (
                        freeze_samples > 0
                        and self._track_samples.get(tid, 0) >= freeze_samples
                        and stable_frames >= stability_frames
                    )
                )
            )
            display_baseline_ready = display_baseline_ready and not pose_blocks_baseline and not inside_roi

            if display_baseline_ready and tid not in self._frozen_display_height:
                self._frozen_display_height[tid] = normalized_bh_live

            lock_candidate_ready = (
                not show_unlocked
                and lock_samples > 0
                and self._track_samples.get(tid, 0) >= lock_samples
                and stable_frames >= lock_stability_frames
                and not pose_blocks_baseline
            )
            lock_baseline_ready = tid in self._frozen_display_height if lock_requires_trusted_baseline else lock_candidate_ready

            if lock_candidate_ready and lock_baseline_ready:
                trusted_height = self._trusted_normalized_height(tid, normalized_bh_live)
                trusted_depth = self._trusted_depth_value(tid, depth_val_live)
                results[tid] = self._lock_track(
                    tid,
                    normalized_height=trusted_height,
                    threshold=live_threshold,
                    depth_val=trusted_depth,
                    depth_engine=depth_engine,
                )
                continue

            if show_height_text:
                if not valid_height_sample:
                    last_valid = self._last_valid_debug.get(tid)
                    if last_valid is None:
                        if show_unlocked:
                            live_text = "Height [invalid sample] U"
                        else:
                            live_text = "Height [pending pose]"
                    else:
                        last_measurement, last_corrected, last_depth, last_multiplier = last_valid
                        raw_display_live = last_measurement * (640.0 / self._frame_h)
                        corrected_display_live = last_corrected * (640.0 / self._frame_h)
                        debug_text = self._format_height_debug(
                            raw_display_px=raw_display_live,
                            corrected_display_px=corrected_display_live,
                            depth_val=last_depth,
                            multiplier=last_multiplier,
                        )
                        live_text = f"Height [STALE | {debug_text}]"
                        if show_unlocked:
                            live_text = f"{live_text} U"
                else:
                    raw_display_live = measurement * (640.0 / self._frame_h)
                    corrected_display_live = normalized_bh_live * (640.0 / self._frame_h)
                    debug_text = self._format_height_debug(
                        raw_display_px=raw_display_live,
                        corrected_display_px=corrected_display_live,
                        depth_val=depth_val_live,
                        multiplier=multiplier_live,
                    )
                    live_text = f"Height [{debug_text}]"
                    if show_unlocked:
                        live_text = f"{live_text} U"
                if pose_obs is not None and bool(getattr(self.cfg, "use_pose_visibility_gate", False)):
                    pose_tag = "P:OK" if pose_obs.full_body_visible else "P:HOLD"
                    live_text = f"{live_text} {pose_tag}"
                results[tid] = (live_text, 1.0)

        stale = (
            set(self._track_samples.keys())
            | set(self._locked_label.keys())
            | set(self._locked_conf.keys())
            | set(self._last_live_height.keys())
            | set(self._stable_frames.keys())
            | set(self._frozen_display_height.keys())
            | set(self._last_valid_debug.keys())
        ) - active_ids
        for sid in stale:
            self._stale_counts[sid] = self._stale_counts.get(sid, 0) + 1
            if self._stale_counts[sid] <= self._stale_ttl:
                continue
            self._locked_label.pop(sid, None)
            self._locked_text.pop(sid, None)
            self._locked_conf.pop(sid, None)
            self._last_live_height.pop(sid, None)
            self._stable_frames.pop(sid, None)
            self._track_samples.pop(sid, None)
            self._frozen_display_height.pop(sid, None)
            self._last_valid_debug.pop(sid, None)
            self._measurement_segments.pop(sid, None)

        return results
