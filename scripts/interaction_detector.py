#!/usr/bin/env python3
"""Interaction heuristics layered on top of grouped tracked people."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class _CarryState:
    locked: bool = False
    high_hits: int = 0
    low_hits: int = 0
    stale_frames: int = 0
    last_score: int = 0
    locked_started_at: float | None = None


class InteractionDetector:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._carry_states: dict[tuple[int, ...], _CarryState] = {}
        self._seen_keys: set[tuple[int, ...]] = set()

    @staticmethod
    def _intersection_area(box_a, box_b) -> float:
        ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
        bx1, by1, bx2, by2 = [float(v) for v in box_b]
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        return float((ix2 - ix1) * (iy2 - iy1))

    @staticmethod
    def _recent_motion(history_map, tid: int):
        hist = history_map.get(int(tid))
        if hist is None or len(hist) < 3:
            return None
        x0, y0 = hist[-3]
        x1, y1 = hist[-1]
        return (float(x1 - x0), float(y1 - y0))

    def get_carry_overlap_hint(
        self,
        *,
        member_ids,
        group_type: str,
        id_to_box,
        track_demographics,
        pose_observations,
        current_centers,
        history_map,
        now_ts: float,
    ) -> tuple[str, int, bool, float]:
        if not getattr(self.cfg, "show_carry_overlap_debug", False):
            return "", 0, False, 0.0
        if group_type != "Minor and Adult" or len(member_ids) != 2:
            return "", 0, False, 0.0
        pair_key = tuple(sorted(int(tid) for tid in member_ids))
        self._seen_keys.add(pair_key)

        adult_tid = None
        minor_tid = None
        for tid in member_ids:
            data = track_demographics.get(int(tid)) if track_demographics else None
            if not data:
                continue
            age_data = data.get("age")
            if not age_data:
                continue
            age_label = str(age_data[1])
            if age_label.startswith("Minor"):
                minor_tid = int(tid)
            elif age_label.startswith("Adult") or age_label.startswith("Senior"):
                adult_tid = int(tid)

        if adult_tid is None or minor_tid is None:
            return "", 0, False, 0.0
        if adult_tid not in id_to_box or minor_tid not in id_to_box:
            return "", 0, False, 0.0

        adult_box = np.asarray(id_to_box[adult_tid], dtype=np.float32)
        minor_box = np.asarray(id_to_box[minor_tid], dtype=np.float32)
        adult_h = max(1.0, float(adult_box[3] - adult_box[1]))
        minor_area = max(1.0, float((minor_box[2] - minor_box[0]) * (minor_box[3] - minor_box[1])))
        overlap_ratio_minor = self._intersection_area(adult_box, minor_box) / minor_area

        score = 0
        total_checks = 5

        minor_cx = float((minor_box[0] + minor_box[2]) * 0.5)
        minor_cy = float((minor_box[1] + minor_box[3]) * 0.5)
        if overlap_ratio_minor >= 0.35:
            score += 1
        if adult_box[0] <= minor_cx <= adult_box[2] and adult_box[1] <= minor_cy <= adult_box[3]:
            score += 1

        adult_pose = pose_observations.get(adult_tid) if pose_observations else None
        if adult_pose is not None and adult_pose.torso_visible_count >= 2 and adult_pose.lower_visible_count <= 1:
            score += 1

        minor_pose = pose_observations.get(minor_tid) if pose_observations else None
        if minor_pose is not None and minor_pose.visible_count >= 3 and minor_pose.lower_visible_count <= 1:
            score += 1

        adult_motion = self._recent_motion(history_map, adult_tid)
        minor_motion = self._recent_motion(history_map, minor_tid)
        if adult_motion is not None and minor_motion is not None:
            adult_speed = math.hypot(adult_motion[0], adult_motion[1])
            minor_speed = math.hypot(minor_motion[0], minor_motion[1])
            if adult_speed > 1.0 and minor_speed > 1.0:
                motion_dot = adult_motion[0] * minor_motion[0] + adult_motion[1] * minor_motion[1]
                motion_cos = motion_dot / max(adult_speed * minor_speed, 1e-6)
                adult_center = current_centers.get(adult_tid)
                minor_center = current_centers.get(minor_tid)
                centers_close = False
                if adult_center is not None and minor_center is not None:
                    centers_close = math.hypot(
                        adult_center[0] - minor_center[0],
                        adult_center[1] - minor_center[1],
                    ) <= adult_h * 0.75
                if motion_cos >= 0.75 and centers_close:
                    score += 1

        score_threshold = max(1, int(getattr(self.cfg, "carry_overlap_score_threshold", 3)))
        lock_frames = max(1, int(getattr(self.cfg, "carry_overlap_lock_frames", 3)))
        release_frames = max(1, int(getattr(self.cfg, "carry_overlap_release_frames", 3)))
        release_threshold = max(0, score_threshold - 1)

        state = self._carry_states.setdefault(pair_key, _CarryState())
        state.stale_frames = 0
        state.last_score = score

        if score >= score_threshold:
            state.high_hits += 1
            state.low_hits = 0
            if not state.locked and state.high_hits >= lock_frames:
                state.locked = True
                state.locked_started_at = now_ts
        elif score <= release_threshold:
            state.low_hits += 1
            if not state.locked:
                state.high_hits = 0
            elif state.low_hits >= release_frames:
                state.locked = False
                state.high_hits = 0
                state.locked_started_at = None
        else:
            state.low_hits = 0

        if state.locked and state.locked_started_at is None:
            state.locked_started_at = now_ts
        elapsed_seconds = 0.0
        if state.locked and state.locked_started_at is not None:
            elapsed_seconds = max(0.0, float(now_ts - state.locked_started_at))

        if state.locked:
            return f"Carry LOCKED {score}/{total_checks}", score, True, elapsed_seconds
        if score >= score_threshold:
            return f"Carry? {score}/{total_checks}", score, False, 0.0
        return f"Carry {score}/{total_checks}", score, False, 0.0

    def begin_frame(self) -> None:
        self._seen_keys.clear()

    def end_frame(self) -> None:
        stale_ttl = max(1, int(getattr(self.cfg, "max_misses", 30)))
        remove_keys: list[tuple[int, ...]] = []
        for pair_key, state in self._carry_states.items():
            if pair_key in self._seen_keys:
                continue
            state.stale_frames += 1
            if state.stale_frames > stale_ttl:
                remove_keys.append(pair_key)
        for pair_key in remove_keys:
            del self._carry_states[pair_key]
