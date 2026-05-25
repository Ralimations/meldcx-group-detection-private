#!/usr/bin/env python3
"""Face-based age classifier using OpenVINO models.

Uses a lightweight face detector + age-gender estimation model to
classify detected people as "Adult" (19-59) or "Senior" (60+).

This complements the height-based classifier which handles "Minor" (1-18).
"""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

from config import OpenVinoDefaults
from scripts.person_detector import non_max_suppression


class FaceAgeClassifier:
    """OpenVINO face detection + age estimation for Adult/Senior classification."""

    def __init__(self, cfg: OpenVinoDefaults) -> None:
        self.cfg = cfg
        self.senior_threshold = cfg.face_age_senior_threshold
        self.history_len = cfg.age_history_len

        face_det_path = Path(cfg.face_detection_model)
        age_model_path = Path(cfg.face_age_gender_model)

        core = ov.Core()

        face_model = core.read_model(str(face_det_path))
        self._face_net = core.compile_model(face_model, cfg.device)
        self._face_input = self._face_net.input(0)
        self._face_output = self._face_net.output(0)
        face_shape = self._face_input.shape
        self._face_h = face_shape[2]
        self._face_w = face_shape[3]

        age_model = core.read_model(str(age_model_path))
        self._age_net = core.compile_model(age_model, cfg.device)
        self._age_input = self._age_net.input(0)
        self._age_out_age = self._age_net.output("age_conv3")
        self._age_out_gender = self._age_net.output("prob")
        age_shape = self._age_input.shape
        self._age_h = age_shape[2]
        self._age_w = age_shape[3]

        self._vote_history: dict[int, deque[str]] = defaultdict(
            lambda: deque(maxlen=self.history_len)
        )
        self._locked: dict[int, str] = {}
        self._gender_sum: dict[int, float] = defaultdict(float)
        self._gender_count: dict[int, int] = defaultdict(int)
        self._gender_locked: dict[int, tuple[str, float]] = {}
        self._frame_count = 0
        self.last_detected_faces: list[tuple[int, int, int, int, float]] = []
        self.last_raw_results: dict[int, dict[str, float | str]] = {}

        print(
            f"[FaceAge] Loaded face detector: {face_det_path.name} "
            f"({self._face_w}x{self._face_h})"
        )
        print(
            f"[FaceAge] Loaded shared age/gender model: {age_model_path.name} "
            f"({self._age_w}x{self._age_h})"
        )

    def _detect_faces(self, frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        """Detect faces in the frame. Returns list of (x1, y1, x2, y2, conf)."""
        conf_threshold = getattr(self.cfg, "face_detection_conf", 0.3)
        overlap_threshold = getattr(self.cfg, "face_overlap_threshold", 0.4)
        h, w = frame.shape[:2]
        blob = cv2.resize(frame, (self._face_w, self._face_h))
        blob = blob.transpose(2, 0, 1).reshape(1, 3, self._face_h, self._face_w).astype(np.float32)

        result = self._face_net({self._face_input: blob})[self._face_output]
        face_boxes: list[tuple[int, int, int, int]] = []
        face_scores: list[float] = []
        for det in result[0, 0]:
            conf = det[2]
            if conf < conf_threshold:
                continue
            x1 = max(0, int(det[3] * w))
            y1 = max(0, int(det[4] * h))
            x2 = min(w, int(det[5] * w))
            y2 = min(h, int(det[6] * h))
            if x2 > x1 and y2 > y1:
                face_boxes.append((x1, y1, x2, y2))
                face_scores.append(float(conf))
        if not face_boxes:
            return []
        boxes_np = np.asarray(face_boxes, dtype=np.float32)
        scores_np = np.asarray(face_scores, dtype=np.float32)
        keep_idx = non_max_suppression(boxes_np, scores_np, float(overlap_threshold))
        return [
            (
                int(boxes_np[i][0]),
                int(boxes_np[i][1]),
                int(boxes_np[i][2]),
                int(boxes_np[i][3]),
                float(scores_np[i]),
            )
            for i in keep_idx
        ]

    def _estimate_age_gender(
        self,
        frame: np.ndarray,
        face_box: tuple[int, int, int, int, float],
    ) -> tuple[float, str, float, float] | None:
        """Estimate age and gender from a face crop."""
        x1, y1, x2, y2, _conf = face_box
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None

        blob = cv2.resize(face_crop, (self._age_w, self._age_h))
        blob = blob.transpose(2, 0, 1).reshape(1, 3, self._age_h, self._age_w).astype(np.float32)

        results = self._age_net({self._age_input: blob})
        age = float(results[self._age_out_age][0, 0, 0, 0]) * 100.0
        gender_prob = results[self._age_out_gender][0]

        max_gender_conf = float(np.max(gender_prob))
        gender = "Male" if gender_prob[1] > gender_prob[0] else "Female"
        age_conf = max_gender_conf

        return age, gender, float(gender_prob[1]), age_conf

    @staticmethod
    def _face_in_person(
        face: tuple[int, int, int, int, float],
        person: tuple[float, ...],
    ) -> bool:
        """Check if a face box is inside a person box."""
        fx1, fy1, fx2, fy2, _fconf = face
        px1, py1, px2, py2 = float(person[0]), float(person[1]), float(person[2]), float(person[3])
        face_cx = (fx1 + fx2) / 2
        face_cy = (fy1 + fy2) / 2
        return px1 <= face_cx <= px2 and py1 <= face_cy <= py2

    def update(
        self,
        frame: np.ndarray,
        boxes: np.ndarray,
        person_ids: np.ndarray,
    ) -> tuple[dict[int, tuple[str, float]], dict[int, tuple[str, float]], dict[int, tuple[float, float]]]:
        """Run inference on the frame. Returns (age_labels, gender_labels, adult_samples)."""
        results: dict[int, tuple[str, float]] = {}
        gender_results: dict[int, tuple[str, float]] = {}
        adult_samples: dict[int, tuple[float, float]] = {}
        active_ids: set[int] = set()
        raw_results: dict[int, dict[str, float | str]] = {}
        self._frame_count += 1
        log_this_frame = (self._frame_count % 30 == 0)
        enable_face_age = bool(
            getattr(self.cfg, "enable_face_analysis", False)
            and getattr(self.cfg, "face_age_override", False)
        )

        self.last_detected_faces = []

        for box, pid in zip(boxes, person_ids):
            tid = int(pid)
            active_ids.add(tid)

            if tid in self._gender_locked and (not enable_face_age or tid in self._locked):
                if enable_face_age and tid in self._locked:
                    results[tid] = (self._locked[tid], 1.0)
                gender_results[tid] = self._gender_locked[tid]
                continue

            x1, y1, x2, y2 = [int(v) for v in box[:4]]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue

            person_crop = frame[y1:y2, x1:x2]
            faces = self._detect_faces(person_crop)
            if not faces:
                continue
            matched_face = max(faces, key=lambda face: float(face[4]))
            face_abs = (
                matched_face[0] + x1,
                matched_face[1] + y1,
                matched_face[2] + x1,
                matched_face[3] + y1,
                matched_face[4],
            )
            self.last_detected_faces.append(face_abs)

            result = self._estimate_age_gender(frame, face_abs)
            if result is None:
                continue

            age, gender, male_prob, age_conf = result
            raw_results[tid] = {
                "age": float(age),
                "gender": str(gender),
                "gender_conf": float(age_conf),
            }
            gender_results[tid] = (gender, age_conf)

            self._gender_sum[tid] += male_prob
            self._gender_count[tid] += 1
            avg_male_prob = self._gender_sum[tid] / self._gender_count[tid]

            if self._gender_count[tid] >= 3:
                if avg_male_prob >= 0.7:
                    self._gender_locked[tid] = ("Male", avg_male_prob)
                elif avg_male_prob <= 0.3:
                    self._gender_locked[tid] = ("Female", 1.0 - avg_male_prob)

            if enable_face_age:
                vote = "Senior" if age >= self.senior_threshold else "Adult"
                print(f"[FaceAge] tid={tid} age={age:.1f} gender={gender} -> {vote}")
                self._vote_history[tid].append(vote)

                hist = self._vote_history[tid]
                if len(hist) >= 3:
                    senior_count = sum(1 for v in hist if v == "Senior")
                    if senior_count >= len(hist) * 0.7:
                        self._locked[tid] = "Senior"
                        results[tid] = ("Senior", 1.0)
                    elif (len(hist) - senior_count) >= len(hist) * 0.7:
                        if tid not in self._locked:
                            idx = np.where(person_ids == tid)[0][0]
                            bh = boxes[idx][3] - boxes[idx][1]
                            adult_samples[tid] = (float(bh), 0.0)

                        self._locked[tid] = "Adult"
                        results[tid] = ("Adult", 1.0)

        for tid in active_ids:
            if tid in self._gender_locked and tid not in gender_results:
                gender_results[tid] = self._gender_locked[tid]
            if enable_face_age and tid in self._locked and tid not in results:
                results[tid] = (self._locked[tid], 1.0)

        stale = set(self._vote_history.keys()) - active_ids
        for sid in stale:
            del self._vote_history[sid]
            self._locked.pop(sid, None)
            self._gender_sum.pop(sid, None)
            self._gender_count.pop(sid, None)
            self._gender_locked.pop(sid, None)

        if log_this_frame:
            if self.last_detected_faces:
                print(f"[FaceAge] Frame {self._frame_count}: {len(self.last_detected_faces)} person-crop faces detected")
            else:
                print(f"[FaceAge] Frame {self._frame_count}: NO person-crop faces detected")

        self.last_raw_results = raw_results
        return results, gender_results, adult_samples
