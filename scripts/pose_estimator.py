#!/usr/bin/env python3
"""Selective pose estimation used to gate trusted height locking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from config import OpenVinoDefaults
from scripts.person_detector import letterbox, non_max_suppression, xywh_to_xyxy

try:
    from openvino import Core
except ImportError:
    from openvino.runtime import Core


@dataclass
class PoseObservation:
    """Latest pose quality signal for one tracked person."""

    full_body_visible: bool
    upright_body: bool
    hands_below_head: bool
    pose_confidence: float
    visible_count: int
    torso_visible_count: int
    lower_visible_count: int
    lowest_visible_ratio: float
    keypoints: np.ndarray | None = None  # Full-frame xyc, shape (17, 3)
    box_xyxy: np.ndarray | None = None   # Full-frame xyxy box used for this pose.


class PoseEstimator:
    """Runs a lightweight YOLO pose model on selected tracked crops."""

    _SKELETON_17 = (
        (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
        (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
        (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
        (1, 3), (2, 4), (3, 5), (4, 6),
    )
    _SKELETON_14 = (
        (0, 1),
        (1, 2), (1, 3),
        (2, 4), (4, 6),
        (3, 5), (5, 7),
        (1, 8), (1, 9),
        (8, 9),
        (8, 10), (10, 12),
        (9, 11), (11, 13),
    )
    _CACHE_MIN_IOU = 0.35

    def __init__(self, cfg: OpenVinoDefaults) -> None:
        self.cfg = cfg
        self.model_path = Path(cfg.pose_model)
        if not self.model_path.exists():
            self.model_path = Path(__file__).parent.parent / str(cfg.pose_model)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Pose model not found: {self.model_path}")

        self.core = Core()
        ov_model = self.core.read_model(model=str(self.model_path))
        self.compiled_model = self.core.compile_model(ov_model, str(cfg.device))
        self.input_port = self.compiled_model.input(0)
        self.output_port = self.compiled_model.output(0)
        self.input_h = int(self.input_port.shape[2])
        self.input_w = int(self.input_port.shape[3])
        output_shape = tuple(int(v) for v in self.output_port.shape)

        self.interval = max(1, int(getattr(cfg, "pose_interval", 3)))
        self.max_tracks_per_frame = max(0, int(getattr(cfg, "pose_max_tracks_per_frame", 0)))
        self.conf_thresh = float(getattr(cfg, "pose_conf", 0.25))
        self.iou_thresh = float(getattr(cfg, "pose_overlap_threshold", 0.45))
        self.crop_padding = max(0.0, float(getattr(cfg, "pose_crop_padding", 0.15)))
        self.keypoint_conf = float(getattr(cfg, "pose_keypoint_conf", 0.35))
        self.min_visible_keypoints = max(1, int(getattr(cfg, "pose_min_visible_keypoints", 8)))
        self.min_torso_keypoints = max(1, int(getattr(cfg, "pose_min_torso_keypoints", 3)))
        self.min_lower_keypoints = max(1, int(getattr(cfg, "pose_min_lower_body_keypoints", 2)))
        self.min_lowest_ratio = float(getattr(cfg, "pose_min_lowest_keypoint_ratio", 0.82))

        self._latest: dict[int, PoseObservation] = {}
        self._frame_counts: dict[int, int] = {}
        self._kpt_count = 17
        self._keypoint_offset = 5
        if len(output_shape) >= 3:
            dim_a = int(output_shape[-2])
            dim_b = int(output_shape[-1])
            cols = dim_a if dim_a < dim_b else dim_b
            if cols >= 6 and (cols - 6) % 3 == 0:
                self._keypoint_offset = 6
                self._kpt_count = (cols - 6) // 3
            elif cols >= 5 and (cols - 5) % 3 == 0:
                self._keypoint_offset = 5
                self._kpt_count = (cols - 5) // 3
        self._configure_pose_schema(self._kpt_count)

        try:
            exec_devices = self.compiled_model.get_property("EXECUTION_DEVICES")
        except Exception:
            exec_devices = [str(cfg.device)]
        print(
            f"[Pose] Loaded {self.model_path.name} "
            f"({self.input_w}x{self.input_h}, every {self.interval} frames, "
            f"kpts={self._kpt_count}, devices={exec_devices})"
        )

    def _configure_pose_schema(self, keypoint_count: int) -> None:
        if keypoint_count == 14:
            self._skeleton = self._SKELETON_14
            self._torso_idx = (1, 2, 3, 8, 9)
            self._lower_idx = (10, 11, 12, 13)
            self._head_idx = (0,)
            self._wrist_idx = (6, 7)
            self._shoulder_idx = (2, 3)
            self._hip_idx = (8, 9)
            self._ankle_idx = (12, 13)
            return

        self._skeleton = self._SKELETON_17
        self._torso_idx = (5, 6, 11, 12)
        self._lower_idx = (13, 14, 15, 16)
        self._head_idx = (0, 1, 2, 3, 4)
        self._wrist_idx = (9, 10)
        self._shoulder_idx = (5, 6)
        self._hip_idx = (11, 12)
        self._ankle_idx = (15, 16)

    def _crop_box(self, frame: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, tuple[int, int]] | None:
        frame_h, frame_w = frame.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)

        pad_x = box_w * self.crop_padding
        pad_y = box_h * self.crop_padding
        cx1 = max(0, int(round(x1 - pad_x)))
        cy1 = max(0, int(round(y1 - pad_y)))
        cx2 = min(frame_w, int(round(x2 + pad_x)))
        cy2 = min(frame_h, int(round(y2 + pad_y)))
        if cx2 <= cx1 or cy2 <= cy1:
            return None

        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None
        return crop, (cx1, cy1)

    def _decode_pose_candidates(
        self,
        image_bgr: np.ndarray,
        *,
        origin_xy: tuple[int, int] = (0, 0),
    ) -> list[PoseObservation]:
        image_h, image_w = image_bgr.shape[:2]
        resized, ratio, dw, dh = letterbox(image_bgr, (self.input_w, self.input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        raw = self.compiled_model([blob])[self.output_port]
        pred = raw[0] if raw.ndim == 3 else raw
        if pred.shape[0] < pred.shape[1] and pred.shape[0] < 100:
            pred = pred.transpose()

        if pred.ndim != 2 or pred.shape[1] < (self._keypoint_offset + self._kpt_count * 3):
            return []

        boxes_xywh = pred[:, :4]
        confidence_scores = pred[:, 4]
        keypoints_raw = pred[:, self._keypoint_offset:]

        force_pose = bool(getattr(self.cfg, "pose_force_run", False))
        keep = confidence_scores >= self.conf_thresh
        if not np.any(keep):
            if not force_pose or confidence_scores.size == 0:
                return []
            best_idx = int(np.argmax(confidence_scores))
            keep = np.zeros_like(confidence_scores, dtype=bool)
            keep[best_idx] = True

        pose_boxes = xywh_to_xyxy(boxes_xywh[keep])
        confidence_scores = confidence_scores[keep]
        keypoints_raw = keypoints_raw[keep]
        keep_idx = non_max_suppression(pose_boxes, confidence_scores, self.iou_thresh)
        if not keep_idx:
            return []

        origin_x, origin_y = origin_xy
        observations: list[PoseObservation] = []
        for cand_idx in keep_idx:
            idx = int(cand_idx)
            cand_box = np.asarray(pose_boxes[idx], dtype=np.float32).copy()
            cand_box[0] = (cand_box[0] - dw) / ratio
            cand_box[1] = (cand_box[1] - dh) / ratio
            cand_box[2] = (cand_box[2] - dw) / ratio
            cand_box[3] = (cand_box[3] - dh) / ratio
            cand_box[0] = np.clip(cand_box[0], 0.0, max(0.0, image_w - 1.0))
            cand_box[1] = np.clip(cand_box[1], 0.0, max(0.0, image_h - 1.0))
            cand_box[2] = np.clip(cand_box[2], 0.0, max(0.0, image_w - 1.0))
            cand_box[3] = np.clip(cand_box[3], 0.0, max(0.0, image_h - 1.0))

            keypoints = keypoints_raw[idx].reshape(-1, 3).astype(np.float32)
            keypoints[:, 0] = (keypoints[:, 0] - dw) / ratio
            keypoints[:, 1] = (keypoints[:, 1] - dh) / ratio
            keypoints[:, 0] = np.clip(keypoints[:, 0], 0.0, max(0.0, image_w - 1.0))
            keypoints[:, 1] = np.clip(keypoints[:, 1], 0.0, max(0.0, image_h - 1.0))

            keypoints_full = keypoints.copy()
            keypoints_full[:, 0] += origin_x
            keypoints_full[:, 1] += origin_y
            box_full = cand_box.copy()
            box_full[0] += origin_x
            box_full[1] += origin_y
            box_full[2] += origin_x
            box_full[3] += origin_y

            visible_mask = keypoints[:, 2] >= self.keypoint_conf
            visible_count = int(np.count_nonzero(visible_mask))
            torso_visible_count = int(np.count_nonzero(keypoints[list(self._torso_idx), 2] >= self.keypoint_conf))
            lower_visible_count = int(np.count_nonzero(keypoints[list(self._lower_idx), 2] >= self.keypoint_conf))
            head_mask = keypoints[list(self._head_idx), 2] >= self.keypoint_conf
            wrist_mask = keypoints[list(self._wrist_idx), 2] >= self.keypoint_conf

            box_h = max(1.0, float(cand_box[3] - cand_box[1]))
            if visible_count > 0:
                lowest_visible_ratio = float((np.max(keypoints[visible_mask, 1]) - cand_box[1]) / box_h)
            else:
                lowest_visible_ratio = 0.0

            upright_body = False
            if torso_visible_count >= 2 and lower_visible_count >= 2 and np.count_nonzero(head_mask) >= 1:
                head_y = float(np.median(keypoints[list(self._head_idx), 1][head_mask]))
                shoulders = keypoints[list(self._shoulder_idx)]
                shoulders_mask = shoulders[:, 2] >= self.keypoint_conf
                hips = keypoints[list(self._hip_idx)]
                hips_mask = hips[:, 2] >= self.keypoint_conf
                ankles = keypoints[list(self._ankle_idx)]
                ankles_mask = ankles[:, 2] >= self.keypoint_conf
                if np.count_nonzero(shoulders_mask) >= 1 and np.count_nonzero(hips_mask) >= 1 and np.count_nonzero(ankles_mask) >= 1:
                    shoulder_y = float(np.median(shoulders[:, 1][shoulders_mask]))
                    hip_y = float(np.median(hips[:, 1][hips_mask]))
                    ankle_y = float(np.median(ankles[:, 1][ankles_mask]))
                    upright_body = head_y < shoulder_y < hip_y < ankle_y

            hands_below_head = False
            if np.count_nonzero(head_mask) >= 1 and np.count_nonzero(wrist_mask) == len(self._wrist_idx):
                head_y = float(np.median(keypoints[list(self._head_idx), 1][head_mask]))
                wrists = keypoints[list(self._wrist_idx)]
                hands_below_head = bool(np.all(wrists[:, 1] >= head_y))

            full_body_visible = (
                visible_count >= self.min_visible_keypoints
                and torso_visible_count >= self.min_torso_keypoints
                and lower_visible_count >= self.min_lower_keypoints
                and lowest_visible_ratio >= self.min_lowest_ratio
            )

            observations.append(
                PoseObservation(
                    full_body_visible=full_body_visible,
                    upright_body=upright_body,
                    hands_below_head=hands_below_head,
                    pose_confidence=float(confidence_scores[idx]),
                    visible_count=visible_count,
                    torso_visible_count=torso_visible_count,
                    lower_visible_count=lower_visible_count,
                    lowest_visible_ratio=lowest_visible_ratio,
                    keypoints=keypoints_full,
                    box_xyxy=box_full,
                )
            )
        return observations

    def _match_observation_to_box(
        self,
        observations: list[PoseObservation],
        box: np.ndarray,
    ) -> PoseObservation | None:
        if not observations:
            return None
        target_box = np.asarray(box[:4], dtype=np.float32)
        target_cx = float((target_box[0] + target_box[2]) * 0.5)
        target_cy = float((target_box[1] + target_box[3]) * 0.5)
        target_scale = max(1.0, float(max(target_box[2] - target_box[0], target_box[3] - target_box[1])))

        best_obs: PoseObservation | None = None
        best_score = float("-inf")
        for obs in observations:
            if obs.box_xyxy is None:
                continue
            iou_score = self._box_iou(obs.box_xyxy, target_box)
            obs_cx = float((obs.box_xyxy[0] + obs.box_xyxy[2]) * 0.5)
            obs_cy = float((obs.box_xyxy[1] + obs.box_xyxy[3]) * 0.5)
            dist = float(np.hypot(obs_cx - target_cx, obs_cy - target_cy))
            center_score = 1.0 - min(1.0, dist / target_scale)
            score = (iou_score * 3.0) + center_score + (obs.pose_confidence * 0.25)
            if score > best_score:
                best_score = score
                best_obs = obs
        return best_obs

    def _infer_pose_full_frame(self, frame: np.ndarray, box: np.ndarray) -> PoseObservation | None:
        return self._match_observation_to_box(self._decode_pose_candidates(frame), box)

    def _infer_pose(self, frame: np.ndarray, box: np.ndarray) -> PoseObservation | None:
        if bool(getattr(self.cfg, "pose_full_frame", False)):
            return self._infer_pose_full_frame(frame, box)

        crop_data = self._crop_box(frame, box)
        if crop_data is None:
            return None

        crop, (crop_x, crop_y) = crop_data
        matched = self._match_observation_to_box(
            self._decode_pose_candidates(crop, origin_xy=(crop_x, crop_y)),
            box,
        )
        return matched

    @staticmethod
    def _box_iou(box_a: np.ndarray | None, box_b: np.ndarray | None) -> float:
        if box_a is None or box_b is None:
            return 0.0
        ax1, ay1, ax2, ay2 = [float(v) for v in box_a[:4]]
        bx1, by1, bx2, by2 = [float(v) for v in box_b[:4]]
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter_area
        if union <= 1e-6:
            return 0.0
        return inter_area / union

    def _project_cached_observation(
        self,
        observation: PoseObservation,
        box: np.ndarray,
    ) -> PoseObservation | None:
        if observation.keypoints is None or observation.box_xyxy is None:
            return observation

        src_box = np.asarray(observation.box_xyxy[:4], dtype=np.float32)
        dst_box = np.asarray(box[:4], dtype=np.float32)
        src_w = max(1.0, float(src_box[2] - src_box[0]))
        src_h = max(1.0, float(src_box[3] - src_box[1]))
        dst_w = max(1.0, float(dst_box[2] - dst_box[0]))
        dst_h = max(1.0, float(dst_box[3] - dst_box[1]))

        projected = np.asarray(observation.keypoints, dtype=np.float32).copy()
        projected[:, 0] = ((projected[:, 0] - src_box[0]) / src_w) * dst_w + dst_box[0]
        projected[:, 1] = ((projected[:, 1] - src_box[1]) / src_h) * dst_h + dst_box[1]
        projected[:, 0] = np.clip(projected[:, 0], min(dst_box[0], dst_box[2]), max(dst_box[0], dst_box[2]))
        projected[:, 1] = np.clip(projected[:, 1], min(dst_box[1], dst_box[3]), max(dst_box[1], dst_box[3]))

        return PoseObservation(
            full_body_visible=observation.full_body_visible,
            upright_body=observation.upright_body,
            hands_below_head=observation.hands_below_head,
            pose_confidence=observation.pose_confidence,
            visible_count=observation.visible_count,
            torso_visible_count=observation.torso_visible_count,
            lower_visible_count=observation.lower_visible_count,
            lowest_visible_ratio=observation.lowest_visible_ratio,
            keypoints=projected,
            box_xyxy=dst_box.copy(),
        )

    def update(
        self,
        frame_bgr: np.ndarray,
        visible_boxes: np.ndarray,
        track_ids: np.ndarray,
        skip_track_ids: set[int] | None = None,
    ) -> dict[int, PoseObservation]:
        """Return the latest pose observations for active tracks."""
        if len(visible_boxes) == 0:
            return {}

        skip_track_ids = skip_track_ids or set()
        results: dict[int, PoseObservation] = {}
        active_ids = {int(tid) for tid in track_ids}
        candidates: list[tuple[float, int, np.ndarray]] = []
        force_pose = bool(getattr(self.cfg, "pose_force_run", False))
        use_full_frame = bool(getattr(self.cfg, "pose_full_frame", False))

        for box, tid_raw in zip(visible_boxes, track_ids):
            tid = int(tid_raw)
            self._frame_counts[tid] = self._frame_counts.get(tid, 0) + 1

            if tid in skip_track_ids:
                continue

            cached = self._latest.get(tid)
            cached_iou = self._box_iou(cached.box_xyxy if cached is not None else None, box)
            cache_drifted = cached is not None and cached_iou < self._CACHE_MIN_IOU
            should_infer = (
                force_pose
                or tid not in self._latest
                or (self._frame_counts[tid] - 1) % self.interval == 0
                or cache_drifted
            )
            if should_infer:
                area = float(max(1.0, box[2] - box[0]) * max(1.0, box[3] - box[1]))
                candidates.append((area, tid, box))
            else:
                if cached is not None:
                    if use_full_frame:
                        results[tid] = cached
                    else:
                        projected = self._project_cached_observation(cached, box)
                        if projected is not None:
                            results[tid] = projected

        candidates.sort(key=lambda item: item[0], reverse=True)
        if use_full_frame:
            frame_observations = self._decode_pose_candidates(frame_bgr)
            for _, tid, box in candidates:
                observation = self._match_observation_to_box(frame_observations, box)
                if observation is not None:
                    self._latest[tid] = observation
                cached = self._latest.get(tid)
                if cached is not None:
                    results[tid] = cached
        else:
            infer_limit = (
                len(candidates)
                if force_pose or self.max_tracks_per_frame <= 0
                else self.max_tracks_per_frame
            )
            for _, tid, box in candidates[:infer_limit]:
                observation = self._infer_pose(frame_bgr, box)
                if observation is not None:
                    self._latest[tid] = observation
                cached = self._latest.get(tid)
                if cached is not None:
                    projected = self._project_cached_observation(cached, box)
                    if projected is not None:
                        results[tid] = projected

            for _, tid, box in candidates[infer_limit:]:
                cached = self._latest.get(tid)
                if cached is not None:
                    projected = self._project_cached_observation(cached, box)
                    if projected is not None:
                        results[tid] = projected

        stale_ids = set(self._latest.keys()) - active_ids
        for sid in stale_ids:
            self._latest.pop(sid, None)
            self._frame_counts.pop(sid, None)

        return results

    def draw(self, frame_bgr: np.ndarray, observations: dict[int, PoseObservation]) -> None:
        """Draw pose keypoints for debugging."""
        if not observations:
            return

        force_draw_all = bool(getattr(self.cfg, "pose_force_run", False))
        show_pose_indices = bool(getattr(self.cfg, "show_pose_indices", False))

        for obs in observations.values():
            if obs.keypoints is None:
                continue
            color = (0, 255, 0) if obs.full_body_visible else (0, 165, 255)
            points = obs.keypoints
            for idx, (x, y, conf) in enumerate(points):
                if force_draw_all or conf >= self.keypoint_conf:
                    point_color = color if conf >= self.keypoint_conf else (180, 180, 180)
                    px = int(x)
                    py = int(y)
                    cv2.circle(frame_bgr, (px, py), 2, point_color, -1)
                    if show_pose_indices:
                        cv2.putText(
                            frame_bgr,
                            str(idx),
                            (px + 3, py - 3),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.28,
                            point_color,
                            1,
                            cv2.LINE_AA,
                        )
            for i, j in self._skeleton:
                if points[i, 2] >= self.keypoint_conf and points[j, 2] >= self.keypoint_conf:
                    cv2.line(
                        frame_bgr,
                        (int(points[i, 0]), int(points[i, 1])),
                        (int(points[j, 0]), int(points[j, 1])),
                        color,
                        1,
                    )
