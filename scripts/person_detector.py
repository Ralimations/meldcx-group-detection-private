#!/usr/bin/env python3
"""Person detection and tracking components."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import math
import heapq
import cv2
import numpy as np

from config import OpenVinoDefaults
from scripts.reid_model import ReIDExtractor

try:
    # OpenVINO 2024+ preferred import path.
    from openvino import Core
except ImportError:
    # Backward compatibility with older OpenVINO releases.
    from openvino.runtime import Core


def letterbox(
    image: np.ndarray,
    new_shape: tuple[int, int],
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, float, float]:
    h0, w0 = image.shape[:2]
    w, h = new_shape

    r = min(w / w0, h / h0)
    new_unpad_w, new_unpad_h = int(round(w0 * r)), int(round(h0 * r))

    dw = (w - new_unpad_w) / 2
    dh = (h - new_unpad_h) / 2

    if (w0, h0) != (new_unpad_w, new_unpad_h):
        image = cv2.resize(image, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return image, r, dw, dh


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    out = boxes.copy()
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def non_max_suppression(
    boxes: np.ndarray,
    confidence_scores: np.ndarray,
    overlap_threshold: float,
) -> list[int]:
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)

    order = confidence_scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter + 1e-7
        iou = inter / union

        inds = np.where(iou <= overlap_threshold)[0]
        order = order[inds + 1]

    return keep


def clip_boxes(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    boxes[:, 0] = boxes[:, 0].clip(0, w - 1)
    boxes[:, 1] = boxes[:, 1].clip(0, h - 1)
    boxes[:, 2] = boxes[:, 2].clip(0, w - 1)
    boxes[:, 3] = boxes[:, 3].clip(0, h - 1)
    return boxes


def box_overlap_ratio(a: np.ndarray, b: np.ndarray) -> float:
    xx1 = max(float(a[0]), float(b[0]))
    yy1 = max(float(a[1]), float(b[1]))
    xx2 = min(float(a[2]), float(b[2]))
    yy2 = min(float(a[3]), float(b[3]))
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return inter / (area_a + area_b - inter + 1e-7)


def filter_to_requested_classes(
    boxes: np.ndarray,
    confidence_scores: np.ndarray,
    class_ids: np.ndarray,
    classes_to_keep: set[int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if classes_to_keep is None:
        return boxes, confidence_scores, class_ids
    if boxes.size == 0:
        return boxes, confidence_scores, class_ids

    mask = np.array([int(class_id) in classes_to_keep for class_id in class_ids], dtype=bool)
    return boxes[mask], confidence_scores[mask], class_ids[mask]


def _group_progress_color(progress: float) -> tuple[int, int, int]:
    """Interpolate bounding box color from green (0.0) -> yellow (0.5) -> red (1.0).
    
    Returns BGR tuple.
    """
    p = max(0.0, min(1.0, progress))
    if p < 0.5:
        # green -> yellow (B stays 0, R goes 0->255, G stays 255)
        t = p / 0.5
        return (0, 255, int(255 * t))
    else:
        # yellow -> red (B stays 0, R stays 255, G goes 255->0)
        t = (p - 0.5) / 0.5
        return (0, int(255 * (1 - t)), 255)


def _parse_hex_color(color_text: str, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    text = str(color_text).strip()
    if not text:
        return fallback
    if text.startswith("#"):
        text = text[1:]
    if len(text) != 6:
        return fallback
    try:
        r = int(text[0:2], 16)
        g = int(text[2:4], 16)
        b = int(text[4:6], 16)
    except ValueError:
        return fallback
    return (b, g, r)


def _parse_label_color_map(
    label_default_color: str | None,
    label_segment_colors: str | None,
) -> tuple[tuple[int, int, int], list[tuple[str, tuple[int, int, int]]]]:
    default_color = _parse_hex_color(label_default_color or "", (0, 255, 255))
    rules: list[tuple[str, tuple[int, int, int]]] = []
    for item in str(label_segment_colors or "").split(","):
        entry = item.strip()
        if not entry or "=" not in entry:
            continue
        prefix, color_text = entry.split("=", 1)
        prefix = prefix.strip()
        if not prefix:
            continue
        rules.append((prefix, _parse_hex_color(color_text, default_color)))
    return default_color, rules


def _segment_color_for_text(
    text: str,
    default_color: tuple[int, int, int],
    color_rules: list[tuple[str, tuple[int, int, int]]],
) -> tuple[int, int, int]:
    stripped = text.strip()
    for prefix, color in color_rules:
        if stripped.startswith(prefix):
            return color
    return default_color


def _tokenize_label_segments(
    text: str,
    default_color: tuple[int, int, int],
    color_rules: list[tuple[str, tuple[int, int, int]]],
) -> list[tuple[str, tuple[int, int, int]]]:
    tokens: list[tuple[str, tuple[int, int, int]]] = []
    chunks = str(text).split(" | ")
    for idx, chunk in enumerate(chunks):
        color = _segment_color_for_text(chunk, default_color, color_rules)
        if idx > 0:
            tokens.append((" | ", default_color))
        for piece in re.findall(r"\S+\s*", chunk):
            if piece:
                tokens.append((piece, color))
    return tokens


def _wrap_colored_tokens(
    tokens: list[tuple[str, tuple[int, int, int]]],
    max_width: int,
    font_scale: float,
    thickness: int,
) -> list[list[tuple[str, tuple[int, int, int]]]]:
    if not tokens or max_width <= 0:
        return []

    def _text_width(text: str) -> int:
        return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0]

    lines: list[list[tuple[str, tuple[int, int, int]]]] = []
    current: list[tuple[str, tuple[int, int, int]]] = []
    current_width = 0

    def _push_line() -> None:
        nonlocal current, current_width
        if current:
            lines.append(current)
        current = []
        current_width = 0

    for token_text, token_color in tokens:
        token_width = _text_width(token_text)
        if token_width <= max_width:
            if current and current_width + token_width > max_width:
                _push_line()
            current.append((token_text, token_color))
            current_width += token_width
            continue

        chunk = ""
        for ch in token_text:
            trial = f"{chunk}{ch}"
            trial_width = _text_width(trial)
            if chunk and current_width + trial_width > max_width:
                current.append((chunk, token_color))
                _push_line()
                chunk = ch
            elif chunk and trial_width > max_width:
                current.append((chunk, token_color))
                _push_line()
                chunk = ch
            else:
                chunk = trial
        if chunk:
            chunk_width = _text_width(chunk)
            if current and current_width + chunk_width > max_width:
                _push_line()
            current.append((chunk, token_color))
            current_width += chunk_width

    _push_line()
    return lines


def draw_detection_boxes(
    image_bgr: np.ndarray,
    boxes: np.ndarray,
    confidence_scores: np.ndarray,
    class_ids: np.ndarray,
    show_body_confidence: bool,
    show_head_confidence: bool,
    show_body: bool = True,
    show_head: bool = True,
    person_ids: np.ndarray | None = None,
    demographics_labels: dict[int, str] | None = None,
    group_progress: dict[int, float] | None = None,
    extra_labels: list[str | None] | None = None,
    label_default_color: str | None = None,
    label_segment_colors: str | None = None,
    cfg=None,
) -> None:
    h_scale = image_bgr.shape[0]
    if cfg is not None:
        label_default_color = label_default_color or getattr(cfg, "label_default_color", None)
        label_segment_colors = label_segment_colors or getattr(cfg, "label_segment_colors", None)
    fs = 0.3 if h_scale < 400 else 0.4 if h_scale < 720 else 0.5
    th = 1
    default_label_color, label_color_rules = _parse_label_color_map(label_default_color, label_segment_colors)

    for i in range(len(boxes)):
        cid = int(class_ids[i])
        
        # Class-specific visibility checks
        if cid == 0 and not show_body: # 0 = person_body
            continue
        if cid == 1 and not show_head: # 1 = person_head
            continue

        x1, y1, x2, y2 = boxes[i].astype(int)
        confidence = confidence_scores[i]
        
        if cid == 0:
            text = f"{confidence:.2f}" if show_body_confidence else ""
        else:
            text = f"{confidence:.2f}" if show_head_confidence else ""
        
        # Legacy support for demographics_labels dict
        if person_ids is not None and demographics_labels is not None:
            tid = int(person_ids[i])
            demo_text = demographics_labels.get(tid, "")
            if demo_text:
                text = f"{demo_text} ({text})" if text else demo_text
        
        # New support for extra_labels list (from tuner)
        if extra_labels is not None and i < len(extra_labels) and extra_labels[i]:
            text = f"{extra_labels[i]} ({text})" if text else extra_labels[i]

        # Determine box color based on group progress
        tid = int(person_ids[i]) if person_ids is not None else -1
        if group_progress and tid in group_progress:
            box_color = _group_progress_color(group_progress[tid])
            progress_val = group_progress[tid]
        else:
            box_color = (0, 255, 0)  # Default green for ungrouped / no tracking
            progress_val = -1.0

        # Draw the bounding box outline
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), box_color, th)

        if tid >= 0 and cid == 0:
            id_text = f"ID {tid}"
            id_fs = max(0.35, fs)
            id_th = max(1, th)
            id_left = max(0, x1 + 2)
            id_top = max(0, y1 + 2)
            (_, id_h), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, id_fs, id_th)
            id_org = (id_left + 1, id_top + id_h + 1)
            cv2.putText(
                image_bgr,
                id_text,
                id_org,
                cv2.FONT_HERSHEY_SIMPLEX,
                id_fs,
                (0, 0, 0),
                id_th + 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image_bgr,
                id_text,
                id_org,
                cv2.FONT_HERSHEY_SIMPLEX,
                id_fs,
                (255, 255, 255),
                id_th,
                cv2.LINE_AA,
            )

        # Draw semi-transparent color tint inside the box for active checking/grouping
        if progress_val >= 0.0:
            overlay = image_bgr[y1:y2, x1:x2].copy()
            tint = np.full_like(overlay, box_color, dtype=np.uint8)
            # Light tint: 8% opacity for subtle effect
            alpha = 0.08 + 0.07 * progress_val  # 8% to 15% as score grows
            cv2.addWeighted(tint, alpha, overlay, 1 - alpha, 0, overlay)
            image_bgr[y1:y2, x1:x2] = overlay
            
        if text:
            frame_w = image_bgr.shape[1]
            max_label_width = max(40, frame_w - x1 - 4)
            colored_tokens = _tokenize_label_segments(text, default_label_color, label_color_rules)
            lines = _wrap_colored_tokens(colored_tokens, max_label_width, fs, th)
            if not lines:
                continue
            line_widths: list[int] = []
            line_heights: list[int] = []
            baseline = 0
            for line in lines:
                width = 0
                max_height = 0
                max_baseline = 0
                for piece, _ in line:
                    (piece_w, piece_h), piece_baseline = cv2.getTextSize(piece, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
                    width += piece_w
                    max_height = max(max_height, piece_h)
                    max_baseline = max(max_baseline, piece_baseline)
                line_widths.append(width)
                line_heights.append(max_height)
                baseline = max(baseline, max_baseline)
            max_line_width = max(line_widths)
            line_gap = max(4, int(6 * fs))
            total_height = sum(line_heights) + line_gap * max(0, len(lines) - 1)

            tx = max(0, min(x1, max(0, frame_w - max_line_width - 2)))
            label_gap = max(8, int(12 * fs))
            if cid == 1:
                box_top = min(
                    max(0, image_bgr.shape[0] - total_height - 1),
                    y2 + label_gap,
                )
            elif y1 >= total_height + label_gap + 2:
                box_top = max(0, y1 - total_height - label_gap)
            else:
                box_top = min(
                    max(0, image_bgr.shape[0] - total_height - 1),
                    y2 + label_gap,
                )

            current_y = box_top
            for line, line_height in zip(lines, line_heights):
                text_y = current_y + line_height
                cursor_x = tx
                for piece, piece_color in line:
                    cv2.putText(
                        image_bgr,
                        piece,
                        (cursor_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fs,
                        (0, 0, 0),
                        th + 2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        image_bgr,
                        piece,
                        (cursor_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fs,
                        piece_color,
                        th,
                    )
                    cursor_x += cv2.getTextSize(piece, cv2.FONT_HERSHEY_SIMPLEX, fs, th)[0][0]
                current_y = text_y + line_gap


class PersonDetector:
    """OpenVINO person detector wrapper."""

    def __init__(
        self,
        model_path: str | Path,
        cfg: OpenVinoDefaults,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = str(cfg.device)
        self.default_image_size = int(cfg.imgsz)
        self.head_confidence_threshold = float(getattr(cfg, "head_detection_conf", getattr(cfg, "person_detection_conf", 0.25)))
        self.head_overlap_threshold = float(getattr(cfg, "head_overlap_threshold", getattr(cfg, "person_overlap_threshold", 0.45)))
        self.core = Core()

        ov_model = self.core.read_model(model=str(self.model_path))
        input_port = ov_model.input(0)
        input_pshape = input_port.partial_shape
        if input_pshape.rank.is_static and int(input_pshape.rank.get_length()) != 4:
            raise RuntimeError(f"Expected NCHW input, got shape: {input_pshape}")

        h_is_static = input_pshape[2].is_static
        w_is_static = input_pshape[3].is_static
        if h_is_static and w_is_static:
            self.input_h = int(input_pshape[2].get_length())
            self.input_w = int(input_pshape[3].get_length())
        else:
            # If imgsz is 0 (Native mode), use 640 as a temporary placeholder until reshaped.
            # YOLO models require input dimensions to be multiples of 32 (max stride).
            init_size = self.default_image_size if self.default_image_size > 0 else 640
            init_size = max(32, (init_size // 32) * 32)
            ov_model.reshape({input_port.get_any_name(): [1, 3, init_size, init_size]})
            self.input_h, self.input_w = init_size, init_size

        print(f"[Detector] Compiling model on {self.device}...", flush=True)
        self.compiled_model = self.core.compile_model(ov_model, self.device)
        try:
            exec_devices = self.compiled_model.get_property("EXECUTION_DEVICES")
            print(f"[Detector] Executing on: {exec_devices}", flush=True)
        except Exception:
            pass

    def reshape(self, new_size: int) -> None:
        """Reshapes the AI model input layer to a new square resolution."""
        if new_size == self.input_h and new_size == self.input_w:
            return

        print(f"[Detector] Reshaping AI input layer to {new_size}x{new_size}...")
        ov_model = self.core.read_model(model=str(self.model_path))
        input_port = ov_model.input(0)

        # Ensure new_size is a multiple of 32 for YOLO
        new_size = (new_size // 32) * 32
        if new_size < 32: new_size = 32

        ov_model.reshape({input_port.get_any_name(): [1, 3, new_size, new_size]})
        self.input_h, self.input_w = new_size, new_size
        self.compiled_model = self.core.compile_model(ov_model, self.device)
        try:
            exec_devices = self.compiled_model.get_property("EXECUTION_DEVICES")
            print(f"[Detector] Executing on: {exec_devices}")
        except Exception:
            pass
        print(f"[Detector] Compiled successfully at {new_size}x{new_size}.")

    def detect_frame(
        self,
        frame_bgr: np.ndarray,
        confidence_threshold: float,
        overlap_threshold: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        orig_h, orig_w = frame_bgr.shape[:2]
        resized, r, dw, dh = letterbox(frame_bgr, (self.input_w, self.input_h))

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        raw = self.compiled_model([blob])[self.compiled_model.output(0)]
        pred = raw[0] if raw.ndim == 3 else raw

        # Support for YOLOv8/v11 (1, features, boxes) vs YOLOv5 (1, boxes, features)
        if pred.shape[0] < pred.shape[1] and pred.shape[0] < 20:
            # features < boxes, likely YOLOv8/v11 format
            pred = pred.transpose()

        if pred.ndim != 2 or pred.shape[1] < 5:
            raise RuntimeError(f"Unexpected output shape: {raw.shape}")

        # YOLOv8/v11 has no separate objectness score; scores are class-specific
        if pred.shape[1] == 4 + len(self.compiled_model.output(0).get_names()) or pred.shape[1] < 10:
            # No objectness score (YOLOv8/v11 style)
            boxes_xywh = pred[:, :4]
            cls_scores = pred[:, 4:]
            class_ids = cls_scores.argmax(axis=1)
            confidence_scores = cls_scores[np.arange(cls_scores.shape[0]), class_ids]
        else:
            # With objectness score (YOLOv5/v7 style, features >= 6)
            boxes_xywh = pred[:, :4]
            obj = pred[:, 4]
            cls_scores = pred[:, 5:]
            class_ids = cls_scores.argmax(axis=1)
            cls_conf = cls_scores[np.arange(cls_scores.shape[0]), class_ids]
            confidence_scores = obj * cls_conf

        class_thresholds = {
            0: float(confidence_threshold),
            1: float(getattr(self, "head_confidence_threshold", confidence_threshold)),
        }
        keep = np.array(
            [
                float(score) >= class_thresholds.get(int(class_id), float(confidence_threshold))
                for score, class_id in zip(confidence_scores, class_ids)
            ],
            dtype=bool,
        )
        if not np.any(keep):
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            )

        boxes = xywh_to_xyxy(boxes_xywh[keep])
        confidence_scores = confidence_scores[keep]
        class_ids = class_ids[keep]

        final_idx: list[int] = []
        for class_id in np.unique(class_ids):
            idx = np.where(class_ids == class_id)[0]
            class_overlap_threshold = (
                self.head_overlap_threshold if int(class_id) == 1 else float(overlap_threshold)
            )
            keep_idx = non_max_suppression(boxes[idx], confidence_scores[idx], class_overlap_threshold)
            final_idx.extend(idx[k] for k in keep_idx)

        if not final_idx:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            )

        final_idx = sorted(final_idx, key=lambda i: float(confidence_scores[i]), reverse=True)
        boxes = boxes[final_idx]
        confidence_scores = confidence_scores[final_idx]
        class_ids = class_ids[final_idx]

        boxes[:, [0, 2]] -= dw
        boxes[:, [1, 3]] -= dh
        boxes /= r
        boxes = clip_boxes(boxes, orig_w, orig_h)

        return boxes, confidence_scores, class_ids


@dataclass
class PersonTrack:
    person_id: int | None
    class_id: int
    box: np.ndarray
    confidence: float
    seen_frames: int = 0
    missing_frames: int = 0
    is_confirmed: bool = False
    hist: np.ndarray | None = None
    embedding: np.ndarray | None = None
    reid_quality_hits: int = 0
    reid_identity_candidate: int | None = None
    reid_identity_candidate_hits: int = 0


class PersonTracker:
    def __init__(self, cfg: OpenVinoDefaults) -> None:
        self.cfg = cfg
        self.overlap_threshold = cfg.track_overlap_threshold
        self.frames_to_confirm = max(1, cfg.min_hits)
        self.max_missing_frames = max(1, cfg.max_misses)
        self.max_missing_to_show = max(0, cfg.track_max_draw_misses)
        self.box_smoothing = float(np.clip(cfg.smooth, 0.0, 0.95))
        self.min_confidence_for_new_track = float(np.clip(cfg.track_min_conf, 0.0, 1.0))
        
        self.reid_window = getattr(cfg, "track_reid_window", 300)
        self.reid_similarity_thresh = getattr(cfg, "track_reid_similarity_thresh", 0.85)
        self.reid_update_interval = max(1, int(getattr(cfg, "track_reid_update_interval", 5)))
        self.reid_backend = str(getattr(cfg, "track_reid_backend", "histogram")).strip().lower()
        if self.reid_backend not in {"off", "histogram", "openvino", "hybrid"}:
            self.reid_backend = "histogram"
        self.reid_require_not_touching_frame_edge = bool(
            getattr(cfg, "track_reid_require_not_touching_frame_edge", True)
        )
        self.reid_min_quality_frames = max(1, int(getattr(cfg, "track_reid_min_quality_frames", 5)))
        self.reid_identity_confirm_frames = max(1, int(getattr(cfg, "track_reid_identity_confirm_frames", 3)))
        self.reid_use_spatial_gate = bool(getattr(cfg, "track_reid_use_spatial_gate", False))
        self.reid_spatial_gate_scale = max(0.0, float(getattr(cfg, "track_reid_spatial_gate_scale", 2.5)))
        self.reid_embedding_similarity_thresh = float(
            getattr(cfg, "track_reid_embedding_similarity_thresh", 0.75)
        )
        self.reid_extractor: ReIDExtractor | None = None
        if self.reid_backend in {"openvino", "hybrid"} and self.reid_window > 0:
            model_path_str = str(getattr(cfg, "track_reid_model", "")).strip()
            if model_path_str:
                model_path = Path(model_path_str)
                try:
                    self.reid_extractor = ReIDExtractor(model_path=model_path, device=cfg.device)
                    print(
                        f"[ReID] Loaded {model_path.name} "
                        f"({self.reid_extractor.input_w}x{self.reid_extractor.input_h}, backend={self.reid_backend})"
                    )
                except Exception as exc:
                    print(f"[Warning] ReID model not available: {exc}")
                    if self.reid_backend == "openvino":
                        print("         Falling back to histogram ReID.")
                        self.reid_backend = "histogram"
                    elif self.reid_backend == "hybrid":
                        print("         Hybrid ReID will use histogram fallback only.")

        self.people: list[PersonTrack] = []
        self.lost_people: list[PersonTrack] = []
        self.next_person_id = 1
        self.unique_confirmed_count = 0
        self.available_person_ids: list[int] = []

    def _assign_confirmed_id(self, person: PersonTrack) -> None:
        if person.person_id is not None:
            return
        if self.available_person_ids:
            person.person_id = heapq.heappop(self.available_person_ids)
            return
        person.person_id = self.next_person_id
        self.next_person_id += 1
        self.unique_confirmed_count += 1

    def _release_person_id(self, person_id: int | None) -> None:
        if person_id is None or person_id <= 0:
            return
        for person in self.people:
            if person.person_id == person_id:
                return
        for person in self.lost_people:
            if person.person_id == person_id:
                return
        if person_id in self.available_person_ids:
            return
        heapq.heappush(self.available_person_ids, person_id)

    def _reid_identity_required(self) -> bool:
        return self.reid_backend in {"openvino", "hybrid"}

    def _extract_histogram(self, frame_bgr: np.ndarray, box: np.ndarray) -> np.ndarray | None:
        x1, y1, x2, y2 = box.astype(int)
        h, w = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None
            
        crop = frame_bgr[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # Focus on the torso (middle 50% vertically) as legs often get occluded / are uniform
        ch, cw = hsv.shape[:2]
        torso = hsv[int(ch*0.2):int(ch*0.7), :]
        if torso.size == 0:
            return None
            
        # 16 bins for Hue, 16 for Saturation
        hist = cv2.calcHist([torso], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    def _extract_embedding(self, frame_bgr: np.ndarray, box: np.ndarray) -> np.ndarray | None:
        if self.reid_extractor is None:
            return None
        return self.reid_extractor.extract(frame_bgr, box)

    def _is_reid_quality_box(self, frame_bgr: np.ndarray, box: np.ndarray) -> bool:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box[:4]]
        if self.reid_require_not_touching_frame_edge:
            if not (x1 > 0 and y1 > 0 and x2 < w and y2 < h):
                return False

        if bool(getattr(self.cfg, "use_roi", False)):
            roi_polygon = getattr(self.cfg, "roi_polygon", ())
            if roi_polygon and len(roi_polygon) >= 3:
                roi_pts = np.array(
                    [(int(px * w), int(py * h)) for px, py in roi_polygon],
                    dtype=np.int32,
                )
                test_points = (
                    (float(x1), float(y1)),
                    (float(x2), float(y1)),
                    (float(x1), float(y2)),
                    (float(x2), float(y2)),
                )
                if any(cv2.pointPolygonTest(roi_pts, pt, False) <= 0 for pt in test_points):
                    return False

        return True

    def _match_identity_from_lost_memory(
        self,
        frame_bgr: np.ndarray,
        box: np.ndarray,
        class_id: int,
    ) -> PersonTrack | None:
        if self.reid_window <= 0 or not self.lost_people:
            return None
        if not self._is_reid_quality_box(frame_bgr, box):
            return None

        det_hist: np.ndarray | None = None
        det_embedding: np.ndarray | None = None
        best_score = -1.0
        best_lost_idx = -1

        for lost_idx, lost_p in enumerate(self.lost_people):
            if lost_p.class_id != class_id:
                continue

            hist_sim = None
            embedding_sim = None

            if self.reid_backend in {"histogram", "hybrid"} and lost_p.hist is not None:
                if det_hist is None:
                    det_hist = self._extract_histogram(frame_bgr, box)
                if det_hist is not None:
                    hist_sim = float(cv2.compareHist(det_hist, lost_p.hist, cv2.HISTCMP_CORREL))

            if self.reid_backend in {"openvino", "hybrid"} and lost_p.embedding is not None and self.reid_extractor is not None:
                if det_embedding is None:
                    det_embedding = self._extract_embedding(frame_bgr, box)
                if det_embedding is not None:
                    embedding_sim = self.reid_extractor.cosine_similarity(det_embedding, lost_p.embedding)

            score = None
            if self.reid_backend == "histogram":
                if hist_sim is not None and hist_sim >= self.reid_similarity_thresh:
                    score = hist_sim
            elif self.reid_backend == "openvino":
                if embedding_sim is not None and embedding_sim >= self.reid_embedding_similarity_thresh:
                    score = embedding_sim
            elif self.reid_backend == "hybrid":
                if embedding_sim is not None and embedding_sim >= self.reid_embedding_similarity_thresh:
                    score = embedding_sim + (0.1 * hist_sim if hist_sim is not None else 0.0)
                elif hist_sim is not None and hist_sim >= self.reid_similarity_thresh:
                    score = hist_sim

            if score is not None and score > best_score:
                best_score = score
                best_lost_idx = lost_idx

        if best_lost_idx < 0:
            return None

        remembered_person = self.lost_people.pop(best_lost_idx)
        if det_hist is not None:
            remembered_person.hist = det_hist if remembered_person.hist is None else cv2.addWeighted(
                remembered_person.hist, 0.7, det_hist, 0.3, 0
            )
        if det_embedding is not None:
            if remembered_person.embedding is None:
                remembered_person.embedding = det_embedding
            else:
                blended = 0.7 * remembered_person.embedding + 0.3 * det_embedding
                norm = float(np.linalg.norm(blended))
                if norm > 1e-12:
                    remembered_person.embedding = blended / norm
        return remembered_person

    def _find_lost_identity_candidate(
        self,
        frame_bgr: np.ndarray,
        box: np.ndarray,
        class_id: int,
    ) -> tuple[PersonTrack | None, np.ndarray | None, np.ndarray | None]:
        if self.reid_window <= 0 or not self.lost_people:
            return None, None, None
        if not self._is_reid_quality_box(frame_bgr, box):
            return None, None, None

        det_hist: np.ndarray | None = None
        det_embedding: np.ndarray | None = None
        best_score = -1.0
        best_match: PersonTrack | None = None

        for lost_p in self.lost_people:
            if lost_p.class_id != class_id:
                continue

            hist_sim = None
            embedding_sim = None

            if self.reid_backend in {"histogram", "hybrid"} and lost_p.hist is not None:
                if det_hist is None:
                    det_hist = self._extract_histogram(frame_bgr, box)
                if det_hist is not None:
                    hist_sim = float(cv2.compareHist(det_hist, lost_p.hist, cv2.HISTCMP_CORREL))

            if self.reid_backend in {"openvino", "hybrid"} and lost_p.embedding is not None and self.reid_extractor is not None:
                if det_embedding is None:
                    det_embedding = self._extract_embedding(frame_bgr, box)
                if det_embedding is not None:
                    embedding_sim = self.reid_extractor.cosine_similarity(det_embedding, lost_p.embedding)

            score = None
            if self.reid_backend == "histogram":
                if hist_sim is not None and hist_sim >= self.reid_similarity_thresh:
                    score = hist_sim
            elif self.reid_backend == "openvino":
                if embedding_sim is not None and embedding_sim >= self.reid_embedding_similarity_thresh:
                    score = embedding_sim
            elif self.reid_backend == "hybrid":
                if embedding_sim is not None and embedding_sim >= self.reid_embedding_similarity_thresh:
                    score = embedding_sim + (0.1 * hist_sim if hist_sim is not None else 0.0)
                elif hist_sim is not None and hist_sim >= self.reid_similarity_thresh:
                    score = hist_sim

            if score is not None and score > best_score:
                best_score = score
                best_match = lost_p

        return best_match, det_hist, det_embedding

    def _remove_lost_identity(self, person_id: int | None) -> None:
        if person_id is None:
            return
        for idx, lost_p in enumerate(self.lost_people):
            if lost_p.person_id == person_id:
                self.lost_people.pop(idx)
                return

    def _update_reid_quality_hits(self, person: PersonTrack, frame_bgr: np.ndarray, box: np.ndarray) -> bool:
        is_quality_box = self._is_reid_quality_box(frame_bgr, box)
        if is_quality_box:
            person.reid_quality_hits += 1
        else:
            person.reid_quality_hits = 0
            person.reid_identity_candidate = None
            person.reid_identity_candidate_hits = 0
        return person.reid_quality_hits >= self.reid_min_quality_frames

    def _finalize_confirmed_identity(self, person: PersonTrack, frame_bgr: np.ndarray, box: np.ndarray) -> None:
        if not person.is_confirmed:
            return
        if not self._reid_identity_required():
            if person.person_id is None:
                self._assign_confirmed_id(person)
            return
        if person.reid_quality_hits < self.reid_min_quality_frames:
            return

        remembered_person, det_hist, det_embedding = self._find_lost_identity_candidate(frame_bgr, box, person.class_id)
        candidate_id = remembered_person.person_id if remembered_person is not None else person.person_id
        if candidate_id is None:
            candidate_id = -1

        if person.reid_identity_candidate != candidate_id:
            person.reid_identity_candidate = candidate_id
            person.reid_identity_candidate_hits = 1
            return

        person.reid_identity_candidate_hits += 1
        if person.reid_identity_candidate_hits < self.reid_identity_confirm_frames:
            return

        if remembered_person is not None:
            previous_person_id = person.person_id
            person.person_id = remembered_person.person_id
            if person.hist is None:
                person.hist = remembered_person.hist if det_hist is None else det_hist
            if person.embedding is None:
                person.embedding = remembered_person.embedding if det_embedding is None else det_embedding
            self._remove_lost_identity(remembered_person.person_id)
            if previous_person_id is not None and previous_person_id != person.person_id:
                self._release_person_id(previous_person_id)
            return

        if person.person_id is None:
            self._assign_confirmed_id(person)

    def _update_track_reid_features(self, person: PersonTrack, frame_bgr: np.ndarray, box: np.ndarray) -> None:
        if not person.is_confirmed or self.reid_window <= 0:
            return
        if person.seen_frames % self.reid_update_interval != 0:
            return
        if person.reid_quality_hits < self.reid_min_quality_frames:
            return

        if self.reid_backend in {"histogram", "hybrid"}:
            new_hist = self._extract_histogram(frame_bgr, box)
            if new_hist is not None:
                if person.hist is None:
                    person.hist = new_hist
                else:
                    person.hist = cv2.addWeighted(person.hist, 0.9, new_hist, 0.1, 0)

        if self.reid_backend in {"openvino", "hybrid"} and self.reid_extractor is not None:
            new_embedding = self._extract_embedding(frame_bgr, box)
            if new_embedding is not None:
                if person.embedding is None:
                    person.embedding = new_embedding
                else:
                    blended = 0.9 * person.embedding + 0.1 * new_embedding
                    norm = float(np.linalg.norm(blended))
                    if norm > 1e-12:
                        person.embedding = blended / norm

    def _restore_lost_person(
        self,
        frame_bgr: np.ndarray,
        box: np.ndarray,
        confidence: float,
        class_id: int,
    ) -> tuple[PersonTrack, bool] | None:
        if self.reid_window <= 0:
            return None

        bx1, by1, bx2, by2 = box
        bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
        bh = max(1e-3, by2 - by1)
        det_hist: np.ndarray | None = None
        det_embedding: np.ndarray | None = None
        best_score = -1.0
        best_lost_idx = -1

        reid_candidates: list[tuple[str, int, PersonTrack]] = []
        for person_idx, person in enumerate(self.people):
            if person.missing_frames > 0 and person.is_confirmed:
                reid_candidates.append(("active", person_idx, person))
        for lost_idx, lost_p in enumerate(self.lost_people):
            reid_candidates.append(("lost", lost_idx, lost_p))

        if not reid_candidates:
            return None

        best_candidate_bucket: str | None = None
        for candidate_bucket, candidate_idx, lost_p in reid_candidates:
            if lost_p.class_id != class_id:
                continue

            lx1, ly1, lx2, ly2 = lost_p.box
            lcx, lcy = (lx1 + lx2) / 2, (ly1 + ly2) / 2
            lh = max(1e-3, ly2 - ly1)
            if self.reid_use_spatial_gate:
                dist = math.hypot(bcx - lcx, bcy - lcy)
                max_h = max(bh, lh)
                if dist >= (max_h * self.reid_spatial_gate_scale):
                    continue

            hist_sim = None
            embedding_sim = None

            if self.reid_backend in {"histogram", "hybrid"} and lost_p.hist is not None:
                if det_hist is None:
                    det_hist = self._extract_histogram(frame_bgr, box)
                if det_hist is not None:
                    hist_sim = float(cv2.compareHist(det_hist, lost_p.hist, cv2.HISTCMP_CORREL))

            if self.reid_backend in {"openvino", "hybrid"} and lost_p.embedding is not None and self.reid_extractor is not None:
                if det_embedding is None:
                    det_embedding = self._extract_embedding(frame_bgr, box)
                if det_embedding is not None:
                    embedding_sim = self.reid_extractor.cosine_similarity(det_embedding, lost_p.embedding)

            score = None
            if self.reid_backend == "histogram":
                if hist_sim is not None and hist_sim >= self.reid_similarity_thresh:
                    score = hist_sim
            elif self.reid_backend == "openvino":
                if embedding_sim is not None and embedding_sim >= self.reid_embedding_similarity_thresh:
                    score = embedding_sim
            elif self.reid_backend == "hybrid":
                if embedding_sim is not None and embedding_sim >= self.reid_embedding_similarity_thresh:
                    score = embedding_sim + (0.1 * hist_sim if hist_sim is not None else 0.0)
                elif hist_sim is not None and hist_sim >= self.reid_similarity_thresh:
                    score = hist_sim

            if score is not None and score > best_score:
                best_score = score
                best_lost_idx = candidate_idx
                best_candidate_bucket = candidate_bucket

        if best_lost_idx < 0 or best_candidate_bucket is None:
            return None

        if best_candidate_bucket == "active":
            restored_person = self.people[best_lost_idx]
            restored_from_lost_pool = False
        else:
            restored_person = self.lost_people.pop(best_lost_idx)
            restored_from_lost_pool = True
        restored_person.box = box.astype(np.float32).copy()
        restored_person.confidence = float(confidence)
        restored_person.missing_frames = 0
        restored_person.seen_frames += 1
        if restored_person.is_confirmed and restored_person.person_id is None:
            self._assign_confirmed_id(restored_person)

        if det_hist is not None:
            if restored_person.hist is None:
                restored_person.hist = det_hist
            else:
                restored_person.hist = cv2.addWeighted(restored_person.hist, 0.7, det_hist, 0.3, 0)
        if det_embedding is not None:
            if restored_person.embedding is None:
                restored_person.embedding = det_embedding
            else:
                blended = 0.7 * restored_person.embedding + 0.3 * det_embedding
                norm = float(np.linalg.norm(blended))
                if norm > 1e-12:
                    restored_person.embedding = blended / norm
        return restored_person, restored_from_lost_pool

    def update(self, frame_bgr: np.ndarray, boxes: np.ndarray, confidence_scores: np.ndarray, class_ids: np.ndarray) -> None:
        matched_people_indices: set[int] = set()
        matched_detection_indices: set[int] = set()

        # Match detections to existing people by descending overlap within the same class.
        candidate_matches: list[tuple[float, int, int]] = []
        for person_idx, person in enumerate(self.people):
            for det_idx in range(len(boxes)):
                if int(class_ids[det_idx]) != person.class_id:
                    continue
                overlap = box_overlap_ratio(person.box, boxes[det_idx])
                if overlap >= self.overlap_threshold:
                    candidate_matches.append((overlap, person_idx, det_idx))

        candidate_matches.sort(key=lambda match: match[0], reverse=True)
        for _, person_idx, det_idx in candidate_matches:
            if person_idx in matched_people_indices or det_idx in matched_detection_indices:
                continue
            person = self.people[person_idx]
            detected_box = boxes[det_idx].astype(np.float32)
            person.box = self.box_smoothing * person.box + (1.0 - self.box_smoothing) * detected_box
            person.confidence = 0.7 * person.confidence + 0.3 * float(confidence_scores[det_idx])
            person.seen_frames += 1
            person.missing_frames = 0
            was_confirmed = person.is_confirmed
            person.is_confirmed = person.is_confirmed or person.seen_frames >= self.frames_to_confirm
            self._update_reid_quality_hits(person, frame_bgr, detected_box)
            if person.is_confirmed:
                self._finalize_confirmed_identity(person, frame_bgr, detected_box)

            matched_people_indices.add(person_idx)
            matched_detection_indices.add(det_idx)
            
            self._update_track_reid_features(person, frame_bgr, detected_box)

        for person_idx, person in enumerate(self.people):
            if person_idx not in matched_people_indices:
                person.missing_frames += 1
                person.reid_quality_hits = 0
                person.reid_identity_candidate = None
                person.reid_identity_candidate_hits = 0

        for det_idx in range(len(boxes)):
            if det_idx in matched_detection_indices:
                continue

            class_id = int(class_ids[det_idx])
            restored_match = self._restore_lost_person(
                frame_bgr,
                boxes[det_idx],
                float(confidence_scores[det_idx]),
                class_id,
            )
            if restored_match is not None:
                restored_person, restored_from_lost_pool = restored_match
                if restored_from_lost_pool:
                    self.people.append(restored_person)
                matched_detection_indices.add(det_idx)
                continue

            # Filter out very low-confidence detections before creating a new person track.
            if float(confidence_scores[det_idx]) < self.min_confidence_for_new_track:
                continue

            person = PersonTrack(
                person_id=None,
                class_id=class_id,
                box=boxes[det_idx].astype(np.float32).copy(),
                confidence=float(confidence_scores[det_idx]),
                seen_frames=1,
                missing_frames=0,
                is_confirmed=self.frames_to_confirm <= 1,
            )
            self._update_reid_quality_hits(person, frame_bgr, person.box)
            if person.reid_quality_hits >= self.reid_min_quality_frames:
                if self.reid_backend in {"histogram", "hybrid"}:
                    person.hist = self._extract_histogram(frame_bgr, person.box)
                if self.reid_backend in {"openvino", "hybrid"} and self.reid_extractor is not None:
                    person.embedding = self._extract_embedding(frame_bgr, person.box)
                
            if person.is_confirmed:
                self._finalize_confirmed_identity(person, frame_bgr, person.box)
                
            self.people.append(person)

        active_people = []
        for person in self.people:
            if person.missing_frames <= self.max_missing_frames:
                active_people.append(person)
            elif person.is_confirmed:
                self.lost_people.append(person)
        self.people = active_people
        
        # Decay and TTL for lost memory
        if self.reid_window > 0:
            retained_lost = []
            for p in self.lost_people:
                p.missing_frames += 1
                if p.missing_frames <= self.reid_window:
                    retained_lost.append(p)
            self.lost_people = retained_lost
        else:
            self.lost_people = []

    def get_visible_people(self, include_tentative: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        visible_people = [
            person for person in self.people
            if (include_tentative or person.is_confirmed) and person.missing_frames <= self.max_missing_to_show
        ]
        if not visible_people:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.int64),
            )

        boxes = np.stack([person.box for person in visible_people]).astype(np.float32)
        confidence_scores = np.array([person.confidence for person in visible_people], dtype=np.float32)
        class_ids = np.array([person.class_id for person in visible_people], dtype=np.int64)
        person_ids = np.array(
            [int(person.person_id) if person.person_id is not None else -1 for person in visible_people],
            dtype=np.int64,
        )
        return boxes, confidence_scores, class_ids, person_ids
