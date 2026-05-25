#!/usr/bin/env python3
"""OpenVINO person ReID embedding extraction."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    from openvino import Core
except ImportError:
    from openvino.runtime import Core


class ReIDExtractor:
    """Extract normalized person embeddings from OpenVINO ReID models."""

    def __init__(self, model_path: str | Path, device: str = "CPU") -> None:
        self.model_path = Path(model_path)
        self.core = Core()
        model = self.core.read_model(str(self.model_path))
        self.compiled_model = self.core.compile_model(model, device)
        self.input = self.compiled_model.input(0)
        self.output = self.compiled_model.output(0)
        shape = self.input.shape
        self.input_h = int(shape[2])
        self.input_w = int(shape[3])

    def extract(self, frame_bgr: np.ndarray, box: np.ndarray) -> np.ndarray | None:
        x1, y1, x2, y2 = [int(v) for v in box[:4]]
        h, w = frame_bgr.shape[:2]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))
        if x2 - x1 < 8 or y2 - y1 < 16:
            return None

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        resized = cv2.resize(crop, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        blob = resized.transpose(2, 0, 1).reshape(1, 3, self.input_h, self.input_w).astype(np.float32)
        embedding = self.compiled_model({self.input: blob})[self.output].reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(embedding))
        if norm <= 1e-12:
            return None
        return embedding / norm

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))
