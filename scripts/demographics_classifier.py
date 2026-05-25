#!/usr/bin/env python3
"""OpenVINO Demographics (Gender/Age) Classifier."""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

try:
    from openvino import Core
except ImportError:
    from openvino.runtime import Core

class DemographicsClassifier:
    """Runs one or more OpenVINO classification models on tracked bounding box crops."""

    def __init__(self, configs: dict[str, dict], device: str, interval: int = 5, stale_ttl: int = 30):
        """
        :param configs: dict like {'gender': {'model': path, 'labels': path, 'history': 15}, ...}
        :param device: OpenVINO target device
        :param interval: Global inference interval (throttling)
        """
        self.device = str(device)
        self.interval = max(1, interval)
        self.stale_ttl = max(1, int(stale_ttl))
        self.core = Core()
        
        self.models = {}
        # track_histories[tid][model_name] = {"sum": np.array, "count": int}
        self.track_histories = defaultdict(lambda: defaultdict(lambda: {"sum": None, "count": 0}))
        self.frame_counts = defaultdict(int)
        self.stale_counts = defaultdict(int)

        for name, cfg in configs.items():
            model_path = Path(cfg['model'])
            labels_path = Path(cfg['labels'])
            history_len = cfg.get('history', 15)
            preprocess = cfg.get('preprocess', 'openvino_zoo')  # 'openvino_zoo' or 'imagenet_rgb'
            
            ov_model = self.core.read_model(model=str(model_path))
            input_port = ov_model.input(0)
            shape = input_port.partial_shape
            
            # Extract input dimensions
            h = int(shape[2].get_length()) if shape.rank.is_static and shape[2].is_static else 256
            w = int(shape[3].get_length()) if shape.rank.is_static and shape[3].is_static else 192

            if not shape.rank.is_static or not (shape[2].is_static and shape[3].is_static):
                ov_model.reshape({input_port.get_any_name(): [1, 3, h, w]})
                
            compiled = self.core.compile_model(ov_model, self.device)
            labels = self._load_labels(labels_path)
            
            self.models[name] = {
                'compiled': compiled,
                'labels': labels,
                'h': h,
                'w': w,
                'history_len': history_len,
                'preprocess': preprocess,
            }

    def _load_labels(self, path: Path) -> list[str]:
        labels: list[str] = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    labels.append(line.split(":", 1)[1].strip().strip("'\""))
                else:
                    labels.append(line)
        return labels

    def predict_crop(self, model_info: dict, frame_bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        crop = frame_bgr[y1:y2, x1:x2]
        resized = cv2.resize(crop, (model_info['w'], model_info['h']))
        blob = resized.transpose(2, 0, 1)[None].astype(np.float32)
        compiled = model_info['compiled']
        raw_out = compiled([blob])[compiled.output(0)]
        return np.squeeze(raw_out)

    def update(self, frame_bgr: np.ndarray, visible_boxes: np.ndarray, track_ids: np.ndarray) -> dict[int, dict[str, dict[str, float]]]:
        """
        Returns {tid: {model_name: {label: prob, ...}, ...}}
        """
        results = {}
        active_ids = set()
        img_h, img_w = frame_bgr.shape[:2]
        
        for box, tid_raw in zip(visible_boxes, track_ids):
            tid = int(tid_raw)
            active_ids.add(tid)
            self.stale_counts[tid] = 0
            
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue

            # Throttling
            should_infer = (self.frame_counts[tid] % self.interval == 0)
            self.frame_counts[tid] += 1
            
            tid_results = {}
            for name, model_info in self.models.items():
                track_data = self.track_histories[tid][name]
                is_locked = track_data['count'] >= model_info['history_len']
                
                if should_infer and not is_locked:
                    scores = self.predict_crop(model_info, frame_bgr, x1, y1, x2, y2)
                    if len(scores) > 0:
                        if track_data['sum'] is None:
                            track_data['sum'] = scores.copy()
                        else:
                            track_data['sum'] += scores
                        track_data['count'] += 1
                
                if track_data['count'] > 0:
                    avg_scores = track_data['sum'] / track_data['count']
                    tid_results[name] = {
                        model_info['labels'][i]: float(s) 
                        for i, s in enumerate(avg_scores) if i < len(model_info['labels'])
                    }
                    
            if tid_results:
                results[tid] = tid_results
                
        # Cleanup stale IDs only after a grace window, so brief misses do not reset PAR history.
        stale_ids = set(self.track_histories.keys()) - active_ids
        for sid in stale_ids:
            self.stale_counts[sid] += 1
            if self.stale_counts[sid] > self.stale_ttl:
                del self.track_histories[sid]
                del self.frame_counts[sid]
                del self.stale_counts[sid]
             
        return results
