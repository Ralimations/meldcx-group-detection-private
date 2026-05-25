#!/usr/bin/env python3
"""Rolling archive recorder helpers."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np


def safe_segment_name(text: str) -> str:
    import re

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    return cleaned.strip("._-") or "source"


class RollingVideoRecorder:
    """Write timestamped MP4 segments and prune them by retention age."""

    def __init__(
        self,
        root_dir: Path,
        source_name: str,
        fps: float,
        frame_size: tuple[int, int],
        segment_minutes: int,
        retention_hours: float,
        on_segment_open=None,
        on_segment_close=None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.source_name = safe_segment_name(source_name)
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.frame_size = (int(frame_size[0]), int(frame_size[1]))
        self.segment_seconds = max(1, int(segment_minutes) * 60)
        self.retention_seconds = max(0.0, float(retention_hours) * 3600.0)
        self._writer: cv2.VideoWriter | None = None
        self._segment_started_at: float | None = None
        self.current_segment_path: Path | None = None
        self._on_segment_open = on_segment_open
        self._on_segment_close = on_segment_close

    def _segment_path_for_time(self, ts: float) -> Path:
        stamp = time.localtime(ts)
        day_dir = self.root_dir / self.source_name / time.strftime("%Y-%m-%d", stamp)
        day_dir.mkdir(parents=True, exist_ok=True)
        return day_dir / f"{time.strftime('%H-%M-%S', stamp)}.mp4"

    def _prune_old_segments(self, now_ts: float) -> None:
        if self.retention_seconds <= 0:
            return
        cutoff = now_ts - self.retention_seconds
        source_root = self.root_dir / self.source_name
        if not source_root.exists():
            return
        for segment_path in source_root.rglob("*.mp4"):
            try:
                if segment_path.stat().st_mtime < cutoff:
                    segment_path.unlink()
            except OSError:
                continue
        for day_dir in sorted(source_root.glob("*")):
            if day_dir.is_dir():
                try:
                    next(day_dir.iterdir())
                except StopIteration:
                    try:
                        day_dir.rmdir()
                    except OSError:
                        pass

    def _open_segment(self, ts: float) -> None:
        self.close()
        segment_path = self._segment_path_for_time(ts)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(segment_path), fourcc, self.fps, self.frame_size)
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open rolling recorder segment: {segment_path}")
        self._writer = writer
        self._segment_started_at = ts
        self.current_segment_path = segment_path
        if self._on_segment_open is not None:
            self._on_segment_open(segment_path, ts)
        self._prune_old_segments(ts)

    def write(self, frame: np.ndarray, ts: float | None = None) -> None:
        now_ts = time.time() if ts is None else float(ts)
        if self._writer is None or self._segment_started_at is None:
            self._open_segment(now_ts)
        elif now_ts - self._segment_started_at >= self.segment_seconds:
            self._open_segment(now_ts)
        self._writer.write(frame)

    def close(self) -> None:
        closing_path = self.current_segment_path
        closing_started_at = self._segment_started_at
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._on_segment_close is not None and closing_path is not None and closing_started_at is not None:
            self._on_segment_close(closing_path, closing_started_at, time.time())
        self._segment_started_at = None
        self.current_segment_path = None
