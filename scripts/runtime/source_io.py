#!/usr/bin/env python3
"""Video and stream input helpers."""

from __future__ import annotations

import threading
import time

import cv2


def open_video_capture(cap_input):
    cap = cv2.VideoCapture(cap_input)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(cap_input, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input: {cap_input}")
    return cap


class FrameGrabber:
    """Background-thread frame reader.

    For live streams (target_fps=0): reads as fast as possible, always
    delivers the latest frame (dropping older unread frames).

    For file-based mimic_live (target_fps>0): reads at the file's FPS
    and delivers each frame exactly once.  If the consumer calls read()
    before a new frame is ready, it gets (ok, None, eof) so it can spin
    without writing duplicate frames to the output video.
    """

    def __init__(self, cap, target_fps=0):
        self._cap = cap
        self._interval = 1.0 / target_fps if target_fps > 0 else 0
        self._is_throttled = target_fps > 0
        self._frame = None
        self._ok = False
        self._eof = False
        self._new = False  # True when the background thread has a frame the consumer hasn't seen
        self._lock = threading.Lock()
        self._stopped = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stopped:
            t0 = time.perf_counter()
            try:
                ok, frame = self._cap.read()
            except Exception:
                with self._lock:
                    self._ok = False
                    self._frame = None
                    self._eof = True
                    self._new = True
                break
            with self._lock:
                self._ok = ok
                self._frame = frame
                self._eof = not ok
                self._new = True
            if not ok:
                break
            if self._interval > 0:
                elapsed = time.perf_counter() - t0
                sleep_time = self._interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def read(self):
        """Return (ok, frame, eof).

        For throttled (file) sources: returns the frame only once per
        background-thread read.  Returns (ok, None, eof) if the
        consumer is faster than the source FPS — the caller should
        ``continue`` and retry.

        For live sources: always returns the latest frame (even if it
        was already seen) to keep the preview responsive.
        """
        with self._lock:
            if self._is_throttled and not self._new:
                # Consumer is faster than source — no new frame yet.
                return self._ok, None, self._eof
            self._new = False
            return self._ok, self._frame, self._eof

    def stop(self):
        self._stopped = True
