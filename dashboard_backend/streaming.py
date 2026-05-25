from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np


def make_placeholder(w: int = 960, h: int = 540, text: str = "Stream offline - press Start Engine") -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (15, 15, 20)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    tx = (w - text_w) // 2
    ty = (h + text_h) // 2
    cv2.putText(img, text, (tx, ty), font, scale, (220, 220, 230), thickness, cv2.LINE_AA)
    return img


def generate_mjpeg(engine: Any):
    """Yield the latest shared MJPEG frame."""
    last_version = -1
    last_placeholder_key = None
    placeholder_bytes = None
    while True:
        data, version = engine.get_frame_jpeg()
        if data is not None and version != last_version:
            last_version = version
        elif data is None:
            cfg = engine.get_config()
            width = max(1, int(cfg.get("preview_width", 960) or 960))
            height = max(1, int(cfg.get("preview_height", 540) or 540))
            text = "Engine warming up - please wait..." if engine.is_running else "Stream offline - press Start Engine"
            placeholder_key = (width, height, text)
            if placeholder_key != last_placeholder_key:
                img = make_placeholder(width, height, text)
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if not ok:
                    time.sleep(0.1)
                    continue
                placeholder_bytes = buf.tobytes()
                last_placeholder_key = placeholder_key
            data = placeholder_bytes
        else:
            time.sleep(0.01)
            continue

        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
        time.sleep(0.01)


def generate_undistort_mjpeg(engine: Any, w: int | None = None, h: int | None = None, side: str = "both"):
    placeholder = make_placeholder(w or 960, h or 540)
    while True:
        img = engine.get_live_tuning_comparison(side=side)
        if img is None:
            img = placeholder
        elif w and h:
            img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_LINEAR)

        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            time.sleep(0.05)
            continue

        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        time.sleep(0.033)
