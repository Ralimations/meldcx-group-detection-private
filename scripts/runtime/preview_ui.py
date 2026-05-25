#!/usr/bin/env python3
"""Preview and ROI persistence helpers."""

from __future__ import annotations

from pathlib import Path

import cv2

from scripts.configuration.runtime_config import (
    load_source_roi_settings,
    save_source_roi_settings,
    source_key_for_runtime,
)


def configure_preview_window(window_name: str, cfg, *, sys_module, ctypes_module) -> None:
    mode = getattr(cfg, "preview_window_mode", "normal").lower()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if mode == "fullscreen":
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    elif mode == "maximized":
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        if sys_module.platform.startswith("win"):
            hwnd = ctypes_module.windll.user32.FindWindowW(None, window_name)
            if hwnd:
                ctypes_module.windll.user32.ShowWindow(hwnd, 3)
    else:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, cfg.preview_width, cfg.preview_height)


def roi_source_key(*, mode: str, camera_index: int, is_stream: bool, source_str: str, source_path: Path | None) -> str:
    if mode == "camera":
        return source_key_for_runtime("", mode="camera", camera_index=camera_index)
    if is_stream:
        return source_key_for_runtime(source_str, mode="source", camera_index=camera_index)
    if source_path is not None:
        return source_key_for_runtime(str(source_path), mode="source", camera_index=camera_index)
    return source_key_for_runtime(source_str, mode="source", camera_index=camera_index)


def save_roi_for_source(source_key: str, roi_polygon, height_roi_polygon) -> None:
    save_source_roi_settings(source_key, roi_polygon, height_roi_polygon)


def load_roi_for_source(source_key: str):
    return load_source_roi_settings(source_key)


def build_editor_state(cfg) -> dict:
    return {
        "mode": "none",
        "dragging_idx": -1,
        "mouse_pos": (0, 0),
        "frame_w": cfg.preview_width,
        "frame_h": cfg.preview_height,
        "roi": list(getattr(cfg, "roi_polygon", [])),
        "height_roi": list(getattr(cfg, "height_roi_polygon", getattr(cfg, "roi_polygon", []))),
        "doorway_roi": list(getattr(cfg, "doorway_roi_polygon", [])),
        "age_adult_threshold_px": float(getattr(cfg, "age_adult_threshold_px", 240.0)),
    }


def norm_to_px(norm_pt, w, h):
    return (int(norm_pt[0] * w), int(norm_pt[1] * h))


def bind_editor_mouse(window_name: str, editor_state: dict) -> None:
    def mouse_callback(event, x, y, flags, param):
        editor_state["mouse_pos"] = (x, y)
        if editor_state["mode"] == "none":
            return
        h = editor_state["frame_h"]
        w = editor_state["frame_w"]
        if h <= 0 or w <= 0:
            return
        if editor_state["mode"] not in ("roi", "height_roi", "doorway_roi"):
            return
        if editor_state["mode"] == "roi":
            current_shape = editor_state["roi"]
        elif editor_state["mode"] == "height_roi":
            current_shape = editor_state["height_roi"]
        else:
            current_shape = editor_state["doorway_roi"]
        if event == cv2.EVENT_LBUTTONDOWN:
            min_dist = 60
            best_idx = -1
            for i, p_norm in enumerate(current_shape):
                px, py = norm_to_px(p_norm, w, h)
                dist = ((px - x) ** 2 + (py - y) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
            if best_idx != -1:
                editor_state["dragging_idx"] = best_idx
        elif event == cv2.EVENT_LBUTTONUP:
            editor_state["dragging_idx"] = -1
        elif event == cv2.EVENT_MOUSEMOVE and editor_state["dragging_idx"] != -1:
            nx = max(0.0, min(1.0, float(x) / w))
            ny = max(0.0, min(1.0, float(y) / h))
            current_shape[editor_state["dragging_idx"]] = (nx, ny)

    cv2.setMouseCallback(window_name, mouse_callback)


def handle_preview_key(raw_key: int, *, editor_state: dict, app) -> tuple[bool, bool]:
    key = raw_key & 0xFF
    end_all = False
    should_break = False
    if raw_key != -1 and editor_state["mode"] != "none":
        print(f"[DEBUG] Key pressed: {raw_key} (masked: {key})")
    if key == ord("e"):
        end_all = True
        should_break = True
    elif key == ord("q") or (key == 27 and editor_state["mode"] == "none"):
        should_break = True
    elif key == ord("r"):
        print(f"[DEBUG] 'r' pressed. Mode: {editor_state['mode']} -> {'roi' if editor_state['mode'] != 'roi' else 'none'}")
        editor_state["mode"] = "none" if editor_state["mode"] == "roi" else "roi"
    elif key == ord("h"):
        print(f"[DEBUG] 'h' pressed. Mode: {editor_state['mode']} -> {'height_roi' if editor_state['mode'] != 'height_roi' else 'none'}")
        editor_state["mode"] = "none" if editor_state["mode"] == "height_roi" else "height_roi"
        editor_state["dragging_idx"] = -1
    elif key == ord("d"):
        if len(editor_state["doorway_roi"]) < 3:
            editor_state["doorway_roi"] = [(0.30, 0.45), (0.70, 0.45), (0.82, 0.80), (0.18, 0.80)]
        editor_state["mode"] = "none" if editor_state["mode"] == "doorway_roi" else "doorway_roi"
        editor_state["dragging_idx"] = -1
    elif editor_state["mode"] != "none":
        if editor_state["mode"] == "height_roi":
            threshold = float(editor_state["age_adult_threshold_px"])
            if key in (ord("+"), ord("=")):
                editor_state["age_adult_threshold_px"] = min(5000.0, threshold + 5.0)
                print(f"[Config] Height age threshold: {editor_state['age_adult_threshold_px']:.1f}px")
                return end_all, should_break
            if key in (ord("-"), ord("_")):
                editor_state["age_adult_threshold_px"] = max(1.0, threshold - 5.0)
                print(f"[Config] Height age threshold: {editor_state['age_adult_threshold_px']:.1f}px")
                return end_all, should_break
        if key in (ord("s"), 13, ord("\r"), ord("v")):
            app._save_editor_config(
                roi_polygon=editor_state["roi"],
                height_roi_polygon=editor_state["height_roi"],
                age_adult_threshold_px=editor_state["age_adult_threshold_px"],
                doorway_roi_polygon=editor_state["doorway_roi"],
            )
            print(f"[Config] {editor_state['mode'].title()} saved for this source")
            editor_state["mode"] = "none"
        elif key == 27:
            editor_state["mode"] = "none"
            editor_state["roi"] = list(getattr(app.cfg, "roi_polygon", []))
            editor_state["height_roi"] = list(getattr(app.cfg, "height_roi_polygon", getattr(app.cfg, "roi_polygon", [])))
            editor_state["doorway_roi"] = list(getattr(app.cfg, "doorway_roi_polygon", []))
            editor_state["age_adult_threshold_px"] = float(getattr(app.cfg, "age_adult_threshold_px", 240.0))
            print("[Config] Edit cancelled.")
    return end_all, should_break
