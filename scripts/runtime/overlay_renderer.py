#!/usr/bin/env python3
"""Overlay and HUD drawing helpers for the runtime preview."""

from __future__ import annotations

import cv2
import numpy as np


def draw_runtime_overlays(
    display_frame,
    *,
    cfg,
    editor_state: dict,
    current_roi,
    current_height_roi,
    current_doorway_roi,
    group_detector,
    room_trip_statuses,
    carry_alert_statuses,
    current_view: str,
    detection_count: int,
    group_count: int,
    proc_fps: float,
    source_fps: float,
    is_last_file_frame: bool,
    norm_to_px,
):
    if cfg.show_roi_mask and current_roi and len(current_roi) >= 3:
        h0, w0 = display_frame.shape[:2]
        pts = np.array([(int(p[0] * w0), int(p[1] * h0)) for p in current_roi], np.int32)
        mask = np.zeros_like(display_frame)
        cv2.fillPoly(mask, [pts], (0, 255, 0))
        cv2.polylines(display_frame, [pts], True, (0, 255, 0), 1)
        alpha = getattr(cfg, "roi_transparency", 0.15)
        cv2.addWeighted(mask, alpha, display_frame, 1.0, 0, display_frame)
        if editor_state["mode"] == "roi":
            for i, p_norm in enumerate(current_roi):
                pt = norm_to_px(p_norm, w0, h0)
                color = (0, 255, 255) if i == editor_state["dragging_idx"] else (0, 0, 255)
                dist = ((pt[0] - editor_state["mouse_pos"][0]) ** 2 + (pt[1] - editor_state["mouse_pos"][1]) ** 2) ** 0.5
                radius = 3 if (dist < 40 or i == editor_state["dragging_idx"]) else 5
                cv2.circle(display_frame, pt, radius, color, -1)
                cv2.circle(display_frame, pt, radius, (255, 255, 255), 1)

    if getattr(cfg, "show_height_roi_mask", True) and current_height_roi and len(current_height_roi) >= 3:
        h0, w0 = display_frame.shape[:2]
        hpts = np.array([(int(p[0] * w0), int(p[1] * h0)) for p in current_height_roi], np.int32)
        height_mask = np.zeros_like(display_frame)
        cv2.fillPoly(height_mask, [hpts], (255, 165, 0))
        cv2.polylines(display_frame, [hpts], True, (255, 165, 0), 1)
        alpha = getattr(cfg, "height_roi_transparency", 0.03)
        cv2.addWeighted(height_mask, alpha, display_frame, 1.0, 0, display_frame)
        if editor_state["mode"] == "height_roi":
            for i, p_norm in enumerate(current_height_roi):
                pt = norm_to_px(p_norm, w0, h0)
                color = (0, 255, 255) if i == editor_state["dragging_idx"] else (255, 165, 0)
                dist = ((pt[0] - editor_state["mouse_pos"][0]) ** 2 + (pt[1] - editor_state["mouse_pos"][1]) ** 2) ** 0.5
                radius = 3 if (dist < 10 or i == editor_state["dragging_idx"]) else 5
                cv2.circle(display_frame, pt, radius, color, -1)
                cv2.circle(display_frame, pt, radius, (255, 255, 255), 1)

    if current_doorway_roi and len(current_doorway_roi) >= 3:
        h0, w0 = display_frame.shape[:2]
        dpts = np.array([(int(p[0] * w0), int(p[1] * h0)) for p in current_doorway_roi], np.int32)
        doorway_mask = np.zeros_like(display_frame)
        cv2.fillPoly(doorway_mask, [dpts], (255, 255, 0))
        cv2.polylines(display_frame, [dpts], True, (255, 255, 0), 1)
        cv2.addWeighted(doorway_mask, 0.08, display_frame, 1.0, 0, display_frame)
        if editor_state["mode"] == "doorway_roi":
            for i, p_norm in enumerate(current_doorway_roi):
                pt = norm_to_px(p_norm, w0, h0)
                color = (0, 255, 255) if i == editor_state["dragging_idx"] else (255, 255, 0)
                cv2.circle(display_frame, pt, 5, color, -1)
                cv2.circle(display_frame, pt, 5, (255, 255, 255), 1)

    if editor_state["mode"] != "none":
        ed_h = display_frame.shape[0]
        ed_fs = max(0.40, ed_h / 1400)
        ed_fs_sm = max(0.32, ed_h / 1800)
        ed_y1 = int(55 * (ed_fs / 0.5))
        ed_y2 = ed_y1 + int(25 * (ed_fs / 0.5))
        mode_labels = {
            "roi": "ROI",
            "height_roi": "Height ROI",
            "doorway_roi": "Doorway ROI",
        }
        mode_text = mode_labels.get(editor_state["mode"], "Editor")
        line1 = f"EDITING {mode_text.upper()} (PAUSED) - Drag points. 's'/Enter=Save"
        if editor_state["mode"] == "height_roi":
            line2 = (
                "Drag polygon points. +/- adjusts height age threshold: "
                f"{float(editor_state['age_adult_threshold_px']):.1f}px"
            )
        elif editor_state["mode"] == "doorway_roi":
            line2 = "Drag doorway ROI points."
        else:
            line2 = "Drag polygon points."

        def draw_status_text(frame, text, x, y, scale, color, thick):
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
            cv2.rectangle(frame, (x - 5, y - th - 5), (x + tw + 5, y + baseline + 5), (0, 0, 0), -1)
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

        draw_status_text(display_frame, line1, 12, ed_y1, ed_fs, (0, 0, 255), 1)
        draw_status_text(display_frame, line2, 12, ed_y2, ed_fs_sm, (0, 180, 255), 1)

    h_scale = display_frame.shape[0]
    fs = 0.3 if h_scale < 400 else 0.5 if h_scale < 720 else 0.6
    th = 1
    if getattr(cfg, "show_ui_overlay", True) or editor_state["mode"] != "none":
        total_groups = len(group_detector.seen_frozen_groups) if group_detector else 0
        source_fps_text = f"{source_fps:.1f}" if source_fps > 0 else "n/a"
        cv2.putText(
            display_frame,
            (
                f"Detections: {detection_count}  Groups: {group_count} (Total: {total_groups})  "
                f"Proc FPS: {proc_fps:.1f}  Source FPS: {source_fps_text}  View: {current_view}"
            ),
            (12, int(35 * (fs / 0.8))),
            cv2.FONT_HERSHEY_SIMPLEX,
            fs,
            (0, 255, 255),
            th,
        )
        if is_last_file_frame:
            cv2.putText(
                display_frame,
                "LAST FRAME - stopping",
                (12, int(60 * (fs / 0.8))),
                cv2.FONT_HERSHEY_SIMPLEX,
                fs,
                (0, 200, 255),
                th,
            )

    bottom_statuses = []
    if getattr(cfg, "show_carry_status", False) and carry_alert_statuses:
        for status in carry_alert_statuses:
            bottom_statuses.append(
                {
                    "text": str(status["text"]),
                    "color": tuple(status.get("color", (0, 140, 255))),
                }
            )

    if getattr(cfg, "show_doorway_status", False) and room_trip_statuses:
        threshold = float(getattr(cfg, "room_presence_alert_seconds", 5.0))
        for status in reversed(room_trip_statuses):
            seconds_in_room = float(status["seconds_in_room"])
            bottom_statuses.append(
                {
                    "text": f"Group {int(status['group_id'])}  {seconds_in_room:.1f}s",
                    "color": (0, 255, 0) if seconds_in_room < threshold else (0, 0, 255),
                }
            )

    if bottom_statuses:
        h0 = display_frame.shape[0]
        fs_room = 0.28 if h0 < 720 else 0.38
        y = h0 - 14
        for status in bottom_statuses:
            color = tuple(int(v) for v in status["color"])
            text = str(status["text"])
            cv2.putText(
                display_frame,
                text,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                fs_room,
                color,
                1,
                cv2.LINE_AA,
            )
            y -= max(14, int(18 * (fs_room / 0.38)))

    return display_frame
