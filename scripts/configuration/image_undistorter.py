#!/usr/bin/env python3
"""Config-driven lens undistortion helper and configuration tuning entrypoint."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "settings.db"

from config import OPENVINO_DEFAULTS, OpenVinoDefaults


def _file_source_key(source_str: str) -> str:
    path = Path(source_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    normalized = Path(os.path.normpath(str(path)))
    project_root_norm = Path(os.path.normpath(str(PROJECT_ROOT)))
    try:
        relative = normalized.relative_to(project_root_norm)
    except ValueError:
        return f"file:{normalized}"
    return f"file:{relative.as_posix()}"


def _source_key_for_input(source: str, *, mode: str = "source", camera_index: int = 0) -> str:
    source_str = str(source).strip().strip("'\"")
    if mode == "camera":
        return f"camera:{int(camera_index)}"
    if source_str.lower().startswith(("rtsp://", "http://", "https://")):
        return f"stream:{source_str}"
    return _file_source_key(source_str)


def _init_undistort_db() -> None:
    con = sqlite3.connect(str(DB_PATH))
    con.execute(
        "CREATE TABLE IF NOT EXISTS source_undistort_settings ("
        "source_key TEXT PRIMARY KEY, "
        "undistort_enable INTEGER NOT NULL, "
        "undistort_model TEXT NOT NULL, "
        "undistort_coeffs_json TEXT NOT NULL, "
        "undistort_fx REAL NOT NULL, "
        "undistort_fy REAL NOT NULL, "
        "undistort_balance REAL NOT NULL, "
        "undistort_alpha REAL NOT NULL, "
        "updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
    )
    con.commit()
    con.close()


def save_source_undistort_settings(
    *,
    source: str,
    enable: bool,
    model: str,
    coeffs: tuple[float, ...],
    fx: float,
    fy: float,
    balance: float,
    alpha: float = 0.0,
    mode: str = "source",
    camera_index: int = 0,
) -> str:
    _init_undistort_db()
    source_key = _source_key_for_input(source, mode=mode, camera_index=camera_index)
    con = sqlite3.connect(str(DB_PATH))
    con.execute(
        "INSERT INTO source_undistort_settings ("
        "source_key, undistort_enable, undistort_model, undistort_coeffs_json, "
        "undistort_fx, undistort_fy, undistort_balance, undistort_alpha, updated_at"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP) "
        "ON CONFLICT(source_key) DO UPDATE SET "
        "undistort_enable = excluded.undistort_enable, "
        "undistort_model = excluded.undistort_model, "
        "undistort_coeffs_json = excluded.undistort_coeffs_json, "
        "undistort_fx = excluded.undistort_fx, "
        "undistort_fy = excluded.undistort_fy, "
        "undistort_balance = excluded.undistort_balance, "
        "undistort_alpha = excluded.undistort_alpha, "
        "updated_at = CURRENT_TIMESTAMP",
        (
            source_key,
            1 if enable else 0,
            str(model),
            json.dumps([float(v) for v in coeffs], ensure_ascii=True),
            float(fx),
            float(fy),
            float(balance),
            float(alpha),
        ),
    )
    con.commit()
    con.close()
    return source_key


def apply_source_undistort_overrides(
    cfg: OpenVinoDefaults,
    *,
    source: str | None = None,
    mode: str | None = None,
    camera_index: int | None = None,
) -> OpenVinoDefaults:
    _init_undistort_db()
    source_key = _source_key_for_input(
        source if source is not None else getattr(cfg, "source", ""),
        mode=mode if mode is not None else str(getattr(cfg, "input_mode", "source")).strip().lower(),
        camera_index=int(camera_index if camera_index is not None else getattr(cfg, "camera_index", 0)),
    )
    con = sqlite3.connect(str(DB_PATH))
    row = con.execute(
        "SELECT undistort_enable, undistort_model, undistort_coeffs_json, undistort_fx, "
        "undistort_fy, undistort_balance, undistort_alpha "
        "FROM source_undistort_settings WHERE source_key = ?",
        (source_key,),
    ).fetchone()
    con.close()
    if row is None:
        return cfg
    return replace(
        cfg,
        undistort_enable=bool(row[0]),
        undistort_model=str(row[1]),
        undistort_coeffs=tuple(float(v) for v in json.loads(row[2])),
        undistort_fx=float(row[3]),
        undistort_fy=float(row[4]),
        undistort_balance=float(row[5]),
        undistort_alpha=float(row[6]),
    )


def copy_source_undistort_settings(from_source_key: str, to_source_key: str) -> None:
    _init_undistort_db()
    con = sqlite3.connect(str(DB_PATH))
    row = con.execute(
        "SELECT undistort_enable, undistort_model, undistort_coeffs_json, undistort_fx, "
        "undistort_fy, undistort_balance, undistort_alpha "
        "FROM source_undistort_settings WHERE source_key = ?",
        (from_source_key,),
    ).fetchone()
    if row is None:
        con.close()
        raise ValueError(f"Missing undistortion config for {from_source_key}")
    con.execute(
        "INSERT INTO source_undistort_settings ("
        "source_key, undistort_enable, undistort_model, undistort_coeffs_json, "
        "undistort_fx, undistort_fy, undistort_balance, undistort_alpha, updated_at"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP) "
        "ON CONFLICT(source_key) DO UPDATE SET "
        "undistort_enable = excluded.undistort_enable, "
        "undistort_model = excluded.undistort_model, "
        "undistort_coeffs_json = excluded.undistort_coeffs_json, "
        "undistort_fx = excluded.undistort_fx, "
        "undistort_fy = excluded.undistort_fy, "
        "undistort_balance = excluded.undistort_balance, "
        "undistort_alpha = excluded.undistort_alpha, "
        "updated_at = CURRENT_TIMESTAMP",
        (
            to_source_key,
            int(row[0]),
            str(row[1]),
            str(row[2]),
            float(row[3]),
            float(row[4]),
            float(row[5]),
            float(row[6]),
        ),
    )
    con.commit()
    con.close()

_init_undistort_db()


@dataclass
class _MapCacheEntry:
    map1: np.ndarray
    map2: np.ndarray


class ImageUndistorter:
    """Apply fisheye or standard lens undistortion using config.py values."""

    def __init__(self, cfg: OpenVinoDefaults) -> None:
        self.cfg = cfg
        self.enabled = bool(getattr(cfg, "undistort_enable", False))
        self.model = str(getattr(cfg, "undistort_model", "fisheye")).strip().lower()
        if self.model not in {"fisheye", "standard"}:
            self.model = "fisheye"

        coeffs = tuple(float(v) for v in getattr(cfg, "undistort_coeffs", (0.0, 0.0, 0.0, 0.0)))
        if self.model == "fisheye":
            coeffs = coeffs[:4] if len(coeffs) >= 4 else coeffs + (0.0,) * (4 - len(coeffs))
        self.coeffs = coeffs
        self.fx_ref = float(getattr(cfg, "undistort_fx", 195.0))
        self.fy_ref = float(getattr(cfg, "undistort_fy", 195.0))
        self.ref_width = max(1, int(getattr(cfg, "undistort_ref_width", 1080)))
        self.ref_height = max(1, int(getattr(cfg, "undistort_ref_height", 1080)))
        self.balance = float(getattr(cfg, "undistort_balance", 0.0))
        self.balance = max(0.0, min(1.0, self.balance))
        self.alpha = float(getattr(cfg, "undistort_alpha", 0.0))
        self.alpha = max(0.0, min(1.0, self.alpha))
        self._cache: dict[tuple[int, int], _MapCacheEntry] = {}

        if self.enabled:
            print(
                f"[Undistort] Enabled ({self.model}) "
                f"fx={self.fx_ref:.2f}, fy={self.fy_ref:.2f}, "
                f"ref={self.ref_width}x{self.ref_height}, coeffs={self.coeffs}"
            )

    def is_effectively_neutral(self) -> bool:
        """Return True when the current settings should behave like a passthrough."""
        return all(abs(v) < 1e-8 for v in self.coeffs)

    def _camera_matrix(self, frame_shape: tuple[int, int]) -> np.ndarray:
        frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
        fx = (self.fx_ref * frame_w) / float(self.ref_width)
        fy = (self.fy_ref * frame_h) / float(self.ref_height)
        cx = frame_w / 2.0
        cy = frame_h / 2.0
        return np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def _maps_for_shape(self, frame_shape: tuple[int, int]) -> _MapCacheEntry:
        frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
        cache_key = (frame_w, frame_h)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        k_mat = self._camera_matrix(frame_shape)
        if self.model == "fisheye":
            d_vec = np.array(self.coeffs[:4], dtype=np.float32).reshape(4, 1)
            new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                k_mat,
                d_vec,
                (frame_w, frame_h),
                np.eye(3, dtype=np.float32),
                balance=self.balance,
                new_size=(frame_w, frame_h),
            )
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                k_mat,
                d_vec,
                np.eye(3, dtype=np.float32),
                new_k,
                (frame_w, frame_h),
                cv2.CV_16SC2,
            )
        else:
            d_vec = np.array(self.coeffs, dtype=np.float32).reshape(-1, 1)
            new_k, _ = cv2.getOptimalNewCameraMatrix(
                k_mat,
                d_vec,
                (frame_w, frame_h),
                self.alpha,
                (frame_w, frame_h),
            )
            map1, map2 = cv2.initUndistortRectifyMap(
                k_mat,
                d_vec,
                None,
                new_k,
                (frame_w, frame_h),
                cv2.CV_16SC2,
            )

        entry = _MapCacheEntry(map1=map1, map2=map2)
        self._cache[cache_key] = entry
        return entry

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Return the undistorted frame, or the input frame if disabled."""
        if not self.enabled:
            return frame
        if frame is None or frame.size == 0:
            return frame
        if self.is_effectively_neutral():
            return frame

        frame_h, frame_w = frame.shape[:2]
        maps = self._maps_for_shape((frame_h, frame_w))
        return cv2.remap(
            frame,
            maps.map1,
            maps.map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )


def _nothing(_: int) -> None:
    pass


def _slider_to_coeff(value: int) -> float:
    return (value - 2000) / 1000.0


def _coeff_to_slider(value: float) -> int:
    return int(round(value * 1000.0)) + 2000


def _apply_saved_reset(window_name: str, cfg: OpenVinoDefaults, coeffs_cfg: tuple[float, float, float, float]) -> None:
    cv2.setTrackbarPos("FX", window_name, max(1, int(getattr(cfg, "undistort_fx", 195.0))))
    cv2.setTrackbarPos("FY", window_name, max(1, int(getattr(cfg, "undistort_fy", 195.0))))
    cv2.setTrackbarPos("K1 x1000", window_name, _coeff_to_slider(coeffs_cfg[0]))
    cv2.setTrackbarPos("K2 x1000", window_name, _coeff_to_slider(coeffs_cfg[1]))
    cv2.setTrackbarPos("K3 x1000", window_name, _coeff_to_slider(coeffs_cfg[2]))
    cv2.setTrackbarPos("K4 x1000", window_name, _coeff_to_slider(coeffs_cfg[3]))
    cv2.setTrackbarPos(
        "Balance x100",
        window_name,
        int(round(float(getattr(cfg, "undistort_balance", 0.0)) * 100.0)),
    )


def _apply_neutral_reset(window_name: str, cfg: OpenVinoDefaults) -> None:
    cv2.setTrackbarPos("FX", window_name, max(1, int(getattr(cfg, "undistort_fx", 195.0))))
    cv2.setTrackbarPos("FY", window_name, max(1, int(getattr(cfg, "undistort_fy", 195.0))))
    cv2.setTrackbarPos("K1 x1000", window_name, _coeff_to_slider(0.0))
    cv2.setTrackbarPos("K2 x1000", window_name, _coeff_to_slider(0.0))
    cv2.setTrackbarPos("K3 x1000", window_name, _coeff_to_slider(0.0))
    cv2.setTrackbarPos("K4 x1000", window_name, _coeff_to_slider(0.0))
    cv2.setTrackbarPos("Balance x100", window_name, 0)


def _stack_preview(left: np.ndarray, right: np.ndarray, max_width: int = 1600) -> np.ndarray:
    preview = cv2.hconcat([left, right])
    h, w = preview.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        preview = cv2.resize(preview, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return preview


def _set_window_fullscreen(window_name: str) -> None:
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 0)
    except cv2.error:
        pass
    cv2.resizeWindow(window_name, 1920, 1080)

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Live undistortion tuning tool")
    parser.add_argument("--source", default=OPENVINO_DEFAULTS.source, help="Video file, image, or stream")
    parser.add_argument(
        "--model",
        default=str(getattr(OPENVINO_DEFAULTS, "undistort_model", "fisheye")),
        choices=("fisheye", "standard"),
        help="Undistortion model to preview",
    )
    args = parser.parse_args(argv)

    cfg_source = apply_source_undistort_overrides(OPENVINO_DEFAULTS, source=args.source, mode="source")

    window_name = "Live Undistortion Tuning"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    _set_window_fullscreen(window_name)

    coeffs_cfg = tuple(float(v) for v in getattr(cfg_source, "undistort_coeffs", (0.0, 0.0, 0.0, 0.0)))
    coeffs_cfg = coeffs_cfg[:4] if len(coeffs_cfg) >= 4 else coeffs_cfg + (0.0,) * (4 - len(coeffs_cfg))

    cv2.createTrackbar("FX", window_name, max(1, int(getattr(cfg_source, "undistort_fx", 195.0))), 4000, _nothing)
    cv2.createTrackbar("FY", window_name, max(1, int(getattr(cfg_source, "undistort_fy", 195.0))), 4000, _nothing)
    cv2.createTrackbar("K1 x1000", window_name, _coeff_to_slider(coeffs_cfg[0]), 4000, _nothing)
    cv2.createTrackbar("K2 x1000", window_name, _coeff_to_slider(coeffs_cfg[1]), 4000, _nothing)
    cv2.createTrackbar("K3 x1000", window_name, _coeff_to_slider(coeffs_cfg[2]), 4000, _nothing)
    cv2.createTrackbar("K4 x1000", window_name, _coeff_to_slider(coeffs_cfg[3]), 4000, _nothing)
    cv2.createTrackbar(
        "Balance x100",
        window_name,
        int(round(float(getattr(cfg_source, "undistort_balance", 0.0)) * 100.0)),
        100,
        _nothing,
    )

    source = str(args.source)
    cap = cv2.VideoCapture(source)
    single_image = None
    if not cap.isOpened():
        single_image = cv2.imread(source)
        if single_image is None:
            raise SystemExit(f"Failed to open source: {source}")

    cfg_base = replace(cfg_source, undistort_enable=True, undistort_model=args.model)
    last_key = None
    undistorter = None

    print("Live undistortion tuning.")
    print("Press 's' to save, 'q' or ESC to quit without saving, 'r' to reset to saved config, and '0' for a neutral reset.")

    while True:
        if single_image is None:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
        else:
            frame = single_image.copy()

        fx = float(max(1, cv2.getTrackbarPos("FX", window_name)))
        fy = float(max(1, cv2.getTrackbarPos("FY", window_name)))
        coeffs = (
            _slider_to_coeff(cv2.getTrackbarPos("K1 x1000", window_name)),
            _slider_to_coeff(cv2.getTrackbarPos("K2 x1000", window_name)),
            _slider_to_coeff(cv2.getTrackbarPos("K3 x1000", window_name)),
            _slider_to_coeff(cv2.getTrackbarPos("K4 x1000", window_name)),
        )
        balance = cv2.getTrackbarPos("Balance x100", window_name) / 100.0

        current_key = (args.model, fx, fy, coeffs, balance, frame.shape[1], frame.shape[0])
        if current_key != last_key:
            cfg_current = replace(
                cfg_base,
                undistort_enable=True,
                undistort_model=args.model,
                undistort_fx=fx,
                undistort_fy=fy,
                undistort_coeffs=coeffs,
                undistort_balance=balance,
            )
            undistorter = ImageUndistorter(cfg_current)
            last_key = current_key

        corrected = undistorter.apply(frame) if undistorter is not None else frame.copy()
        preview = _stack_preview(frame, corrected)

        cv2.putText(preview, "Original", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        half_x = preview.shape[1] // 2
        cv2.putText(preview, "Corrected", (half_x + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        info = (
            f"{args.model} | fx={fx:.1f} fy={fy:.1f} | "
            f"k=({coeffs[0]:.3f}, {coeffs[1]:.3f}, {coeffs[2]:.3f}, {coeffs[3]:.3f}) | "
            f"balance={balance:.2f}"
        )
        cv2.putText(preview, info, (20, preview.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(window_name, preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            source_key = save_source_undistort_settings(
                source=source,
                enable=True,
                model=args.model,
                coeffs=coeffs,
                fx=fx,
                fy=fy,
                balance=balance,
                alpha=float(getattr(cfg_source, "undistort_alpha", 0.0)),
            )
            print(f"Saved undistortion values for {source_key} to settings.db")
            break
        if key in (27, ord("q")):
            print("Exited without saving undistortion values.")
            break
        if key == ord("r"):
            _apply_saved_reset(window_name, cfg_source, coeffs_cfg)
        if key == ord("0"):
            _apply_neutral_reset(window_name, cfg_source)

    if single_image is None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
