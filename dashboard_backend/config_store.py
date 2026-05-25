from __future__ import annotations

import json
from dataclasses import fields

from config import OPENVINO_DEFAULTS, OpenVinoDefaults


def cfg_to_dict(cfg: OpenVinoDefaults) -> dict:
    """Convert OpenVinoDefaults dataclass to a JSON-serialisable dict."""
    data: dict = {}
    for field in fields(cfg):
        value = getattr(cfg, field.name)
        if field.name == "undistort_coeffs":
            data[field.name] = ", ".join(map(str, value))
            continue
        if field.name in {"roi_polygon", "height_roi_polygon", "doorway_roi_polygon"}:
            data[field.name] = [
                [float(point[0]), float(point[1])]
                for point in value
                if isinstance(point, (list, tuple)) and len(point) == 2
            ]
            continue
        if isinstance(value, (list, tuple)):
            continue
        data[field.name] = value
    return data


def dict_to_cfg(base: OpenVinoDefaults, overrides: dict) -> OpenVinoDefaults:
    """Apply dict overrides onto a copy of base config."""
    current = cfg_to_dict(base)
    current.update(overrides)
    field_map = {field.name: field for field in fields(base)}
    kwargs = {}
    for name, value in current.items():
        if name not in field_map:
            continue
        field_type = field_map[name].type
        try:
            if name == "undistort_coeffs":
                if isinstance(value, str):
                    parts = [part.strip() for part in value.split(",")]
                    kwargs[name] = tuple(float(part) for part in parts if part)
                else:
                    kwargs[name] = tuple(value)
                continue

            if name in {"roi_polygon", "height_roi_polygon", "doorway_roi_polygon"}:
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        value = []
                if isinstance(value, (list, tuple)):
                    kwargs[name] = tuple(
                        tuple(float(coord) for coord in point)
                        for point in value
                        if isinstance(point, (list, tuple)) and len(point) == 2
                    )
                continue

            if field_type in ("bool", "Optional[bool]") or field_type is bool:
                kwargs[name] = bool(value)
            elif field_type in ("int", "Optional[int]") or field_type is int:
                kwargs[name] = int(value)
            elif field_type in ("float", "Optional[float]") or field_type is float:
                kwargs[name] = float(value)
            else:
                kwargs[name] = "source" if name == "input_mode" and value in ("network", "file") else value
        except (TypeError, ValueError):
            kwargs[name] = value

    for field in fields(base):
        if field.name not in kwargs:
            kwargs[field.name] = getattr(base, field.name)

    return OpenVinoDefaults(**kwargs)


DEFAULTS_DICT = cfg_to_dict(OPENVINO_DEFAULTS)
ALLOWED_CONFIG_KEYS = frozenset(DEFAULTS_DICT.keys())


def sanitize_config_values(values: dict | None) -> dict:
    """Drop stale dashboard keys that no longer exist in OpenVinoDefaults."""
    if not isinstance(values, dict):
        return {}
    return {
        key: value
        for key, value in values.items()
        if key in ALLOWED_CONFIG_KEYS
    }
