#!/usr/bin/env python3
"""Lightweight smoke checks for the dashboard backend bootstrap."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    import dashboard_server  # noqa: WPS433

    app = dashboard_server.app_flask
    engine = dashboard_server.engine

    routes = {rule.rule for rule in app.url_map.iter_rules()}
    required_routes = {
        "/video_feed",
        "/api/config",
        "/api/engine/status",
        "/api/engine/start",
        "/api/engine/stop",
        "/api/logs",
        "/api/videos",
        "/api/sensors",
        "/api/sensors/load",
        "/api/sensors/deactivate",
        "/api/presets",
    }
    missing_routes = sorted(required_routes - routes)
    if missing_routes:
        print("Missing routes:")
        for route in missing_routes:
            print(f"  - {route}")
        return 1

    status = engine.get_status_snapshot()
    if not {"running", "paused", "people", "groups", "fps"} <= status.keys():
        print("Engine status snapshot is missing expected keys.")
        return 1

    print("Dashboard backend smoke test passed.")
    print(f"Registered routes: {len(routes)}")
    print(f"Active sensor: {engine.get_active_sensor_id()}")
    print(f"Initial status: {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
