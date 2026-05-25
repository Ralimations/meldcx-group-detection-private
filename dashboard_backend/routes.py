from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import cv2
from flask import Response, abort, jsonify, request, send_file

from .persistence import (
    create_sensor,
    delete_sensor,
    fetch_summaries,
    get_summary,
    get_sensor,
    list_presets,
    list_sensors,
    load_preset,
    save_preset,
    update_sensor_name,
)
from .config_store import sanitize_config_values
from .streaming import generate_mjpeg, generate_undistort_mjpeg, make_placeholder


def register_routes(app, engine: Any, param_meta: dict, defaults_dict: dict, project_root: Path, db_path: Path) -> None:
    @app.route("/video_feed")
    def video_feed():
        return Response(
            generate_mjpeg(engine),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/api/frame")
    def frame_snapshot():
        data, _version = engine.get_frame_snapshot_jpeg()
        if data is None:
            cfg = engine.get_config()
            width = max(1, int(cfg.get("preview_width", 960) or 960))
            height = max(1, int(cfg.get("preview_height", 540) or 540))
            text = "Engine warming up - please wait..." if engine.is_running else "Stream offline - press Start Engine"
            img = make_placeholder(width, height, text)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ok:
                return "Encoding error", 500
            data = buf.tobytes()
        return Response(data, mimetype="image/jpeg")

    @app.route("/api/workspace/snapshot", methods=["POST"])
    def workspace_snapshot():
        engine.capture_tuning_snapshot()
        return jsonify({"ok": True})

    @app.route("/api/workspace/frame")
    def workspace_frame():
        img = engine.get_workspace_snapshot()
        if img is None:
            cfg = engine.get_config()
            width = max(1, int(cfg.get("preview_width", 960) or 960))
            height = max(1, int(cfg.get("preview_height", 540) or 540))
            text = "No frame captured yet - open Live Monitoring first"
            img = make_placeholder(width, height, text)

        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            return "Encoding error", 500
        return Response(buf.tobytes(), mimetype="image/jpeg")

    @app.route("/api/undistort/snapshot", methods=["POST"])
    def undistort_snapshot():
        engine.capture_tuning_snapshot()
        return jsonify({"ok": True})

    @app.route("/api/undistort/comparison")
    def undistort_comparison():
        side = request.args.get("side", "both")
        img = engine.get_tuning_comparison(side=side)
        if img is None:
            img = engine.get_live_tuning_comparison(side=side)
        if img is None:
            img = make_placeholder(1280, 480)
            cv2.putText(img, "No snapshot captured", (400, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return "Encoding error", 500
        return Response(buf.tobytes(), mimetype="image/jpeg")

    @app.route("/api/undistort/frame")
    def undistort_frame():
        side = request.args.get("side", "both")
        img = engine.get_live_tuning_comparison(side=side)
        if img is None:
            cfg = engine.get_config()
            width = max(1, int(cfg.get("preview_width", 960) or 960))
            height = max(1, int(cfg.get("preview_height", 540) or 540))
            if side == "both":
                width = width * 2
            img = make_placeholder(width, height)

        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return "Encoding error", 500
        return Response(buf.tobytes(), mimetype="image/jpeg")

    @app.route("/api/undistort/video_feed")
    def undistort_video_feed():
        try:
            w = request.args.get("w", type=int)
            h = request.args.get("h", type=int)
            side = request.args.get("side", type=str, default="both")
        except Exception:
            w, h, side = None, None, "both"

        return Response(
            generate_undistort_mjpeg(engine, w, h, side),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/api/config", methods=["GET"])
    def get_config():
        return jsonify({"values": engine.get_config(), "meta": param_meta})

    @app.route("/api/config", methods=["POST"])
    def set_config():
        updates = sanitize_config_values(request.get_json(force=True) or {})
        engine.set_config(updates)
        return jsonify({"ok": True})

    @app.route("/api/engine/status", methods=["GET"])
    def get_status():
        return jsonify(engine.get_status_snapshot())

    @app.route("/api/engine/start", methods=["POST"])
    def start_engine():
        engine.start()
        return jsonify({"ok": True})

    @app.route("/api/engine/stop", methods=["POST"])
    def stop_engine():
        threading.Thread(target=engine.stop, daemon=True).start()
        return jsonify({"ok": True})

    @app.route("/api/engine/pause", methods=["POST"])
    def pause_engine():
        return jsonify({"paused": engine.pause()})

    @app.route("/api/playback", methods=["GET"])
    def get_playback():
        return jsonify(engine.get_playback_status())

    @app.route("/api/playback", methods=["POST"])
    def update_playback():
        body = request.get_json(force=True) or {}
        status = engine.control_playback(
            playing=body.get("playing"),
            seek_frame=body.get("seek_frame"),
            speed=body.get("speed"),
            loop=body.get("loop"),
            restart=bool(body.get("restart", False)),
        )
        return jsonify(status)

    @app.route("/api/engine/run_default", methods=["POST"])
    def run_default():
        engine.start()
        return jsonify({"ok": True})

    @app.route("/api/reset", methods=["POST"])
    def reset_engine():
        threading.Thread(target=engine.reset, daemon=True).start()
        return jsonify({"ok": True})

    @app.route("/api/logs", methods=["GET"])
    def get_logs():
        return jsonify({"logs": engine.get_logs()})

    @app.route("/api/logs/clear", methods=["POST"])
    def clear_logs():
        engine.clear_logs()
        return jsonify({"ok": True})

    @app.route("/api/summaries", methods=["GET"])
    def get_summaries():
        return jsonify(fetch_summaries(db_path))

    @app.route("/api/summaries/<int:summary_id>/artifact/<string:kind>", methods=["GET"])
    def get_summary_artifact(summary_id: int, kind: str):
        summary = get_summary(db_path, summary_id)
        if summary is None:
            abort(404)

        key_map = {
            "video": "annotated_video_path",
            "json": "detection_json_path",
        }
        artifact_key = key_map.get(kind)
        if artifact_key is None:
            abort(404)

        artifact_path = summary.get(artifact_key)
        if not artifact_path:
            abort(404)

        path = Path(artifact_path)
        if not path.is_absolute():
            path = project_root / artifact_path
        if not path.exists() or not path.is_file():
            abort(404)

        mimetype = "video/mp4" if kind == "video" else "application/json"
        return send_file(path, as_attachment=True, download_name=path.name, mimetype=mimetype)

    @app.route("/api/videos", methods=["GET"])
    def get_videos():
        media_dir = project_root / "media"
        videos = sorted(
            path.name
            for path in media_dir.iterdir()
            if path.suffix.lower() in (".mp4", ".mkv", ".avi", ".mov")
        ) if media_dir.is_dir() else []
        current_source = engine.get_config().get("source", "")
        is_network = current_source.startswith(("rtsp://", "http://", "https://"))
        current = Path(current_source).name if (current_source and not is_network) else ""
        return jsonify({"videos": videos, "current": current})

    @app.route("/api/sensors", methods=["GET"])
    def get_sensors():
        return jsonify({"sensors": list_sensors(db_path), "active_id": engine.get_active_sensor_id()})

    @app.route("/api/sensors", methods=["POST"])
    def create_sensor_route():
        body = request.get_json(force=True) or {}
        name = body.get("name", "New Sensor")
        initial_config = dict(defaults_dict)
        if "config" in body:
            initial_config.update(sanitize_config_values(body["config"]))
        sensor_id = create_sensor(db_path, name, initial_config)
        engine._log(f"[Sensor] Created '{name}' (ID: {sensor_id})")
        return jsonify({"ok": True, "id": sensor_id})

    @app.route("/api/sensors/<int:sensor_id>", methods=["DELETE"])
    def delete_sensor_route(sensor_id: int):
        delete_sensor(db_path, sensor_id)
        if engine._active_sensor_id == sensor_id:
            engine._active_sensor_id = None
            engine.stop()
        engine._log(f"[Sensor] Deleted sensor {sensor_id}")
        return jsonify({"ok": True})

    @app.route("/api/sensors/<int:sensor_id>", methods=["PATCH"])
    def update_sensor_route(sensor_id: int):
        body = request.get_json(force=True) or {}
        name = str(body.get("name", "")).strip()
        if not name:
            return jsonify({"error": "name required"}), 400

        loaded = get_sensor(db_path, sensor_id)
        if loaded is None:
            return jsonify({"error": "sensor not found"}), 404

        update_sensor_name(db_path, sensor_id, name)
        engine._log(f"[Sensor] Renamed sensor {sensor_id} to '{name}'")
        return jsonify({"ok": True})

    @app.route("/api/sensors/load", methods=["POST"])
    def load_sensor_route():
        body = request.get_json(force=True) or {}
        sensor_id = body.get("id")
        if not sensor_id:
            return jsonify({"error": "id required"}), 400

        loaded = get_sensor(db_path, sensor_id)
        if loaded is None:
            return jsonify({"error": "sensor not found"}), 404

        name, values = loaded
        values = sanitize_config_values(values)
        with engine._lock:
            engine._config_values = values
            engine._config_dirty = True
            engine._active_sensor_id = sensor_id

        engine._log(f"[Sensor] Loaded and activated '{name}' (ID: {sensor_id})")
        if engine.is_running:
            threading.Thread(target=engine.reset, daemon=True).start()
        return jsonify({"ok": True, "values": values})

    @app.route("/api/sensors/deactivate", methods=["POST"])
    def deactivate_sensor_route():
        engine._active_sensor_id = None
        engine.stop()
        engine._log("[Sensor] Deactivated current sensor. Engine stopped.")
        return jsonify({"ok": True})

    @app.route("/api/switch", methods=["POST"])
    def switch_input():
        body = request.get_json(force=True) or {}
        mode = body.get("mode", "source")
        if mode == "camera":
            engine.switch_input("camera", body.get("camera", 0))
        else:
            video = body.get("video", "")
            is_url = video.startswith(("rtsp://", "http://", "https://"))
            data = f"media/{video}" if video and not is_url and not video.startswith("media/") else video
            engine.switch_input("source", data)
        return jsonify({"ok": True})

    @app.route("/api/defaults", methods=["POST"])
    def reset_defaults():
        return jsonify({"values": engine.reset_to_defaults()})

    @app.route("/api/presets", methods=["GET"])
    def get_presets():
        return jsonify({"presets": list_presets(db_path)})

    @app.route("/api/save", methods=["POST"])
    def save_preset_route():
        body = request.get_json(force=True) or {}
        name = body.get("name", "").strip()
        if not name:
            return jsonify({"error": "name required"}), 400
        save_preset(db_path, name, engine.get_config())
        engine._log(f"[Preset] Saved: {name}")
        return jsonify({"ok": True})

    @app.route("/api/load", methods=["POST"])
    def load_preset_route():
        body = request.get_json(force=True) or {}
        name = body.get("name", "").strip()
        values = load_preset(db_path, name)
        if values is None:
            return jsonify({"error": "preset not found"}), 404
        engine.set_config(values)
        engine._log(f"[Preset] Loaded: {name}")
        return jsonify({"values": values})
