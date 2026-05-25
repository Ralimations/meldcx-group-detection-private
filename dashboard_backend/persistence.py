from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .config_store import sanitize_config_values


def init_db(db_path: Path) -> None:
    con = sqlite3.connect(str(db_path))
    con.execute("CREATE TABLE IF NOT EXISTS presets (name TEXT PRIMARY KEY, config_json TEXT)")
    con.execute(
        "CREATE TABLE IF NOT EXISTS sensors ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "name TEXT NOT NULL, "
        "config_json TEXT NOT NULL"
        ")"
    )
    con.execute(
        "CREATE TABLE IF NOT EXISTS test_summaries ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "timestamp TEXT, "
        "source TEXT, "
        "max_people INTEGER, "
        "max_groups INTEGER, "
        "avg_fps REAL, "
        "duration REAL, "
        "annotated_video_path TEXT, "
        "detection_json_path TEXT)"
    )
    existing_columns = {
        row[1]
        for row in con.execute("PRAGMA table_info(test_summaries)").fetchall()
    }
    if "annotated_video_path" not in existing_columns:
        con.execute("ALTER TABLE test_summaries ADD COLUMN annotated_video_path TEXT")
    if "detection_json_path" not in existing_columns:
        con.execute("ALTER TABLE test_summaries ADD COLUMN detection_json_path TEXT")
    con.commit()
    con.close()


def fetch_summaries(db_path: Path) -> list[dict]:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT * FROM test_summaries ORDER BY id DESC").fetchall()
    con.close()
    return [dict(row) for row in rows]


def get_summary(db_path: Path, summary_id: int) -> dict | None:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    row = con.execute("SELECT * FROM test_summaries WHERE id = ?", (summary_id,)).fetchone()
    con.close()
    return dict(row) if row else None


def save_summary(
    db_path: Path,
    *,
    timestamp: str,
    source: str,
    max_people: int,
    max_groups: int,
    avg_fps: float,
    duration: float,
    annotated_video_path: str | None,
    detection_json_path: str | None,
) -> None:
    con = sqlite3.connect(str(db_path))
    con.execute(
        "INSERT INTO test_summaries "
        "(timestamp, source, max_people, max_groups, avg_fps, duration, annotated_video_path, detection_json_path) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            timestamp,
            source,
            max_people,
            max_groups,
            avg_fps,
            duration,
            annotated_video_path,
            detection_json_path,
        ),
    )
    con.commit()
    con.close()


def list_sensors(db_path: Path) -> list[dict]:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT id, name, config_json FROM sensors ORDER BY id DESC").fetchall()
    con.close()
    sensors = []
    for row in rows:
        item = dict(row)
        item["config"] = sanitize_config_values(json.loads(item.pop("config_json")))
        sensors.append(item)
    return sensors


def create_sensor(db_path: Path, name: str, config: dict) -> int:
    clean_config = sanitize_config_values(config)
    con = sqlite3.connect(str(db_path))
    cursor = con.cursor()
    cursor.execute(
        "INSERT INTO sensors (name, config_json) VALUES (?, ?)",
        (name, json.dumps(clean_config)),
    )
    sensor_id = cursor.lastrowid
    con.commit()
    con.close()
    return sensor_id


def delete_sensor(db_path: Path, sensor_id: int) -> None:
    con = sqlite3.connect(str(db_path))
    con.execute("DELETE FROM sensors WHERE id = ?", (sensor_id,))
    con.commit()
    con.close()


def update_sensor_name(db_path: Path, sensor_id: int, name: str) -> None:
    con = sqlite3.connect(str(db_path))
    con.execute("UPDATE sensors SET name = ? WHERE id = ?", (name, sensor_id))
    con.commit()
    con.close()


def get_sensor(db_path: Path, sensor_id: int) -> tuple[str, dict] | None:
    con = sqlite3.connect(str(db_path))
    row = con.execute("SELECT name, config_json FROM sensors WHERE id = ?", (sensor_id,)).fetchone()
    con.close()
    if not row:
        return None
    name, config_json = row
    return name, sanitize_config_values(json.loads(config_json))


def list_presets(db_path: Path) -> list[str]:
    con = sqlite3.connect(str(db_path))
    rows = con.execute("SELECT name FROM presets ORDER BY name").fetchall()
    con.close()
    return [row[0] for row in rows]


def save_preset(db_path: Path, name: str, values: dict) -> None:
    clean_values = sanitize_config_values(values)
    con = sqlite3.connect(str(db_path))
    con.execute(
        "INSERT OR REPLACE INTO presets (name, config_json) VALUES (?, ?)",
        (name, json.dumps(clean_values)),
    )
    con.commit()
    con.close()


def load_preset(db_path: Path, name: str) -> dict | None:
    con = sqlite3.connect(str(db_path))
    row = con.execute("SELECT config_json FROM presets WHERE name=?", (name,)).fetchone()
    con.close()
    if not row:
        return None
    return sanitize_config_values(json.loads(row[0]))
