#!/usr/bin/env python3
"""Persist archive segments and group lifecycle events."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from scripts.runtime.archive_recorder import safe_segment_name


class GroupEventLogger:
    """Persist archive segments and group lifecycle events to settings.db."""

    def __init__(self, db_path: Path, root_dir: Path, source_name: str) -> None:
        self.db_path = Path(db_path)
        self.root_dir = Path(root_dir)
        self.source_name = safe_segment_name(source_name)
        self._active_groups: dict[int, dict] = {}
        self._active_segment_id: int | None = None
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.db_path))
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def _init_db(self) -> None:
        con = self._connect()
        con.execute(
            "CREATE TABLE IF NOT EXISTS recording_segments ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "source TEXT NOT NULL, "
            "segment_path TEXT NOT NULL, "
            "segment_start_ts REAL NOT NULL, "
            "segment_end_ts REAL, "
            "created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_recording_segments_source_time "
            "ON recording_segments (source, segment_start_ts)"
        )
        con.execute(
            "CREATE TABLE IF NOT EXISTS group_events ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "source TEXT NOT NULL, "
            "group_id INTEGER NOT NULL, "
            "event_type TEXT NOT NULL, "
            "event_ts REAL NOT NULL, "
            "event_local TEXT NOT NULL, "
            "member_ids_json TEXT NOT NULL, "
            "member_count INTEGER NOT NULL, "
            "group_type TEXT, "
            "score_pct INTEGER NOT NULL, "
            "bbox_json TEXT NOT NULL, "
            "view TEXT, "
            "segment_id INTEGER, "
            "segment_path TEXT, "
            "created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, "
            "FOREIGN KEY(segment_id) REFERENCES recording_segments(id))"
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_group_events_source_time "
            "ON group_events (source, event_ts)"
        )
        con.commit()
        con.close()

    def _relative_segment_path(self, segment_path: Path | None) -> str | None:
        if segment_path is None:
            return None
        try:
            return str(segment_path.relative_to(self.root_dir))
        except ValueError:
            return str(segment_path)

    def on_segment_open(self, segment_path: Path, ts: float) -> None:
        rel_path = self._relative_segment_path(segment_path)
        con = self._connect()
        cur = con.execute(
            "INSERT INTO recording_segments (source, segment_path, segment_start_ts) VALUES (?, ?, ?)",
            (self.source_name, rel_path, float(ts)),
        )
        self._active_segment_id = int(cur.lastrowid)
        con.commit()
        con.close()

    def on_segment_close(self, segment_path: Path, started_at: float, ended_at: float) -> None:
        rel_path = self._relative_segment_path(segment_path)
        con = self._connect()
        if self._active_segment_id is not None:
            con.execute(
                "UPDATE recording_segments SET segment_end_ts = ? WHERE id = ?",
                (float(ended_at), self._active_segment_id),
            )
        else:
            con.execute(
                "UPDATE recording_segments SET segment_end_ts = ? "
                "WHERE source = ? AND segment_path = ? AND segment_start_ts = ?",
                (float(ended_at), self.source_name, rel_path, float(started_at)),
            )
        con.commit()
        con.close()
        self._active_segment_id = None

    def _write_event(self, *, event_type: str, ts: float, snapshot: dict, segment_path: Path | None) -> None:
        con = self._connect()
        con.execute(
            "INSERT INTO group_events ("
            "source, group_id, event_type, event_ts, event_local, member_ids_json, member_count, "
            "group_type, score_pct, bbox_json, view, segment_id, segment_path"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                self.source_name,
                int(snapshot["group_id"]),
                event_type,
                float(ts),
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)),
                json.dumps(list(snapshot.get("member_ids", ())), ensure_ascii=True),
                int(snapshot.get("member_count", 0)),
                snapshot.get("group_type", ""),
                int(snapshot.get("score_pct", 0)),
                json.dumps(list(snapshot.get("bbox", ())), ensure_ascii=True),
                snapshot.get("view", ""),
                self._active_segment_id,
                self._relative_segment_path(segment_path),
            ),
        )
        con.commit()
        con.close()

    def sync(self, snapshots: dict[int, dict], ts: float, segment_path: Path | None = None) -> None:
        normalized = {int(group_id): dict(snapshot) for group_id, snapshot in snapshots.items()}
        for group_id, snapshot in normalized.items():
            previous = self._active_groups.get(group_id)
            if previous is None:
                self._write_event(event_type="started", ts=ts, snapshot=snapshot, segment_path=segment_path)
            else:
                changed = any(
                    previous.get(field) != snapshot.get(field)
                    for field in ("member_ids", "member_count", "group_type", "view")
                )
                if changed:
                    self._write_event(event_type="updated", ts=ts, snapshot=snapshot, segment_path=segment_path)

        for group_id, snapshot in list(self._active_groups.items()):
            if group_id not in normalized:
                self._write_event(event_type="ended", ts=ts, snapshot=snapshot, segment_path=segment_path)

        self._active_groups = normalized

    def close(self, ts: float, segment_path: Path | None = None) -> None:
        self.sync({}, ts, segment_path=segment_path)
