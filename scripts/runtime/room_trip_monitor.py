#!/usr/bin/env python3
"""Doorway ROI entry timing for room monitoring."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class _MemberState:
    last_anchor: tuple[float, float] | None = None
    was_in_doorway: bool = False
    visible: bool = False


@dataclass
class RoomGroupState:
    member_ids: set[int] = field(default_factory=set)
    members: dict[int, _MemberState] = field(default_factory=dict)
    last_seen_at: float | None = None
    in_room_since: float | None = None
    alerted: bool = False
    all_gone_frames: int = 0
    needs_exit_first: bool = False


class RoomTripMonitor:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.states: dict[int, RoomGroupState] = {}
        self.stale_ttl = max(
            1,
            int(getattr(cfg, "track_reid_window", getattr(cfg, "max_misses", 30))),
        )

    def update(
        self,
        group_snapshots: dict[int, dict],
        now_ts: float,
        *,
        boxes_by_person_id: dict[int, tuple[float, float, float, float]],
        frame_shape: tuple[int, int],
    ) -> list[dict[str, object]]:
        doorway_roi = tuple(
            (float(p[0]), float(p[1]))
            for p in getattr(self.cfg, "doorway_roi_polygon", ())
        )
        if not getattr(self.cfg, "use_doorway_monitor", False) or len(doorway_roi) < 3:
            self.states.clear()
            return []

        frame_h, frame_w = frame_shape
        doorway_pts = np.array(
            [(int(px * frame_w), int(py * frame_h)) for px, py in doorway_roi],
            dtype=np.int32,
        )
        threshold = float(getattr(self.cfg, "room_presence_alert_seconds", 5.0))

        # ── 1. Register new "Minor and Adult" groups, remember member IDs ──
        for group_id, snapshot in group_snapshots.items():
            if snapshot.get("group_type") != "Minor and Adult":
                continue
            gid = int(group_id)
            if gid not in self.states:
                self.states[gid] = RoomGroupState()
            state = self.states[gid]
            for mid in snapshot.get("member_ids", ()):
                mid = int(mid)
                state.member_ids.add(mid)
                if mid not in state.members:
                    state.members[mid] = _MemberState()

        # ── 2. Update every tracked member's position independently ──
        statuses: list[dict[str, object]] = []
        remove_ids: list[int] = []

        for gid, state in self.states.items():
            any_visible = False

            for mid in state.member_ids:
                ms = state.members.setdefault(mid, _MemberState())
                box = boxes_by_person_id.get(mid)

                if box is not None:
                    x1, y1, x2, y2 = (float(v) for v in box)
                    anchor = ((x1 + x2) * 0.5, y2)
                    ms.last_anchor = anchor
                    ms.visible = True
                    any_visible = True

                    if not state.needs_exit_first:
                        if cv2.pointPolygonTest(doorway_pts, anchor, False) >= 0:
                            ms.was_in_doorway = True
                else:
                    ms.visible = False

            if any_visible:
                state.last_seen_at = now_ts
                state.all_gone_frames = 0
            else:
                state.all_gone_frames += 1

            # ── 3. Determine if the group went into the room ──
            any_entered = any(ms.was_in_doorway for ms in state.members.values())
            all_gone = all(not ms.visible for ms in state.members.values())
            any_in_doorway_now = any(
                ms.visible and ms.was_in_doorway
                for ms in state.members.values()
            )

            # Any member reappeared → trip is over, reset timer.
            if state.in_room_since is not None and any_visible:
                print(f"[Room] Group {gid} Minor+Adult — member reappeared, timer reset")
                state.in_room_since = None
                state.alerted = False
                state.needs_exit_first = True
                for ms in state.members.values():
                    ms.was_in_doorway = False

            # Unlock re-arming once all visible members are outside the ROI.
            if state.needs_exit_first and any_visible:
                all_outside = all(
                    cv2.pointPolygonTest(doorway_pts, ms.last_anchor, False) < 0
                    for ms in state.members.values()
                    if ms.visible and ms.last_anchor is not None
                )
                if all_outside:
                    state.needs_exit_first = False
                    for ms in state.members.values():
                        ms.was_in_doorway = False

            # If the group moved fully back outside before disappearing,
            # clear the pending doorway-entry memory so off-scene disappearance
            # is not misread as a fresh entry.
            if state.in_room_since is None and any_visible:
                visible_members = [
                    ms
                    for ms in state.members.values()
                    if ms.visible and ms.last_anchor is not None
                ]
                if visible_members:
                    all_outside = all(
                        cv2.pointPolygonTest(doorway_pts, ms.last_anchor, False) < 0
                        for ms in visible_members
                    )
                    if all_outside:
                        for ms in state.members.values():
                            ms.was_in_doorway = False

            # All members gone after doorway entry → they went in.
            if any_entered and all_gone and state.in_room_since is None:
                state.in_room_since = (
                    state.last_seen_at if state.last_seen_at is not None else now_ts
                )
                print(
                    f"[Room] Group {gid} Minor+Adult — all members gone "
                    f"after doorway entry, timing started"
                )

            # ── 4. Emit elapsed time ──
            if state.in_room_since is not None:
                elapsed = max(0.0, now_ts - state.in_room_since)
                if not state.alerted and elapsed >= threshold:
                    print(
                        f"[Room] Group {gid} Minor+Adult in room "
                        f"for {elapsed:.1f}s — ALERT"
                    )
                    state.alerted = True
                statuses.append(
                    {
                        "group_id": gid,
                        "seconds_in_room": elapsed,
                        "alerted": bool(state.alerted),
                    }
                )

            # Expire groups that have been gone way too long.
            if state.all_gone_frames > self.stale_ttl:
                remove_ids.append(gid)

        for gid in remove_ids:
            del self.states[gid]

        statuses.sort(key=lambda s: int(s["group_id"]))
        return statuses
