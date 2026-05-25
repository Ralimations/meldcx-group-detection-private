import math
import cv2
import numpy as np
from collections import defaultdict, deque
from config import OpenVinoDefaults
from scripts.interaction_detector import InteractionDetector

class GroupDetector:
    def __init__(self, cfg: OpenVinoDefaults):
        self.cfg = cfg
        history_len = cfg.group_history_len
        self.history = defaultdict(lambda: deque(maxlen=history_len))
        self.h_history = defaultdict(lambda: deque(maxlen=history_len))
        # Keep track of the 'age' of an ID (how many total frames it has been tracked)
        self.track_ages = defaultdict(int)
        # Track how many frames each ID has been idle (for persistent idle detection)
        self.idle_frame_counts = defaultdict(int)
        self.track_stale_counts = defaultdict(int)
        self.pair_stale_counts = defaultdict(int)
        self.state_retention_frames = max(1, int(getattr(cfg, "max_misses", 30)))
        
        self.max_relative_dist_check = cfg.group_max_relative_dist_check
        self.max_relative_dist_lock = cfg.group_max_relative_dist_lock
        self.min_history = cfg.group_min_history
        
        # Confidence score system (replaces binary frame counter)
        self.pair_scores = defaultdict(float)
        self.pair_peak_scores = defaultdict(float)
        self.lock_threshold = cfg.group_lock_threshold
        self.max_score = cfg.group_max_score
        self.speed_ratio_thresh = getattr(cfg, "group_speed_ratio_threshold", 0.6)
        self.static_vs_moving_blocker_enabled = getattr(cfg, "group_static_vs_moving_blocker", True)
        
        # To store state details for rendering
        self.pair_states = {} # {(id1, id2): "Checking" | "Locked" | "Weakening"}
        
        # Frozen group identity — once a group locks, its members get a permanent group ID
        # Prevents separately-locked groups from merging via union-find
        self.frozen_groups = {}  # tid -> frozen_group_id
        self.next_group_id = 1
        
        # Persistent group memory — remembers which group ID a track or pair
        # belonged to so that re-locking reuses the same ID instead of minting a new one
        self.tid_group_memory = {}    # tid -> last known group_id
        self.pair_group_memory = {}   # (min_id, max_id) -> group_id
        
        # Relative position stability — track angle between pair centers over time
        self.pair_angle_history = defaultdict(lambda: deque(maxlen=cfg.group_pair_angle_history_len))
        
        # Merged detection — only for warning display (throttled at ~1s intervals)
        self.merged_warn_interval = cfg.group_merged_warn_interval
        self.last_merged_warn_frame = 0
        self.cached_merged_warnings = []  # [(box, n_heads, tid)]
        
        # Origin tracking — remember where a person first entered the frame
        # If two people meet but started on opposite sides of the screen, they are passing
        self.track_origins_x = {}  # tid -> first seen X center coordinate
        self.origin_gap_threshold = cfg.group_origin_gap_threshold
        self.frame_count = 0  # global frame counter
        
        # Total groups tracking — unique frozen group IDs ever seen (never re-increments)
        self.seen_frozen_groups = set()
        
        # Perspective-specific variables
        self.leveled_min_size_ratio = getattr(cfg, "leveled_min_size_ratio", 0.55)
        self.leveled_max_foot_y_relative = getattr(cfg, "leveled_max_foot_y_relative", 0.4)
        self.leveled_combo_size_ratio = getattr(cfg, "leveled_combo_size_ratio", 0.75)
        self.leveled_combo_foot_y_relative = getattr(cfg, "leveled_combo_foot_y_relative", 0.25)
        self.leveled_z_velocity_blocking = getattr(cfg, "leveled_z_velocity_blocking", True)

        self.topdown_min_size_ratio = getattr(cfg, "topdown_min_size_ratio", 0.20)
        self.topdown_max_foot_y_relative = getattr(cfg, "topdown_max_foot_y_relative", 1.0)
        self.topdown_combo_size_ratio = getattr(cfg, "topdown_combo_size_ratio", 0.30)
        self.topdown_combo_foot_y_relative = getattr(cfg, "topdown_combo_foot_y_relative", 0.8)
        self.topdown_z_velocity_blocking = getattr(cfg, "topdown_z_velocity_blocking", False)

        self.groups = []
        self.last_group_snapshots = {}
        self.interaction_detector = InteractionDetector(cfg)
        self.group_colors = [
            (0, 255, 255),   # Bright Cyan
            (0, 255, 0),     # Bright Lime
            (255, 0, 255),   # Bright Magenta
            (0, 165, 255),   # Bright Orange 
            (255, 255, 0),   # Bright Yellow
            (0, 0, 255)      # Bright Blue
        ]

    def update(self, boxes, ids, current_view: str = "LEVELED"):
        current_centers = {}
        id_to_box = {}
        
        for i, (box, track_id) in enumerate(zip(boxes, ids)):
            # Use FOOT-BASE (bottom center) instead of geometric center.
            # Foot position is far more stable than center/width during arm gestures.
            cx = (box[0] + box[2]) / 2.0
            cy = float(box[3])   # bottom edge
            tid = int(track_id)
            
            # Record origin if this is the first time seeing this track
            if tid not in self.track_origins_x:
                self.track_origins_x[tid] = cx
                
            self.history[tid].append((cx, cy))
            self.h_history[tid].append(box[3] - box[1])
            self.track_ages[tid] += 1
            self.track_stale_counts[tid] = 0
            current_centers[tid] = (cx, cy)
            id_to_box[tid] = box
        
        # Update idle frame counts based on velocity (computed after velocities are available)
        # We defer this to after velocity computation below
            
        velocities = {}
        z_velocities = {}
        for tid, hist in self.history.items():
            if tid in current_centers and len(hist) >= self.min_history:
                # XY Velocity: Shorter sliding window (10 frames) for responsive movement
                window = min(10, len(hist))
                dx = hist[-1][0] - hist[-window][0]
                dy = hist[-1][1] - hist[-window][1]
                dt = window
                velocities[tid] = (dx / dt, dy / dt)
                
                # Z Velocity (Scale): Linear regression over the entire history (up to 30 frames)
                # Box heights can jitter frame-to-frame. A simple difference is too noisy.
                # Fitting a line gives a highly stable trend of whether the person is 
                # approaching (growing) or walking away (shrinking).
                h_arr = list(self.h_history[tid])
                n = len(h_arr)
                if n >= self.min_history:
                    # Simple linear regression: slope = Cov(x, y) / Var(x) where x is frame index
                    x_mean = (n - 1) / 2.0
                    y_mean = sum(h_arr) / n
                    num = sum((i - x_mean) * (h - y_mean) for i, h in enumerate(h_arr))
                    den = sum((i - x_mean) ** 2 for i in range(n))
                    z_velocities[tid] = num / den if den > 0 else 0.0
        
        # Update per-track idle frame counts
        for tid in current_centers:
            if tid in velocities:
                spd = math.hypot(velocities[tid][0], velocities[tid][1])
                h = self.h_history[tid][-1] if self.h_history[tid] else 1.0
                
                # Normalize idle check: speed < 1% of person height.
                # This makes it resolution-independent (robust at 1080p/4K).
                if spd < (h * 0.01):
                    self.idle_frame_counts[tid] += 1
            else:
                # Not enough history yet — assume idle
                self.idle_frame_counts[tid] += 1
                
        active_ids = list(velocities.keys())
        pairs = []
        for i in range(len(active_ids)):
            for j in range(i + 1, len(active_ids)):
                id1, id2 = active_ids[i], active_ids[j]
                c1, c2 = current_centers[id1], current_centers[id2]
                v1, v2 = velocities[id1], velocities[id2]
                box1, box2 = id_to_box[id1], id_to_box[id2]
                
                h1 = box1[3] - box1[1]
                h2 = box2[3] - box2[1]
                avg_height = (h1 + h2) / 2
                
                # ── DEPTH-AWARE PROXIMITY ──
                # Use foot-base (bottom-center of bounding box) instead of
                # box center.  Foot position is a much better proxy for
                # real-world ground-plane distance.
                foot1_x = (box1[0] + box1[2]) / 2.0
                foot1_y = box1[3]   # bottom edge
                foot2_x = (box2[0] + box2[2]) / 2.0
                foot2_y = box2[3]
                foot_dist = math.hypot(foot1_x - foot2_x, foot1_y - foot2_y)
                relative_dist = foot_dist / max(avg_height, 1.0)
                
                # ── BOX-SIZE RATIO ──
                size_ratio = min(h1, h2) / max(h1, h2, 1.0)
                
                # ── DEPTH COMPATIBILITY ──
                depth_compatible = True
                foot_y_gap = abs(foot1_y - foot2_y)
                foot_y_relative = foot_y_gap / max(avg_height, 1.0)
                
                if current_view == "TOP-DOWN":
                    min_size_ratio = self.topdown_min_size_ratio
                    max_foot_y_relative = self.topdown_max_foot_y_relative
                    combo_size_ratio = self.topdown_combo_size_ratio
                    combo_foot_y_relative = self.topdown_combo_foot_y_relative
                else:
                    min_size_ratio = self.leveled_min_size_ratio
                    max_foot_y_relative = self.leveled_max_foot_y_relative
                    combo_size_ratio = self.leveled_combo_size_ratio
                    combo_foot_y_relative = self.leveled_combo_foot_y_relative
                
                if size_ratio < min_size_ratio:
                    depth_compatible = False
                elif foot_y_relative > max_foot_y_relative:
                    depth_compatible = False
                elif size_ratio < combo_size_ratio and foot_y_relative > combo_foot_y_relative:
                    depth_compatible = False
                
                # Z velocity blocking config (view-dependent)
                if current_view == "TOP-DOWN":
                    enforce_z_blocking = self.topdown_z_velocity_blocking
                else:
                    enforce_z_blocking = self.leveled_z_velocity_blocking
                
                # Speed magnitude for each person
                s1 = math.hypot(v1[0], v1[1])
                s2 = math.hypot(v2[0], v2[1])
                
                # Determine if each person is idle (normalized: speed < 1% of height)
                is_idle_1 = s1 < (h1 * 0.01)
                is_idle_2 = s2 < (h2 * 0.01)
                both_idle = is_idle_1 and is_idle_2
                
                # ── Z-AXIS (SCALE) MOVEMENT ──
                # scale_rate is the per-frame % change in box height, based on the stable regression line
                scale_rate1 = (z_velocities[id1] / max(h1, 1.0)) if id1 in z_velocities else 0.0
                scale_rate2 = (z_velocities[id2] / max(h2, 1.0)) if id2 in z_velocities else 0.0
                
                # Loosen Z-active threshold to 0.5% (was 0.25%) to allow for 
                # breathing/weight-shifting without breaking "standing together"
                z1_active = abs(scale_rate1) > 0.005  
                z2_active = abs(scale_rate2) > 0.005
                
                # Opposite Z-direction blocker:
                # One person growing (approaching camera) while the other is
                # shrinking (walking away) → they are passing each other.
                z_opposite = False
                if enforce_z_blocking and z1_active and z2_active:
                    if (scale_rate1 > 0) != (scale_rate2 > 0):   # opposite signs
                        z_opposite = True
                
                z_moving_together = False
                if enforce_z_blocking and z1_active and z2_active and not z_opposite:
                    if abs(scale_rate1 - scale_rate2) < 0.015:
                        z_moving_together = True
                
                pair_key = (min(id1, id2), max(id1, id2))
                
                # Direction and speed checks (only meaningful when both are moving in X/Y)
                direction_ok = False
                speed_ok = False
                xy_opposite = False
                both_xy_moving = not is_idle_1 and not is_idle_2
                if both_xy_moving:
                    dot = v1[0]*v2[0] + v1[1]*v2[1]
                    mag1 = math.hypot(v1[0], v1[1])
                    mag2 = math.hypot(v2[0], v2[1])
                    cos_sim = dot / max(mag1 * mag2, 1e-6)
                    direction_ok = cos_sim > 0.85  # Stricter: ~31° tolerance (up from 0.75/41°)
                    
                    # Hard blocker: opposite XY directions (> 90° apart)
                    # "if person a,b are moving in opposite direction,
                    #  they are never in the same group"
                    if cos_sim < 0:
                        xy_opposite = True
                    
                    # Speed ratio check
                    speed_ratio = min(s1, s2) / max(s1, s2, 1e-6)
                    speed_ok = speed_ratio > self.speed_ratio_thresh  # one isn't X% faster than the other
                
                # Persistent idle detection: if a track has been idle >85% of its lifetime
                # and has existed for >30 frames, it's "persistently idle"
                age_1 = self.track_ages[id1]
                age_2 = self.track_ages[id2]
                idle_ratio_1 = self.idle_frame_counts.get(id1, 0) / max(age_1, 1)
                idle_ratio_2 = self.idle_frame_counts.get(id2, 0) / max(age_2, 1)
                persistently_idle_1 = idle_ratio_1 > 0.85 and age_1 > 30
                persistently_idle_2 = idle_ratio_2 > 0.85 and age_2 > 30
                
                # --- NEW: STATIC VS MOVING BLOCKER ---
                # "Statue" vs "Fast Walker" blocker
                # Block grouping if one person is long-term static and the other is actively moving fast.
                # Only lifts if the moving person also slows down/stops.
                static_vs_moving = False
                if self.static_vs_moving_blocker_enabled:
                    # 1.5% height threshold is the "fast walker" line
                    if persistently_idle_1 and s2 > (h2 * 0.015):
                        static_vs_moving = True
                    elif persistently_idle_2 and s1 > (h1 * 0.015):
                        static_vs_moving = True
                
                # Score adjustment
                current_score = self.pair_scores[pair_key]
                
                # Distance gates: loose for checking, tight for full gain
                dist_check_ok = relative_dist < self.max_relative_dist_check
                dist_lock_ok = relative_dist < self.max_relative_dist_lock
                
                # Relative position stability check
                pair_angle = math.atan2(c2[1] - c1[1], c2[0] - c1[0])
                self.pair_angle_history[pair_key].append(pair_angle)
                position_stable = True
                angle_hist = self.pair_angle_history[pair_key]
                if len(angle_hist) >= 5:
                    angles = list(angle_hist)
                    sin_avg = sum(math.sin(a) for a in angles) / len(angles)
                    cos_avg = sum(math.cos(a) for a in angles) / len(angles)
                    resultant = math.hypot(sin_avg, cos_avg)
                    position_stable = resultant > 0.7  # ~45° max spread
                
                # --- NEW: ORIGIN BLOCKER ---
                # Compare where they originally came from. If they started on completely
                # opposite sides of the screen (e.g. left edge vs right edge) but are now close,
                # they must be strangers walking past each other to cross paths.
                origin1_x = self.track_origins_x.get(id1, c1[0])
                origin2_x = self.track_origins_x.get(id2, c2[0])
                origin_gap = abs(origin1_x - origin2_x)
                
                # If their origins were > 400 pixels apart horizontally, permanently block grouping
                # (Assuming common 1080p/720p widths, a 400px+ gap means they didn't enter together)
                origins_opposite = origin_gap > self.origin_gap_threshold
                
                standing_together = both_idle and not z1_active and not z2_active

                # RELAX ORIGIN BLOCKER: If they stop and stand together, ignore their different origins.
                # This allows people arriving from different sides to "meet" in the middle.
                if origins_opposite and standing_together:
                    origins_opposite = False

                # ── FINAL GROUPING DECISION ──
                # Hard blockers — if any of these are true, never group this pair
                hard_blocked = z_opposite or not depth_compatible or xy_opposite or origins_opposite or static_vs_moving
                
                xy_moving_together = both_xy_moving and direction_ok and speed_ok
                moving_together = xy_moving_together or z_moving_together
                
                conditions_met = (
                    not hard_blocked
                    and dist_check_ok
                    and position_stable
                    and (moving_together or standing_together)
                )
                
                if conditions_met:
                    # Conditions met — accumulate score with tiered adaptive gain
                    # NOTE: idle people standing together get the SAME gain as moving together.
                    # Standing close is itself a strong social signal.
                    if relative_dist < 0.3:
                        gain = 0.40  # Ultra Proximity (~3 frames to lock)
                    elif relative_dist < 0.5:
                        gain = 0.25  # High Proximity (~4-5 frames to lock)
                    else:
                        # Standard Proximity (0.5 to 0.8)
                        gain = 0.10  # ~10 frames to lock
                    
                    self.pair_scores[pair_key] = min(current_score + gain, self.max_score)
                    self.pair_peak_scores[pair_key] = max(self.pair_peak_scores[pair_key], self.pair_scores[pair_key])
                    
                    if self.pair_scores[pair_key] >= self.lock_threshold:
                        pairs.append((id1, id2))
                        if current_score < self.lock_threshold:
                            # First lock (or re-lock) — assign frozen group identity
                            fg1 = self.frozen_groups.get(id1)
                            fg2 = self.frozen_groups.get(id2)
                            if fg1 is None and fg2 is None:
                                # Check memory: did this pair or either track have a group before?
                                remembered_gid = self.pair_group_memory.get(pair_key)
                                if remembered_gid is None:
                                    remembered_gid = self.tid_group_memory.get(id1) or self.tid_group_memory.get(id2)
                                if remembered_gid is not None:
                                    gid = remembered_gid
                                else:
                                    gid = self.next_group_id
                                    self.next_group_id += 1
                                self.frozen_groups[id1] = gid
                                self.frozen_groups[id2] = gid
                                # Store in memory
                                self.tid_group_memory[id1] = gid
                                self.tid_group_memory[id2] = gid
                                self.pair_group_memory[pair_key] = gid
                            elif fg1 is not None and fg2 is None:
                                self.frozen_groups[id2] = fg1
                                self.tid_group_memory[id2] = fg1
                            elif fg2 is not None and fg1 is None:
                                self.frozen_groups[id1] = fg2
                                self.tid_group_memory[id1] = fg2
                            # If both already in different frozen groups, don't reassign
                            if self.cfg.verbose_logging:
                                print(f"  --> [SUCCESS] Pair {pair_key} Officially Grouped! (score: {self.pair_scores[pair_key]:.2f})")
                        self.pair_states[pair_key] = "Locked"
                    elif self.pair_peak_scores[pair_key] >= self.lock_threshold and self.pair_scores[pair_key] >= 0.5:
                        # Was locked before, recovering in grace period
                        pairs.append((id1, id2))
                        self.pair_states[pair_key] = "Weakening"
                    else:
                        self.pair_states[pair_key] = "Checking"
                        if self.cfg.verbose_logging:
                            print(f"[GROUPING] Pair {pair_key} score: {self.pair_scores[pair_key]:.2f}/{self.lock_threshold:.1f}")
                else:
                    # Conditions not met — determine decay rate
                    was_locked = self.pair_peak_scores[pair_key] >= self.lock_threshold
                    both_idle = is_idle_1 and is_idle_2
                    
                    # Previously-locked pair still close together → very slow decay
                    # Don't require both_idle — jitter on live streams makes idle detection noisy
                    if was_locked and dist_check_ok and not hard_blocked:
                        decay = 0.02
                    elif hard_blocked:
                        decay = 0.15  # aggressive decay for depth mismatch / opposite Z
                    else:
                        decay = 0.10
                    
                    if current_score > 0.1:
                        if xy_opposite:
                            reason = "Opposite XY direction"
                        elif z_opposite:
                            reason = "Opposite Z (passing)"
                        elif origins_opposite:
                            reason = "Opposite Origins (strangers)"
                        elif static_vs_moving:
                            reason = "Static vs Moving blocker"
                        elif not depth_compatible:
                            reason = f"Depth mismatch (size:{size_ratio:.2f} footY:{foot_y_relative:.2f})"
                        elif not (moving_together or standing_together):
                            reason = "Not moving or standing together"
                        elif not position_stable:
                            reason = "Position drift"
                        else:
                            reason = "Distance"
                        if self.cfg.verbose_logging:
                            print(f"[GROUPING] Pair {pair_key} decaying ({reason}). Score: {current_score:.2f} -> {max(0, current_score - decay):.2f}")
                    
                    self.pair_scores[pair_key] = max(0, current_score - decay)
                    
                    # Check if still in grace period (was locked, score still above 0.5)
                    if was_locked and self.pair_scores[pair_key] >= 0.5:
                        pairs.append((id1, id2))
                        self.pair_states[pair_key] = "Weakening"
                    elif self.pair_scores[pair_key] <= 0:
                        if pair_key in self.pair_states:
                            del self.pair_states[pair_key]
                    
        # Direction-gated union-find grouping
        # Prevents separate pairs walking in different directions from merging
        # into one large group via a shared neighbor.
        parent = {uid: uid for uid in active_ids}
        def find(i):
            if parent[i] == i: return i
            parent[i] = find(parent[i])
            return parent[i]
        
        # Build current group lists for direction checking
        def get_group_members(root):
            return [uid for uid in active_ids if find(uid) == root]
        
        def groups_direction_compatible(group_a, group_b):
            """Check if all members of group_a have compatible direction with all members of group_b."""
            for a in group_a:
                if a not in velocities:
                    continue
                va = velocities[a]
                sa = math.hypot(va[0], va[1])
                ha = self.h_history[a][-1] if self.h_history[a] else 1.0
                if sa < (ha * 0.01):
                    continue  # skip idle members
                for b in group_b:
                    if b not in velocities:
                        continue
                    vb = velocities[b]
                    sb = math.hypot(vb[0], vb[1])
                    hb = self.h_history[b][-1] if self.h_history[b] else 1.0
                    if sb < (hb * 0.01):
                        continue  # skip idle members
                    # Cosine similarity between velocity vectors
                    dot = va[0]*vb[0] + va[1]*vb[1]
                    cos_sim = dot / max(sa * sb, 1e-6)
                    if cos_sim < 0.5:  # broader threshold than pairwise (60° vs 41°)
                        return False
            return True
            
        def frozen_groups_compatible(group_a, group_b):
            """Check that merging won't combine different frozen groups."""
            fgs_a = set(self.frozen_groups[m] for m in group_a if m in self.frozen_groups)
            fgs_b = set(self.frozen_groups[m] for m in group_b if m in self.frozen_groups)
            if fgs_a and fgs_b:
                # Both sides have frozen group members — only merge if same frozen group
                return not fgs_a.isdisjoint(fgs_b)
            return True
        
        for id1, id2 in pairs:
            r1, r2 = find(id1), find(id2)
            if r1 != r2:
                group_a = get_group_members(r1)
                group_b = get_group_members(r2)
                # Block merge if groups belong to different frozen groups
                if not frozen_groups_compatible(group_a, group_b):
                    continue
                # Also check direction compatibility
                if groups_direction_compatible(group_a, group_b):
                    parent[r1] = r2
                
        groups_dict = defaultdict(list)
        for uid in active_ids:
            groups_dict[find(uid)].append(uid)
            
        self.groups = [v for v in groups_dict.values() if len(v) > 1]
        
        # Cleanup stale tracks
        active_set = set(current_centers.keys())
        
        stale_pairs = [k for k in list(self.pair_scores.keys()) if k[0] not in active_set or k[1] not in active_set]
        for k in stale_pairs:
            self.pair_stale_counts[k] += 1
            if self.pair_stale_counts[k] <= self.state_retention_frames:
                continue
            del self.pair_scores[k]
            if k in self.pair_peak_scores:
                del self.pair_peak_scores[k]
            if k in self.pair_states:
                del self.pair_states[k]
            if k in self.pair_angle_history:
                del self.pair_angle_history[k]
            if k in self.pair_group_memory:
                del self.pair_group_memory[k]
            if k in self.pair_stale_counts:
                del self.pair_stale_counts[k]
        active_pairs = [k for k in list(self.pair_scores.keys()) if k[0] in active_set and k[1] in active_set]
        for k in active_pairs:
            self.pair_stale_counts[k] = 0
            
        keys_to_clean = []
        for k in self.history.keys():
            if k not in active_set:
                keys_to_clean.append(k)
        for k in keys_to_clean:
            self.track_stale_counts[k] += 1
            if self.track_stale_counts[k] <= self.state_retention_frames:
                continue
            if k in self.track_ages:
                del self.track_ages[k]
            if k in self.idle_frame_counts:
                del self.idle_frame_counts[k]
            if k in self.frozen_groups:
                del self.frozen_groups[k]
            if k in self.track_origins_x:
                del self.track_origins_x[k]
            if k in self.history:
                del self.history[k]
            if k in self.h_history:
                del self.h_history[k]
            if k in self.tid_group_memory:
                del self.tid_group_memory[k]
            if k in self.track_stale_counts:
                del self.track_stale_counts[k]
        for k in active_set:
            self.track_stale_counts[k] = 0
                     
        return self.groups, current_centers

    def draw_groups(self, frame, boxes, track_ids, current_view: str = "LEVELED", cls_ids=None, labels=None,
                    raw_boxes=None, raw_cls_ids=None, track_demographics=None, pose_observations=None,
                    now_ts: float = 0.0):
        """Draw groups and return (frame, active_group_count, group_progress).
        
        group_progress is a dict {tid: float} where float is 0.0 to 1.0
        indicating how close a person is to being grouped.
        """
        self.last_group_snapshots = {}
        self.interaction_detector.begin_frame()

        if track_ids is None or len(boxes) == 0:
            return frame, 0, {}
        
        self.frame_count += 1
            
        groups, current_centers = self.update(boxes, track_ids, current_view)
        id_to_box = {int(tid): box for tid, box in zip(track_ids, boxes)}
        
        # Head-inside-body detection using raw (unfiltered) detections
        # Identify head detections (class 1) and body detections (class 0) from raw data
        head_boxes = []
        body_count = 0
        head_count = 0
        merged_warnings = []  # [(body_box, n_heads)]
        
        if raw_boxes is not None and raw_cls_ids is not None and labels is not None:
            for rbox, rcls in zip(raw_boxes, raw_cls_ids):
                rcls_int = int(rcls)
                label = labels[rcls_int] if 0 <= rcls_int < len(labels) else ""
                if "head" in label.lower():
                    head_count += 1
                    # Store head center point
                    hcx = (rbox[0] + rbox[2]) / 2.0
                    hcy = (rbox[1] + rbox[3]) / 2.0
                    head_boxes.append((hcx, hcy, rbox))
                elif "body" in label.lower() or "person" in label.lower():
                    body_count += 1
            
            # For each tracked body box, count how many head centers fall inside it
            for tid, box in id_to_box.items():
                bx1, by1, bx2, by2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                heads_inside = 0
                for hcx, hcy, _ in head_boxes:
                    if bx1 <= hcx <= bx2 and by1 <= hcy <= by2:
                        heads_inside += 1
                if heads_inside >= 2:
                    merged_warnings.append((box, heads_inside, tid))
            
            # Throttle merged warnings to ~1 second intervals
            if merged_warnings and (self.frame_count - self.last_merged_warn_frame) >= self.merged_warn_interval:
                self.cached_merged_warnings = merged_warnings
                self.last_merged_warn_frame = self.frame_count
            elif not merged_warnings:
                self.cached_merged_warnings = []
        elif cls_ids is not None and labels is not None:
            # Fallback: count from tracked cls_ids (no raw data available)
            for cid in cls_ids:
                cid_int = int(cid)
                label = labels[cid_int] if 0 <= cid_int < len(labels) else ""
                if "head" in label.lower():
                    head_count += 1
                elif "body" in label.lower() or "person" in label.lower():
                    body_count += 1
        
        # Track seen frozen groups for total count (after all detection logic)
        active_fgs = set(self.frozen_groups.values())
        self.seen_frozen_groups.update(active_fgs)
        
        # Color definitions
        COLOR_WEAKENING = (0, 165, 255)   # Orange
        
        # Build frozen_group_id -> color mapping for distinct group colors
        def get_group_color(tid):
            fgid = self.frozen_groups.get(tid, 0)
            return self.group_colors[fgid % len(self.group_colors)]
        
        id_to_group_idx = {}
        # Map tid to its frozen group ID for display numbering
        id_to_frozen_gid = {}
        for g_idx, group in enumerate(groups):
            for tid in group:
                id_to_group_idx[tid] = g_idx
                id_to_frozen_gid[tid] = self.frozen_groups.get(tid, 0)


        semantic_grouping_enabled = bool(getattr(self.cfg, "semantic_grouping", True))
        semantic_flags = {
            "Pair": bool(getattr(self.cfg, "semantic_pair", True)),
            "Pair of Minors": bool(getattr(self.cfg, "semantic_pair_of_minors", True)),
            "Minor and Adult": bool(getattr(self.cfg, "semantic_minor_and_adult", True)),
            "Pair of Adults": bool(getattr(self.cfg, "semantic_pair_of_adults", True)),
            "Couple": bool(getattr(self.cfg, "semantic_couple", True)),
            "Group of Minors": bool(getattr(self.cfg, "semantic_group_of_minors", True)),
            "Group of Adults": bool(getattr(self.cfg, "semantic_group_of_adults", True)),
            "Family": bool(getattr(self.cfg, "semantic_family", True)),
            "Small Group": bool(getattr(self.cfg, "semantic_small_group", True)),
            "Large Group": bool(getattr(self.cfg, "semantic_large_group", True)),
        }

        # Helper for group type classification (semantic)
        def get_group_type(member_ids, demographics):
            if not semantic_grouping_enabled:
                return ""
            if not member_ids:
                return ""
            n = len(member_ids)
            
            # Extract demographics counts
            n_minors = 0
            n_adults = 0
            n_males = 0
            n_females = 0
            
            for tid in member_ids:
                if demographics and tid in demographics:
                    data = demographics[tid]
                    # Age check
                    age_data = data.get("age")
                    if age_data:
                        age_val = age_data[1]
                        if age_val.startswith("Minor"):
                            n_minors += 1
                        elif age_val.startswith("Adult") or age_val.startswith("Senior"):
                            n_adults += 1
                    
                    # Gender check
                    gender_data = data.get("gender")
                    if gender_data:
                        gender_val = gender_data[1]
                        if gender_val.startswith("Male"):
                            n_males += 1
                        elif gender_val.startswith("Female"):
                            n_females += 1

            # Semantic Logic
            if n == 2:
                # If we have demographics for both
                if n_minors == 2 and semantic_flags["Pair of Minors"]:
                    return "Pair of Minors"
                if n_minors == 1 and n_adults == 1 and semantic_flags["Minor and Adult"]:
                    return "Minor and Adult"
                if n_adults == 2:
                    if n_males == 1 and n_females == 1 and semantic_flags["Couple"]:
                        return "Couple"
                    if semantic_flags["Pair of Adults"]:
                        return "Pair of Adults"
                if semantic_flags["Pair"]:
                    return "Pair"
                return ""
            
            if n >= 3:
                if n_minors == n and semantic_flags["Group of Minors"]:
                    return "Group of Minors"
                if n_adults == n and semantic_flags["Group of Adults"]:
                    return "Group of Adults"
                if n_minors > 0 and n_adults > 0 and semantic_flags["Family"]:
                    return "Family"
                
                if n <= 4 and semantic_flags["Small Group"]:
                    return "Small Group"
                if semantic_flags["Large Group"]:
                    return "Large Group"
                return ""

            return ""

        # Pre-compute group types for each frozen group ID
        fgid_to_type = {}
        for fgid in self.seen_frozen_groups:
            members = [tid for tid, g in id_to_frozen_gid.items() if g == fgid]
            if members:
                g_type = get_group_type(members, track_demographics)
                fgid_to_type[fgid] = g_type
        
        # Draw unified bounding boxes and labels for groups
        # We process by frozen group ID rather than by individual track ID to draw one large box per group
        
        # --- Pre-compute: Per-person group progress scores ---
        # For tracks not yet in a locked group, compute progress from pair checking scores
        # For tracks in a locked group, progress = 1.0
        group_progress = {}
        
        highest_checking_scores = defaultdict(float)
        for (pid1, pid2), state in self.pair_states.items():
            if state == "Checking":
                score = self.pair_scores.get((pid1, pid2), 0)
                highest_checking_scores[pid1] = max(highest_checking_scores[pid1], score)
                highest_checking_scores[pid2] = max(highest_checking_scores[pid2], score)
                
        for tid, max_score in highest_checking_scores.items():
            if tid in id_to_box and tid not in id_to_frozen_gid:
                # Normalize to 0.0 - 1.0 range
                group_progress[tid] = min(1.0, max_score / self.lock_threshold)
        
        # People in locked/frozen groups get 1.0
        for tid in id_to_frozen_gid:
            group_progress[tid] = 1.0

        # --- Draw Unified Locked Groups ---
        # Group tracked IDs by their frozen group ID
        fgid_to_tids = defaultdict(list)
        for tid, fgid in id_to_frozen_gid.items():
            fgid_to_tids[fgid].append(tid)
            
        for fgid, member_tids in fgid_to_tids.items():
            if not member_tids:
                continue
            
            # 1. Determine the unified bounding box encompassing all members and collect points for convex polygon
            min_x, min_y = float('inf'), float('inf')
            max_x, max_y = float('-inf'), float('-inf')
            points_list = []
            
            for tid in member_tids:
                if tid in id_to_box:
                    x1, y1, x2, y2 = id_to_box[tid]
                    min_x = min(min_x, x1)
                    min_y = min(min_y, y1)
                    max_x = max(max_x, x2)
                    max_y = max(max_y, y2)
                    points_list.extend([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                    
            if not points_list:
                continue # Safety check
                
            ux1, uy1, ux2, uy2 = int(min_x), int(min_y), int(max_x), int(max_y)
            
            # 2. Determine the unified visual state (Locked vs Weakening)
            is_locked = False
            is_weakening = False
            
            # If any pair involving any member is locked, the whole group box is locked
            for tid in member_tids:
                for (pid1, pid2), state in self.pair_states.items():
                    if tid == pid1 or tid == pid2:
                        if state == "Locked":
                            is_locked = True
                        elif state == "Weakening":
                            is_weakening = True
                            
            if not is_locked and not is_weakening:
                is_locked = True  # Default fallback if pair state expired but group persists
                
            # 3. Skip old demographics — using gender-neutral labels now
            
            # 4. Draw the unified box
            # Dynamic Color Coding: based on the maximum pair score among the members
            max_score = 0.0
            for tid in member_tids:
                for (pid1, pid2), score in self.pair_scores.items():
                    if tid == pid1 or tid == pid2:
                        max_score = max(max_score, score)

            # Map score (0.0 to 1.0) to color logic (Inverted: Green=0%, Red=Locked)
            if max_score >= 0.75:
                group_color = (0, 0, 255)      # Red (75%+ -> Locked)
            elif max_score >= 0.50:
                group_color = (0, 165, 255)    # Orange (50% - 74%)
            elif max_score >= 0.25:
                group_color = (0, 255, 255)    # Yellow (25% - 49%)
            else:
                group_color = (0, 255, 0)      # Green (0% - 24%)

            group_type = fgid_to_type.get(fgid, "")
            score_pct = min(100, int(max_score * 100))
            carry_hint, carry_score, carry_locked, carry_elapsed_seconds = self.interaction_detector.get_carry_overlap_hint(
                member_ids=member_tids,
                group_type=group_type,
                id_to_box=id_to_box,
                track_demographics=track_demographics,
                pose_observations=pose_observations,
                current_centers=current_centers,
                history_map=self.history,
                now_ts=now_ts,
            )
            label_text = f"Group {fgid} ({score_pct}%)" + (f" ({group_type})" if group_type else "")
            if carry_hint:
                label_text += f" [{carry_hint}]"
            self.last_group_snapshots[int(fgid)] = {
                "group_id": int(fgid),
                "member_ids": tuple(sorted(int(tid) for tid in member_tids)),
                "member_count": len(member_tids),
                "group_type": group_type,
                "interaction_type": carry_hint,
                "interaction_score": carry_score,
                "interaction_locked": bool(carry_locked),
                "interaction_elapsed_seconds": float(carry_elapsed_seconds),
                "score_pct": score_pct,
                "bbox": (ux1, uy1, ux2, uy2),
                "view": str(current_view),
            }
            
            # Compute Convex Hull polygon points
            pts = np.array(points_list, dtype=np.int32)
            hull = cv2.convexHull(pts)
            
            h_scale = frame.shape[0]
            fs = 0.4 if h_scale < 400 else 0.6 if h_scale < 720 else 0.8
            th = 1 if h_scale < 400 else 2
            line_th = max(1, th * 2)

            if is_locked:
                cv2.polylines(frame, [hull], isClosed=True, color=group_color, thickness=line_th)
                cv2.putText(frame, label_text, (ux1, max(0, uy2 + int(30 * (fs/0.8)))), cv2.FONT_HERSHEY_SIMPLEX, fs, group_color, th)
            elif is_weakening:
                cv2.polylines(frame, [hull], isClosed=True, color=group_color, thickness=max(1, line_th - 1))
                cv2.putText(frame, label_text, (ux1, max(0, uy2 + int(30 * (fs/0.8)))), cv2.FONT_HERSHEY_SIMPLEX, fs, group_color, th)
                

            # 5. Draw connection lines between members' feet (Andrea's dynamic logic)
            feet_positions = []
            for tid in member_tids:
                if tid in id_to_box:
                    ix1, iy1, ix2, iy2 = id_to_box[tid]
                    # Approximate feet position: x center, y near the bottom of the box
                    fx = int((ix1 + ix2) // 2)
                    fy = int(iy2)
                    feet_positions.append((fx, fy))
                    
            # Connect them in a chain, validating distance
            for k in range(len(feet_positions) - 1):
                p1 = feet_positions[k]
                p2 = feet_positions[k+1]
                dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                
                # Sanity check before drawing the line
                if dist < 300.0:
                    cv2.line(frame, p1, p2, group_color, 2, cv2.LINE_AA)

        # Expose active non-frozen group candidates too, so downstream doorway logic
        # can react before the group fully locks.
        for member_tids in groups:
            if not member_tids:
                continue
            if any(int(tid) in self.frozen_groups for tid in member_tids):
                continue

            points_list = []
            min_x, min_y = float("inf"), float("inf")
            max_x, max_y = float("-inf"), float("-inf")
            for tid in member_tids:
                if tid not in id_to_box:
                    continue
                x1, y1, x2, y2 = id_to_box[tid]
                min_x = min(min_x, x1)
                min_y = min(min_y, y1)
                max_x = max(max_x, x2)
                max_y = max(max_y, y2)
                points_list.extend([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
            if not points_list:
                continue

            sorted_members = tuple(sorted(int(tid) for tid in member_tids))
            synthetic_gid = -sum((idx + 1) * tid for idx, tid in enumerate(sorted_members))
            if synthetic_gid == 0:
                synthetic_gid = -1
            group_type = get_group_type(sorted_members, track_demographics)
            carry_hint, carry_score, carry_locked, carry_elapsed_seconds = self.interaction_detector.get_carry_overlap_hint(
                member_ids=sorted_members,
                group_type=group_type,
                id_to_box=id_to_box,
                track_demographics=track_demographics,
                pose_observations=pose_observations,
                current_centers=current_centers,
                history_map=self.history,
                now_ts=now_ts,
            )
            max_score = 0.0
            for tid in sorted_members:
                for (pid1, pid2), score in self.pair_scores.items():
                    if tid == pid1 or tid == pid2:
                        max_score = max(max_score, score)
            self.last_group_snapshots[int(synthetic_gid)] = {
                "group_id": int(synthetic_gid),
                "member_ids": sorted_members,
                "member_count": len(sorted_members),
                "group_type": group_type,
                "interaction_type": carry_hint,
                "interaction_score": carry_score,
                "interaction_locked": bool(carry_locked),
                "interaction_elapsed_seconds": float(carry_elapsed_seconds),
                "score_pct": min(100, int(max_score * 100)),
                "bbox": (int(min_x), int(min_y), int(max_x), int(max_y)),
                "view": str(current_view),
            }
        
        # Draw merged detection warnings (throttled — only from cached snapshots)
        if getattr(self.cfg, "show_merged_warnings", False):
            COLOR_MERGED = (0, 165, 255)  # Orange
            for mbox, n_heads, mtid in self.cached_merged_warnings:
                mx1, my1, mx2, my2 = map(int, mbox)
                cv2.putText(frame, f"{n_heads} heads / 1 body",
                            (mx1, max(0, my1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_MERGED, 2)

        self.interaction_detector.end_frame()
        
        # Display HUD
        h_scale = frame.shape[0]
        fs = 0.4 if h_scale < 400 else 0.5 if h_scale < 720 else 0.7
        th = 1 if h_scale < 400 else 2
        
        total_groups = len(self.seen_frozen_groups)
        active_groups = len(groups)
                
        return frame, active_groups, group_progress

    def get_group_snapshots(self):
        return dict(self.last_group_snapshots)
