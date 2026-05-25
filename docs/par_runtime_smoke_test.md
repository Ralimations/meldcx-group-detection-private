# PAR Runtime Smoke Test

## Goal

Verify that the merged Python runtime works on one real input source with detection, PAR labels, pose gating, height ROI locking, and dashboard controls active together.

## Preconditions

- Confirm the input source is reachable and stable.
- Confirm the dashboard backend starts and the engine can be started from the UI.
- Confirm the configured PAR assets exist:
  - `models/attributes/person-attributes-recognition-crossroad-0234.xml`
  - `models/attributes/person-attributes-recognition-crossroad-0234.bin`
  - `models/attributes/person-attributes-recognition-crossroad-0234.labels.txt`
- Confirm the pose model exists:
  - `models/pose/yolo11n-pose.xml`
  - `models/pose/yolo11n-pose.bin`

## Suggested Starting Settings

- `imgsz = 640`
- `skip_frames = 0`
- `detect_age = true`
- `show_age = true`
- `age_use_height_fallback = true`
- `body_par_use_for_age = false`
- `demographics_interval = 0`
- `use_pose_visibility_gate = true`
- `pose_interval = 0`
- `pose_max_tracks_per_frame = 0`
- `show_pose_keypoints = false`
- `show_height_measurement_line = false`
## Smoke Test Flow

1. Start the dashboard and backend.
2. Select one real camera or one representative video source.
3. Start the engine and confirm frames stream without startup failure.
4. Verify person detection boxes and track IDs appear and remain stable.
5. Verify gender and attribute labels still appear when enabled.
6. Verify age labels appear only when `detect_age` and `show_age` are enabled.
7. Move a subject through the configured height ROI and confirm height-derived labels transition from waiting or pending states into a locked result.
8. Turn `show_pose_keypoints` on and confirm pose markers appear only as an overlay change, not a logic break.
9. Turn `show_height_measurement_line` on and confirm the measurement line appears without destabilizing labels.
10. Toggle `pose_interval`, `demographics_interval`, and `skip_frames` upward and confirm the engine remains responsive while labels update less often.
11. Stop, pause, resume, and reset from the dashboard and confirm the engine remains recoverable.

## Pass Criteria

- No startup crash during engine initialization.
- No runtime exception when demographics, pose, and height logic are active together.
- Detection and tracking remain visible throughout the test.
- Height ROI logic produces understandable intermediate states and can lock on valid samples.
- Dashboard controls update runtime behavior without desynchronizing the stream.

## Follow-up Checks If It Fails

- If labels are missing, inspect `detect_age`, `show_age`, `body_par_use_for_age`, and the PAR label file path.
- If height age never locks, inspect the ROI placement, pose visibility gate, and sample/stability thresholds.
- If performance drops sharply, reduce `imgsz`, increase `skip_frames`, increase `demographics_interval`, and increase `pose_interval` before changing model assets.
- If PAR outputs look wrong, compare the loaded labels file against the model artifact actually configured in `config.py`.
