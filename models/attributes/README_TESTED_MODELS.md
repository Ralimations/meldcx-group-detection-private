# Retained PAR Models In This Repo

This folder now keeps only the PAR assets that are still relevant to the current runtime.

## Active Runtime Asset

### `person-attributes-recognition-crossroad-0234`

Status:
- Retained.
- This is the current body PAR model wired in `config.py`.

Files:
- `person-attributes-recognition-crossroad-0234.xml`
- `person-attributes-recognition-crossroad-0234.bin`
- `person-attributes-recognition-crossroad-0234.labels.txt`

Purpose:
- Body-based gender inference.
- Clothing and carry-item attributes.
- Optional PAR-to-age label mapping through the `body_par_age_*` settings.

Notes:
- This is the only retained body PAR artifact because it matches the current non-dashboard runtime.
- Older experimental or superseded PAR artifacts were removed during repository cleanup once they were no longer referenced by code.

## Related Runtime Assets Outside This Folder

- Person detector: `models/person/best.xml`
- Pose model: `models/pose/yolo11n-pose.xml`
- Optional face override models: `models/face_age/...`

## Cleanup Rule

Only keep model artifacts that are referenced by the active runtime or intentionally retained for an active fallback path.
