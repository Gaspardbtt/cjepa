"""Read Gaspard's windows file (one JSON window per line)."""

import json
from pathlib import Path
from typing import Any

REQUIRED_KEYS = (
    "video_id",
    "event_index",
    "collision_frame",
    "collision_objects",
    "history_frames",
    "target_frame",
    "prediction_step",
    "required_object_ids",
    "slot_to_object",
    "status",
)


def load_windows(path: Path, measured_only: bool = True) -> list[dict[str, Any]]:
    """Return windows sorted by (video_id, event_index), slot_to_object keys as int."""
    if not path.exists():
        raise FileNotFoundError(f"Windows file not found: {path}")
    windows = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                w = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({e})") from e
            missing = [k for k in REQUIRED_KEYS if k not in w]
            if missing:
                raise ValueError(f"{path}:{line_no}: missing keys {missing}")
            if measured_only and w["status"] != "measured":
                continue
            w["slot_to_object"] = {
                int(s): int(o) for s, o in w["slot_to_object"].items()
            }
            windows.append(w)
    windows.sort(key=lambda w: (w["video_id"], w["event_index"]))
    return windows
