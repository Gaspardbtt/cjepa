"""Extract the slots of the videos used by the windows from the full HF pkl.

The pkl ({split: {"<id>_pixels.mp4": [128, 7, 128]}}) is loaded once (several GB);
the subset is saved as an npz with one array per video, keyed by str(video_id).

Usage: uv run python -m ablation.extract_slots [--limit N]
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

from ablation import config
from ablation.windows import load_windows


def extract(
    pkl_path: Path, windows_path: Path, out_path: Path, split: str, limit: int | None
) -> None:
    windows = load_windows(windows_path, measured_only=True)
    if limit is not None:
        windows = windows[:limit]
    video_ids = sorted({w["video_id"] for w in windows})
    if not video_ids:
        raise ValueError(f"No measured window in {windows_path}")
    print(f"{len(windows)} windows, {len(video_ids)} videos")

    if not pkl_path.exists():
        raise FileNotFoundError(f"Slots pkl not found: {pkl_path}")
    print(f"Loading {pkl_path} (full file)...")
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    if split not in data:
        raise KeyError(f"Split {split!r} not in pkl (splits: {list(data)})")
    split_slots = data[split]

    missing = [v for v in video_ids if config.slot_key(v) not in split_slots]
    if missing:
        example = next(iter(split_slots))
        raise KeyError(
            f"{len(missing)} videos missing from pkl[{split!r}], e.g. "
            f"{config.slot_key(missing[0])!r}; pkl key example: {example!r}"
        )

    subset = {}
    for v in video_ids:
        slots = np.asarray(split_slots[config.slot_key(v)], dtype=np.float32)
        if slots.ndim != 3 or slots.shape[1:] != (config.NUM_SLOTS, config.SLOT_DIM):
            raise ValueError(f"Video {v}: expected [T, 7, 128], got {slots.shape}")
        subset[str(v)] = slots
    del data, split_slots

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **subset)
    print(f"Saved {len(subset)} videos to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", type=Path, default=config.SLOTS_PKL_PATH)
    parser.add_argument("--windows", type=Path, default=config.WINDOWS_PATH)
    parser.add_argument("--out", type=Path, default=config.SLOTS_SUBSET_PATH)
    parser.add_argument("--split", default=config.SLOTS_SPLIT)
    parser.add_argument("--limit", type=int, default=None, help="first N windows only")
    args = parser.parse_args()
    extract(args.pkl, args.windows, args.out, args.split, args.limit)


if __name__ == "__main__":
    main()
