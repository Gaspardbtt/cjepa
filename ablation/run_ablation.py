"""Run the 8-condition history ablation on every measured window (no threshold filter).

Condition 0 = reference (full history); condition k+1 = slot k hidden at steps 1-5.
err[i, c, s] = ||z_hat[c, s] - z[s]||^2 at the target position (frame T+1).

Saved in errors_<model>.npz:
  err            [n, 8, 7]       main analysis (h=1)
  err_h          [n, 8, 10, 7]   same conditions at horizons h=1..10 (NaN past the video)
  err_tgt_hidden [n, 7, 7, 10]   [i, t, k, h-1]: error on slot t when slots t AND k are
                                 hidden at steps 1-5 (k == t: only t hidden)
  err_copy       [n, 10, 7]      copy-last baseline ||z[T-1] - z[frame_h]||^2

Usage:
  uv run python -m ablation.run_ablation --model cjepa --device cpu [--limit N] [--check]
  debug on one window, outside results/:
  ... --windows matching_preprocess_val/windows.json --results-dir <dir> --check
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from einops import rearrange

from ablation import config
from ablation.windows import load_windows

if str(config.REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(config.REPO_ROOT))
from src.cjepa_predictor import MaskedSlotPredictor

NUM_MASKED_SLOTS = {"cjepa": 4, "ocjepa": 0}  # training setting, unused at inference


def load_predictor(model: str, device: str) -> MaskedSlotPredictor:
    ckpt_path = config.CKPT_PATHS[model]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    predictor = MaskedSlotPredictor(
        num_slots=config.NUM_SLOTS,
        slot_dim=config.SLOT_DIM,
        history_frames=config.HISTORY_FRAMES,
        pred_frames=config.PRED_FRAMES,
        num_masked_slots=NUM_MASKED_SLOTS[model],
        seed=0,
        depth=config.DEPTH,
        heads=config.HEADS,
        dim_head=config.DIM_HEAD,
        mlp_dim=config.MLP_DIM,
        dropout=config.DROPOUT,
    )
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    predictor.load_state_dict(state_dict, strict=True)
    predictor.eval()  # disables dropout
    return predictor.to(device)


def build_masks() -> torch.Tensor:
    """[N_CONDITIONS, S, T] bool, True = visible."""
    masks = torch.zeros(
        config.N_CONDITIONS, config.NUM_SLOTS, config.TOTAL_FRAMES, dtype=torch.bool
    )
    masks[:, :, : config.HISTORY_FRAMES] = (
        True  # reference: full history, future hidden
    )
    for k in range(config.NUM_SLOTS):
        masks[k + 1, k, list(config.HIDDEN_STEPS)] = False
    return masks


def build_target_hidden_masks() -> torch.Tensor:
    """[S*S, S, T] bool; index t*S + k hides slots t and k at steps 1-5."""
    S = config.NUM_SLOTS
    masks = torch.zeros(S * S, S, config.TOTAL_FRAMES, dtype=torch.bool)
    masks[:, :, : config.HISTORY_FRAMES] = True
    for t in range(S):
        for k in range(S):
            masks[t * S + k, [t, k], 1 : config.HISTORY_FRAMES] = False
    return masks


def masked_input(
    slots: torch.Tensor, masks: torch.Tensor, predictor: MaskedSlotPredictor
) -> torch.Tensor:
    """Batched version of probing.utils.prepare_masked_input (same formula).

    slots: [T, S, D] (history at steps 0-5, anything at 6-15); masks: [C, S, T].
    Visible token = slot + time_pe[t]; hidden = mask_token + time_pe[t] + id_proj(z_t0).
    Returns [C, T*S, D].
    """
    T = slots.shape[0]
    time_pe = predictor.time_pos_embed[0, :T]  # [T, 1, D]
    anchor_queries = predictor.id_projector(slots[0])  # [S, D]
    query = predictor.mask_token + time_pe + anchor_queries.unsqueeze(0)  # [T, S, D]
    visible = slots + time_pe  # [T, S, D]
    mask_tsd = masks.transpose(1, 2).unsqueeze(-1)  # [C, T, S, 1]
    x = torch.where(mask_tsd, visible.unsqueeze(0), query.unsqueeze(0))  # [C, T, S, D]
    return rearrange(x, "c t s d -> c (t s) d")


@torch.no_grad()
def predict_conditions(
    history: torch.Tensor, masks: torch.Tensor, predictor: MaskedSlotPredictor
) -> torch.Tensor:
    """history: [6, S, D] -> predictions at all steps for each condition [C, T, S, D]."""
    S, D = history.shape[1:]
    slots = torch.zeros(config.TOTAL_FRAMES, S, D, device=history.device)
    slots[: config.HISTORY_FRAMES] = history
    out = predictor.transformer(masked_input(slots, masks, predictor))
    out = predictor.to_out(out)
    return rearrange(out, "c (t s) d -> c t s d", t=config.TOTAL_FRAMES, s=S)


def window_inputs(
    window: dict[str, Any], video_slots: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return history [6, S, D] and future slots [10, S, D] (h=1..10, NaN past video)."""
    if window["prediction_step"] != 1:
        raise ValueError(f"prediction_step must be 1, got {window['prediction_step']}")
    frames = list(window["history_frames"]) + [window["target_frame"]]
    if len(window["history_frames"]) != config.HISTORY_FRAMES:
        raise ValueError(f"Expected 6 history frames, got {window['history_frames']}")
    if min(frames) < 0 or max(frames) >= video_slots.shape[0]:
        raise IndexError(f"Frames {frames} out of range [0, {video_slots.shape[0]})")
    future = np.full((config.PRED_FRAMES, *video_slots.shape[1:]), np.nan, np.float32)
    for h in config.HORIZONS:
        frame = config.horizon_frame(window["target_frame"], h)
        if frame < video_slots.shape[0]:
            future[h - 1] = video_slots[frame]
    return video_slots[window["history_frames"]], future


@torch.no_grad()
def check_window(
    history: torch.Tensor,
    target: torch.Tensor,
    err: np.ndarray,
    window: dict[str, Any],
    masks: torch.Tensor,
    predictor: MaskedSlotPredictor,
) -> None:
    """Dev sanity checks (NOTES.md), printed for one window."""
    pred = predict_conditions(history, masks, predictor)[0, config.TARGET_POSITION]
    ref = predictor.inference(history.unsqueeze(0))[0, 0]
    max_diff = (pred - ref).abs().max().item()
    print(f"  [check] cond 0 vs inference(): max |diff| = {max_diff:.2e}")
    if not torch.allclose(pred, ref, atol=1e-5):
        raise AssertionError("Reference condition differs from predictor.inference()")

    again = predict_conditions(history, masks, predictor)[: config.N_CONDITIONS, config.TARGET_POSITION]
    again_err = ((again - target) ** 2).sum(-1).cpu().numpy()
    if not np.array_equal(again_err, err):
        raise AssertionError("Two passes give different errors (non-deterministic)")
    print("  [check] deterministic: OK")

    pair = predict_conditions(history, build_target_hidden_masks().to(history.device), predictor)
    pair_err = ((pair[:, config.TARGET_POSITION] - target) ** 2).sum(-1).cpu().numpy()
    S = config.NUM_SLOTS
    self_pair = np.array([pair_err[t * S + t, t] for t in range(S)])
    self_main = np.array([err[t + 1, t] for t in range(S)])
    if not np.allclose(self_pair, self_main, atol=1e-4):
        raise AssertionError(f"Pair (t, t) != condition t+1: {self_pair} vs {self_main}")
    print("  [check] target-hidden (t, t) == condition t+1: OK")

    copy_err = ((history[-1] - target) ** 2).sum(-1).cpu().numpy()
    s2o = window["slot_to_object"]
    obj_slots = sorted(s2o)
    background = [s for s in range(config.NUM_SLOTS) if s not in s2o]
    print(f"  [check] object slots {obj_slots}, background slots {background}")
    print(f"  [check] err ref   (obj slots): {np.round(err[0, obj_slots], 3)}")
    print(f"  [check] copy-last (obj slots): {np.round(copy_err[obj_slots], 3)}")
    for t in obj_slots:
        self_d = err[t + 1, t] - err[0, t]
        bg_d = [round(float(err[b + 1, t] - err[0, t]), 4) for b in background]
        print(
            f"  [check] target slot {t}: D(self) = {self_d:.3f}, D(background) = {bg_d}"
        )


def run(
    model: str,
    device: str,
    limit: int | None,
    check: bool,
    windows_path: Path,
    results_dir: Path,
) -> None:
    torch.manual_seed(0)
    windows = load_windows(windows_path, measured_only=True)
    if limit is not None:
        windows = windows[:limit]
    if not windows:
        raise ValueError(f"No measured window in {windows_path}")
    if not config.SLOTS_SUBSET_PATH.exists():
        raise FileNotFoundError(
            f"{config.SLOTS_SUBSET_PATH} missing: run `python -m ablation.extract_slots`"
        )
    slots_npz = np.load(config.SLOTS_SUBSET_PATH)
    predictor = load_predictor(model, device)
    masks = build_masks().to(device)
    all_masks = torch.cat([masks, build_target_hidden_masks().to(device)])
    S, H, C = config.NUM_SLOTS, config.PRED_FRAMES, config.N_CONDITIONS
    future_pos = slice(config.HISTORY_FRAMES, config.TOTAL_FRAMES)

    n = len(windows)
    errors = np.empty((n, C, S), np.float32)
    errors_h = np.empty((n, C, H, S), np.float32)
    errors_tgt_hidden = np.empty((n, S, S, H), np.float32)
    errors_copy = np.empty((n, H, S), np.float32)
    for i, w in enumerate(windows):
        key = str(w["video_id"])
        if key not in slots_npz:
            raise KeyError(f"Video {key} not in {config.SLOTS_SUBSET_PATH}")
        history_np, future_np = window_inputs(w, slots_npz[key])
        history = torch.from_numpy(history_np).float().to(device)
        future = torch.from_numpy(future_np).float().to(device)  # [H, S, D]
        target = future[0]
        pred = predict_conditions(history, all_masks, predictor)[:, future_pos]  # [C+S*S, H, S, D]
        sq = ((pred - future) ** 2).sum(-1).cpu().numpy()  # [C+S*S, H, S]
        errors_h[i] = sq[:C]
        errors[i] = sq[:C, 0]
        pair = sq[C:].reshape(S, S, H, S)  # [t, k, h, slot]
        errors_tgt_hidden[i] = pair[np.arange(S), :, :, np.arange(S)]  # slot = t
        errors_copy[i] = ((history[-1] - future) ** 2).sum(-1).cpu().numpy()
        if check:
            print(f"Window {w['video_id']}/{w['event_index']}:")
            check_window(history, target, errors[i], w, all_masks, predictor)
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(windows)} windows")

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / config.errors_path(model).name
    np.savez(
        out_path,
        err=errors,
        err_h=errors_h,
        err_tgt_hidden=errors_tgt_hidden,
        err_copy=errors_copy,
    )
    write_index(windows, results_dir / config.ERRORS_INDEX_PATH.name)
    print(f"Saved err {errors.shape} to {out_path}")


def write_index(windows: list[dict[str, Any]], path: Path) -> None:
    """Row i <-> err[i]. Same for both models (same windows, same order)."""
    fields = ["video_id", "event_index", "collision_frame", "target_frame"]
    rows = [{k: w[k] for k in fields} for w in windows]
    if path.exists():
        with path.open() as f:
            existing = [{k: int(v) for k, v in r.items()} for r in csv.DictReader(f)]
        if existing != rows:
            raise ValueError(
                f"{path} exists with a different window list; remove it first"
            )
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(config.CKPT_PATHS))
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    parser.add_argument("--limit", type=int, default=None, help="first N windows only")
    parser.add_argument("--check", action="store_true", help="run dev sanity checks")
    parser.add_argument("--windows", type=Path, default=config.WINDOWS_PATH)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    args = parser.parse_args()
    run(args.model, args.device, args.limit, args.check, args.windows, args.results_dir)


if __name__ == "__main__":
    main()
