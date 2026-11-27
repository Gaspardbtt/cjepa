"""Analyze ablation errors: accepted windows -> instances -> metrics tables.

Instance = (window, target in {A, B}); partner = the other collision object;
candidates = slots of required_object_ids except the target.
D[k] = err[k+1, t] - err[0, t]; self = D[t] (excluded from the ranking).

Outputs (in --out-dir, default results/):
  instances.csv            one row per (thresholds, model, instance)
  table_main.csv           metrics at the main thresholds (95/5)
  table_robustness.csv     same at 90/10
  table_by_candidates.csv  top-1 and chance per number of candidates
  table_horizon.csv        exploratory: horizons h=1..10 + aggregate h=1..5, for the
                           standard ablation and the target-hidden variant
                           (D_th(k) = err(t | t and k hidden) - err(t | t hidden))

Usage:
  uv run python -m ablation.analyze
  uv run python -m ablation.analyze --windows matching_preprocess_val/windows_example.json \
      --results-dir <dir> --split all --no-threshold      # debug, prints only
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ablation import config
from ablation.windows import load_windows

STAT_NAMES = ("mean", "min", "q1", "median", "q3", "max")


# --- Window filtering --------------------------------------------------------
def is_accepted(w: dict[str, Any], thresholds: dict[str, float] | None) -> bool:
    flags_ok = (
        w["status"] == "measured"
        and w["reference_complete"]
        and w["identities_stable"]
        and w["measurements_complete"]
    )
    if not flags_ok or thresholds is None:
        return flags_ok
    return (
        w["min_target_coverage"] >= thresholds["min_target_coverage"]
        and w["max_other_coverage"] <= thresholds["max_other_coverage"]
    )


def in_split(video_id: int, split: str) -> bool:
    if split == "all":
        return True
    lo, hi = config.SPLITS[split]
    return lo <= video_id <= hi


# --- Join errors <-> windows ---------------------------------------------------
def load_errors(results_dir: Path, model: str) -> np.ndarray | None:
    path = results_dir / f"errors_{model}.npz"
    if not path.exists():
        return None
    err = np.load(path)["err"]
    if err.ndim != 3 or err.shape[1:] != (config.N_CONDITIONS, config.NUM_SLOTS):
        raise ValueError(f"{path}: expected [n, 8, 7], got {err.shape}")
    return err


def load_index(path: Path) -> list[dict[str, int]]:
    if not path.exists():
        raise FileNotFoundError(f"Errors index not found: {path}")
    with path.open() as f:
        return [{k: int(v) for k, v in row.items()} for row in csv.DictReader(f)]


def join(
    windows: list[dict[str, Any]], index: list[dict[str, int]]
) -> dict[tuple[int, int], int]:
    """(video_id, event_index) -> row of err; checks frames agree with windows."""
    by_key = {(w["video_id"], w["event_index"]): w for w in windows}
    rows = {}
    for i, r in enumerate(index):
        key = (r["video_id"], r["event_index"])
        if key in rows:
            raise ValueError(f"Duplicate key {key} in errors index")
        w = by_key.get(key)
        if w is None:
            continue
        if (r["collision_frame"], r["target_frame"]) != (
            w["collision_frame"],
            w["target_frame"],
        ):
            raise ValueError(f"Frames mismatch for {key} between index and windows")
        rows[key] = i
    return rows


# --- Instances ---------------------------------------------------------------
def object_to_slot(w: dict[str, Any]) -> dict[int, int]:
    obj2slot: dict[int, int] = {}
    for s, o in w["slot_to_object"].items():
        if o in obj2slot:
            raise ValueError(
                f"Window {w['video_id']}/{w['event_index']}: object {o} has 2 slots"
            )
        obj2slot[o] = s
    return obj2slot


def window_instances(w: dict[str, Any], err: np.ndarray) -> list[dict[str, Any]]:
    """Two instances (target A, target B) from err [8, 7] of one window."""
    obj2slot = object_to_slot(w)
    key = f"{w['video_id']}/{w['event_index']}"
    unmapped = [o for o in w["required_object_ids"] if o not in obj2slot]
    if unmapped:
        raise ValueError(f"Window {key}: required objects without slot {unmapped}")
    a, b = w["collision_objects"]
    instances = []
    for target_obj, partner_obj in ((a, b), (b, a)):
        t, p = obj2slot[target_obj], obj2slot[partner_obj]
        cand = [obj2slot[o] for o in w["required_object_ids"] if o != target_obj]
        if len(cand) < 2:
            raise ValueError(f"Window {key}: fewer than 2 candidates")
        d = {k: float(err[k + 1, t] - err[0, t]) for k in cand}
        d_max = max(d.values())
        winners = [k for k in cand if d[k] == d_max]
        others = [d[k] for k in cand if k != p]
        d_self = float(err[t + 1, t] - err[0, t])
        instances.append(
            {
                "video_id": w["video_id"],
                "event_index": w["event_index"],
                "target_object": target_obj,
                "partner_object": partner_obj,
                "target_slot": t,
                "partner_slot": p,
                "n_candidates": len(cand),
                "success": (1.0 / len(winners)) if p in winners else 0.0,
                "chance": 1.0 / len(cand),
                "d_partner": d[p],
                "d_others_mean": float(np.mean(others)),
                "d_others": ";".join(f"{x:.6g}" for x in others),
                "d_self": d_self,
                "ratio_partner_self": d[p] / d_self if d_self > 0 else float("nan"),
                "err_ref_target": float(err[0, t]),
            }
        )
    return instances


# --- Statistics --------------------------------------------------------------
def bootstrap_ci(values: np.ndarray, video_ids: np.ndarray) -> tuple[float, float]:
    """CI of the instance mean, resampling videos (cluster bootstrap)."""
    videos, inv = np.unique(video_ids, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(videos))
    counts = np.bincount(inv, minlength=len(videos)).astype(float)
    rng = np.random.default_rng(config.BOOTSTRAP_SEED)
    idx = rng.integers(0, len(videos), size=(config.N_BOOTSTRAP, len(videos)))
    boot = sums[idx].sum(1) / counts[idx].sum(1)
    alpha = (1 - config.CI_LEVEL) / 2
    lo, hi = np.quantile(boot, [alpha, 1 - alpha])
    return float(lo), float(hi)


def describe(values: np.ndarray, prefix: str) -> dict[str, float]:
    """Boxplot stats (mean, min, q1, median, q3, max); NaN if empty."""
    if values.size == 0:
        return {f"{prefix}_{s}": float("nan") for s in STAT_NAMES}
    q1, med, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    stats = (values.mean(), values.min(), q1, med, q3, values.max())
    return {f"{prefix}_{s}": float(v) for s, v in zip(STAT_NAMES, stats, strict=True)}


def model_metrics(inst: list[dict[str, Any]]) -> dict[str, Any]:
    vid = np.array([r["video_id"] for r in inst])
    success = np.array([r["success"] for r in inst])
    chance = np.array([r["chance"] for r in inst])
    d_p = np.array([r["d_partner"] for r in inst])
    d_o = np.array([r["d_others_mean"] for r in inst])
    d_o_all = np.array(
        [float(x) for r in inst for x in r["d_others"].split(";") if x], dtype=float
    )
    ratio = np.array([r["ratio_partner_self"] for r in inst])
    ratio_ok = ratio[~np.isnan(ratio)]
    positive = (d_p > 0).astype(float)
    diff = d_p - d_o

    m: dict[str, Any] = {
        "n_windows": len({(r["video_id"], r["event_index"]) for r in inst}),
        "n_instances": len(inst),
        "n_videos": len(np.unique(vid)),
        "top1": success.mean(),
        "top1_ci_lo": bootstrap_ci(success, vid)[0],
        "top1_ci_hi": bootstrap_ci(success, vid)[1],
        "chance": chance.mean(),
        "chance_ci_lo": bootstrap_ci(chance, vid)[0],
        "chance_ci_hi": bootstrap_ci(chance, vid)[1],
        "frac_d_partner_pos": positive.mean(),
        "frac_d_partner_pos_ci_lo": bootstrap_ci(positive, vid)[0],
        "frac_d_partner_pos_ci_hi": bootstrap_ci(positive, vid)[1],
        "diff_partner_others_mean": diff.mean(),
        "diff_partner_others_ci_lo": bootstrap_ci(diff, vid)[0],
        "diff_partner_others_ci_hi": bootstrap_ci(diff, vid)[1],
        "diff_partner_others_median": float(np.median(diff)),
    }
    m |= describe(d_p, "d_partner")
    m |= describe(d_o, "d_others_inst_mean")  # one value per instance
    m |= describe(d_o_all, "d_others_pooled")  # every non-partner candidate
    m |= describe(ratio_ok, "ratio_partner_self")
    m["ratio_n_excluded_self_le_0"] = int(np.isnan(ratio).sum())
    return m


def chance_row(inst: list[dict[str, Any]]) -> dict[str, Any]:
    vid = np.array([r["video_id"] for r in inst])
    chance = np.array([r["chance"] for r in inst])
    lo, hi = bootstrap_ci(chance, vid)
    return {
        "n_windows": len({(r["video_id"], r["event_index"]) for r in inst}),
        "n_instances": len(inst),
        "n_videos": len(np.unique(vid)),
        "top1": chance.mean(),
        "top1_ci_lo": lo,
        "top1_ci_hi": hi,
    }


def by_candidates(inst: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for n in sorted({r["n_candidates"] for r in inst}):
        group = [r for r in inst if r["n_candidates"] == n]
        vid = np.array([r["video_id"] for r in group])
        success = np.array([r["success"] for r in group])
        lo, hi = bootstrap_ci(success, vid)
        rows.append(
            {
                "n_candidates": n,
                "n_instances": len(group),
                "n_videos": len(np.unique(vid)),
                "top1": success.mean(),
                "top1_ci_lo": lo,
                "top1_ci_hi": hi,
                "chance": 1.0 / n,
            }
        )
    return rows


# --- Exploratory: horizons and target-hidden variant --------------------------
def load_annotation(video_id: int) -> dict[str, Any]:
    lo = video_id // 1000 * 1000
    path = config.ANNOTATION_DIR / f"annotation_{lo}-{lo + 1000}" / f"annotation_{video_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Annotation not found: {path}")
    with path.open() as f:
        return json.load(f)


def max_clean_horizon(w: dict[str, Any], annotation: dict[str, Any]) -> int:
    """Largest h such that every horizon h' <= h is usable (annotations only):
    frame inside the video, all required objects in view, and no other collision
    involving A or B in (target_frame, frame_h']."""
    trajectory = {f["frame_id"]: f for f in annotation["motion_trajectory"]}
    participants = set(w["collision_objects"])
    others = [
        int(e["frame_id"])
        for j, e in enumerate(annotation["collision"])
        if j != w["event_index"] and participants & {int(o) for o in e["object_ids"]}
    ]
    required = set(w["required_object_ids"])
    best = 0
    for h in config.HORIZONS:
        frame = config.horizon_frame(w["target_frame"], h)
        if frame not in trajectory:
            break
        in_view = {o["object_id"] for o in trajectory[frame]["objects"] if o["inside_camera_view"]}
        if not required <= in_view:
            break
        if any(w["target_frame"] < f <= frame for f in others):
            break
        best = h
    return best


def horizon_index(npz: dict[str, np.ndarray], hs: list[int]) -> list[int]:
    """Position of each horizon h in the saved arrays (see npz['horizons'])."""
    stored = [int(h) for h in npz["horizons"]]
    missing = [h for h in hs if h not in stored]
    if missing:
        raise ValueError(f"Horizons {missing} not saved (saved: {stored}); rerun run_ablation")
    return [stored.index(h) for h in hs]


def as_ablation_err(npz: dict[str, np.ndarray], variant: str, hs: list[int]) -> np.ndarray:
    """[n, 8, 7] array read like `err` by window_instances, errors summed over hs.

    standard:      E[0, t] = err(t | ref),      E[k+1, t] = err(t | k hidden)
    target_hidden: E[0, t] = err(t | t hidden), E[k+1, t] = err(t | t and k hidden)
    """
    idx = horizon_index(npz, hs)
    if variant == "standard":
        return npz["err_h"][:, :, idx, :].sum(2)
    pair = npz["err_tgt_hidden"][:, :, :, idx].sum(3)  # [n, t, k]
    e = np.empty((pair.shape[0], config.N_CONDITIONS, config.NUM_SLOTS), pair.dtype)
    e[:, 1:, :] = pair.transpose(0, 2, 1)  # e[i, k+1, t] = pair[i, t, k]
    e[:, 0, :] = np.diagonal(pair, axis1=1, axis2=2)  # e[i, 0, t] = pair[i, t, t]
    return e


HORIZON_COLUMNS = (
    "n_windows", "n_instances", "n_videos", "top1", "top1_ci_lo", "top1_ci_hi", "chance",
    "frac_d_partner_pos", "frac_d_partner_pos_ci_lo", "frac_d_partner_pos_ci_hi",
    "diff_partner_others_mean", "diff_partner_others_ci_lo", "diff_partner_others_ci_hi",
    "diff_partner_others_median", "d_partner_median", "d_others_inst_mean_median",
)


def horizon_rows(
    windows: list[dict[str, Any]],
    npz_by_model: dict[str, dict[str, np.ndarray]],
    rows: dict[tuple[int, int], int],
    split: str,
    horizons: list[int],
) -> list[dict[str, Any]]:
    clean_h: dict[tuple[int, int], int] = {}
    annotations: dict[int, dict[str, Any]] = {}
    out = []
    for th_name, th in config.THRESHOLDS.items():
        accepted = [w for w in windows if is_accepted(w, th) and in_split(w["video_id"], split)]
        for w in accepted:
            key = (w["video_id"], w["event_index"])
            if key not in clean_h:
                if w["video_id"] not in annotations:
                    annotations[w["video_id"]] = load_annotation(w["video_id"])
                clean_h[key] = max_clean_horizon(w, annotations[w["video_id"]])
        settings = [(str(h), [h]) for h in horizons]
        if set(config.AGG_HORIZONS) <= set(horizons):
            settings.append((f"sum_1-{max(config.AGG_HORIZONS)}", list(config.AGG_HORIZONS)))
        for label, hs in settings:
            ws = [w for w in accepted if clean_h[(w["video_id"], w["event_index"])] >= max(hs)]
            for model, npz in npz_by_model.items():
                variants = ["standard"] + (["target_hidden"] if "err_tgt_hidden" in npz else [])
                for variant in variants:
                    base = {"thresholds": th_name, "model": model, "variant": variant,
                            "horizon": label}
                    if not ws:
                        out.append(base | {"n_windows": 0})
                        continue
                    err = as_ablation_err(npz, variant, hs)
                    inst = build_instances(ws, err, rows, None, "all")
                    m = model_metrics(inst)
                    row = base | {k: m[k] for k in HORIZON_COLUMNS}
                    if variant == "standard":
                        idx = horizon_index(npz, hs)
                        copy = np.array([
                            npz["err_copy"][rows[(r["video_id"], r["event_index"])], idx,
                                            r["target_slot"]].sum()
                            for r in inst
                        ])
                        ref = np.array([r["err_ref_target"] for r in inst])
                        row |= {
                            "err_ref_median": float(np.median(ref)),
                            "err_copy_median": float(np.median(copy)),
                            "frac_ref_below_copy": float((ref < copy).mean()),
                            "ratio_partner_self_median": m["ratio_partner_self_median"],
                        }
                    out.append(row)
    return out


# --- Main ----------------------------------------------------------------------
def build_instances(
    windows: list[dict[str, Any]],
    err: np.ndarray,
    rows: dict[tuple[int, int], int],
    thresholds: dict[str, float] | None,
    split: str,
) -> list[dict[str, Any]]:
    instances = []
    for w in windows:
        key = (w["video_id"], w["event_index"])
        if not (is_accepted(w, thresholds) and in_split(w["video_id"], split)):
            continue
        if key not in rows:
            raise KeyError(f"Accepted window {key} has no ablation errors")
        instances.extend(window_instances(w, err[rows[key]]))
    return instances


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Nothing to write to {path}")
    fields = list(rows[0])
    for r in rows[1:]:
        fields += [k for k in r if k not in fields]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, default=config.WINDOWS_PATH)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--split", default="eval", choices=[*config.SPLITS, "all"])
    parser.add_argument(
        "--no-threshold",
        action="store_true",
        help="debug: flags only, print, write nothing",
    )
    parser.add_argument(
        "--horizons", type=int, nargs="+", default=None,
        help="horizons for table_horizon (default: all saved by run_ablation)",
    )
    args = parser.parse_args()

    windows = load_windows(args.windows, measured_only=True)
    index = load_index(args.results_dir / config.ERRORS_INDEX_PATH.name)
    rows = join(windows, index)
    errors = {
        m: e
        for m in config.CKPT_PATHS
        if (e := load_errors(args.results_dir, m)) is not None
    }
    if not errors:
        raise FileNotFoundError(f"No errors_<model>.npz in {args.results_dir}")
    for m, e in errors.items():
        if len(e) != len(index):
            raise ValueError(
                f"errors_{m}.npz has {len(e)} rows, index has {len(index)}"
            )

    threshold_sets: dict[str, dict[str, float] | None] = (
        {"none": None} if args.no_threshold else dict(config.THRESHOLDS)
    )
    all_instances, tables, cand_rows = [], {}, []
    for th_name, th in threshold_sets.items():
        table = []
        for model, err in errors.items():
            inst = build_instances(windows, err, rows, th, args.split)
            if not inst:
                raise ValueError(f"No instance for {model} at thresholds {th_name}")
            all_instances += [{"thresholds": th_name, "model": model} | r for r in inst]
            table.append({"model": model} | model_metrics(inst))
            cand_rows += [
                {"thresholds": th_name, "model": model} | r for r in by_candidates(inst)
            ]
        table.append({"model": "chance"} | chance_row(inst))
        tables[th_name] = table

    if args.no_threshold:
        for r in all_instances:
            print(
                {
                    k: r[k]
                    for k in (
                        "model",
                        "video_id",
                        "target_slot",
                        "partner_slot",
                        "n_candidates",
                        "success",
                        "chance",
                        "d_partner",
                        "d_others",
                        "d_self",
                        "ratio_partner_self",
                    )
                }
            )
        for r in tables["none"]:
            print({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "instances.csv", all_instances)
    write_csv(args.out_dir / "table_main.csv", tables["main"])
    write_csv(args.out_dir / "table_robustness.csv", tables["robustness"])
    write_csv(args.out_dir / "table_by_candidates.csv", cand_rows)

    npz_by_model = {}
    for m in errors:
        z = np.load(args.results_dir / f"errors_{m}.npz")
        if not {"err_h", "err_copy"} <= set(z.files):
            print(f"errors_{m}.npz has no horizon data: rerun run_ablation for table_horizon")
            continue
        npz_by_model[m] = {k: z[k] for k in z.files if k != "err"}
        if "horizons" not in z.files:  # files saved before --horizons existed
            npz_by_model[m]["horizons"] = np.arange(1, z["err_h"].shape[2] + 1)
        if "err_tgt_hidden" not in z.files:
            print(f"errors_{m}.npz has no target-hidden data (run_ablation --target-hidden)")
    if npz_by_model:
        saved = [{int(h) for h in z["horizons"]} for z in npz_by_model.values()]
        horizons = sorted(args.horizons) if args.horizons else sorted(set.intersection(*saved))
        write_csv(
            args.out_dir / "table_horizon.csv",
            horizon_rows(windows, npz_by_model, rows, args.split, horizons),
        )


if __name__ == "__main__":
    main()
