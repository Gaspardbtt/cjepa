"""Frozen paths and constants for the collision ablation. Single source of truth."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- Inputs -----------------------------------------------------------------
WINDOWS_PATH = REPO_ROOT / "matching_preprocess_val" / "windows.jsonl"  # all windows
DEBUG_WINDOWS_PATH = REPO_ROOT / "matching_preprocess_val" / "windows.json"  # 1 window
SLOTS_PKL_PATH = REPO_ROOT / "checkpoints" / "clevrer_videosaur_slots.pkl"
SLOTS_SUBSET_PATH = REPO_ROOT / "data" / "slots_val_subset.npz"
SLOTS_SPLIT = "val"
CKPT_DIR = REPO_ROOT / "checkpoints" / "cjepa-ckpts"
CKPT_PATHS = {
    "cjepa": CKPT_DIR / "clevrer_videosaur_4_epoch_30_object.ckpt",  # 4/7 masked
    "ocjepa": CKPT_DIR / "clevrer_videosaur_0_epoch_30_object.ckpt",  # 0/7 masked
}


def slot_key(video_id: int) -> str:
    """Key of a video in the slots pkl (format used by calibrate_matching.py)."""
    return f"{video_id}_pixels.mp4"


# --- Outputs ----------------------------------------------------------------
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
ERRORS_INDEX_PATH = RESULTS_DIR / "errors_index.csv"


def errors_path(model: str) -> Path:
    return RESULTS_DIR / f"errors_{model}.npz"


# --- Predictor architecture (must match checkpoints, loaded strict) ---------
NUM_SLOTS = 7
SLOT_DIM = 128
HISTORY_FRAMES = 6
PRED_FRAMES = 10
TOTAL_FRAMES = HISTORY_FRAMES + PRED_FRAMES
DEPTH = 6
HEADS = 16
DIM_HEAD = 64
MLP_DIM = 2048
DROPOUT = 0.1

# --- Ablation protocol (frozen, see CLAUDE.md journal 24/11 and 27/11) ------
TARGET_POSITION = HISTORY_FRAMES  # prediction_step=1 -> position 6 = frame T+1
HIDDEN_STEPS = range(1, HISTORY_FRAMES)  # hide steps 1-5, anchor t=0 kept
N_CONDITIONS = 1 + NUM_SLOTS  # 0 = reference, k+1 = slot k hidden

# --- Exploratory analyses (frozen 27/11, see Journal) ------------------------
FRAME_STEP = 2  # frameskip used in training (scripts/clevrer/train_cjepa_from_slot.sh)
HORIZONS = range(1, PRED_FRAMES + 1)  # h=1..10 <-> positions 6..15 <-> frame T+1+2(h-1)
AGG_HORIZONS = range(1, 6)  # aggregate: errors summed over h=1..5 (positions 6-10)
ANNOTATION_DIR = REPO_ROOT / "data" / "CLEVRER" / "clevrer_annotation" / "val"


def horizon_frame(target_frame: int, h: int) -> int:
    """Video frame predicted at horizon h (h=1 is target_frame)."""
    return target_frame + FRAME_STEP * (h - 1)

# --- Window acceptance thresholds (frozen 26/11) ----------------------------
THRESHOLDS = {
    "main": {"min_target_coverage": 0.95, "max_other_coverage": 0.05},
    "robustness": {"min_target_coverage": 0.90, "max_other_coverage": 0.10},
}

# --- Splits (inclusive video_id ranges) -------------------------------------
SPLITS = {
    "dev": (10000, 10099),
    "eval": (10100, 14999),
}

# --- Statistics -------------------------------------------------------------
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 0
CI_LEVEL = 0.95
