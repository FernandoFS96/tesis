"""
evaluate.py
==============================================================================
Unified validation script: compares all models on all tasks.

Loads:
    - 17 Specialist MLPs    (from --mlp_run/specialist_XX/best.pt)
    - 1  DR-MLP             (from --mlp_run/dr_mlp/best.pt)
    - 1  ANP dual-target    (from --anp_run,       predicts SoC% + Cycle)
    - 1  ANP SoC-only       (from --anp_soc_run,   predicts SoC% only)
    - 1  ANP Cycle-only     (from --anp_cycle_run, predicts Cycle only)

Single-target ANP models are evaluated on both targets: the target they were trained on is reported normally; the target they did not predict is NaN.
This allows a fair side-by-side comparison in the same table and heatmaps.

Evaluation protocol (identical for all models):
    - Context window : first ctx_cycles × meas_per_cycle rows (default 1 800)
    - Target window  : next  tgt_cycles × meas_per_cycle rows (default 1 800)
    - Tasks evaluated: all 25 synthetic datasets (17 train + 5 val + 3 test)

Outputs (saved to --out_dir, default ./validation/<timestamp>/):
    mae_SoC_pct.csv          — MAE SoC (%) for all models × all tasks
    mae_Cycle.csv            — MAE Cycle for all models × all tasks
    mae_comparison.csv       — combined wide-format table
    mae_soc_heatmap.png      — heatmap SoC
    mae_cycle_heatmap.png    — heatmap Cycle
    bar_comparison.png       — grouped bar chart (avg per split)
    summary.txt              — human-readable summary table

Usage:
    python evaluate.py \
        --mlp_run           ../train/runs_mlp/20260511_121741 \
        --anp_run           ../train/runs/anp_all/20260512_124715 \
        --anp_soc_run       ../train/runs/anp_SoC/20260512_114601 \
        --anp_cycle_run     ../train/runs/anp_Cycle/20260512_114703 \
        --anp_soc_reduced_run   ../train/runs/anp_SoC_reduced/20260513_110544 \
        --anp_cycle_reduced_run ../train/runs/anp_Cycle_reduced/20260513_114323 \
        --data_dir          ../csic_real_synth_load/prepared_data

Location: csic/validation/evaluate.py
==============================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ── Resolve package root so imports work regardless of CWD ───────────────────
_VAL_DIR   = Path(__file__).resolve().parent          # csic/validation/
_CSIC_ROOT = _VAL_DIR.parent                          # csic/
_TRAIN_DIR = _CSIC_ROOT / "train"

for _p in [str(_CSIC_ROOT), str(_TRAIN_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from train.train_utils import (
    load_prepared_data,
    validate_targets,
    get_task_splits,
    sort_task_by_cycle,
    REDUCED_FEATURE_SETS,
    get_feature_indices,
    filter_x,
)

# Import model architectures
try:
    from models.anp import LatentModel
except ImportError:
    from models.anp import LatentModel

try:
    from train.train_mlp import MLP as _TrainMLP #type: ignore 
except ImportError:
    # Inline MLP definition as fallback (same architecture as train_mlp.py)
    import torch.nn.init as init

    class _TrainMLP(nn.Module):
        def __init__(self, input_dim, output_dim, neurons=128, dropout=0.1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, neurons), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(neurons, neurons),   nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(neurons, neurons),   nn.ReLU(),
                nn.Linear(neurons, output_dim),
            )
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    if m.out_features == output_dim:
                        init.xavier_uniform_(m.weight)
                    else:
                        init.kaiming_normal_(m.weight, nonlinearity="relu")
                    nn.init.zeros_(m.bias)

        def forward(self, x):
            return self.net(x)

MLP = _TrainMLP


# ==============================================================================
# DATA HELPERS
# ==============================================================================

def extract_window(
    X:        pd.DataFrame,
    y:        pd.DataFrame,
    ctx_rows: int,
    tgt_rows: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract context and target row windows from a cycle-sorted task."""
    T       = len(X)
    ctx_end = min(ctx_rows, T)
    tgt_end = min(ctx_end + tgt_rows, T)
    X_arr   = X.values.astype(np.float32)
    y_arr   = y.values.astype(np.float32)
    return X_arr[:ctx_end], y_arr[:ctx_end], X_arr[ctx_end:tgt_end], y_arr[ctx_end:tgt_end]


def compute_mae(
    pred:          np.ndarray,
    true:          np.ndarray,
    denorm_values: dict,
    target_cols:   List[str],
) -> Dict[str, float]:
    """Denormalized MAE per target column. Clips SoC predictions to [0, 100]."""
    result = {}
    for i, col in enumerate(target_cols):
        m = denorm_values["y_mean"].get(col, 0.0)
        s = denorm_values["y_std"].get(col, 1.0)
        
        # Denormalize predictions
        pred_denorm = pred[:, i] * s + m
        true_denorm = true[:, i] * s + m
        
        # Clip SoC predictions to [0, 100] since it's a battery percentage
        if col == "SoC (%)":
            pred_denorm = np.clip(pred_denorm, 0.0, 100.0)
        
        result[col] = float(np.abs(pred_denorm - true_denorm).mean())
    return result


# ==============================================================================
# MODEL LOADING
# ==============================================================================

def load_mlp_specialists(
    mlp_run_dir: Path,
    input_dim:   int,
    output_dim:  int,
    device:      torch.device,
    n_specialists: int = 17,
) -> List[Tuple[str, nn.Module]]:
    """
    Load all specialist MLP checkpoints from a runs_mlp directory.

    Searches for specialist_01/best.pt … specialist_NN/best.pt.
    Also reads config.json to recover the neurons/dropout used during training.

    Args:
        mlp_run_dir:   Path to a runs_mlp/<timestamp>/ directory.
        input_dim:     Model input dimension.
        output_dim:    Model output dimension.
        device:        Torch device.
        n_specialists: Expected number of specialists.

    Returns:
        List of (label, model) tuples, model in eval mode.
    """
    # Read training config for architecture params
    cfg_path = mlp_run_dir / "config.json"
    neurons, dropout = 128, 0.1
    if cfg_path.exists():
        with cfg_path.open() as f:
            cfg = json.load(f)
        neurons = cfg.get("neurons", 128)
        dropout = cfg.get("dropout", 0.1)

    models = []
    for i in range(1, n_specialists + 1):
        label    = f"specialist_{i:02d}"
        ckpt_path = mlp_run_dir / label / "best.pt"
        if not ckpt_path.exists():
            print(f"  ⚠  {label}: best.pt not found at {ckpt_path} — skipping")
            continue
        model = MLP(input_dim, output_dim, neurons, dropout)
        ckpt  = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.eval().to(device)
        models.append((label, model))
        print(f"  ✓  {label} loaded")
    return models


def load_dr_mlp(
    mlp_run_dir: Path,
    input_dim:   int,
    output_dim:  int,
    device:      torch.device,
) -> Optional[Tuple[str, nn.Module]]:
    """
    Load the DR-MLP checkpoint from a runs_mlp directory.

    Returns:
        ('dr_mlp', model) tuple, or None if checkpoint not found.
    """
    cfg_path  = mlp_run_dir / "config.json"
    neurons, dropout = 128, 0.1
    if cfg_path.exists():
        with cfg_path.open() as f:
            cfg = json.load(f)
        neurons = cfg.get("neurons", 128)
        dropout = cfg.get("dropout", 0.1)

    ckpt_path = mlp_run_dir / "dr_mlp" / "best.pt"
    if not ckpt_path.exists():
        print(f"  ⚠  DR-MLP: best.pt not found at {ckpt_path} — skipping")
        return None

    model = MLP(input_dim, output_dim, neurons, dropout)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    print(f"  ✓  DR-MLP loaded")
    return ("dr_mlp", model)


def load_anp_model(
    run_dir:          Path,
    input_dim:        int,
    all_target_cols:  List[str],
    device:           torch.device,
    label:            str = "anp",
) -> Optional[Tuple[str, nn.Module, List[str], Optional[List[str]]]]:
    """
    Load an ANP checkpoint and detect which targets and features it was trained on.
    Returns (label, model, model_target_cols, feature_cols) or None if not found.
    feature_cols is None for full-feature models, or a list of column names for reduced-feature models.
    """
    ckpt_path = run_dir / "best.pt"
    cfg_path  = run_dir / "config.json"

    if not ckpt_path.exists():
        print(f"  ⚠  {label}: best.pt not found at {ckpt_path} — skipping")
        return None

    num_hidden   = 128
    attn_dropout = 0.1
    target_col   = "all"
    use_reduced  = False

    if cfg_path.exists():
        with cfg_path.open() as f:
            cfg_data = json.load(f)
        num_hidden   = (cfg_data.get("num_hidden")
                        or cfg_data.get("params", {}).get("num_hidden", 128))
        attn_dropout = cfg_data.get("attn_dropout", 0.1)
        target_col   = cfg_data.get("target_col", "all")
        use_reduced  = cfg_data.get("use_reduced_features", False)

    # Determine which targets this model predicts
    if target_col == "all":
        model_target_cols = all_target_cols
    elif target_col in all_target_cols:
        model_target_cols = [target_col]
    else:
        raw = torch.load(ckpt_path, map_location="cpu")
        out_keys = [k for k in raw["model"] if "mean_projection" in k and "weight" in k]
        out_dim  = raw["model"][out_keys[0]].shape[0] if out_keys else len(all_target_cols)
        model_target_cols = all_target_cols[:out_dim]

    # Determine which X features this model was trained on
    feature_cols    = REDUCED_FEATURE_SETS.get(target_col) if use_reduced else None
    model_input_dim = len(feature_cols) if feature_cols is not None else input_dim
    output_dim      = len(model_target_cols)

    model = LatentModel(
        num_hidden=num_hidden,
        input_dim=model_input_dim,
        output_dim=output_dim,
        attn_dropout=attn_dropout,
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    val_mae = ckpt.get("val_MAE", ckpt.get("val_loss", "?"))
    feat_str = f"reduced({model_input_dim})" if feature_cols else f"all({input_dim})"
    print(f"  ✓  {label:<22} targets={model_target_cols}  "
          f"num_hidden={num_hidden}  features={feat_str}  val_MAE={val_mae}")
    return (label, model, model_target_cols, feature_cols)


# ==============================================================================
# EVALUATION
# ==============================================================================

@torch.no_grad()
def eval_mlp(
    model:         nn.Module,
    X_tgt:         np.ndarray,
    y_tgt:         np.ndarray,
    device:        torch.device,
    denorm_values: dict,
    target_cols:   List[str],
) -> Dict[str, float]:
    """
    Evaluate an MLP on a target window.

    The MLP receives individual rows and predicts targets directly, no context mechanism.

    Args:
        model:         MLP in eval mode.
        X_tgt:         Feature array (Nt, D).
        y_tgt:         Target array  (Nt, O).
        device:        Torch device.
        denorm_values: Denormalization scalers.
        target_cols:   Target column names.

    Returns:
        Dict {col: mae} in original units.
    """
    X_t  = torch.tensor(X_tgt, dtype=torch.float32).to(device)
    pred = model(X_t).cpu().numpy()
    return compute_mae(pred, y_tgt, denorm_values, target_cols)


@torch.no_grad()
def eval_anp(
    model:             nn.Module,
    X_ctx:             np.ndarray,
    y_ctx:             np.ndarray,
    X_tgt:             np.ndarray,
    y_tgt:             np.ndarray,
    device:            torch.device,
    denorm_values:     dict,
    all_target_cols:   List[str],
    model_target_cols: List[str],
) -> Dict[str, float]:
    """
    Evaluate an ANP on a single task using context → target prediction.

    Handles both dual-target and single-target models transparently:
    - y_ctx and y_tgt are filtered to the model's trained targets before being passed to the model, so input dimensions always match.
    - Targets that the model does not predict are returned as NaN, allowing all models to be compared in the same result table.

    Args:
        model:             ANP LatentModel in eval mode.
        X_ctx / y_ctx:     Context features and targets (Nc, D/O_all).
        X_tgt / y_tgt:     Target features and ground truth (Nt, D/O_all).
        device:            Torch device.
        denorm_values:     Denormalization scalers (full dict for all targets).
        all_target_cols:   All targets present in the data.
        model_target_cols: Subset of targets this model was trained on.

    Returns:
        Dict {col: mae} for all_target_cols. NaN for unmodelled targets.
    """
    # Filter y to only the targets this model knows about
    if model_target_cols != all_target_cols:
        col_idx   = [all_target_cols.index(c) for c in model_target_cols]
        y_ctx_m   = y_ctx[:, col_idx]
        y_tgt_m   = y_tgt[:, col_idx]
    else:
        y_ctx_m = y_ctx
        y_tgt_m = y_tgt

    ctx_x = torch.tensor(X_ctx,   dtype=torch.float32).unsqueeze(0).to(device)
    ctx_y = torch.tensor(y_ctx_m, dtype=torch.float32).unsqueeze(0).to(device)
    tgt_x = torch.tensor(X_tgt,   dtype=torch.float32).unsqueeze(0).to(device)

    # Prior-only inference (no target_y → realistic deployment scenario)
    pred_mean, _, _, _, _ = model(ctx_x, ctx_y, tgt_x, target_y=None)
    pred = pred_mean.squeeze(0).cpu().numpy()   # (Nt, O_model)

    # Compute MAE only for the model's targets
    mae_model = compute_mae(pred, y_tgt_m, denorm_values, model_target_cols)

    # Fill full result dict: NaN for targets not predicted by this model
    result = {col: float("nan") for col in all_target_cols}
    result.update(mae_model)
    return result


# ==============================================================================
# PLOTTING
# ==============================================================================

_DPI = 150


def plot_heatmaps(
    df_soc:   pd.DataFrame,
    df_cycle: pd.DataFrame,
    out_dir:  Path,
) -> None:
    """Save MAE heatmaps for SoC and Cycle (all models × all tasks)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Vertical separator column index (after last train task)
    train_count = sum(1 for c in df_soc.columns if c.startswith("train_"))

    for df, metric, cmap, fname in [
        (df_soc,   "MAE SoC (%)",  "YlOrRd", "mae_soc_heatmap.png"),
        (df_cycle, "MAE Cycle",    "YlOrRd", "mae_cycle_heatmap.png"),
    ]:
        n_models = len(df)
        n_tasks  = len(df.columns)
        fig_h    = max(5, n_models * 0.45)
        fig_w    = max(10, n_tasks * 0.6)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        vals = df.values.astype(float)
        im   = ax.imshow(vals, cmap=cmap, aspect="auto",
                         vmin=np.nanmin(vals), vmax=np.nanpercentile(vals, 95))

        # Axes labels
        ax.set_xticks(range(n_tasks))
        ax.set_xticklabels(df.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n_models))
        ax.set_yticklabels(df.index, fontsize=8)

        # Annotate cells
        for r in range(n_models):
            for c in range(n_tasks):
                v = vals[r, c]
                if not np.isnan(v):
                    ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                            fontsize=6, color="black")

        # Highlight best (minimum) value per column with green box
        for c in range(n_tasks):
            col_vals = vals[:, c]
            if not np.all(np.isnan(col_vals)):
                min_row = np.nanargmin(col_vals)
                rect = matplotlib.patches.Rectangle(
                    (c - 0.5, min_row - 0.5), 1, 1,
                    linewidth=2.5, edgecolor='lime', facecolor='none'
                )
                ax.add_patch(rect)

        # Vertical separator between train / val / test
        if train_count > 0:
            ax.axvline(train_count - 0.5, color="white", linewidth=2)
        val_count = sum(1 for c in df.columns if c.startswith("val_"))
        if val_count > 0:
            ax.axvline(train_count + val_count - 0.5, color="white", linewidth=2)

        # Horizontal separator before first ANP row
        anp_labels = {"anp", "anp_soc", "anp_cycle"}
        anp_rows   = [i for i, m in enumerate(df.index) if m in anp_labels]
        if anp_rows:
            ax.axhline(min(anp_rows) - 0.5, color="white", linewidth=2)

        plt.colorbar(im, ax=ax, label=metric, shrink=0.8)
        ax.set_title(f"{metric} — all models × all tasks\n"
                     f"(train | val | test columns separated by white lines)",
                     fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓  {fname}")


def plot_bar_comparison(
    df_soc:      pd.DataFrame,
    df_cycle:    pd.DataFrame,
    out_dir:     Path,
) -> None:
    """
    Grouped bar chart: average MAE per model for train / val / test splits.
    One figure for SoC, one for Cycle.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    train_cols = [c for c in df_soc.columns if c.startswith("train_")]
    val_cols   = [c for c in df_soc.columns if c.startswith("val_")]
    test_cols  = [c for c in df_soc.columns if c.startswith("test_")]

    models = list(df_soc.index)
    x      = np.arange(len(models))
    width  = 0.25

    # Color palette: specialists grey, DR-MLP orange, ANP variants in blue tones
    def bar_color(label: str) -> str:
        if label == "anp":          return "#1C7293"   # teal  — dual-target
        if label == "anp_soc":      return "#028090"   # darker teal — SoC only
        if label == "anp_cycle":    return "#21295C"   # navy  — Cycle only
        if label == "dr_mlp":       return "#D4860A"   # amber
        return "#9AB8C8"                               # grey  — specialists

    colors = [bar_color(m) for m in models]

    for df, metric, fname in [
        (df_soc,   "MAE SoC (%)",  "bar_soc.png"),
        (df_cycle, "MAE Cycle",    "bar_cycle.png"),
    ]:
        fig, ax = plt.subplots(figsize=(max(12, len(models) * 0.7), 5))

        for k, (split_cols, split_label, offset) in enumerate([
            (train_cols, "Train", -width),
            (val_cols,   "Val",    0),
            (test_cols,  "Test",   width),
        ]):
            if not split_cols:
                continue
            avgs = df[split_cols].mean(axis=1).values
            bars = ax.bar(x + offset, avgs, width,
                          label=split_label,
                          color=[c for c in colors],
                          alpha=[1.0, 0.6, 0.35][k],
                          edgecolor="white", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(metric)
        ax.set_title(f"Average {metric} by model and split "
                     f"(grey=specialist, orange=DR-MLP, blue=ANP)")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓  {fname}")


# ==============================================================================
# MAIN
# ==============================================================================

def run(
    mlp_run_dir:     Path,
    anp_run_dir:     Optional[Path],
    anp_soc_run_dir: Optional[Path],
    anp_cycle_run_dir: Optional[Path],
    anp_soc_reduced_run_dir: Optional[Path],
    anp_cycle_reduced_run_dir: Optional[Path],
    data_dir:        str,
    out_dir:         Path,
    train_task_ids:  List[int],
    val_task_ids:    List[int],
    test_task_ids:   List[int],
    ctx_cycles:      int,
    tgt_cycles:      int,
    meas_per_cycle:  int,
) -> None:
    """
    Full validation pipeline.

    Evaluates all MLP and ANP variants on every synthetic task and produces comparison tables, heatmaps, and bar charts.

    Args:
        mlp_run_dir:       Path to runs_mlp/<timestamp>/ directory.
        anp_run_dir:       Path to dual-target ANP run (None to skip).
        anp_soc_run_dir:   Path to SoC-only ANP run   (None to skip).
        anp_cycle_run_dir: Path to Cycle-only ANP run  (None to skip).
        data_dir:          Path to the directory containing prepared_data.pkl.
        out_dir:           Output directory for validation results.
        train_task_ids:    0-based dataset indices used as training tasks.
        val_task_ids:      0-based dataset indices used as val tasks.
        test_task_ids:     0-based dataset indices used as test tasks.
        ctx_cycles:        Context window size in cycles.
        tgt_cycles:        Target  window size in cycles.
        meas_per_cycle:    Measurements per cycle in the dataset.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx_rows = ctx_cycles * meas_per_cycle
    tgt_rows = tgt_cycles * meas_per_cycle

    print(f"\n🔧  Device  : {device}")
    print(f"📁  Out dir : {out_dir}")
    print(f"   ctx_rows = {ctx_rows}  ({ctx_cycles} cycles × {meas_per_cycle})")
    print(f"   tgt_rows = {tgt_rows}  ({tgt_cycles} cycles × {meas_per_cycle})")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\n📂  Loading data from: {data_dir}")
    data = load_prepared_data(data_dir)
    validate_targets(data)

    target_cols = list(data["normalized_synth_datasets"][0][1].columns)
    input_dim   = data["normalized_synth_datasets"][0][0].shape[1]
    output_dim  = len(target_cols)
    denorm_values = {
        "y_mean": data["denorm_values"]["y_mean"],
        "y_std":  data["denorm_values"]["y_std"],
    }
    print(f"   input_dim={input_dim}  output_dim={output_dim}  targets={target_cols}")

    x_col_names = list(data["normalized_synth_datasets"][0][0].columns)
    # All 25 tasks evaluated
    all_task_ids = list(range(len(data["normalized_synth_datasets"])))
    all_tasks    = [
        sort_task_by_cycle(*data["normalized_synth_datasets"][i])
        for i in all_task_ids
    ]

    # Task labels
    def task_label(i: int) -> str:
        if i in train_task_ids: return f"train_{train_task_ids.index(i)+1:02d}"
        if i in val_task_ids:   return f"val_{val_task_ids.index(i)+1:02d}"
        if i in test_task_ids:  return f"test_{test_task_ids.index(i)+1:02d}"
        return f"task_{i:02d}"

    task_labels = [task_label(i) for i in all_task_ids]

    # Pre-extract windows for all tasks
    windows = [
        extract_window(X, y, ctx_rows, tgt_rows)
        for X, y in all_tasks
    ]

    # ── Load models ───────────────────────────────────────────────────────────
    print("\n📦  Loading models...")

    # Each entry: (label, model, type, model_target_cols)
    # type ∈ {"mlp", "anp"}
    # model_target_cols: which targets the model predicts
    all_models: List[Tuple[str, nn.Module, str, List[str], Optional[List[str]]]] = []

    # MLP Specialists
    specialists = load_mlp_specialists(
        mlp_run_dir, input_dim, output_dim, device,
        n_specialists=len(train_task_ids),
    )
    for label, model in specialists:
        all_models.append((label, model, "mlp", target_cols, None))

    # DR-MLP
    dr = load_dr_mlp(mlp_run_dir, input_dim, output_dim, device)
    if dr:
        all_models.append((dr[0], dr[1], "mlp", target_cols, None))

    # ANP dual-target
    if anp_run_dir is not None:
        anp = load_anp_model(anp_run_dir, input_dim, target_cols, device, label="anp")
        if anp:
            all_models.append((anp[0], anp[1], "anp", anp[2], anp[3]))

    # ANP SoC-only
    if anp_soc_run_dir is not None:
        anp_soc = load_anp_model(
            anp_soc_run_dir, input_dim, target_cols, device, label="anp_soc"
        )
        if anp_soc:
            all_models.append((anp_soc[0], anp_soc[1], "anp", anp_soc[2], anp_soc[3]))


    # ANP Cycle-only
    if anp_cycle_run_dir is not None:
        anp_cyc = load_anp_model(
            anp_cycle_run_dir, input_dim, target_cols, device, label="anp_cycle"
        )
        if anp_cyc:
            all_models.append((anp_cyc[0], anp_cyc[1], "anp", anp_cyc[2], anp_cyc[3]))
    
    # ANP SoC-only — reduced features
    if anp_soc_reduced_run_dir is not None:
        anp_soc_r = load_anp_model(
            anp_soc_reduced_run_dir, input_dim, target_cols,
            device, label="anp_soc_reduced"
        )
        if anp_soc_r:
            all_models.append((anp_soc_r[0], anp_soc_r[1], "anp",
                                anp_soc_r[2], anp_soc_r[3]))

    # ANP Cycle-only — reduced features
    if anp_cycle_reduced_run_dir is not None:
        anp_cyc_r = load_anp_model(
            anp_cycle_reduced_run_dir, input_dim, target_cols,
            device, label="anp_cycle_reduced"
        )
        if anp_cyc_r:
            all_models.append((anp_cyc_r[0], anp_cyc_r[1], "anp",
                                anp_cyc_r[2], anp_cyc_r[3]))

    print(f"\n  Total models loaded: {len(all_models)}")

    # ── Evaluate all models on all tasks ──────────────────────────────────────
    print("\n📊  Evaluating...\n")

    # results[model_label][task_label] = {col: mae}
    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for m_label, model, m_type, m_target_cols, feat_cols in all_models:
        feat_idx = get_feature_indices(x_col_names, feat_cols)
        results[m_label] = {}
        for t_label, (X_ctx, y_ctx, X_tgt, y_tgt) in zip(task_labels, windows):
            if m_type == "mlp":
                mae = eval_mlp(model, X_tgt, y_tgt, device, denorm_values, target_cols)
            else:
                mae = eval_anp(
                    model,
                    filter_x(X_ctx, feat_idx),   # ← filtrado aquí
                    y_ctx,
                    filter_x(X_tgt, feat_idx),   # ← filtrado aquí
                    y_tgt,
                    device, denorm_values, target_cols, m_target_cols
                )
            results[m_label][t_label] = mae

        # Progress summary (NaN-safe)
        avg_soc = np.nanmean([
            results[m_label][t].get("SoC (%)", float("nan"))
            for t in task_labels
        ])
        avg_cyc = np.nanmean([
            results[m_label][t].get("Cycle", float("nan"))
            for t in task_labels
        ])
        soc_str = f"{avg_soc:.3f}%" if not np.isnan(avg_soc) else "  N/A  "
        cyc_str = f"{avg_cyc:.2f}"  if not np.isnan(avg_cyc) else "  N/A  "
        print(f"  {m_label:<22}  avg SoC MAE={soc_str}  avg Cycle MAE={cyc_str}")

    # ── Build DataFrames ──────────────────────────────────────────────────────
    model_labels_ordered = [m for m, _, _, _, _ in all_models]

    soc_col   = "SoC (%)" if "SoC (%)" in target_cols else target_cols[0]
    cycle_col = "Cycle"   if "Cycle"   in target_cols else target_cols[-1]

    def build_df(col: str) -> pd.DataFrame:
        rows = {}
        for m_label in model_labels_ordered:
            rows[m_label] = {
                t: results[m_label].get(t, {}).get(col, float("nan"))
                for t in task_labels
            }
        return pd.DataFrame(rows).T  # (models, tasks)

    df_soc   = build_df(soc_col)
    df_cycle = build_df(cycle_col)

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    def safe(s): return s.replace(" ","_").replace("(","").replace(")","").replace("%","pct")

    df_soc.to_csv(out_dir / f"mae_{safe(soc_col)}.csv")
    df_cycle.to_csv(out_dir / f"mae_{safe(cycle_col)}.csv")
    print(f"\n  ✓  mae_{safe(soc_col)}.csv")
    print(f"  ✓  mae_{safe(cycle_col)}.csv")

    # Combined wide-format CSV
    combined_rows = []
    for m_label in model_labels_ordered:
        row = {"model": m_label}
        for t_label in task_labels:
            for col in target_cols:
                row[f"{t_label}/mae_{safe(col)}"] = \
                    results[m_label].get(t_label, {}).get(col, float("nan"))
        combined_rows.append(row)
    pd.DataFrame(combined_rows).set_index("model").to_csv(
        out_dir / "mae_comparison.csv"
    )
    print(f"  ✓  mae_comparison.csv")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n📈  Saving plots...")
    plot_heatmaps(df_soc, df_cycle, out_dir)
    plot_bar_comparison(df_soc, df_cycle, out_dir)

    # ── Summary table ─────────────────────────────────────────────────────────
    train_labels = [t for t in task_labels if t.startswith("train_")]
    val_labels   = [t for t in task_labels if t.startswith("val_")]
    test_labels  = [t for t in task_labels if t.startswith("test_")]

    lines = []
    lines.append("=" * 95)
    lines.append("VALIDATION SUMMARY — average MAE per split (original units)  |  NaN = target not predicted")
    lines.append("=" * 95)
    lines.append(
        f"\n{'Model':<22} {'Train SoC':>10} {'Train Cyc':>10} "
        f"{'Val SoC':>9} {'Val Cyc':>9} {'Test SoC':>10} {'Test Cyc':>10}"
    )
    lines.append("-" * 95)

    def avg(m_label, labels, col):
        vals = [results[m_label].get(t, {}).get(col, float("nan")) for t in labels]
        vals = [v for v in vals if not np.isnan(v)]
        return np.mean(vals) if vals else float("nan")

    def fmt(v):
        return f"{v:>10.3f}" if not np.isnan(v) else f"{'N/A':>10}"

    for m_label in model_labels_ordered:
        anp_tag = ""
        if m_label == "anp":       anp_tag = "  ← ANP dual"
        elif m_label == "anp_soc": anp_tag = "  ← ANP SoC-only"
        elif m_label == "anp_cycle": anp_tag = "  ← ANP Cycle-only"
        lines.append(
            f"{m_label:<22}"
            f" {fmt(avg(m_label, train_labels, soc_col))}"
            f" {fmt(avg(m_label, train_labels, cycle_col))}"
            f" {fmt(avg(m_label, val_labels, soc_col))}"
            f" {fmt(avg(m_label, val_labels, cycle_col))}"
            f" {fmt(avg(m_label, test_labels, soc_col))}"
            f" {fmt(avg(m_label, test_labels, cycle_col))}"
            f"{anp_tag}"
        )

    # Aggregate rows
    lines.append("-" * 95)
    specialist_labels = [m for m in model_labels_ordered if m.startswith("specialist_")]

    def group_avg(model_list, split_labels, col):
        vals = []
        for m in model_list:
            for t in split_labels:
                v = results[m].get(t, {}).get(col, float("nan"))
                if not np.isnan(v):
                    vals.append(v)
        return np.mean(vals) if vals else float("nan")

    for group_label, group_models in [
        ("AVG Specialists",   specialist_labels),
        ("DR-MLP",            ["dr_mlp"]    if "dr_mlp"    in model_labels_ordered else []),
        ("ANP (dual)",        ["anp"]       if "anp"       in model_labels_ordered else []),
        ("ANP (SoC-only)",    ["anp_soc"]   if "anp_soc"   in model_labels_ordered else []),
        ("ANP (Cycle-only)",  ["anp_cycle"] if "anp_cycle" in model_labels_ordered else []),
    ]:
        if not group_models:
            continue
        lines.append(
            f"{group_label:<22}"
            f" {fmt(group_avg(group_models, train_labels, soc_col))}"
            f" {fmt(group_avg(group_models, train_labels, cycle_col))}"
            f" {fmt(group_avg(group_models, val_labels, soc_col))}"
            f" {fmt(group_avg(group_models, val_labels, cycle_col))}"
            f" {fmt(group_avg(group_models, test_labels, soc_col))}"
            f" {fmt(group_avg(group_models, test_labels, cycle_col))}"
        )

    summary = "\n".join(lines)
    print("\n" + summary)

    with open(out_dir / "summary.txt", "w") as f:
        f.write(summary)
    print(f"\n  ✓  summary.txt")
    print(f"\n✅  Validation complete. All outputs in: {out_dir}\n")


# ==============================================================================
# CLI
# ==============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified validation — Specialist MLPs, DR-MLP and ANP variants",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mlp_run", type=str, required=True, help="Path to the runs_mlp/<timestamp>/ directory")
    p.add_argument("--anp_run", type=str, default=None, help="Path to the dual-target ANP run directory (optional)")
    p.add_argument("--anp_soc_run", type=str, default=None, help="Path to the SoC-only ANP run directory (optional)")
    p.add_argument("--anp_cycle_run", type=str, default=None, help="Path to the Cycle-only ANP run directory (optional)")
    p.add_argument("--data_dir", type=str, default="../csic_real_synth_load/prepared_data")
    p.add_argument("--out_dir",  type=str, default="", help="Output directory (default: ./validation/results/)")

    # Evaluation window
    p.add_argument("--ctx_cycles",     type=int, default=60)
    p.add_argument("--tgt_cycles",     type=int, default=60)
    p.add_argument("--meas_per_cycle", type=int, default=30)

    # Task split (must match the training run)
    p.add_argument("--train_ids", type=int, nargs="+", default=list(range(17)))
    p.add_argument("--val_ids",   type=int, nargs="+", default=list(range(17, 22)))
    p.add_argument("--test_ids",  type=int, nargs="+", default=list(range(22, 25)))

    p.add_argument("--anp_soc_reduced_run",   type=str, default=None, help="Path to SoC-only ANP run trained with reduced features")
    p.add_argument("--anp_cycle_reduced_run", type=str, default=None, help="Path to Cycle-only ANP run trained with reduced features")

    return p.parse_args()


def main() -> None:
    args    = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(__file__).resolve().parent / "results")

    run(
        mlp_run_dir       = Path(args.mlp_run),
        anp_run_dir       = Path(args.anp_run)       if args.anp_run       else None,
        anp_soc_run_dir   = Path(args.anp_soc_run)   if args.anp_soc_run   else None,
        anp_cycle_run_dir = Path(args.anp_cycle_run) if args.anp_cycle_run else None,
        anp_soc_reduced_run_dir   = Path(args.anp_soc_reduced_run)   if args.anp_soc_reduced_run   else None,
        anp_cycle_reduced_run_dir = Path(args.anp_cycle_reduced_run) if args.anp_cycle_reduced_run else None,
        data_dir          = args.data_dir,
        out_dir           = out_dir,
        train_task_ids    = args.train_ids,
        val_task_ids      = args.val_ids,
        test_task_ids     = args.test_ids,
        ctx_cycles        = args.ctx_cycles,
        tgt_cycles        = args.tgt_cycles,
        meas_per_cycle    = args.meas_per_cycle,
    )

if __name__ == "__main__":
    main()