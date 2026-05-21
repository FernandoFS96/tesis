"""
train_utils.py
==============================================================================
Utility functions for ANP battery training pipeline.

Covers:
    - Data loading and validation
    - Task splitting
    - Episode and batch construction (cycle-based, professor's approach)
    - Denormalized MAE metric
    - Evaluation loop
    - Training curve plotting (called automatically at end of training)

All functions are stateless and import-safe: no side effects on import.
==============================================================================
"""

from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for remote servers
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_prepared_data(data_dir: str) -> dict:
    """
    Load the prepared_data.pkl file, searching candidate paths relative to
    both the current working directory and the script's parent directories.

    Args:
        data_dir: Path (absolute or relative) to the directory containing
                  prepared_data.pkl.

    Returns:
        Deserialized data dictionary produced by load.py.

    Raises:
        FileNotFoundError: if prepared_data.pkl is not found in any candidate.
        RuntimeError: if the pkl cannot be deserialized (version mismatch).
    """
    data_dir_path = Path(data_dir).expanduser()
    script_dir    = Path(__file__).resolve().parent
    csic_root     = script_dir.parent

    if data_dir_path.is_absolute():
        candidates = [data_dir_path]
    else:
        candidates = [
            Path.cwd() / data_dir_path,
            script_dir  / data_dir_path,
            csic_root   / data_dir_path,
            csic_root   / "csic_real_synth_load" / "prepared_data",
        ]

    # Deduplicate while preserving order
    seen, unique = set(), []
    for c in candidates:
        r = c.resolve()
        if r not in seen:
            seen.add(r)
            unique.append(r)

    pkl_path = None
    for c in unique:
        trial = c / "prepared_data.pkl"
        if trial.exists():
            pkl_path = trial
            break

    if pkl_path is None:
        tried = "\n".join(f"  - {p / 'prepared_data.pkl'}" for p in unique)
        raise FileNotFoundError(
            f"prepared_data.pkl not found in any candidate path:\n{tried}"
        )

    print(f"✓ Loading PKL: {pkl_path}")
    with pkl_path.open("rb") as f:
        try:
            return pickle.load(f)
        except AttributeError as exc:
            raise RuntimeError(
                "Cannot deserialize prepared_data.pkl — likely a pandas version "
                "mismatch. Re-generate the file by running load.py in the same "
                "environment used to create it."
            ) from exc


def validate_targets(
    data: dict,
    expected_targets: Tuple[str, ...] = ("SoC (%)", "Cycle"),
) -> None:
    """
    Assert that the pkl contains every expected target column.

    Args:
        data: Dictionary returned by load_prepared_data.
        expected_targets: Tuple of target column names that must be present.

    Raises:
        ValueError: with a human-readable fix suggestion if targets are missing.
    """
    actual  = list(data["normalized_synth_datasets"][0][1].columns)
    missing = [t for t in expected_targets if t not in actual]
    if missing:
        raise ValueError(
            f"\n{'='*60}\n"
            f"  MISSING TARGETS IN PKL: {missing}\n"
            f"  Current targets:        {actual}\n\n"
            f"  FIX: edit load.py  →  'targets': {list(expected_targets)}\n"
            f"  Then re-run:  python load.py\n"
            f"{'='*60}\n"
        )
    print(f"✓ Targets verified: {actual}")


# ==============================================================================
# TASK SPLITTING
# ==============================================================================

def get_task_splits(
    data: dict,
    train_ids: List[int],
    val_ids:   List[int],
    test_ids:  List[int],
) -> Tuple[list, list, list]:
    """
    Partition the synthetic datasets into train / val / test task lists.

    Each task is a (X_df, y_df) tuple of DataFrames.

    Args:
        data:      Dictionary returned by load_prepared_data.
        train_ids: 0-based indices of datasets used for training.
        val_ids:   0-based indices of datasets used for validation.
        test_ids:  0-based indices of datasets used for testing.

    Returns:
        (train_tasks, val_tasks, test_tasks) — lists of (X_df, y_df) tuples.

    Raises:
        IndexError: if any id is out of range.
    """
    synth = data["normalized_synth_datasets"]
    n     = len(synth)

    def _get(ids: List[int]) -> list:
        out = []
        for i in ids:
            if i >= n:
                raise IndexError(
                    f"task_id={i} out of range — only {n} synthetic datasets available."
                )
            out.append(synth[i])
        return out

    train_tasks = _get(train_ids)
    val_tasks   = _get(val_ids)
    test_tasks  = _get(test_ids)

    print(f"✓ Task split — train: {len(train_tasks)}  "
          f"val: {len(val_tasks)}  test: {len(test_tasks)}")
    return train_tasks, val_tasks, test_tasks


# ==============================================================================
# EPISODE AND BATCH CONSTRUCTION
# ==============================================================================

def sort_task_by_cycle(
    X: pd.DataFrame,
    y: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sort task rows in ascending cycle order (stable sort preserves intra-cycle
    measurement order for equal cycle values).

    Args:
        X: Feature DataFrame (T, D).
        y: Target DataFrame  (T, O).

    Returns:
        (X_sorted, y_sorted) with reset integer index.
    """
    if "Cycle" in X.columns:
        order = X["Cycle"].argsort(kind="stable")
        return (
            X.iloc[order].reset_index(drop=True),
            y.iloc[order].reset_index(drop=True),
        )
    return X.reset_index(drop=True), y.reset_index(drop=True)


def make_episode_fixed(
    X:               pd.DataFrame,
    y:               pd.DataFrame,
    ctx_rows:        int,
    tgt_rows:        int,
    device:          torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a single ANP episode using a fixed number of context and target rows.

    Following the professor's approach:
      - Context  = the first `ctx_rows` rows of the sorted trajectory
                   (e.g. first 50 cycles × 30 measurements = 1 500 rows).
      - Target   = the next `tgt_rows` rows immediately after the context
                   (e.g. cycles 51-100 → rows 1 500 – 2 999).

    Both windows are taken as contiguous slices — no random subsampling —
    so the model always sees exactly the same temporal prefix and predicts
    the immediately following segment.

    Args:
        X:         Feature DataFrame, already sorted by cycle (T, D).
        y:         Target  DataFrame, already sorted by cycle (T, O).
        ctx_rows:  Number of rows to use as context.
        tgt_rows:  Number of rows to use as target.
        device:    Torch device for the returned tensors.

    Returns:
        ctx_x  (ctx_rows, D)
        ctx_y  (ctx_rows, O)
        tgt_x  (tgt_rows, D)
        tgt_y  (tgt_rows, O)

    Note:
        Tensors are returned WITHOUT a batch dimension so they can be stacked
        by make_batch().
    """
    T = len(X)

    ctx_end = min(ctx_rows, T)
    tgt_end = min(ctx_end + tgt_rows, T)

    ctx_idx = np.arange(ctx_end)
    tgt_idx = np.arange(ctx_end, tgt_end)

    X_t = torch.tensor(X.values, dtype=torch.float32)
    y_t = torch.tensor(y.values, dtype=torch.float32)

    return (
        X_t[ctx_idx],   # (ctx_rows, D)
        y_t[ctx_idx],   # (ctx_rows, O)
        X_t[tgt_idx],   # (tgt_rows, D)
        y_t[tgt_idx],   # (tgt_rows, O)
    )


def make_batch(
    tasks:      list,
    batch_size: int,
    ctx_rows:   int,
    tgt_rows:   int,
    device:     torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a training batch by stacking `batch_size` episodes along dim 0.

    Tasks are sampled with replacement (valid in meta-learning; allows any batch_size regardless of the number of available tasks).

    Args:
        tasks:      List of (X_df, y_df) tuples, already cycle-sorted.
        batch_size: Number of episodes to stack.
        ctx_rows:   Context window size in rows (passed to make_episode_fixed).
        tgt_rows:   Target  window size in rows (passed to make_episode_fixed).
        device:     Target device for the output tensors.

    Returns:
        context_x  (B, ctx_rows, D)
        context_y  (B, ctx_rows, O)
        target_x   (B, tgt_rows, D)
        target_y   (B, tgt_rows, O)
    """
    cx_list, cy_list, tx_list, ty_list = [], [], [], []

    if batch_size <= len(tasks):
        chosen = random.sample(tasks, k=batch_size)  # sin reemplazo
    else:
        chosen = random.choices(tasks, k=batch_size)  # con reemplazo si necesario

    for X, y in chosen:
        cx, cy, tx, ty = make_episode_fixed(X, y, ctx_rows, tgt_rows, device)
        cx_list.append(cx)
        cy_list.append(cy)
        tx_list.append(tx)
        ty_list.append(ty)

    return (
        torch.stack(cx_list).to(device),
        torch.stack(cy_list).to(device),
        torch.stack(tx_list).to(device),
        torch.stack(ty_list).to(device),
    )


# ==============================================================================
# METRICS
# ==============================================================================

def denorm_mae(
    pred_mean:    torch.Tensor,
    target_y:     torch.Tensor,
    denorm_values: dict,
    target_cols:   List[str],
) -> Dict[str, float]:
    """
    Compute denormalized Mean Absolute Error for every target column.

    Applies the inverse StandardScaler transform:
        original = normalized * std + mean

    Args:
        pred_mean:     Model predictions,  shape (B, Nt, O).
        target_y:      Ground truth values, shape (B, Nt, O).
        denorm_values: Dict with keys 'y_mean' and 'y_std' (scalars per column).
        target_cols:   Ordered list of target column names, length O.

    Returns:
        Dict mapping each column name to its scalar MAE in original units.
    """
    result = {}
    for i, col in enumerate(target_cols):
        mean_val = denorm_values["y_mean"].get(col, 0.0)
        std_val  = denorm_values["y_std"].get(col, 1.0)
        pred_dn  = pred_mean[:, :, i].detach().cpu() * std_val + mean_val  # (B, Nt)
        true_dn  = target_y[:, :, i].detach().cpu() * std_val + mean_val  # (B, Nt)
        result[col] = (pred_dn - true_dn).abs().mean().item()
    return result


# ==============================================================================
# EVALUATION LOOP
# ==============================================================================

@torch.no_grad()
def evaluate(
    model:          nn.Module,
    tasks:          list,
    ctx_rows:       int,
    tgt_rows:       int,
    device:         torch.device,
    denorm_values:  dict,
    target_cols:    List[str],
    beta:           float = 1.0,
    split_name:     str   = "val",
) -> dict:
    """
    Evaluate the model on a set of tasks using fixed context/target windows.

    All tasks are assembled into a single batch for efficiency.  The context
    is always the first `ctx_rows` rows; the target is the following `tgt_rows`
    rows — identical to the training setup, but deterministic.

    Args:
        model:         ANP LatentModel in eval mode after this call.
        tasks:         List of (X_df, y_df) tuples, cycle-sorted.
        ctx_rows:      Number of context rows (same as training).
        tgt_rows:      Number of target rows  (same as training).
        device:        Torch device.
        denorm_values: Dict with 'y_mean' and 'y_std' for MAE computation.
        target_cols:   Ordered list of target column names.
        beta:          KL weight used in the ELBO loss.
        split_name:    Prefix for returned metric keys ('val' or 'test').

    Returns:
        Dict of {f'{split_name}/{metric_name}': float} entries covering
        loss, NLL, KL, and denormalized MAE per target column.
    """
    model.eval()

    cx_list, cy_list, tx_list, ty_list = [], [], [], []
    for X, y in tasks:
        cx, cy, tx, ty = make_episode_fixed(X, y, ctx_rows, tgt_rows, device)
        cx_list.append(cx)
        cy_list.append(cy)
        tx_list.append(tx)
        ty_list.append(ty)

    ctx_x = torch.stack(cx_list).to(device)
    ctx_y = torch.stack(cy_list).to(device)
    tgt_x = torch.stack(tx_list).to(device)
    tgt_y = torch.stack(ty_list).to(device)

    pred_mean, pred_var, loss, kl, nll = model(
        ctx_x, ctx_y, tgt_x, tgt_y, beta=beta
    )

    mae = denorm_mae(pred_mean, tgt_y, denorm_values, target_cols)

    results: dict = {
        f"{split_name}/loss": loss.item(),
        f"{split_name}/nll":  nll.item(),
        f"{split_name}/kl":   kl.item(),
    }
    for col, val in mae.items():
        safe = (col.replace(" ", "_").replace("(", "")
                   .replace(")", "").replace("%", "pct"))
        results[f"{split_name}/mae_{safe}"] = val

    model.train()
    return results


# ==============================================================================
# PLOTTING
# ==============================================================================

# Plot styling constants
_STYLE = {
    "train_color": "#1C7293",
    "val_color":   "#C0392B",
    "kl_color":    "#D4860A",
    "nll_color":   "#237A3D",
    "lr_color":    "#9AB8C8",
    "alpha":       0.85,
    "linewidth":   1.8,
    "fig_dpi":     150,
}


def _save(fig: Figure, path: Path, tight: bool = True) -> None:
    """Save figure and close it."""
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=_STYLE["fig_dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}")


def plot_training_curves(
    metrics_df: pd.DataFrame,
    plots_dir:  Path,
    target_cols: List[str],
) -> None:
    """
    Generate and save all training diagnostic plots.

    Plots produced:
        01_loss_curves.png       — train/val ELBO loss over epochs
        02_nll_kl_curves.png     — NLL and KL components separately
        03_lr_schedule.png       — learning rate schedule
        04_mae_soc.png           — validation MAE for SoC (%) over epochs
        05_mae_cycle.png         — validation MAE for Cycle over epochs
        06_overview.png          — 2×3 summary figure of all key metrics

    Args:
        metrics_df:  DataFrame with one row per epoch (loaded from metrics.csv).
        plots_dir:   Directory where PNG files will be saved (created if absent).
        target_cols: Ordered list of target column names (e.g. ['SoC (%)', 'Cycle']).
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📈  Saving training plots → {plots_dir}")

    epochs = metrics_df["epoch"].values

    # ── Helper: safe column access ────────────────────────────────────────────
    def col(name: str) -> Optional[np.ndarray]:
        if name in metrics_df.columns:
            # Use to_numpy() to ensure a plain numpy ndarray is returned
            return metrics_df[name].to_numpy()
        return None

    def safe_col_name(target: str) -> str:
        return (target.replace(" ", "_").replace("(", "")
                      .replace(")", "").replace("%", "pct"))

    # ── 1. Loss curves ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    train_loss = col("train/loss")
    val_loss   = col("val/loss")

    if train_loss is not None:
        ax.plot(epochs, train_loss, label="Train loss",
                color=_STYLE["train_color"],
                linewidth=_STYLE["linewidth"], alpha=_STYLE["alpha"])
    if val_loss is not None:
        ax.plot(epochs, val_loss, label="Val loss",
                color=_STYLE["val_color"],
                linewidth=_STYLE["linewidth"], alpha=_STYLE["alpha"])

    ax.set_xlabel("Epoch")
    ax.set_ylabel("ELBO Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, plots_dir / "01_loss_curves.png")

    # ── 2. NLL and KL components ──────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for ax, key, label, color in [
        (ax1, "train/nll", "NLL",          _STYLE["nll_color"]),
        (ax2, "train/kl",  "KL divergence", _STYLE["kl_color"]),
    ]:
        values = col(key)
        if values is not None:
            ax.plot(epochs, values, color=color,
                    linewidth=_STYLE["linewidth"], alpha=_STYLE["alpha"],
                    label=f"Train {label}")
        val_key = key.replace("train/", "val/")
        val_values = col(val_key)
        if val_values is not None:
            ax.plot(epochs, val_values, color=_STYLE["val_color"],
                    linewidth=_STYLE["linewidth"], alpha=_STYLE["alpha"],
                    linestyle="--", label=f"Val {label}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(f"{label} over Training")
        ax.legend()
        ax.grid(True, alpha=0.3)

    _save(fig, plots_dir / "02_nll_kl_curves.png")

    # ── 3. Learning rate schedule ─────────────────────────────────────────────
    lr_vals = col("lr")
    if lr_vals is not None:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(epochs, lr_vals, color=_STYLE["lr_color"],
                linewidth=_STYLE["linewidth"], alpha=_STYLE["alpha"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule (Cosine Annealing)")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        _save(fig, plots_dir / "03_lr_schedule.png")

    # ── 4 & 5. MAE per target ─────────────────────────────────────────────────
    for plot_idx, target in enumerate(target_cols, start=4):
        safe   = safe_col_name(target)
        val_key = f"val/mae_{safe}"
        mae_vals = col(val_key)
        if mae_vals is None:
            continue

        fig, ax = plt.subplots(figsize=(10, 4))
        # Filter out NaN values that appear in epochs before first validation
        mask = ~np.isnan(mae_vals)
        ax.plot(epochs[mask], mae_vals[mask],
                color=_STYLE["val_color"],
                linewidth=_STYLE["linewidth"], alpha=_STYLE["alpha"],
                label=f"Val MAE — {target}")

        # Annotate best value
        best_val = np.nanmin(mae_vals)
        best_ep  = epochs[np.nanargmin(mae_vals)]
        ax.axhline(best_val, color="gray", linestyle=":", linewidth=1.0,
                   label=f"Best = {best_val:.4f} (epoch {best_ep})")

        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"MAE [{target}]")
        ax.set_title(f"Validation MAE — {target}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        _save(fig, plots_dir / f"0{plot_idx}_mae_{safe}.png")

    # ── 6. Overview figure (2 × 3 grid) ──────────────────────────────────────
    fig = plt.figure(figsize=(16, 8))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    panels = [
        # (subplot position, x_data, [(y_data, label, color, linestyle)])
        (gs[0, 0], epochs, [
            (col("train/loss"), "Train",  _STYLE["train_color"], "-"),
            (col("val/loss"),   "Val",    _STYLE["val_color"],   "-"),
        ], "ELBO Loss", "Loss"),
        (gs[0, 1], epochs, [
            (col("train/nll"), "Train NLL", _STYLE["nll_color"], "-"),
            (col("val/nll"),   "Val NLL",   _STYLE["val_color"], "--"),
        ], "Negative Log-Likelihood", "NLL"),
        (gs[0, 2], epochs, [
            (col("train/kl"), "KL", _STYLE["kl_color"], "-"),
        ], "KL Divergence", "KL"),
        (gs[1, 2], epochs, [
            (col("lr"), "LR", _STYLE["lr_color"], "-"),
        ], "Learning Rate", "LR (log scale)"),
    ]

    for i, (pos, x, lines, title, ylabel) in enumerate(panels):
        ax = fig.add_subplot(pos)
        for y_vals, label, color, ls in lines:
            if y_vals is not None:
                mask = ~np.isnan(y_vals.astype(float))
                ax.plot(x[mask], y_vals[mask], label=label,
                        color=color, linestyle=ls,
                        linewidth=1.5, alpha=_STYLE["alpha"])
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
        if "LR" in ylabel:
            ax.set_yscale("log")

    # MAE panels
    for j, target in enumerate(target_cols[:2]):
        ax = fig.add_subplot(gs[1, j])
        safe = safe_col_name(target)
        mae_vals = col(f"val/mae_{safe}")
        if mae_vals is not None:
            mask = ~np.isnan(mae_vals.astype(float))
            ax.plot(epochs[mask], mae_vals[mask],
                    color=_STYLE["val_color"], linewidth=1.5,
                    alpha=_STYLE["alpha"], label=f"Val MAE")
            best = np.nanmin(mae_vals)
            ax.axhline(best, color="gray", linestyle=":", linewidth=1.0,
                       label=f"Best = {best:.4f}")
        ax.set_title(f"MAE — {target}", fontsize=10, fontweight="bold")
        ax.set_ylabel(f"MAE [{target}]", fontsize=9)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    fig.suptitle("Training Overview", fontsize=14, fontweight="bold", y=1.01)
    _save(fig, plots_dir / "06_overview.png", tight=False)
    print(f"  ✓  All plots saved to: {plots_dir}\n")


def generate_all_plots(
    run_dir:     Path,
    target_cols: List[str],
) -> None:
    """
    Entry point called at the end of training.

    Loads metrics.csv from run_dir, generates all plots, and saves them
    to run_dir/plots/.

    Args:
        run_dir:     Path to the run directory containing metrics.csv.
        target_cols: Ordered list of target column names.
    """
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        print(f"⚠  metrics.csv not found in {run_dir} — skipping plots.")
        return

    metrics_df = pd.read_csv(metrics_path)
    plots_dir  = run_dir / "plots"
    plot_training_curves(metrics_df, plots_dir, target_cols)

# ==============================================================================
# FEATURE REDUCTION — RF-based compact feature sets
# (from Experiment 6, V3 synthetic datasets)
# ==============================================================================

# Feature sets identified by Random Forest feature importance (MRR ranking).
# Keys match the target_col values used in Config.
REDUCED_FEATURE_SETS: dict = {
    "SoC (%)": [
        "Potential",
        "Phase_45", "Phase_46", "Phase_47", "Phase_48", "Phase_49",  # very-low freq
        "Phase_3",  "Phase_4",  "Phase_5",  "Phase_6",  "Phase_7",   # high freq
    ],
    "Cycle": [
        "Potential",
        "Phase_30", "Phase_31", "Phase_27", "Phase_28", "Phase_26",  # mid freq
        "Zmag_35",  "Zmag_36",  "Zmag_37",  "Zmag_40",               # low freq magnitude
        "Zim_33",   "Zre_38",   "Zre_36",                            # low freq cartesian
    ],
    "all": None,  # no filtering — use all 201 features
}

def apply_feature_reduction(
    data:       dict,
    target_col: str,
) -> dict:
    """
    Filter X features in the prepared dataset to the compact set identified by the RF feature importance study (Experiment 6, V3 synthetic datasets).

    Filtering is only applied when target_col is 'SoC (%)' or 'Cycle'. For target_col='all' the data is returned unchanged.

    The function modifies a shallow copy of the data dict, the original is not mutated.

    Args:
        data:       Dictionary returned by load_prepared_data().
        target_col: Which target is being trained. Controls which feature subset is selected ('SoC (%)', 'Cycle', or 'all').

    Returns:
        data dict with X DataFrames filtered to the selected feature columns.
        The 'denorm_values' entry is updated to only contain X stats for the kept columns (y_mean / y_std are not touched).

    Raises:
        ValueError: if any requested feature is not found in the dataset X.
    """
    feature_cols = REDUCED_FEATURE_SETS.get(target_col)

    if feature_cols is None:
        # 'all' or unknown target — no filtering
        return data

    # Validate that all requested features exist in the data
    sample_X = data["normalized_synth_datasets"][0][0]
    missing   = [f for f in feature_cols if f not in sample_X.columns]
    if missing:
        raise ValueError(
            f"\nFeature reduction failed — the following columns were not found in the dataset X:\n  {missing}\n"
            f"Available columns (first 20): {list(sample_X.columns[:20])}\n"
            f"Check that the feature names match exactly (case-sensitive)."
        )

    print(f"\n🔬  Feature reduction active for target '{target_col}'")
    print(f"   Using {len(feature_cols)} / {len(sample_X.columns)} features:")
    print(f"   {feature_cols}")

    # Filter all synthetic datasets
    filtered_synth = [
        (X[feature_cols], y)
        for X, y in data["normalized_synth_datasets"]
    ]

    # Filter real dataset if present
    real_X, real_y = data["normalized_real_dataset"]
    filtered_real  = (real_X[feature_cols], real_y)

    # Update denorm_values for X (only keep stats for selected features)
    orig_x_mean = data["denorm_values"]["y_mean"]  # note: y_mean/y_std are for y
    orig_x_mean_X = data["denorm_values"].get("X_mean", {})
    orig_x_std_X  = data["denorm_values"].get("X_std",  {})
    filtered_x_mean = {k: v for k, v in orig_x_mean_X.items() if k in feature_cols}
    filtered_x_std  = {k: v for k, v in orig_x_std_X.items()  if k in feature_cols}

    # Build updated data dict (shallow copy — y and denorm y are unchanged)
    updated_data = {
        **data,
        "normalized_synth_datasets": filtered_synth,
        "normalized_real_dataset":   filtered_real,
        "denorm_values": {
            **data["denorm_values"],
            "X_mean": filtered_x_mean,
            "X_std":  filtered_x_std,
        },
    }

    return updated_data

def aggregate_by_cycle(
    X: pd.DataFrame,
    y: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate EIS measurements to one representative point per cycle.

    Replaces the ~30 rows per cycle with a single row whose features are the column-wise mean across all measurements in that cycle.
    This eliminates the intra-cycle SoC variation from the input, leaving only the inter-cycle degradation signal in X.

    Target aggregation:
        - Cycle: constant within each group → mean equals cycle number
        - SoC(%): mean SoC of the cycle (represents average charge state)

    Args:
        X: Feature DataFrame (T, D), cycle-sorted, without Cycle column.
        y: Target DataFrame  (T, O), must contain 'Cycle' column.

    Returns:
        X_agg (n_cycles, D) and y_agg (n_cycles, O) — one row per cycle.

    Raises:
        ValueError: if 'Cycle' is not present in y.
    """
    if "Cycle" not in y.columns:
        raise ValueError(
            "aggregate_by_cycle requires 'Cycle' in y. "
            "Ensure target_col includes Cycle or is 'all'."
        )

    cycle_ids = y["Cycle"].values

    # X: mean of all EIS measurements within each cycle
    X_work = X.copy()
    X_work["_cycle_key"] = cycle_ids
    X_agg = (
        X_work.groupby("_cycle_key", sort=True)
              .mean()
              .reset_index(drop=True)
    )

    # y: mean per cycle (Cycle target is constant within group,
    #    SoC target becomes mean SoC of the cycle)
    y_work = y.copy()
    y_work["_cycle_key"] = cycle_ids
    y_agg = (
        y_work.groupby("_cycle_key", sort=True)
              .mean()
              .reset_index(drop=True)
    )

    return X_agg, y_agg

def get_feature_indices(
    x_col_names:  List[str],
    feature_cols: Optional[List[str]],
) -> Optional[List[int]]:
    """
    Return the numpy column indices corresponding to feature_cols.
    Returns None when feature_cols is None (no filtering needed).

    Args:
        x_col_names:  Ordered list of all X column names in the dataset.
        feature_cols: Subset of column names to keep, or None.

    Raises:
        ValueError: if any name in feature_cols is absent from x_col_names.
    """
    if feature_cols is None:
        return None
    missing = [c for c in feature_cols if c not in x_col_names]
    if missing:
        raise ValueError(
            f"Feature columns not found in data X: {missing}\n"
            f"Check that REDUCED_FEATURE_SETS names match the pkl column names."
        )
    return [x_col_names.index(c) for c in feature_cols]


def filter_x(X: np.ndarray, feat_idx: Optional[List[int]]) -> np.ndarray:
    """Return X[:, feat_idx] if feat_idx is not None, else X unchanged."""
    return X if feat_idx is None else X[:, feat_idx]

@torch.no_grad()
def enrich_with_soc_predictions(
    tasks_raw:        List[Tuple[pd.DataFrame, pd.DataFrame]],
    tasks_agg:        List[Tuple[pd.DataFrame, pd.DataFrame]],
    anp_soc_model:    nn.Module,
    soc_feat_cols:    Optional[List[str]],
    device:           torch.device,
    ctx_cycles:       int = 60,
    meas_per_cycle:   int = 30,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Enrich cycle-aggregated tasks with per-cycle SoC statistics predicted
    by a pre-trained ANP-SoC model.

    For each task, the ANP-SoC model predicts SoC for every measurement row
    using the first ctx_cycles cycles as context (prior path — no target_y).
    Four statistics are computed per cycle from those predictions and appended
    as new columns to the cycle-aggregated X DataFrame:

        soc_pred_mean  — mean predicted SoC across the cycle's measurements
        soc_pred_min   — minimum predicted SoC (depth of discharge)
        soc_pred_max   — maximum predicted SoC (top of charge)
        soc_pred_range — soc_pred_max − soc_pred_min (amplitude)

    These features give the ANP-Cycle explicit information about the charge
    regime of each cycle, separating the intra-cycle SoC variation from the
    inter-cycle degradation signal.

    Args:
        tasks_raw:      Measurement-level tasks [(X_df, y_df), ...].
                        X_df: (T, D), y_df: (T, O) with 'Cycle' column.
        tasks_agg:      Cycle-aggregated tasks [(X_agg, y_agg), ...].
                        X_agg: (N_cycles, D), y_agg: (N_cycles, O).
        anp_soc_model:  Loaded ANP-SoC LatentModel in eval mode.
        soc_feat_cols:  X column names used by ANP-SoC (None = all columns).
        device:         Torch device.
        ctx_cycles:     Context window size in cycles.
        meas_per_cycle: Measurements per cycle.

    Returns:
        Enriched cycle-aggregated tasks [(X_enriched, y_agg), ...]
        where X_enriched has 4 extra columns.
    """
    from models.anp import LatentModel   # local import — avoids circular deps

    ctx_rows = ctx_cycles * meas_per_cycle
    enriched = []

    for task_idx, ((X_raw, y_raw), (X_agg, y_agg)) in enumerate(
        zip(tasks_raw, tasks_agg)
    ):
        T = len(X_raw)

        # Context: first ctx_cycles cycles at measurement level
        ctx_end  = min(ctx_rows, T)
        X_ctx_df = X_raw.iloc[:ctx_end]
        y_ctx_df = y_raw.iloc[:ctx_end]

        # Filter X to ANP-SoC feature set if it used reduced features
        if soc_feat_cols is not None:
            X_ctx_np = X_ctx_df[soc_feat_cols].values.astype(np.float32)
            X_all_np = X_raw[soc_feat_cols].values.astype(np.float32)
        else:
            X_ctx_np = X_ctx_df.values.astype(np.float32)
            X_all_np = X_raw.values.astype(np.float32)

        # y_ctx: only SoC column (ANP-SoC is single-target)
        soc_col_name = "SoC (%)"
        y_ctx_np = y_ctx_df[[soc_col_name]].values.astype(np.float32)

        # Run ANP-SoC inference for ALL rows (prior path)
        ctx_x = torch.tensor(X_ctx_np).unsqueeze(0).to(device)
        ctx_y = torch.tensor(y_ctx_np).unsqueeze(0).to(device)
        tgt_x = torch.tensor(X_all_np).unsqueeze(0).to(device)
        soc_mean, _, _, _, _ = anp_soc_model(ctx_x, ctx_y, tgt_x, target_y=None)
        soc_pred = soc_mean.squeeze(0).cpu().numpy()[:, 0]  # (T,)

        # Compute per-cycle statistics
        # Use ground truth Cycle column from y_raw to group rows
        cycle_ids = y_raw["Cycle"].values
        stats_rows = []
        for cyc in sorted(set(cycle_ids)):
            mask = cycle_ids == cyc
            p    = soc_pred[mask]
            stats_rows.append({
                "soc_pred_mean":  float(p.mean()),
                "soc_pred_min":   float(p.min()),
                "soc_pred_max":   float(p.max()),
                "soc_pred_range": float(p.max() - p.min()),
            })
        stats_df = pd.DataFrame(stats_rows).reset_index(drop=True)

        # Concatenate with cycle-aggregated X
        X_enriched = pd.concat(
            [X_agg.reset_index(drop=True), stats_df],
            axis=1
        )
        enriched.append((X_enriched, y_agg))

        if (task_idx + 1) % 5 == 0 or task_idx == 0:
            print(f"     enriched {task_idx+1}/{len(tasks_raw)} tasks  "
                  f"(soc_pred range: [{soc_pred.min():.2f}, {soc_pred.max():.2f}])")

    print(f"     new X shape: {enriched[0][0].shape}  "
          f"(+4 soc_pred columns)")
    return enriched