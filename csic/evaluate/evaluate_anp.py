"""
evaluate_anp.py
==============================================================================
Comprehensive inference evaluation script for the trained ANP model.

Five evaluation tests:

    TEST 1 — Context impact
        Vary context size (5, 10, 20, 30, 40, 50, 60 cycles) and measure MAE on the next 60 cycles of target. 
        Shows how much context the model needs to make accurate predictions.

    TEST 2 — Prediction horizon
        Fix context at 60 cycles and extend the target window as far as possible into the future. 
        Shows how MAE degrades with distance from the context window.

    TEST 3 — Uncertainty calibration
        Check whether the ANP's predicted variance is informative: does higher predicted uncertainty correlate with higher actual error?
        A well-calibrated model should show this correlation.

    TEST 4 — Prior vs posterior collapse
        Measure KL divergence between prior (context only) and posterior (context + target) as context size grows. 
        A healthy model should show decreasing KL as more context is provided.

    TEST 5 — Cross-task context robustness
        Use context from a different task than the one being predicted.
        Tests whether the model can transfer information across batteries.

Usage:
    python evaluate_anp.py \
        --anp_run ../train/runs/20260505_114753 \
        --data_dir ../csic_real_synth_load/prepared_data

With Optuna run:
    python evaluate_anp.py \
        --anp_run ../train/optuna_results/trial_019 \
        --data_dir ../csic_real_synth_load/prepared_data

    # Run only specific tests
    python evaluate_anp.py --anp_run ... --tests 1 2 3

Location: csic/validation/evaluate_anp.py
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
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ── Path setup ────────────────────────────────────────────────────────────────
_VAL_DIR   = Path(__file__).resolve().parent
_CSIC_ROOT = _VAL_DIR.parent
_TRAIN_DIR = _CSIC_ROOT / "train"
for _p in [str(_CSIC_ROOT), str(_TRAIN_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from train.train_utils import (
    load_prepared_data,
    validate_targets,
    sort_task_by_cycle,
)

from models.anp import LatentModel


# ==============================================================================
# CONSTANTS
# ==============================================================================

MEAS_PER_CYCLE  = 30
TRAIN_CTX_CYC   = 60   # context used during training
CONTEXT_SIZES   = [2, 5, 10, 20, 30, 40, 50, 60]   # test 1 context sweep
HORIZON_CTX_CYC = 60   # fixed context for horizon test (test 2)
HORIZON_STEP    = 60   # report MAE every N cycles in the future
DPI             = 150

# Colors
C_TRAIN  = "#1C7293"
C_VAL    = "#C0392B"
C_AMBER  = "#D4860A"
C_GREEN  = "#237A3D"
C_GREY   = "#9AB8C8"


# ==============================================================================
# HELPERS
# ==============================================================================

def load_anp(run_dir: Path, input_dim: int, output_dim: int,
             device: torch.device) -> nn.Module:
    ckpt_path = run_dir / "best.pt"
    cfg_path  = run_dir / "config.json"
    num_hidden = 128
    if cfg_path.exists():
        with cfg_path.open() as f:
            cfg = json.load(f)
        # train_anp.py stores num_hidden at top level
        # optuna_anp.py stores it inside params{}
        num_hidden = (cfg.get("num_hidden")
                      or cfg.get("params", {}).get("num_hidden")
                      or 128)
    model = LatentModel(num_hidden=num_hidden,
                        input_dim=input_dim, output_dim=output_dim)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    print(f"  ✓  ANP loaded  (num_hidden={num_hidden}  "
          f"epoch={ckpt.get('epoch','?')}  "
          f"val_MAE={ckpt.get('val_MAE', ckpt.get('val_loss','?'))})")
    return model


def denormalize(arr: np.ndarray, col: str, dv: dict) -> np.ndarray:
    return arr * dv["y_std"].get(col, 1.0) + dv["y_mean"].get(col, 0.0)


def compute_mae_dn(pred: np.ndarray, true: np.ndarray,
                   target_cols: List[str], dv: dict) -> Dict[str, float]:
    result = {}
    for i, col in enumerate(target_cols):
        p = denormalize(pred[:, i], col, dv)
        t = denormalize(true[:, i], col, dv)
        result[col] = float(np.abs(p - t).mean())
    return result


@torch.no_grad()
def anp_predict(
    model:    nn.Module,
    X_ctx:    np.ndarray,
    y_ctx:    np.ndarray,
    X_tgt:    np.ndarray,
    device:   torch.device,
    n_passes: int = 5,          # ← único parámetro nuevo
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run N stochastic forward passes and average predictions.

    The ANP samples z from the prior at each forward pass, so repeated
    calls with the same inputs give different outputs. Averaging N passes
    reduces this stochastic variance without retraining.

    Combination via law of total variance:
        ensemble_mean = mean(mean_i)
        ensemble_var  = mean(var_i + mean_i²) - ensemble_mean²
    """
    ctx_x = torch.tensor(X_ctx).unsqueeze(0).to(device)
    ctx_y = torch.tensor(y_ctx).unsqueeze(0).to(device)
    tgt_x = torch.tensor(X_tgt).unsqueeze(0).to(device)

    all_means, all_vars = [], []

    for i in range(n_passes):
        mean, var, _, _, _ = model(ctx_x, ctx_y, tgt_x, target_y=None)
        all_means.append(mean.squeeze(0).cpu().numpy())
        all_vars.append(var.squeeze(0).cpu().numpy())

    all_means = np.stack(all_means)   # (n_passes, Nt, O)
    all_vars  = np.stack(all_vars)    # (n_passes, Nt, O)

    ensemble_mean = all_means.mean(axis=0)
    ensemble_var  = (all_vars + all_means**2).mean(axis=0) - ensemble_mean**2
    ensemble_std  = np.sqrt(np.maximum(ensemble_var, 1e-8))

    return ensemble_mean, ensemble_std


@torch.no_grad()
def anp_prior_posterior_kl(
    model:   nn.Module,
    X_ctx:   np.ndarray,
    y_ctx:   np.ndarray,
    X_tgt:   np.ndarray,
    y_tgt:   np.ndarray,
    device:  torch.device,
    n_passes: int = 5,
) -> float:
    """
    Compute KL(posterior || prior) for a given context/target pair.
    Returns the scalar KL value (in nats).
    """
    ctx_x = torch.tensor(X_ctx).unsqueeze(0).to(device)
    ctx_y = torch.tensor(y_ctx).unsqueeze(0).to(device)
    tgt_x = torch.tensor(X_tgt).unsqueeze(0).to(device)
    tgt_y = torch.tensor(y_tgt).unsqueeze(0).to(device)

    kl_values = []
    for _ in range(n_passes):
        _, _, _, kl, _ = model(ctx_x, ctx_y, tgt_x, target_y=tgt_y, beta=1.0)
        kl_values.append(float(kl.item()))

    return float(np.mean(kl_values))


def save(fig: Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}")


# ==============================================================================
# TEST 1 — CONTEXT IMPACT
# ==============================================================================

def test_context_impact(
    model:        nn.Module,
    tasks:        list,
    task_labels:  List[str],
    target_cols:  List[str],
    dv:           dict,
    device:       torch.device,
    out_dir:      Path,
    ctx_sizes:    List[int] = CONTEXT_SIZES,
    tgt_cycles:   int       = 60,
    n_passes:     int       = 5,
) -> pd.DataFrame:
    """
    TEST 1: How does MAE change as we give the model more context?

    For each context size c in ctx_sizes:
        Context = first c × MEAS_PER_CYCLE rows
        Target  = next  tgt_cycles × MEAS_PER_CYCLE rows  (always 60 cycles)

    This keeps the target window fixed and only changes the amount of
    context provided, isolating the effect of context size on prediction
    quality.

    Args:
        ctx_sizes: List of context sizes in cycles to evaluate.
        tgt_cycles: Number of target cycles (fixed across all context sizes).

    Returns:
        DataFrame with columns [ctx_cycles, task, col, mae].
    """
    tgt_rows = tgt_cycles * MEAS_PER_CYCLE
    rows     = []

    for ctx_cyc in ctx_sizes:
        ctx_rows = ctx_cyc * MEAS_PER_CYCLE
        print(f"    ctx={ctx_cyc:3d} cycles ({ctx_rows} rows)...")

        for t_label, (X, y) in zip(task_labels, tasks):
            T       = len(X)
            ctx_end = min(ctx_rows, T)
            tgt_end = min(ctx_end + tgt_rows, T)

            if tgt_end <= ctx_end:
                continue

            X_ctx = X[:ctx_end]
            y_ctx = y[:ctx_end]
            X_tgt = X[ctx_end:tgt_end]
            y_tgt = y[ctx_end:tgt_end]

            mean, std = anp_predict(model, X_ctx, y_ctx, X_tgt, device, n_passes=n_passes)
            mae       = compute_mae_dn(mean, y_tgt, target_cols, dv)

            for col, val in mae.items():
                rows.append({
                    "ctx_cycles": ctx_cyc,
                    "task":       t_label,
                    "target":     col,
                    "mae":        val,
                    "pred_std_mean": float(std.mean()),
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "test1_context_impact.csv", index=False)
    print(f"  ✓  test1_context_impact.csv")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(target_cols),
                             figsize=(7 * len(target_cols), 5))
    if len(target_cols) == 1:
        axes = [axes]

    colors = plt.get_cmap("tab10", len(task_labels))

    for ax, col in zip(axes, target_cols):
        col_df = df[df["target"] == col]

        # Individual task lines
        for ti, t_label in enumerate(task_labels):
            t_df = col_df[col_df["task"] == t_label]
            ax.plot(t_df["ctx_cycles"], t_df["mae"],
                    color=colors(ti), alpha=0.4, linewidth=1.0,
                    marker="o", markersize=4)

        # Mean across tasks
        mean_df = col_df.groupby("ctx_cycles")["mae"].agg(["mean", "std"])
        ax.plot(mean_df.index, mean_df["mean"],
                color=C_TRAIN, linewidth=2.5, marker="o", markersize=7,
                label="Mean across tasks", zorder=5)
        ax.fill_between(mean_df.index,
                        mean_df["mean"] - mean_df["std"],
                        mean_df["mean"] + mean_df["std"],
                        alpha=0.15, color=C_TRAIN)

        # Mark training context size
        ax.axvline(TRAIN_CTX_CYC, color=C_AMBER, linestyle="--",
                   linewidth=1.5, label=f"Training ctx ({TRAIN_CTX_CYC} cyc)")

        ax.set_xlabel("Context size (cycles)")
        ax.set_ylabel(f"MAE [{col}]")
        ax.set_title(f"Test 1 — Context impact on {col}\n"
                     f"(target = next {tgt_cycles} cycles, fixed)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.set_xticks(ctx_sizes)

    save(fig, out_dir / "test1_context_impact.png")
    return df


# ==============================================================================
# TEST 2 — PREDICTION HORIZON
# ==============================================================================

def test_prediction_horizon(
    model:       nn.Module,
    tasks:       list,
    task_labels: List[str],
    target_cols: List[str],
    dv:          dict,
    device:      torch.device,
    out_dir:     Path,
    ctx_cycles:  int = HORIZON_CTX_CYC,
    step_cycles: int = HORIZON_STEP,
    n_passes:    int = 5,
) -> pd.DataFrame:
    """
    TEST 2: How does MAE grow as we predict further into the future?

    Context is fixed at ctx_cycles (60). The target window is divided into
    consecutive non-overlapping blocks of step_cycles cycles each.
    MAE is computed independently for each block.

    This shows the model's effective prediction horizon before accuracy
    degrades unacceptably.

    Args:
        ctx_cycles:  Fixed context size in cycles.
        step_cycles: Block size for reporting MAE (cycles).

    Returns:
        DataFrame with columns [horizon_start, horizon_end, task, target, mae].
    """
    ctx_rows  = ctx_cycles  * MEAS_PER_CYCLE
    step_rows = step_cycles * MEAS_PER_CYCLE
    rows      = []

    for t_label, (X, y) in zip(task_labels, tasks):
        T        = len(X)
        ctx_end  = min(ctx_rows, T)
        X_ctx    = X[:ctx_end]
        y_ctx    = y[:ctx_end]

        # Full remaining trajectory as target (memory-safe with no_grad)
        X_rest   = X[ctx_end:]
        y_rest   = y[ctx_end:]
        n_rest   = len(X_rest)

        if n_rest == 0:
            continue

        print(f"    {t_label}: ctx={ctx_cycles} cycles, "
              f"remaining={n_rest} rows "
              f"({n_rest // MEAS_PER_CYCLE} cycles)")

        # Predict entire remaining trajectory at once
        mean_full, std_full = anp_predict(
            model, X_ctx, y_ctx, X_rest, device, n_passes=n_passes
        )

        # Split into horizon blocks and compute MAE per block
        for block_start in range(0, n_rest, step_rows):
            block_end = min(block_start + step_rows, n_rest)
            if block_end <= block_start:
                break

            pred_block = mean_full[block_start:block_end]
            true_block = y_rest[block_start:block_end]
            std_block  = std_full[block_start:block_end]
            mae        = compute_mae_dn(pred_block, true_block, target_cols, dv)

            h_start = ctx_cycles + block_start // MEAS_PER_CYCLE
            h_end   = ctx_cycles + block_end   // MEAS_PER_CYCLE

            for col, val in mae.items():
                rows.append({
                    "horizon_start_cyc": h_start,
                    "horizon_end_cyc":   h_end,
                    "horizon_mid_cyc":   (h_start + h_end) / 2,
                    "task":              t_label,
                    "target":            col,
                    "mae":               val,
                    "pred_std_mean":     float(std_block.mean()),
                    "n_rows":            block_end - block_start,
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "test2_prediction_horizon.csv", index=False)
    print(f"  ✓  test2_prediction_horizon.csv")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(target_cols),
                             figsize=(9 * len(target_cols), 5))
    if len(target_cols) == 1:
        axes = [axes]

    colors = plt.get_cmap("tab10", len(task_labels))

    for ax, col in zip(axes, target_cols):
        col_df = df[df["target"] == col]

        for ti, t_label in enumerate(task_labels):
            t_df = col_df[col_df["task"] == t_label].sort_values("horizon_mid_cyc")
            ax.plot(t_df["horizon_mid_cyc"], t_df["mae"],
                    color=colors(ti), alpha=0.45, linewidth=1.0,
                    label=t_label, marker=".")

        # Mean across tasks
        mean_df = col_df.groupby("horizon_mid_cyc")["mae"].agg(["mean","std"])
        ax.plot(mean_df.index, mean_df["mean"],
                color=C_TRAIN, linewidth=2.5, zorder=5, label="Mean")
        ax.fill_between(mean_df.index,
                        mean_df["mean"] - mean_df["std"],
                        mean_df["mean"] + mean_df["std"],
                        alpha=0.15, color=C_TRAIN)

        # Mark training target boundary
        ax.axvline(ctx_cycles + TRAIN_CTX_CYC, color=C_AMBER,
                   linestyle="--", linewidth=1.5,
                   label=f"Training target end ({ctx_cycles + TRAIN_CTX_CYC})")
        ax.axvline(ctx_cycles, color=C_GREEN, linestyle=":",
                   linewidth=1.2, label=f"Context end ({ctx_cycles})")

        ax.set_xlabel("Prediction horizon (cycle number)")
        ax.set_ylabel(f"MAE [{col}]")
        ax.set_title(f"Test 2 — Prediction horizon [{col}]\n"
                     f"(context = first {ctx_cycles} cycles, fixed)")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.25)

    save(fig, out_dir / "test2_prediction_horizon.png")
    return df


# ==============================================================================
# TEST 3 — UNCERTAINTY CALIBRATION
# ==============================================================================

def test_uncertainty_calibration(
    model:       nn.Module,
    tasks:       list,
    task_labels: List[str],
    target_cols: List[str],
    dv:          dict,
    device:      torch.device,
    out_dir:     Path,
    ctx_cycles:  int = HORIZON_CTX_CYC,
    tgt_cycles:  int = 60,
    n_bins:      int = 10,
    n_passes:    int = 5,
) -> pd.DataFrame:
    """
    TEST 3: Is the ANP's predicted uncertainty informative?

    Bins all target predictions by predicted standard deviation and checks
    whether actual absolute error is higher in high-uncertainty bins.

    A well-calibrated model should show a positive correlation between
    predicted std and actual error (|pred - true|).

    Also computes Spearman correlation between std and |error|.

    Args:
        n_bins: Number of bins for the reliability diagram.

    Returns:
        DataFrame with binned calibration statistics.
    """
    ctx_rows = ctx_cycles * MEAS_PER_CYCLE
    tgt_rows = tgt_cycles * MEAS_PER_CYCLE
    rows     = []

    for t_label, (X, y) in zip(task_labels, tasks):
        T       = len(X)
        ctx_end = min(ctx_rows, T)
        tgt_end = min(ctx_end + tgt_rows, T)
        if tgt_end <= ctx_end:
            continue

        X_ctx  = X[:ctx_end];   y_ctx = y[:ctx_end]
        X_tgt  = X[ctx_end:tgt_end]; y_tgt = y[ctx_end:tgt_end]

        mean, std = anp_predict(model, X_ctx, y_ctx, X_tgt, device, n_passes=n_passes)

        for i, col in enumerate(target_cols):
            pred_dn = denormalize(mean[:, i], col, dv)
            true_dn = denormalize(y_tgt[:, i], col, dv)
            std_dn  = std[:, i] * dv["y_std"].get(col, 1.0)
            abs_err = np.abs(pred_dn - true_dn)

            for j in range(len(pred_dn)):
                rows.append({
                    "task":    t_label,
                    "target":  col,
                    "pred_std": float(std_dn[j]),
                    "abs_err":  float(abs_err[j]),
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "test3_uncertainty_raw.csv", index=False)

    # Bin by predicted std and compute mean abs_err per bin
    cal_rows = []
    for col in target_cols:
        col_df    = df[df["target"] == col]
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(col_df["pred_std"], quantiles)

        for b in range(n_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            mask   = (col_df["pred_std"] >= lo) & (col_df["pred_std"] < hi)
            if mask.sum() == 0:
                continue
            mean_std = col_df[mask]["pred_std"].mean()
            mean_err = col_df[mask]["abs_err"].mean()
            cal_rows.append({
                "target":   col,
                "bin_lo":   lo,
                "bin_hi":   hi,
                "mean_std": mean_std,
                "mean_abs_err": mean_err,
                "n":        int(mask.sum()),
            })

        # Spearman correlation
        from scipy import stats
        rho, pval = stats.spearmanr(col_df["pred_std"], col_df["abs_err"])
        print(f"    {col}: Spearman ρ(std, |error|) = {rho:.3f}  p={pval:.4f}")

    cal_df = pd.DataFrame(cal_rows)
    cal_df.to_csv(out_dir / "test3_calibration_bins.csv", index=False)
    print(f"  ✓  test3_calibration_bins.csv")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(target_cols),
                             figsize=(6 * len(target_cols), 5))
    if len(target_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, target_cols):
        col_cal = cal_df[cal_df["target"] == col]
        ax.scatter(col_cal["mean_std"], col_cal["mean_abs_err"],
                   s=80, color=C_TRAIN, zorder=3)
        ax.plot(col_cal["mean_std"], col_cal["mean_abs_err"],
                color=C_TRAIN, linewidth=1.5, alpha=0.7)

        # Perfect calibration line (std ≈ error)
        lim = max(col_cal["mean_std"].max(), col_cal["mean_abs_err"].max()) * 1.1
        ax.plot([0, lim], [0, lim], color=C_GREY, linestyle="--",
                linewidth=1.0, label="Perfect calibration (std = |error|)")

        ax.set_xlabel(f"Mean predicted std [{col}]")
        ax.set_ylabel(f"Mean |error| [{col}]")
        ax.set_title(f"Test 3 — Uncertainty calibration [{col}]\n"
                     f"(above diagonal = underconfident, below = overconfident)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    save(fig, out_dir / "test3_uncertainty_calibration.png")
    return cal_df


# ==============================================================================
# TEST 4 — PRIOR vs POSTERIOR (KL COLLAPSE CHECK)
# ==============================================================================

def test_prior_posterior_kl(
    model:       nn.Module,
    tasks:       list,
    task_labels: List[str],
    target_cols: List[str],
    dv:          dict,
    device:      torch.device,
    out_dir:     Path,
    ctx_sizes:   List[int] = CONTEXT_SIZES,
    tgt_cycles:  int       = 60,
    n_passes:    int       = 5,
) -> pd.DataFrame:
    """
    TEST 4: Does providing more context reduce posterior uncertainty?

    Computes KL(posterior || prior) for each context size.
    A healthy ANP should show increasing KL with more context — meaning
    the posterior is more different from the prior when context is informative.
    A flat or near-zero KL across all context sizes indicates KL collapse.

    Returns:
        DataFrame with columns [ctx_cycles, task, kl].
    """
    tgt_rows = tgt_cycles * MEAS_PER_CYCLE
    rows     = []

    for ctx_cyc in ctx_sizes:
        ctx_rows = ctx_cyc * MEAS_PER_CYCLE
        for t_label, (X, y) in zip(task_labels, tasks):
            T        = len(X)
            ctx_end  = min(ctx_rows, T)
            tgt_end  = min(ctx_end + tgt_rows, T)
            if tgt_end <= ctx_end:
                continue

            kl = anp_prior_posterior_kl(
                model,
                X[:ctx_end], y[:ctx_end],
                X[ctx_end:tgt_end], y[ctx_end:tgt_end],
                device,
                n_passes=n_passes,
            )
            rows.append({"ctx_cycles": ctx_cyc, "task": t_label, "kl": kl})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "test4_prior_posterior_kl.csv", index=False)
    print(f"  ✓  test4_prior_posterior_kl.csv")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    colors  = plt.get_cmap("tab10", len(task_labels))

    for ti, t_label in enumerate(task_labels):
        t_df = df[df["task"] == t_label]
        ax.plot(t_df["ctx_cycles"], t_df["kl"],
                color=colors(ti), alpha=0.45, linewidth=1.0,
                marker="o", markersize=4, label=t_label)

    mean_kl = df.groupby("ctx_cycles")["kl"].agg(["mean", "std"])
    ax.plot(mean_kl.index, mean_kl["mean"],
            color=C_TRAIN, linewidth=2.5, zorder=5,
            marker="o", markersize=7, label="Mean KL")
    ax.fill_between(mean_kl.index,
                    mean_kl["mean"] - mean_kl["std"],
                    mean_kl["mean"] + mean_kl["std"],
                    alpha=0.15, color=C_TRAIN)

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Context size (cycles)")
    ax.set_ylabel("KL(posterior || prior)  [nats]")
    ax.set_title("Test 4 — Prior vs Posterior KL divergence\n"
                 "KL → 0 for all context sizes indicates KL collapse")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.25)
    ax.set_xticks(ctx_sizes)

    save(fig, out_dir / "test4_prior_posterior_kl.png")
    return df


# ==============================================================================
# TEST 5 — CROSS-TASK CONTEXT ROBUSTNESS
# ==============================================================================

def test_cross_task_robustness(
    model:       nn.Module,
    tasks:       list,
    task_labels: List[str],
    target_cols: List[str],
    dv:          dict,
    device:      torch.device,
    out_dir:     Path,
    ctx_cycles:  int = HORIZON_CTX_CYC,
    tgt_cycles:  int = 60,
    n_passes:    int = 5,
) -> pd.DataFrame:
    """
    TEST 5: Does the model generalize when given context from a different task?

    For each task i, predict its cycles (ctx+1) to (ctx+tgt) using:
        (a) Own context  — context from task i itself    (matched)
        (b) Cross context — context from every other task j≠i  (mismatched)

    Compares matched vs mismatched MAE.
    If the ANP truly performs meta-learning, matched context should give
    substantially better predictions than cross-task context.

    Returns:
        DataFrame with columns [pred_task, ctx_task, matched, target, mae].
    """
    ctx_rows = ctx_cycles * MEAS_PER_CYCLE
    tgt_rows = tgt_cycles * MEAS_PER_CYCLE
    rows     = []

    for pi, (t_pred_label, (X_pred, y_pred)) in enumerate(
        zip(task_labels, tasks)
    ):
        T       = len(X_pred)
        ctx_end = min(ctx_rows, T)
        tgt_end = min(ctx_end + tgt_rows, T)
        if tgt_end <= ctx_end:
            continue

        X_tgt  = X_pred[ctx_end:tgt_end]
        y_tgt  = y_pred[ctx_end:tgt_end]

        for ci, (t_ctx_label, (X_ctx_src, y_ctx_src)) in enumerate(
            zip(task_labels, tasks)
        ):
            ctx_end_c = min(ctx_rows, len(X_ctx_src))
            X_ctx     = X_ctx_src[:ctx_end_c]
            y_ctx     = y_ctx_src[:ctx_end_c]

            mean, _ = anp_predict(model, X_ctx, y_ctx, X_tgt, device, n_passes=n_passes)
            mae     = compute_mae_dn(mean, y_tgt, target_cols, dv)
            matched = (pi == ci)

            for col, val in mae.items():
                rows.append({
                    "pred_task": t_pred_label,
                    "ctx_task":  t_ctx_label,
                    "matched":   matched,
                    "target":    col,
                    "mae":       val,
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "test5_cross_task_robustness.csv", index=False)
    print(f"  ✓  test5_cross_task_robustness.csv")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(target_cols),
                             figsize=(7 * len(target_cols), 5))
    if len(target_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, target_cols):
        col_df = df[df["target"] == col]

        matched_mae  = col_df[col_df["matched"]]["mae"]
        mismatch_mae = col_df[~col_df["matched"]]["mae"]

        ax.boxplot(
            [matched_mae.values, mismatch_mae.values],
            tick_labels=["Own context\n(matched)", "Cross context\n(mismatched)"],
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
        )
        bp_data = ax.containers
        for patch, color in zip(
            [p for p in ax.patches if hasattr(p, "get_facecolor")],
            [C_TRAIN, C_VAL],
        ):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_ylabel(f"MAE [{col}]")
        ax.set_title(f"Test 5 — Cross-task context robustness [{col}]\n"
                     f"Matched: {matched_mae.mean():.3f}  "
                     f"Mismatched: {mismatch_mae.mean():.3f}")
        ax.grid(True, axis="y", alpha=0.25)

    save(fig, out_dir / "test5_cross_task_robustness.png")
    return df


# ==============================================================================
# MAIN
# ==============================================================================

def run(
    anp_run_dir:    Path,
    data_dir:       str,
    out_dir:        Path,
    tests:          List[int],
    train_task_ids: List[int],
    val_task_ids:   List[int],
    test_task_ids:  List[int],
    eval_split:     str = "val",
    n_passes:       int = 5,
) -> None:
    """
    Run selected evaluation tests.

    Args:
        anp_run_dir:    Path to the ANP run directory.
        data_dir:       Path to prepared_data.pkl directory.
        out_dir:        Output directory for results.
        tests:          List of test indices to run (1-5).
        eval_split:     Which split to evaluate: 'val', 'test', or 'all'.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧  Device  : {device}")
    print(f"📁  Out dir : {out_dir}")
    print(f"   Tests   : {tests}")
    print(f"   Split   : {eval_split}")

    # ── Load data ─────────────────────────────────────────────────────────────
    data = load_prepared_data(data_dir)
    validate_targets(data)

    target_cols = list(data["normalized_synth_datasets"][0][1].columns)
    input_dim   = data["normalized_synth_datasets"][0][0].shape[1]
    output_dim  = len(target_cols)
    dv          = {
        "y_mean": data["denorm_values"]["y_mean"],
        "y_std":  data["denorm_values"]["y_std"],
    }

    print(f"   Targets : {target_cols}")

    # Select tasks for evaluation
    if eval_split == "val":
        eval_ids = val_task_ids
    elif eval_split == "test":
        eval_ids = test_task_ids
    elif eval_split == "train":
        eval_ids = train_task_ids
    else:  # all
        eval_ids = train_task_ids + val_task_ids + test_task_ids

    def label(i):
        if i in train_task_ids: return f"train_{train_task_ids.index(i)+1:02d}"
        if i in val_task_ids:   return f"val_{val_task_ids.index(i)+1:02d}"
        if i in test_task_ids:  return f"test_{test_task_ids.index(i)+1:02d}"
        return f"task_{i:02d}"

    # Re-sort (sort_task_by_cycle returns DataFrames)
    tasks_sorted_clean = []
    for i in eval_ids:
        X_df, y_df = sort_task_by_cycle(*data["normalized_synth_datasets"][i])
        tasks_sorted_clean.append((X_df.values.astype(np.float32),
                                   y_df.values.astype(np.float32)))

    task_labels = [label(i) for i in eval_ids]
    print(f"   Tasks   : {task_labels}\n")

    # ── Load ANP ─────────────────────────────────────────────────────────────
    print("📦  Loading ANP...")
    model = load_anp(anp_run_dir, input_dim, output_dim, device)

    # ── Run tests ─────────────────────────────────────────────────────────────
    if 1 in tests:
        print(f"\n{'='*55}")
        print("  TEST 1 — Context impact")
        print(f"{'='*55}")
        test_context_impact(
            model, tasks_sorted_clean, task_labels,
            target_cols, dv, device, out_dir, n_passes=n_passes
        )

    if 2 in tests:
        print(f"\n{'='*55}")
        print("  TEST 2 — Prediction horizon")
        print(f"{'='*55}")
        test_prediction_horizon(
            model, tasks_sorted_clean, task_labels,
            target_cols, dv, device, out_dir, n_passes=n_passes
        )

    if 3 in tests:
        print(f"\n{'='*55}")
        print("  TEST 3 — Uncertainty calibration")
        print(f"{'='*55}")
        try:
            test_uncertainty_calibration(
                model, tasks_sorted_clean, task_labels,
                target_cols, dv, device, out_dir, n_passes=n_passes
            )
        except ImportError:
            print("  ⚠  scipy not available — skipping Test 3. "
                  "Install with: pip install scipy")

    if 4 in tests:
        print(f"\n{'='*55}")
        print("  TEST 4 — Prior vs Posterior KL")
        print(f"{'='*55}")
        test_prior_posterior_kl(
            model, tasks_sorted_clean, task_labels,
            target_cols, dv, device, out_dir, n_passes=n_passes
        )

    if 5 in tests:
        print(f"\n{'='*55}")
        print("  TEST 5 — Cross-task context robustness")
        print(f"{'='*55}")
        test_cross_task_robustness(
            model, tasks_sorted_clean, task_labels,
            target_cols, dv, device, out_dir, n_passes=n_passes
        )

    print(f"\n✅  All tests complete. Results in: {out_dir}\n")


# ==============================================================================
# CLI
# ==============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ANP inference evaluation — context impact, horizon, calibration, KL collapse, cross-task robustness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--anp_run",   type=str, required=True, help="Path to the ANP run directory (contains best.pt)")
    p.add_argument("--data_dir",  type=str, default="../csic_real_synth_load/prepared_data")
    p.add_argument("--out_dir",   type=str, default="", help="Output directory (default: ./anp_eval/<timestamp>/)")
    p.add_argument("--tests",     type=int, nargs="+", default=[1, 2, 3, 4, 5], help="Which tests to run (1-5, default: all)")
    p.add_argument("--split",     type=str, default="all", choices=["train", "val", "test", "all"], help="Which task split to evaluate")
    p.add_argument("--train_ids", type=int, nargs="+", default=list(range(17)))
    p.add_argument("--val_ids",   type=int, nargs="+", default=list(range(17, 22)))
    p.add_argument("--test_ids",  type=int, nargs="+", default=list(range(22, 25)))
    p.add_argument("--n_passes", type=int, default=5, help="Number of stochastic forward passes to average (default: 5)")
    return p.parse_args() 


def main() -> None:
    args    = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(__file__).resolve().parent / "anp_eval")
    run(
        anp_run_dir    = Path(args.anp_run),
        data_dir       = args.data_dir,
        out_dir        = out_dir,
        tests          = args.tests,
        train_task_ids = args.train_ids,
        val_task_ids   = args.val_ids,
        test_task_ids  = args.test_ids,
        eval_split     = args.split,
        n_passes       = args.n_passes,
    )

if __name__ == "__main__":
    main()
