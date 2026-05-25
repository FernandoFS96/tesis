"""
evaluate_anp.py
==============================================================================
Comprehensive inference evaluation script for all trained ANP model variants.

Supports:
    - ANP dual-target        (--anp_run)
    - ANP SoC-only           (--anp_soc_run)
    - ANP Cycle-only         (--anp_cycle_run)
    - ANP SoC-only reduced   (--anp_soc_reduced_run)
    - ANP Cycle-only reduced (--anp_cycle_reduced_run)

Each model is auto-detected from its checkpoint and config.json:
    - model_target_cols : which targets it predicts
    - feature_cols      : which X columns to use (None = all 202)
    - aggregate_by_cycle: whether to aggregate EIS measurements per cycle

Results are saved to model-specific subdirectories under --out_dir.

Five evaluation tests (controllable via --tests):

    TEST 1 — Context impact
        Vary context size and measure MAE on next 60 cycles.

    TEST 2 — Prediction horizon
        Fix context at 60 cycles, extend target as far as possible.

    TEST 3 — Uncertainty calibration
        Check correlation between predicted std and actual error.

    TEST 4 — Prior vs posterior KL collapse
        Measure KL(posterior || prior) as context size grows.

    TEST 5 — Cross-task context robustness
        Predict task i using context from task j != i.

Inference uses n_passes stochastic forward passes (default 5) averaged via the law of total variance to reduce the ANP's inherent stochasticity.

Usage:
    python evaluate_anp.py \
        --anp_run            ../train/runs/anp_all/20260512_124715 \
        --anp_soc_run        ../train/runs/anp_SoC/20260512_114601 \
        --anp_cycle_run      ../train/runs/anp_Cycle/20260512_114703 \
        --anp_soc_reduced_run  ../train/optuna_results/anp_soc_reduced/trial_163 \
        --anp_cycle_reduced_run ../train/runs/anp_Cycle_reduced/20260519_122631 \
        --data_dir           ../csic_real_synth_load/prepared_data \
        --tests 1 2 3 4 5 \
        --split test

Location: csic/validation/evaluate_anp.py
==============================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    REDUCED_FEATURE_SETS,
    get_feature_indices,
    filter_x,
    aggregate_by_cycle,
)
from models.anp import LatentModel


# ==============================================================================
# CONSTANTS
# ==============================================================================

MEAS_PER_CYCLE  = 30
TRAIN_CTX_CYC   = 60
CONTEXT_SIZES   = [2, 5, 10, 20, 30, 40, 50, 60]
HORIZON_STEP    = 60   # report MAE every N cycles in Test 2
DPI             = 300

C_MAIN  = "#1C7293"
C_VAL   = "#C0392B"
C_AMBER = "#D4860A"
C_GREEN = "#237A3D"
C_GREY  = "#9AB8C8"


# ==============================================================================
# MODEL INFO DATACLASS
# ==============================================================================

@dataclass
class ModelInfo:
    """
    All metadata needed to run inference with one ANP model variant.

    Attributes:
        label            : Display name (e.g. 'anp_soc_reduced').
        model            : Loaded LatentModel in eval mode.
        all_target_cols  : All targets present in the data (e.g. ['SoC (%)', 'Cycle']).
        model_target_cols: Targets this model predicts (subset of all_target_cols).
        target_idx       : Indices into all_target_cols for model_target_cols.
        feature_cols     : X column names for reduced models, None for full-feature.
        feat_idx         : Numpy column indices for X filtering (None = no filtering).
        aggregate        : True if the model was trained on cycle-aggregated data.
        rows_per_cycle   : 1 if aggregate else MEAS_PER_CYCLE. Used to convert cycle counts to row counts in all tests.
        run_dir          : Path to the model's run directory.
        out_dir          : Model-specific output subdirectory.
    """
    label:             str
    model:             nn.Module
    all_target_cols:   List[str]
    model_target_cols: List[str]
    target_idx:        List[int]
    feature_cols:      Optional[List[str]]
    feat_idx:          Optional[List[int]]
    aggregate:         bool
    rows_per_cycle:    int
    run_dir:           Path
    out_dir:           Path


# ==============================================================================
# MODEL LOADING
# ==============================================================================

def load_anp_model(
    run_dir:         Path,
    input_dim:       int,
    all_target_cols: List[str],
    all_col_names:   List[str],
    device:          torch.device,
    label:           str,
    base_out_dir:    Path,
) -> Optional[ModelInfo]:
    """
    Load an ANP checkpoint and build a ModelInfo object.

    Auto-detects from checkpoint + config.json:
        - model_target_cols  (from target_col config field)
        - model_input_dim    (from latent_encoder weight shape in checkpoint)
        - feature_cols       (from REDUCED_FEATURE_SETS if input_dim < full)
        - aggregate_by_cycle (from config field)

    Args:
        run_dir:         Path to run directory containing best.pt.
        input_dim:       Full dataset input dimension.
        all_target_cols: All target columns in the data.
        all_col_names:   All X column names in the data.
        device:          Torch device.
        label:           Display label for output files and console.
        base_out_dir:    Parent output dir; model results go in base_out_dir/label/.

    Returns:
        ModelInfo or None if checkpoint not found.
    """
    ckpt_path = run_dir / "best.pt"
    cfg_path  = run_dir / "config.json"

    if not ckpt_path.exists():
        print(f"  ⚠  {label}: best.pt not found at {ckpt_path} — skipping")
        return None

    num_hidden   = 128
    attn_dropout = 0.1
    target_col   = "all"
    aggregate    = False
    ctx_cycles_m = TRAIN_CTX_CYC
    meas_p_cycle = MEAS_PER_CYCLE
    cfg_data     = {}

    if cfg_path.exists():
        with cfg_path.open() as f:
            cfg_data = json.load(f)
        num_hidden   = (cfg_data.get("num_hidden")
                        or cfg_data.get("params", {}).get("num_hidden", 128))
        attn_dropout = cfg_data.get("attn_dropout", 0.1)
        target_col   = cfg_data.get("target_col", "all")
        aggregate    = cfg_data.get("aggregate_by_cycle", False)
        ctx_cycles_m = cfg_data.get("ctx_cycles", TRAIN_CTX_CYC)
        meas_p_cycle = cfg_data.get("measurements_per_cycle", MEAS_PER_CYCLE)

    # Which targets does this model predict?
    if target_col == "all":
        model_target_cols = all_target_cols
    elif target_col in all_target_cols:
        model_target_cols = [target_col]
    else:
        raw_cpu = torch.load(ckpt_path, map_location="cpu")
        out_keys = [k for k in raw_cpu["model"]
                    if "mean_projection" in k and "weight" in k]
        out_dim = raw_cpu["model"][out_keys[0]].shape[0] if out_keys else len(all_target_cols)
        model_target_cols = all_target_cols[:out_dim]

    output_dim = len(model_target_cols)
    target_idx = [all_target_cols.index(c) for c in model_target_cols]

    # Infer model_input_dim robustly from checkpoint weight shape
    raw_cpu = torch.load(ckpt_path, map_location="cpu")
    lat_key = next(
        (k for k in raw_cpu["model"]
         if "latent_encoder.input_projection.linear_layer.weight" in k), None
    )
    if lat_key:
        model_input_dim = raw_cpu["model"][lat_key].shape[1] - output_dim
    else:
        model_input_dim = input_dim

    # Determine feature_cols for X filtering
    if model_input_dim == input_dim:
        feature_cols = None
        feat_idx     = None
    else:
        feature_cols = REDUCED_FEATURE_SETS.get(target_col)
        if feature_cols is None:
            raise ValueError(
                f"'{label}' has input_dim={model_input_dim} (data={input_dim}), "
                f"but REDUCED_FEATURE_SETS has no entry for target_col='{target_col}'."
            )
        if len(feature_cols) != model_input_dim:
            raise ValueError(
                f"'{label}': checkpoint input_dim={model_input_dim} but "
                f"REDUCED_FEATURE_SETS['{target_col}'] has {len(feature_cols)} features."
            )
        feat_idx = get_feature_indices(all_col_names, feature_cols)

    rows_per_cycle = 1 if aggregate else meas_p_cycle

    # Build and load model
    model = LatentModel(num_hidden=num_hidden, input_dim=model_input_dim,
                        output_dim=output_dim, attn_dropout=attn_dropout)
    model.load_state_dict(raw_cpu["model"])
    model.eval().to(device)

    val_mae  = raw_cpu.get("val_MAE", raw_cpu.get("val_loss", "?"))
    feat_str = f"reduced({model_input_dim})" if feature_cols else f"all({input_dim})"
    agg_str  = "  agg=cycle" if aggregate else ""
    print(f"  ✓  {label:<24} targets={model_target_cols}  "
          f"features={feat_str}{agg_str}  val_MAE={val_mae}")

    out_dir = base_out_dir / label
    out_dir.mkdir(parents=True, exist_ok=True)

    return ModelInfo(
        label=label, model=model,
        all_target_cols=all_target_cols, model_target_cols=model_target_cols,
        target_idx=target_idx, feature_cols=feature_cols, feat_idx=feat_idx,
        aggregate=aggregate, rows_per_cycle=rows_per_cycle,
        run_dir=run_dir, out_dir=out_dir,
    )


# ==============================================================================
# INFERENCE HELPERS
# ==============================================================================

def denormalize(arr: np.ndarray, col: str, dv: dict) -> np.ndarray:
    return arr * dv["y_std"].get(col, 1.0) + dv["y_mean"].get(col, 0.0)


def compute_mae_dn(
    pred:        np.ndarray,
    true:        np.ndarray,
    target_cols: List[str],
    dv:          dict,
) -> Dict[str, float]:
    """
    Compute MAE per target column in original (denormalized) units.
    Columns that are entirely NaN (unmodelled targets) return NaN.
    """
    result = {}
    for i, col in enumerate(target_cols):
        if np.all(np.isnan(pred[:, i])):
            result[col] = float("nan")
        else:
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
    n_passes: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run n_passes stochastic forward passes and combine via law of total variance.
    Returns (ensemble_mean, ensemble_std) — both shape (Nt, O_model).
    """
    ctx_x = torch.tensor(X_ctx).unsqueeze(0).to(device)
    ctx_y = torch.tensor(y_ctx).unsqueeze(0).to(device)
    tgt_x = torch.tensor(X_tgt).unsqueeze(0).to(device)
    all_means, all_vars = [], []
    for _ in range(n_passes):
        mean, var, _, _, _ = model(ctx_x, ctx_y, tgt_x, target_y=None)
        all_means.append(mean.squeeze(0).cpu().numpy())
        all_vars.append(var.squeeze(0).cpu().numpy())
    all_means = np.stack(all_means)
    all_vars  = np.stack(all_vars)
    ens_mean  = all_means.mean(axis=0)
    ens_var   = (all_vars + all_means**2).mean(axis=0) - ens_mean**2
    ens_std   = np.sqrt(np.maximum(ens_var, 1e-8))
    return ens_mean, ens_std


@torch.no_grad()
def anp_kl(
    model:    nn.Module,
    X_ctx:    np.ndarray,
    y_ctx:    np.ndarray,
    X_tgt:    np.ndarray,
    y_tgt:    np.ndarray,
    device:   torch.device,
    n_passes: int = 5,
) -> float:
    """Compute KL(posterior || prior) averaged over n_passes."""
    ctx_x = torch.tensor(X_ctx).unsqueeze(0).to(device)
    ctx_y = torch.tensor(y_ctx).unsqueeze(0).to(device)
    tgt_x = torch.tensor(X_tgt).unsqueeze(0).to(device)
    tgt_y = torch.tensor(y_tgt).unsqueeze(0).to(device)
    kls = []
    for _ in range(n_passes):
        _, _, _, kl, _ = model(ctx_x, ctx_y, tgt_x, target_y=tgt_y, beta=1.0)
        kls.append(float(kl.item()))
    return float(np.mean(kls))


def predict_mi(
    mi:       ModelInfo,
    X_ctx:    np.ndarray,
    y_ctx:    np.ndarray,
    X_tgt:    np.ndarray,
    device:   torch.device,
    n_passes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference for a ModelInfo, handling X/y filtering and output expansion.

    Filters X to mi.feat_idx and y_ctx to mi.target_idx before the forward pass, 
    then expands results to all_target_cols shape with NaN for targets the model does not predict.

    Returns:
        mean (Nt, len(all_target_cols)) and std (Nt, len(all_target_cols)).
    """
    X_ctx_m = filter_x(X_ctx, mi.feat_idx)
    X_tgt_m = filter_x(X_tgt, mi.feat_idx)
    y_ctx_m = y_ctx[:, mi.target_idx] if mi.target_idx != list(range(len(mi.all_target_cols))) else y_ctx

    mean_m, std_m = anp_predict(mi.model, X_ctx_m, y_ctx_m, X_tgt_m, device, n_passes)

    # Expand to all_target_cols
    Nt  = len(X_tgt)
    n_a = len(mi.all_target_cols)
    mean_full = np.full((Nt, n_a), float("nan"), dtype=np.float32)
    std_full  = np.full((Nt, n_a), float("nan"), dtype=np.float32)
    for mi_i, col in enumerate(mi.model_target_cols):
        fi = mi.all_target_cols.index(col)
        mean_full[:, fi] = mean_m[:, mi_i]
        std_full[:, fi]  = std_m[:, mi_i]
    return mean_full, std_full


def kl_mi(
    mi:       ModelInfo,
    X_ctx:    np.ndarray,
    y_ctx:    np.ndarray,
    X_tgt:    np.ndarray,
    y_tgt:    np.ndarray,
    device:   torch.device,
    n_passes: int,
) -> float:
    """Run KL computation for a ModelInfo with appropriate filtering."""
    X_ctx_m = filter_x(X_ctx, mi.feat_idx)
    X_tgt_m = filter_x(X_tgt, mi.feat_idx)
    y_ctx_m = y_ctx[:, mi.target_idx]
    y_tgt_m = y_tgt[:, mi.target_idx]
    return anp_kl(mi.model, X_ctx_m, y_ctx_m, X_tgt_m, y_tgt_m, device, n_passes)


def save(fig: Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓  {path.name}")


# ==============================================================================
# TEST 1 — CONTEXT IMPACT
# ==============================================================================

def test_context_impact(
    mi:          ModelInfo,
    tasks:       list,
    task_labels: List[str],
    dv:          dict,
    device:      torch.device,
    ctx_sizes:   List[int] = CONTEXT_SIZES,
    tgt_cycles:  int       = 60,
    n_passes:    int       = 5,
) -> pd.DataFrame:
    """
    TEST 1: How does MAE change as context grows?

    For each context size c (in cycles):
        ctx_rows = c * mi.rows_per_cycle
        tgt_rows = tgt_cycles * mi.rows_per_cycle
    Target window is always fixed at tgt_cycles cycles.
    Works transparently for both measurement-level and cycle-aggregated models.
    """
    rows = []
    for ctx_cyc in ctx_sizes:
        ctx_rows = ctx_cyc  * mi.rows_per_cycle
        tgt_rows = tgt_cycles * mi.rows_per_cycle
        print(f"    ctx={ctx_cyc:3d} cycles ...")

        for t_label, (X, y) in zip(task_labels, tasks):
            T       = len(X)
            ctx_end = min(ctx_rows, T)
            tgt_end = min(ctx_end + tgt_rows, T)
            if tgt_end <= ctx_end:
                continue
            mean, std = predict_mi(mi, X[:ctx_end], y[:ctx_end],
                                    X[ctx_end:tgt_end], device, n_passes)
            mae = compute_mae_dn(mean, y[ctx_end:tgt_end], mi.all_target_cols, dv)
            for col, val in mae.items():
                rows.append({"ctx_cycles": ctx_cyc, "task": t_label,
                             "target": col, "mae": val,
                             "pred_std_mean": float(np.nanmean(std))})

    df = pd.DataFrame(rows)
    df.to_csv(mi.out_dir / "test1_context_impact.csv", index=False)

    target_cols_plot = [c for c in mi.all_target_cols
                        if not df[df["target"]==c]["mae"].isna().all()]
    fig, axes = plt.subplots(1, len(target_cols_plot),
                             figsize=(7 * len(target_cols_plot), 5))
    if len(target_cols_plot) == 1:
        axes = [axes]
    colors = plt.get_cmap("tab10", len(task_labels))
    for ax, col in zip(axes, target_cols_plot):
        col_df = df[df["target"] == col].dropna(subset=["mae"])
        for ti, t_label in enumerate(task_labels):
            t_df = col_df[col_df["task"] == t_label]
            ax.plot(t_df["ctx_cycles"], t_df["mae"], color=colors(ti),
                    alpha=0.4, linewidth=1.0, marker="o", markersize=4)
        mean_df = col_df.groupby("ctx_cycles")["mae"].agg(["mean","std"])
        ax.plot(mean_df.index, mean_df["mean"], color=C_MAIN, linewidth=2.5,
                marker="o", markersize=7, label="Mean across tasks", zorder=5)
        ax.fill_between(mean_df.index, mean_df["mean"]-mean_df["std"],
                        mean_df["mean"]+mean_df["std"], alpha=0.15, color=C_MAIN)
        ax.axvline(TRAIN_CTX_CYC, color=C_AMBER, linestyle="--",
                   linewidth=1.5, label=f"Training ctx ({TRAIN_CTX_CYC} cyc)")
        ax.set_xlabel("Context size (cycles)")
        ax.set_ylabel(f"MAE [{col}]")
        ax.set_title(f"Test 1 — Context impact [{mi.label}] — {col}\n"
                     f"(target = next {tgt_cycles} cycles, fixed)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.25); ax.set_xticks(ctx_sizes)
    save(fig, mi.out_dir / "test1_context_impact.png")
    return df


# ==============================================================================
# TEST 2 — PREDICTION HORIZON
# ==============================================================================

def test_prediction_horizon(
    mi:          ModelInfo,
    tasks:       list,
    task_labels: List[str],
    dv:          dict,
    device:      torch.device,
    ctx_cycles:  int = TRAIN_CTX_CYC,
    step_cycles: int = HORIZON_STEP,
    n_passes:    int = 5,
) -> pd.DataFrame:
    """
    TEST 2: How does MAE grow with prediction horizon?

    Context fixed at ctx_cycles. Target split into step_cycles-cycle blocks.
    Works for both measurement-level and cycle-aggregated models.
    """
    ctx_rows  = ctx_cycles  * mi.rows_per_cycle
    step_rows = step_cycles * mi.rows_per_cycle
    rows = []

    for t_label, (X, y) in zip(task_labels, tasks):
        T       = len(X)
        ctx_end = min(ctx_rows, T)
        X_rest  = X[ctx_end:]; y_rest = y[ctx_end:]
        n_rest  = len(X_rest)
        if n_rest == 0:
            continue
        print(f"    {t_label}: ctx={ctx_cycles} cyc, "
              f"remaining={n_rest // mi.rows_per_cycle} cyc")

        mean_full, std_full = predict_mi(mi, X[:ctx_end], y[:ctx_end],
                                          X_rest, device, n_passes)

        for block_start in range(0, n_rest, step_rows):
            block_end = min(block_start + step_rows, n_rest)
            if block_end <= block_start:
                break
            pred_b = mean_full[block_start:block_end]
            true_b = y_rest[block_start:block_end]
            std_b  = std_full[block_start:block_end]
            mae    = compute_mae_dn(pred_b, true_b, mi.all_target_cols, dv)
            h_s    = ctx_cycles + block_start // mi.rows_per_cycle
            h_e    = ctx_cycles + block_end   // mi.rows_per_cycle
            for col, val in mae.items():
                rows.append({"horizon_start_cyc": h_s, "horizon_end_cyc": h_e,
                             "horizon_mid_cyc": (h_s+h_e)/2, "task": t_label,
                             "target": col, "mae": val,
                             "pred_std_mean": float(np.nanmean(std_b))})

    df = pd.DataFrame(rows)
    df.to_csv(mi.out_dir / "test2_prediction_horizon.csv", index=False)

    target_cols_plot = [c for c in mi.all_target_cols
                        if not df[df["target"]==c]["mae"].isna().all()]
    fig, axes = plt.subplots(1, len(target_cols_plot),
                             figsize=(9 * len(target_cols_plot), 5))
    if len(target_cols_plot) == 1:
        axes = [axes]
    colors = plt.get_cmap("tab10", len(task_labels))
    for ax, col in zip(axes, target_cols_plot):
        col_df = df[df["target"] == col].dropna(subset=["mae"])
        for ti, t_label in enumerate(task_labels):
            t_df = col_df[col_df["task"]==t_label].sort_values("horizon_mid_cyc")
            ax.plot(t_df["horizon_mid_cyc"], t_df["mae"],
                    color=colors(ti), alpha=0.45, linewidth=1.0,
                    label=t_label, marker=".")
        mean_df = col_df.groupby("horizon_mid_cyc")["mae"].agg(["mean","std"])
        ax.plot(mean_df.index, mean_df["mean"], color=C_MAIN, linewidth=2.5,
                zorder=5, label="Mean")
        ax.fill_between(mean_df.index, mean_df["mean"]-mean_df["std"],
                        mean_df["mean"]+mean_df["std"], alpha=0.15, color=C_MAIN)
        ax.axvline(ctx_cycles + TRAIN_CTX_CYC, color=C_AMBER, linestyle="--",
                   linewidth=1.5, label=f"Training target end ({ctx_cycles+TRAIN_CTX_CYC})")
        ax.axvline(ctx_cycles, color=C_GREEN, linestyle=":", linewidth=1.2,
                   label=f"Context end ({ctx_cycles})")
        ax.set_xlabel("Prediction horizon (cycle number)")
        ax.set_ylabel(f"MAE [{col}]")
        ax.set_title(f"Test 2 — Horizon [{mi.label}] — {col}\n"
                     f"(context = first {ctx_cycles} cycles, fixed)")
        ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.25)
    save(fig, mi.out_dir / "test2_prediction_horizon.png")
    return df


# ==============================================================================
# TEST 3 — UNCERTAINTY CALIBRATION
# ==============================================================================

def test_uncertainty_calibration(
    mi:          ModelInfo,
    tasks:       list,
    task_labels: List[str],
    dv:          dict,
    device:      torch.device,
    ctx_cycles:  int = TRAIN_CTX_CYC,
    tgt_cycles:  int = 60,
    n_bins:      int = 10,
    n_passes:    int = 5,
) -> pd.DataFrame:
    """
    TEST 3: Is the ANP's predicted uncertainty informative?
    Only runs for targets the model actually predicts (skips NaN columns).
    """
    ctx_rows = ctx_cycles * mi.rows_per_cycle
    tgt_rows = tgt_cycles * mi.rows_per_cycle
    rows = []

    for t_label, (X, y) in zip(task_labels, tasks):
        T = len(X); ctx_end = min(ctx_rows, T); tgt_end = min(ctx_end+tgt_rows, T)
        if tgt_end <= ctx_end:
            continue
        mean, std = predict_mi(mi, X[:ctx_end], y[:ctx_end],
                                X[ctx_end:tgt_end], device, n_passes)
        for i, col in enumerate(mi.all_target_cols):
            if np.all(np.isnan(mean[:, i])):
                continue
            pred_dn = denormalize(mean[:, i], col, dv)
            true_dn = denormalize(y[ctx_end:tgt_end, i], col, dv)
            std_dn  = std[:, i] * dv["y_std"].get(col, 1.0)
            abs_err = np.abs(pred_dn - true_dn)
            for j in range(len(pred_dn)):
                rows.append({"task": t_label, "target": col,
                             "pred_std": float(std_dn[j]),
                             "abs_err": float(abs_err[j])})

    df = pd.DataFrame(rows)
    df.to_csv(mi.out_dir / "test3_uncertainty_raw.csv", index=False)

    cal_rows = []
    for col in df["target"].unique():
        col_df    = df[df["target"] == col]
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(col_df["pred_std"], quantiles)
        for b in range(n_bins):
            lo, hi = bin_edges[b], bin_edges[b+1]
            mask = (col_df["pred_std"] >= lo) & (col_df["pred_std"] < hi)
            if mask.sum() == 0:
                continue
            cal_rows.append({"target": col, "bin_lo": lo, "bin_hi": hi,
                             "mean_std": col_df[mask]["pred_std"].mean(),
                             "mean_abs_err": col_df[mask]["abs_err"].mean(),
                             "n": int(mask.sum())})
        try:
            from scipy import stats
            rho, pval = stats.spearmanr(col_df["pred_std"], col_df["abs_err"])
            print(f"    {col}: Spearman ρ(std, |error|) = {rho:.3f}  p={pval:.4f}")
        except ImportError:
            pass

    cal_df = pd.DataFrame(cal_rows)
    cal_df.to_csv(mi.out_dir / "test3_calibration_bins.csv", index=False)

    target_cols_plot = cal_df["target"].unique().tolist()
    fig, axes = plt.subplots(1, len(target_cols_plot),
                             figsize=(6 * len(target_cols_plot), 5))
    if len(target_cols_plot) == 1:
        axes = [axes]
    for ax, col in zip(axes, target_cols_plot):
        col_cal = cal_df[cal_df["target"] == col]
        ax.scatter(col_cal["mean_std"], col_cal["mean_abs_err"],
                   s=80, color=C_MAIN, zorder=3)
        ax.plot(col_cal["mean_std"], col_cal["mean_abs_err"],
                color=C_MAIN, linewidth=1.5, alpha=0.7)
        lim = max(col_cal["mean_std"].max(), col_cal["mean_abs_err"].max()) * 1.1
        ax.plot([0, lim], [0, lim], color=C_GREY, linestyle="--",
                linewidth=1.0, label="Perfect calibration (std = |error|)")
        ax.set_xlabel(f"Mean predicted std [{col}]")
        ax.set_ylabel(f"Mean |error| [{col}]")
        ax.set_title(f"Test 3 — Uncertainty calibration [{mi.label}] — {col}\n"
                     f"(above diagonal = underconfident, below = overconfident)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
    save(fig, mi.out_dir / "test3_uncertainty_calibration.png")
    return cal_df


# ==============================================================================
# TEST 4 — PRIOR vs POSTERIOR KL
# ==============================================================================

def test_prior_posterior_kl(
    mi:          ModelInfo,
    tasks:       list,
    task_labels: List[str],
    dv:          dict,
    device:      torch.device,
    ctx_sizes:   List[int] = CONTEXT_SIZES,
    tgt_cycles:  int       = 60,
    n_passes:    int       = 5,
) -> pd.DataFrame:
    """
    TEST 4: Does the posterior diverge from the prior with more context?
    KL → 0 for all context sizes = KL collapse.
    """
    tgt_rows = tgt_cycles * mi.rows_per_cycle
    rows = []
    for ctx_cyc in ctx_sizes:
        ctx_rows = ctx_cyc * mi.rows_per_cycle
        for t_label, (X, y) in zip(task_labels, tasks):
            T = len(X); ctx_end = min(ctx_rows, T); tgt_end = min(ctx_end+tgt_rows, T)
            if tgt_end <= ctx_end:
                continue
            kl = kl_mi(mi, X[:ctx_end], y[:ctx_end],
                        X[ctx_end:tgt_end], y[ctx_end:tgt_end], device, n_passes)
            rows.append({"ctx_cycles": ctx_cyc, "task": t_label, "kl": kl})

    df = pd.DataFrame(rows)
    df.to_csv(mi.out_dir / "test4_prior_posterior_kl.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors  = plt.get_cmap("tab10", len(task_labels))
    for ti, t_label in enumerate(task_labels):
        t_df = df[df["task"] == t_label]
        ax.plot(t_df["ctx_cycles"], t_df["kl"], color=colors(ti),
                alpha=0.45, linewidth=1.0, marker="o", markersize=4, label=t_label)
    mean_kl = df.groupby("ctx_cycles")["kl"].agg(["mean","std"])
    ax.plot(mean_kl.index, mean_kl["mean"], color=C_MAIN, linewidth=2.5,
            zorder=5, marker="o", markersize=7, label="Mean KL")
    ax.fill_between(mean_kl.index, mean_kl["mean"]-mean_kl["std"],
                    mean_kl["mean"]+mean_kl["std"], alpha=0.15, color=C_MAIN)
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Context size (cycles)"); ax.set_ylabel("KL(posterior || prior) [nats]")
    ax.set_title(f"Test 4 — Prior vs Posterior KL [{mi.label}]\n"
                 f"KL → 0 for all context sizes indicates KL collapse")
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.25); ax.set_xticks(ctx_sizes)
    save(fig, mi.out_dir / "test4_prior_posterior_kl.png")
    return df


# ==============================================================================
# TEST 5 — CROSS-TASK ROBUSTNESS
# ==============================================================================

def test_cross_task_robustness(
    mi:          ModelInfo,
    tasks:       list,
    task_labels: List[str],
    dv:          dict,
    device:      torch.device,
    ctx_cycles:  int = TRAIN_CTX_CYC,
    tgt_cycles:  int = 60,
    n_passes:    int = 5,
) -> pd.DataFrame:
    """
    TEST 5: Does matched context (own task) outperform mismatched context?
    Confirms whether the model truly performs meta-learning.
    """
    ctx_rows = ctx_cycles * mi.rows_per_cycle
    tgt_rows = tgt_cycles * mi.rows_per_cycle
    rows = []

    for pi, (t_pred_label, (X_pred, y_pred)) in enumerate(zip(task_labels, tasks)):
        T = len(X_pred); ctx_end = min(ctx_rows, T); tgt_end = min(ctx_end+tgt_rows, T)
        if tgt_end <= ctx_end:
            continue
        X_tgt = X_pred[ctx_end:tgt_end]; y_tgt = y_pred[ctx_end:tgt_end]

        for ci, (t_ctx_label, (X_ctx_src, y_ctx_src)) in enumerate(zip(task_labels, tasks)):
            ctx_end_c = min(ctx_rows, len(X_ctx_src))
            mean, _   = predict_mi(mi, X_ctx_src[:ctx_end_c], y_ctx_src[:ctx_end_c],
                                    X_tgt, device, n_passes)
            mae = compute_mae_dn(mean, y_tgt, mi.all_target_cols, dv)
            for col, val in mae.items():
                if not np.isnan(val):
                    rows.append({"pred_task": t_pred_label, "ctx_task": t_ctx_label,
                                 "matched": pi==ci, "target": col, "mae": val})

    df = pd.DataFrame(rows)
    df.to_csv(mi.out_dir / "test5_cross_task_robustness.csv", index=False)

    target_cols_plot = df["target"].unique().tolist()
    fig, axes = plt.subplots(1, len(target_cols_plot),
                             figsize=(7 * len(target_cols_plot), 5))
    if len(target_cols_plot) == 1:
        axes = [axes]
    for ax, col in zip(axes, target_cols_plot):
        col_df = df[df["target"] == col]
        matched_mae  = col_df[col_df["matched"]]["mae"]
        mismatch_mae = col_df[~col_df["matched"]]["mae"]
        bp = ax.boxplot([matched_mae.values, mismatch_mae.values],
                        tick_labels=["Own context\n(matched)", "Cross context\n(mismatched)"],
                        patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))
        for patch, color in zip(bp["boxes"], [C_MAIN, C_VAL]):
            patch.set_facecolor(color); patch.set_alpha(0.6)
        ax.set_ylabel(f"MAE [{col}]")
        ax.set_title(f"Test 5 — Cross-task robustness [{mi.label}] — {col}\n"
                     f"Matched: {matched_mae.mean():.3f}  Mismatched: {mismatch_mae.mean():.3f}")
        ax.grid(True, axis="y", alpha=0.25)
    save(fig, mi.out_dir / "test5_cross_task_robustness.png")
    return df


# ==============================================================================
# PER-MODEL RUN
# ==============================================================================

def run_for_model(
    mi:          ModelInfo,
    tasks_raw:   list,
    tasks_agg:   Optional[list],
    task_labels: List[str],
    dv:          dict,
    device:      torch.device,
    tests:       List[int],
    n_passes:    int,
) -> None:
    """
    Run all selected tests for a single ModelInfo.

    Automatically selects cycle-aggregated task data for models trained with aggregate_by_cycle=True.

    Args:
        tasks_raw:  Measurement-level task data (Nc×MEAS, ...).
        tasks_agg:  Cycle-level aggregated task data, or None if not needed.
        task_labels: Labels for each task.
        dv:          Denormalization values.
        device:      Torch device.
        tests:       Which test indices to run.
        n_passes:    Stochastic forward pass count.
    """
    tasks = tasks_agg if mi.aggregate else tasks_raw
    if tasks is None:
        print(f"  ⚠  {mi.label}: aggregate=True but tasks_agg not available — skipping")
        return

    print(f"\n{'='*62}")
    print(f"  MODEL: {mi.label}")
    print(f"  targets={mi.model_target_cols}  "
          f"features={'reduced' if mi.feature_cols else 'full'}  "
          f"agg={mi.aggregate}  rows_per_cycle={mi.rows_per_cycle}")
    print(f"  output → {mi.out_dir}")
    print(f"{'='*62}")

    if 1 in tests:
        print(f"\n  ── TEST 1 — Context impact")
        test_context_impact(mi, tasks, task_labels, dv, device, n_passes=n_passes)

    if 2 in tests:
        print(f"\n  ── TEST 2 — Prediction horizon")
        test_prediction_horizon(mi, tasks, task_labels, dv, device, n_passes=n_passes)

    if 3 in tests:
        print(f"\n  ── TEST 3 — Uncertainty calibration")
        try:
            test_uncertainty_calibration(mi, tasks, task_labels, dv, device, n_passes=n_passes)
        except ImportError:
            print("    ⚠  scipy not available — skipping. pip install scipy")

    if 4 in tests:
        print(f"\n  ── TEST 4 — Prior vs Posterior KL")
        test_prior_posterior_kl(mi, tasks, task_labels, dv, device, n_passes=n_passes)

    if 5 in tests:
        print(f"\n  ── TEST 5 — Cross-task robustness")
        test_cross_task_robustness(mi, tasks, task_labels, dv, device, n_passes=n_passes)

    print(f"\n  ✅  {mi.label} complete → {mi.out_dir}\n")


# ==============================================================================
# MAIN RUN
# ==============================================================================

def run(
    model_run_dirs: Dict[str, Optional[Path]],
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
    Run inference evaluation for all specified ANP model variants.

    Args:
        model_run_dirs: Dict mapping label → run_dir (None = skip this model).
        data_dir:       Path to prepared_data.pkl directory.
        out_dir:        Base output directory; each model gets a subdirectory.
        tests:          Test indices to run for every model (1-5).
        eval_split:     'train' | 'val' | 'test' | 'all'.
        n_passes:       Stochastic forward passes to average per prediction.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧  Device   : {device}")
    print(f"📁  Base dir : {out_dir}")
    print(f"   Tests    : {tests}")
    print(f"   Split    : {eval_split}")
    print(f"   n_passes : {n_passes}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\n📂  Loading data from: {data_dir}")
    data = load_prepared_data(data_dir)
    validate_targets(data)

    all_target_cols = list(data["normalized_synth_datasets"][0][1].columns)
    input_dim       = data["normalized_synth_datasets"][0][0].shape[1]
    all_col_names   = list(data["normalized_synth_datasets"][0][0].columns)
    dv = {"y_mean": data["denorm_values"]["y_mean"],
          "y_std":  data["denorm_values"]["y_std"]}

    print(f"   Targets  : {all_target_cols}")
    print(f"   input_dim: {input_dim}")

    # Task selection
    split_map = {
        "train": train_task_ids,
        "val":   val_task_ids,
        "test":  test_task_ids,
        "all":   train_task_ids + val_task_ids + test_task_ids,
    }
    eval_ids = split_map[eval_split]

    def label_fn(i):
        if i in train_task_ids: return f"train_{train_task_ids.index(i)+1:02d}"
        if i in val_task_ids:   return f"val_{val_task_ids.index(i)+1:02d}"
        if i in test_task_ids:  return f"test_{test_task_ids.index(i)+1:02d}"
        return f"task_{i:02d}"

    task_labels = [label_fn(i) for i in eval_ids]
    print(f"   Tasks    : {task_labels}\n")

    # Measurement-level tasks (always built)
    tasks_raw = []
    for i in eval_ids:
        X_df, y_df = sort_task_by_cycle(*data["normalized_synth_datasets"][i])
        tasks_raw.append((X_df.values.astype(np.float32),
                          y_df.values.astype(np.float32)))

    # ── Load models ────────────────────────────────────────────────────────────
    print("📦  Loading models...")
    model_infos: List[ModelInfo] = []
    for label, run_dir in model_run_dirs.items():
        if run_dir is None:
            continue
        mi = load_anp_model(run_dir, input_dim, all_target_cols, all_col_names,
                            device, label, out_dir)
        if mi is not None:
            model_infos.append(mi)

    if not model_infos:
        print("\n⚠  No models loaded — check paths.")
        return

    # ── Cycle-aggregated tasks (built once if needed) ──────────────────────────
    needs_agg = any(mi.aggregate for mi in model_infos)
    if needs_agg:
        print("\n🔄  Pre-computing cycle-level aggregated tasks...")
        tasks_agg = []
        for i in eval_ids:
            X_df, y_df = sort_task_by_cycle(*data["normalized_synth_datasets"][i])
            X_a, y_a   = aggregate_by_cycle(X_df, y_df)
            tasks_agg.append((X_a.values.astype(np.float32),
                               y_a.values.astype(np.float32)))
        print(f"   {len(tasks_agg)} tasks aggregated "
              f"({tasks_agg[0][0].shape[0]} rows/task)")
    else:
        tasks_agg = None

    # ── Run all models ─────────────────────────────────────────────────────────
    for mi in model_infos:
        run_for_model(mi, tasks_raw, tasks_agg, task_labels,
                      dv, device, tests, n_passes)

    print(f"\n✅  All models complete. Results in: {out_dir}\n")


# ==============================================================================
# CLI
# ==============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "ANP inference evaluation — tests 1-5 for all ANP model variants. "
            "Each model gets its own output subdirectory."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model paths
    p.add_argument("--anp_run", type=str, default=None, help="ANP dual-target run directory")
    p.add_argument("--anp_soc_run", type=str, default=None, help="ANP SoC-only run directory")
    p.add_argument("--anp_cycle_run", type=str, default=None, help="ANP Cycle-only run directory")
    p.add_argument("--anp_soc_reduced_run", type=str, default=None, help="ANP SoC-only reduced-features run directory")
    p.add_argument("--anp_cycle_reduced_run",type=str, default=None, help="ANP Cycle-only reduced-features run directory (auto-detects aggregate_by_cycle from config.json)")
    # Data and output
    p.add_argument("--data_dir", type=str, default="../csic_real_synth_load/prepared_data")
    p.add_argument("--out_dir", type=str, default="", help="Base output directory (default: ./anp_eval/)")
    # Evaluation settings
    p.add_argument("--tests", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="Tests to run (1-5)")
    p.add_argument("--split",    type=str, default="val", choices=["train", "val", "test", "all"], help="Task split to evaluate")
    p.add_argument("--n_passes", type=int, default=5, help="Stochastic forward passes to average per prediction")
    # Task split config
    p.add_argument("--train_ids", type=int, nargs="+", default=list(range(17)))
    p.add_argument("--val_ids",   type=int, nargs="+", default=list(range(17, 22)))
    p.add_argument("--test_ids",  type=int, nargs="+", default=list(range(22, 25)))
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(__file__).resolve().parent / "anp_eval")

    model_run_dirs = {
        "anp":              Path(args.anp_run)              if args.anp_run              else None,
        "anp_soc":          Path(args.anp_soc_run)          if args.anp_soc_run          else None,
        "anp_cycle":        Path(args.anp_cycle_run)        if args.anp_cycle_run        else None,
        "anp_soc_reduced":  Path(args.anp_soc_reduced_run)  if args.anp_soc_reduced_run  else None,
        "anp_cycle_reduced":Path(args.anp_cycle_reduced_run)if args.anp_cycle_reduced_run else None,
    }

    run(
        model_run_dirs = model_run_dirs,
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
