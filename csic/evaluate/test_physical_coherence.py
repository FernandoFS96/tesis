"""
test_physical_coherence.py
==============================================================================
Tests whether trained models produce physically incoherent predictions during inference and reports how frequently violations occur.

Physical constraints checked:
    SoC (%):
        1. Range violation     — predicted SoC outside [0, 100] %
        2. Large spike         — |ΔSoC| > spike_threshold % between consecutive measurements within the same cycle
        3. Monotonicity (soft) — SoC trend across cycles should be roughly non-increasing (battery degrades over time).
                                 Flagged if mean SoC increases > mono_threshold % from first quarter to last quarter of trajectory.

    Cycle:
        4. Negative prediction — predicted Cycle < 0
        5. Non-monotone        — predicted cycle number decreases across the trajectory (cycles should always increase)

Single-target ANP models are handled transparently:
    - ANP-SoC  only predicts SoC  → Cycle checks are skipped (shown as N/A)
    - ANP-Cycle only predicts Cycle → SoC checks are skipped (shown as N/A)

Models evaluated:
    - ANP dual-target  (--anp_run)
    - ANP SoC-only     (--anp_soc_run)
    - ANP Cycle-only   (--anp_cycle_run)
    - DR-MLP           (--mlp_run/dr_mlp/best.pt)
    - Optionally: a specific specialist (--specialist_id)

Output (saved to --out_dir):
    coherence_report.txt      — full violation report per model and task
    coherence_summary.csv     — aggregated violation rates per model
    violation_plots/          — time-series plots showing violation examples
    comparison_plots/         — all models overlaid on the same axes per task

Usage:
    python test_physical_coherence.py \
        --mlp_run             ../train/runs_mlp/20260511_121741 \
        --anp_run             ../train/runs/anp_all/20260512_124715 \
        --anp_soc_run         ../train/runs/anp_SoC/20260512_114601 \
        --anp_cycle_run       ../train/runs/anp_Cycle/20260512_114703 \
        --anp_soc_reduced_run   ../train/runs/anp_SoC_reduced/20260513_110544 \
        --anp_cycle_reduced_run ../train/runs/anp_Cycle_reduced/20260514_122442 \
        --data_dir            ../csic_real_synth_load/prepared_data

Location: csic/validation/test_physical_coherence.py
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
import matplotlib.gridspec as gridspec
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
)

from models.anp import LatentModel

try:
    from train.train_mlp import MLP  #type: ignore 
except ImportError:
    import torch.nn.init as init

    class MLP(nn.Module):
        def __init__(self, input_dim, output_dim, neurons=128, dropout=0.1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, neurons), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(neurons, neurons),   nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(neurons, neurons),   nn.ReLU(),
                nn.Linear(neurons, output_dim),
            )
        def forward(self, x): return self.net(x)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class CoherenceConfig:
    # Physical thresholds
    soc_min:          float = 0.0    # minimum valid SoC (%)
    soc_max:          float = 100.0  # maximum valid SoC (%)
    spike_threshold:  float = 30.0   # max |ΔSoC| between consecutive rows (%)
    mono_threshold:   float = 10.0   # max allowed SoC increase across trajectory (%)
    cycle_min:        float = 0.0    # minimum valid cycle number

    # Evaluation window
    ctx_cycles:             int = 60
    tgt_cycles:             int = 60
    measurements_per_cycle: int = 30

    # Which tasks to evaluate
    train_task_ids: List[int] = field(default_factory=lambda: list(range(17)))
    val_task_ids:   List[int] = field(default_factory=lambda: list(range(17, 22)))
    test_task_ids:  List[int] = field(default_factory=lambda: list(range(22, 25)))

    @property
    def ctx_rows(self) -> int:
        return self.ctx_cycles * self.measurements_per_cycle

    @property
    def tgt_rows(self) -> int:
        return self.tgt_cycles * self.measurements_per_cycle


# ==============================================================================
# DATA HELPERS
# ==============================================================================

def denormalize(
    arr:           np.ndarray,
    col:           str,
    denorm_values: dict,
) -> np.ndarray:
    """Denormalize a 1D array for a given target column."""
    m = denorm_values["y_mean"].get(col, 0.0)
    s = denorm_values["y_std"].get(col, 1.0)
    return arr * s + m


def extract_window(
    X:        pd.DataFrame,
    y:        pd.DataFrame,
    ctx_rows: int,
    tgt_rows: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    T       = len(X)
    ctx_end = min(ctx_rows, T)
    tgt_end = min(ctx_end + tgt_rows, T)
    Xa = X.values.astype(np.float32)
    ya = y.values.astype(np.float32)
    return Xa[:ctx_end], ya[:ctx_end], Xa[ctx_end:tgt_end], ya[ctx_end:tgt_end]


# ==============================================================================
# MODEL LOADING
# ==============================================================================
def load_anp_model(
    run_dir:          Path,
    input_dim:        int,
    all_target_cols:  List[str],
    device:           torch.device,
    label:            str = "ANP",
) -> Optional[Tuple[str, nn.Module, List[str], Optional[List[str]]]]:
    """
    Load an ANP checkpoint and detect which targets and features it was trained on.

    Reads config.json for num_hidden, attn_dropout, and target_col.
    Infers model_input_dim directly from checkpoint weight shapes, robust to missing 'use_reduced_features' field in config.json.

    Returns:
        (label, model, model_target_cols, feature_cols) or None if not found.
        feature_cols is None for full-feature models, or a list of column names for reduced-feature models.
    """
    ckpt_path = run_dir / "best.pt"
    cfg_path  = run_dir / "config.json"

    if not ckpt_path.exists():
        print(f"  ⚠  {label}: checkpoint not found: {ckpt_path}")
        return None

    num_hidden   = 128
    attn_dropout = 0.1
    target_col   = "all"

    if cfg_path.exists():
        with cfg_path.open() as f:
            cfg_data = json.load(f)
        num_hidden   = (cfg_data.get("num_hidden")
                        or cfg_data.get("params", {}).get("num_hidden", 128))
        attn_dropout = cfg_data.get("attn_dropout", 0.1)
        target_col   = cfg_data.get("target_col", "all")

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

    output_dim = len(model_target_cols)

    # ── Robustly determine model_input_dim from the checkpoint itself ──────────
    # Works even if use_reduced_features was not saved in config.json.
    raw     = torch.load(ckpt_path, map_location="cpu")
    lat_key = next(
        (k for k in raw["model"]
         if "latent_encoder.input_projection.linear_layer.weight" in k), None
    )
    if lat_key:
        # weight shape: [num_hidden, input_dim + output_dim]
        model_input_dim = raw["model"][lat_key].shape[1] - output_dim
    else:
        model_input_dim = input_dim  # fallback: assume full features

    # Determine feature_cols for X filtering during evaluation
    if model_input_dim == input_dim:
        feature_cols = None          # full feature set — no filtering needed
    else:
        feature_cols = REDUCED_FEATURE_SETS.get(target_col)
        if feature_cols is None:
            raise ValueError(
                f"'{label}' has input_dim={model_input_dim} (not {input_dim}), "
                f"but REDUCED_FEATURE_SETS has no entry for target_col='{target_col}'. "
                f"Add the reduced feature list to REDUCED_FEATURE_SETS in train_utils.py."
            )
        if len(feature_cols) != model_input_dim:
            raise ValueError(
                f"'{label}': checkpoint input_dim={model_input_dim} but "
                f"REDUCED_FEATURE_SETS['{target_col}'] has {len(feature_cols)} features. "
                f"These must match — check REDUCED_FEATURE_SETS in train_utils.py."
            )

    model = LatentModel(num_hidden=num_hidden, input_dim=model_input_dim,
                        output_dim=output_dim, attn_dropout=attn_dropout)
    model.load_state_dict(raw["model"])   # reuse already-loaded raw dict
    model.eval().to(device)

    val_mae  = raw.get("val_MAE", raw.get("val_loss", "?"))
    feat_str = f"reduced({model_input_dim})" if feature_cols else f"all({input_dim})"
    print(f"  ✓  {label:<22} targets={model_target_cols}  "
          f"num_hidden={num_hidden}  features={feat_str}  val_MAE={val_mae}")
    return (label, model, model_target_cols, feature_cols)


def load_mlp(ckpt_path: Path, input_dim: int, output_dim: int,
             device: torch.device, neurons: int = 128,
             dropout: float = 0.1) -> Optional[nn.Module]:
    if not ckpt_path.exists():
        print(f"  ⚠  MLP checkpoint not found: {ckpt_path}")
        return None
    model = MLP(input_dim, output_dim, neurons, dropout)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    print(f"  ✓  MLP loaded from {ckpt_path}")
    return model


# ==============================================================================
# INFERENCE
# ==============================================================================

@torch.no_grad()
def predict_anp(
    model:             nn.Module,
    X_ctx:             np.ndarray,
    y_ctx:             np.ndarray,
    X_tgt:             np.ndarray,
    device:            torch.device,
    all_target_cols:   List[str],
    model_target_cols: List[str],
) -> np.ndarray:
    """
    Run ANP inference.

    Filters y_ctx to the model's trained targets before the forward pass.
    Returns predictions expanded to (Nt, len(all_target_cols)) with NaN for any target the model does not predict, so downstream checks can treat all models uniformly.

    Args:
        all_target_cols:   All targets present in the data.
        model_target_cols: Subset of targets this model was trained on.

    Returns:
        pred (Nt, len(all_target_cols)) in normalized space, NaN for
        columns not predicted by this model.
    """
    # Filter y_ctx to model's targets
    if model_target_cols != all_target_cols:
        col_idx = [all_target_cols.index(c) for c in model_target_cols]
        y_ctx_m = y_ctx[:, col_idx]
    else:
        y_ctx_m = y_ctx

    ctx_x = torch.tensor(X_ctx).unsqueeze(0).to(device)
    ctx_y = torch.tensor(y_ctx_m).unsqueeze(0).to(device)
    tgt_x = torch.tensor(X_tgt).unsqueeze(0).to(device)
    pred_mean, _, _, _, _ = model(ctx_x, ctx_y, tgt_x, target_y=None)
    pred_model = pred_mean.squeeze(0).cpu().numpy()   # (Nt, O_model)

    # Expand to full target shape — NaN for unmodelled targets
    Nt = len(X_tgt)
    pred_full = np.full((Nt, len(all_target_cols)), float("nan"), dtype=np.float32)
    for model_i, col in enumerate(model_target_cols):
        full_i = all_target_cols.index(col)
        pred_full[:, full_i] = pred_model[:, model_i]

    return pred_full


@torch.no_grad()
def predict_mlp(
    model:  nn.Module,
    X_tgt:  np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Run MLP inference. Returns predictions (Nt, O) in normalized space."""
    X_t = torch.tensor(X_tgt).to(device)
    return model(X_t).cpu().numpy()


# ==============================================================================
# PHYSICAL COHERENCE CHECKS
# ==============================================================================

@dataclass
class ViolationReport:
    """Stores violation counts and details for a single (model, task) pair."""
    model_label:  str
    task_label:   str
    n_predictions: int

    # SoC violations
    soc_below_min:     int = 0   # predictions < 0 %
    soc_above_max:     int = 0   # predictions > 100 %
    soc_spike:         int = 0   # |ΔSoC| > threshold between consecutive rows
    soc_not_monotone:  bool = False  # mean SoC increases across trajectory

    # Cycle violations
    cycle_negative:    int = 0   # predictions < 0
    cycle_not_monotone: bool = False  # cycle numbers decrease across trajectory

    # Ground truth for reference
    soc_pred_min:  float = 0.0
    soc_pred_max:  float = 0.0
    soc_pred_mean: float = 0.0
    cycle_pred_min: float = 0.0
    cycle_pred_max: float = 0.0

    def total_soc_violations(self) -> int:
        return self.soc_below_min + self.soc_above_max + self.soc_spike

    def violation_rate(self) -> float:
        """Fraction of predictions with at least one SoC range violation."""
        return (self.soc_below_min + self.soc_above_max) / max(1, self.n_predictions)

    def has_any_violation(self) -> bool:
        return (self.total_soc_violations() > 0 or
                self.soc_not_monotone or
                self.cycle_negative > 0 or
                self.cycle_not_monotone)


def check_coherence(
    pred_dn:     np.ndarray,     # (Nt, O) denormalized predictions, NaN for missing targets
    target_cols: List[str],
    cfg:         CoherenceConfig,
    model_label: str,
    task_label:  str,
) -> ViolationReport:
    """
    Run all physical coherence checks on a prediction array.

    Columns that are entirely NaN (targets not predicted by single-target models) are silently skipped, their violation counts stay at zero.

    Args:
        pred_dn:     Denormalized predictions, shape (Nt, O). NaN values indicate targets not predicted by this model.
        target_cols: Ordered list of all target column names.
        cfg:         CoherenceConfig with physical thresholds.
        model_label: Label for the model being evaluated.
        task_label:  Label for the task being evaluated.

    Returns:
        ViolationReport with checks populated for predicted targets only.
    """
    Nt = len(pred_dn)
    report = ViolationReport(
        model_label=model_label,
        task_label=task_label,
        n_predictions=Nt,
    )

    # ── SoC checks ────────────────────────────────────────────────────────────
    if "SoC (%)" in target_cols:
        soc_idx  = target_cols.index("SoC (%)")
        soc_pred = pred_dn[:, soc_idx]

        # Skip entirely if this model does not predict SoC
        if not np.all(np.isnan(soc_pred)):
            report.soc_pred_min  = float(np.nanmin(soc_pred))
            report.soc_pred_max  = float(np.nanmax(soc_pred))
            report.soc_pred_mean = float(np.nanmean(soc_pred))

            report.soc_below_min = int((soc_pred < cfg.soc_min).sum())
            report.soc_above_max = int((soc_pred > cfg.soc_max).sum())

            if Nt > 1:
                deltas = np.abs(np.diff(soc_pred))
                report.soc_spike = int((deltas > cfg.spike_threshold).sum())

            q = max(1, Nt // 4)
            mean_first = np.nanmean(soc_pred[:q])
            mean_last  = np.nanmean(soc_pred[-q:])
            report.soc_not_monotone = bool(
                (mean_last - mean_first) > cfg.mono_threshold
            )

    # ── Cycle checks ──────────────────────────────────────────────────────────
    if "Cycle" in target_cols:
        cyc_idx  = target_cols.index("Cycle")
        cyc_pred = pred_dn[:, cyc_idx]

        # Skip entirely if this model does not predict Cycle
        if not np.all(np.isnan(cyc_pred)):
            report.cycle_pred_min = float(np.nanmin(cyc_pred))
            report.cycle_pred_max = float(np.nanmax(cyc_pred))

            report.cycle_negative = int((cyc_pred < cfg.cycle_min).sum())

            if Nt > 1:
                x     = np.arange(Nt)
                slope = np.polyfit(x, cyc_pred, 1)[0]
                report.cycle_not_monotone = bool(slope < -1.0)

    return report


# ==============================================================================
# PLOTTING
# ==============================================================================

def plot_violation_example(
    pred_dn:     np.ndarray,
    true_dn:     np.ndarray,
    target_cols: List[str],
    report:      ViolationReport,
    cfg:         CoherenceConfig,
    out_path:    Path,
) -> None:
    """
    Plot predicted vs ground truth time series, highlighting violation regions.

    Args:
        pred_dn:     Denormalized predictions (Nt, O).
        true_dn:     Denormalized ground truth (Nt, O).
        target_cols: Target column names.
        report:      ViolationReport for this (model, task) pair.
        cfg:         CoherenceConfig with physical thresholds.
        out_path:    Path where the PNG will be saved.
    """
    n_targets = len(target_cols)
    fig = plt.figure(figsize=(13, 4 * n_targets))
    gs  = gridspec.GridSpec(n_targets, 1, figure=fig, hspace=0.45)

    x = np.arange(len(pred_dn))

    for i, col in enumerate(target_cols):
        ax = fig.add_subplot(gs[i])
        p  = pred_dn[:, i]
        t  = true_dn[:, i]

        ax.plot(x, t, color="#9AB8C8", linewidth=1.2, alpha=0.8,
                label="Ground truth", zorder=2)
        ax.plot(x, p, color="#1C7293", linewidth=1.4,
                label="Prediction", zorder=3)

        if "SoC" in col:
            # Shade physical bounds
            ax.axhline(cfg.soc_min, color="#C0392B", linestyle="--",
                       linewidth=1.0, alpha=0.6, label=f"Min ({cfg.soc_min}%)")
            ax.axhline(cfg.soc_max, color="#C0392B", linestyle="--",
                       linewidth=1.0, alpha=0.6, label=f"Max ({cfg.soc_max}%)")

            # Highlight out-of-range predictions
            below = p < cfg.soc_min
            above = p > cfg.soc_max
            if below.any():
                ax.scatter(x[below], p[below], color="#C0392B", s=20,
                           zorder=5, label=f"Below 0% ({below.sum()} pts)")
            if above.any():
                ax.scatter(x[above], p[above], color="#D4860A", s=20,
                           zorder=5, label=f"Above 100% ({above.sum()} pts)")

            # Highlight spikes
            if len(p) > 1:
                deltas    = np.abs(np.diff(p))
                spike_idx = np.where(deltas > cfg.spike_threshold)[0]
                for si in spike_idx:
                    ax.axvspan(si, si + 1, alpha=0.25, color="#C0392B")

        ax.set_ylabel(col)
        ax.set_xlabel("Target row index")
        ax.set_title(
            f"{report.model_label} — {report.task_label} — {col}\n"
            f"Range violations: {report.soc_below_min + report.soc_above_max}  "
            f"Spikes: {report.soc_spike}  "
            f"Not monotone: {report.soc_not_monotone}"
            if "SoC" in col else
            f"{report.model_label} — {report.task_label} — {col}\n"
            f"Negative: {report.cycle_negative}  "
            f"Not monotone: {report.cycle_not_monotone}"
        )
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        f"Physical coherence — {report.model_label} on {report.task_label}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_comparison_coherence(
    preds_dn:    Dict[str, np.ndarray],    # {model_label: (Nt, O)}
    true_dn:     np.ndarray,               # (Nt, O)
    reports:     Dict[str, ViolationReport],
    target_cols: List[str],
    cfg:         CoherenceConfig,
    out_path:    Path,
) -> None:
    """
    Plot ANP and DR-MLP predictions side by side on the same axes for direct comparison of physical coherence.

    Args:
        preds_dn: Dict mapping model label to denormalized predictions (Nt, O).
        true_dn:  Denormalized ground truth (Nt, O).
        reports:  Dict mapping model label to its ViolationReport.
        target_cols: Target column names.
        cfg:      CoherenceConfig with physical thresholds.
        out_path: Output PNG path.
    """
    # Color palette — one per model, consistent across subplots
    MODEL_COLORS = {
        "ANP":       "#1C7293",   # teal  — dual-target
        "ANP-SoC":   "#028090",   # darker teal — SoC only
        "ANP-Cycle": "#21295C",   # navy  — Cycle only
        "DR-MLP":    "#D4860A",   # amber
        "Specialist_01": "#237A3D",
    }
    DEFAULT_COLORS = ["#8E44AD", "#E74C3C", "#16A085"]
    model_labels   = list(preds_dn.keys())

    def model_color(label: str) -> str:
        if label in MODEL_COLORS:
            return MODEL_COLORS[label]
        idx = model_labels.index(label) % len(DEFAULT_COLORS)
        return DEFAULT_COLORS[idx]

    n_targets  = len(target_cols)
    task_label = next(iter(reports.values())).task_label

    fig = plt.figure(figsize=(14, 4 * n_targets))
    gs  = gridspec.GridSpec(n_targets, 1, figure=fig, hspace=0.50)

    x = np.arange(len(true_dn))

    for i, col in enumerate(target_cols):
        ax = fig.add_subplot(gs[i])

        # Ground truth
        ax.plot(x, true_dn[:, i], color="#9AB8C8", linewidth=1.5,
                alpha=0.8, label="Ground truth", zorder=2)

        if "SoC" in col:
            ax.axhline(cfg.soc_min, color="#C0392B", linestyle="--",
                       linewidth=1.0, alpha=0.5,
                       label=f"Bounds [{cfg.soc_min}, {cfg.soc_max}%]")
            ax.axhline(cfg.soc_max, color="#C0392B", linestyle="--",
                       linewidth=1.0, alpha=0.5)

        # One line per model — skip if model did not predict this target (all NaN)
        for m_label, pred_dn in preds_dn.items():
            color  = model_color(m_label)
            report = reports.get(m_label)
            p      = pred_dn[:, i]

            # Skip this model for this target if it was not predicted
            if np.all(np.isnan(p)):
                continue

            # Violation summary for legend
            if report and "SoC" in col:
                n_viol = report.soc_below_min + report.soc_above_max
                viol_str = f" [⚠ {n_viol} range viol.]" if n_viol > 0 else " [✓ clean]"
            elif report and "Cycle" in col:
                viol_str = f" [⚠ {report.cycle_negative} neg.]" \
                    if report.cycle_negative > 0 else " [✓ clean]"
            else:
                viol_str = ""

            ax.plot(x, p, color=color, linewidth=1.6,
                    label=f"{m_label}{viol_str}", zorder=3)

            # Mark out-of-range points for this model
            if "SoC" in col:
                below = p < cfg.soc_min
                above = p > cfg.soc_max
                if below.any():
                    ax.scatter(x[below], p[below], color=color, marker="v",
                               s=30, zorder=5, edgecolors="black",
                               linewidths=0.4)
                if above.any():
                    ax.scatter(x[above], p[above], color=color, marker="^",
                               s=30, zorder=5, edgecolors="black",
                               linewidths=0.4)

        ax.set_ylabel(col)
        ax.set_xlabel("Target row index")
        ax.set_title(f"{task_label} — {col}")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.25)

    # Build violation summary for suptitle
    summary_parts = []
    for m_label, rep in reports.items():
        if rep.has_any_violation():
            summary_parts.append(f"{m_label}: ⚠")
        else:
            summary_parts.append(f"{m_label}: ✓")

    fig.suptitle(
        f"Physical coherence comparison — {task_label}\n"
        + "  |  ".join(summary_parts),
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ==============================================================================
# MAIN EVALUATION
# ==============================================================================

def run(
    anp_run_dir:              Optional[Path],
    anp_soc_run_dir:          Optional[Path],
    anp_cycle_run_dir:        Optional[Path],
    anp_soc_reduced_run_dir:  Optional[Path],
    anp_cycle_reduced_run_dir: Optional[Path],
    mlp_run_dir:              Optional[Path],
    data_dir:                 str,
    out_dir:                  Path,
    cfg:                      CoherenceConfig,
    all_tasks:                bool = False,
    specialist_id:            Optional[int] = None,
    plot_violations:          bool = True,
    plot_clean:               bool = False,
) -> None:
    """
    Full coherence test pipeline.

    Args:
        anp_run_dir:     Path to runs/<timestamp>/ (None to skip ANP).
        mlp_run_dir:     Path to runs_mlp/<timestamp>/ (None to skip MLPs).
        data_dir:        Path to prepared_data.pkl directory.
        out_dir:         Output directory for reports and plots.
        cfg:             CoherenceConfig with thresholds.
        all_tasks:       If True, evaluate all 25 tasks; otherwise val+test only.
        specialist_id:   1-based specialist index to include (optional).
        plot_violations: Save plots for tasks with violations.
        plot_clean:      Also save plots for tasks with no violations.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "violation_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧  Device  : {device}")
    print(f"📁  Out dir : {out_dir}")

    # ── Load data ─────────────────────────────────────────────────────────────
    data = load_prepared_data(data_dir)
    validate_targets(data)

    target_cols   = list(data["normalized_synth_datasets"][0][1].columns)
    input_dim     = data["normalized_synth_datasets"][0][0].shape[1]
    output_dim    = len(target_cols)
    denorm_values = {
        "y_mean": data["denorm_values"]["y_mean"],
        "y_std":  data["denorm_values"]["y_std"],
    }

    # Task selection
    if all_tasks:
        task_ids = list(range(len(data["normalized_synth_datasets"])))
    else:
        task_ids = cfg.val_task_ids + cfg.test_task_ids

    def task_label(i: int) -> str:
        if i in cfg.train_task_ids: return f"train_{cfg.train_task_ids.index(i)+1:02d}"
        if i in cfg.val_task_ids:   return f"val_{cfg.val_task_ids.index(i)+1:02d}"
        if i in cfg.test_task_ids:  return f"test_{cfg.test_task_ids.index(i)+1:02d}"
        return f"task_{i:02d}"

    # Pre-sort and extract windows
    tasks_data = []
    for i in task_ids:
        X, y  = sort_task_by_cycle(*data["normalized_synth_datasets"][i])
        X_ctx, y_ctx, X_tgt, y_tgt = extract_window(
            X, y, cfg.ctx_rows, cfg.tgt_rows
        )
        tasks_data.append((task_label(i), X_ctx, y_ctx, X_tgt, y_tgt))

    print(f"\n   Tasks evaluated : {len(tasks_data)}")
    print(f"   Target columns  : {target_cols}")
    print(f"\n📦  Loading models...")

    # x_col_names needed to map feature names → numpy indices for reduced models
    x_col_names = list(data["normalized_synth_datasets"][0][0].columns)

    # ── Load models ───────────────────────────────────────────────────────────
    # Each entry: (label, model, type, model_target_cols, feature_cols)
    # feature_cols: list of X column names for reduced models, None for full
    models_to_test: List[Tuple[str, nn.Module, str, List[str], Optional[List[str]]]] = []

    if anp_run_dir is not None:
        anp = load_anp_model(anp_run_dir, input_dim, target_cols, device, label="ANP")
        if anp:
            models_to_test.append((anp[0], anp[1], "anp", anp[2], anp[3]))

    if anp_soc_run_dir is not None:
        anp_soc = load_anp_model(
            anp_soc_run_dir, input_dim, target_cols, device, label="ANP-SoC"
        )
        if anp_soc:
            models_to_test.append((anp_soc[0], anp_soc[1], "anp", anp_soc[2], anp_soc[3]))

    if anp_cycle_run_dir is not None:
        anp_cyc = load_anp_model(
            anp_cycle_run_dir, input_dim, target_cols, device, label="ANP-Cycle"
        )
        if anp_cyc:
            models_to_test.append((anp_cyc[0], anp_cyc[1], "anp", anp_cyc[2], anp_cyc[3]))

    if anp_soc_reduced_run_dir is not None:
        anp_soc_r = load_anp_model(
            anp_soc_reduced_run_dir, input_dim, target_cols,
            device, label="ANP-SoC-red"
        )
        if anp_soc_r:
            models_to_test.append((anp_soc_r[0], anp_soc_r[1], "anp",
                                    anp_soc_r[2], anp_soc_r[3]))

    if anp_cycle_reduced_run_dir is not None:
        anp_cyc_r = load_anp_model(
            anp_cycle_reduced_run_dir, input_dim, target_cols,
            device, label="ANP-Cycle-red"
        )
        if anp_cyc_r:
            models_to_test.append((anp_cyc_r[0], anp_cyc_r[1], "anp",
                                    anp_cyc_r[2], anp_cyc_r[3]))

    if mlp_run_dir is not None:
        cfg_path = mlp_run_dir / "config.json"
        neurons, dropout = 128, 0.1
        if cfg_path.exists():
            with cfg_path.open() as f:
                mlp_cfg = json.load(f)
            neurons = mlp_cfg.get("neurons", 128)
            dropout = mlp_cfg.get("dropout", 0.1)

        dr = load_mlp(mlp_run_dir / "dr_mlp" / "best.pt",
                      input_dim, output_dim, device, neurons, dropout)
        if dr:
            models_to_test.append(("DR-MLP", dr, "mlp", target_cols, None))

        if specialist_id is not None:
            spec_path = mlp_run_dir / f"specialist_{specialist_id:02d}" / "best.pt"
            spec = load_mlp(spec_path, input_dim, output_dim, device, neurons, dropout)
            if spec:
                models_to_test.append((f"Specialist_{specialist_id:02d}", spec, "mlp",
                                        target_cols, None))

    if not models_to_test:
        print("\n  ⚠  No models loaded — check paths and try again.")
        return

    print(f"\n  Models loaded: {[m for m, _, _, _, _ in models_to_test]}")

    # ── Run coherence checks ──────────────────────────────────────────────────
    print(f"\n🔬  Running coherence checks...\n")

    all_reports: List[ViolationReport] = []

    # Collect predictions and reports keyed by task for comparison plots
    # Structure: task_predictions[t_label] = {m_label: pred_dn}
    #            task_reports[t_label]     = {m_label: report}
    task_predictions: Dict[str, Dict[str, np.ndarray]] = {}
    task_reports:     Dict[str, Dict[str, ViolationReport]] = {}
    task_true_dn:     Dict[str, np.ndarray] = {}

    for m_label, model, m_type, m_target_cols, feat_cols in models_to_test:
        feat_idx = get_feature_indices(x_col_names, feat_cols)
        print(f"  ── {m_label} ─────────────────────────────────────")
        for t_label, X_ctx, y_ctx, X_tgt, y_tgt in tasks_data:

            # Predict in normalized space (NaN for unmodelled targets)
            if m_type == "anp":
                pred_norm = predict_anp(
                    model,
                    filter_x(X_ctx, feat_idx),   # filter X to model's features
                    y_ctx,
                    filter_x(X_tgt, feat_idx),   # filter X to model's features
                    device,
                    target_cols, m_target_cols
                )
            else:
                pred_norm = predict_mlp(model, X_tgt, device)

            # Denormalize
            pred_dn = np.stack([
                denormalize(pred_norm[:, i], col, denorm_values)
                for i, col in enumerate(target_cols)
            ], axis=1)
            true_dn = np.stack([
                denormalize(y_tgt[:, i], col, denorm_values)
                for i, col in enumerate(target_cols)
            ], axis=1)

            # Run checks
            report = check_coherence(
                pred_dn, target_cols, cfg, m_label, t_label
            )
            all_reports.append(report)

            # Accumulate for comparison plot
            if t_label not in task_predictions:
                task_predictions[t_label] = {}
                task_reports[t_label]     = {}
                task_true_dn[t_label]     = true_dn
            task_predictions[t_label][m_label] = pred_dn
            task_reports[t_label][m_label]     = report

            # Console output
            viol_str = ""
            if report.soc_below_min > 0:
                viol_str += f"  SoC<0: {report.soc_below_min}"
            if report.soc_above_max > 0:
                viol_str += f"  SoC>100: {report.soc_above_max}"
            if report.soc_spike > 0:
                viol_str += f"  spikes: {report.soc_spike}"
            if report.soc_not_monotone:
                viol_str += "  SoC↑trend"
            if report.cycle_negative > 0:
                viol_str += f"  Cyc<0: {report.cycle_negative}"
            if report.cycle_not_monotone:
                viol_str += "  Cyc↓trend"

            status = "✓ clean" if not report.has_any_violation() else f"⚠{viol_str}"
            soc_range = (f"SoC∈[{report.soc_pred_min:6.1f},{report.soc_pred_max:6.1f}]%"
                         if report.soc_pred_max > 0 or report.soc_pred_min < 0
                         else "SoC=N/A          ")
            cyc_range = (f"Cyc∈[{report.cycle_pred_min:6.0f},{report.cycle_pred_max:6.0f}]"
                         if report.cycle_pred_max > 0 or report.cycle_pred_min < 0
                         else "Cyc=N/A        ")
            print(f"   {t_label:<14}  {soc_range}  {cyc_range}  {status}")

            # Individual model plot (existing behaviour)
            #should_plot = (plot_violations and report.has_any_violation()) or plot_clean
            #if should_plot:
            #    plot_path = plots_dir / f"{m_label}_{t_label}.png"
            #    plot_violation_example(
            #        pred_dn, true_dn, target_cols, report, cfg, plot_path
            #    )

    # ── Comparison plots (all models on the same axes, per task) ─────────────
    print(f"\n  Generating comparison plots...")
    comp_dir = out_dir / "comparison_plots"
    comp_dir.mkdir(parents=True, exist_ok=True)

    for t_label in task_predictions:
        # Only plot if more than one model is available for this task
        if len(task_predictions[t_label]) < 2:
            continue
        any_violation = any(
            r.has_any_violation()
            for r in task_reports[t_label].values()
        )
        if (plot_violations and any_violation) or plot_clean:
            plot_comparison_coherence(
                preds_dn    = task_predictions[t_label],
                true_dn     = task_true_dn[t_label],
                reports     = task_reports[t_label],
                target_cols = target_cols,
                cfg         = cfg,
                out_path    = comp_dir / f"comparison_{t_label}.png",
            )
    print(f"  ✓  Comparison plots → {comp_dir}")

    # ── Build summary CSV ─────────────────────────────────────────────────────
    summary_rows = []
    for r in all_reports:
        summary_rows.append({
            "model":               r.model_label,
            "task":                r.task_label,
            "n_predictions":       r.n_predictions,
            "soc_below_min":       r.soc_below_min,
            "soc_above_max":       r.soc_above_max,
            "soc_spike":           r.soc_spike,
            "soc_not_monotone":    int(r.soc_not_monotone),
            "cycle_negative":      r.cycle_negative,
            "cycle_not_monotone":  int(r.cycle_not_monotone),
            "violation_rate_%":    round(r.violation_rate() * 100, 2),
            "soc_pred_min":        round(r.soc_pred_min, 2),
            "soc_pred_max":        round(r.soc_pred_max, 2),
            "soc_pred_mean":       round(r.soc_pred_mean, 2),
            "cycle_pred_min":      round(r.cycle_pred_min, 1),
            "cycle_pred_max":      round(r.cycle_pred_max, 1),
            "any_violation":       int(r.has_any_violation()),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "coherence_summary.csv", index=False)
    print(f"\n  ✓  coherence_summary.csv")

    # ── Aggregated report ─────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 70)
    lines.append("PHYSICAL COHERENCE REPORT")
    lines.append(f"Thresholds: SoC ∈ [{cfg.soc_min}, {cfg.soc_max}]%  "
                 f"spike > {cfg.spike_threshold}%  "
                 f"mono_threshold = {cfg.mono_threshold}%")
    lines.append("=" * 70)

    for m_label in [m for m, _, _, _, _ in models_to_test]:
        m_reports = [r for r in all_reports if r.model_label == m_label]
        n_tasks   = len(m_reports)
        n_total   = sum(r.n_predictions for r in m_reports)

        lines.append(f"\n── {m_label} ({n_tasks} tasks, {n_total:,} predictions) ──")
        lines.append(f"{'Violation type':<35} {'Count':>8} {'% of preds':>12}")
        lines.append("-" * 57)

        checks = [
            ("SoC below 0%",             sum(r.soc_below_min for r in m_reports)),
            ("SoC above 100%",           sum(r.soc_above_max for r in m_reports)),
            ("SoC spike (>30%/step)",    sum(r.soc_spike for r in m_reports)),
            ("SoC trend not decreasing", sum(r.soc_not_monotone for r in m_reports)),
            ("Cycle negative",           sum(r.cycle_negative for r in m_reports)),
            ("Cycle trend decreasing",   sum(r.cycle_not_monotone for r in m_reports)),
            ("Tasks with any violation", sum(r.has_any_violation() for r in m_reports)),
        ]

        for name, count in checks:
            if "Tasks" in name:
                lines.append(f"  {name:<33} {count:>8} / {n_tasks} tasks")
            else:
                pct = count / max(1, n_total) * 100
                lines.append(f"  {name:<33} {count:>8}   {pct:>10.2f}%")

        # SoC range summary (only if model predicts SoC)
        soc_mins  = [r.soc_pred_min for r in m_reports if r.soc_pred_max != 0.0 or r.soc_pred_min != 0.0]
        soc_maxes = [r.soc_pred_max for r in m_reports if r.soc_pred_max != 0.0 or r.soc_pred_min != 0.0]
        if soc_mins:
            lines.append(f"\n  SoC predicted range across all tasks:")
            lines.append(f"    Global min: {min(soc_mins):.2f}%  "
                         f"Global max: {max(soc_maxes):.2f}%  "
                         f"(valid: [0, 100]%)")
        else:
            lines.append(f"\n  SoC: not predicted by this model (N/A)")

    report_text = "\n".join(lines)
    print("\n" + report_text)

    with open(out_dir / "coherence_report.txt", "w") as f:
        f.write(report_text)
    print(f"\n  ✓  coherence_report.txt")
    print(f"\n✅  Done. Outputs in: {out_dir}\n")


# ==============================================================================
# CLI
# ==============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Physical coherence test for ANP variants and MLP predictions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--anp_run",              type=str, default=None, help="Path to dual-target ANP run directory")
    p.add_argument("--anp_soc_run",          type=str, default=None, help="Path to SoC-only ANP run directory")
    p.add_argument("--anp_cycle_run",        type=str, default=None, help="Path to Cycle-only ANP run directory")
    p.add_argument("--anp_soc_reduced_run",  type=str, default=None, help="Path to SoC-only ANP run with reduced features")
    p.add_argument("--anp_cycle_reduced_run",type=str, default=None, help="Path to Cycle-only ANP run with reduced features")
    p.add_argument("--mlp_run",              type=str, default=None, help="Path to runs_mlp/<timestamp>/ directory")
    p.add_argument("--data_dir",       type=str, default="../csic_real_synth_load/prepared_data")
    p.add_argument("--out_dir",        type=str, default="", help="Output directory (default: ./coherence/)")
    p.add_argument("--all_tasks",      action="store_true", help="Evaluate all 25 tasks instead of val+test only")
    p.add_argument("--specialist_id",  type=int, default=None, help="1-based specialist MLP index to include (e.g. 5)")
    p.add_argument("--plot_clean",     action="store_true", help="Also save plots for tasks with no violations")
    p.add_argument("--no_plots",       action="store_true", help="Skip all plots (faster, CSV only)")

    # Physical thresholds
    p.add_argument("--soc_min",        type=float, default=0.0)
    p.add_argument("--soc_max",        type=float, default=100.0)
    p.add_argument("--spike_threshold",type=float, default=30.0)
    p.add_argument("--mono_threshold", type=float, default=10.0)

    # Evaluation window
    p.add_argument("--ctx_cycles",     type=int, default=60)
    p.add_argument("--tgt_cycles",     type=int, default=60)
    p.add_argument("--meas_per_cycle", type=int, default=30)

    # Task split
    p.add_argument("--train_ids", type=int, nargs="+", default=list(range(17)))
    p.add_argument("--val_ids",   type=int, nargs="+", default=list(range(17, 22)))
    p.add_argument("--test_ids",  type=int, nargs="+", default=list(range(22, 25)))

    return p.parse_args()


def main() -> None:
    args    = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(__file__).resolve().parent / "coherence")

    cfg = CoherenceConfig(
        soc_min          = args.soc_min,
        soc_max          = args.soc_max,
        spike_threshold  = args.spike_threshold,
        mono_threshold   = args.mono_threshold,
        ctx_cycles       = args.ctx_cycles,
        tgt_cycles       = args.tgt_cycles,
        measurements_per_cycle = args.meas_per_cycle,
        train_task_ids   = args.train_ids,
        val_task_ids     = args.val_ids,
        test_task_ids    = args.test_ids,
    )

    run(
        anp_run_dir              = Path(args.anp_run)              if args.anp_run              else None,
        anp_soc_run_dir          = Path(args.anp_soc_run)          if args.anp_soc_run          else None,
        anp_cycle_run_dir        = Path(args.anp_cycle_run)        if args.anp_cycle_run        else None,
        anp_soc_reduced_run_dir  = Path(args.anp_soc_reduced_run)  if args.anp_soc_reduced_run  else None,
        anp_cycle_reduced_run_dir= Path(args.anp_cycle_reduced_run)if args.anp_cycle_reduced_run else None,
        mlp_run_dir              = Path(args.mlp_run)              if args.mlp_run              else None,
        data_dir        = args.data_dir,
        out_dir         = out_dir,
        cfg             = cfg,
        all_tasks       = args.all_tasks,
        specialist_id   = args.specialist_id,
        plot_violations = not args.no_plots,
        plot_clean      = args.plot_clean,
    )


if __name__ == "__main__":
    main()