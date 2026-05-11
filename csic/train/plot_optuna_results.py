"""
plot_optuna_results.py
==============================================================================
Visualise Optuna hyperparameter search results at any point during or after the optimisation. 
Reads directly from the study.db SQLite database so it can be run while optuna_anp.py is still running in another process.

Generates:
    01_optimization_history.png   — best val MAE found per trial over time
    02_param_importance.png       — fANOVA hyperparameter importance ranking
    03_parallel_coordinate.png    — all trials as parallel coordinate lines
    04_slice_num_hidden.png       — val MAE distribution per num_hidden value
    05_slice_batch_size.png       — val MAE distribution per batch_size value
    06_slice_lr.png               — val MAE distribution per lr value
    07_slice_beta.png             — val MAE distribution per beta value
    08_slice_attn_dropout.png     — val MAE distribution per attn_dropout value
    09_contour_lr_beta.png        — 2D contour: interaction lr × beta
    10_contour_hidden_lr.png      — 2D contour: interaction num_hidden × lr
    11_top10_trials.png           — bar chart of top-10 trials by val MAE
    summary_table.csv             — all completed trials sorted by val MAE

Usage:
    # From the train/ directory
    python plot_optuna_results.py --study_dir ./optuna_results

    # Specify a custom study name if you changed it
    python plot_optuna_results.py --study_dir ./optuna_results \
                                  --study_name anp_battery
==============================================================================
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler


# ── Plot style ─────────────────────────────────────────────────────────────────
PLT_DPI  = 150
PLT_LW   = 1.6
BEST_COL = "#1C7293"
VAL_COL  = "#C0392B"
GREY     = "#9AB8C8"


def _save(fig: Figure, path: Path, tight: bool = True) -> None:
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=PLT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}")


def load_study(study_dir: Path, study_name: str) -> optuna.Study:
    """Load an existing Optuna study from its SQLite database."""
    db_path = study_dir / "study.db"
    if not db_path.exists():
        raise FileNotFoundError(
            f"study.db not found in {study_dir}.\n"
            f"Make sure --study_dir points to the optuna_results/ directory."
        )
    storage = f"sqlite:///{db_path}"
    study   = optuna.load_study(study_name=study_name, storage=storage)
    return study


def get_completed_df(study: optuna.Study) -> pd.DataFrame:
    """Return a DataFrame of completed trials sorted by val loss."""
    trials_df = study.trials_dataframe()
    completed  = trials_df[trials_df["state"] == "COMPLETE"].copy()
    completed  = completed.sort_values("value").reset_index(drop=True)
    return completed


# ==============================================================================
# INDIVIDUAL PLOTS
# ==============================================================================

def plot_optimization_history(
    study:    optuna.Study,
    out_dir:  Path,
) -> None:
    """Plot best val MAE found so far at each trial number."""
    trials_df  = get_completed_df(study)
    if trials_df.empty:
        print("  ⚠  No completed trials yet — skipping optimization history")
        return

    numbers    = trials_df["number"].values
    val_losses = trials_df["value"].values

    # Sort by trial number to get chronological order
    order      = np.argsort(numbers)
    numbers    = numbers[order]
    val_losses = val_losses[order]

    # Running best
    running_best = np.minimum.accumulate(val_losses)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.scatter(numbers, val_losses, s=25, alpha=0.55, color=GREY,
               label="Trial val MAE", zorder=2)
    ax.plot(numbers, running_best, color=BEST_COL, linewidth=PLT_LW,
            label="Best so far", zorder=3)

    best_idx = np.argmin(val_losses)
    ax.axvline(numbers[best_idx], color=VAL_COL, linestyle="--",
               linewidth=1.0, label=f"Best trial #{numbers[best_idx]}")

    ax.set_xlabel("Trial number")
    ax.set_ylabel("Val MAE (SoC %)")
    ax.set_title(f"Optimization history — {len(numbers)} completed trials")
    ax.legend()
    ax.grid(True, alpha=0.25)
    _save(fig, out_dir / "01_optimization_history.png")


def plot_param_importance(
    study:   optuna.Study,
    out_dir: Path,
) -> None:
    """fANOVA importance — requires ≥ 4 completed trials."""
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) < 4:
        print(f"  ⚠  Only {len(completed)} completed trials — "
              f"importance needs ≥ 4. Skipping.")
        return

    try:
        from optuna.importance import get_param_importances
        importances = get_param_importances(study)
    except Exception as exc:
        print(f"  ⚠  Could not compute importance: {exc}")
        return

    params = list(importances.keys())
    values = list(importances.values())

    fig, ax = plt.subplots(figsize=(8, 4))
    colors  = [BEST_COL if v == max(values) else GREY for v in values]
    bars    = ax.barh(params[::-1], values[::-1], color=colors[::-1],
                      edgecolor="white")
    ax.set_xlabel("Importance (fANOVA)")
    ax.set_title(f"Hyperparameter importance — {len(completed)} trials")
    ax.grid(True, axis="x", alpha=0.25)

    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    _save(fig, out_dir / "02_param_importance.png")


def plot_parallel_coordinate(
    study:   optuna.Study,
    out_dir: Path,
) -> None:
    """Parallel coordinate plot — colour-coded by val loss."""
    df = get_completed_df(study)
    if df.empty:
        return

    param_cols = [c for c in df.columns if c.startswith("params_")]
    if not param_cols:
        print("  ⚠  No param columns found in trials DataFrame — skipping parallel coord")
        return

    params     = [c.replace("params_", "") for c in param_cols]
    val_losses = df["value"].values
    norm       = Normalize(np.min(val_losses), np.percentile(val_losses, 80))
    cmap       = matplotlib.colormaps.get_cmap("RdYlGn_r")

    n_params = len(params)
    fig, axes = plt.subplots(1, n_params - 1, figsize=(3 * (n_params - 1), 5),
                             sharey=False)
    if n_params == 2:
        axes = [axes]

    for i, (ax, p1, p2) in enumerate(zip(axes, params[:-1], params[1:])):
        col1 = f"params_{p1}"
        col2 = f"params_{p2}"

        # Normalise each param to [0, 1] for the parallel axis
        # Use to_numpy to ensure a plain ndarray (avoids ExtensionArray issues)
        v1 = df[col1].to_numpy(dtype=float)
        v2 = df[col2].to_numpy(dtype=float)
        v1n = (v1 - v1.min()) / (v1.max() - v1.min() + 1e-9)
        v2n = (v2 - v2.min()) / (v2.max() - v2.min() + 1e-9)

        for j in range(len(df)):
            ax.plot([0, 1], [v1n[j], v2n[j]],
                    color=cmap(norm(val_losses[j])), alpha=0.4, linewidth=0.8)

        # Tick labels showing actual values
        unique1 = np.unique(v1)
        unique2 = np.unique(v2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([p1, p2], fontsize=9)
        ax.set_yticks([])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label="Val MAE", shrink=0.8)
    fig.suptitle("Parallel coordinate — trial hyperparameters\n"
                 "(green = low MAE, red = high MAE)", fontsize=11)
    _save(fig, out_dir / "03_parallel_coordinate.png")


def plot_slice(
    study:      optuna.Study,
    param_name: str,
    out_dir:    Path,
    plot_index: int,
) -> None:
    """
    For a single categorical hyperparameter: boxplot of val loss per value,
    with individual trial dots overlaid.
    """
    df = get_completed_df(study)
    col = f"params_{param_name}"
    if col not in df.columns:
        print(f"  ⚠  Column '{col}' not found — skipping slice for {param_name}")
        return

    values     = df[col].values
    val_losses = df["value"].values
    categories = sorted(set(values), key=lambda x: float(x))

    # Group val losses by category value
    grouped = {str(cat): val_losses[values == cat] for cat in categories}

    fig, ax = plt.subplots(figsize=(8, 4))

    positions = range(len(categories))
    bp = ax.boxplot(
        [grouped[str(cat)] for cat in categories],
        positions=list(positions),
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.0),
        flierprops=dict(marker=".", markersize=3, alpha=0.4),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(BEST_COL)
        patch.set_alpha(0.5)

    # Overlay individual points with jitter
    for i, cat in enumerate(categories):
        y = grouped[str(cat)]
        x = np.random.normal(i, 0.07, size=len(y))
        ax.scatter(x, y, s=20, alpha=0.7, color=VAL_COL, zorder=3)

    ax.set_xticks(list(positions))
    ax.set_xticklabels([str(c) for c in categories])
    ax.set_xlabel(param_name)
    ax.set_ylabel("Val MAE (SoC %)")
    ax.set_title(f"Val MAE by {param_name} value  "
                 f"({len(df)} completed trials)")
    ax.grid(True, axis="y", alpha=0.25)
    _save(fig, out_dir / f"0{plot_index}_slice_{param_name}.png")


def plot_contour(
    study:   optuna.Study,
    param_x: str,
    param_y: str,
    out_dir: Path,
    plot_index: int,
) -> None:
    """
    2D scatter coloured by val loss to reveal interactions between two params.
    Uses log scale for lr-like parameters.
    """
    df  = get_completed_df(study)
    cx  = f"params_{param_x}"
    cy  = f"params_{param_y}"
    if cx not in df.columns or cy not in df.columns:
        return

    x   = df[cx].values.astype(float)
    y_p = df[cy].values.astype(float)
    # Ensure z is a numeric numpy array so we can use numpy functions safely
    z   = df["value"].astype(float).values
    # Use numpy min/percentile to avoid accessing pandas ExtensionArray methods
    z_min = np.min(z)
    z_p80 = np.percentile(z, 80)
    norm = Normalize(z_min, z_p80)

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(x, y_p, c=z, cmap="RdYlGn_r", norm=norm,
                    s=60, alpha=0.8, edgecolors="white", linewidths=0.4)
    plt.colorbar(sc, ax=ax, label="Val MAE")

    # Annotate best trial
    best_idx = np.argmin(z)
    ax.scatter(x[best_idx], y_p[best_idx], s=150, marker="*",
               color="black", zorder=5, label=f"Best (trial #{df['number'].values[best_idx]})")
    ax.legend(fontsize=8)

    log_x = param_x in ("lr",)
    log_y = param_y in ("lr",)
    if log_x: ax.set_xscale("log")
    if log_y: ax.set_yscale("log")

    ax.set_xlabel(param_x)
    ax.set_ylabel(param_y)
    ax.set_title(f"Interaction: {param_x} × {param_y}")
    ax.grid(True, alpha=0.2)
    _save(fig, out_dir / f"{plot_index:02d}_contour_{param_x}_{param_y}.png")


def plot_top10(
    study:   optuna.Study,
    out_dir: Path,
) -> None:
    """Horizontal bar chart of the top-10 trials by val loss."""
    df = get_completed_df(study)
    if df.empty:
        return

    top10 = df.head(10)
    labels = [f"#{int(r.number):03d}  "
              + "  ".join(f"{c.replace('params_','')}={r[c]}"
                          for c in df.columns if c.startswith("params_"))
              for _, r in top10.iterrows()]

    fig, ax = plt.subplots(figsize=(12, max(4, len(top10) * 0.55)))
    colors = [BEST_COL] + [GREY] * (len(top10) - 1)
    ax.barh(range(len(top10))[::-1], top10["value"].values,
            color=colors, edgecolor="white")
    ax.set_yticks(range(len(top10))[::-1])
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Val MAE (SoC %)")
    ax.set_title(f"Top-{len(top10)} trials by val MAE  "
                 f"(blue = best)")
    ax.grid(True, axis="x", alpha=0.25)
    _save(fig, out_dir / "11_top10_trials.png")


def save_summary_table(
    study:   optuna.Study,
    out_dir: Path,
) -> None:
    """Save a clean CSV of all completed trials sorted by val loss."""
    df = get_completed_df(study)
    if df.empty:
        print("  ⚠  No completed trials to save")
        return

    # Rename param columns for readability
    rename = {c: c.replace("params_", "") for c in df.columns
              if c.startswith("params_")}
    out_df = df.rename(columns=rename)
    keep   = ["number", "value"] + list(rename.values()) + \
             [c for c in out_df.columns if "duration" in c.lower()]
    keep   = [c for c in keep if c in out_df.columns]
    out_df[keep].to_csv(out_dir / "summary_table.csv", index=False)
    print(f"  ✓  summary_table.csv  ({len(out_df)} completed trials)")


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    p = argparse.ArgumentParser(description="Visualise Optuna results from study.db at any time",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--study_dir",  type=str, required=True, help="Path to the optuna_results/ directory containing study.db")
    p.add_argument("--study_name", type=str, default="anp_battery", help="Optuna study name (must match the one used in optuna_anp.py)")
    p.add_argument("--out_dir",    type=str, default="./optuna_results/plots", help="Where to save plots (default: same as study_dir)")
    args = p.parse_args()

    study_dir = Path(args.study_dir)
    out_dir   = Path(args.out_dir) if args.out_dir else study_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📂  Loading study from : {study_dir / 'study.db'}")
    study = load_study(study_dir, args.study_name)

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    total     = len(study.trials)
    print(f"   Completed trials : {len(completed)} / {total}")

    if not completed:
        print("\n  No completed trials yet. Run optuna_anp.py first.\n")
        return

    best = study.best_trial
    print(f"   Best val MAE     : {best.value:.4f}  (trial #{best.number:03d})")
    print(f"   Best params      : {best.params}")
    print(f"\n📈  Generating plots → {out_dir}\n")

    # 01 — Optimization history
    plot_optimization_history(study, out_dir)

    # 02 — Hyperparameter importance
    plot_param_importance(study, out_dir)

    # 03 — Parallel coordinate
    plot_parallel_coordinate(study, out_dir)

    # 04-08 — Slice plots (one per hyperparameter)
    params_to_slice = [
        ("num_hidden",   4),
        ("batch_size",   5),
        ("lr",           6),
        ("beta",         7),
        ("attn_dropout", 8),
    ]
    for param, idx in params_to_slice:
        plot_slice(study, param, out_dir, idx)

    # 09-10 — Contour interaction plots
    plot_contour(study, "lr",         "beta",       out_dir, 9)
    plot_contour(study, "num_hidden", "lr",          out_dir, 10)

    # 11 — Top-10 bar chart
    plot_top10(study, out_dir)

    # Summary CSV
    save_summary_table(study, out_dir)

    print(f"\n✅  Done. {len(os.listdir(out_dir))} files in {out_dir}\n")


if __name__ == "__main__":
    import os
    main()
