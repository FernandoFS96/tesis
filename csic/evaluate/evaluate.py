"""
evaluate.py
==============================================================================
Unified validation script: compares all models on all tasks.
Loads:
    - 17 Specialist MLPs       (from --mlp_run/specialist_XX/best.pt)
    - 1  DR-MLP                (from --mlp_run/dr_mlp/best.pt)
    - 1  ANP dual-target       (from --anp_run)
    - 1  ANP SoC-only          (from --anp_soc_run)
    - 1  ANP Cycle-only        (from --anp_cycle_run)
    - 1  ANP SoC-only reduced  (from --anp_soc_reduced_run)
    - 1  ANP Cycle-only reduced (from --anp_cycle_reduced_run)
      May be aggregated (aggregate_by_cycle) and/or SoC-enriched (enrich_soc_predictions). Both are auto-detected from config.json.

All model variants are handled transparently:
    - Single-target models predict only their trained target; the other is NaN.
    - Reduced-feature models filter X to their feature set automatically.
    - Cycle-aggregated models use cycle-level windows (1 row/cycle).
    - SoC-enriched models get per-cycle SoC statistics prepended to X.

Evaluation protocol (identical for all non-aggregated models):
    - Context : first ctx_cycles × meas_per_cycle rows (default 1 800)
    - Target  : next  tgt_cycles × meas_per_cycle rows (default 1 800)
    - Tasks   : all 25 synthetic datasets (17 train + 5 val + 3 test)

Outputs (saved to --out_dir, default ./validation/results/):
    mae_SoC_pct.csv          MAE SoC(%) for all models × all tasks
    mae_Cycle.csv            MAE Cycle  for all models × all tasks
    mae_comparison.csv       Combined wide-format table
    mae_soc_heatmap.png      Heatmap SoC
    mae_cycle_heatmap.png    Heatmap Cycle
    bar_soc.png / bar_cycle.png   Grouped bar charts per split
    summary.txt              Human-readable summary table

Usage:
    python evaluate.py \
        --mlp_run              ../train/runs_mlp/20260511_121741 \
        --anp_run              ../train/runs/anp_all/20260512_124715 \
        --anp_soc_run          ../train/runs/anp_SoC/20260512_114601 \
        --anp_cycle_run        ../train/runs/anp_Cycle_reduced/20260519_122631 \
        --anp_soc_reduced_run  ../train/optuna_results/anp_soc_reduced/trial_163 \
        --anp_cycle_reduced_run ../train/runs/anp_Cycle_reduced_agg_enriched/20260521_124941 \
        --data_dir             ../csic_real_synth_load/prepared_data

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

# ── Resolve package root ──────────────────────────────────────────────────────
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
    enrich_with_soc_predictions,   # ← new
)

try:
    from models.anp import LatentModel
except ImportError:
    from models.anp import LatentModel

try:
    from train.train_mlp import MLP as _TrainMLP  # type: ignore
except ImportError:
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
    X: pd.DataFrame, y: pd.DataFrame,
    ctx_rows: int, tgt_rows: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract context and target row windows from a cycle-sorted task."""
    T = len(X); ctx_end = min(ctx_rows, T); tgt_end = min(ctx_end + tgt_rows, T)
    Xa = X.values.astype(np.float32); ya = y.values.astype(np.float32)
    return Xa[:ctx_end], ya[:ctx_end], Xa[ctx_end:tgt_end], ya[ctx_end:tgt_end]


def extract_window_np(
    X: np.ndarray, y: np.ndarray,
    ctx_rows: int, tgt_rows: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Same as extract_window but for already-converted numpy arrays."""
    T = len(X); ctx_end = min(ctx_rows, T); tgt_end = min(ctx_end + tgt_rows, T)
    return X[:ctx_end], y[:ctx_end], X[ctx_end:tgt_end], y[ctx_end:tgt_end]


def compute_mae(
    pred: np.ndarray, true: np.ndarray,
    denorm_values: dict, target_cols: List[str],
) -> Dict[str, float]:
    """Denormalized MAE per target column. Clips SoC predictions to [0, 100]."""
    result = {}
    for i, col in enumerate(target_cols):
        m = denorm_values["y_mean"].get(col, 0.0)
        s = denorm_values["y_std"].get(col, 1.0)
        pred_dn = pred[:, i] * s + m
        true_dn = true[:, i] * s + m
        if col == "SoC (%)":
            pred_dn = np.clip(pred_dn, 0.0, 100.0)
        result[col] = float(np.abs(pred_dn - true_dn).mean())
    return result


# ==============================================================================
# MODEL LOADING
# ==============================================================================

def load_mlp_specialists(
    mlp_run_dir: Path, input_dim: int, output_dim: int,
    device: torch.device, n_specialists: int = 17,
) -> List[Tuple[str, nn.Module]]:
    cfg_path = mlp_run_dir / "config.json"
    neurons, dropout = 128, 0.1
    if cfg_path.exists():
        with cfg_path.open() as f:
            cfg = json.load(f)
        neurons = cfg.get("neurons", 128); dropout = cfg.get("dropout", 0.1)
    models = []
    for i in range(1, n_specialists + 1):
        label = f"specialist_{i:02d}"
        ckpt_path = mlp_run_dir / label / "best.pt"
        if not ckpt_path.exists():
            print(f"  ⚠  {label}: best.pt not found — skipping"); continue
        model = MLP(input_dim, output_dim, neurons, dropout)
        ckpt  = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.eval().to(device)
        models.append((label, model))
        print(f"  ✓  {label} loaded")
    return models


def load_dr_mlp(
    mlp_run_dir: Path, input_dim: int, output_dim: int,
    device: torch.device,
) -> Optional[Tuple[str, nn.Module]]:
    cfg_path = mlp_run_dir / "config.json"
    neurons, dropout = 128, 0.1
    if cfg_path.exists():
        with cfg_path.open() as f:
            cfg = json.load(f)
        neurons = cfg.get("neurons", 128); dropout = cfg.get("dropout", 0.1)
    ckpt_path = mlp_run_dir / "dr_mlp" / "best.pt"
    if not ckpt_path.exists():
        print(f"  ⚠  DR-MLP: best.pt not found — skipping"); return None
    model = MLP(input_dim, output_dim, neurons, dropout)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    print(f"  ✓  DR-MLP loaded")
    return ("dr_mlp", model)


def load_anp_model(
    run_dir:         Path,
    input_dim:       int,
    all_target_cols: List[str],
    device:          torch.device,
    label:           str = "anp",
) -> Optional[Tuple[str, nn.Module, List[str], Optional[List[str]], Dict]]:
    """
    Load an ANP checkpoint and auto-detect its full configuration.

    Reads config.json for architecture params and pipeline flags
    (target_col, use_reduced_features, aggregate_by_cycle,
    enrich_soc_predictions, anp_soc_run_dir).
    Infers model_input_dim directly from checkpoint weight shapes —
    robust to missing config fields.

    Returns:
        5-tuple (label, model, model_target_cols, feature_cols, agg_cfg)
        or None if best.pt not found.
            feature_cols : X column names for reduced models, None for full.
            agg_cfg      : dict with 'aggregate', 'enrich', 'soc_run_dir',
                           'ctx_rows', 'tgt_rows'.
    """
    ckpt_path = run_dir / "best.pt"
    cfg_path  = run_dir / "config.json"

    if not ckpt_path.exists():
        print(f"  ⚠  {label}: best.pt not found at {ckpt_path} — skipping")
        return None

    num_hidden   = 128
    attn_dropout = 0.1
    target_col   = "all"
    cfg_data     = {}   # always initialised so later .get() calls are safe

    if cfg_path.exists():
        with cfg_path.open() as f:
            cfg_data = json.load(f)
        num_hidden   = (cfg_data.get("num_hidden")
                        or cfg_data.get("params", {}).get("num_hidden", 128))
        attn_dropout = cfg_data.get("attn_dropout", 0.1)
        target_col   = cfg_data.get("target_col", "all")

    # ── Target columns ────────────────────────────────────────────────────────
    if target_col == "all":
        model_target_cols = all_target_cols
    elif target_col in all_target_cols:
        model_target_cols = [target_col]
    else:
        raw_cpu  = torch.load(ckpt_path, map_location="cpu")
        out_keys = [k for k in raw_cpu["model"]
                    if "mean_projection" in k and "weight" in k]
        out_dim  = raw_cpu["model"][out_keys[0]].shape[0] if out_keys else len(all_target_cols)
        model_target_cols = all_target_cols[:out_dim]

    output_dim = len(model_target_cols)

    # ── Input dim from checkpoint (immune to missing config flags) ─────────────
    raw_cpu = torch.load(ckpt_path, map_location="cpu")
    lat_key = next(
        (k for k in raw_cpu["model"]
         if "latent_encoder.input_projection.linear_layer.weight" in k), None
    )
    model_input_dim = (raw_cpu["model"][lat_key].shape[1] - output_dim
                       if lat_key else input_dim)

    # ── Feature cols for X filtering ──────────────────────────────────────────
    if model_input_dim == input_dim:
        feature_cols = None   # full feature set
    else:
        feature_cols = REDUCED_FEATURE_SETS.get(target_col)
        if feature_cols is None:
            raise ValueError(
                f"'{label}' checkpoint input_dim={model_input_dim} "
                f"(data={input_dim}) but REDUCED_FEATURE_SETS has no entry "
                f"for target_col='{target_col}'."
            )
        # SoC-enriched models add 4 columns on top of the EIS feature set
        enrich_check = cfg_data.get("enrich_soc_predictions", False)
        n_extra      = 4 if enrich_check else 0
        expected_dim = len(feature_cols) + n_extra
        if expected_dim != model_input_dim:
            raise ValueError(
                f"'{label}': checkpoint input_dim={model_input_dim} but "
                f"REDUCED_FEATURE_SETS['{target_col}'] has {len(feature_cols)} "
                f"features" + (f" + 4 SoC enrichment = {expected_dim}"
                               if enrich_check else "")
                + ". Check REDUCED_FEATURE_SETS in train_utils.py."
            )

    # ── Aggregation / enrichment config ───────────────────────────────────────
    aggregate    = cfg_data.get("aggregate_by_cycle",     False)
    enrich       = cfg_data.get("enrich_soc_predictions", False)
    soc_run_dir  = cfg_data.get("anp_soc_run_dir",        "")
    ctx_cycles_m = cfg_data.get("ctx_cycles",             60)
    tgt_cycles_m = cfg_data.get("tgt_cycles",             60)
    meas_p_cycle = cfg_data.get("measurements_per_cycle", 30)

    model_ctx_rows = ctx_cycles_m if aggregate else ctx_cycles_m * meas_p_cycle
    model_tgt_rows = tgt_cycles_m if aggregate else tgt_cycles_m * meas_p_cycle

    agg_cfg = {
        "aggregate":   aggregate,
        "enrich":      enrich,      # True → use windows_enriched
        "soc_run_dir": soc_run_dir, # path to ANP-SoC checkpoint
        "ctx_rows":    model_ctx_rows,
        "tgt_rows":    model_tgt_rows,
    }

    # ── Build and load model ──────────────────────────────────────────────────
    model = LatentModel(num_hidden=num_hidden, input_dim=model_input_dim,
                        output_dim=output_dim, attn_dropout=attn_dropout)
    model.load_state_dict(raw_cpu["model"])
    model.eval().to(device)

    val_mae  = raw_cpu.get("val_MAE", raw_cpu.get("val_loss", "?"))
    feat_str = f"reduced({model_input_dim})" if feature_cols else f"all({input_dim})"
    agg_str  = ("  agg=cycle+SoC" if (aggregate and enrich)
                else "  agg=cycle" if aggregate else "")
    print(f"  ✓  {label:<22} targets={model_target_cols}  "
          f"features={feat_str}{agg_str}  val_MAE={val_mae}")
    return (label, model, model_target_cols, feature_cols, agg_cfg)


# ==============================================================================
# EVALUATION
# ==============================================================================

@torch.no_grad()
def eval_mlp(
    model: nn.Module, X_tgt: np.ndarray, y_tgt: np.ndarray,
    device: torch.device, denorm_values: dict, target_cols: List[str],
) -> Dict[str, float]:
    X_t  = torch.tensor(X_tgt, dtype=torch.float32).to(device)
    pred = model(X_t).cpu().numpy()
    return compute_mae(pred, y_tgt, denorm_values, target_cols)


@torch.no_grad()
def eval_anp(
    model: nn.Module,
    X_ctx: np.ndarray, y_ctx: np.ndarray,
    X_tgt: np.ndarray, y_tgt: np.ndarray,
    device: torch.device,
    denorm_values: dict,
    all_target_cols: List[str],
    model_target_cols: List[str],
) -> Dict[str, float]:
    """
    Evaluate an ANP on a single task via context → target prediction.

    Filters y_ctx/y_tgt to the model's trained targets before the forward
    pass. Returns NaN for targets the model does not predict, so all models
    can be compared in the same result table.
    """
    if model_target_cols != all_target_cols:
        col_idx = [all_target_cols.index(c) for c in model_target_cols]
        y_ctx_m = y_ctx[:, col_idx]
        y_tgt_m = y_tgt[:, col_idx]
    else:
        y_ctx_m = y_ctx; y_tgt_m = y_tgt

    ctx_x = torch.tensor(X_ctx,   dtype=torch.float32).unsqueeze(0).to(device)
    ctx_y = torch.tensor(y_ctx_m, dtype=torch.float32).unsqueeze(0).to(device)
    tgt_x = torch.tensor(X_tgt,   dtype=torch.float32).unsqueeze(0).to(device)
    pred_mean, _, _, _, _ = model(ctx_x, ctx_y, tgt_x, target_y=None)
    pred = pred_mean.squeeze(0).cpu().numpy()

    mae_model = compute_mae(pred, y_tgt_m, denorm_values, model_target_cols)
    result    = {col: float("nan") for col in all_target_cols}
    result.update(mae_model)
    return result


# ==============================================================================
# PLOTTING
# ==============================================================================

_DPI = 300
PLOT_TICK_FONTSIZE       = 10
PLOT_AXIS_LABEL_FONTSIZE = 14
PLOT_TITLE_FONTSIZE      = 16
PLOT_LEGEND_FONTSIZE     = 12
PLOT_CELL_FONTSIZE       = 9
PLOT_COLORBAR_FONTSIZE   = 12
PLOT_SEPARATOR_LINEWIDTH = 4
PLOT_HIGHLIGHT_LINEWIDTH = 5


def plot_heatmaps(df_soc: pd.DataFrame, df_cycle: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def drop_empty(df):
        return df.loc[~df.isna().all(axis=1)]

    train_count = sum(1 for c in df_soc.columns if c.startswith("train_"))

    for df, metric, fname in [
        (df_soc,   "MAE SoC (%)", "mae_soc_heatmap.png"),
        (df_cycle, "MAE Cycle",   "mae_cycle_heatmap.png"),
    ]:
        df = drop_empty(df)
        n_models, n_tasks = len(df), len(df.columns)
        fig, ax = plt.subplots(figsize=(max(10, n_tasks * 0.8), max(5, n_models * 0.55)))
        vals = df.values.astype(float)
        im   = ax.imshow(vals, cmap="YlOrRd", aspect="auto",
                         vmin=np.nanmin(vals), vmax=np.nanpercentile(vals, 95))
        ax.set_xticks(range(n_tasks))
        ax.set_xticklabels(df.columns, rotation=45, ha="right",
                           fontsize=PLOT_TICK_FONTSIZE)
        ax.set_yticks(range(n_models))
        ax.set_yticklabels(df.index, fontsize=PLOT_TICK_FONTSIZE)
        for r in range(n_models):
            for c in range(n_tasks):
                v = vals[r, c]
                if not np.isnan(v):
                    ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                            fontsize=PLOT_CELL_FONTSIZE, color="black")
        for c in range(n_tasks):
            col_vals = vals[:, c]
            if not np.all(np.isnan(col_vals)):
                min_row = np.nanargmin(col_vals)
                ax.add_patch(matplotlib.patches.Rectangle(
                    (c - 0.5, min_row - 0.5), 1, 1,
                    linewidth=PLOT_HIGHLIGHT_LINEWIDTH, edgecolor="lime",
                    facecolor="none"))
        ax.axvline(train_count - 0.5, color="white",
                   linewidth=PLOT_SEPARATOR_LINEWIDTH)
        val_count = sum(1 for c in df.columns if c.startswith("val_"))
        ax.axvline(train_count + val_count - 0.5, color="white",
                   linewidth=PLOT_SEPARATOR_LINEWIDTH)
        anp_rows = [i for i, m in enumerate(df.index) if m in {"anp","anp_soc","anp_cycle"}]
        if anp_rows:
            ax.axhline(min(anp_rows) - 0.5, color="white",
                       linewidth=PLOT_SEPARATOR_LINEWIDTH)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.tick_params(labelsize=PLOT_TICK_FONTSIZE)
        cbar.set_label(metric, fontsize=PLOT_COLORBAR_FONTSIZE)
        ax.set_title(f"{metric} — all models × all tasks\n"
                     f"(train | val | test columns separated by white lines)",
                     fontweight="bold", fontsize=PLOT_TITLE_FONTSIZE)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓  {fname}")


def plot_bar_comparison(df_soc: pd.DataFrame, df_cycle: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def drop_empty(df):
        return df.loc[~df.isna().all(axis=1)]

    for df, metric, fname in [
        (df_soc,   "MAE SoC (%)", "bar_soc.png"),
        (df_cycle, "MAE Cycle",   "bar_cycle.png"),
    ]:
        df = drop_empty(df)
        train_cols = [c for c in df.columns if c.startswith("train_")]
        val_cols   = [c for c in df.columns if c.startswith("val_")]
        test_cols  = [c for c in df.columns if c.startswith("test_")]
        models = list(df.index)
        x, width = np.arange(len(models)), 0.25

        def bar_color(label: str) -> str:
            if label == "anp":             return "#1C7293"
            if label == "anp_soc":         return "#028090"
            if label == "anp_cycle":       return "#21295C"
            if label == "anp_soc_red":     return "#2E86AB"
            if label == "anp_cycle_red":   return "#5C4A72"
            if label == "dr_mlp":          return "#D4860A"
            return "#9AB8C8"

        colors = [bar_color(m) for m in models]
        fig, ax = plt.subplots(figsize=(max(12, len(models) * 0.8), 5))
        for k, (split_cols, split_label, offset) in enumerate([
            (train_cols, "Train", -width),
            (val_cols,   "Val",    0),
            (test_cols,  "Test",   width),
        ]):
            if not split_cols: continue
            avgs = df[split_cols].mean(axis=1).values
            ax.bar(x + offset, avgs, width, label=split_label, color=colors,
                   alpha=[1.0, 0.6, 0.35][k], edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right",
                           fontsize=PLOT_TICK_FONTSIZE)
        ax.set_ylabel(metric, fontsize=PLOT_AXIS_LABEL_FONTSIZE)
        ax.set_title(f"Average {metric} by model and split "
                     f"(grey=specialist, orange=DR-MLP, blue=ANP)",
                     fontsize=PLOT_TITLE_FONTSIZE)
        ax.legend(fontsize=PLOT_LEGEND_FONTSIZE)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓  {fname}")


# ==============================================================================
# MAIN RUN
# ==============================================================================

def run(
    mlp_run_dir:               Path,
    anp_run_dir:               Optional[Path],
    anp_soc_run_dir:           Optional[Path],
    anp_cycle_run_dir:         Optional[Path],
    anp_soc_reduced_run_dir:   Optional[Path],
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
    x_col_names  = list(data["normalized_synth_datasets"][0][0].columns)
    print(f"   input_dim={input_dim}  output_dim={output_dim}  targets={target_cols}")

    all_task_ids = list(range(len(data["normalized_synth_datasets"])))
    all_tasks    = [
        sort_task_by_cycle(*data["normalized_synth_datasets"][i])
        for i in all_task_ids
    ]

    def task_label(i: int) -> str:
        if i in train_task_ids: return f"train_{train_task_ids.index(i)+1:02d}"
        if i in val_task_ids:   return f"val_{val_task_ids.index(i)+1:02d}"
        if i in test_task_ids:  return f"test_{test_task_ids.index(i)+1:02d}"
        return f"task_{i:02d}"

    task_labels = [task_label(i) for i in all_task_ids]

    # Measurement-level windows (shared by non-aggregated models)
    windows = [extract_window(X, y, ctx_rows, tgt_rows) for X, y in all_tasks]

    # ── Load models ───────────────────────────────────────────────────────────
    print("\n📦  Loading models...")

    # 6-tuple: (label, model, type, model_target_cols, feature_cols, agg_cfg)
    _EMPTY_AGG = {"aggregate": False, "enrich": False, "soc_run_dir": "",
                  "ctx_rows": ctx_rows, "tgt_rows": tgt_rows}
    all_models: List[Tuple[str, nn.Module, str, List[str],
                           Optional[List[str]], Dict]] = []

    for label, model in load_mlp_specialists(
        mlp_run_dir, input_dim, output_dim, device, len(train_task_ids)
    ):
        all_models.append((label, model, "mlp", target_cols, None, _EMPTY_AGG))

    dr = load_dr_mlp(mlp_run_dir, input_dim, output_dim, device)
    if dr:
        all_models.append((dr[0], dr[1], "mlp", target_cols, None, _EMPTY_AGG))

    for lbl, path in [
        ("anp",           anp_run_dir),
        ("anp_soc",       anp_soc_run_dir),
        ("anp_cycle",     anp_cycle_run_dir),
        ("anp_soc_red",   anp_soc_reduced_run_dir),
        ("anp_cycle_red", anp_cycle_reduced_run_dir),
    ]:
        if path is None:
            continue
        res = load_anp_model(path, input_dim, target_cols, device, label=lbl)
        if res:
            all_models.append((res[0], res[1], "anp", res[2], res[3], res[4]))

    print(f"\n  Total models loaded: {len(all_models)}")

    # ── Pre-compute cycle-level aggregated windows ────────────────────────────
    needs_agg = any(m[5].get("aggregate", False) for m in all_models)
    if needs_agg:
        print("\n  🔄  Pre-computing cycle-level aggregated windows...")
        tasks_agg   = [aggregate_by_cycle(X, y) for X, y in all_tasks]
        windows_agg = [extract_window(X, y, ctx_cycles, tgt_cycles)
                       for X, y in tasks_agg]
        print(f"     ctx={ctx_cycles} rows  tgt={tgt_cycles} rows  (1 row/cycle)")
    else:
        windows_agg = windows    # alias — unused if needs_agg is False

    # ── Pre-compute SoC-enriched windows ─────────────────────────────────────
    needs_enrich = any(m[5].get("enrich", False) for m in all_models)
    if needs_enrich:
        soc_run_dir_str = next(
            m[5]["soc_run_dir"] for m in all_models if m[5].get("enrich")
        )
        if not soc_run_dir_str:
            print("  ⚠  enrich=True but soc_run_dir is empty — skipping enrichment")
            windows_enriched = windows_agg if needs_agg else windows
        else:
            print("\n  🔬  Pre-computing SoC-enriched cycle-level windows...")
            soc_run      = Path(soc_run_dir_str)
            soc_ckpt     = soc_run / "best.pt"
            soc_cfg_path = soc_run / "config.json"

            soc_num_hidden, soc_attn_dropout, soc_feat_cols = 128, 0.1, None
            if soc_cfg_path.exists():
                with soc_cfg_path.open() as f:
                    scd = json.load(f)
                soc_num_hidden   = (scd.get("num_hidden")
                                    or scd.get("params", {}).get("num_hidden", 128))
                soc_attn_dropout = scd.get("attn_dropout", 0.1)
                soc_target_col   = scd.get("target_col", "SoC (%)")
                if scd.get("use_reduced_features", False):
                    soc_feat_cols = REDUCED_FEATURE_SETS.get(soc_target_col)

            raw_soc  = torch.load(soc_ckpt, map_location="cpu")
            lat_k    = next((k for k in raw_soc["model"]
                             if "latent_encoder.input_projection.linear_layer.weight" in k), None)
            soc_idim = (raw_soc["model"][lat_k].shape[1] - 1 if lat_k else input_dim)

            soc_model = LatentModel(num_hidden=soc_num_hidden, input_dim=soc_idim,
                                    output_dim=1, attn_dropout=soc_attn_dropout)
            soc_model.load_state_dict(raw_soc["model"])
            soc_model.eval().to(device)
            print(f"     ANP-SoC: {soc_run.name}  "
                  f"(input_dim={soc_idim}  "
                  f"features={'reduced' if soc_feat_cols else 'all'})")

            # EIS feature set used by the enriched Cycle model (e.g. 13 cols)
            enriched_entry = next(m for m in all_models if m[5].get("enrich"))
            eis_feat_cols  = enriched_entry[4]   # feat_cols from the 6-tuple

            tasks_raw_list = []
            tasks_agg_list = []
            for X_df, y_df in all_tasks:
                tasks_raw_list.append((X_df, y_df))
                X_a, y_a = aggregate_by_cycle(X_df, y_df)
                if eis_feat_cols is not None:
                    X_a = X_a[eis_feat_cols]    # 202 → 13 EIS Cycle features
                tasks_agg_list.append((X_a, y_a))

            enriched_tasks = enrich_with_soc_predictions(
                tasks_raw     = tasks_raw_list,
                tasks_agg     = tasks_agg_list,
                anp_soc_model = soc_model,
                soc_feat_cols = soc_feat_cols,
                device        = device,
                ctx_cycles    = ctx_cycles,
                meas_per_cycle= meas_per_cycle,
            )

            del soc_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            windows_enriched = [
                extract_window(X_e, y_e, ctx_cycles, tgt_cycles)
                for X_e, y_e in enriched_tasks
            ]
            print(f"     ✓  {len(windows_enriched)} enriched windows  "
                  f"(X ctx shape: {windows_enriched[0][0].shape})")
    else:
        windows_enriched = windows    # alias — unused

    # ── Evaluate all models on all tasks ──────────────────────────────────────
    print("\n📊  Evaluating...\n")
    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for m_label, model, m_type, m_target_cols, feat_cols, agg_cfg in all_models:
        # Enriched models: X is already exactly right — no further column filtering
        if agg_cfg.get("enrich"):
            feat_idx  = None
            m_windows = windows_enriched
        elif agg_cfg.get("aggregate"):
            feat_idx  = get_feature_indices(x_col_names, feat_cols)
            m_windows = windows_agg
        else:
            feat_idx  = get_feature_indices(x_col_names, feat_cols)
            m_windows = windows

        results[m_label] = {}
        for t_label, (X_ctx, y_ctx, X_tgt, y_tgt) in zip(task_labels, m_windows):
            if m_type == "mlp":
                mae = eval_mlp(model, X_tgt, y_tgt, device, denorm_values, target_cols)
            else:
                mae = eval_anp(
                    model,
                    filter_x(X_ctx, feat_idx),
                    y_ctx,
                    filter_x(X_tgt, feat_idx),
                    y_tgt,
                    device, denorm_values, target_cols, m_target_cols
                )
            results[m_label][t_label] = mae

        avg_soc = np.nanmean([results[m_label][t].get("SoC (%)", float("nan"))
                              for t in task_labels])
        avg_cyc = np.nanmean([results[m_label][t].get("Cycle", float("nan"))
                              for t in task_labels])
        soc_str = f"{avg_soc:.3f}%" if not np.isnan(avg_soc) else "  N/A  "
        cyc_str = f"{avg_cyc:.2f}"  if not np.isnan(avg_cyc) else "  N/A  "
        print(f"  {m_label:<24}  avg SoC MAE={soc_str}  avg Cycle MAE={cyc_str}")

    # ── Build DataFrames ──────────────────────────────────────────────────────
    model_labels_ordered = [m for m, *_ in all_models]
    soc_col   = "SoC (%)" if "SoC (%)" in target_cols else target_cols[0]
    cycle_col = "Cycle"   if "Cycle"   in target_cols else target_cols[-1]

    def build_df(col: str) -> pd.DataFrame:
        return pd.DataFrame({
            m: {t: results[m].get(t, {}).get(col, float("nan")) for t in task_labels}
            for m in model_labels_ordered
        }).T

    df_soc   = build_df(soc_col)
    df_cycle = build_df(cycle_col)

    def safe(s):
        return s.replace(" ","_").replace("(","").replace(")","").replace("%","pct")

    df_soc.to_csv(out_dir / f"mae_{safe(soc_col)}.csv")
    df_cycle.to_csv(out_dir / f"mae_{safe(cycle_col)}.csv")
    print(f"\n  ✓  mae_{safe(soc_col)}.csv")
    print(f"  ✓  mae_{safe(cycle_col)}.csv")

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

    def avg(m, labels, col):
        vals = [v for t in labels
                if not np.isnan(v := results[m].get(t, {}).get(col, float("nan")))]
        return np.mean(vals) if vals else float("nan")

    def fmt(v): return f"{v:>10.3f}" if not np.isnan(v) else f"{'N/A':>10}"

    lines = ["=" * 95,
             "VALIDATION SUMMARY — average MAE per split (original units)  |  NaN = target not predicted",
             "=" * 95,
             f"\n{'Model':<24} {'Train SoC':>10} {'Train Cyc':>10} "
             f"{'Val SoC':>9} {'Val Cyc':>9} {'Test SoC':>10} {'Test Cyc':>10}",
             "-" * 95]

    ANP_TAGS = {
        "anp":           "  ← ANP dual",
        "anp_soc":       "  ← ANP SoC-only",
        "anp_cycle":     "  ← ANP Cycle-only",
        "anp_soc_red":   "  ← ANP SoC-only (reduced)",
        "anp_cycle_red": "  ← ANP Cycle-only (reduced)",
    }
    for m_label in model_labels_ordered:
        tag = ANP_TAGS.get(m_label, "")
        lines.append(
            f"{m_label:<24}"
            f" {fmt(avg(m_label, train_labels, soc_col))}"
            f" {fmt(avg(m_label, train_labels, cycle_col))}"
            f" {fmt(avg(m_label, val_labels, soc_col))}"
            f" {fmt(avg(m_label, val_labels, cycle_col))}"
            f" {fmt(avg(m_label, test_labels, soc_col))}"
            f" {fmt(avg(m_label, test_labels, cycle_col))}"
            f"{tag}"
        )

    lines.append("-" * 95)
    specialist_labels = [m for m in model_labels_ordered if m.startswith("specialist_")]

    def group_avg(grp, split_labels, col):
        vals = [v for m in grp for t in split_labels
                if not np.isnan(v := results[m].get(t, {}).get(col, float("nan")))]
        return np.mean(vals) if vals else float("nan")

    for grp_label, grp_models in [
        ("AVG Specialists",     specialist_labels),
        ("DR-MLP",              ["dr_mlp"]        if "dr_mlp"        in model_labels_ordered else []),
        ("ANP (dual)",          ["anp"]            if "anp"           in model_labels_ordered else []),
        ("ANP (SoC-only)",      ["anp_soc"]        if "anp_soc"       in model_labels_ordered else []),
        ("ANP (Cycle-only)",    ["anp_cycle"]      if "anp_cycle"     in model_labels_ordered else []),
        ("ANP (SoC reduced)",   ["anp_soc_red"]    if "anp_soc_red"   in model_labels_ordered else []),
        ("ANP (Cycle reduced)", ["anp_cycle_red"]  if "anp_cycle_red" in model_labels_ordered else []),
    ]:
        if not grp_models: continue
        lines.append(
            f"{grp_label:<24}"
            f" {fmt(group_avg(grp_models, train_labels, soc_col))}"
            f" {fmt(group_avg(grp_models, train_labels, cycle_col))}"
            f" {fmt(group_avg(grp_models, val_labels, soc_col))}"
            f" {fmt(group_avg(grp_models, val_labels, cycle_col))}"
            f" {fmt(group_avg(grp_models, test_labels, soc_col))}"
            f" {fmt(group_avg(grp_models, test_labels, cycle_col))}"
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
    p = argparse.ArgumentParser(
        description="Unified validation — Specialist MLPs, DR-MLP and ANP variants",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mlp_run",              type=str, required=True)
    p.add_argument("--anp_run",              type=str, default=None)
    p.add_argument("--anp_soc_run",          type=str, default=None)
    p.add_argument("--anp_cycle_run",        type=str, default=None)
    p.add_argument("--anp_soc_reduced_run",  type=str, default=None)
    p.add_argument("--anp_cycle_reduced_run",type=str, default=None)
    p.add_argument("--data_dir",       type=str, default="../csic_real_synth_load/prepared_data")
    p.add_argument("--out_dir",        type=str, default="")
    p.add_argument("--ctx_cycles",     type=int, default=60)
    p.add_argument("--tgt_cycles",     type=int, default=60)
    p.add_argument("--meas_per_cycle", type=int, default=30)
    p.add_argument("--train_ids",      type=int, nargs="+", default=list(range(17)))
    p.add_argument("--val_ids",        type=int, nargs="+", default=list(range(17, 22)))
    p.add_argument("--test_ids",       type=int, nargs="+", default=list(range(22, 25)))
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    out_dir = (Path(args.out_dir) if args.out_dir
               else Path(__file__).resolve().parent / "results")
    run(
        mlp_run_dir               = Path(args.mlp_run),
        anp_run_dir               = Path(args.anp_run)               if args.anp_run               else None,
        anp_soc_run_dir           = Path(args.anp_soc_run)           if args.anp_soc_run           else None,
        anp_cycle_run_dir         = Path(args.anp_cycle_run)         if args.anp_cycle_run         else None,
        anp_soc_reduced_run_dir   = Path(args.anp_soc_reduced_run)   if args.anp_soc_reduced_run   else None,
        anp_cycle_reduced_run_dir = Path(args.anp_cycle_reduced_run) if args.anp_cycle_reduced_run else None,
        data_dir        = args.data_dir,
        out_dir         = out_dir,
        train_task_ids  = args.train_ids,
        val_task_ids    = args.val_ids,
        test_task_ids   = args.test_ids,
        ctx_cycles      = args.ctx_cycles,
        tgt_cycles      = args.tgt_cycles,
        meas_per_cycle  = args.meas_per_cycle,
    )


if __name__ == "__main__":
    main()
