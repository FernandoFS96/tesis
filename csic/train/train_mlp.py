"""
train_mlp.py
==============================================================================
Specialist MLP and Domain-Randomized (DR) MLP baseline training script.

Purpose:
    Provide a supervised baseline against which the ANP meta-learning model can be compared, following the professor's recommendation:
        "Train 17 specialist MLPs (one per task) + 3 OOD, validate on all tasks to confirm the numbers are what we expect."

Models trained:
    - 17 Specialist MLPs  — one per training task, fitted only on that task's first ctx_cycles cycles.
    - 1 DR-MLP            — trained on all 17 training tasks concatenated, acting as a domain-randomized generalist baseline.

Evaluation window (same as ANP):
    - Context (train input): cycles 1  – ctx_cycles      (default 50)
    - Target  (train label): cycles ctx_cycles+1 – 2*ctx_cycles (default 51-100)
    - OOD eval:              same window on unseen val/test tasks

Final output — comparison table (comparison/mae_comparison.csv):
    Rows:    all models (17 specialists + DR-MLP)
    Columns: MAE_SoC and MAE_Cycle for each of the 17 train + N val tasks

Usage:
    python train_mlp.py
    python train_mlp.py --data_dir ../csic_real_synth_load/prepared_data \
                        --neurons 128 --epochs 1000 --ctx_cycles 50

Output structure:
    runs_mlp/<timestamp>/
        specialist_01/   best.pt, metrics.csv, plots/
        ...
        specialist_17/
        dr_mlp/          best.pt, metrics.csv, plots/
        comparison/
            mae_comparison.csv
            mae_soc_heatmap.png
            mae_cycle_heatmap.png
            summary.txt
==============================================================================
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Reuse data utilities from the ANP pipeline
from train_utils import (
    load_prepared_data,
    validate_targets,
    get_task_splits,
    sort_task_by_cycle,
)


# ==============================================================================
# MLP MODEL
# ==============================================================================

class MLP(nn.Module):
    """
    Three-layer MLP regression model.

    Architecture matches the hidden dimension of the ANP (num_hidden=128)
    to keep the comparison fair in terms of representational capacity per layer.
    The total parameter count is intentionally much smaller than the ANP —
    that is expected for a task-specific supervised baseline.

    Args:
        input_dim:  Number of input features (201 for EIS + Potential).
        output_dim: Number of regression targets (2: SoC%, Cycle).
        neurons:    Hidden layer width (default 128, matching ANP num_hidden).
        dropout:    Dropout probability applied after each hidden activation.
    """

    def __init__(
        self,
        input_dim:  int,
        output_dim: int,
        neurons:    int   = 128,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, neurons),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(neurons, neurons),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(neurons, neurons),
            nn.ReLU(),
            nn.Linear(neurons, output_dim),
        )

        # Weight initialisation consistent with original mlp.py
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                if m.out_features == output_dim:
                    init.xavier_uniform_(m.weight)
                else:
                    init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class Config:
    """All hyperparameters for the MLP baseline experiment."""

    # ── Data ──────────────────────────────────────────────────────────────────
    data_dir: str = "../csic_real_synth_load/prepared_data"

    # Task split — must match the ANP experiment for a fair comparison
    train_task_ids: List[int] = field(default_factory=lambda: list(range(17)))
    val_task_ids:   List[int] = field(default_factory=lambda: list(range(17, 22)))
    test_task_ids:  List[int] = field(default_factory=lambda: list(range(22, 25)))

    # ── Model ─────────────────────────────────────────────────────────────────
    input_dim:  int   = 201
    output_dim: int   = 2
    neurons:    int   = 128    # matches ANP num_hidden
    dropout:    float = 0.1

    # ── Episode windows (same as ANP) ─────────────────────────────────────────
    ctx_cycles:             int = 50
    tgt_cycles:             int = 50
    measurements_per_cycle: int = 30

    # ── Training ──────────────────────────────────────────────────────────────
    epochs:         int   = 500
    early_stopping: int   = 50    # specialist overfits faster than ANP
    lr:             float = 1e-3
    lr_min:         float = 1e-5
    weight_decay:   float = 1e-4
    batch_size:     int   = 256   # row-level batching (not task-level)
    seed:           int   = 42

    # ── Output ────────────────────────────────────────────────────────────────
    run_dir: str = ""

    def __post_init__(self) -> None:
        if not self.run_dir:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = f"./runs_mlp/{ts}"

    @property
    def ctx_rows(self) -> int:
        return self.ctx_cycles * self.measurements_per_cycle

    @property
    def tgt_rows(self) -> int:
        return self.tgt_cycles * self.measurements_per_cycle


# ==============================================================================
# DATA HELPERS
# ==============================================================================

def extract_windows(
    X:        pd.DataFrame,
    y:        pd.DataFrame,
    ctx_rows: int,
    tgt_rows: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract context (train) and target (val) row windows from a task.

    Args:
        X:        Feature DataFrame, cycle-sorted (T, D).
        y:        Target DataFrame,  cycle-sorted (T, O).
        ctx_rows: Number of rows in the context window (train).
        tgt_rows: Number of rows in the target window (val/eval).

    Returns:
        X_ctx, y_ctx  — numpy arrays (ctx_rows, D/O)  — training data
        X_tgt, y_tgt  — numpy arrays (tgt_rows, D/O)  — evaluation data
    """
    T       = len(X)
    ctx_end = min(ctx_rows, T)
    tgt_end = min(ctx_end + tgt_rows, T)

    X_arr = X.values.astype(np.float32)
    y_arr = y.values.astype(np.float32)

    return (
        X_arr[:ctx_end],
        y_arr[:ctx_end],
        X_arr[ctx_end:tgt_end],
        y_arr[ctx_end:tgt_end],
    )


def make_row_batches(
    X:          np.ndarray,
    y:          np.ndarray,
    batch_size: int,
    shuffle:    bool = True,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Yield (X_batch, y_batch) tensors from row arrays.

    Unlike the ANP which batches tasks, the MLP batches individual rows.

    Args:
        X:          Feature array (N, D).
        y:          Target array  (N, O).
        batch_size: Number of rows per batch.
        shuffle:    Whether to shuffle before batching.

    Returns:
        List of (X_batch, y_batch) tensor tuples.
    """
    N   = len(X)
    idx = np.random.permutation(N) if shuffle else np.arange(N)
    batches = []
    for start in range(0, N, batch_size):
        b_idx  = idx[start:start + batch_size]
        batches.append((
            torch.tensor(X[b_idx], dtype=torch.float32),
            torch.tensor(y[b_idx], dtype=torch.float32),
        ))
    return batches


# ==============================================================================
# METRICS
# ==============================================================================

def compute_mae(
    pred:         np.ndarray,
    true:         np.ndarray,
    denorm_values: dict,
    target_cols:   List[str],
) -> Dict[str, float]:
    """
    Compute denormalized MAE per target column.

    Args:
        pred:          Model predictions  (N, O) numpy array.
        true:          Ground truth       (N, O) numpy array.
        denorm_values: Dict with 'y_mean' and 'y_std'.
        target_cols:   Ordered list of target column names.

    Returns:
        Dict {column_name: mae_in_original_units}.
    """
    result = {}
    for i, col in enumerate(target_cols):
        mean_v = denorm_values["y_mean"].get(col, 0.0)
        std_v  = denorm_values["y_std"].get(col, 1.0)
        p_dn   = pred[:, i] * std_v + mean_v
        t_dn   = true[:, i] * std_v + mean_v
        result[col] = float(np.abs(p_dn - t_dn).mean())
    return result


# ==============================================================================
# PLOTTING
# ==============================================================================

_C = {
    "train": "#1C7293",
    "val":   "#C0392B",
    "dpi":   150,
    "lw":    1.8,
}


def plot_mlp_curves(
    metrics_df:  pd.DataFrame,
    target_cols: List[str],
    plots_dir:   Path,
    title:       str = "",
) -> None:
    """
    Generate and save training diagnostic plots for a single MLP model.

    Plots:
        01_loss.png     — train/val MSE loss
        02_mae_SoC.png  — val MAE for SoC(%)
        03_mae_Cycle.png — val MAE for Cycle

    Args:
        metrics_df:  DataFrame with one row per epoch from metrics.csv.
        target_cols: Ordered list of target column names.
        plots_dir:   Directory where PNG files will be saved.
        title:       Optional title prefix for all plots.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    epochs = metrics_df["epoch"].values

    def _col(name: str) -> np.ndarray | None:
        # Ensure we always return a NumPy ndarray (not a pandas ExtensionArray)
        if name not in metrics_df.columns:
            return None
        vals = metrics_df[name].to_numpy()
        # Coerce to a float ndarray to avoid ExtensionArray/ categorical types
        return np.asarray(vals, dtype=float)

    def _save(fig: Figure, path: Path) -> None:
        fig.tight_layout()
        fig.savefig(path, dpi=_C["dpi"], bbox_inches="tight")
        plt.close(fig)

    # Loss
    fig, ax = plt.subplots(figsize=(9, 4))
    for key, label, color in [
        ("train/loss", "Train MSE", _C["train"]),
        ("val/loss",   "Val MSE",   _C["val"]),
    ]:
        v = _col(key)
        if v is not None:
            ax.plot(epochs, v, label=label, color=color, linewidth=_C["lw"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"{title} — Loss")
    ax.legend(); ax.grid(True, alpha=0.3)
    _save(fig, plots_dir / "01_loss.png")

    # MAE per target
    for i, col in enumerate(target_cols, start=2):
        safe = col.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct")
        v = _col(f"val/mae_{safe}")
        if v is None:
            continue
        fig, ax = plt.subplots(figsize=(9, 4))
        mask = ~np.isnan(v.astype(float))
        ax.plot(epochs[mask], v[mask], color=_C["val"], linewidth=_C["lw"],
                label=f"Val MAE — {col}")
        best = np.nanmin(v)
        ax.axhline(best, color="gray", linestyle=":", linewidth=1.0,
                   label=f"Best = {best:.2f}")
        ax.set_xlabel("Epoch"); ax.set_ylabel(f"MAE [{col}]")
        ax.set_title(f"{title} — MAE {col}")
        ax.legend(); ax.grid(True, alpha=0.3)
        _save(fig, plots_dir / f"0{i}_mae_{safe}.png")


def plot_comparison_heatmaps(
    df_soc:   pd.DataFrame,
    df_cycle: pd.DataFrame,
    out_dir:  Path,
) -> None:
    """
    Save heatmaps of MAE_SoC and MAE_Cycle for all models × all tasks.

    Args:
        df_soc:   DataFrame (models × tasks) with SoC MAE values.
        df_cycle: DataFrame (models × tasks) with Cycle MAE values.
        out_dir:  Directory where PNGs are saved.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for df, metric, cmap, fname in [
        (df_soc,   "MAE SoC (%)",  "YlOrRd", "mae_soc_heatmap.png"),
        (df_cycle, "MAE Cycle",    "YlOrRd", "mae_cycle_heatmap.png"),
    ]:
        n_models = len(df)
        n_tasks  = len(df.columns)
        fig_h    = max(4, n_models * 0.4)
        fig_w    = max(8, n_tasks  * 0.55)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(df.values.astype(float), cmap=cmap, aspect="auto")

        ax.set_xticks(range(n_tasks))
        ax.set_xticklabels(df.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n_models))
        ax.set_yticklabels(df.index, fontsize=8)

        plt.colorbar(im, ax=ax, label=metric)

        # Annotate cells
        for r in range(n_models):
            for c in range(n_tasks):
                val = df.values[r, c]
                if not np.isnan(val):
                    ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                            fontsize=6.5, color="black")

        ax.set_title(f"{metric} — all models × all tasks", fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓  {fname}")


# ==============================================================================
# SINGLE MODEL TRAINING
# ==============================================================================

def train_one_mlp(
    model:         MLP,
    X_train:       np.ndarray,
    y_train:       np.ndarray,
    X_val:         np.ndarray,
    y_val:         np.ndarray,
    cfg:           Config,
    device:        torch.device,
    model_dir:     Path,
    denorm_values: dict,
    target_cols:   List[str],
    model_name:    str = "MLP",
) -> pd.DataFrame:
    """
    Train a single MLP model and save checkpoint, metrics CSV, and plots.

    Training procedure:
        - Row-level MSE loss, Adam optimizer, cosine LR annealing.
        - Early stopping on validation MSE with patience = cfg.early_stopping.
        - best.pt saved whenever val loss improves.

    Args:
        model:         Initialized MLP to train.
        X_train:       Training features  (N_ctx, D).
        y_train:       Training targets   (N_ctx, O).
        X_val:         Validation features (N_tgt, D).
        y_val:         Validation targets  (N_tgt, O).
        cfg:           Config dataclass.
        device:        Torch device.
        model_dir:     Directory for this model's outputs.
        denorm_values: Denormalization scalers for MAE computation.
        target_cols:   Target column names.
        model_name:    Label used in plot titles.

    Returns:
        metrics_df: DataFrame with one row per epoch.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr_min)
    criterion = nn.MSELoss()

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    best_soc_mae           = float("inf")
    epochs_without_improve = 0
    rows                   = []

    epoch_pbar = tqdm(
        range(1, cfg.epochs + 1),
        desc=f"{model_name} epochs",
        dynamic_ncols=True,
        leave=False,
    )

    for epoch in epoch_pbar:
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        batches    = make_row_batches(X_train, y_train, cfg.batch_size, shuffle=True)
        batch_losses = []
        batch_pbar = tqdm(
            batches,
            desc=f"{model_name} batches",
            dynamic_ncols=True,
            leave=False,
        )
        for X_b, y_b in batch_pbar:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            batch_pbar.set_postfix({"loss": f"{loss.item():.2f}"})
        train_loss = float(np.mean(batch_losses))

        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            pred_val  = model(X_val_t).cpu().numpy()
            val_loss  = float(criterion(
                torch.tensor(pred_val), torch.tensor(y_val)
            ).item())

        mae = compute_mae(pred_val, y_val, denorm_values, target_cols)
        soc_mae = mae.get("SoC (%)", float("inf"))
        row = {"epoch": epoch, "train/loss": train_loss, "val/loss": val_loss}
        for col, val in mae.items():
            safe = col.replace(" ", "_").replace("(","").replace(")","").replace("%","pct")
            row[f"val/mae_{safe}"] = val
        rows.append(row)
        epoch_pbar.set_postfix(
            {
                "train_loss": f"{train_loss:.2f}",
                "val_loss": f"{val_loss:.2f}",
                "val_mae_soc": f"{soc_mae:.2f}",
                "best_soc_mae": f"{best_soc_mae:.2f}",
            }
        )

        # ── Early stopping ─────────────────────────────────────────────────────
        if soc_mae < best_soc_mae:
            best_soc_mae = soc_mae
            epochs_without_improve = 0
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "val_mae": best_soc_mae},
                model_dir / "best.pt",
            )
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= cfg.early_stopping:
                break

    epoch_pbar.close()

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(model_dir / "metrics.csv", index=False)
    plot_mlp_curves(metrics_df, target_cols, model_dir / "plots", title=model_name)

    return metrics_df


# ==============================================================================
# EVALUATION
# ==============================================================================

@torch.no_grad()
def eval_model_on_task(
    model:         MLP,
    X_eval:        np.ndarray,
    y_eval:        np.ndarray,
    device:        torch.device,
    denorm_values: dict,
    target_cols:   List[str],
) -> Dict[str, float]:
    """
    Evaluate a trained MLP on a given (X_eval, y_eval) set.

    Args:
        model:         Trained MLP in eval mode after this call.
        X_eval:        Feature array (N, D).
        y_eval:        Target array  (N, O).
        device:        Torch device.
        denorm_values: Denormalization scalers.
        target_cols:   Target column names.

    Returns:
        Dict {col_name: mae} in original units.
    """
    model.eval()
    X_t    = torch.tensor(X_eval, dtype=torch.float32).to(device)
    pred   = model(X_t).cpu().numpy()
    return compute_mae(pred, y_eval, denorm_values, target_cols)


# ==============================================================================
# MAIN EXPERIMENT
# ==============================================================================

def run(cfg: Config) -> None:
    """
    Full experiment pipeline:
        1. Load data and build task splits.
        2. Train 17 specialist MLPs (one per training task).
        3. Train 1 DR-MLP on all 17 training tasks concatenated.
        4. Evaluate every model on every task (train + val).
        5. Save comparison table and heatmaps.

    Args:
        cfg: Experiment configuration.
    """
    import random
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🔧  Device : {device}")
    print(f"📁  Run dir: {run_dir}\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    data = load_prepared_data(cfg.data_dir)
    validate_targets(data)

    target_cols    = list(data["normalized_synth_datasets"][0][1].columns)
    cfg.output_dim = len(target_cols)
    cfg.input_dim  = data["normalized_synth_datasets"][0][0].shape[1]

    denorm_values = {
        "y_mean": data["denorm_values"]["y_mean"],
        "y_std":  data["denorm_values"]["y_std"],
    }

    train_tasks, val_tasks, test_tasks = get_task_splits(
        data, cfg.train_task_ids, cfg.val_task_ids, cfg.test_task_ids
    )

    # Cycle-sort all tasks once
    def presort(tasks):
        return [sort_task_by_cycle(X, y) for X, y in tasks]

    train_sorted = presort(train_tasks)
    val_sorted   = presort(val_tasks)
    test_sorted  = presort(test_tasks)

    # Pre-extract context/target windows for all tasks
    def extract_all(tasks):
        return [extract_windows(X, y, cfg.ctx_rows, cfg.tgt_rows)
                for X, y in tasks]

    train_windows = extract_all(train_sorted)
    val_windows   = extract_all(val_sorted)
    test_windows  = extract_all(test_sorted)

    # Save config
    cfg_dict = asdict(cfg)
    cfg_dict.update({"target_cols": target_cols,
                     "ctx_rows": cfg.ctx_rows, "tgt_rows": cfg.tgt_rows})
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)

    # ── Tracking ──────────────────────────────────────────────────────────────
    # Nested dict: results[model_label][task_label] = {col: mae}
    results: dict = {}

    # Task labels for the comparison table
    train_labels = [f"train_{i+1:02d}" for i in cfg.train_task_ids]
    val_labels   = [f"val_{i+1:02d}"   for i in cfg.val_task_ids]
    test_labels  = [f"test_{i+1:02d}"  for i in cfg.test_task_ids]
    all_eval_labels = train_labels + val_labels + test_labels

    # ── 1. Train specialist MLPs ───────────────────────────────────────────────
    print("=" * 60)
    print("  PHASE 1 — Specialist MLPs (one per training task)")
    print("=" * 60)

    specialist_models: list = []

    specialist_iter = tqdm(
        list(zip(cfg.train_task_ids, train_windows)),
        desc="Specialist models",
        dynamic_ncols=True,
    )

    for task_idx, (task_id, (X_ctx, y_ctx, X_tgt, y_tgt)) in enumerate(
        specialist_iter
    ):
        label      = f"specialist_{task_idx+1:02d}"
        model_dir  = run_dir / label
        model_name = f"Specialist {task_idx+1:02d} (task {task_id+1})"

        print(f"\n  [{task_idx+1:02d}/{len(cfg.train_task_ids)}]  {model_name}")
        print(f"   ctx_rows={len(X_ctx)}  tgt_rows={len(X_tgt)}")
        specialist_iter.set_postfix({"current": label})

        model = MLP(cfg.input_dim, cfg.output_dim, cfg.neurons, cfg.dropout)
        print(f"   params: {model.count_params():,}")

        train_one_mlp(
            model, X_ctx, y_ctx, X_tgt, y_tgt,
            cfg, device, model_dir, denorm_values, target_cols, model_name
        )

        # Load best checkpoint
        ckpt = torch.load(model_dir / "best.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        specialist_models.append((label, model))

        # Evaluate on ALL tasks
        results[label] = {}

        for t_label, (Xc, yc, Xt, yt) in zip(train_labels, train_windows):
            results[label][t_label] = eval_model_on_task(
                model, Xt, yt, device, denorm_values, target_cols
            )
        for t_label, (Xc, yc, Xt, yt) in zip(val_labels, val_windows):
            results[label][t_label] = eval_model_on_task(
                model, Xt, yt, device, denorm_values, target_cols
            )
        for t_label, (Xc, yc, Xt, yt) in zip(test_labels, test_windows):
            results[label][t_label] = eval_model_on_task(
                model, Xt, yt, device, denorm_values, target_cols
            )

        print(f"   Best val MAE — "
              f"SoC: {min(results[label][t]['SoC (%)'] for t in train_labels[:1]):.3f}%  "
              f"Cycle: {min(results[label][t]['Cycle'] for t in train_labels[:1]):.2f}")

    # ── 2. Train DR-MLP ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 2 — DR-MLP (all 17 tasks concatenated)")
    print("=" * 60)

    # Concatenate context windows from all training tasks
    X_dr = np.concatenate([w[0] for w in train_windows], axis=0)
    y_dr = np.concatenate([w[1] for w in train_windows], axis=0)

    # For validation: concatenate all target windows
    X_dr_val = np.concatenate([w[2] for w in train_windows], axis=0)
    y_dr_val = np.concatenate([w[3] for w in train_windows], axis=0)

    print(f"\n  DR-MLP train set: {len(X_dr):,} rows  "
          f"({len(train_windows)} tasks × ~{len(X_dr)//len(train_windows)} rows/task)")

    dr_model     = MLP(cfg.input_dim, cfg.output_dim, cfg.neurons, cfg.dropout)
    dr_model_dir = run_dir / "dr_mlp"
    print(f"  params: {dr_model.count_params():,}")

    train_one_mlp(
        dr_model, X_dr, y_dr, X_dr_val, y_dr_val,
        cfg, device, dr_model_dir, denorm_values, target_cols, "DR-MLP"
    )

    ckpt = torch.load(dr_model_dir / "best.pt", map_location=device)
    dr_model.load_state_dict(ckpt["model"])

    results["dr_mlp"] = {}
    for t_label, (Xc, yc, Xt, yt) in zip(train_labels, train_windows):
        results["dr_mlp"][t_label] = eval_model_on_task(
            dr_model, Xt, yt, device, denorm_values, target_cols
        )
    for t_label, (Xc, yc, Xt, yt) in zip(val_labels, val_windows):
        results["dr_mlp"][t_label] = eval_model_on_task(
            dr_model, Xt, yt, device, denorm_values, target_cols
        )
    for t_label, (Xc, yc, Xt, yt) in zip(test_labels, test_windows):
        results["dr_mlp"][t_label] = eval_model_on_task(
            dr_model, Xt, yt, device, denorm_values, target_cols
        )

    # ── 3. Build comparison tables ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 3 — Comparison table")
    print("=" * 60)

    comp_dir = run_dir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    all_model_labels = [label for label, _ in specialist_models] + ["dr_mlp"]

    # One DataFrame per target
    tables: Dict[str, pd.DataFrame] = {}
    for col in target_cols:
        rows_dict = {}
        for m_label in all_model_labels:
            rows_dict[m_label] = {
                t_label: results[m_label].get(t_label, {}).get(col, float("nan"))
                for t_label in all_eval_labels
            }
        tables[col] = pd.DataFrame(rows_dict).T  # models × tasks

    # Save per-target CSVs and combined CSV
    combined_rows = []
    for col in target_cols:
        safe = col.replace(" ", "_").replace("(","").replace(")","").replace("%","pct")
        tables[col].to_csv(comp_dir / f"mae_{safe}.csv")

    # Combined: multi-level column CSV
    for m_label in all_model_labels:
        row = {"model": m_label}
        for t_label in all_eval_labels:
            for col in target_cols:
                safe = col.replace(" ", "_").replace("(","").replace(")","").replace("%","pct")
                row[f"{t_label}/mae_{safe}"] = \
                    results[m_label].get(t_label, {}).get(col, float("nan"))
        combined_rows.append(row)

    combined_df = pd.DataFrame(combined_rows).set_index("model")
    combined_df.to_csv(comp_dir / "mae_comparison.csv")
    print(f"\n  ✓  mae_comparison.csv saved")

    # ── 4. Heatmaps ───────────────────────────────────────────────────────────
    soc_col   = "SoC (%)" if "SoC (%)" in target_cols else target_cols[0]
    cycle_col = "Cycle"   if "Cycle"   in target_cols else target_cols[-1]

    plot_comparison_heatmaps(
        tables[soc_col], tables[cycle_col], comp_dir
    )

    # ── 5. Summary printout ───────────────────────────────────────────────────
    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("COMPARISON SUMMARY — MAE in original units")
    summary_lines.append("=" * 70)

    # Average MAE per model on train tasks vs val tasks
    summary_lines.append(
        f"\n{'Model':<22} {'Train MAE SoC':>14} {'Train MAE Cyc':>14} "
        f"{'Val MAE SoC':>12} {'Val MAE Cyc':>12} {'Test MAE SoC':>13} {'Test MAE Cyc':>13}"
    )
    summary_lines.append("-" * 100)

    for m_label in all_model_labels:
        def avg(labels, col):
            vals = [results[m_label].get(t, {}).get(col, float("nan")) for t in labels]
            vals = [v for v in vals if not np.isnan(v)]
            return np.mean(vals) if vals else float("nan")

        tr_soc  = avg(train_labels, soc_col)
        tr_cyc  = avg(train_labels, cycle_col)
        val_soc = avg(val_labels,   soc_col)
        val_cyc = avg(val_labels,   cycle_col)
        tst_soc = avg(test_labels,  soc_col)
        tst_cyc = avg(test_labels,  cycle_col)

        summary_lines.append(
            f"{m_label:<22} {tr_soc:>14.3f} {tr_cyc:>14.2f} "
            f"{val_soc:>12.3f} {val_cyc:>12.2f} {tst_soc:>13.3f} {tst_cyc:>13.2f}"
        )

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    with open(comp_dir / "summary.txt", "w") as f:
        f.write(summary_text)

    print(f"\n✅  All outputs saved to: {run_dir}\n")


# ==============================================================================
# CLI
# ==============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Specialist MLP + DR-MLP baseline experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir",       type=str,   default=None)
    p.add_argument("--run_dir",        type=str,   default="")
    p.add_argument("--neurons",        type=int,   default=128,
                   help="Hidden layer width (matches ANP num_hidden)")
    p.add_argument("--dropout",        type=float, default=0.1)
    p.add_argument("--ctx_cycles",     type=int,   default=60)
    p.add_argument("--tgt_cycles",     type=int,   default=60)
    p.add_argument("--meas_per_cycle", type=int,   default=30,
                   dest="measurements_per_cycle")
    p.add_argument("--epochs",         type=int,   default=1000)
    p.add_argument("--early_stop",     type=int,   default=200,
                   dest="early_stopping")
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--batch_size",     type=int,   default=256)
    p.add_argument("--weight_decay",   type=float, default=1e-4)
    p.add_argument("--seed",           type=int,   default=18)
    p.add_argument("--train_ids",      type=int,   nargs="+", default=None)
    p.add_argument("--val_ids",        type=int,   nargs="+", default=None)
    p.add_argument("--test_ids",       type=int,   nargs="+", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = Config(
        data_dir               = args.data_dir or Config.data_dir,
        run_dir                = args.run_dir,
        neurons                = args.neurons,
        dropout                = args.dropout,
        ctx_cycles             = args.ctx_cycles,
        tgt_cycles             = args.tgt_cycles,
        measurements_per_cycle = args.measurements_per_cycle,
        epochs                 = args.epochs,
        early_stopping         = args.early_stopping,
        lr                     = args.lr,
        batch_size             = args.batch_size,
        weight_decay           = args.weight_decay,
        seed                   = args.seed,
    )
    if args.train_ids is not None: cfg.train_task_ids = args.train_ids
    if args.val_ids   is not None: cfg.val_task_ids   = args.val_ids
    if args.test_ids  is not None: cfg.test_task_ids  = args.test_ids

    run(cfg)


if __name__ == "__main__":
    main()
