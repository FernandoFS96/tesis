#!/usr/bin/env python3
"""
finetune_ood.py
===============
Fine-tunes Optuna-optimized ANP/RANP models trained on high-variance data
towards a low-variance (OoD) target domain, comparing multiple parameter-
efficient strategies and data budgets.

Scenario
--------
  Source model : trained on HIGH-variance data   (highvar Optuna best)
  Target domain: LOW-variance data               (the OoD gap to close)
  Oracle       : model trained natively on lowvar (upper-bound target)

Fine-tuning strategies
----------------------
  decoder_heads    – only output heads of Decoder (mean_projection + log_var_projection)
  decoder_full     – entire Decoder module
  decoder_det_last – Decoder + last cross-attention block of DeterministicEncoder
  decoder_lat_last – Decoder + mu/log_var heads of LatentEncoder
  decoder_det_full – Decoder + all cross-attention blocks of DeterministicEncoder

Data budgets
------------
  n_traj in {10, 20, 50, 100, 200, 300, all}

Baselines (computed automatically)
------------------------------------
  ood_baseline – highvar model evaluated on lowvar data, zero fine-tuning
  oracle       – lowvar model evaluated on lowvar data (native performance)

Outputs
-------
  results/finetune_ood/
    topology_<topo>/model_<anp|ranp>/
      finetune_summary.csv
      baselines.json
      checkpoints/
        <strategy>/n_traj_<n>/
          finetuned_checkpoint.pth.tar
          finetune_log.csv
      plots/
        mae_vs_ntraj_ctx<pct>.png
        gap_closed_vs_ntraj_ctx<pct>.png
        time_vs_ntraj.png
        pareto_time_vs_mae_ctx<pct>.png
        context_sweep_ntraj_<n>.png
        finetune_curves.png

Usage
-----
    cd <project-root>
    python finetune_ood.py \
        --optuna-root  /home/fernando/tesis/underwater-localization-topologies/src/training/results/optuna \
        --lowvar-data  /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_low_variance \
        --topologies   ellipsoidal,random,aligned \
        --model-types  anp,ranp \
        --study-version v2 \
        --n-traj       10,20,50,100,200,300,all \
        --strategies   decoder_heads,decoder_full,decoder_det_last,decoder_det_full,decoder_lat_last \
        --lr           1e-4 \
        --epochs       1000 \
        --patience     150 \
        --context-fracs 0.1,0.2,0.3,0.4,0.5,0.6,0.7 \
        --output-dir   results/finetune_ood \
        --device       cuda

    Example for a single config:
    python finetune_ood.py \
        --optuna-root  /home/fernando/tesis/underwater-localization-topologies/src/training/results/optuna \
        --lowvar-data  /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_low_variance \
        --topologies   ellipsoidal,random,aligned \
        --model-types  anp \
        --study-version v2 \
        --n-traj       10,20,50,100,200,300,all \
        --strategies   decoder_heads,decoder_full,decoder_det_last,decoder_det_full,decoder_lat_last \
        --lr           5e-4 \
        --epochs       1000 \
        --patience     100 \
        --context-fracs 0.1,0.2,0.3,0.4,0.5,0.6,0.7 \
        --output-dir   results/finetune_ood \
        --n-seed       5 \
        --skip-existing \
        --device       cuda \

    python finetune_ood.py \
        --optuna-root  /home/fernando/tesis/underwater-localization-topologies/src/training/results/optuna \
        --lowvar-data  /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_low_variance \
        --topologies   ellipsoidal,random,aligned \
        --model-types  ranp \
        --study-version v2 \
        --n-traj       10,20,50,100,200,300,all \
        --strategies   decoder_full,decoder_det_last,rnn_proj_only,rnn_proj_decoder,rnn_full_decoder \
        --lr           5e-4 \
        --epochs       1000 \
        --patience     100 \
        --context-fracs 0.1,0.2,0.3,0.4,0.5,0.6,0.7 \
        --output-dir   results/finetune_ood \
        --n-seed       5 \
        --skip-existing \
        --device       cuda
Notes
-----
- Uses existing val split for early-stopping; test split for final evaluation.
- Normalization stats recomputed from lowvar training data (target domain).
- Each strategy × n_traj combination starts from a fresh deep-copy of the
  source (highvar) model, so runs are fully independent.
- The --skip-existing flag skips configs that already have a saved checkpoint,
  allowing restartable runs.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
from tqdm import tqdm
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── project imports (run from project root) ──────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.load_optuna_model import (
    load_optuna_best_model,
    resolve_optuna_best_model_dir,
)

# =============================================================================
# Constants
# =============================================================================

NUM_SENSORS     = 10
NUM_TIME_POINTS = 201
OUTPUT_DIM      = 3

DEFAULT_STRATEGIES = [
    "decoder_heads",
    "decoder_full",
    "decoder_det_last",
    "decoder_det_full",
    "decoder_lat_last",
]
DEFAULT_N_TRAJ = [10, 20, 50, 100, 200, 300, "all"]

STRATEGY_COLORS = {
    "decoder_heads":    "#2980b9",
    "decoder_full":     "#27ae60",
    "decoder_det_last": "#e67e22",
    "decoder_det_full": "#c0392b",
    "decoder_lat_last": "#8e44ad",
    # RANP-specific strategies (touch TemporalEncoder)
    "rnn_proj_only":    "#1abc9c",
    "rnn_proj_decoder": "#f39c12",
    "rnn_full_decoder": "#e74c3c",
}
STRATEGY_LABELS = {
    "decoder_heads":    "Heads only  (mean+var proj.)",
    "decoder_full":     "Full Decoder",
    "decoder_det_last": "Full Decoder + Det.Enc. last cross-attn",
    "decoder_det_full": "Full Decoder + Full Det.Enc.",
    "decoder_lat_last": "Full Decoder + Lat.Enc. μ/σ heads",
    # RANP-specific strategies (touch TemporalEncoder)
    "rnn_proj_only":    "RANP: RNN input_proj + LayerNorm",
    "rnn_proj_decoder": "RANP: RNN input_proj + LayerNorm + Full Decoder",
    "rnn_full_decoder": "RANP: Full TemporalEncoder + Full Decoder",
}

# =============================================================================
# Data utilities
# =============================================================================

def load_topology_data(data_dir: str | Path, topology: str):
    """Load train / val / test pickles for a given topology directory."""
    topo_dir = Path(data_dir) / f"topology_{topology}"
    if not topo_dir.exists():
        raise FileNotFoundError(f"Topology dir not found: {topo_dir}")

    with open(topo_dir / "train_data.pkl", "rb") as f:
        train_data = pickle.load(f)
    with open(topo_dir / "val_data.pkl", "rb") as f:
        val_data = pickle.load(f)

    test_path = topo_dir / "test_data.pkl"
    test_data = None
    if test_path.exists():
        with open(test_path, "rb") as f:
            test_data = pickle.load(f)

    return train_data, val_data, test_data


def compute_y_stats(data) -> Tuple[torch.Tensor, torch.Tensor]:
    Y     = np.concatenate([y for _, y in data], axis=0)
    mean  = torch.tensor(Y.mean(0), dtype=torch.float32)
    std   = torch.tensor(Y.std(0) + 1e-6, dtype=torch.float32)
    return mean, std


def compute_x_means(data, num_time_points: int, num_sensors: int) -> torch.Tensor:
    """Returns (S, P) sensor-mean tensor used as mask fill values."""
    X  = np.concatenate([x for x, _ in data], axis=0)          # (N*T, Dx)
    X3 = X.reshape(X.shape[0], num_time_points, num_sensors)    # (N*T, P, S)
    return torch.tensor(X3.mean(0).T, dtype=torch.float32)      # (S, P)


def apply_mask_and_append(
    x_batch:    torch.Tensor,   # (B, T, Dx)
    sensor_mask: torch.Tensor,  # (B, S) float, 1=available
    x_means_SP: torch.Tensor,   # (S, P)
    num_time_points: int,
    num_sensors: int,
) -> torch.Tensor:
    """Fill dropped sensors with training mean and append binary mask features."""
    B, T, Dx = x_batch.shape
    P, S     = num_time_points, num_sensors

    x4  = x_batch.view(B, T, P, S)
    mu  = x_means_SP.T.view(1, 1, P, S).to(x_batch.device, dtype=x_batch.dtype)
    m   = sensor_mask.view(B, 1, 1, S)
    x4m = x4 * m + mu * (1.0 - m)

    x_masked  = x4m.reshape(B, T, Dx)
    mask_feat = sensor_mask.view(B, 1, S).expand(B, T, S)
    return torch.cat([x_masked, mask_feat], dim=-1)             # (B, T, Dx+S)


def make_dataloader(data, batch_size: int, shuffle: bool = False) -> DataLoader:
    xs = torch.tensor(np.stack([x for x, _ in data]), dtype=torch.float32)
    ys = torch.tensor(np.stack([y for _, y in data]), dtype=torch.float32)
    return DataLoader(TensorDataset(xs, ys), batch_size=batch_size, shuffle=shuffle)


def build_inverse_holdout_indices(
    total_points: int,
    holdout_frac: float,
    context_frac: float,
    device: str | torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Context sits immediately before the holdout tail (inverse protocol)."""
    n_holdout    = max(1, int(round(holdout_frac * total_points)))
    holdout_start = total_points - n_holdout
    max_ctx      = max(1, holdout_start)
    ctx_size     = max(1, min(max_ctx, int(round(context_frac * max_ctx))))
    ctx_start    = holdout_start - ctx_size
    ctx_idx      = torch.arange(ctx_start, holdout_start, device=device)
    tar_idx      = torch.arange(holdout_start, total_points, device=device)
    return ctx_idx, tar_idx


# =============================================================================
# Fine-tuning parameter selection & freezing
# =============================================================================

def get_finetune_params(
    model: nn.Module,
    strategy: str,
    model_type: str = "anp",
) -> List[nn.Parameter]:
    """
    Return the parameter list to be optimized for each strategy.
    All other model parameters will be frozen.

    ANP strategies (work on both ANP and RANP)
    -------------------------------------------
    decoder_heads    : mean_projection + log_var_projection  (output heads only)
    decoder_full     : entire model.decoder
    decoder_det_last : decoder_full + last cross-attn block of DeterministicEncoder
    decoder_det_full : decoder_full + full DeterministicEncoder
    decoder_lat_last : decoder_full + mu/log_var heads of LatentEncoder

    RANP-only strategies (require model.temporal_encoder)
    -----------------------------------------------------
    rnn_proj_only    : TemporalEncoder.input_proj + LayerNorm only
    rnn_proj_decoder : rnn_proj_only + full Decoder
    rnn_full_decoder : full TemporalEncoder + full Decoder
    """
    if strategy not in STRATEGY_LABELS:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Valid options: {list(STRATEGY_LABELS.keys())}"
        )

    ranp_strategies = {"rnn_proj_only", "rnn_proj_decoder", "rnn_full_decoder"}
    if strategy in ranp_strategies and model_type != "ranp":
        raise ValueError(
            f"Strategy '{strategy}' is only valid for RANP models "
            f"(model_type='{model_type}' given)."
        )

    params: List[nn.Parameter] = []

    if strategy == "decoder_heads":
        params += list(model.decoder.mean_projection.parameters()) #type: ignore
        params += list(model.decoder.log_var_projection.parameters()) #type: ignore

    elif strategy in ("decoder_full", "decoder_det_last", "decoder_det_full", "decoder_lat_last"):
        params += list(model.decoder.parameters()) #type: ignore

        if strategy == "decoder_det_last":
            params += list(
                model.deterministic_encoder.cross_attentions[-1].parameters() #type: ignore
            )

        elif strategy == "decoder_det_full":
            params += list(model.deterministic_encoder.parameters()) #type: ignore

        elif strategy == "decoder_lat_last":
            params += list(model.latent_encoder.mu.parameters()) #type: ignore
            params += list(model.latent_encoder.log_var.parameters()) #type: ignore

    elif strategy == "rnn_proj_only":
        # Only input_proj + LayerNorm: lightest RNN components, directly see domain features.
        params += list(model.temporal_encoder.input_proj.parameters()) #type: ignore
        params += list(model.temporal_encoder.norm.parameters()) #type: ignore

    elif strategy == "rnn_proj_decoder":
        # input_proj + LayerNorm + full Decoder: combines both adaptation axes.
        params += list(model.temporal_encoder.input_proj.parameters()) #type: ignore
        params += list(model.temporal_encoder.norm.parameters()) #type: ignore
        params += list(model.decoder.parameters()) #type: ignore

    elif strategy == "rnn_full_decoder":
        # Full TemporalEncoder (input_proj, LayerNorm, RNN weights) + full Decoder.
        params += list(model.temporal_encoder.parameters()) #type: ignore
        params += list(model.decoder.parameters()) #type: ignore

    return params

# =============================================================================
# Unified forward pass (ANP / RANP)
# =============================================================================

def freeze_all_except(model: nn.Module, active_params: List[nn.Parameter]) -> None:
    """Freeze every parameter not in active_params."""
    active_ids = {id(p) for p in active_params}
    for p in model.parameters():
        p.requires_grad_(id(p) in active_ids)


# =============================================================================
# Unified forward pass (ANP / RANP)
# =============================================================================

def model_forward(
    model:      nn.Module,
    model_type: str,
    x_aug:      torch.Tensor,   # (B, T, Dx+S)
    ctx_idx:    torch.Tensor,   # (Nc,)
    ctx_y:      torch.Tensor,   # (B, Nc, 3)
    tar_idx:    torch.Tensor,   # (Nt,)
    tar_y:      Optional[torch.Tensor] = None,
    beta:       float = 1.0,
):
    """Dispatch to the correct forward signature depending on model_type."""
    if model_type == "anp":
        ctx_x = x_aug[:, ctx_idx, :]
        tar_x = x_aug[:, tar_idx, :]
        return model(ctx_x, ctx_y, tar_x, tar_y, beta=beta)
    elif model_type == "ranp":
        return model(
            x_seq            = x_aug,
            context_indices  = ctx_idx,
            context_y        = ctx_y,
            target_indices   = tar_idx,
            target_y         = tar_y,
            beta             = beta,
        )
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'")


# =============================================================================
# Fine-tuning loop
# =============================================================================

def subsample_data(data: list, n: int | str, seed: int = 18) -> list:
    """Return at most *n* trajectories sampled without replacement."""
    if n == "all" or int(n) >= len(data):
        return data
    rng     = np.random.default_rng(seed)
    indices = rng.choice(len(data), size=int(n), replace=False)
    return [data[i] for i in sorted(indices)]


def finetune_model(
    model:       nn.Module,
    model_type:  str,
    strategy:    str,
    train_data:  list,
    val_data:    list,
    y_mean:      torch.Tensor,
    y_std:       torch.Tensor,
    x_means_SP:  torch.Tensor,
    n_traj:      int | str,
    lr:          float,
    epochs:      int,
    patience:    int,
    batch_size:  int,
    device:      str | torch.device,
    save_dir:    Path,
    holdout_frac:    float = 0.2,
    es_context_frac: float = 0.4,
    seed:            int   = 0,
) -> Dict:
    """
    Fine-tune *model* in-place using the given strategy and data budget.

    Returns a metadata dict with training stats (best val MAE, time, etc.).
    The best checkpoint is written to save_dir / finetuned_checkpoint.pth.tar.
    The per-epoch log is written to save_dir / finetune_log.csv.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── subsample fine-tuning data ───────────────────────────────────────────
    # ── reproducibilidad de la run ────────────────────────────────────────────
    torch.manual_seed(seed)
    np.random.seed(seed)

    ft_data      = subsample_data(train_data, n_traj, seed=seed)
    train_loader = make_dataloader(ft_data, batch_size, shuffle=True)
    val_loader   = make_dataloader(val_data, batch_size, shuffle=False)

    y_mean     = y_mean.to(device)
    y_std      = y_std.to(device)
    x_means_SP = x_means_SP.to(device)

    # ── freeze / unfreeze ────────────────────────────────────────────────────
    active_params = get_finetune_params(model, strategy, model_type)
    freeze_all_except(model, active_params)
    n_trainable = sum(p.numel() for p in active_params)
    n_total     = sum(p.numel() for p in model.parameters())
    print(
        f"    Trainable: {n_trainable:,} / {n_total:,} params  "
        f"({100*n_trainable/max(1,n_total):.1f}%)"
    )

    optimizer = torch.optim.Adam(active_params, lr=lr, weight_decay=1e-5)

    # ── training state ───────────────────────────────────────────────────────
    best_val_mae      = float("inf")
    patience_counter  = 0
    best_state        = copy.deepcopy(model.state_dict())
    log_rows: list    = []
    t_start           = time.time()

    epoch_iter = tqdm(
        range(epochs),
        desc=f"FT[{strategy}|n={n_traj}]",
        leave=False,
        dynamic_ncols=True,
        disable=not sys.stdout.isatty(),
    )

    for epoch in epoch_iter:

        # ── train ────────────────────────────────────────────────────────────
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            B, T, _ = x_batch.shape

            sensor_mask = torch.ones(B, NUM_SENSORS, device=device)
            x_aug = apply_mask_and_append(
                x_batch, sensor_mask, x_means_SP, NUM_TIME_POINTS, NUM_SENSORS
            )

            # random context size in [10%, 90%] — consistent with pretraining
            min_ctx  = max(1, int(0.1 * T))
            max_ctx  = min(int(0.9 * T), T - 1)
            ctx_size = (
                torch.randint(min_ctx, max_ctx + 1, (1,)).item()
                if max_ctx > min_ctx else min_ctx
            )
            ctx_idx = torch.arange(ctx_size, device=device)
            tar_idx = torch.arange(ctx_size, T, device=device)

            y_norm = (y_batch - y_mean) / y_std
            ctx_y  = y_norm[:, ctx_idx, :]
            tar_y  = y_norm[:, tar_idx, :]

            optimizer.zero_grad()
            _, _, loss, _, _ = model_forward(
                model, model_type, x_aug, ctx_idx, ctx_y, tar_idx, tar_y, beta=1.0
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(active_params, max_norm=1.0)
            optimizer.step()

        # ── validation (inverse holdout) ─────────────────────────────────────
        model.eval()
        val_mae_acc = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                B, T, _ = x_batch.shape

                sensor_mask = torch.ones(B, NUM_SENSORS, device=device)
                x_aug = apply_mask_and_append(
                    x_batch, sensor_mask, x_means_SP, NUM_TIME_POINTS, NUM_SENSORS
                )

                y_norm  = (y_batch - y_mean) / y_std
                ctx_idx, tar_idx = build_inverse_holdout_indices(
                    T, holdout_frac, es_context_frac, device
                )
                ctx_y = y_norm[:, ctx_idx, :]

                y_pred_norm, *_ = model_forward(
                    model, model_type, x_aug, ctx_idx, ctx_y, tar_idx
                )
                y_pred = y_pred_norm * y_std + y_mean
                mae    = F.l1_loss(y_pred, y_batch[:, tar_idx, :], reduction="mean").item()
                val_mae_acc  += mae
                n_val_batches += 1

        val_mae = val_mae_acc / max(1, n_val_batches)
        elapsed = time.time() - t_start
        log_rows.append([epoch + 1, val_mae, elapsed])

        if val_mae < best_val_mae:
            best_val_mae     = val_mae
            patience_counter = 0
            best_state       = copy.deepcopy(model.state_dict())
            torch.save({"model": best_state}, save_dir / "finetuned_checkpoint.pth.tar")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"    Early stop  epoch={epoch+1}  "
                f"best_val_mae={best_val_mae:.4f} m"
            )
            break

        epoch_iter.set_postfix({
            "val_mae": f"{val_mae:.4f}",
            "best": f"{best_val_mae:.4f}",
            "ES": f"{patience_counter}/{patience}",
        })

    total_time = time.time() - t_start

    # restore best weights
    model.load_state_dict(best_state)

    # persist epoch log
    with open(save_dir / "finetune_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "val_mae_m", "elapsed_s"])
        w.writerows(log_rows)

    return {
        "best_val_mae":       best_val_mae,
        "total_time_s":       total_time,
        "n_epochs":           len(log_rows),
        "n_traj_used":        len(ft_data),
        "n_trainable_params": n_trainable,
    }


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_model(
    model:       nn.Module,
    model_type:  str,
    loader:      DataLoader,
    y_mean:      torch.Tensor,
    y_std:       torch.Tensor,
    x_means_SP:  torch.Tensor,
    context_fracs: List[float],
    device:      str | torch.device,
    holdout_frac: float = 0.2,
) -> Dict[float, float]:
    """
    Evaluate model on *loader* for each context fraction using the
    inverse_context_holdout protocol.  Returns {frac: mean_MAE_m}.
    """
    y_mean     = y_mean.to(device)
    y_std      = y_std.to(device)
    x_means_SP = x_means_SP.to(device)

    mae_sums  = {f: 0.0 for f in context_fracs}
    n_batches = 0

    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            B, T, _ = x_batch.shape

            sensor_mask = torch.ones(B, NUM_SENSORS, device=device)
            x_aug = apply_mask_and_append(
                x_batch, sensor_mask, x_means_SP, NUM_TIME_POINTS, NUM_SENSORS
            )
            y_norm = (y_batch - y_mean) / y_std

            for frac in context_fracs:
                ctx_idx, tar_idx = build_inverse_holdout_indices(
                    T, holdout_frac, frac, device
                )
                ctx_y = y_norm[:, ctx_idx, :]
                y_pred_norm, *_ = model_forward(
                    model, model_type, x_aug, ctx_idx, ctx_y, tar_idx
                )
                y_pred = y_pred_norm * y_std + y_mean
                mae    = F.l1_loss(
                    y_pred, y_batch[:, tar_idx, :], reduction="mean"
                ).item()
                mae_sums[frac] += mae

            n_batches += 1

    return {f: mae_sums[f] / max(1, n_batches) for f in context_fracs}


# =============================================================================
# Plotting helpers
# =============================================================================

def _n_traj_x_axis(n_traj_values):
    """Return integer x-tick positions and string labels for n_traj."""
    ticks  = list(range(len(n_traj_values)))
    labels = [str(n) for n in n_traj_values]
    return ticks, labels


def plot_mae_vs_ntraj(
    mae_results:       Dict,
    mae_std_results:   Dict,
    ood_by_frac:       Dict[float, float],
    oracle_by_frac:    Dict[float, float],
    strategies:        List[str],
    n_traj_values:     List,
    context_frac:      float,
    save_path:         Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ticks, labels = _n_traj_x_axis(n_traj_values)

    ood    = ood_by_frac[context_frac]
    oracle = oracle_by_frac[context_frac]
    ax.axhline(ood,    color="#e74c3c", ls="--", lw=1.8, label="OoD baseline (no FT)")
    ax.axhline(oracle, color="#2c3e50", ls="--", lw=1.8, label="Oracle (native domain)")
    ax.fill_between(
        [-0.3, len(n_traj_values) - 0.7], oracle, ood,
        alpha=0.07, color="#e74c3c"
    )

    for strategy in strategies:
        y     = np.array([
            mae_results.get((strategy, str(n)), {}).get(context_frac, float("nan"))
            for n in n_traj_values
        ], dtype=float)
        y_std = np.array([
            mae_std_results.get((strategy, str(n)), {}).get(context_frac, 0.0)
            for n in n_traj_values
        ], dtype=float)
        ax.plot(
            ticks, y, marker="o",
            color=STRATEGY_COLORS[strategy],
            label=STRATEGY_LABELS[strategy],
            linewidth=2, markersize=7
        )
        ax.fill_between(
            ticks, y - y_std, y + y_std,
            alpha=0.15, color=STRATEGY_COLORS[strategy]
        )

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlabel("N trajectories for fine-tuning", fontsize=11)
    ax.set_ylabel("MAE  (m)  —  inverse holdout", fontsize=11)
    ax.set_title(
        f"MAE vs. data budget   |   ctx = {int(context_frac*100)}%",
        fontsize=12
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"  [plot] {save_path}")


def plot_gap_closed(
    mae_results:       Dict,
    mae_std_results:   Dict,
    ood_by_frac:       Dict[float, float],
    oracle_by_frac:    Dict[float, float],
    strategies:        List[str],
    n_traj_values:     List,
    context_frac:      float,
    save_path:         Path,
) -> None:
    ood    = ood_by_frac[context_frac]
    oracle = oracle_by_frac[context_frac]
    gap    = ood - oracle
    if gap <= 0:
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    ticks, labels = _n_traj_x_axis(n_traj_values)

    ax.axhline(100.0, color="#2c3e50", ls="--", lw=1.5, label="Oracle  (100 %)")
    ax.axhline(  0.0, color="#e74c3c", ls="--", lw=1.5, label="OoD baseline  (0 %)")

    for strategy in strategies:
        y, y_lo, y_hi = [], [], []
        for n in n_traj_values:
            mae  = mae_results.get((strategy, str(n)), {}).get(context_frac, float("nan"))
            std  = mae_std_results.get((strategy, str(n)), {}).get(context_frac, 0.0)
            pct  = 100.0 * (ood - mae) / gap if np.isfinite(mae) else float("nan")
            plo  = 100.0 * (ood - (mae + std)) / gap if np.isfinite(mae) else float("nan")
            phi  = 100.0 * (ood - (mae - std)) / gap if np.isfinite(mae) else float("nan")
            y.append(pct); y_lo.append(plo); y_hi.append(phi)
        y_arr = np.array(y, dtype=float)
        ax.plot(
            ticks, y_arr, marker="o",
            color=STRATEGY_COLORS[strategy],
            label=STRATEGY_LABELS[strategy],
            linewidth=2, markersize=7
        )
        ax.fill_between(
            ticks, np.array(y_lo, dtype=float), np.array(y_hi, dtype=float),
            alpha=0.15, color=STRATEGY_COLORS[strategy]
        )

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlabel("N trajectories for fine-tuning", fontsize=11)
    ax.set_ylabel("% of OoD gap closed", fontsize=11)
    ax.set_title(
        f"Gap closure  |  ctx = {int(context_frac*100)}%"
        f"  (gap = {gap:.2f} m)",
        fontsize=12
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"  [plot] {save_path}")


def plot_time_vs_ntraj(
    time_results:  Dict,
    strategies:    List[str],
    n_traj_values: List,
    save_path:     Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ticks, labels = _n_traj_x_axis(n_traj_values)

    for strategy in strategies:
        y = [time_results.get((strategy, str(n)), float("nan")) for n in n_traj_values]
        ax.plot(
            ticks, y, marker="s",
            color=STRATEGY_COLORS[strategy],
            label=STRATEGY_LABELS[strategy],
            linewidth=2, markersize=7
        )

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlabel("N trajectories for fine-tuning", fontsize=11)
    ax.set_ylabel("Fine-tuning time (s)", fontsize=11)
    ax.set_title("Fine-tuning time vs. data budget", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"  [plot] {save_path}")


def plot_pareto(
    time_results:   Dict,
    mae_results:    Dict,
    strategies:     List[str],
    n_traj_values:  List,
    context_frac:   float,
    ood_mae:        float,
    oracle_mae:     float,
    save_path:      Path,
) -> None:
    """Pareto scatter: x = time, y = MAE.  Each point annotated with n_traj."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axhline(ood_mae,    color="#e74c3c", ls="--", lw=1.2, alpha=0.75, label="OoD baseline")
    ax.axhline(oracle_mae, color="#2c3e50", ls="--", lw=1.2, alpha=0.75, label="Oracle")

    for strategy in strategies:
        xs, ys, lbls = [], [], []
        for n in n_traj_values:
            t = time_results.get((strategy, str(n)), float("nan"))
            m = mae_results.get((strategy, str(n)), {}).get(context_frac, float("nan"))
            if np.isfinite(t) and np.isfinite(m):
                xs.append(t)
                ys.append(m)
                lbls.append(str(n))
        if not xs:
            continue
        ax.plot(xs, ys, "-o",
                color=STRATEGY_COLORS[strategy],
                label=STRATEGY_LABELS[strategy],
                linewidth=1.5, markersize=7)
        for x, y, lbl in zip(xs, ys, lbls):
            ax.annotate(lbl, (x, y), textcoords="offset points",
                        xytext=(4, 4), fontsize=7, alpha=0.85)

    ax.set_xlabel("Fine-tuning time (s)", fontsize=11)
    ax.set_ylabel(f"MAE (m) — ctx = {int(context_frac*100)}%", fontsize=11)
    ax.set_title("Pareto frontier: time vs. quality", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"  [plot] {save_path}")


def plot_context_sweep(
    mae_results:        Dict,
    context_fracs:      List[float],
    ood_by_frac:        Dict[float, float],
    oracle_by_frac:     Dict[float, float],
    strategies:         List[str],
    n_traj_highlight,
    save_path:          Path,
) -> None:
    """MAE vs. context fraction for each strategy at a fixed n_traj."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = [int(f * 100) for f in context_fracs]

    ax.plot(x, [ood_by_frac[f]    for f in context_fracs],
            color="#e74c3c", ls="--", lw=1.8, marker="s", markersize=6,
            label="OoD baseline")
    ax.plot(x, [oracle_by_frac[f] for f in context_fracs],
            color="#2c3e50", ls="--", lw=1.8, marker="s", markersize=6,
            label="Oracle")

    for strategy in strategies:
        y = [
            mae_results.get((strategy, str(n_traj_highlight)), {}).get(f, float("nan"))
            for f in context_fracs
        ]
        ax.plot(x, y, marker="o",
                color=STRATEGY_COLORS[strategy],
                label=STRATEGY_LABELS[strategy],
                linewidth=2, markersize=6)

    ax.set_xlabel("Context fraction (%)", fontsize=11)
    ax.set_ylabel("MAE  (m)", fontsize=11)
    ax.set_title(
        f"MAE vs. context fraction  |  n_traj = {n_traj_highlight}",
        fontsize=12
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"  [plot] {save_path}")


def plot_finetune_curves(
    log_base_dir:  Path,
    strategies:    List[str],
    n_traj_values: List,
    ood_mae:       float,
    oracle_mae:    float,
    save_path:     Path,
) -> None:
    """Val-MAE curves over fine-tuning epochs for 3 representative n_traj values."""
    n_sel = min(3, len(n_traj_values))
    idxs  = [0, len(n_traj_values) // 2, len(n_traj_values) - 1][:n_sel]
    sel_n = [n_traj_values[i] for i in idxs]

    fig, axes = plt.subplots(1, n_sel, figsize=(7 * n_sel, 5), sharey=True)
    if n_sel == 1:
        axes = [axes]

    for ax, n in zip(axes, sel_n):
        ax.axhline(ood_mae,    color="#e74c3c", ls="--", lw=1.2, alpha=0.7, label="OoD baseline")
        ax.axhline(oracle_mae, color="#2c3e50", ls="--", lw=1.2, alpha=0.7, label="Oracle")

        for strategy in strategies:
            log_path = log_base_dir / strategy / f"n_traj_{n}" / "finetune_log.csv"
            if not log_path.exists():
                continue
            epochs_l, maes = [], []
            with open(log_path, newline="") as f:
                for row in csv.DictReader(f):
                    epochs_l.append(int(row["epoch"]))
                    maes.append(float(row["val_mae_m"]))
            ax.plot(epochs_l, maes,
                    color=STRATEGY_COLORS[strategy],
                    label=STRATEGY_LABELS[strategy],
                    linewidth=1.6, alpha=0.9)

        ax.set_title(f"n_traj = {n}", fontsize=10)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Val MAE (m)", fontsize=11)

    handles = (
        [Line2D([0], [0], color=STRATEGY_COLORS[s], lw=2, label=STRATEGY_LABELS[s])
         for s in strategies]
        + [
            Line2D([0], [0], color="#e74c3c", lw=1.5, ls="--", label="OoD baseline"),
            Line2D([0], [0], color="#2c3e50", lw=1.5, ls="--", label="Oracle"),
        ]
    )
    fig.legend(handles=handles, loc="upper right", fontsize=8, frameon=True)
    fig.suptitle("Fine-tuning curves — val MAE over epochs", fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"  [plot] {save_path}")


def plot_summary_heatmap(
    mae_results:    Dict,
    strategies:     List[str],
    n_traj_values:  List,
    context_frac:   float,
    ood_mae:        float,
    oracle_mae:     float,
    save_path:      Path,
) -> None:
    """Heatmap of % gap closed (strategies × n_traj)."""
    gap = ood_mae - oracle_mae
    if gap <= 0:
        return

    mat = np.full((len(strategies), len(n_traj_values)), np.nan)
    for i, strat in enumerate(strategies):
        for j, n in enumerate(n_traj_values):
            mae = mae_results.get((strat, str(n)), {}).get(context_frac, float("nan"))
            if np.isfinite(mae):
                mat[i, j] = 100.0 * (ood_mae - mae) / gap

    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(n_traj_values)), 5))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(n_traj_values)))
    ax.set_xticklabels([str(n) for n in n_traj_values])
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels([STRATEGY_LABELS[s] for s in strategies], fontsize=9)
    ax.set_xlabel("N trajectories", fontsize=11)
    ax.set_title(
        f"% of OoD gap closed  |  ctx = {int(context_frac*100)}%\n"
        f"(gap = {gap:.2f} m;  OoD = {ood_mae:.2f} m;  oracle = {oracle_mae:.2f} m)",
        fontsize=11
    )
    for i in range(len(strategies)):
        for j in range(len(n_traj_values)):
            txt = f"{mat[i,j]:.0f}%" if np.isfinite(mat[i, j]) else "—"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="black" if 20 < mat[i, j] < 80 else "white"
                    if np.isfinite(mat[i, j]) else "gray")

    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="% gap closed")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"  [plot] {save_path}")


# =============================================================================
# Main orchestration
# =============================================================================

def aggregate_seed_results(
    seed_mae_list:  List[Dict[float, float]],
    seed_time_list: List[float],
) -> Tuple[Dict[float, float], Dict[float, float], float, float]:
    """
    Agrega resultados de N seeds independientes de fine-tuning.
    Devuelve (mae_mean, mae_std, time_mean, time_std) donde
    mae_mean y mae_std son dicts {context_frac: valor}.
    """
    fracs = list(seed_mae_list[0].keys())
    mae_mean = {
        f: float(np.mean([r[f] for r in seed_mae_list]))
        for f in fracs
    }
    mae_std = {
        f: float(np.std([r[f] for r in seed_mae_list], ddof=1))
        if len(seed_mae_list) > 1 else 0.0
        for f in fracs
    }
    time_mean = float(np.mean(seed_time_list))
    time_std  = float(np.std(seed_time_list, ddof=1)) if len(seed_time_list) > 1 else 0.0
    return mae_mean, mae_std, time_mean, time_std


def run_experiment(
    topology:      str,
    model_type:    str,
    args:          argparse.Namespace,
    context_fracs: List[float],
) -> Dict:
    """
    Full fine-tuning + evaluation pipeline for one (topology, model_type) pair.
    Returns a dict with mae_results, time_results, ood_by_frac, oracle_by_frac.
    """
    print(f"\n{'='*72}")
    print(f"  Topology : {topology}    Model : {model_type.upper()}")
    print(f"{'='*72}")

    out_base = Path(args.output_dir) / f"topology_{topology}" / f"model_{model_type}"
    out_base.mkdir(parents=True, exist_ok=True)

    # ── 1. Load lowvar data (target domain) ─────────────────────────────────
    print("\n[data] Loading low-variance data (target domain)…")
    lv_train, lv_val, lv_test = load_topology_data(args.lowvar_data, topology)
    eval_data = lv_test if lv_test is not None else lv_val
    print(
        f"  train={len(lv_train)}  val={len(lv_val)}  "
        f"{'test=' + str(len(lv_test)) if lv_test else 'test=N/A (using val)'}"
    )

    y_mean_lv, y_std_lv = compute_y_stats(lv_train)
    x_means_lv = compute_x_means(lv_train, NUM_TIME_POINTS, NUM_SENSORS)

    val_loader  = make_dataloader(lv_val,   args.batch_size)
    eval_loader = make_dataloader(eval_data, args.batch_size)

    # ── 2. Load source (highvar) model ───────────────────────────────────────
    print("\n[model] Loading HIGH-VAR source model…")
    hv_study = f"{model_type}_masked_highvar_{topology}_{args.study_version}"
    hv_dir   = resolve_optuna_best_model_dir(
        results_dir=args.optuna_root,
        study_name=hv_study,
        model_type=model_type,
        version=args.study_version,
    )
    hv_model, hv_hparams, hv_meta = load_optuna_best_model(
        best_model_dir  = hv_dir,
        topology        = topology,
        model_type      = model_type,
        num_sensors     = NUM_SENSORS,
        num_time_points = NUM_TIME_POINTS,
        output_dim      = OUTPUT_DIM,
        device          = args.device,
    )
    print(f"  Loaded from : {hv_dir}")
    print(f"  Optuna trial: {hv_meta['trial_number'] if hv_meta else '?'}")

    # ── 3. Load oracle (lowvar) model ────────────────────────────────────────
    print("\n[model] Loading LOW-VAR oracle model…")
    lv_study = f"{model_type}_masked_lowvar_{topology}_{args.study_version}"
    lv_dir   = resolve_optuna_best_model_dir(
        results_dir=args.optuna_root,
        study_name=lv_study,
        model_type=model_type,
        version=args.study_version,
    )
    lv_model, _, _ = load_optuna_best_model(
        best_model_dir  = lv_dir,
        topology        = topology,
        model_type      = model_type,
        num_sensors     = NUM_SENSORS,
        num_time_points = NUM_TIME_POINTS,
        output_dim      = OUTPUT_DIM,
        device          = args.device,
    )
    print(f"  Loaded from : {lv_dir}")

    # ── 4. Compute baselines ─────────────────────────────────────────────────
    print("\n[baseline] OoD baseline (highvar model → lowvar data, no FT)…")
    ood_by_frac = evaluate_model(
        hv_model, model_type, eval_loader,
        y_mean_lv, y_std_lv, x_means_lv,
        context_fracs, args.device, holdout_frac=args.holdout_frac,
    )
    print(f"  Mean MAE across fracs: {np.mean(list(ood_by_frac.values())):.4f} m")

    print("[baseline] Oracle (lowvar model → lowvar data)…")
    oracle_by_frac = evaluate_model(
        lv_model, model_type, eval_loader,
        y_mean_lv, y_std_lv, x_means_lv,
        context_fracs, args.device, holdout_frac=args.holdout_frac,
    )
    print(f"  Mean MAE across fracs: {np.mean(list(oracle_by_frac.values())):.4f} m")

    # persist baselines
    baselines = {
        "ood_baseline":  {str(f): v for f, v in ood_by_frac.items()},
        "oracle":        {str(f): v for f, v in oracle_by_frac.items()},
    }
    with open(out_base / "baselines.json", "w") as bf:
        json.dump(baselines, bf, indent=2)

    # ── 5. Fine-tuning loop ───────────────────────────────────────────────────
    strategies    = args.strategies
    n_traj_values = args.n_traj
    mae_results      : Dict = {}
    mae_std_results  : Dict = {}
    time_results     : Dict = {}
    summary_rows     : List = []

    for strategy in strategies:
        print(f"\n[ft] ── Strategy: {strategy} ──")
        for n in n_traj_values:
            n_str  = str(n)
            n_seeds = args.n_seeds

            # acumuladores por seed
            seed_mae_list:    List[Dict[float, float]] = []
            seed_time_list:   List[float]              = []
            seed_epochs:      List[int]                = []
            seed_n_params:    List[int]                = []
            seed_n_traj_used: List[int]                = []

            for seed in range(n_seeds):
                seed_dir = out_base / "checkpoints" / strategy / f"n_traj_{n_str}" / f"seed_{seed}"
                ckpt     = seed_dir / "finetuned_checkpoint.pth.tar"

                if args.skip_existing and ckpt.exists():
                    eval_json = seed_dir / "eval_results.json"
                    if eval_json.exists():
                        with open(eval_json) as ef:
                            saved = json.load(ef)
                        seed_mae_list.append({float(k): v for k, v in saved["mae_by_frac"].items()})
                        seed_time_list.append(saved["total_time_s"])
                        seed_epochs.append(saved.get("n_epochs", 0))
                        seed_n_params.append(saved.get("n_trainable_params", 0))
                        seed_n_traj_used.append(saved.get("n_traj_used", 0))
                        print(f"  [skip] strategy={strategy}  n_traj={n_str}  seed={seed}")
                    continue

                print(f"  n_traj={n_str}  seed={seed}")

                model_copy = copy.deepcopy(hv_model)
                model_copy.to(args.device)

                ft_meta = finetune_model(
                    model        = model_copy,
                    model_type   = model_type,
                    strategy     = strategy,
                    train_data   = lv_train,
                    val_data     = lv_val,
                    y_mean       = y_mean_lv,
                    y_std        = y_std_lv,
                    x_means_SP   = x_means_lv,
                    n_traj       = n,
                    lr           = args.lr,
                    epochs       = args.epochs,
                    patience     = args.patience,
                    batch_size   = args.batch_size,
                    device       = args.device,
                    save_dir     = seed_dir,
                    holdout_frac     = args.holdout_frac,
                    es_context_frac  = args.es_context_frac,
                    seed         = seed,
                )

                model_copy.eval()
                ft_eval = evaluate_model(
                    model_copy, model_type, eval_loader,
                    y_mean_lv, y_std_lv, x_means_lv,
                    context_fracs, args.device, holdout_frac=args.holdout_frac,
                )

                with open(seed_dir / "eval_results.json", "w") as ef:
                    json.dump({
                        "mae_by_frac":        {str(k): v for k, v in ft_eval.items()},
                        "total_time_s":       ft_meta["total_time_s"],
                        "n_epochs":           ft_meta["n_epochs"],
                        "n_trainable_params": ft_meta["n_trainable_params"],
                        "n_traj_used":        ft_meta["n_traj_used"],
                    }, ef, indent=2)

                seed_mae_list.append(ft_eval)
                seed_time_list.append(ft_meta["total_time_s"])
                seed_epochs.append(ft_meta["n_epochs"])
                seed_n_params.append(ft_meta["n_trainable_params"])
                seed_n_traj_used.append(ft_meta["n_traj_used"])

            if not seed_mae_list:
                continue

            # ── agregar seeds ──────────────────────────────────────────────────
            mae_mean, mae_std, time_mean, time_std = aggregate_seed_results(
                seed_mae_list, seed_time_list
            )
            mae_results[(strategy, n_str)]     = mae_mean
            mae_std_results[(strategy, n_str)] = mae_std
            time_results[(strategy, n_str)]    = time_mean

            mean_ood    = np.mean(list(ood_by_frac.values()))
            mean_oracle = np.mean(list(oracle_by_frac.values()))
            mean_ft     = np.mean(list(mae_mean.values()))
            mean_std    = float(np.mean(list(mae_std.values())))
            gap         = mean_ood - mean_oracle
            pct_closed  = 100.0 * (mean_ood - mean_ft) / max(gap, 1e-6)

            print(
                f"    → MAE={mean_ft:.4f}±{mean_std:.4f} m  |  "
                f"gap={pct_closed:.1f}%  |  "
                f"time={time_mean:.1f}±{time_std:.1f}s  |  "
                f"seeds={len(seed_mae_list)}"
            )

            row = {
                "topology":           topology,
                "model_type":         model_type,
                "strategy":           strategy,
                "n_traj":             n_str,
                "n_seeds":            len(seed_mae_list),
                "n_traj_used":        int(np.mean(seed_n_traj_used)),
                "n_trainable_params": seed_n_params[0] if seed_n_params else 0,
                "total_time_s_mean":  round(time_mean, 2),
                "total_time_s_std":   round(time_std,  2),
                "n_epochs_mean":      round(float(np.mean(seed_epochs)), 1),
                "n_epochs_std":       round(float(np.std(seed_epochs, ddof=1)) if len(seed_epochs) > 1 else 0.0, 1),
                "test_mean_mae":      round(mean_ft,  6),
                "test_std_mae":       round(mean_std, 6),
                "ood_baseline_mean":  round(mean_ood,    6),
                "oracle_mean":        round(mean_oracle,  6),
                "gap_closed_pct":     round(pct_closed,   2),
                **{
                    f"test_mae_ctx{int(f*100)}": round(mae_mean[f], 6)
                    for f in context_fracs
                },
                **{
                    f"test_std_ctx{int(f*100)}": round(mae_std[f], 6)
                    for f in context_fracs
                },
                **{
                    f"ood_mae_ctx{int(f*100)}":    round(ood_by_frac[f], 6)
                    for f in context_fracs
                },
                **{
                    f"oracle_mae_ctx{int(f*100)}": round(oracle_by_frac[f], 6)
                    for f in context_fracs
                },
            }
            summary_rows.append(row)

        # ── 6. Save summary CSV ───────────────────────────────────────────────────
    if summary_rows:
        csv_path = out_base / "finetune_summary.csv"
        fieldnames = list(summary_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\n[csv] Saved → {csv_path}")

    # ── 7. Plots ──────────────────────────────────────────────────────────────
    plots_dir = out_base / "plots"
    main_frac = args.context_frac

    for frac in context_fracs:
        pct = int(frac * 100)
        plot_mae_vs_ntraj(
            mae_results, mae_std_results, ood_by_frac, oracle_by_frac,
            strategies, n_traj_values, frac,
            plots_dir / f"mae_vs_ntraj_ctx{pct}.png",
        )
        plot_gap_closed(
            mae_results, mae_std_results, ood_by_frac, oracle_by_frac,
            strategies, n_traj_values, frac,
            plots_dir / f"gap_closed_ctx{pct}.png",
        )
        plot_pareto(
            time_results, mae_results,
            strategies, n_traj_values, frac,
            ood_by_frac[frac], oracle_by_frac[frac],
            plots_dir / f"pareto_ctx{pct}.png",
        )
        plot_summary_heatmap(
            mae_results, strategies, n_traj_values, frac,
            ood_by_frac[frac], oracle_by_frac[frac],
            plots_dir / f"heatmap_gap_closed_ctx{pct}.png",
        )

    plot_time_vs_ntraj(
        time_results, strategies, n_traj_values,
        plots_dir / "time_vs_ntraj.png",
    )

    plot_finetune_curves(
        log_base_dir = out_base / "checkpoints",
        strategies   = strategies,
        n_traj_values= n_traj_values,
        ood_mae      = ood_by_frac[main_frac],
        oracle_mae   = oracle_by_frac[main_frac],
        save_path    = plots_dir / "finetune_curves.png",
    )

    # context sweep at the largest data budget
    plot_context_sweep(
        mae_results, context_fracs,
        ood_by_frac, oracle_by_frac,
        strategies, n_traj_values[-1],
        plots_dir / f"context_sweep_ntraj_max.png",
    )
    # and at the smallest
    plot_context_sweep(
        mae_results, context_fracs,
        ood_by_frac, oracle_by_frac,
        strategies, n_traj_values[0],
        plots_dir / f"context_sweep_ntraj_min.png",
    )

    return {
        "mae_results":   mae_results,
        "time_results":  time_results,
        "ood_by_frac":   ood_by_frac,
        "oracle_by_frac": oracle_by_frac,
    }


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune OoD ANP/RANP models (highvar → lowvar adaptation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── paths ────────────────────────────────────────────────────────────────
    p.add_argument(
        "--optuna-root", required=True,
        help="Root of Optuna results (contains anp/ and ranp/ subdirectories)."
    )
    p.add_argument(
        "--lowvar-data", required=True,
        help="Root of low-variance processed data (topology_<x>/ folders inside)."
    )
    p.add_argument(
        "--output-dir", default="results/finetune_ood",
        help="Directory where all outputs (checkpoints, CSVs, plots) are saved."
    )

    # ── scope ────────────────────────────────────────────────────────────────
    p.add_argument(
        "--topologies", default="ellipsoidal",
        help="Comma-separated topologies to process: aligned,ellipsoidal,random."
    )
    p.add_argument(
        "--model-types", default="anp",
        help="Comma-separated model types: anp,ranp."
    )
    p.add_argument(
        "--study-version", default="v2",
        help="Optuna study version tag (e.g. v1, v2)."
    )

    # ── fine-tuning ──────────────────────────────────────────────────────────
    p.add_argument(
        "--strategies",
        default=",".join(DEFAULT_STRATEGIES),
        help=(
            "Comma-separated fine-tuning strategies. "
            "ANP/RANP: decoder_heads, decoder_full, decoder_det_last, decoder_det_full, decoder_lat_last. "
            "RANP-only: rnn_proj_only, rnn_proj_decoder, rnn_full_decoder."
        ),
    )
    p.add_argument(
        "--n-traj", default="10,20,50,100,200,300,all",
        help="Comma-separated data budgets (use 'all' for full training set)."
    )
    p.add_argument("--lr",       type=float, default=1e-4,
                   help="Fine-tuning learning rate.")
    p.add_argument("--epochs",   type=int,   default=1000,
                   help="Maximum fine-tuning epochs.")
    p.add_argument("--patience", type=int,   default=150,
                   help="Early-stopping patience (inverse holdout val MAE).")
    p.add_argument("--batch-size", type=int, default=8)

    # ── evaluation ───────────────────────────────────────────────────────────
    p.add_argument(
        "--holdout-frac", type=float, default=0.2,
        help="Fraction of trajectory reserved as target in inverse holdout."
    )
    p.add_argument(
        "--es-context-frac", type=float, default=0.4,
        help="Context fraction used during early-stopping validation."
    )
    p.add_argument(
        "--context-frac", type=float, default=0.3,
        help="Primary context fraction for summary plots."
    )
    p.add_argument(
        "--context-fracs", default="0.1,0.2,0.3,0.4,0.5,0.6",
        help="Comma-separated context fractions for the full evaluation sweep."
    )

    # ── misc ─────────────────────────────────────────────────────────────────
    p.add_argument("--device", default="cuda",
                   help="Torch device: cpu | cuda | cuda:0 …")
    p.add_argument(
        "--skip-existing", action="store_true",
        help="Skip (topology, model, strategy, n_traj) combos that already have a checkpoint."
    )
    p.add_argument(
        "--n-seeds", type=int, default=1,
        help="Número de seeds independientes de fine-tuning. "
             "Si >1, reporta media ± std sobre las runs."
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # parse list args
    args.topologies  = [t.strip() for t in args.topologies.split(",") if t.strip()]
    args.model_types = [m.strip() for m in args.model_types.split(",") if m.strip()]
    args.strategies  = [s.strip() for s in args.strategies.split(",") if s.strip()]

    raw_n = [x.strip() for x in args.n_traj.split(",") if x.strip()]
    args.n_traj = []
    for x in raw_n:
        args.n_traj.append("all" if x.lower() == "all" else int(x))

    context_fracs = [float(x) for x in args.context_fracs.split(",") if x.strip()]
    if args.context_frac not in context_fracs:
        context_fracs = sorted(set(context_fracs + [args.context_frac]))

    # validate strategies
    for s in args.strategies:
        if s not in STRATEGY_LABELS:
            raise ValueError(
                f"Unknown strategy '{s}'. Valid: {list(STRATEGY_LABELS.keys())}"
            )

    print("=" * 72)
    print("  finetune_ood.py — OoD domain adaptation experiment")
    print("=" * 72)
    print(f"  Topologies  : {args.topologies}")
    print(f"  Model types : {args.model_types}")
    print(f"  Strategies  : {args.strategies}")
    print(f"  Data budgets: {args.n_traj}")
    print(f"  Ctx fracs   : {context_fracs}")
    print(f"  N seeds     : {args.n_seeds}")
    print(f"  Device      : {args.device}")
    print(f"  Output dir  : {args.output_dir}")
    print()

    for topology in args.topologies:
        for model_type in args.model_types:
            try:
                run_experiment(
                    topology       = topology,
                    model_type     = model_type,
                    args           = args,
                    context_fracs  = context_fracs,
                )
            except FileNotFoundError as exc:
                print(f"\n[skip] {topology}/{model_type}: {exc}\n")

    print(f"\n{'='*72}")
    print(f"  Done.  Results in: {args.output_dir}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
