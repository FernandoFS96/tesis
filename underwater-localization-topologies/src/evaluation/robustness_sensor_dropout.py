#!/usr/bin/env python3
"""
robustness_sensor_dropout.py
============================
Evaluates ANP and RANP robustness to sensor dropout on low-variance data.

Five experiments, all run per-topology (ellipsoidal, random, aligned):

  E1 – Oracle          : all 10 sensors active, MAE/NLL over context fraction sweep.
  E2 – Random k-dropout: sweep active sensors 1→10, averaged over random draws.
  E3 – Sensor importance: single-sensor ablation (k=1 active) and leave-one-out (k=9 active).
  E4 – Spatial dropout : topology-specific structured removal.
                         Ellipsoidal → arc removal (consecutive angular sector).
                         Aligned     → edge failure (from one side) and center-gap.
                         Random      → radius-based removal from array centroid.
  E5 – Fine-tuning     : adapt masked model to a fixed reduced config (worst sensors from E3b)
                         using decoder-only fine-tuning at multiple data budgets.

Models compared (4 total):
  anp_basic   – ANP,  p_drop=0, num_hidden=128
  anp_masked  – ANP,  Optuna best, loaded via load_optuna_best_model_from_study
  ranp_basic  – RANP, p_drop=0, num_hidden=128, rnn lstm/1-layer
  ranp_masked – RANP, Optuna best, loaded via load_optuna_best_model_from_study

Outputs (per topology):
  results/robustness_sensor_dropout/topology_<name>/
    E1_oracle/           MAE+NLL CSVs, context-sweep plots
    E2_random_dropout/   MAE vs k_active CSVs, degradation plots
    E3_sensor_importance/ per-sensor bar charts, CSVs
    E4_spatial_dropout/  structured removal plots, CSVs
    E5_finetune/         adaptation curves, summary CSVs
    summary/             cross-topology comparison plots

Usage
-----
    python robustness_sensor_dropout.py \\
        --data-dir      /path/to/data_processed_topologies_low_variance \\
        --optuna-root   /home/fernando/tesis/underwater-localization-topologies/src/training/results/optuna \\
        --basic-anp-dir /path/to/results/ANP_topologies_no_masked/masked_dropbernoulli_p0.0_train_mean_first \\
        --basic-ranp-dir /path/to/results/RANP_topologies_no_masked/lowvar/ranp_dropbernoulli_p0.0_train_mean_first_rnn-lstm_h64_l1 \\
        --topologies    ellipsoidal,random,aligned \\
        --study-version v2 \\
        --output-dir    results/robustness_sensor_dropout \\
        --device        cuda

    python robustness_sensor_dropout.py \
        --data-dir /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_low_variance \
        --optuna-root /home/fernando/tesis/underwater-localization-topologies/src/training/results/optuna\
        --basic-anp-dir /home/fernando/tesis/underwater-localization-topologies/src/training/results/ANP_topologies_no_masked/masked_dropbernoulli_p0.0_train_mean_first \
        --basic-ranp-dir /home/fernando/tesis/underwater-localization-topologies/src/training/results/RANP_topologies_no_masked/lowvar/ranp_dropbernoulli_p0.0_train_mean_first_rnn-lstm_h64_l1 \
        --topologies ellipsoidal,random,aligned \
        --study-version v2 \
        --output-dir results/robustness_sensor_dropout \
        --device cuda \
        --skip-e5           # remove this flag when you want fine-tuning

  # To include sensor positions for E4 (random topology radius-based removal):
    --sensor-positions-dir /path/to/raw/data   (contains channel_option_*/topology_*/channel_info/)

  # To skip the time-consuming fine-tuning experiment:
    --skip-e5
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import pickle
import random
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ── project root on sys.path ─────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import src.models.anp as anp_module
import src.models.r_anp as ranp_module
from src.utils.load_optuna_model import load_optuna_best_model_from_study

# ── import NavigationTrajectoryDataset ─────────────────────────────────────
from src.utils.nav_dataset import NavigationTrajectoryDataset

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════
NUM_SENSORS      = 10
NUM_TIME_POINTS  = 201
OUTPUT_DIM       = 3
INPUT_DIM        = NUM_TIME_POINTS * NUM_SENSORS + NUM_SENSORS   # 2020

# Basic (non-Optuna) model hyperparameters
BASIC_NUM_HIDDEN = 256
BASIC_RNN_TYPE   = "lstm"
BASIC_RNN_LAYERS = 1
BASIC_RNN_DROPOUT = 0.15

# Model display names and colors for plots
MODEL_NAMES   = ["anp_basic", "anp_masked", "ranp_basic", "ranp_masked"]
MODEL_LABELS  = {
    "anp_basic":   "ANP (basic)",
    "anp_masked":  "ANP (masked/Optuna)",
    "ranp_basic":  "RANP (basic)",
    "ranp_masked": "RANP (masked/Optuna)",
}
MODEL_COLORS  = {
    "anp_basic":   "#4878d0",
    "anp_masked":  "#1f4e96",
    "ranp_basic":  "#e88b2b",
    "ranp_masked": "#a84800",
}
MODEL_STYLES  = {
    "anp_basic":   "--",
    "anp_masked":  "-",
    "ranp_basic":  "--",
    "ranp_masked": "-",
}

HOLDOUT_FRAC  = 0.20   # fraction of trajectory reserved for target evaluation
FIXED_CTX_FRAC = 0.40  # default context fraction used in E2/E3/E4


# ══════════════════════════════════════════════════════════════════════════════
# Data utilities
# ══════════════════════════════════════════════════════════════════════════════

def _infer_num_hidden_from_checkpoint(ckpt_path: str, model_type: str) -> int:
    """
    Infer num_hidden from saved weight shapes to avoid hardcoding mismatches.
    ANP : reads latent_encoder.mu.linear_layer.weight  → shape [H, H]
    RANP: reads temporal_encoder.rnn.weight_ih_l0      → shape [4H, input] (LSTM)
           or  temporal_encoder.rnn.weight_ih_l0       → shape [3H, input] (GRU)
    """
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
    if model_type == "anp":
        return sd["latent_encoder.mu.linear_layer.weight"].shape[0]
    else:  # ranp
        w = sd["temporal_encoder.rnn.weight_ih_l0"]
        # LSTM: 4 gates → 4H; GRU: 3 gates → 3H
        divisor = 4 if w.shape[0] % 4 == 0 else 3
        return w.shape[0] // divisor
    
def load_topology_data(data_dir: str, topology: str):
    topo_dir = Path(data_dir) / f"topology_{topology}"
    with open(topo_dir / "train_data.pkl", "rb") as f:
        train_data = pickle.load(f)
    with open(topo_dir / "test_data.pkl", "rb") as f:
        test_data = pickle.load(f)
    with open(topo_dir / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return train_data, test_data, metadata


def compute_y_stats(train_data, device):
    Y = np.concatenate([y for _, y in train_data], axis=0)
    y_mean = torch.tensor(Y.mean(0), dtype=torch.float32, device=device)
    y_std  = torch.tensor(Y.std(0) + 1e-6, dtype=torch.float32, device=device)
    return y_mean, y_std


def compute_x_sensor_means(train_data) -> np.ndarray:
    """Returns (S, P) mean array matching training convention."""
    X  = np.concatenate([x for x, _ in train_data], axis=0)   # (N*T, Dx)
    X3 = X.reshape(X.shape[0], NUM_TIME_POINTS, NUM_SENSORS)   # (N*T, P, S)
    return X3.mean(axis=0).T                                    # (S, P)


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_basic_anp(ckpt_path: str, device: torch.device) -> nn.Module:
    num_hidden = _infer_num_hidden_from_checkpoint(ckpt_path, "anp")
    model = anp_module.LatentModel(
        num_hidden=num_hidden,
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  [ANP-basic]  num_hidden={num_hidden} ← {ckpt_path}")
    return model


def load_basic_ranp(ckpt_path: str, device: torch.device) -> nn.Module:
    num_hidden = _infer_num_hidden_from_checkpoint(ckpt_path, "ranp")
    model = ranp_module.LatentModel(
        num_hidden=num_hidden,
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        rnn_type=BASIC_RNN_TYPE,
        rnn_layers=BASIC_RNN_LAYERS,
        rnn_dropout=BASIC_RNN_DROPOUT,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  [RANP-basic] num_hidden={num_hidden} ← {ckpt_path}")
    return model


def load_optuna_model(optuna_root: str, model_type: str, topology: str,
                      study_version: str, device: torch.device):
    """Load best Optuna model for a given type and topology."""
    study_name = f"{model_type}_masked_lowvar_{topology}_{study_version}"
    model, hparams, meta = load_optuna_best_model_from_study(
        results_dir=optuna_root,
        study_name=study_name,
        topology=topology,
        model_type=model_type,
        num_sensors=NUM_SENSORS,
        num_time_points=NUM_TIME_POINTS,
        output_dim=OUTPUT_DIM,
        device=device,
    )
    model.eval()
    label = "ANP-masked" if model_type == "anp" else "RANP-masked"
    print(f"  [{label}]   ← {study_name}  (num_hidden={hparams.get('num_hidden')})")
    return model


def load_all_models(args, topology: str, device: torch.device) -> Dict[str, nn.Module]:
    """Load all four models for a given topology. Returns dict keyed by model name."""
    models = {}

    # Basic ANP
    anp_ckpt = Path(args.basic_anp_dir) / f"topology_{topology}" / "best_checkpoint.pth.tar"
    if anp_ckpt.exists():
        models["anp_basic"] = load_basic_anp(str(anp_ckpt), device)
    else:
        print(f"  [WARN] ANP-basic checkpoint not found: {anp_ckpt}")

    # Basic RANP
    ranp_ckpt = Path(args.basic_ranp_dir) / f"topology_{topology}" / "best_checkpoint.pth.tar"
    if ranp_ckpt.exists():
        models["ranp_basic"] = load_basic_ranp(str(ranp_ckpt), device)
    else:
        print(f"  [WARN] RANP-basic checkpoint not found: {ranp_ckpt}")

    # Optuna ANP
    try:
        models["anp_masked"] = load_optuna_model(
            args.optuna_root, "anp", topology, args.study_version, device)
    except (FileNotFoundError, Exception) as e:
        print(f"  [WARN] ANP-masked could not be loaded: {e}")

    # Optuna RANP
    try:
        models["ranp_masked"] = load_optuna_model(
            args.optuna_root, "ranp", topology, args.study_version, device)
    except (FileNotFoundError, Exception) as e:
        print(f"  [WARN] RANP-masked could not be loaded: {e}")

    return models


# ══════════════════════════════════════════════════════════════════════════════
# Masking & augmentation
# ══════════════════════════════════════════════════════════════════════════════

def apply_sensor_mask(
    x_batch: torch.Tensor,        # (B, T, Dx)  Dx = P*S
    active_sensors: List[int],    # indices in 0..S-1 that remain ON
    x_means_SP: torch.Tensor,     # (S, P)
) -> torch.Tensor:                # (B, T, Dx+S)
    """
    Zero-out masked sensors (replacing with train-mean) and append the binary
    mask as additional features — exactly as done in training.
    """
    B, T, Dx = x_batch.shape
    P, S = NUM_TIME_POINTS, NUM_SENSORS

    sensor_mask = torch.zeros(B, S, device=x_batch.device, dtype=x_batch.dtype)
    for s in active_sensors:
        sensor_mask[:, s] = 1.0

    x4 = x_batch.view(B, T, P, S)
    mu  = x_means_SP.T.view(1, 1, P, S).to(device=x_batch.device, dtype=x_batch.dtype)
    fill_val = mu.expand(B, T, P, S)

    m = sensor_mask.view(B, 1, 1, S)
    x4_masked = x4 * m + fill_val * (1.0 - m)
    x_masked  = x4_masked.reshape(B, T, Dx)

    mask_feat = sensor_mask.view(B, 1, S).expand(B, T, S)
    return torch.cat([x_masked, mask_feat], dim=-1)   # (B, T, Dx+S)


def augment_full_mask(x_batch: torch.Tensor) -> torch.Tensor:
    """All sensors active — just append an all-ones mask."""
    B, T, Dx = x_batch.shape
    mask_feat = torch.ones(B, T, NUM_SENSORS, device=x_batch.device, dtype=x_batch.dtype)
    return torch.cat([x_batch, mask_feat], dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
# Prediction wrappers (handle different forward signatures)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict(model: nn.Module, model_name: str,
            x_aug: torch.Tensor, ctx_idx: torch.Tensor,
            ctx_y: torch.Tensor, tar_idx: torch.Tensor,
            tar_y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unified predict returning (mean_norm, var_norm, kl, nll)."""
    is_ranp = "ranp" in model_name
    if is_ranp:
        mean, var, _, kl, nll = model(
            x_seq=x_aug,
            context_indices=ctx_idx,
            context_y=ctx_y,
            target_indices=tar_idx,
            target_y=tar_y,
            beta=1.0,
        )
    else:
        context_x = x_aug[:, ctx_idx, :]
        target_x  = x_aug[:, tar_idx, :]
        mean, var, _, kl, nll = model(context_x, ctx_y, target_x, tar_y, beta=1.0)
    return mean, var, kl, nll


# ══════════════════════════════════════════════════════════════════════════════
# Core evaluation: MAE + NLL for one mask configuration
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_mask(
    models: Dict[str, nn.Module],
    test_data: list,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    x_means_SP: torch.Tensor,
    active_sensors: Optional[List[int]],   # None = all sensors
    ctx_frac: float,
    device: torch.device,
    batch_size: int = 8,
    holdout_frac: float = HOLDOUT_FRAC,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all models with a given sensor mask and context fraction.

    Returns: results[model_name] = {"mae": float, "nll": float, "mean_std": float}

    The evaluation target is always the fixed held-out TAIL (last holdout_frac of T).
    Context comes from the first ctx_frac fraction of the non-holdout window.
    """
    ds     = NavigationTrajectoryDataset(test_data)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    accum = {name: {"mae": [], "nll": [], "mean_std": []} for name in models}

    for x_raw, y_raw in loader:
        x_raw, y_raw = x_raw.to(device), y_raw.to(device)
        B, T, _ = x_raw.shape

        # Fixed held-out target: last holdout_frac of T
        n_holdout   = max(1, int(round(holdout_frac * T)))
        holdout_start = T - n_holdout
        tar_idx = torch.arange(holdout_start, T, device=device)

        # Context: first ctx_frac of the non-holdout window
        max_ctx = holdout_start - 1
        n_ctx   = max(1, min(max_ctx, int(round(ctx_frac * max_ctx))))
        ctx_idx = torch.arange(n_ctx, device=device)

        # Augment with mask
        if active_sensors is None:
            x_aug = augment_full_mask(x_raw)
        else:
            x_aug = apply_sensor_mask(x_raw, active_sensors, x_means_SP)

        y_norm = (y_raw - y_mean) / y_std
        ctx_y  = y_norm[:, ctx_idx, :]
        tar_y  = y_norm[:, tar_idx, :]

        for name, model in models.items():
            mean_n, var_n, kl, nll = predict(model, name, x_aug, ctx_idx, ctx_y, tar_idx, tar_y)
            pred = mean_n * y_std + y_mean
            mae  = F.l1_loss(pred, y_raw[:, tar_idx, :], reduction="mean").item()
            std_pred = var_n.sqrt().mean().item()  # mean predicted std (calibration)
            accum[name]["mae"].append(mae)
            accum[name]["nll"].append(nll.item() if nll is not None else float("nan"))
            accum[name]["mean_std"].append(std_pred)

    return {
        name: {
            "mae":      float(np.mean(v["mae"])),
            "nll":      float(np.nanmean(v["nll"])),
            "mean_std": float(np.mean(v["mean_std"])),
        }
        for name, v in accum.items()
    }


# ══════════════════════════════════════════════════════════════════════════════
# E1 – Oracle evaluation (all sensors, context fraction sweep)
# ══════════════════════════════════════════════════════════════════════════════

def run_e1_oracle(models, test_data, y_mean, y_std, x_means_SP,
                  ctx_fracs, device, out_dir, batch_size=8):
    """All sensors active; sweep context fractions."""
    print("\n  [E1] Oracle evaluation …")
    out_dir = Path(out_dir) / "E1_oracle"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []  # for CSV
    results = {name: {"mae": [], "nll": [], "mean_std": []} for name in models}

    for frac in tqdm(ctx_fracs, desc="  E1 ctx sweep"):
        res = evaluate_mask(
            models, test_data, y_mean, y_std, x_means_SP,
            active_sensors=None, ctx_frac=frac,
            device=device, batch_size=batch_size,
        )
        for name in models:
            results[name]["mae"].append(res[name]["mae"])
            results[name]["nll"].append(res[name]["nll"])
            results[name]["mean_std"].append(res[name]["mean_std"])
        rows.append({"ctx_frac": frac, **{f"{n}_mae": res[n]["mae"] for n in models},
                     **{f"{n}_nll": res[n]["nll"] for n in models},
                     **{f"{n}_std": res[n]["mean_std"] for n in models}})

    # Save CSV
    _save_csv(rows, out_dir / "e1_oracle_results.csv")

    # Plot MAE
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name in models:
        axes[0].plot(ctx_fracs, results[name]["mae"],
                     label=MODEL_LABELS.get(name, name),
                     color=MODEL_COLORS.get(name, "gray"),
                     linestyle=MODEL_STYLES.get(name, "-"), linewidth=2)
        axes[1].plot(ctx_fracs, results[name]["mean_std"],
                     label=MODEL_LABELS.get(name, name),
                     color=MODEL_COLORS.get(name, "gray"),
                     linestyle=MODEL_STYLES.get(name, "-"), linewidth=2)

    axes[0].set_xlabel("Context fraction"); axes[0].set_ylabel("MAE (m)")
    axes[0].set_title("E1 – Oracle MAE vs Context Fraction (all sensors)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Context fraction"); axes[1].set_ylabel("Mean predicted σ (m)")
    axes[1].set_title("E1 – Predicted uncertainty vs Context Fraction")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "e1_oracle_ctx_sweep.png", dpi=150)
    plt.close(fig)

    print(f"  E1 done → {out_dir}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# E2 – Random k-dropout sweep
# ══════════════════════════════════════════════════════════════════════════════

def run_e2_random_dropout(models, test_data, y_mean, y_std, x_means_SP,
                          ctx_frac, device, out_dir, batch_size=8,
                          n_draws=30, seed=42):
    """Sweep number of active sensors 1→10, averaging over random draws."""
    print("\n  [E2] Random k-dropout sweep …")
    out_dir = Path(out_dir) / "E2_random_dropout"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    k_values = list(range(1, NUM_SENSORS + 1))  # 1 active … 10 active

    results  = {name: {"mae": [], "nll": [], "mean_std": [], "mae_std": []}
                for name in models}
    rows = []

    for k in tqdm(k_values, desc="  E2 k sweep"):
        # Generate all draws (random subsets of size k)
        draws = [sorted(rng.sample(range(NUM_SENSORS), k)) for _ in range(n_draws)]

        mae_per_draw  = {name: [] for name in models}
        nll_per_draw  = {name: [] for name in models}
        std_per_draw  = {name: [] for name in models}

        for active in draws:
            res = evaluate_mask(
                models, test_data, y_mean, y_std, x_means_SP,
                active_sensors=active, ctx_frac=ctx_frac,
                device=device, batch_size=batch_size,
            )
            for name in models:
                mae_per_draw[name].append(res[name]["mae"])
                nll_per_draw[name].append(res[name]["nll"])
                std_per_draw[name].append(res[name]["mean_std"])

        row = {"k_active": k}
        for name in models:
            m = float(np.mean(mae_per_draw[name]))
            s = float(np.std(mae_per_draw[name]))
            results[name]["mae"].append(m)
            results[name]["mae_std"].append(s)
            results[name]["nll"].append(float(np.nanmean(nll_per_draw[name])))
            results[name]["mean_std"].append(float(np.mean(std_per_draw[name])))
            row[f"{name}_mae"] = m
            row[f"{name}_mae_std"] = s
            row[f"{name}_nll"] = results[name]["nll"][-1]
        rows.append(row)

    _save_csv(rows, out_dir / "e2_random_dropout_results.csv")

    # Plot MAE ± std vs k_active
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in models:
        mae_arr = np.array(results[name]["mae"])
        std_arr = np.array(results[name]["mae_std"])
        ax.plot(k_values, mae_arr, label=MODEL_LABELS.get(name, name),
                color=MODEL_COLORS.get(name), linestyle=MODEL_STYLES.get(name), linewidth=2)
        ax.fill_between(k_values, mae_arr - std_arr, mae_arr + std_arr,
                        color=MODEL_COLORS.get(name), alpha=0.12)
    ax.axvline(x=NUM_SENSORS, color="black", linestyle=":", alpha=0.5, label="All sensors (oracle)")
    ax.invert_xaxis()
    ax.set_xlabel("Number of active sensors"); ax.set_ylabel("MAE (m)")
    ax.set_title(f"E2 – Degradation vs Active Sensors (ctx={ctx_frac:.0%}, n_draws={n_draws})")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "e2_random_dropout_mae.png", dpi=150)
    plt.close(fig)

    # Plot predicted uncertainty vs k_active
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in models:
        ax.plot(k_values, results[name]["mean_std"],
                label=MODEL_LABELS.get(name, name),
                color=MODEL_COLORS.get(name), linestyle=MODEL_STYLES.get(name), linewidth=2)
    ax.invert_xaxis()
    ax.set_xlabel("Number of active sensors"); ax.set_ylabel("Mean predicted σ (m)")
    ax.set_title("E2 – Calibration: predicted uncertainty vs Active Sensors")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "e2_random_dropout_uncertainty.png", dpi=150)
    plt.close(fig)

    print(f"  E2 done → {out_dir}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# E3 – Sensor importance: single-sensor ablation & leave-one-out
# ══════════════════════════════════════════════════════════════════════════════

def run_e3_sensor_importance(models, test_data, y_mean, y_std, x_means_SP,
                             ctx_frac, device, out_dir, batch_size=8):
    """
    E3a – Only sensor s active (k=1), for s in 0..9   → most informative sensor.
    E3b – All except sensor s active (k=9), for s in 0..9 → most critical sensor.
    Returns e3b_ranking[model_name] = sorted list of (sensor_idx, delta_mae)
    """
    print("\n  [E3] Sensor importance ablations …")
    out_dir = Path(out_dir) / "E3_sensor_importance"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Oracle MAE (all sensors)
    oracle = evaluate_mask(models, test_data, y_mean, y_std, x_means_SP,
                           active_sensors=None, ctx_frac=ctx_frac,
                           device=device, batch_size=batch_size)
    oracle_mae = {n: oracle[n]["mae"] for n in models}

    rows_single, rows_loo = [], []
    single_mae  = {name: [] for name in models}
    loo_mae     = {name: [] for name in models}
    loo_delta   = {name: [] for name in models}

    for s in tqdm(range(NUM_SENSORS), desc="  E3 sensors"):
        # E3a: only sensor s
        res_single = evaluate_mask(
            models, test_data, y_mean, y_std, x_means_SP,
            active_sensors=[s], ctx_frac=ctx_frac,
            device=device, batch_size=batch_size,
        )
        # E3b: all except sensor s
        loo_active = [i for i in range(NUM_SENSORS) if i != s]
        res_loo = evaluate_mask(
            models, test_data, y_mean, y_std, x_means_SP,
            active_sensors=loo_active, ctx_frac=ctx_frac,
            device=device, batch_size=batch_size,
        )

        row_s = {"sensor": s}
        row_l = {"sensor": s}
        for name in models:
            single_mae[name].append(res_single[name]["mae"])
            loo_mae[name].append(res_loo[name]["mae"])
            delta = res_loo[name]["mae"] - oracle_mae[name]
            loo_delta[name].append(delta)
            row_s[f"{name}_mae"] = res_single[name]["mae"]
            row_l[f"{name}_mae"] = res_loo[name]["mae"]
            row_l[f"{name}_delta_mae"] = delta
        rows_single.append(row_s)
        rows_loo.append(row_l)

    _save_csv(rows_single, out_dir / "e3a_single_sensor_mae.csv")
    _save_csv(rows_loo,    out_dir / "e3b_leave_one_out_mae.csv")

    sensor_ids = list(range(NUM_SENSORS))

    # ── Plot E3a: single-sensor MAE (bar chart) ───────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    x = np.arange(NUM_SENSORS)
    w = 0.2
    n_m = len(models)
    offsets = np.linspace(-(n_m-1)*w/2, (n_m-1)*w/2, n_m)
    for i, name in enumerate(models):
        axes[0].bar(x + offsets[i], single_mae[name], width=w,
                    label=MODEL_LABELS.get(name, name), color=MODEL_COLORS.get(name), alpha=0.8)
    for name in models:
        axes[0].axhline(oracle_mae[name], linestyle="--",
                        color=MODEL_COLORS.get(name), alpha=0.5, linewidth=1)
    axes[0].set_xticks(x); axes[0].set_xticklabels([f"S{s}" for s in sensor_ids])
    axes[0].set_xlabel("Active sensor"); axes[0].set_ylabel("MAE (m)")
    axes[0].set_title("E3a – Single-sensor MAE (lower = more informative)")
    axes[0].legend(fontsize=8); axes[0].grid(axis="y", alpha=0.3)

    # ── Plot E3b: leave-one-out ΔMAE (bar chart) ─────────────────────────────
    for i, name in enumerate(models):
        axes[1].bar(x + offsets[i], loo_delta[name], width=w,
                    label=MODEL_LABELS.get(name, name), color=MODEL_COLORS.get(name), alpha=0.8)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels([f"S{s}" for s in sensor_ids])
    axes[1].set_xlabel("Removed sensor"); axes[1].set_ylabel("ΔMAE vs oracle (m)")
    axes[1].set_title("E3b – Leave-one-out ΔMAE (higher = more critical)")
    axes[1].legend(fontsize=8); axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "e3_sensor_importance.png", dpi=150)
    plt.close(fig)

    # Build ranking: most critical sensors per model (from E3b)
    e3b_ranking = {}
    for name in models:
        ranked = sorted(enumerate(loo_delta[name]), key=lambda x: x[1], reverse=True)
        e3b_ranking[name] = ranked  # list of (sensor_idx, delta_mae)

    # Save ranking summary
    with open(out_dir / "e3b_sensor_ranking.txt", "w") as f:
        f.write("Most critical sensors (leave-one-out ΔMAE, descending):\n")
        f.write("="*60 + "\n")
        for name in models:
            f.write(f"\n{MODEL_LABELS.get(name, name)}:\n")
            for rank, (s, d) in enumerate(e3b_ranking[name], 1):
                f.write(f"  #{rank}  Sensor {s}  ΔMAE={d:+.4f} m\n")

    print(f"  E3 done → {out_dir}")
    return e3b_ranking, oracle_mae


# ══════════════════════════════════════════════════════════════════════════════
# E4 – Spatial dropout (topology-specific)
# ══════════════════════════════════════════════════════════════════════════════

def _get_spatial_removal_schedule(topology: str,
                                  sensor_positions: Optional[np.ndarray]) -> Dict[str, List[List[int]]]:
    """
    Returns removal schedules keyed by mode name.
    Each schedule is a list of 10 entries (one per removal step k=0..9),
    where each entry is the list of ACTIVE sensor indices.

    Ellipsoidal → "arc" mode (consecutive angular sector removed)
    Aligned     → "edge_left", "edge_right", "center_gap"
    Random      → "radius" (centroid-based, requires sensor_positions) + "random_fallback"
    """
    S = NUM_SENSORS
    schedules = {}

    if topology == "ellipsoidal":
        # Sensors are equally spaced at angles 0, 36, 72, ... 324 degrees (CCW).
        # Arc removal: remove k consecutive sensors starting from sensor 0.
        # We test all 10 starting positions and report mean + worst-case.
        arc_schedules = {}
        for start in range(S):
            per_k = []
            for k in range(S + 1):  # k=0 (all on) ... k=S (all off)
                removed = [(start + j) % S for j in range(k)]
                active  = [s for s in range(S) if s not in removed]
                per_k.append(active)
            arc_schedules[f"arc_start{start}"] = per_k
        schedules["arc"] = arc_schedules

    elif topology == "aligned":
        # Sensors 0..9 ordered left to right.
        # Edge-left: remove k sensors from the left (0, 1, ..., k-1)
        edge_left = [[s for s in range(S) if s >= k] for k in range(S + 1)]
        schedules["edge_left"] = edge_left

        # Edge-right: remove k sensors from the right (9, 8, ..., 9-k+1)
        edge_right = [[s for s in range(S) if s < S - k] for k in range(S + 1)]
        schedules["edge_right"] = edge_right

        # Center-gap: remove k sensors from the center outward
        center_gap = []
        for k in range(S + 1):
            half  = k // 2
            extra = k % 2
            lo = S // 2 - half
            hi = S // 2 + half + extra - 1
            removed = list(range(lo, hi + 1)) if k > 0 else []
            active  = [s for s in range(S) if s not in removed]
            center_gap.append(active)
        schedules["center_gap"] = center_gap

    elif topology == "random":
        if sensor_positions is not None:
            # sensor_positions: (3, S) → use xy plane
            xy = sensor_positions[:2, :].T   # (S, 2)
            centroid = xy.mean(axis=0)
            dists = np.linalg.norm(xy - centroid, axis=1)  # (S,)
            order_by_dist = np.argsort(dists)[::-1]  # farthest first → removed first
            radius_schedule = []
            for k in range(S + 1):
                removed = list(order_by_dist[:k])
                active  = [s for s in range(S) if s not in removed]
                radius_schedule.append(active)
            schedules["radius"] = radius_schedule
        else:
            print("  [E4] No sensor positions provided for random topology; "
                  "using random removal fallback (same as E2 but deterministic).")
            rng = random.Random(7)
            order = list(range(S)); rng.shuffle(order)
            random_schedule = []
            for k in range(S + 1):
                removed = order[:k]
                active  = [s for s in range(S) if s not in removed]
                random_schedule.append(active)
            schedules["radius_fallback"] = random_schedule

    return schedules


def run_e4_spatial_dropout(models, test_data, y_mean, y_std, x_means_SP,
                           topology, ctx_frac, device, out_dir,
                           sensor_positions=None, batch_size=8):
    """Topology-specific structured sensor removal."""
    print(f"\n  [E4] Spatial dropout ({topology}) …")
    out_dir = Path(out_dir) / "E4_spatial_dropout"
    out_dir.mkdir(parents=True, exist_ok=True)

    schedules = _get_spatial_removal_schedule(topology, sensor_positions)

    all_results = {}
    for mode_name, schedule in schedules.items():
        print(f"    Mode: {mode_name}")
        if isinstance(schedule, dict):
            # ellipsoidal: dict of arc_startX → per_k list
            # Aggregate across starting positions
            per_k_per_model = {name: {k: [] for k in range(NUM_SENSORS + 1)} for name in models}
            for start_key, per_k_list in tqdm(schedule.items(), desc=f"    {mode_name}"):
                for k, active in enumerate(per_k_list):
                    if len(active) == 0:
                        continue
                    res = evaluate_mask(models, test_data, y_mean, y_std, x_means_SP,
                                        active_sensors=active, ctx_frac=ctx_frac,
                                        device=device, batch_size=batch_size)
                    for name in models:
                        per_k_per_model[name][k].append(res[name]["mae"])

            rows = []
            mode_results = {name: {"k_removed": [], "mae_mean": [], "mae_worst": []} for name in models}
            for k in range(NUM_SENSORS + 1):
                row = {"k_removed": k}
                for name in models:
                    vals = per_k_per_model[name][k]
                    if vals:
                        m = float(np.mean(vals)); w = float(np.max(vals))
                    else:
                        m = w = float("nan")
                    mode_results[name]["k_removed"].append(k)
                    mode_results[name]["mae_mean"].append(m)
                    mode_results[name]["mae_worst"].append(w)
                    row[f"{name}_mae_mean"]  = m
                    row[f"{name}_mae_worst"] = w
                rows.append(row)
            _save_csv(rows, out_dir / f"e4_{mode_name}_results.csv")

        else:
            # Linear schedule (aligned / random)
            rows = []
            mode_results = {name: {"k_removed": [], "mae": []} for name in models}
            for k, active in enumerate(tqdm(schedule, desc=f"    {mode_name}")):
                if len(active) == 0:
                    break
                res = evaluate_mask(models, test_data, y_mean, y_std, x_means_SP,
                                    active_sensors=active, ctx_frac=ctx_frac,
                                    device=device, batch_size=batch_size)
                row = {"k_removed": k}
                for name in models:
                    mode_results[name]["k_removed"].append(k)
                    mode_results[name]["mae"].append(res[name]["mae"])
                    row[f"{name}_mae"] = res[name]["mae"]
                rows.append(row)
            _save_csv(rows, out_dir / f"e4_{mode_name}_results.csv")

        all_results[mode_name] = mode_results

        # Plot for this mode
        fig, ax = plt.subplots(figsize=(9, 5))
        for name in models:
            mr = mode_results[name]
            k_arr = mr["k_removed"]
            if "mae_mean" in mr:
                ax.plot(k_arr, mr["mae_mean"],
                        label=(MODEL_LABELS.get(name) or name) + " (mean)",
                        color=MODEL_COLORS.get(name), linestyle=MODEL_STYLES.get(name), linewidth=2)
                ax.plot(k_arr, mr["mae_worst"],
                        label=(MODEL_LABELS.get(name) or name) + " (worst)",
                        color=MODEL_COLORS.get(name), linestyle=":", linewidth=1.2)
            else:
                ax.plot(k_arr, mr["mae"],
                        label=MODEL_LABELS.get(name) or name,
                        color=MODEL_COLORS.get(name), linestyle=MODEL_STYLES.get(name), linewidth=2)

        mode_label = mode_name.replace("_", " ").title()
        ax.set_xlabel("Sensors removed"); ax.set_ylabel("MAE (m)")
        ax.set_title(f"E4 – Spatial Dropout [{topology}] – {mode_label}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_dir / f"e4_{mode_name}_mae.png", dpi=150)
        plt.close(fig)

    print(f"  E4 done → {out_dir}")
    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# E5 – Fine-tuning adaptation (decoder-only, fixed reduced sensor config)
# ══════════════════════════════════════════════════════════════════════════════

def _freeze_for_decoder_only(model: nn.Module, model_name: str):
    """Freeze all parameters except the decoder."""
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze decoder only
    for param in model.decoder.parameters(): #type: ignore
        param.requires_grad = True


def _finetune_model(model: nn.Module, model_name: str,
                    train_data: list, val_data: list,
                    active_sensors: List[int],
                    y_mean: torch.Tensor, y_std: torch.Tensor,
                    x_means_SP: torch.Tensor,
                    n_traj: int, lr: float, epochs: int, patience: int,
                    device: torch.device, batch_size: int = 8,
                    ctx_frac: float = FIXED_CTX_FRAC,
                    holdout_frac: float = HOLDOUT_FRAC) -> Tuple[nn.Module, List[float], List[float]]:
    """Fine-tune decoder of a deep-copied model on n_traj trajectories."""
    ft_model = copy.deepcopy(model)
    _freeze_for_decoder_only(ft_model, model_name)
    ft_model.train()

    # Sub-sample training trajectories
    rng = random.Random(99)
    subset = rng.sample(train_data, min(n_traj, len(train_data)))

    ds     = NavigationTrajectoryDataset(subset)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_ds = NavigationTrajectoryDataset(val_data)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    opt = torch.optim.Adam(
        [p for p in ft_model.parameters() if p.requires_grad], lr=lr)

    best_val_mae  = float("inf")
    best_state    = copy.deepcopy(ft_model.state_dict())
    no_improve    = 0
    train_log, val_log = [], []

    for epoch in range(epochs):
        ft_model.train()
        ep_loss = []
        for x_raw, y_raw in loader:
            x_raw, y_raw = x_raw.to(device), y_raw.to(device)
            B, T, _ = x_raw.shape
            x_aug = apply_sensor_mask(x_raw, active_sensors, x_means_SP)
            y_norm = (y_raw - y_mean) / y_std

            n_holdout = max(1, int(round(holdout_frac * T)))
            holdout_start = T - n_holdout
            max_ctx  = holdout_start - 1
            n_ctx    = max(1, min(max_ctx, int(round(ctx_frac * max_ctx))))
            ctx_idx  = torch.arange(n_ctx, device=device)
            tar_idx  = torch.arange(holdout_start, T, device=device)

            ctx_y = y_norm[:, ctx_idx, :]
            tar_y = y_norm[:, tar_idx, :]

            opt.zero_grad()
            mean_n, var_n, kl, nll = predict(ft_model, model_name, x_aug,
                                              ctx_idx, ctx_y, tar_idx, tar_y)
            # Simple NLL loss for fine-tuning (beta=0 – no KL term to avoid collapsing)
            loss = nll if nll is not None else F.l1_loss(mean_n, tar_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ft_model.parameters(), max_norm=1.0)
            opt.step()
            ep_loss.append(loss.item())
        train_log.append(float(np.mean(ep_loss)))

        # Validation
        ft_model.eval()
        val_maes = []
        with torch.no_grad():
            for x_raw, y_raw in val_loader:
                x_raw, y_raw = x_raw.to(device), y_raw.to(device)
                B, T, _ = x_raw.shape
                x_aug = apply_sensor_mask(x_raw, active_sensors, x_means_SP)
                y_norm = (y_raw - y_mean) / y_std
                n_holdout = max(1, int(round(holdout_frac * T)))
                holdout_start = T - n_holdout
                max_ctx = holdout_start - 1
                n_ctx   = max(1, min(max_ctx, int(round(ctx_frac * max_ctx))))
                ctx_idx = torch.arange(n_ctx, device=device)
                tar_idx = torch.arange(holdout_start, T, device=device)
                ctx_y   = y_norm[:, ctx_idx, :]
                tar_y   = y_norm[:, tar_idx, :]
                mean_n, _, _, _ = predict(ft_model, model_name, x_aug,
                                          ctx_idx, ctx_y, tar_idx, tar_y)
                pred = mean_n * y_std + y_mean
                val_maes.append(F.l1_loss(pred, y_raw[:, tar_idx, :]).item())

        val_mae = float(np.mean(val_maes))
        val_log.append(val_mae)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state   = copy.deepcopy(ft_model.state_dict())
            no_improve   = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    ft_model.load_state_dict(best_state)
    ft_model.eval()
    return ft_model, train_log, val_log


def run_e5_finetune(models, train_data, val_data, test_data,
                    y_mean, y_std, x_means_SP,
                    e3b_ranking,
                    ctx_frac, device, out_dir, batch_size=8,
                    n_traj_budgets=None, k_removed_sensors=3,
                    ft_lr=1e-4, ft_epochs=500, ft_patience=100):
    """
    Fine-tune masked models on reduced sensor configurations identified in E3b.
    Compares: no-FT baseline (basic + masked) vs. masked+fine-tuned.
    """
    print(f"\n  [E5] Fine-tuning adaptation (k_removed={k_removed_sensors}) …")
    if n_traj_budgets is None:
        n_traj_budgets = [10, 25, 50]
    out_dir = Path(out_dir) / "E5_finetune"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine the fixed damaged sensor configuration:
    # Use the k most critical sensors from the E3b ranking of anp_masked (or first available).
    ref_model = "anp_masked" if "anp_masked" in e3b_ranking else list(e3b_ranking.keys())[0]
    critical_sensors = [s for s, _ in e3b_ranking[ref_model][:k_removed_sensors]]
    active_config    = sorted([s for s in range(NUM_SENSORS) if s not in critical_sensors])

    print(f"    Damaged config (ref: {ref_model}): removed sensors {critical_sensors}")
    print(f"    Active sensors: {active_config}")

    with open(out_dir / "e5_sensor_config.txt", "w") as f:
        f.write(f"Removed sensors (most critical from E3b/{ref_model}): {critical_sensors}\n")
        f.write(f"Active sensors: {active_config}\n")
        f.write(f"k_removed: {k_removed_sensors}\n")

    # Baseline: evaluate all models on the damaged config without fine-tuning
    baseline = evaluate_mask(models, test_data, y_mean, y_std, x_means_SP,
                             active_sensors=active_config, ctx_frac=ctx_frac,
                             device=device, batch_size=batch_size)

    masked_models = {k: v for k, v in models.items() if "masked" in k}
    rows = []
    ft_results = {name: {"n_traj": [], "mae": []} for name in masked_models}

    for n_traj in n_traj_budgets:
        for name, model in masked_models.items():
            print(f"    Fine-tuning {name}, n_traj={n_traj} …")
            ft_model, train_log, val_log = _finetune_model(
                model, name, train_data, val_data,
                active_sensors=active_config,
                y_mean=y_mean, y_std=y_std, x_means_SP=x_means_SP,
                n_traj=n_traj, lr=ft_lr, epochs=ft_epochs, patience=ft_patience,
                device=device, batch_size=batch_size, ctx_frac=ctx_frac,
            )
            res = evaluate_mask({name: ft_model}, test_data, y_mean, y_std, x_means_SP,
                                active_sensors=active_config, ctx_frac=ctx_frac,
                                device=device, batch_size=batch_size)
            ft_mae = res[name]["mae"]
            ft_results[name]["n_traj"].append(n_traj)
            ft_results[name]["mae"].append(ft_mae)
            rows.append({
                "model": name, "n_traj": n_traj,
                "ft_mae": ft_mae,
                "baseline_mae": baseline[name]["mae"],
                "gap_closed_pct": (
                    100 * (baseline[name]["mae"] - ft_mae) / baseline[name]["mae"]
                    if baseline[name]["mae"] > 0 else 0.0
                ),
            })

            # Save fine-tuning curve
            curve_dir = out_dir / f"{name}_n{n_traj}"
            curve_dir.mkdir(exist_ok=True)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(val_log, label="Val MAE", color="darkorange")
            ax2 = ax.twinx()
            ax2.plot(train_log, color="steelblue", alpha=0.5, label="Train loss")
            ax.set_xlabel("Epoch"); ax.set_ylabel("Val MAE (m)")
            ax2.set_ylabel("Train loss")
            ax.set_title(f"E5 Fine-tuning: {name} n_traj={n_traj}")
            ax.legend(loc="upper right"); ax2.legend(loc="center right")
            plt.tight_layout()
            fig.savefig(curve_dir / "finetune_curve.png", dpi=150)
            plt.close(fig)

    _save_csv(rows, out_dir / "e5_finetune_summary.csv")

    # Summary plot: MAE vs n_traj for masked models + baselines
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in models:
        bl = baseline[name]["mae"]
        ax.axhline(bl, linestyle=MODEL_STYLES.get(name, "--"),
                   color=MODEL_COLORS.get(name), linewidth=1.5, alpha=0.6,
                   label=f"{MODEL_LABELS.get(name, name)} (no FT)")

    for name in masked_models:
        ax.plot(ft_results[name]["n_traj"], ft_results[name]["mae"],
                marker="o", linewidth=2, color=MODEL_COLORS.get(name),
                label=f"{MODEL_LABELS.get(name, name)} (fine-tuned)")

    ax.set_xlabel("Fine-tuning trajectories (n_traj)")
    ax.set_ylabel("MAE (m)")
    ax.set_title(f"E5 – Fine-tuning adaptation (removed: sensors {critical_sensors})")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "e5_mae_vs_ntraj.png", dpi=150)
    plt.close(fig)

    print(f"  E5 done → {out_dir}")
    return ft_results


# ══════════════════════════════════════════════════════════════════════════════
# Cross-topology summary plot
# ══════════════════════════════════════════════════════════════════════════════

def plot_cross_topology_summary(all_topology_results: Dict, out_dir: Path):
    """Compare E2 degradation curves across topologies for each model."""
    out_dir.mkdir(parents=True, exist_ok=True)
    topologies = list(all_topology_results.keys())
    if not topologies:
        return

    # One figure per model: E2 MAE vs k_active, one line per topology
    model_names_available = list(all_topology_results[topologies[0]].get("E2", {}).keys())
    colors_topo = {"ellipsoidal": "#2ca02c", "random": "#9467bd", "aligned": "#d62728"}

    for model_name in model_names_available:
        fig, ax = plt.subplots(figsize=(9, 5))
        for topo in topologies:
            e2 = all_topology_results[topo].get("E2", {})
            if model_name not in e2:
                continue
            mae = e2[model_name]["mae"]
            k_values = list(range(1, len(mae) + 1))
            ax.plot(k_values, mae, label=topo.capitalize(),
                    color=colors_topo.get(topo, "gray"), linewidth=2)
        ax.invert_xaxis()
        ax.set_xlabel("Active sensors"); ax.set_ylabel("MAE (m)")
        ax.set_title(f"E2 Cross-topology comparison – {MODEL_LABELS.get(model_name, model_name)}")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_dir / f"cross_topology_e2_{model_name}.png", dpi=150)
        plt.close(fig)

    print(f"  Cross-topology summary → {out_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# Utility: save CSV
# ══════════════════════════════════════════════════════════════════════════════

def _save_csv(rows: List[dict], path: Path):
    if not rows:
        return
    path = Path(path)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Load sensor positions for E4 (optional)
# ══════════════════════════════════════════════════════════════════════════════

def try_load_sensor_positions(sensor_positions_dir: Optional[str],
                               topology: str) -> Optional[np.ndarray]:
    """Load sensor positions (3, S) for E4 if raw data dir is provided."""
    if sensor_positions_dir is None:
        return None
    raw_dir = Path(sensor_positions_dir)
    # Find first available channel option
    candidates = sorted(raw_dir.glob(f"channel_option_*/{topology}/channel_info/sensor_positions_*.npy"))
    if not candidates:
        print(f"  [E4] No sensor position files found in {sensor_positions_dir}")
        return None
    pos = np.load(candidates[0])
    print(f"  [E4] Sensor positions loaded from {candidates[0]}")
    return pos  # (3, S)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Sensor dropout robustness evaluation for ANP/RANP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Paths
    p.add_argument("--data-dir",     required=True,
                   help="Root of processed low-variance data (topology_*/ folders)")
    p.add_argument("--optuna-root",  required=True,
                   help="Root of Optuna results (anp/ and ranp/ subdirs)")
    p.add_argument("--basic-anp-dir", required=True,
                   help="Root dir for basic ANP models (topology_*/ inside)")
    p.add_argument("--basic-ranp-dir", required=True,
                   help="Root dir for basic RANP models (topology_*/ inside)")
    p.add_argument("--output-dir",   default="results/robustness_sensor_dropout",
                   help="Output directory for all results")
    p.add_argument("--sensor-positions-dir", default=None,
                   help="Optional: raw data dir for loading sensor positions (E4 random topology)")

    # Scope
    p.add_argument("--topologies",   default="ellipsoidal,random,aligned",
                   help="Comma-separated topologies to evaluate")
    p.add_argument("--study-version", default="v2",
                   help="Optuna study version tag (e.g. v2)")

    # Evaluation
    p.add_argument("--ctx-frac", type=float, default=0.40, help="Fixed context fraction used in E2, E3, E4, E5")
    p.add_argument("--ctx-fracs", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8", help="Context fractions for E1 oracle sweep")
    p.add_argument("--holdout-frac", type=float, default=0.20, help="Tail fraction used as evaluation target")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-draws", type=int, default=30, help="Random draws per k-value in E2")
    p.add_argument("--seed", type=int, default=18)
    p.add_argument("--device", default="cuda")

    # E5 fine-tuning
    p.add_argument("--skip-e5", action="store_true", help="Skip the fine-tuning experiment (E5)")
    p.add_argument("--e5-n-traj", default="20,50,100", help="Comma-separated data budgets for E5 fine-tuning")
    p.add_argument("--e5-k-removed", type=int, default=3, help="Number of most critical sensors to permanently remove in E5")
    p.add_argument("--e5-lr",        type=float, default=5e-4)
    p.add_argument("--e5-epochs",    type=int, default=1000)
    p.add_argument("--e5-patience",  type=int, default=100)

    return p.parse_args()

def _already_done(sentinel_path: Path) -> bool:
    """Return True if the sentinel output file already exists (experiment was completed)."""
    if sentinel_path.exists():
        print(f"  [skip] Already done — found {sentinel_path.relative_to(sentinel_path.parents[3])}")
        return True
    return False

def _load_e3b_ranking(ranking_path: Path) -> dict:
    """
    Re-parse the e3b_sensor_ranking.txt file written by run_e3_sensor_importance
    so that E5 can consume it even when E3 is skipped.
    Returns dict: model_name → list of (sensor_idx, delta_mae)
    """
    import re
    ranking = {}
    current_model = None
    with open(ranking_path, "r") as f:
        for line in f:
            line = line.rstrip()
            # Detect model header lines (not a rank line, not a separator)
            if line and not line.startswith("Most") and not line.startswith("=") \
                    and not re.match(r"\s*#\d+", line):
                # Strip trailing colon → model label
                candidate = line.strip().rstrip(":")
                if candidate:
                    current_model = candidate
                    ranking[current_model] = []
            elif current_model and re.match(r"\s*#\d+", line):
                m = re.search(r"Sensor\s+(\d+)\s+ΔMAE=([+-]?\d+\.\d+)", line)
                if m:
                    ranking[current_model].append(
                        (int(m.group(1)), float(m.group(2)))
                    )
    return ranking

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Device: {device}")

    topologies   = [t.strip() for t in args.topologies.split(",") if t.strip()]
    ctx_fracs    = [float(f) for f in args.ctx_fracs.split(",") if f.strip()]
    e5_n_traj    = [int(n) for n in args.e5_n_traj.split(",") if n.strip()]
    out_root     = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_topology_results = {}

    for topology in topologies:
        print(f"\n{'='*72}")
        print(f"  Topology: {topology.upper()}")
        print(f"{'='*72}")

        topo_out = out_root / f"topology_{topology}"
        topo_out.mkdir(parents=True, exist_ok=True)

        # ── Load data ─────────────────────────────────────────────────────────
        print("\n[data] Loading …")
        train_data, test_data, metadata = load_topology_data(args.data_dir, topology)
        y_mean, y_std   = compute_y_stats(train_data, device)
        x_means_np      = compute_x_sensor_means(train_data)
        x_means_SP      = torch.tensor(x_means_np, dtype=torch.float32, device=device)
        val_data        = None  # load for E5 only (lazy)
        print(f"  Train: {len(train_data)} | Test: {len(test_data)}")

        # ── Load models ───────────────────────────────────────────────────────
        print("\n[models] Loading …")
        models = load_all_models(args, topology, device)
        if not models:
            print(f"  No models loaded for {topology}, skipping.")
            continue
        print(f"  Loaded: {list(models.keys())}")

        topo_results = {}

        # ─────────────────────────────────────────────────────────────────────
        # E1 – Oracle
        # ─────────────────────────────────────────────────────────────────────
        e1_sentinel = topo_out / "E1_oracle" / "e1_oracle_results.csv"
        if _already_done(e1_sentinel):
            # Reload results from CSV so downstream experiments still have data
            e1_res = None  # E1 results not needed by later experiments
        else:
            e1_res = run_e1_oracle(
                models, test_data, y_mean, y_std, x_means_SP,
                ctx_fracs=ctx_fracs, device=device,
                out_dir=topo_out, batch_size=args.batch_size,
            )
        topo_results["E1"] = e1_res

        # ─────────────────────────────────────────────────────────────────────
        # E2 – Random k-dropout
        # ─────────────────────────────────────────────────────────────────────
        e2_sentinel = topo_out / "E2_random_dropout" / "e2_random_dropout_results.csv"
        if _already_done(e2_sentinel):
            e2_res = None
        else:
            e2_res = run_e2_random_dropout(
                models, test_data, y_mean, y_std, x_means_SP,
                ctx_frac=args.ctx_frac, device=device,
                out_dir=topo_out, batch_size=args.batch_size,
                n_draws=args.n_draws, seed=args.seed,
            )
        topo_results["E2"] = e2_res

        # ─────────────────────────────────────────────────────────────────────
        # E3 – Sensor importance
        # ─────────────────────────────────────────────────────────────────────
        e3_sentinel = topo_out / "E3_sensor_importance" / "e3b_sensor_ranking.txt"
        if _already_done(e3_sentinel):
            # Reload ranking from the saved text file for E5 to consume
            e3b_ranking = _load_e3b_ranking(e3_sentinel)
            oracle_mae  = {}
        else:
            e3b_ranking, oracle_mae = run_e3_sensor_importance(
                models, test_data, y_mean, y_std, x_means_SP,
                ctx_frac=args.ctx_frac, device=device,
                out_dir=topo_out, batch_size=args.batch_size,
            )
        topo_results["E3_ranking"] = e3b_ranking
        topo_results["oracle_mae"] = oracle_mae

        # ─────────────────────────────────────────────────────────────────────
        # E4 – Spatial dropout
        # ─────────────────────────────────────────────────────────────────────
        e4_dir      = topo_out / "E4_spatial_dropout"
        e4_sentinel = next(e4_dir.glob("e4_*.csv"), None) if e4_dir.exists() else None
        if e4_sentinel is not None:
            print(f"  [skip] Already done — found {e4_sentinel.name} in E4_spatial_dropout/")
            e4_res = None
        else:
            sensor_positions = try_load_sensor_positions(args.sensor_positions_dir, topology)
            e4_res = run_e4_spatial_dropout(
                models, test_data, y_mean, y_std, x_means_SP,
                topology=topology, ctx_frac=args.ctx_frac,
                device=device, out_dir=topo_out,
                sensor_positions=sensor_positions, batch_size=args.batch_size,
            )
        topo_results["E4"] = e4_res

        # ─────────────────────────────────────────────────────────────────────
        # E5 – Fine-tuning adaptation
        # ─────────────────────────────────────────────────────────────────────
        if not args.skip_e5:
            e5_sentinel = topo_out / "E5_finetune" / "e5_finetune_summary.csv"
            if _already_done(e5_sentinel):
                e5_res = None
            else:
                topo_dir = Path(args.data_dir) / f"topology_{topology}"
                with open(topo_dir / "val_data.pkl", "rb") as f:
                    val_data = pickle.load(f)
                e5_res = run_e5_finetune(
                    models, train_data, val_data, test_data,
                    y_mean, y_std, x_means_SP,
                    e3b_ranking=e3b_ranking,
                    ctx_frac=args.ctx_frac, device=device,
                    out_dir=topo_out, batch_size=args.batch_size,
                    n_traj_budgets=e5_n_traj,
                    k_removed_sensors=args.e5_k_removed,
                    ft_lr=args.e5_lr, ft_epochs=args.e5_epochs,
                    ft_patience=args.e5_patience,
                )
            topo_results["E5"] = e5_res

    # ── Cross-topology summary ────────────────────────────────────────────────
    print("\n[summary] Cross-topology plots …")
    plot_cross_topology_summary(all_topology_results, out_root / "summary")

    print(f"\n{'='*72}")
    print(f"  All done. Results in: {out_root.resolve()}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
