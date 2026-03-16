"""
eval_anp_vs_ranp_masked.py
==========================
Side-by-side evaluation of ANP (masked) vs RANP (masked) on the
low-variance / random-topology test set.

Tests produced
--------------
1. MAE per theta group         - bar chart + CSV table  (ANP / RANP-LSTM / RANP-GRU)
2. MAE vs context fraction     - line plot (context sweep from 5 % to 95 %)
3. NLL vs context fraction     - line plot
4. Predicted trajectories      - one figure per theta value, showing GT, ANP, RANP-LSTM, RANP-GRU with ±1σ shading
5. Summary statistics table    - printed and saved as a .txt

RANP-GRU is optional: pass --ranp-gru-ckpt to include it; omit to skip.

Usage
-----
# Run from the repo root:
cd /home/fernando/tesis/underwater-localization-topologies
python src/evaluation/eval_anp_vs_ranp_masked.py

# Usage example with all arguments specified:
python eval_anp_vs_ranp_masked.py \
    --data-dir /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_low_variance \
    --topology random \
    --anp-ckpt /home/fernando/tesis/underwater-localization-topologies/src/training/results/ANP_topologies_masked/lowvar/masked_dropbernoulli_p0.2_train_mean_first/topology_random/best_checkpoint.pth.tar \
    --ranp-ckpt /home/fernando/tesis/underwater-localization-topologies/src/training/results/RANP_topologies_masked/ranp_dropbernoulli_p0.2_train_mean_first_rnn-lstm_h128_l1/topology_random/best_checkpoint.pth.tar \
    --output-dir results/eval_anp_vs_ranp/lowvar_random \
    --ctx-fracs 0.05,0.10,0.20,0.30,0.50,0.70,0.90 \
    --fixed-ctx-frac 0.40 \
    --num-traj-plots 2 \
    --seed 18

python eval_anp_vs_ranp_masked.py \
    --data-dir /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_low_variance \
    --topology ellipsoidal \
    --anp-ckpt /home/fernando/tesis/underwater-localization-topologies/src/training/results/ANP_topologies_masked/lowvar/masked_dropbernoulli_p0.2_train_mean_first/topology_ellipsoidal/best_checkpoint.pth.tar \
    --ranp-ckpt /home/fernando/tesis/underwater-localization-topologies/src/training/results/RANP_topologies_masked/ranp_dropbernoulli_p0.2_train_mean_first_rnn-lstm_h128_l1/topology_ellipsoidal/best_checkpoint.pth.tar \
    --ranp-gru-ckpt /home/fernando/tesis/underwater-localization-topologies/src/training/results/RANP_topologies_masked/ranp_dropbernoulli_p0.2_train_mean_first_rnn-gru_h128_l1/topology_ellipsoidal/best_checkpoint.pth.tar \
    --output-dir results/eval_anp_vs_ranp/lowvar_ellipsoidal \
    --ctx-fracs 0.05,0.10,0.20,0.30,0.50,0.70,0.90 \
    --fixed-ctx-frac 0.40 \
    --num-traj-plots 2 \
    --seed 18
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── make sure project root is importable ──────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import src.models.anp as anp_module
import src.models.r_anp as ranp_module
from src.utils.nav_dataset import NavigationTrajectoryDataset

# ══════════════════════════════════════════════════════════════════════════════
# Default paths (override via CLI)
# ══════════════════════════════════════════════════════════════════════════════
_DEFAULT_DATA_DIR = str(
    _REPO_ROOT / "data/data/data_processed_topologies_low_variance"
)
_DEFAULT_ANP_CKPT = str(
    _REPO_ROOT
    / "src/training/results/ANP_topologies_masked/lowvar"
    / "masked_dropbernoulli_p0.2_train_mean_first/topology_random/best_checkpoint.pth.tar"
)
_DEFAULT_RANP_CKPT = str(
    _REPO_ROOT
    / "src/training/results/RANP_topologies_masked"
    / "ranp_dropbernoulli_p0.2_train_mean_first_rnn-lstm_h128_l1"
    / "topology_random/best_checkpoint.pth.tar"
)
_DEFAULT_RANP_GRU_CKPT = str(
    _REPO_ROOT
    / "src/training/results/RANP_topologies_masked"
    / "ranp_dropbernoulli_p0.2_train_mean_first_rnn-gru_h128_l1"
    / "topology_random/best_checkpoint.pth.tar"
)
_DEFAULT_OUTPUT_DIR = str(
    _REPO_ROOT / "results/eval_anp_vs_ranp/lowvar_random"
)

# ══════════════════════════════════════════════════════════════════════════════
# Model hyper-parameters (must match training)
# ══════════════════════════════════════════════════════════════════════════════
NUM_HIDDEN   = 128
NUM_SENSORS  = 10
NUM_TIME_PTS = 201
INPUT_DIM    = NUM_TIME_PTS * NUM_SENSORS + NUM_SENSORS   # 2020
OUTPUT_DIM   = 3

# ══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_data(data_dir: str, topology: str) -> Tuple[list, list, dict]:
    topo_dir = Path(data_dir) / f"topology_{topology}"
    with open(topo_dir / "train_data.pkl", "rb") as f:
        train_data = pickle.load(f)
    with open(topo_dir / "test_data.pkl", "rb") as f:
        test_data = pickle.load(f)
    with open(topo_dir / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return train_data, test_data, metadata


def compute_y_stats(train_data: list, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    Y = np.concatenate([y for _, y in train_data], axis=0)
    y_mean = torch.tensor(Y.mean(axis=0), dtype=torch.float32, device=device)
    y_std  = torch.tensor(Y.std(axis=0) + 1e-6, dtype=torch.float32, device=device)
    return y_mean, y_std


def compute_x_sensor_means(train_data: list, num_time_points: int, num_sensors: int
) -> np.ndarray:
    """Returns (S, P) mean array."""
    X = np.concatenate([x for x, _ in train_data], axis=0)
    X3 = X.reshape(X.shape[0], num_time_points, num_sensors)
    return X3.mean(axis=0).T  # (S, P)


def group_by_theta(test_data: list, test_thetas: list
) -> Dict[float, list]:
    groups: Dict[float, list] = {}
    for sample, theta in zip(test_data, test_thetas):
        groups.setdefault(theta, []).append(sample)
    return groups


# ══════════════════════════════════════════════════════════════════════════════
# Masking / augmentation (replica from training scripts)
# ══════════════════════════════════════════════════════════════════════════════

def augment_x_with_full_mask(x_batch: torch.Tensor,
    x_means_SP: torch.Tensor,
    num_time_points: int,
    num_sensors: int,
) -> torch.Tensor:
    """Append an all-ones mask (no sensor dropout) to x_batch.

    Args:
        x_batch   : (B, T, Dx)
        x_means_SP: (S, P) - not used since mask=all ones, included for API symmetry
    Returns:
        x_aug     : (B, T, Dx + S)
    """
    B, T, Dx = x_batch.shape
    mask_feat = torch.ones(B, T, num_sensors, device=x_batch.device, dtype=x_batch.dtype)
    return torch.cat([x_batch, mask_feat], dim=-1)   # (B, T, Dx+S)


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_anp(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    """Load the plain ANP (src.models.anp.LatentModel)."""
    model = anp_module.LatentModel(
        num_hidden=NUM_HIDDEN,
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[ANP]  loaded from {ckpt_path}")
    return model


def load_ranp(ckpt_path: str,
    device: torch.device,
    rnn_type: str = "lstm",
    rnn_layers: int = 1,
    rnn_dropout: float = 0.0,
) -> torch.nn.Module:
    """Load a recurrent ANP (src.models.r_anp.LatentModel) with the given RNN type."""
    model = ranp_module.LatentModel(
        num_hidden=NUM_HIDDEN,
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        rnn_type=rnn_type,
        rnn_layers=rnn_layers,
        rnn_dropout=rnn_dropout,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[RANP-{rnn_type.upper()}] loaded from {ckpt_path}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Prediction wrappers (handle the two different forward signatures)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_anp(model: torch.nn.Module,
    x_aug: torch.Tensor,       # (B, T, INPUT_DIM)
    ctx_idx: torch.Tensor,     # (Nc,)
    ctx_y: torch.Tensor,       # (B, Nc, output_dim)  - normalised
    tar_idx: torch.Tensor,     # (Nt,)
    tar_y: torch.Tensor,       # (B, Nt, output_dim)  - normalised
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (mean_norm, var_norm, loss, kl, nll)."""
    context_x = x_aug[:, ctx_idx, :]
    target_x  = x_aug[:, tar_idx, :]
    mean, var, loss, kl, nll = model(context_x, ctx_y, target_x, tar_y, beta=1.0)
    return mean, var, loss, kl, nll


@torch.no_grad()
def predict_ranp(model: torch.nn.Module,
    x_aug: torch.Tensor,
    ctx_idx: torch.Tensor,
    ctx_y: torch.Tensor,
    tar_idx: torch.Tensor,
    tar_y: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (mean_norm, var_norm, loss, kl, nll)."""
    mean, var, loss, kl, nll = model(
        x_seq=x_aug,
        context_indices=ctx_idx,
        context_y=ctx_y,
        target_indices=tar_idx,
        target_y=tar_y,
        beta=1.0,
    )
    return mean, var, loss, kl, nll


# ══════════════════════════════════════════════════════════════════════════════
# Context index sampling (deterministic "first N" mode, matching training)
# ══════════════════════════════════════════════════════════════════════════════

def ctx_indices_first(total: int, n: int, device: torch.device) -> torch.Tensor:
    return torch.arange(n, device=device)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1 - MAE per theta group at a fixed context fraction
# ══════════════════════════════════════════════════════════════════════════════

def eval_mae_per_theta(anp_model: torch.nn.Module,
    ranp_model: torch.nn.Module,
    theta_groups: Dict[float, list],
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    x_means_SP: torch.Tensor,
    ctx_frac: float,
    device: torch.device,
    batch_size: int = 8,
    ranp_gru_model: torch.nn.Module = None,
) -> Dict[str, Dict[float, float]]:
    """Compute non-context MAE for ANP, RANP-LSTM (and optionally RANP-GRU) per theta group.

    Returns nested dict: results[model_name][theta] = mae_value
    """
    results = {"ANP": {}, "RANP": {}}
    if ranp_gru_model is not None:
        results["RANP-GRU"] = {}

    for theta, group in sorted(theta_groups.items()):
        ds     = NavigationTrajectoryDataset(group)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        anp_maes, ranp_maes, gru_maes = [], [], []
        for x_raw, y_raw in loader:
            x_raw, y_raw = x_raw.to(device), y_raw.to(device)
            B, T, _ = x_raw.shape

            x_aug   = augment_x_with_full_mask(x_raw, x_means_SP, NUM_TIME_PTS, NUM_SENSORS)
            y_norm  = (y_raw - y_mean) / y_std

            n_ctx   = max(1, min(T - 1, int(round(ctx_frac * T))))
            ctx_idx = ctx_indices_first(T, n_ctx, device)
            tar_idx = torch.arange(n_ctx, T, device=device)  # strictly post-context

            ctx_y = y_norm[:, ctx_idx, :]
            tar_y = y_norm[:, tar_idx, :]

            # ANP
            mean_anp, _, _, _, _ = predict_anp(anp_model, x_aug, ctx_idx, ctx_y, tar_idx, tar_y)
            pred_anp = mean_anp * y_std + y_mean
            anp_maes.append(F.l1_loss(pred_anp, y_raw[:, tar_idx, :], reduction="mean").item())

            # RANP-LSTM
            mean_ranp, _, _, _, _ = predict_ranp(ranp_model, x_aug, ctx_idx, ctx_y, tar_idx, tar_y)
            pred_ranp = mean_ranp * y_std + y_mean
            ranp_maes.append(F.l1_loss(pred_ranp, y_raw[:, tar_idx, :], reduction="mean").item())

            # RANP-GRU (optional)
            if ranp_gru_model is not None:
                mean_gru, _, _, _, _ = predict_ranp(ranp_gru_model, x_aug, ctx_idx, ctx_y, tar_idx, tar_y)
                pred_gru = mean_gru * y_std + y_mean
                gru_maes.append(F.l1_loss(pred_gru, y_raw[:, tar_idx, :], reduction="mean").item())

        results["ANP"][theta]  = float(np.mean(anp_maes))
        results["RANP"][theta] = float(np.mean(ranp_maes))
        if ranp_gru_model is not None:
            results["RANP-GRU"][theta] = float(np.mean(gru_maes))

    return results


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2 - MAE / NLL / KL vs context fraction (sweep)
# ══════════════════════════════════════════════════════════════════════════════

def eval_vs_context_fraction(anp_model: torch.nn.Module,
    ranp_model: torch.nn.Module,
    test_data: list,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    x_means_SP: torch.Tensor,
    ctx_fracs: List[float],
    device: torch.device,
    batch_size: int = 8,
    holdout_frac: float = 0.20,
    ranp_gru_model: torch.nn.Module = None,
) -> Dict[str, Dict[str, List[float]]]:
    """Sweep context fractions and record MAE + NLL + KL for each model.

    MAE is computed on a **fixed held-out tail** of the trajectory (last `holdout_frac` fraction of time steps) so that the evaluated points are identical across all context sizes. 
    This avoids the artefact where MAE appears to grow with context because the non-context set shifts towards harder, later time steps.

    Context fracs that would overlap with the held-out tail are capped
    at (1 - holdout_frac - 1/T).

    Returns:
        out[model_name]["mae"] = [mae_at_frac0, mae_at_frac1, ...]
        out[model_name]["nll"] = [nll_at_frac0, ...]
        out[model_name]["kl"]  = [kl_at_frac0, ...]
    """
    out = {
        "ANP":  {"mae": [], "nll": [], "kl": []},
        "RANP": {"mae": [], "nll": [], "kl": []},
    }
    if ranp_gru_model is not None:
        out["RANP-GRU"] = {"mae": [], "nll": [], "kl": []}
    ds     = NavigationTrajectoryDataset(test_data)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    for frac in tqdm(ctx_fracs, desc="Context fraction sweep"):
        anp_maes, ranp_maes, gru_maes = [], [], []
        anp_nlls, ranp_nlls, gru_nlls = [], [], []
        anp_kls, ranp_kls, gru_kls = [], [], []

        for x_raw, y_raw in loader:
            x_raw, y_raw = x_raw.to(device), y_raw.to(device)
            B, T, _ = x_raw.shape

            # Fixed held-out evaluation set: always the last `holdout_frac` steps.
            n_holdout  = max(1, int(round(holdout_frac * T)))
            holdout_idx = torch.arange(T - n_holdout, T, device=device)  # fixed tail

            # Context: first N steps, capped so it never enters the held-out tail.
            max_ctx = T - n_holdout - 1
            n_ctx   = max(1, min(max_ctx, int(round(frac * T))))
            ctx_idx = ctx_indices_first(T, n_ctx, device)
            tar_idx = torch.arange(T, device=device)

            x_aug  = augment_x_with_full_mask(x_raw, x_means_SP, NUM_TIME_PTS, NUM_SENSORS)
            y_norm = (y_raw - y_mean) / y_std

            ctx_y = y_norm[:, ctx_idx, :]
            tar_y = y_norm[:, tar_idx, :]

            # ANP
            mean_a, var_a, _, kl_a, nll_a = predict_anp(anp_model, x_aug, ctx_idx, ctx_y, tar_idx, tar_y)
            pred_a = mean_a * y_std + y_mean
            anp_maes.append(F.l1_loss(pred_a[:, holdout_idx, :], y_raw[:, holdout_idx, :], reduction="mean").item())
            anp_nlls.append(nll_a.item())
            anp_kls.append(kl_a.item())

            # RANP
            mean_r, var_r, _, kl_r, nll_r = predict_ranp(ranp_model, x_aug, ctx_idx, ctx_y, tar_idx, tar_y)
            pred_r = mean_r * y_std + y_mean
            ranp_maes.append(F.l1_loss(pred_r[:, holdout_idx, :], y_raw[:, holdout_idx, :], reduction="mean").item())
            ranp_nlls.append(nll_r.item())
            ranp_kls.append(kl_r.item())

            # RANP-GRU (optional)
            if ranp_gru_model is not None:
                mean_g, _, _, kl_g, nll_g = predict_ranp(ranp_gru_model, x_aug, ctx_idx, ctx_y, tar_idx, tar_y)
                pred_g = mean_g * y_std + y_mean
                gru_maes.append(F.l1_loss(pred_g[:, holdout_idx, :], y_raw[:, holdout_idx, :], reduction="mean").item())
                gru_nlls.append(nll_g.item())
                gru_kls.append(kl_g.item())

        out["ANP"]["mae"].append(float(np.mean(anp_maes)))
        out["ANP"]["nll"].append(float(np.mean(anp_nlls)))
        out["ANP"]["kl"].append(float(np.mean(anp_kls)))
        out["RANP"]["mae"].append(float(np.mean(ranp_maes)))
        out["RANP"]["nll"].append(float(np.mean(ranp_nlls)))
        out["RANP"]["kl"].append(float(np.mean(ranp_kls)))
        if ranp_gru_model is not None:
            out["RANP-GRU"]["mae"].append(float(np.mean(gru_maes)))
            out["RANP-GRU"]["nll"].append(float(np.mean(gru_nlls)))
            out["RANP-GRU"]["kl"].append(float(np.mean(gru_kls)))

    return out


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3 - Trajectory plots (per theta group)
# ══════════════════════════════════════════════════════════════════════════════

def plot_trajectories_for_theta(anp_model: torch.nn.Module,
    ranp_model: torch.nn.Module,
    theta_groups: Dict[float, list],
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    x_means_SP: torch.Tensor,
    ctx_frac: float,
    output_dir: Path,
    num_traj_plots: int,
    device: torch.device,
    seed: int = 42,
    ranp_gru_model: torch.nn.Module = None,
) -> None:
    """For each theta value, plot up to num_traj_plots trajectories.

    Each figure has 2 rows (x and y — z is always 0 and is skipped) and shows:
    - Ground truth (black)
    - ANP prediction ± 1σ (blue)
    - RANP prediction ± 1σ (orange)
    - Context region shaded in grey
    """
    rng = np.random.default_rng(seed)
    # Only x and y; z is always 0 in this dataset.
    dim_indices = [0, 1]
    dim_labels  = ["x (m)", "y (m)"]
    colors = {"ANP": "#1f77b4", "RANP": "#ff7f0e", "RANP-GRU": "#2ca02c"}

    traj_dir = output_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    for theta, group in sorted(theta_groups.items()):
        n_plot = min(num_traj_plots, len(group))
        idxs   = rng.choice(len(group), size=n_plot, replace=False)

        for plot_idx, sample_i in enumerate(idxs):
            x_raw_np, y_raw_np = group[sample_i]   # (T, Dx), (T, 3)
            T = x_raw_np.shape[0]

            x_raw = torch.tensor(x_raw_np[None], dtype=torch.float32, device=device)   # (1,T,Dx)
            y_raw = torch.tensor(y_raw_np[None], dtype=torch.float32, device=device)   # (1,T,3)

            x_aug  = augment_x_with_full_mask(x_raw, x_means_SP, NUM_TIME_PTS, NUM_SENSORS)
            y_norm = (y_raw - y_mean) / y_std

            n_ctx   = max(1, min(T - 1, int(round(ctx_frac * T))))
            ctx_idx = ctx_indices_first(T, n_ctx, device)
            tar_idx = torch.arange(T, device=device)

            ctx_y = y_norm[:, ctx_idx, :]
            tar_y = y_norm[:, tar_idx, :]

            mean_a, var_a, _, _, _ = predict_anp(anp_model, x_aug, ctx_idx, ctx_y, tar_idx, tar_y)
            mean_r, var_r, _, _, _ = predict_ranp(ranp_model, x_aug, ctx_idx, ctx_y, tar_idx, tar_y)
            if ranp_gru_model is not None:
                mean_g, var_g, _, _, _ = predict_ranp(ranp_gru_model, x_aug, ctx_idx, ctx_y, tar_idx, tar_y)

            # denormalise
            pred_a  = (mean_a * y_std + y_mean).squeeze(0).cpu().numpy()   # (T,3)
            std_a   = (torch.sqrt(var_a) * y_std).squeeze(0).cpu().numpy()
            pred_r  = (mean_r * y_std + y_mean).squeeze(0).cpu().numpy()
            std_r   = (torch.sqrt(var_r) * y_std).squeeze(0).cpu().numpy()
            if ranp_gru_model is not None:
                pred_g = (mean_g * y_std + y_mean).squeeze(0).cpu().numpy()
                std_g  = (torch.sqrt(var_g) * y_std).squeeze(0).cpu().numpy()
            gt      = y_raw_np                                               # (T,3)
            t_axis  = np.arange(T)

            fig, axes = plt.subplots(len(dim_indices), 1, figsize=(12, 6), sharex=True)
            fig.suptitle(
                f"Trajectory — θ = {theta:.1f}  (sample {plot_idx+1}/{n_plot})\n"
                f"Context: {n_ctx}/{T} points (first {ctx_frac*100:.0f}%)",
                fontsize=11,
            )

            for row, (dim_idx, ax) in enumerate(zip(dim_indices, axes)):
                # context shading
                ax.axvspan(0, n_ctx - 1, alpha=0.08, color="grey", label="Context region" if row == 0 else "_")

                # ground truth
                ax.plot(t_axis, gt[:, dim_idx], "k-", lw=1.5, label="Ground truth" if row == 0 else "_")
                # context / target markers on GT
                _ctx_np = ctx_idx.cpu().numpy()
                _tar_np = np.arange(n_ctx, T)
                ax.scatter(_ctx_np, gt[_ctx_np, dim_idx], c="red", s=25, zorder=6,
                           label="Context pts" if row == 0 else "_")
                ax.scatter(_tar_np, gt[_tar_np, dim_idx], c="#90EE90", s=15, zorder=5,
                           edgecolors="none", label="Target pts" if row == 0 else "_")

                # ANP
                ax.plot(t_axis, pred_a[:, dim_idx], color=colors["ANP"], lw=1.3,
                        label="ANP" if row == 0 else "_")
                #ax.fill_between(
                #    t_axis,
                #    pred_a[:, dim_idx] - std_a[:, dim_idx],
                #    pred_a[:, dim_idx] + std_a[:, dim_idx],
                #    alpha=0.2, color=colors["ANP"],
                #)

                # RANP-LSTM
                ax.plot(t_axis, pred_r[:, dim_idx], color=colors["RANP"], lw=1.3,
                        label="RANP-LSTM" if row == 0 else "_")
                #ax.fill_between(
                #    t_axis,
                #    pred_r[:, dim_idx] - std_r[:, dim_idx],
                #    pred_r[:, dim_idx] + std_r[:, dim_idx],
                #    alpha=0.2, color=colors["RANP"],
                #)

                # RANP-GRU (optional)
                if ranp_gru_model is not None:
                    ax.plot(t_axis, pred_g[:, dim_idx], color=colors["RANP-GRU"], lw=1.3,
                            label="RANP-GRU" if row == 0 else "_")
                    #ax.fill_between(
                    #    t_axis,
                    #    pred_g[:, dim_idx] - std_g[:, dim_idx],
                    #    pred_g[:, dim_idx] + std_g[:, dim_idx],
                    #    alpha=0.2, color=colors["RANP-GRU"],
                    #)

                ax.set_ylabel(dim_labels[row])
                ax.grid(True, alpha=0.3)

            axes[0].legend(loc="upper right", fontsize=8)
            axes[-1].set_xlabel("Trajectory step")
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            fname = traj_dir / f"traj_theta{theta:.1f}_s{plot_idx+1}.png"
            plt.savefig(fname, dpi=150)
            plt.close(fig)

    print(f"  Trajectory plots saved to {traj_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4 - 2D trajectory path plot (XY plane)
# ══════════════════════════════════════════════════════════════════════════════

def plot_2d_paths_per_theta(anp_model: torch.nn.Module,
    ranp_model: torch.nn.Module,
    theta_groups: Dict[float, list],
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    x_means_SP: torch.Tensor,
    ctx_frac: float,
    output_dir: Path,
    num_traj_plots: int,
    device: torch.device,
    seed: int = 42,
    ranp_gru_model: torch.nn.Module = None,
) -> None:
    """XY-plane trajectory paths for both models vs ground truth."""
    rng = np.random.default_rng(seed)
    colors = {"ANP": "#1f77b4", "RANP": "#ff7f0e", "RANP-GRU": "#2ca02c"}

    path_dir = output_dir / "paths_xy"
    path_dir.mkdir(parents=True, exist_ok=True)

    for theta, group in sorted(theta_groups.items()):
        n_plot = min(num_traj_plots, len(group))
        idxs   = rng.choice(len(group), size=n_plot, replace=False)

        fig, axes = plt.subplots(1, n_plot, figsize=(6 * n_plot, 5))
        if n_plot == 1:
            axes = [axes]
        fig.suptitle(f"XY trajectory paths — θ = {theta:.1f}  (ctx {ctx_frac*100:.0f}%)", fontsize=11)

        for plot_idx, (sample_i, ax) in enumerate(zip(idxs, axes)):
            x_raw_np, y_raw_np = group[sample_i]
            T = x_raw_np.shape[0]

            x_raw = torch.tensor(x_raw_np[None], dtype=torch.float32, device=device)
            y_raw = torch.tensor(y_raw_np[None], dtype=torch.float32, device=device)

            x_aug  = augment_x_with_full_mask(x_raw, x_means_SP, NUM_TIME_PTS, NUM_SENSORS)
            y_norm = (y_raw - y_mean) / y_std

            n_ctx   = max(1, min(T - 1, int(round(ctx_frac * T))))
            ctx_idx = ctx_indices_first(T, n_ctx, device)
            tar_idx = torch.arange(T, device=device)
            ctx_y   = y_norm[:, ctx_idx, :]
            tar_y   = y_norm[:, tar_idx, :]

            mean_a, _, _, _, _ = predict_anp(anp_model, x_aug, ctx_idx, ctx_y, tar_idx, tar_y)
            mean_r, _, _, _, _ = predict_ranp(ranp_model, x_aug, ctx_idx, ctx_y, tar_idx, tar_y)
            if ranp_gru_model is not None:
                mean_g, _, _, _, _ = predict_ranp(ranp_gru_model, x_aug, ctx_idx, ctx_y, tar_idx, tar_y)

            pred_a = (mean_a * y_std + y_mean).squeeze(0).cpu().numpy()
            pred_r = (mean_r * y_std + y_mean).squeeze(0).cpu().numpy()
            if ranp_gru_model is not None:
                pred_g = (mean_g * y_std + y_mean).squeeze(0).cpu().numpy()
            gt     = y_raw_np

            ax.plot(gt[:, 0], gt[:, 1], "k-", lw=1.5, label="GT")
            ax.plot(pred_a[:, 0], pred_a[:, 1], color=colors["ANP"], lw=1.3, linestyle="--", label="ANP")
            ax.plot(pred_r[:, 0], pred_r[:, 1], color=colors["RANP"], lw=1.3, linestyle="--", label="RANP-LSTM")
            if ranp_gru_model is not None:
                ax.plot(pred_g[:, 0], pred_g[:, 1], color=colors["RANP-GRU"], lw=1.3, linestyle=":", label="RANP-GRU")
            # mark context / target on GT path
            ax.scatter(gt[:n_ctx, 0], gt[:n_ctx, 1], c="red", s=20, zorder=6, label="Context pts")
            ax.scatter(gt[n_ctx:, 0], gt[n_ctx:, 1], c="#90EE90", s=15, zorder=5, edgecolors="none", label="Target pts")

            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_title(f"Sample {plot_idx+1}")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.set_aspect("equal", adjustable="datalim")

        plt.tight_layout()
        fname = path_dir / f"path_xy_theta{theta:.1f}.png"
        plt.savefig(fname, dpi=150)
        plt.close(fig)

    print(f"  XY path plots saved to {path_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

def plot_mae_per_theta(results: Dict[str, Dict[float, float]],
    output_dir: Path,
    ctx_frac: float,
) -> None:
    thetas = sorted(results["ANP"].keys())
    x      = np.arange(len(thetas))

    model_names  = list(results.keys())
    model_colors = {"ANP": "#1f77b4", "RANP": "#ff7f0e", "RANP-GRU": "#2ca02c"}
    n_models = len(model_names)
    width    = 0.8 / n_models
    offsets  = np.linspace(-(n_models - 1) / 2 * width, (n_models - 1) / 2 * width, n_models)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, offset in zip(model_names, offsets):
        vals = [results[name][t] for t in thetas]
        ax.bar(x + offset, vals, width, label=name,
               color=model_colors.get(name, "grey"), alpha=0.85)

    ax.set_xlabel("θ (sensor orientation)")
    ax.set_ylabel("MAE (m)")
    ax.set_title(f"MAE per θ group — context {ctx_frac*100:.0f}%  (non-context points)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"θ={t:.1f}" for t in thetas])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "mae_per_theta.png", dpi=150)
    plt.close(fig)
    print(f"  mae_per_theta.png saved")


def plot_mae_vs_context(sweep: Dict[str, Dict[str, List[float]]],
    ctx_fracs: List[float],
    output_dir: Path,
    holdout_frac: float = 0.20,
) -> None:
    model_styles = {
        "ANP":      ("o-", "#1f77b4"),
        "RANP":     ("s-", "#ff7f0e"),
        "RANP-GRU": ("^-", "#2ca02c"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, metric, ylabel in zip(
        axes,
        ["mae", "nll", "kl"],
        ["MAE (m)", "NLL (nats)", "KL (nats)"],
    ):
        for name, data in sweep.items():
            marker, color = model_styles.get(name, ("x-", "grey"))
            ax.plot([f * 100 for f in ctx_fracs], data[metric],
                    marker, color=color, label=name, lw=2)
        ax.set_xlabel("Context fraction (%)")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{ylabel} vs context fraction")

    model_names_str = " vs ".join(sweep.keys())
    plt.suptitle(
        f"{model_names_str} — performance across context sizes\n"
        f"(MAE evaluated on fixed last {holdout_frac*100:.0f}% of trajectory)",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(output_dir / "mae_nll_kl_vs_context.png", dpi=150)
    plt.close(fig)
    print(f"  mae_nll_kl_vs_context.png saved")


# ══════════════════════════════════════════════════════════════════════════════
# Variance histograms
# ══════════════════════════════════════════════════════════════════════════════

def plot_variance_histograms(
    anp_model: torch.nn.Module,
    ranp_model: torch.nn.Module,
    test_data: list,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    x_means_SP: torch.Tensor,
    ctx_fracs: List[float],
    output_dir: Path,
    device: torch.device,
    batch_size: int = 8,
    holdout_frac: float = 0.20,
    ranp_gru_model: torch.nn.Module = None,
    bins: int = 60,
) -> None:
    """Plot variance distributions for all context levels in a single figure.

    The figure uses 2D histograms with context fraction on the x-axis and the
    predicted standard deviation σ (in metres) on the y-axis. A subplot is
    created for each model/output-dimension pair.
    """
    dim_labels = ["σ_x (m)", "σ_y (m)"]
    dim_indices = [0, 1]

    ds     = NavigationTrajectoryDataset(test_data)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model_names = ["ANP", "RANP"] + (["RANP-GRU"] if ranp_gru_model is not None else [])
    model_cmaps = {
        "ANP": "Blues",
        "RANP": "Oranges",
        "RANP-GRU": "Greens",
    }

    sigma_vals: Dict[str, List[List[float]]] = {
        name: [[], []] for name in model_names
    }
    ctx_vals: Dict[str, List[List[float]]] = {
        name: [[], []] for name in model_names
    }

    for frac in tqdm(ctx_fracs, desc="Variance histogram sweep"):
        for x_raw, y_raw in loader:
            x_raw, y_raw = x_raw.to(device), y_raw.to(device)
            B, T, _ = x_raw.shape

            n_holdout   = max(1, int(round(holdout_frac * T)))
            holdout_idx = torch.arange(T - n_holdout, T, device=device)

            max_ctx = T - n_holdout - 1
            n_ctx   = max(1, min(max_ctx, int(round(frac * T))))
            ctx_idx = ctx_indices_first(T, n_ctx, device)
            tar_idx = torch.arange(T, device=device)

            x_aug  = augment_x_with_full_mask(x_raw, x_means_SP, NUM_TIME_PTS, NUM_SENSORS)
            y_norm = (y_raw - y_mean) / y_std
            ctx_y  = y_norm[:, ctx_idx, :]
            tar_y  = y_norm[:, tar_idx, :]

            _, var_a, _, _, _ = predict_anp(anp_model, x_aug, ctx_idx, ctx_y, tar_idx, tar_y)
            _, var_r, _, _, _ = predict_ranp(ranp_model, x_aug, ctx_idx, ctx_y, tar_idx, tar_y)
            var_g = None
            if ranp_gru_model is not None:
                _, var_g, _, _, _ = predict_ranp(ranp_gru_model, x_aug, ctx_idx, ctx_y, tar_idx, tar_y)

            # σ in metres: sqrt(var_norm) * y_std  →  (B, holdout, 3)
            std_a = (torch.sqrt(var_a[:, holdout_idx, :]) * y_std).cpu().numpy()  # (B, Nh, 3)
            std_r = (torch.sqrt(var_r[:, holdout_idx, :]) * y_std).cpu().numpy()
            if var_g is not None:
                std_g = (torch.sqrt(var_g[:, holdout_idx, :]) * y_std).cpu().numpy()

            for di in dim_indices:
                anp_vals = std_a[:, :, di].ravel().tolist()
                ranp_vals = std_r[:, :, di].ravel().tolist()
                sigma_vals["ANP"][di].extend(anp_vals)
                sigma_vals["RANP"][di].extend(ranp_vals)
                ctx_vals["ANP"][di].extend([frac * 100] * len(anp_vals))
                ctx_vals["RANP"][di].extend([frac * 100] * len(ranp_vals))
                if var_g is not None:
                    gru_vals = std_g[:, :, di].ravel().tolist()
                    sigma_vals["RANP-GRU"][di].extend(gru_vals)
                    ctx_vals["RANP-GRU"][di].extend([frac * 100] * len(gru_vals))

    fig, axes = plt.subplots(
        len(model_names),
        len(dim_indices),
        figsize=(6 * len(dim_indices), 3.8 * len(model_names)),
        sharex=True,
        squeeze=False,
    )
    fig.suptitle(
        "Predicted σ distribution vs context fraction\n"
        f"(holdout last {holdout_frac*100:.0f}% of trajectory)",
        fontsize=11,
    )

    ctx_bins = np.array(sorted(ctx_fracs)) * 100.0
    if len(ctx_bins) == 1:
        ctx_edges = np.array([ctx_bins[0] - 0.5, ctx_bins[0] + 0.5])
    else:
        mids = 0.5 * (ctx_bins[:-1] + ctx_bins[1:])
        left = ctx_bins[0] - (mids[0] - ctx_bins[0])
        right = ctx_bins[-1] + (ctx_bins[-1] - mids[-1])
        ctx_edges = np.concatenate(([left], mids, [right]))

    for row, name in enumerate(model_names):
        for col, (di, dlabel) in enumerate(zip(dim_indices, dim_labels)):
            ax = axes[row, col]
            x = np.array(ctx_vals[name][di], dtype=float)
            y = np.array(sigma_vals[name][di], dtype=float)
            if y.size == 0:
                continue

            y_max = float(np.percentile(y, 99.5))
            y_edges = np.linspace(0.0, max(y_max, 1e-6), bins + 1)
            hist = ax.hist2d(x, y, bins=[ctx_edges, y_edges], cmap=model_cmaps.get(name, "Greys"))
            fig.colorbar(hist[3], ax=ax, label="Count")

            ax.set_title(f"{name} — {dlabel}")
            ax.set_xlabel("Context fraction (%)")
            ax.set_ylabel(dlabel)
            ax.set_xticks(ctx_bins)
            ax.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / "variance_histograms_vs_context.png", dpi=150)
    plt.close(fig)

    print(f"  Variance histogram plot saved to {output_dir / 'variance_histograms_vs_context.png'}")


# ══════════════════════════════════════════════════════════════════════════════
# Summary CSV / TXT
# ══════════════════════════════════════════════════════════════════════════════

def save_summary(mae_per_theta: Dict[str, Dict[float, float]],
    sweep: Dict[str, Dict[str, List[float]]],
    ctx_fracs: List[float],
    fixed_ctx_frac: float,
    output_dir: Path,
) -> None:
    model_names = list(mae_per_theta.keys())   # e.g. ["ANP", "RANP"] or [..., "RANP-GRU"]
    col_w = 12

    header_cols  = "  ".join(f"{n:>{col_w}}" for n in model_names)
    delta_header = "  ".join(
        f"{'Δ('+n+'-ANP)':>{col_w}}" for n in model_names if n != "ANP"
    )

    lines = [
        "=" * 70,
        "  Models: " + " / ".join(model_names) + " — Evaluation Summary",
        f"  Fixed context fraction: {fixed_ctx_frac*100:.0f}%",
        "=" * 70,
        "",
        "MAE per θ group (non-context points)",
        "-" * 60,
        f"{'θ':>6}  {header_cols}  {delta_header}",
    ]

    for theta in sorted(mae_per_theta["ANP"]):
        vals      = [mae_per_theta[n][theta] for n in model_names]
        val_str   = "  ".join(f"{v:>{col_w}.4f}" for v in vals)
        delta_str = "  ".join(
            f"{mae_per_theta[n][theta] - mae_per_theta['ANP'][theta]:>+{col_w}.4f}"
            for n in model_names if n != "ANP"
        )
        lines.append(f"{theta:>6.1f}  {val_str}  {delta_str}")

    avgs          = {n: float(np.mean(list(mae_per_theta[n].values()))) for n in model_names}
    avg_val_str   = "  ".join(f"{avgs[n]:>{col_w}.4f}" for n in model_names)
    avg_delta_str = "  ".join(
        f"{avgs[n] - avgs['ANP']:>+{col_w}.4f}" for n in model_names if n != "ANP"
    )
    lines += [
        "-" * 60,
        f"{'avg':>6}  {avg_val_str}  {avg_delta_str}",
        "",
        "MAE vs context fraction",
        "-" * 60,
        f"{'ctx%':>6}  " + "  ".join(f"{n+' MAE':>{col_w}}" for n in model_names),
    ]
    for i, frac in enumerate(ctx_fracs):
        row = "  ".join(f"{sweep[n]['mae'][i]:>{col_w}.4f}" for n in model_names)
        lines.append(f"{frac*100:>5.0f}%  {row}")

    lines += [
        "",
        "NLL vs context fraction",
        "-" * 60,
        f"{'ctx%':>6}  " + "  ".join(f"{n+' NLL':>{col_w}}" for n in model_names),
    ]
    for i, frac in enumerate(ctx_fracs):
        row = "  ".join(f"{sweep[n]['nll'][i]:>{col_w}.4f}" for n in model_names)
        lines.append(f"{frac*100:>5.0f}%  {row}")

    lines += [
        "",
        "KL vs context fraction",
        "-" * 60,
        f"{'ctx%':>6}  " + "  ".join(f"{n+' KL':>{col_w}}" for n in model_names),
    ]
    for i, frac in enumerate(ctx_fracs):
        row = "  ".join(f"{sweep[n]['kl'][i]:>{col_w}.4f}" for n in model_names)
        lines.append(f"{frac*100:>5.0f}%  {row}")

    summary_str = "\n".join(lines)
    print("\n" + summary_str)

    with open(output_dir / "summary.txt", "w") as f:
        f.write(summary_str + "\n")

    # CSV for easy import
    import csv
    non_anp = [n for n in model_names if n != "ANP"]
    with open(output_dir / "mae_per_theta.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["theta"] + model_names + ["Δ(" + n + "-ANP)" for n in non_anp])
        for theta in sorted(mae_per_theta["ANP"]):
            vals   = [mae_per_theta[n][theta] for n in model_names]
            deltas = [mae_per_theta[n][theta] - mae_per_theta["ANP"][theta] for n in non_anp]
            w.writerow([theta] + vals + deltas)

    with open(output_dir / "sweep_vs_context.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["ctx_frac"]
            + [n + "_mae" for n in model_names]
            + [n + "_nll" for n in model_names]
            + [n + "_kl" for n in model_names]
        )
        for i, frac in enumerate(ctx_fracs):
            maes = [sweep[n]["mae"][i] for n in model_names]
            nlls = [sweep[n]["nll"][i] for n in model_names]
            kls  = [sweep[n]["kl"][i] for n in model_names]
            w.writerow([frac] + maes + nlls + kls)

    print(f"  Summary written to {output_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ANP vs RANP (masked, low-var, random topology)")
    p.add_argument("--data-dir",       default=_DEFAULT_DATA_DIR)
    p.add_argument("--topology",       default="random")
    p.add_argument("--anp-ckpt",       default=_DEFAULT_ANP_CKPT)
    p.add_argument("--ranp-ckpt",      default=_DEFAULT_RANP_CKPT)
    p.add_argument("--ranp-gru-ckpt",  default=None, help="Checkpoint for RANP-GRU variant. Omit to skip GRU evaluation.")
    p.add_argument("--output-dir",     default=_DEFAULT_OUTPUT_DIR)
    p.add_argument("--ctx-fracs",      default="0.05,0.10,0.20,0.30,0.50,0.70,0.90", help="Comma-separated list of context fractions for the sweep")
    p.add_argument("--fixed-ctx-frac", type=float, default=0.30, help="Context fraction used for per-theta MAE and trajectory plots")
    p.add_argument("--num-traj-plots", type=int, default=2, help="Number of sample trajectories to plot per theta group")
    p.add_argument("--batch-size",     type=int, default=8)
    p.add_argument("--seed",           type=int, default=18)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ctx_fracs = [float(f) for f in args.ctx_fracs.split(",")]

    # ── Load data ────────────────────────────────────────────────────────────
    print("\n[1/7] Loading data …")
    train_data, test_data, metadata = load_data(args.data_dir, args.topology)
    theta_groups = group_by_theta(test_data, metadata["test_thetas"])
    print(f"  Train: {len(train_data)} | Test: {len(test_data)}")
    print(f"  θ values: {sorted(theta_groups)}")

    y_mean, y_std = compute_y_stats(train_data, device)
    x_means_np    = compute_x_sensor_means(train_data, NUM_TIME_PTS, NUM_SENSORS)
    x_means_SP    = torch.tensor(x_means_np, dtype=torch.float32, device=device)

    # ── Load models ──────────────────────────────────────────────────────────
    print("\n[2/7] Loading models …")
    anp_model  = load_anp(args.anp_ckpt,  device)
    ranp_model = load_ranp(args.ranp_ckpt, device, rnn_type="lstm")

    ranp_gru_model = None
    if args.ranp_gru_ckpt is not None:
        if os.path.exists(args.ranp_gru_ckpt):
            ranp_gru_model = load_ranp(args.ranp_gru_ckpt, device, rnn_type="gru")
        else:
            print(f"  [WARNING] RANP-GRU checkpoint not found, skipping: {args.ranp_gru_ckpt}")

    # ── Test 1: MAE per theta group ──────────────────────────────────────────
    print(f"\n[3/7] MAE per θ group  (ctx = {args.fixed_ctx_frac*100:.0f}%) …")
    mae_per_theta = eval_mae_per_theta(anp_model, ranp_model, theta_groups,
        y_mean, y_std, x_means_SP,
        ctx_frac=args.fixed_ctx_frac,
        device=device,
        batch_size=args.batch_size,
        ranp_gru_model=ranp_gru_model,
    )
    plot_mae_per_theta(mae_per_theta, output_dir, args.fixed_ctx_frac)

    # ── Test 2: MAE / NLL vs context fraction ────────────────────────────────
    print("\n[4/7] Context fraction sweep …")
    HOLDOUT_FRAC = 0.20
    sweep = eval_vs_context_fraction(anp_model, ranp_model, test_data,
        y_mean, y_std, x_means_SP,
        ctx_fracs=ctx_fracs,
        device=device,
        batch_size=args.batch_size,
        holdout_frac=HOLDOUT_FRAC,
        ranp_gru_model=ranp_gru_model,
    )
    plot_mae_vs_context(sweep, ctx_fracs, output_dir, holdout_frac=HOLDOUT_FRAC)

    # ── Test 3: Trajectory plots (per-dim, per theta) ────────────────────────
    print(f"\n[5/7] Trajectory plots (ctx = {args.fixed_ctx_frac*100:.0f}%) …")
    plot_trajectories_for_theta(anp_model, ranp_model, theta_groups,
        y_mean, y_std, x_means_SP,
        ctx_frac=args.fixed_ctx_frac,
        output_dir=output_dir,
        num_traj_plots=args.num_traj_plots,
        device=device,
        seed=args.seed,
        ranp_gru_model=ranp_gru_model,
    )

    # ── Test 4: XY path plots ────────────────────────────────────────────────
    plot_2d_paths_per_theta(anp_model, ranp_model, theta_groups,
        y_mean, y_std, x_means_SP,
        ctx_frac=args.fixed_ctx_frac,
        output_dir=output_dir,
        num_traj_plots=args.num_traj_plots,
        device=device,
        seed=args.seed,
        ranp_gru_model=ranp_gru_model,
    )

    # ── Test 5: Variance histograms ──────────────────────────────────────────
    print("\n[6/7] Variance histograms …")
    HOLDOUT_FRAC = 0.20
    plot_variance_histograms(anp_model, ranp_model, test_data,
        y_mean, y_std, x_means_SP,
        ctx_fracs=ctx_fracs,
        output_dir=output_dir,
        device=device,
        batch_size=args.batch_size,
        holdout_frac=HOLDOUT_FRAC,
        ranp_gru_model=ranp_gru_model,
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n[7/7] Saving summary …")
    save_summary(mae_per_theta, sweep, ctx_fracs, args.fixed_ctx_frac, output_dir)

    print(f"\nAll results written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
