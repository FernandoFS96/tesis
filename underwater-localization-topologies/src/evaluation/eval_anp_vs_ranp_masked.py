"""
eval_anp_vs_ranp_masked.py
==========================
Side-by-side evaluation of ANP (masked) vs RANP (masked) on the
low-variance / random-topology test set.

Tests produced
--------------
1. MAE per theta group         - bar chart + CSV table
2. MAE vs context fraction     - line plot (context sweep from 5 % to 95 %)
3. NLL vs context fraction     - line plot
4. Predicted trajectories      - one figure per theta value, showing GT, ANP, RANP with ±1σ shading for all three output dimensions (x, y, z)
5. Summary statistics table    - printed and saved as a .txt

Usage
-----
# Run from the repo root:
cd /home/fernando/tesis/underwater-localization-topologies
python src/evaluation/eval_anp_vs_ranp_masked.py

# Optional CLI overrides (all have sensible defaults):
python eval_anp_vs_ranp_masked.py \
    --data-dir /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_low_variance \
    --topology random \
    --anp-ckpt /home/fernando/tesis/underwater-localization-topologies/src/training/results/ANP_topologies_masked/lowvar/masked_dropbernoulli_p0.2_train_mean_first/topology_random/best_checkpoint.pth.tar \
    --ranp-ckpt /home/fernando/tesis/underwater-localization-topologies/src/training/results/RANP_topologies_masked/ranp_dropbernoulli_p0.2_train_mean_first_rnn-lstm_h256_l1/topology_random/best_checkpoint.pth.tar \
    --output-dir results/eval_anp_vs_ranp/lowvar_random \
    --ctx-fracs 0.05,0.10,0.20,0.30,0.50,0.70,0.90 \
    --fixed-ctx-frac 0.30 \
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
    / "ranp_dropbernoulli_p0.2_train_mean_first_rnn-lstm_h256_l1"
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


def compute_y_stats(
    train_data: list, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    Y = np.concatenate([y for _, y in train_data], axis=0)
    y_mean = torch.tensor(Y.mean(axis=0), dtype=torch.float32, device=device)
    y_std  = torch.tensor(Y.std(axis=0) + 1e-6, dtype=torch.float32, device=device)
    return y_mean, y_std


def compute_x_sensor_means(
    train_data: list, num_time_points: int, num_sensors: int
) -> np.ndarray:
    """Returns (S, P) mean array."""
    X = np.concatenate([x for x, _ in train_data], axis=0)
    X3 = X.reshape(X.shape[0], num_time_points, num_sensors)
    return X3.mean(axis=0).T  # (S, P)


def group_by_theta(
    test_data: list, test_thetas: list
) -> Dict[float, list]:
    groups: Dict[float, list] = {}
    for sample, theta in zip(test_data, test_thetas):
        groups.setdefault(theta, []).append(sample)
    return groups


# ══════════════════════════════════════════════════════════════════════════════
# Masking / augmentation (replica from training scripts)
# ══════════════════════════════════════════════════════════════════════════════

def augment_x_with_full_mask(
    x_batch: torch.Tensor,
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


def load_ranp(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    """Load the recurrent ANP (src.models.r_anp.LatentModel)."""
    model = ranp_module.LatentModel(
        num_hidden=NUM_HIDDEN,
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        rnn_type="lstm",
        rnn_layers=1,
        rnn_dropout=0.0,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[RANP] loaded from {ckpt_path}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Prediction wrappers (handle the two different forward signatures)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_anp(
    model: torch.nn.Module,
    x_aug: torch.Tensor,       # (B, T, INPUT_DIM)
    ctx_idx: torch.Tensor,     # (Nc,)
    ctx_y: torch.Tensor,       # (B, Nc, output_dim)  - normalised
    tar_idx: torch.Tensor,     # (Nt,)
    tar_y: torch.Tensor,       # (B, Nt, output_dim)  - normalised
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (mean_norm, var_norm, loss, nll)."""
    context_x = x_aug[:, ctx_idx, :]
    target_x  = x_aug[:, tar_idx, :]
    mean, var, loss, kl, nll = model(context_x, ctx_y, target_x, tar_y, beta=1.0)
    return mean, var, loss, nll


@torch.no_grad()
def predict_ranp(
    model: torch.nn.Module,
    x_aug: torch.Tensor,
    ctx_idx: torch.Tensor,
    ctx_y: torch.Tensor,
    tar_idx: torch.Tensor,
    tar_y: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (mean_norm, var_norm, loss, nll)."""
    mean, var, loss, kl, nll = model(
        x_seq=x_aug,
        context_indices=ctx_idx,
        context_y=ctx_y,
        target_indices=tar_idx,
        target_y=tar_y,
        beta=1.0,
    )
    return mean, var, loss, nll


# ══════════════════════════════════════════════════════════════════════════════
# Context index sampling (deterministic "first N" mode, matching training)
# ══════════════════════════════════════════════════════════════════════════════

def ctx_indices_first(total: int, n: int, device: torch.device) -> torch.Tensor:
    return torch.arange(n, device=device)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1 - MAE per theta group at a fixed context fraction
# ══════════════════════════════════════════════════════════════════════════════

def eval_mae_per_theta(
    anp_model: torch.nn.Module,
    ranp_model: torch.nn.Module,
    theta_groups: Dict[float, list],
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    x_means_SP: torch.Tensor,
    ctx_frac: float,
    device: torch.device,
    batch_size: int = 8,
) -> Dict[str, Dict[float, float]]:
    """Compute non-context MAE for ANP and RANP for each theta group.

    Returns nested dict: results[model_name][theta] = mae_value
    """
    results = {"ANP": {}, "RANP": {}}

    for theta, group in sorted(theta_groups.items()):
        ds     = NavigationTrajectoryDataset(group)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        anp_maes, ranp_maes = [], []
        for x_raw, y_raw in loader:
            x_raw, y_raw = x_raw.to(device), y_raw.to(device)
            B, T, _ = x_raw.shape

            x_aug   = augment_x_with_full_mask(x_raw, x_means_SP, NUM_TIME_PTS, NUM_SENSORS)
            y_norm  = (y_raw - y_mean) / y_std

            n_ctx   = max(1, min(T - 1, int(round(ctx_frac * T))))
            ctx_idx = ctx_indices_first(T, n_ctx, device)
            tar_idx = torch.arange(T, device=device)

            non_ctx = torch.ones(T, dtype=torch.bool, device=device)
            non_ctx[ctx_idx] = False

            ctx_y = y_norm[:, ctx_idx, :]
            tar_y = y_norm[:, tar_idx, :]

            # ANP
            mean_anp, _, _, _ = predict_anp(anp_model, x_aug, ctx_idx, ctx_y, tar_idx, tar_y)
            pred_anp = mean_anp * y_std + y_mean
            anp_maes.append(F.l1_loss(pred_anp[:, non_ctx, :], y_raw[:, non_ctx, :], reduction="mean").item())

            # RANP
            mean_ranp, _, _, _ = predict_ranp(ranp_model, x_aug, ctx_idx, ctx_y, tar_idx, tar_y)
            pred_ranp = mean_ranp * y_std + y_mean
            ranp_maes.append(F.l1_loss(pred_ranp[:, non_ctx, :], y_raw[:, non_ctx, :], reduction="mean").item())

        results["ANP"][theta]  = float(np.mean(anp_maes))
        results["RANP"][theta] = float(np.mean(ranp_maes))

    return results


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2 - MAE / NLL vs context fraction (sweep)
# ══════════════════════════════════════════════════════════════════════════════

def eval_vs_context_fraction(
    anp_model: torch.nn.Module,
    ranp_model: torch.nn.Module,
    test_data: list,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    x_means_SP: torch.Tensor,
    ctx_fracs: List[float],
    device: torch.device,
    batch_size: int = 8,
    holdout_frac: float = 0.20,
) -> Dict[str, Dict[str, List[float]]]:
    """Sweep context fractions and record MAE + NLL for each model.

    MAE is computed on a **fixed held-out tail** of the trajectory
    (last `holdout_frac` fraction of time steps) so that the evaluated
    points are identical across all context sizes. This avoids the
    artefact where MAE appears to grow with context because the non-context
    set shifts towards harder, later time steps.

    Context fracs that would overlap with the held-out tail are capped
    at (1 - holdout_frac - 1/T).

    Returns:
        out[model_name]["mae"] = [mae_at_frac0, mae_at_frac1, ...]
        out[model_name]["nll"] = [nll_at_frac0, ...]
    """
    out = {
        "ANP":  {"mae": [], "nll": []},
        "RANP": {"mae": [], "nll": []},
    }
    ds     = NavigationTrajectoryDataset(test_data)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    for frac in tqdm(ctx_fracs, desc="Context fraction sweep"):
        anp_maes, ranp_maes = [], []
        anp_nlls, ranp_nlls = [], []

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
            mean_a, var_a, _, nll_a = predict_anp(anp_model, x_aug, ctx_idx, ctx_y, tar_idx, tar_y)
            pred_a = mean_a * y_std + y_mean
            anp_maes.append(F.l1_loss(pred_a[:, holdout_idx, :], y_raw[:, holdout_idx, :], reduction="mean").item())
            anp_nlls.append(nll_a.item())

            # RANP
            mean_r, var_r, _, nll_r = predict_ranp(ranp_model, x_aug, ctx_idx, ctx_y, tar_idx, tar_y)
            pred_r = mean_r * y_std + y_mean
            ranp_maes.append(F.l1_loss(pred_r[:, holdout_idx, :], y_raw[:, holdout_idx, :], reduction="mean").item())
            ranp_nlls.append(nll_r.item())

        out["ANP"]["mae"].append(float(np.mean(anp_maes)))
        out["ANP"]["nll"].append(float(np.mean(anp_nlls)))
        out["RANP"]["mae"].append(float(np.mean(ranp_maes)))
        out["RANP"]["nll"].append(float(np.mean(ranp_nlls)))

    return out


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3 - Trajectory plots (per theta group)
# ══════════════════════════════════════════════════════════════════════════════

def plot_trajectories_for_theta(
    anp_model: torch.nn.Module,
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
    colors = {"ANP": "#1f77b4", "RANP": "#ff7f0e"}

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

            mean_a, var_a, _, _ = predict_anp(anp_model, x_aug, ctx_idx, ctx_y, tar_idx, tar_y)
            mean_r, var_r, _, _ = predict_ranp(ranp_model, x_aug, ctx_idx, ctx_y, tar_idx, tar_y)

            # denormalise
            pred_a  = (mean_a * y_std + y_mean).squeeze(0).cpu().numpy()   # (T,3)
            std_a   = (torch.sqrt(var_a) * y_std).squeeze(0).cpu().numpy()
            pred_r  = (mean_r * y_std + y_mean).squeeze(0).cpu().numpy()
            std_r   = (torch.sqrt(var_r) * y_std).squeeze(0).cpu().numpy()
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

                # ANP
                ax.plot(t_axis, pred_a[:, dim_idx], color=colors["ANP"], lw=1.3,
                        label="ANP" if row == 0 else "_")
                ax.fill_between(
                    t_axis,
                    pred_a[:, dim_idx] - std_a[:, dim_idx],
                    pred_a[:, dim_idx] + std_a[:, dim_idx],
                    alpha=0.2, color=colors["ANP"],
                )

                # RANP
                ax.plot(t_axis, pred_r[:, dim_idx], color=colors["RANP"], lw=1.3,
                        label="RANP" if row == 0 else "_")
                ax.fill_between(
                    t_axis,
                    pred_r[:, dim_idx] - std_r[:, dim_idx],
                    pred_r[:, dim_idx] + std_r[:, dim_idx],
                    alpha=0.2, color=colors["RANP"],
                )

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

def plot_2d_paths_per_theta(
    anp_model: torch.nn.Module,
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
) -> None:
    """XY-plane trajectory paths for both models vs ground truth."""
    rng = np.random.default_rng(seed)
    colors = {"ANP": "#1f77b4", "RANP": "#ff7f0e"}

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

            mean_a, _, _, _ = predict_anp(anp_model, x_aug, ctx_idx, ctx_y, tar_idx, tar_y)
            mean_r, _, _, _ = predict_ranp(ranp_model, x_aug, ctx_idx, ctx_y, tar_idx, tar_y)

            pred_a = (mean_a * y_std + y_mean).squeeze(0).cpu().numpy()
            pred_r = (mean_r * y_std + y_mean).squeeze(0).cpu().numpy()
            gt     = y_raw_np

            ax.plot(gt[:, 0], gt[:, 1], "k-", lw=1.5, label="GT")
            ax.plot(pred_a[:, 0], pred_a[:, 1], color=colors["ANP"], lw=1.3, linestyle="--", label="ANP")
            ax.plot(pred_r[:, 0], pred_r[:, 1], color=colors["RANP"], lw=1.3, linestyle="--", label="RANP")
            # mark context
            ax.scatter(gt[:n_ctx, 0], gt[:n_ctx, 1], c="red", s=10, zorder=5, label="Context pts")

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

def plot_mae_per_theta(
    results: Dict[str, Dict[float, float]],
    output_dir: Path,
    ctx_frac: float,
) -> None:
    thetas = sorted(results["ANP"].keys())
    x      = np.arange(len(thetas))
    width  = 0.35

    anp_vals  = [results["ANP"][t]  for t in thetas]
    ranp_vals = [results["RANP"][t] for t in thetas]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, anp_vals,  width, label="ANP",  color="#1f77b4", alpha=0.85)
    ax.bar(x + width/2, ranp_vals, width, label="RANP", color="#ff7f0e", alpha=0.85)

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


def plot_mae_vs_context(
    sweep: Dict[str, Dict[str, List[float]]],
    ctx_fracs: List[float],
    output_dir: Path,
    holdout_frac: float = 0.20,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, ylabel in zip(
        axes,
        ["mae", "nll"],
        ["MAE (m)", "NLL (nats)"],
    ):
        ax.plot([f * 100 for f in ctx_fracs], sweep["ANP"][metric],
                "o-", color="#1f77b4", label="ANP",  lw=2)
        ax.plot([f * 100 for f in ctx_fracs], sweep["RANP"][metric],
                "s-", color="#ff7f0e", label="RANP", lw=2)
        ax.set_xlabel("Context fraction (%)")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{ylabel} vs context fraction")

    plt.suptitle(
        f"ANP vs RANP — performance across context sizes\n"
        f"(MAE evaluated on fixed last {holdout_frac*100:.0f}% of trajectory)",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(output_dir / "mae_nll_vs_context.png", dpi=150)
    plt.close(fig)
    print(f"  mae_nll_vs_context.png saved")


# ══════════════════════════════════════════════════════════════════════════════
# Summary CSV / TXT
# ══════════════════════════════════════════════════════════════════════════════

def save_summary(
    mae_per_theta: Dict[str, Dict[float, float]],
    sweep: Dict[str, Dict[str, List[float]]],
    ctx_fracs: List[float],
    fixed_ctx_frac: float,
    output_dir: Path,
) -> None:
    lines = [
        "=" * 62,
        "  ANP vs RANP — Evaluation Summary",
        f"  Topology: random | Data: low variance",
        f"  Fixed context fraction: {fixed_ctx_frac*100:.0f}%",
        "=" * 62,
        "",
        "MAE per θ group (non-context points)",
        "-" * 44,
        f"{'θ':>6}  {'ANP (m)':>10}  {'RANP (m)':>10}  {'Δ (RANP-ANP)':>14}",
    ]

    for theta in sorted(mae_per_theta["ANP"]):
        a = mae_per_theta["ANP"][theta]
        r = mae_per_theta["RANP"][theta]
        lines.append(f"{theta:>6.1f}  {a:>10.4f}  {r:>10.4f}  {r-a:>+14.4f}")

    avg_anp  = float(np.mean(list(mae_per_theta["ANP"].values())))
    avg_ranp = float(np.mean(list(mae_per_theta["RANP"].values())))
    lines += [
        "-" * 44,
        f"{'avg':>6}  {avg_anp:>10.4f}  {avg_ranp:>10.4f}  {avg_ranp-avg_anp:>+14.4f}",
        "",
        "MAE vs context fraction",
        "-" * 44,
        f"{'ctx%':>6}  {'ANP MAE':>10}  {'RANP MAE':>10}",
    ]
    for frac, a_mae, r_mae in zip(ctx_fracs, sweep["ANP"]["mae"], sweep["RANP"]["mae"]):
        lines.append(f"{frac*100:>5.0f}%  {a_mae:>10.4f}  {r_mae:>10.4f}")

    lines += [
        "",
        "NLL vs context fraction",
        "-" * 44,
        f"{'ctx%':>6}  {'ANP NLL':>10}  {'RANP NLL':>10}",
    ]
    for frac, a_nll, r_nll in zip(ctx_fracs, sweep["ANP"]["nll"], sweep["RANP"]["nll"]):
        lines.append(f"{frac*100:>5.0f}%  {a_nll:>10.4f}  {r_nll:>10.4f}")

    summary_str = "\n".join(lines)
    print("\n" + summary_str)

    with open(output_dir / "summary.txt", "w") as f:
        f.write(summary_str + "\n")

    # CSV for easy import
    import csv
    with open(output_dir / "mae_per_theta.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["theta", "ANP_mae", "RANP_mae", "delta"])
        for theta in sorted(mae_per_theta["ANP"]):
            a = mae_per_theta["ANP"][theta]
            r = mae_per_theta["RANP"][theta]
            w.writerow([theta, a, r, r - a])

    with open(output_dir / "sweep_vs_context.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ctx_frac", "ANP_mae", "RANP_mae", "ANP_nll", "RANP_nll"])
        for frac, am, rm, an, rn in zip(
            ctx_fracs,
            sweep["ANP"]["mae"], sweep["RANP"]["mae"],
            sweep["ANP"]["nll"], sweep["RANP"]["nll"],
        ):
            w.writerow([frac, am, rm, an, rn])

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
    print("\n[1/6] Loading data …")
    train_data, test_data, metadata = load_data(args.data_dir, args.topology)
    theta_groups = group_by_theta(test_data, metadata["test_thetas"])
    print(f"  Train: {len(train_data)} | Test: {len(test_data)}")
    print(f"  θ values: {sorted(theta_groups)}")

    y_mean, y_std = compute_y_stats(train_data, device)
    x_means_np    = compute_x_sensor_means(train_data, NUM_TIME_PTS, NUM_SENSORS)
    x_means_SP    = torch.tensor(x_means_np, dtype=torch.float32, device=device)

    # ── Load models ──────────────────────────────────────────────────────────
    print("\n[2/6] Loading models …")
    anp_model  = load_anp(args.anp_ckpt,   device)
    ranp_model = load_ranp(args.ranp_ckpt, device)

    # ── Test 1: MAE per theta group ──────────────────────────────────────────
    print(f"\n[3/6] MAE per θ group  (ctx = {args.fixed_ctx_frac*100:.0f}%) …")
    mae_per_theta = eval_mae_per_theta(
        anp_model, ranp_model, theta_groups,
        y_mean, y_std, x_means_SP,
        ctx_frac=args.fixed_ctx_frac,
        device=device,
        batch_size=args.batch_size,
    )
    plot_mae_per_theta(mae_per_theta, output_dir, args.fixed_ctx_frac)

    # ── Test 2: MAE / NLL vs context fraction ────────────────────────────────
    print("\n[4/6] Context fraction sweep …")
    HOLDOUT_FRAC = 0.20
    sweep = eval_vs_context_fraction(
        anp_model, ranp_model, test_data,
        y_mean, y_std, x_means_SP,
        ctx_fracs=ctx_fracs,
        device=device,
        batch_size=args.batch_size,
        holdout_frac=HOLDOUT_FRAC,
    )
    plot_mae_vs_context(sweep, ctx_fracs, output_dir, holdout_frac=HOLDOUT_FRAC)

    # ── Test 3: Trajectory plots (per-dim, per theta) ────────────────────────
    print(f"\n[5/6] Trajectory plots (ctx = {args.fixed_ctx_frac*100:.0f}%) …")
    plot_trajectories_for_theta(
        anp_model, ranp_model, theta_groups,
        y_mean, y_std, x_means_SP,
        ctx_frac=args.fixed_ctx_frac,
        output_dir=output_dir,
        num_traj_plots=args.num_traj_plots,
        device=device,
        seed=args.seed,
    )

    # ── Test 4: XY path plots ────────────────────────────────────────────────
    plot_2d_paths_per_theta(
        anp_model, ranp_model, theta_groups,
        y_mean, y_std, x_means_SP,
        ctx_frac=args.fixed_ctx_frac,
        output_dir=output_dir,
        num_traj_plots=args.num_traj_plots,
        device=device,
        seed=args.seed,
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n[6/6] Saving summary …")
    save_summary(mae_per_theta, sweep, ctx_fracs, args.fixed_ctx_frac, output_dir)

    print(f"\nAll results written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
