"""
Comparative evaluation: ANP (set-based, masked) vs Sequential ANP (SNP, masked).

Produces per topology:
  1. MAE heatmap (models × theta) with best-per-theta highlight
  2. MAE vs context % curves (ANP vs SNP)
  3. Trajectory prediction plots (GT, ANP, SNP) — XY bird's-eye
  4. Per-axis time-series with ±kσ confidence bands
  5. Per-step MAE evolution along the trajectory
  6. Cross-topology summary text + comparison plot

Usage example:
  cd /home/fernando/tesis/underwater-localization-topologies
  python -m src.evaluation.evaluate_anp_vs_snp \
    --data-dir  data/data/data_processed_topologies_low_variance \
    --anp-dir   src/training/results/ANP_topologies_masked/masked_dropbernoulli_p0.2_train_mean_first \
    --snp-dir   src/training/results/ANP_topologies_masked/masked_dropbernoulli_p0.2_train_mean_first_snp-lstm_l1_d0.1 \
    --output-dir results/eval_anp_vs_snp \
    --topologies random aligned ellipsoidal
"""

import os
import sys
import csv
import pickle
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

# Make sure src package is importable when running as __main__
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.r_anp import LatentModel, SequentialLatentModel
from src.utils.nav_dataset import NavigationTrajectoryDataset


# ====================================================================
# Helpers
# ====================================================================

def _compute_y_stats(train_data, device: torch.device):
    """y_mean, y_std from training data — matches training normalization."""
    Y = np.concatenate([y for _, y in train_data], axis=0)
    y_mean = torch.tensor(Y.mean(axis=0), dtype=torch.float32, device=device)
    y_std  = torch.tensor(Y.std(axis=0) + 1e-6, dtype=torch.float32, device=device)
    return y_mean, y_std


def _compute_x_sensor_means(train_data, P: int, S: int):
    """Per-sensor mean (S, P) for mask-fill."""
    X = np.concatenate([x for x, _ in train_data], axis=0)
    Dx = X.shape[1]
    assert Dx == P * S, f"Dx={Dx} ≠ P*S={P*S}"
    X3 = X.reshape(X.shape[0], P, S)
    return X3.mean(axis=0).T  # (S, P)


def _apply_mask_all_sensors(x_batch, x_means_SP, P, S, device):
    """
    Apply "all sensors available" masking (no dropout) and append
    explicit mask features — same as training does at eval time.
    Returns (B, T, Dx+S).
    """
    B, T, Dx = x_batch.shape
    sensor_mask = torch.ones(B, S, device=device)             # all sensors on
    mask_feat = sensor_mask.unsqueeze(1).expand(B, T, S)       # (B, T, S)
    x_aug = torch.cat([x_batch, mask_feat], dim=-1)            # (B, T, Dx+S)
    return x_aug


def _sample_context_indices(total: int, n_ctx: int, g: torch.Generator, device):
    perm = torch.randperm(total, generator=g, device=device)
    return perm[:n_ctx].sort().values


# ====================================================================
# Evaluator
# ====================================================================

class ANPvsSNPEvaluator:
    """Compare a set-based masked ANP with a Sequential masked ANP."""

    def __init__(
        self,
        data_dir: Path,
        anp_result_dir: Path,
        snp_result_dir: Path,
        output_dir: Path,
        num_sensors: int = 10,
        num_time_points: int = 201,
        num_hidden: int = 128,
        snp_rnn_type: str = "lstm",
        snp_rnn_layers: int = 1,
        snp_rnn_dropout: float = 0.1,
        seed: int = 42,
        batch_size: int = 8,
    ):
        self.data_dir = Path(data_dir)
        self.anp_dir  = Path(anp_result_dir)
        self.snp_dir  = Path(snp_result_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.S = num_sensors
        self.P = num_time_points
        self.num_hidden = num_hidden
        self.snp_rnn_type = snp_rnn_type
        self.snp_rnn_layers = snp_rnn_layers
        self.snp_rnn_dropout = snp_rnn_dropout
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        # Reproducibility
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Caches
        self._y_stats_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._x_means_cache: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_pickle(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def load_topology_data(self, topology: str):
        """Returns (train_data, test_data, theta_groups, theta_values)."""
        tdir = self.data_dir / f"topology_{topology}"
        train_data = self._load_pickle(tdir / "train_data.pkl")
        test_data  = self._load_pickle(tdir / "test_data.pkl")
        metadata   = self._load_pickle(tdir / "metadata.pkl")

        theta_groups: Dict[float, list] = {}
        for sample, theta in zip(test_data, metadata["test_thetas"]):
            theta_groups.setdefault(theta, []).append(sample)

        theta_values = sorted(theta_groups.keys())
        return train_data, test_data, theta_groups, theta_values

    def get_y_stats(self, topology: str, train_data=None):
        if topology not in self._y_stats_cache:
            if train_data is None:
                tdir = self.data_dir / f"topology_{topology}"
                train_data = self._load_pickle(tdir / "train_data.pkl")
            self._y_stats_cache[topology] = _compute_y_stats(train_data, self.device)
        return self._y_stats_cache[topology]

    def get_x_means(self, topology: str, train_data=None):
        if topology not in self._x_means_cache:
            if train_data is None:
                tdir = self.data_dir / f"topology_{topology}"
                train_data = self._load_pickle(tdir / "train_data.pkl")
            np_means = _compute_x_sensor_means(train_data, self.P, self.S)
            self._x_means_cache[topology] = torch.tensor(
                np_means, dtype=torch.float32, device=self.device
            )
        return self._x_means_cache[topology]

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_anp(self, topology: str, input_dim: int, output_dim: int) -> LatentModel:
        path = self.anp_dir / f"topology_{topology}" / "best_checkpoint.pth.tar"
        if not path.exists():
            raise FileNotFoundError(f"ANP checkpoint not found: {path}")
        model = LatentModel(num_hidden=self.num_hidden,
                            input_dim=input_dim, output_dim=output_dim)
        ckpt = torch.load(path, map_location=self.device)
        model.load_state_dict(ckpt["model"])
        model.to(self.device).eval()
        print(f"  Loaded ANP  ({path.parent.name})")
        return model

    def _load_snp(self, topology: str, input_dim: int, output_dim: int) -> SequentialLatentModel:
        path = self.snp_dir / f"topology_{topology}" / "best_checkpoint.pth.tar"
        if not path.exists():
            raise FileNotFoundError(f"SNP checkpoint not found: {path}")
        model = SequentialLatentModel(
            num_hidden=self.num_hidden,
            input_dim=input_dim, output_dim=output_dim,
            rnn_type=self.snp_rnn_type,
            rnn_layers=self.snp_rnn_layers,
            rnn_dropout=self.snp_rnn_dropout,
        )
        ckpt = torch.load(path, map_location=self.device)
        model.load_state_dict(ckpt["model"])
        model.to(self.device).eval()
        print(f"  Loaded SNP  ({path.parent.name})")
        return model

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _norm_y(y, y_mean, y_std):
        return (y - y_mean.view(1, 1, -1)) / y_std.view(1, 1, -1)

    @staticmethod
    def _denorm_y(y_norm, y_mean, y_std):
        return y_norm * y_std.view(1, 1, -1) + y_mean.view(1, 1, -1)

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _predict_anp(self, model, x_aug, y, y_mean, y_std, ctx_idx):
        """Set-based ANP: context/target split."""
        y_norm = self._norm_y(y, y_mean, y_std)
        cx = x_aug[:, ctx_idx, :]
        cy = y_norm[:, ctx_idx, :]
        tx = x_aug  # all points as targets
        pred_norm, var_norm, *_ = model(cx, cy, tx)
        pred = self._denorm_y(pred_norm, y_mean, y_std)
        var_real = var_norm * (y_std.view(1, 1, -1) ** 2)
        return pred, var_real

    @torch.no_grad()
    def _predict_snp(self, model, x_aug, y, y_mean, y_std):
        """Sequential ANP: full-sequence inference (prior only)."""
        y_norm = self._norm_y(y, y_mean, y_std)
        pred_norm, var_norm, *_ = model(x_aug, y_norm, target_y=None)
        pred = self._denorm_y(pred_norm, y_mean, y_std)
        var_real = var_norm * (y_std.view(1, 1, -1) ** 2)
        return pred, var_real

    # ==================================================================
    # 1. MAE heatmap  (models × theta)
    # ==================================================================

    def compute_mae_heatmap(
        self,
        topology: str,
        theta_groups: Dict[float, list],
        theta_values: List[float],
        anp_model, snp_model,
        y_mean, y_std,
        context_percent: int = 30,
    ) -> Tuple[np.ndarray, List[str]]:
        """Returns (mae_matrix[n_models, n_thetas], model_names)."""
        model_names = ["ANP (set-based)", "Sequential ANP"]
        mae_matrix = np.zeros((2, len(theta_values)))

        g = torch.Generator(device=self.device)
        g.manual_seed(self.seed)

        for j, theta in enumerate(tqdm(theta_values, desc="    θ groups")):
            ds = NavigationTrajectoryDataset(theta_groups[theta])
            loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
            errs = {n: [] for n in model_names}

            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                T = x.size(1)
                n_ctx = max(1, min(T - 1, int(context_percent / 100 * T)))
                ctx_idx = _sample_context_indices(T, n_ctx, g, self.device)
                non_ctx = torch.ones(T, dtype=torch.bool, device=self.device)
                non_ctx[ctx_idx] = False

                x_aug = _apply_mask_all_sensors(x, None, self.P, self.S, self.device)

                pred_anp, _ = self._predict_anp(anp_model, x_aug, y, y_mean, y_std, ctx_idx)
                mae_anp = F.l1_loss(pred_anp[:, non_ctx], y[:, non_ctx], reduction="none").mean(dim=[1, 2])
                errs["ANP (set-based)"].extend(mae_anp.cpu().numpy())

                pred_snp, _ = self._predict_snp(snp_model, x_aug, y, y_mean, y_std)
                mae_snp = F.l1_loss(pred_snp[:, non_ctx], y[:, non_ctx], reduction="none").mean(dim=[1, 2])
                errs["Sequential ANP"].extend(mae_snp.cpu().numpy())

            for i, n in enumerate(model_names):
                mae_matrix[i, j] = np.mean(errs[n])

        return mae_matrix, model_names

    def plot_heatmap(self, mae_matrix, model_names, theta_values, topology, save_dir):
        fig, ax = plt.subplots(figsize=(max(10, 2 * len(theta_values)), 4))
        best_idx = np.nanargmin(mae_matrix, axis=0)

        sns.heatmap(
            mae_matrix, annot=True, fmt=".3f", cmap="viridis",
            xticklabels=[f"{t:.1f}" for t in theta_values],
            yticklabels=model_names,
            cbar_kws={"label": "MAE"}, ax=ax, annot_kws={"size": 11},
        )
        for j, i in enumerate(best_idx):
            rect = patches.Rectangle((j, i), 1, 1, fill=False,
                                     edgecolor="blue", linewidth=3)
            ax.add_patch(rect)

        ax.set_xlabel("θ (channel variance)", fontsize=13)
        ax.set_ylabel("Model", fontsize=13)
        ax.set_title(f"MAE Comparison — {topology.capitalize()}\n"
                     f"(blue border = best per θ)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_dir / f"heatmap_{topology}.png", dpi=200, bbox_inches="tight")
        plt.close()

    # ==================================================================
    # 2. MAE vs context % curves
    # ==================================================================

    def evaluate_context_sensitivity(
        self, topology, theta_groups, theta_values,
        anp_model, snp_model, y_mean, y_std,
    ):
        ctx_pcts = list(range(5, 95, 5))
        results = {
            "ANP (set-based)": {c: [] for c in ctx_pcts},
            "Sequential ANP":  {c: [] for c in ctx_pcts},
        }

        g = torch.Generator(device=self.device)

        for ctx_pct in tqdm(ctx_pcts, desc="    Context sweep"):
            g.manual_seed(self.seed)
            for theta in theta_values:
                ds = NavigationTrajectoryDataset(theta_groups[theta])
                loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    T = x.size(1)
                    n_ctx = max(1, min(T - 1, int(ctx_pct / 100 * T)))
                    ctx_idx = _sample_context_indices(T, n_ctx, g, self.device)
                    non_ctx = torch.ones(T, dtype=torch.bool, device=self.device)
                    non_ctx[ctx_idx] = False
                    x_aug = _apply_mask_all_sensors(x, None, self.P, self.S, self.device)

                    pred_anp, _ = self._predict_anp(anp_model, x_aug, y, y_mean, y_std, ctx_idx)
                    mae_a = F.l1_loss(pred_anp[:, non_ctx], y[:, non_ctx], reduction="none").mean(dim=[1, 2])
                    results["ANP (set-based)"][ctx_pct].extend(mae_a.cpu().numpy())

                    pred_snp, _ = self._predict_snp(snp_model, x_aug, y, y_mean, y_std)
                    mae_s = F.l1_loss(pred_snp[:, non_ctx], y[:, non_ctx], reduction="none").mean(dim=[1, 2])
                    results["Sequential ANP"][ctx_pct].extend(mae_s.cpu().numpy())

        means = {
            name: {c: np.mean(v) for c, v in d.items()}
            for name, d in results.items()
        }
        return ctx_pcts, means, results

    def plot_context_curves(self, ctx_pcts, means, topology, save_dir):
        fig, ax = plt.subplots(figsize=(12, 6))
        for name, color in [("ANP (set-based)", "#2E86AB"), ("Sequential ANP", "#F18F01")]:
            ys = [means[name][c] for c in ctx_pcts]
            ax.plot(ctx_pcts, ys, "-o", linewidth=2.5, markersize=5,
                    color=color, label=name)
            best_c = min(means[name], key=means[name].get)
            ax.plot(best_c, means[name][best_c], "*", markersize=14, color=color)

        ax.set_xlabel("Context %", fontsize=13)
        ax.set_ylabel("MAE (non-context points)", fontsize=13)
        ax.set_title(f"MAE vs Context Size — {topology.capitalize()}", fontsize=15, fontweight="bold")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(save_dir / f"context_curves_{topology}.png", dpi=200, bbox_inches="tight")
        plt.close()

    # ==================================================================
    # 3. Trajectory prediction plots (XY bird's-eye)
    # ==================================================================

    def plot_trajectories(
        self, topology, theta_groups, anp_model, snp_model,
        y_mean, y_std, save_dir,
        target_thetas=(0.1, 0.3), n_traj=10, context_percent=30,
    ):
        for theta in target_thetas:
            if theta not in theta_groups:
                theta = min(theta_groups, key=lambda t: abs(t - theta))
            group_data = theta_groups[theta]
            samples = random.sample(group_data, min(n_traj, len(group_data)))

            theta_dir = save_dir / f"trajectories_theta_{theta:.1f}"
            theta_dir.mkdir(parents=True, exist_ok=True)

            g = torch.Generator(device=self.device)
            g.manual_seed(self.seed)

            for idx, (x_np, y_np) in enumerate(samples):
                x = torch.FloatTensor(x_np).unsqueeze(0).to(self.device)
                y = torch.FloatTensor(y_np).unsqueeze(0).to(self.device)
                T = x.size(1)
                n_ctx = max(1, min(T - 1, int(context_percent / 100 * T)))
                ctx_idx = _sample_context_indices(T, n_ctx, g, self.device)

                x_aug = _apply_mask_all_sensors(x, None, self.P, self.S, self.device)

                pred_anp, _ = self._predict_anp(anp_model, x_aug, y, y_mean, y_std, ctx_idx)
                pred_snp, _ = self._predict_snp(snp_model, x_aug, y, y_mean, y_std)

                gt  = y.squeeze(0).cpu().numpy()[:, :2]
                anp = pred_anp.squeeze(0).cpu().numpy()[:, :2]
                snp = pred_snp.squeeze(0).cpu().numpy()[:, :2]

                fig, ax = plt.subplots(figsize=(10, 8))
                ax.plot(gt[:, 0], gt[:, 1], "--", color="blue", lw=2.5, label="Ground Truth", alpha=0.8)
                ax.plot(anp[:, 0], anp[:, 1], "-", color="green", lw=2, label="ANP (set-based)")
                ax.plot(snp[:, 0], snp[:, 1], "-", color="red", lw=2, label="Sequential ANP")

                # context boundary marker
                ctx_idx_np = ctx_idx.cpu().numpy()
                ax.scatter(gt[ctx_idx_np, 0], gt[ctx_idx_np, 1],
                           marker="x", s=20, color="gray", alpha=0.4, zorder=3, label="Context pts")

                ax.plot(gt[0, 0], gt[0, 1], "go", ms=10, zorder=5, label="Start")
                ax.plot(gt[-1, 0], gt[-1, 1], "rs", ms=10, zorder=5, label="End")

                ax.set_xlabel("X [m]", fontsize=14)
                ax.set_ylabel("Y [m]", fontsize=14)
                ax.set_title(f"Trajectory {idx+1} — {topology.capitalize()} (θ={theta:.1f})",
                             fontsize=15, fontweight="bold")
                ax.legend(fontsize=11, loc="best")
                ax.set_aspect("equal", adjustable="box")
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(theta_dir / f"trajectory_{idx+1}.png", dpi=200, bbox_inches="tight")
                plt.close()

            print(f"    Saved {len(samples)} trajectory plots → {theta_dir}")

    # ==================================================================
    # 4. Per-axis time-series with ±kσ
    # ==================================================================

    def plot_axiswise_ci(
        self, topology, theta_groups, anp_model, snp_model,
        y_mean, y_std, save_dir,
        target_theta=0.1, n_traj=3, k=1.0, context_percent=30,
    ):
        if target_theta not in theta_groups:
            target_theta = min(theta_groups, key=lambda t: abs(t - target_theta))
        group_data = theta_groups[target_theta]
        samples = random.sample(group_data, min(n_traj, len(group_data)))

        n = len(samples)
        fig, axes = plt.subplots(n, 2, figsize=(18, 6 * n))
        axes = np.atleast_2d(axes)

        g = torch.Generator(device=self.device)
        g.manual_seed(self.seed)

        for i, (x_np, y_np) in enumerate(samples):
            x = torch.FloatTensor(x_np).unsqueeze(0).to(self.device)
            y = torch.FloatTensor(y_np).unsqueeze(0).to(self.device)
            T = x.size(1)
            n_ctx = max(1, min(T - 1, int(context_percent / 100 * T)))
            ctx_idx = _sample_context_indices(T, n_ctx, g, self.device)

            x_aug = _apply_mask_all_sensors(x, None, self.P, self.S, self.device)

            pred_anp, var_anp = self._predict_anp(anp_model, x_aug, y, y_mean, y_std, ctx_idx)
            pred_snp, var_snp = self._predict_snp(snp_model, x_aug, y, y_mean, y_std)

            gt_np   = y.squeeze(0).cpu().numpy()
            anp_np  = pred_anp.squeeze(0).cpu().numpy()
            snp_np  = pred_snp.squeeze(0).cpu().numpy()
            std_anp = np.sqrt(var_anp.squeeze(0).cpu().numpy())
            std_snp = np.sqrt(var_snp.squeeze(0).cpu().numpy())
            ts = np.arange(T)

            for d in range(2):
                ax = axes[i, d]
                label = "X" if d == 0 else "Y"
                ax.plot(ts, gt_np[:, d], "--", color="blue", lw=2, label="GT")

                # ANP
                ax.plot(ts, anp_np[:, d], color="green", lw=1.5, label="ANP mean")
                lo = anp_np[:, d] - k * std_anp[:, d]
                hi = anp_np[:, d] + k * std_anp[:, d]
                ax.fill_between(ts, lo, hi, color="green", alpha=0.15, label=f"ANP ±{k}σ")

                # SNP
                ax.plot(ts, snp_np[:, d], color="red", lw=1.5, label="SNP mean")
                lo = snp_np[:, d] - k * std_snp[:, d]
                hi = snp_np[:, d] + k * std_snp[:, d]
                ax.fill_between(ts, lo, hi, color="red", alpha=0.15, label=f"SNP ±{k}σ")

                # Mark context boundary
                ctx_boundary = n_ctx
                ax.axvline(ctx_boundary, color="gray", ls=":", lw=1, alpha=0.6)
                ax.set_title(f"Traj {i+1} — {label}-axis  ({topology}, θ={target_theta:.1f})", fontsize=13)
                ax.legend(fontsize=9, loc="best")
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out = save_dir / f"axiswise_ci_{topology}_theta{target_theta:.1f}_k{k:.1f}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"    Saved axis-wise CI plot → {out}")

    # ==================================================================
    # 5. Per-step MAE evolution along the trajectory
    # ==================================================================

    def compute_per_step_mae(
        self, topology, theta_groups, theta_values,
        anp_model, snp_model, y_mean, y_std, context_percent=30,
    ):
        """Average MAE at each time-step t (averaged over all test trajectories)."""
        g = torch.Generator(device=self.device)
        g.manual_seed(self.seed)

        # We'll accumulate sum & count per step
        first_T = None
        anp_step_sum: torch.Tensor = torch.zeros(1, device=self.device)
        snp_step_sum: torch.Tensor = torch.zeros(1, device=self.device)
        count = 0

        for theta in theta_values:
            ds = NavigationTrajectoryDataset(theta_groups[theta])
            loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                B, T, _ = x.shape
                if first_T is None:
                    first_T = T
                    anp_step_sum = torch.zeros(T, device=self.device)
                    snp_step_sum = torch.zeros(T, device=self.device)

                n_ctx = max(1, min(T - 1, int(context_percent / 100 * T)))
                ctx_idx = _sample_context_indices(T, n_ctx, g, self.device)
                x_aug = _apply_mask_all_sensors(x, None, self.P, self.S, self.device)

                pred_anp, _ = self._predict_anp(anp_model, x_aug, y, y_mean, y_std, ctx_idx)
                pred_snp, _ = self._predict_snp(snp_model, x_aug, y, y_mean, y_std)

                # MAE per step, averaged over output dims, then over batch
                mae_anp_step = (pred_anp - y).abs().mean(dim=-1).mean(dim=0)  # (T,)
                mae_snp_step = (pred_snp - y).abs().mean(dim=-1).mean(dim=0)

                anp_step_sum += mae_anp_step * B
                snp_step_sum += mae_snp_step * B
                count += B

        anp_per_step = (anp_step_sum / count).cpu().numpy()
        snp_per_step = (snp_step_sum / count).cpu().numpy()
        return anp_per_step, snp_per_step

    def plot_per_step_mae(self, anp_per_step, snp_per_step, topology, save_dir, context_percent=30):
        T = len(anp_per_step)
        ts = np.arange(T)

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(ts, anp_per_step, color="green", lw=2, label="ANP (set-based)")
        ax.plot(ts, snp_per_step, color="red", lw=2, label="Sequential ANP")

        ctx_boundary = max(1, int(context_percent / 100 * T))
        ax.axvline(ctx_boundary, color="gray", ls=":", lw=1.5, alpha=0.7,
                   label=f"Context boundary ({context_percent}%)")

        ax.set_xlabel("Time step", fontsize=13)
        ax.set_ylabel("MAE", fontsize=13)
        ax.set_title(f"Per-Step MAE — {topology.capitalize()}", fontsize=15, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / f"per_step_mae_{topology}.png", dpi=200, bbox_inches="tight")
        plt.close()

    # ==================================================================
    # 6. Per-theta context curves (one plot with all thetas, ANP vs SNP)
    # ==================================================================

    def compute_per_theta_context_curves(
        self, theta_groups, theta_values,
        anp_model, snp_model, y_mean, y_std,
    ):
        ctx_pcts = list(range(5, 95, 5))
        # results[model_name][theta][ctx_pct] = mean_mae
        results = {
            "ANP (set-based)": {th: {} for th in theta_values},
            "Sequential ANP":  {th: {} for th in theta_values},
        }

        g = torch.Generator(device=self.device)

        for ctx_pct in tqdm(ctx_pcts, desc="    Per-θ context sweep"):
            for theta in theta_values:
                g.manual_seed(self.seed)
                ds = NavigationTrajectoryDataset(theta_groups[theta])
                loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
                anp_maes, snp_maes = [], []
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    T = x.size(1)
                    n_ctx = max(1, min(T - 1, int(ctx_pct / 100 * T)))
                    ctx_idx = _sample_context_indices(T, n_ctx, g, self.device)
                    non_ctx = torch.ones(T, dtype=torch.bool, device=self.device)
                    non_ctx[ctx_idx] = False
                    x_aug = _apply_mask_all_sensors(x, None, self.P, self.S, self.device)

                    pred_a, _ = self._predict_anp(anp_model, x_aug, y, y_mean, y_std, ctx_idx)
                    anp_maes.extend(
                        F.l1_loss(pred_a[:, non_ctx], y[:, non_ctx], reduction="none")
                        .mean(dim=[1, 2]).cpu().numpy()
                    )
                    pred_s, _ = self._predict_snp(snp_model, x_aug, y, y_mean, y_std)
                    snp_maes.extend(
                        F.l1_loss(pred_s[:, non_ctx], y[:, non_ctx], reduction="none")
                        .mean(dim=[1, 2]).cpu().numpy()
                    )

                results["ANP (set-based)"][theta][ctx_pct] = np.mean(anp_maes)
                results["Sequential ANP"][theta][ctx_pct]  = np.mean(snp_maes)

        return ctx_pcts, results

    def plot_per_theta_context_curves(self, ctx_pcts, results, theta_values, topology, save_dir):
        """One figure per theta showing ANP vs SNP context curves."""
        n_theta = len(theta_values)
        cols = min(n_theta, 4)
        rows = (n_theta + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), squeeze=False)

        for idx, theta in enumerate(theta_values):
            r, c = divmod(idx, cols)
            ax = axes[r][c]
            for name, color in [("ANP (set-based)", "#2E86AB"), ("Sequential ANP", "#F18F01")]:
                ys = [results[name][theta][cp] for cp in ctx_pcts]
                ax.plot(ctx_pcts, ys, "-o", lw=2, ms=4, color=color, label=name)
            ax.set_title(f"θ = {theta:.1f}", fontsize=13)
            ax.set_xlabel("Context %"); ax.set_ylabel("MAE")
            ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

        # hide unused subplots
        for idx in range(n_theta, rows * cols):
            r, c = divmod(idx, cols)
            axes[r][c].set_visible(False)

        fig.suptitle(f"Context Sensitivity per θ — {topology.capitalize()}",
                     fontsize=16, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.savefig(save_dir / f"context_per_theta_{topology}.png",
                    dpi=200, bbox_inches="tight")
        plt.close()

    # ==================================================================
    # Run per topology
    # ==================================================================

    def run_topology(self, topology: str, context_percent: int = 30):
        print(f"\n{'='*60}")
        print(f"  Evaluating topology: {topology.upper()}")
        print(f"{'='*60}")

        topo_dir = self.output_dir / topology
        topo_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        print("  Loading data...")
        train_data, test_data, theta_groups, theta_values = self.load_topology_data(topology)
        y_mean, y_std = self.get_y_stats(topology, train_data)

        first = next(iter(theta_groups.values()))[0]
        input_dim = first[0].shape[-1] + self.S  # Dx + S (mask features)
        output_dim = first[1].shape[-1]

        # Load models
        print("  Loading models...")
        anp = self._load_anp(topology, input_dim, output_dim)
        snp = self._load_snp(topology, input_dim, output_dim)

        results_summary: Dict[str, Any] = {"topology": topology}

        # 1. Heatmap
        print("  [1/6] MAE heatmap...")
        mae_mat, m_names = self.compute_mae_heatmap(
            topology, theta_groups, theta_values, anp, snp, y_mean, y_std,
            context_percent=context_percent,
        )
        self.plot_heatmap(mae_mat, m_names, theta_values, topology, topo_dir)
        results_summary["mae_matrix"] = mae_mat
        results_summary["model_names"] = m_names
        results_summary["theta_values"] = theta_values

        # 2. Context sensitivity (overall)
        print("  [2/6] Context sensitivity...")
        ctx_pcts, ctx_means, ctx_raw = self.evaluate_context_sensitivity(
            topology, theta_groups, theta_values, anp, snp, y_mean, y_std,
        )
        self.plot_context_curves(ctx_pcts, ctx_means, topology, topo_dir)
        results_summary["ctx_means"] = ctx_means

        # 3. Trajectory plots
        print("  [3/6] Trajectory plots...")
        self.plot_trajectories(
            topology, theta_groups, anp, snp, y_mean, y_std, topo_dir,
            target_thetas=[theta_values[0], theta_values[-1]] if len(theta_values) > 1 else theta_values,
            n_traj=10, context_percent=context_percent,
        )

        # 4. Axis-wise CI
        print("  [4/6] Axis-wise CI plots...")
        for theta in [theta_values[0], theta_values[-1]] if len(theta_values) > 1 else theta_values:
            self.plot_axiswise_ci(
                topology, theta_groups, anp, snp, y_mean, y_std, topo_dir,
                target_theta=theta, n_traj=3, k=1.0, context_percent=context_percent,
            )

        # 5. Per-step MAE
        print("  [5/6] Per-step MAE...")
        anp_step, snp_step = self.compute_per_step_mae(
            topology, theta_groups, theta_values, anp, snp, y_mean, y_std,
            context_percent=context_percent,
        )
        self.plot_per_step_mae(anp_step, snp_step, topology, topo_dir, context_percent)
        results_summary["anp_per_step"] = anp_step
        results_summary["snp_per_step"] = snp_step

        # 6. Per-theta context curves
        print("  [6/6] Per-θ context curves...")
        ctx_pcts_t, ctx_theta_res = self.compute_per_theta_context_curves(
            theta_groups, theta_values, anp, snp, y_mean, y_std,
        )
        self.plot_per_theta_context_curves(ctx_pcts_t, ctx_theta_res, theta_values, topology, topo_dir)

        # Save numerical results
        self._save_results_txt(topo_dir, topology, mae_mat, m_names, theta_values,
                               ctx_means, ctx_pcts, anp_step, snp_step)

        return results_summary

    # ------------------------------------------------------------------
    # Textual report
    # ------------------------------------------------------------------

    def _save_results_txt(self, topo_dir, topology, mae_mat, m_names, theta_values,
                          ctx_means, ctx_pcts, anp_step, snp_step):
        path = topo_dir / "results.txt"
        with open(path, "w") as f:
            f.write(f"ANP vs Sequential ANP — Topology: {topology.upper()}\n")
            f.write("=" * 70 + "\n\n")

            # Heatmap table
            f.write("MAE per theta:\n")
            f.write(f"{'Model':<22}")
            for th in theta_values:
                f.write(f"θ={th:<7.1f}")
            f.write("Mean\n")
            f.write("-" * 70 + "\n")
            for i, name in enumerate(m_names):
                f.write(f"{name:<22}")
                vals = []
                for j in range(len(theta_values)):
                    v = mae_mat[i, j]
                    f.write(f"{v:<9.4f}")
                    vals.append(v)
                f.write(f"{np.mean(vals):.4f}\n")

            # Winner
            anp_mean = np.mean(mae_mat[0])
            snp_mean = np.mean(mae_mat[1])
            winner = "Sequential ANP" if snp_mean < anp_mean else "ANP (set-based)"
            delta = abs(anp_mean - snp_mean) / max(anp_mean, snp_mean) * 100
            f.write(f"\nWinner: {winner}  (Δ = {delta:.1f}%)\n")

            # Best context
            f.write("\nBest context % (overall):\n")
            for name in ctx_means:
                best_c = min(ctx_means[name], key=ctx_means[name].get)
                f.write(f"  {name}: {best_c}% (MAE={ctx_means[name][best_c]:.4f})\n")

            # Per-step summary
            f.write(f"\nPer-step MAE (mean over all steps):\n")
            f.write(f"  ANP:  {anp_step.mean():.4f}\n")
            f.write(f"  SNP:  {snp_step.mean():.4f}\n")

        print(f"    Results saved → {path}")

    # ------------------------------------------------------------------
    # Cross-topology summary
    # ------------------------------------------------------------------

    def cross_topology_summary(self, all_results: Dict[str, dict]):
        summary_dir = self.output_dir
        path = summary_dir / "cross_topology_summary.txt"

        with open(path, "w") as f:
            f.write("Cross-Topology Comparison: ANP vs Sequential ANP\n")
            f.write("=" * 70 + "\n\n")
            for topo, res in all_results.items():
                mae_mat = res["mae_matrix"]
                anp_mean = np.mean(mae_mat[0])
                snp_mean = np.mean(mae_mat[1])
                winner = "SNP" if snp_mean < anp_mean else "ANP"
                delta = abs(anp_mean - snp_mean) / max(anp_mean, snp_mean) * 100
                f.write(f"{topo.upper():<15}  ANP={anp_mean:.4f}  SNP={snp_mean:.4f}"
                        f"  → {winner} wins (Δ={delta:.1f}%)\n")

        print(f"\nCross-topology summary → {path}")

        # Plot
        topos = list(all_results.keys())
        anp_vals = [np.mean(all_results[t]["mae_matrix"][0]) for t in topos]
        snp_vals = [np.mean(all_results[t]["mae_matrix"][1]) for t in topos]

        x = np.arange(len(topos))
        w = 0.35
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - w / 2, anp_vals, w, label="ANP (set-based)", color="#2E86AB")
        bars2 = ax.bar(x + w / 2, snp_vals, w, label="Sequential ANP", color="#F18F01")

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10)

        ax.set_xticks(x)
        ax.set_xticklabels([t.capitalize() for t in topos], fontsize=12)
        ax.set_ylabel("Mean MAE", fontsize=13)
        ax.set_title("ANP vs Sequential ANP — All Topologies", fontsize=15, fontweight="bold")
        ax.legend(fontsize=12); ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(summary_dir / "cross_topology_bar.png", dpi=200, bbox_inches="tight")
        plt.close()

        # Per-step overlay
        fig, ax = plt.subplots(figsize=(14, 5))
        colors_topo = {"random": "#A23B72", "aligned": "#F18F01", "ellipsoidal": "#2E86AB"}
        for topo, res in all_results.items():
            c = colors_topo.get(topo, "gray")
            T = len(res["anp_per_step"])
            ts = np.arange(T)
            ax.plot(ts, res["anp_per_step"], ls="--", color=c, lw=1.5,
                    label=f"ANP — {topo}")
            ax.plot(ts, res["snp_per_step"], ls="-", color=c, lw=2,
                    label=f"SNP — {topo}")
        ax.set_xlabel("Time step", fontsize=13)
        ax.set_ylabel("MAE", fontsize=13)
        ax.set_title("Per-Step MAE — All Topologies", fontsize=15, fontweight="bold")
        ax.legend(fontsize=9, ncol=2); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(summary_dir / "cross_topology_per_step_mae.png", dpi=200, bbox_inches="tight")
        plt.close()

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def run(self, topologies: List[str], context_percent: int = 30):
        all_results = {}
        for topo in topologies:
            try:
                res = self.run_topology(topo, context_percent=context_percent)
                all_results[topo] = res
            except Exception as e:
                print(f"  ERROR on {topo}: {e}")
                import traceback; traceback.print_exc()

        if len(all_results) > 1:
            self.cross_topology_summary(all_results)

        print(f"\n{'='*60}")
        print(f"All done. Results in: {self.output_dir}")
        print(f"{'='*60}")


# ====================================================================
# CLI
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare set-based ANP vs Sequential ANP per topology",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="Root of processed topology data")
    parser.add_argument("--anp-dir", type=Path, required=True, help="Root dir of ANP (set-based, masked) checkpoints")
    parser.add_argument("--snp-dir", type=Path, required=True, help="Root dir of Sequential ANP checkpoints")
    parser.add_argument("--output-dir", type=Path, default=Path("results/eval_anp_vs_snp"))
    parser.add_argument("--topologies", nargs="+", default=["random", "aligned", "ellipsoidal"])
    parser.add_argument("--context-percent", type=int, default=30, help="Context %% for heatmap, trajectories, per-step MAE")
    parser.add_argument("--num-sensors", type=int, default=10)
    parser.add_argument("--num-time-points", type=int, default=201)
    parser.add_argument("--num-hidden", type=int, default=128)
    parser.add_argument("--snp-rnn-type", type=str, default="lstm")
    parser.add_argument("--snp-rnn-layers", type=int, default=1)
    parser.add_argument("--snp-rnn-dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    evaluator = ANPvsSNPEvaluator(
        data_dir=args.data_dir,
        anp_result_dir=args.anp_dir,
        snp_result_dir=args.snp_dir,
        output_dir=args.output_dir,
        num_sensors=args.num_sensors,
        num_time_points=args.num_time_points,
        num_hidden=args.num_hidden,
        snp_rnn_type=args.snp_rnn_type,
        snp_rnn_layers=args.snp_rnn_layers,
        snp_rnn_dropout=args.snp_rnn_dropout,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    evaluator.run(args.topologies, context_percent=args.context_percent)


if __name__ == "__main__":
    main()
