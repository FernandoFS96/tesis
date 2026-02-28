#!/usr/bin/env python3
"""
Evaluate baseline ANP vs recurrent ANP (RANP-style) on processed topology datasets.

What this script produces (per topology):
1) Heatmaps of MAE and NLL by model x theta-group at a fixed context fraction.
2) Delta heatmap (RANP - ANP) for MAE.
3) Context curve: mean MAE over theta groups vs context percentages.
4) Coverage and interval-width plots.
5) Robustness curves vs sensor dropout at test time.
6) Qualitative trajectory plots comparing GT vs ANP vs RANP at fixed context.
7) CSV summaries for all aggregated metrics.

Assumptions aligned with your training pipeline:
- Data comes from processed topology folders created by data_process_topology.py.
- Each topology folder contains train_data.pkl, test_data.pkl, metadata.pkl.
- Checkpoints are under:
    <experiment_root>/topology_<name>/best_checkpoint.pth.tar
  (falls back to last_checkpoint.pth.tar if needed)
- Targets are evaluated on NON-context points by default, matching your training/validation style.

Examples
--------
python evaluate_anp_vs_ranp.py \
  --data-dir /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_low_variance \
  --baseline-root /home/fernando/tesis/underwater-localization-topologies/src/training/results/ANP_topologies_masked/masked_dropbernoulli_p0.2_train_mean_first \
  --recurrent-root /home/fernando/tesis/underwater-localization-topologies/src/training/results/ANP_topologies_masked/masked_dropbernoulli_p0.2_train_mean_first_rnn-lstm_h128_l1_d0.0 \
  --output-dir /home/fernando/tesis/underwater-localization-topologies/results/eval_anp_vs_ranp_2 \
  --device cuda

Optional forecasting-only evaluation:
python evaluate_anp_vs_ranp.py ... --target-mode future_only
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------
# Model imports
# ---------------------------
from src.models.anp import LatentModel

class TemporalEncoder(nn.Module):
    """Fallback temporal encoder for loading recurrent checkpoints."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        rnn_type: str = "lstm",
        layer_norm: bool = True,
    ):
        super().__init__()
        rnn_type = rnn_type.lower()
        if rnn_type not in {"lstm", "gru"}:
            raise ValueError(f"Unsupported rnn_type={rnn_type}")
        rnn_dropout = dropout if num_layers > 1 else 0.0
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=rnn_dropout,
                batch_first=True,
                bidirectional=False,
            )
        else:
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=rnn_dropout,
                batch_first=True,
                bidirectional=False,
            )
        self.norm = nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        h_seq, _ = self.rnn(x_seq)
        return self.norm(h_seq)


# ---------------------------
# Dataset helpers
# ---------------------------
class EvalDataset(Dataset):
    def __init__(self, data: Sequence[Tuple[np.ndarray, np.ndarray]], thetas: Sequence[float]):
        self.data = data
        self.thetas = np.asarray(thetas, dtype=np.float32)
        if len(self.data) != len(self.thetas):
            raise ValueError(f"data/thetas length mismatch: {len(self.data)} vs {len(self.thetas)}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        x, y = self.data[idx]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(self.thetas[idx], dtype=torch.float32),
            torch.tensor(idx, dtype=torch.long),
        )


@dataclass
class ModelBundle:
    name: str
    model: nn.Module
    rnn_encoder: Optional[nn.Module]
    device: torch.device
    uses_rnn: bool
    checkpoint_path: str

    def eval(self):
        self.model.eval()
        if self.rnn_encoder is not None:
            self.rnn_encoder.eval()


# ---------------------------
# Utility functions (aligned with training script)
# ---------------------------
def compute_y_stats(train_data):
    Y = np.concatenate([y for _, y in train_data], axis=0)
    y_mean = torch.tensor(Y.mean(axis=0), dtype=torch.float32)
    y_std = torch.tensor(Y.std(axis=0) + 1e-6, dtype=torch.float32)
    return y_mean, y_std


def compute_x_sensor_means(train_data, num_time_points: int, num_sensors: int):
    X = np.concatenate([x for x, _ in train_data], axis=0)
    Dx = X.shape[1]
    P, S = num_time_points, num_sensors
    if Dx != P * S:
        raise ValueError(f"Dx={Dx} but expected P*S={P*S}")
    X3 = X.reshape(X.shape[0], P, S)
    mean_PS = X3.mean(axis=0)
    return mean_PS.T  # (S,P)


def load_topology_split(data_dir: str, topology: str):
    topo_dir = os.path.join(data_dir, f"topology_{topology}")
    paths = {
        "train": os.path.join(topo_dir, "train_data.pkl"),
        "val": os.path.join(topo_dir, "val_data.pkl"),
        "test": os.path.join(topo_dir, "test_data.pkl"),
        "meta": os.path.join(topo_dir, "metadata.pkl"),
    }
    missing = [k for k, v in paths.items() if not os.path.exists(v)]
    if missing:
        raise FileNotFoundError(f"Missing files for topology={topology}: {missing}")
    with open(paths["train"], "rb") as f:
        train_data = pickle.load(f)
    with open(paths["val"], "rb") as f:
        val_data = pickle.load(f)
    with open(paths["test"], "rb") as f:
        test_data = pickle.load(f)
    with open(paths["meta"], "rb") as f:
        metadata = pickle.load(f)
    return train_data, val_data, test_data, metadata


def sample_sensor_mask(B: int, S: int, mode: str, p_drop: float, device: torch.device, generator=None):
    if p_drop <= 0:
        return torch.ones((B, S), device=device)

    if mode == "bernoulli":
        if generator is None:
            keep = (torch.rand(B, S, device=device) > p_drop)
        else:
            keep = (torch.rand(B, S, generator=generator, device=device) > p_drop)
    elif mode == "k_uniform":
        keep = torch.zeros(B, S, dtype=torch.bool, device=device)
        for b in range(B):
            if generator is None:
                k = torch.randint(1, S + 1, (1,), device=device).item()
                idx = torch.randperm(S, device=device)[:k]
            else:
                k = torch.randint(1, S + 1, (1,), generator=generator, device=device).item()
                idx = torch.randperm(S, generator=generator, device=device)[:k]
            keep[b, idx] = True
    else:
        raise ValueError(f"Unknown sensor_drop_mode: {mode}")

    all_off = ~keep.any(dim=1)
    if all_off.any():
        n = int(all_off.sum().item())
        idx = torch.randint(0, S, (n,), generator=generator, device=device)
        keep[all_off, idx] = True
    return keep.float()


def apply_sensor_dropout_and_append_mask(
    x_batch: torch.Tensor,
    sensor_mask: torch.Tensor,
    x_means_SP: torch.Tensor,
    num_time_points: int,
    num_sensors: int,
    fill: str = "train_mean",
):
    B, T, Dx = x_batch.shape
    P, S = num_time_points, num_sensors
    if Dx != P * S:
        raise ValueError(f"x_batch Dx={Dx} but expected {P*S}")
    x4 = x_batch.view(B, T, P, S)

    if fill == "zero":
        fill_val = torch.zeros((B, T, P, S), device=x_batch.device, dtype=x_batch.dtype)
    elif fill == "train_mean":
        mu = x_means_SP.T.view(1, 1, P, S).to(device=x_batch.device, dtype=x_batch.dtype)
        fill_val = mu.expand(B, T, P, S)
    else:
        raise ValueError(f"Unknown fill={fill}")

    m = sensor_mask.view(B, 1, 1, S)
    x4_masked = x4 * m + fill_val * (1.0 - m)
    x_masked = x4_masked.reshape(B, T, Dx)
    mask_feat = sensor_mask.view(B, 1, S).expand(B, T, S)
    return torch.cat([x_masked, mask_feat], dim=-1)


def context_indices(total_points: int, frac: float, device: torch.device, mode: str = "first", generator=None):
    k = max(1, min(total_points - 1, int(round(frac * total_points))))
    if mode == "first":
        return torch.arange(k, device=device)
    if mode == "random":
        perm = torch.randperm(total_points, device=device, generator=generator)
        return perm[:k].sort().values
    raise ValueError(f"Unknown mode={mode}")


def target_indices_from_context(total_points: int, ctx_idx: torch.Tensor, device: torch.device, target_mode: str):
    if target_mode == "all_points":
        return torch.arange(total_points, device=device)
    if target_mode == "future_only":
        mask = torch.ones(total_points, dtype=torch.bool, device=device)
        mask[ctx_idx] = False
        return torch.arange(total_points, device=device)[mask]
    raise ValueError(f"Unknown target_mode={target_mode}")


def infer_model_dims_from_state_dict(model_sd: Dict[str, torch.Tensor]):
    num_hidden = model_sd["latent_encoder.input_projection.linear_layer.weight"].shape[0]
    output_dim = model_sd["decoder.mean_projection.linear_layer.weight"].shape[0]
    input_dim = model_sd["decoder.target_projection.linear_layer.weight"].shape[1]
    return num_hidden, input_dim, output_dim


def infer_rnn_config(rnn_sd: Dict[str, torch.Tensor]):
    weight_keys = [k for k in rnn_sd.keys() if k.startswith("rnn.weight_ih_l")]
    if not weight_keys:
        raise ValueError("Could not infer RNN config: no rnn.weight_ih_l* keys found")
    weight0 = rnn_sd["rnn.weight_ih_l0"]
    gates_times_hidden, input_dim = weight0.shape
    if gates_times_hidden % 4 == 0:
        rnn_type = "lstm"
        hidden_dim = gates_times_hidden // 4
    elif gates_times_hidden % 3 == 0:
        rnn_type = "gru"
        hidden_dim = gates_times_hidden // 3
    else:
        raise ValueError(f"Could not infer RNN type from weight_ih_l0 shape={tuple(weight0.shape)}")
    num_layers = len(weight_keys)
    return {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "rnn_type": rnn_type,
        "dropout": 0.0,
    }


def find_checkpoint(experiment_root: str, topology: str) -> str:
    candidates = [
        os.path.join(experiment_root, f"topology_{topology}", "best_checkpoint.pth.tar"),
        os.path.join(experiment_root, f"topology_{topology}", "last_checkpoint.pth.tar"),
        os.path.join(experiment_root, f"{topology}", "best_checkpoint.pth.tar"),
        os.path.join(experiment_root, f"{topology}", "last_checkpoint.pth.tar"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Could not find checkpoint for topology={topology} under {experiment_root}. Tried: {candidates}"
    )


def load_model_bundle(name: str, checkpoint_path: str, device: torch.device) -> ModelBundle:
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model" not in ckpt:
        raise KeyError(f"Checkpoint at {checkpoint_path} does not contain key 'model'")

    model_sd = ckpt["model"]
    num_hidden, input_dim, output_dim = infer_model_dims_from_state_dict(model_sd)
    model = LatentModel(num_hidden=num_hidden, input_dim=input_dim, output_dim=output_dim).to(device)
    model.load_state_dict(model_sd)

    rnn_encoder = None
    uses_rnn = "rnn_encoder" in ckpt
    if uses_rnn:
        rnn_cfg = infer_rnn_config(ckpt["rnn_encoder"])
        rnn_encoder = TemporalEncoder(**rnn_cfg).to(device)
        rnn_encoder.load_state_dict(ckpt["rnn_encoder"])

    bundle = ModelBundle(
        name=name,
        model=model,
        rnn_encoder=rnn_encoder,
        device=device,
        uses_rnn=uses_rnn,
        checkpoint_path=checkpoint_path,
    )
    bundle.eval()
    return bundle


def z_from_coverage(level: float, device: torch.device) -> float:
    p = torch.tensor((1.0 + level) / 2.0, device=device)
    return float(torch.distributions.Normal(0.0, 1.0).icdf(p).item())


# ---------------------------
# Evaluation core
# ---------------------------
def predictive_pass(
    bundle: ModelBundle,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    x_means_SP: torch.Tensor,
    num_time_points: int,
    num_sensors: int,
    context_frac: float,
    ctx_mode: str,
    target_mode: str,
    mask_fill: str,
    sensor_drop_mode: str,
    sensor_drop_p: float,
    coverage_levels: Sequence[float],
    generator=None,
):
    device = bundle.device
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    B, T, _ = x_batch.shape

    sensor_mask = sample_sensor_mask(B, num_sensors, sensor_drop_mode, sensor_drop_p, device=device, generator=generator)
    x_aug = apply_sensor_dropout_and_append_mask(
        x_batch, sensor_mask, x_means_SP,
        num_time_points=num_time_points,
        num_sensors=num_sensors,
        fill=mask_fill,
    )
    x_features = bundle.rnn_encoder(x_aug) if bundle.rnn_encoder is not None else x_aug

    y_norm = (y_batch - y_mean) / y_std

    ctx_idx = context_indices(T, context_frac, device=device, mode=ctx_mode, generator=generator)
    tar_idx = target_indices_from_context(T, ctx_idx, device=device, target_mode=target_mode)

    context_x = x_features[:, ctx_idx, :]
    context_y = y_norm[:, ctx_idx, :]
    target_x = x_features[:, tar_idx, :]
    target_y = y_norm[:, tar_idx, :]

    with torch.no_grad():
        pred_mean_norm, pred_var_norm, _, _, _ = bundle.model(
            context_x, context_y, target_x, target_y=None, beta=1.0
        )

    pred_mean = pred_mean_norm * y_std + y_mean
    pred_var = pred_var_norm * (y_std ** 2)
    pred_std = torch.sqrt(torch.clamp(pred_var, min=1e-10))
    gt = y_batch[:, tar_idx, :]

    # Evaluation region:
    # - all_points  -> predict all points but score only non-context points
    # - future_only -> tar_idx already excludes context, so score all tar_idx
    if target_mode == "all_points":
        eval_mask = torch.ones(len(tar_idx), dtype=torch.bool, device=device)
        eval_mask[ctx_idx] = False
    else:
        eval_mask = torch.ones(len(tar_idx), dtype=torch.bool, device=device)

    pred_mean_eval = pred_mean[:, eval_mask, :]
    pred_var_eval = pred_var[:, eval_mask, :]
    pred_std_eval = pred_std[:, eval_mask, :]
    gt_eval = gt[:, eval_mask, :]

    # pointwise metrics per sample
    abs_err = (pred_mean_eval - gt_eval).abs()
    mae_per_sample = abs_err.mean(dim=(1, 2))

    nll_point = 0.5 * torch.log(2 * torch.pi * pred_var_eval) + 0.5 * ((gt_eval - pred_mean_eval) ** 2) / pred_var_eval
    nll_per_sample = nll_point.mean(dim=(1, 2))

    coverage_per_level = {}
    width_per_level = {}
    for level in coverage_levels:
        z = z_from_coverage(level, device=device)
        lower = pred_mean_eval - z * pred_std_eval
        upper = pred_mean_eval + z * pred_std_eval
        cov = ((gt_eval >= lower) & (gt_eval <= upper)).float().mean(dim=(1, 2))
        width = (upper - lower).mean(dim=(1, 2))
        coverage_per_level[level] = cov
        width_per_level[level] = width

    return {
        "ctx_idx": ctx_idx.detach().cpu(),
        "tar_idx": tar_idx.detach().cpu(),
        "sensor_mask": sensor_mask.detach().cpu(),
        "pred_mean": pred_mean.detach().cpu(),
        "pred_var": pred_var.detach().cpu(),
        "pred_std": pred_std.detach().cpu(),
        "gt": gt.detach().cpu(),
        "mae": mae_per_sample.detach().cpu(),
        "nll": nll_per_sample.detach().cpu(),
        "coverage": {k: v.detach().cpu() for k, v in coverage_per_level.items()},
        "width": {k: v.detach().cpu() for k, v in width_per_level.items()},
    }


def evaluate_model_over_loader(
    bundle: ModelBundle,
    loader: DataLoader,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    x_means_SP: torch.Tensor,
    num_time_points: int,
    num_sensors: int,
    context_frac: float,
    ctx_mode: str,
    target_mode: str,
    mask_fill: str,
    sensor_drop_mode: str,
    sensor_drop_p: float,
    coverage_levels: Sequence[float],
    seed: int,
):
    metrics = []
    g = torch.Generator(device=bundle.device)
    g.manual_seed(seed)

    for x_batch, y_batch, theta_batch, idx_batch in loader:
        out = predictive_pass(
            bundle=bundle,
            x_batch=x_batch,
            y_batch=y_batch,
            y_mean=y_mean,
            y_std=y_std,
            x_means_SP=x_means_SP,
            num_time_points=num_time_points,
            num_sensors=num_sensors,
            context_frac=context_frac,
            ctx_mode=ctx_mode,
            target_mode=target_mode,
            mask_fill=mask_fill,
            sensor_drop_mode=sensor_drop_mode,
            sensor_drop_p=sensor_drop_p,
            coverage_levels=coverage_levels,
            generator=g,
        )

        for i in range(len(theta_batch)):
            row = {
                "theta": float(theta_batch[i].item()),
                "sample_idx": int(idx_batch[i].item()),
                "mae": float(out["mae"][i].item()),
                "nll": float(out["nll"][i].item()),
            }
            for level in coverage_levels:
                row[f"coverage_{level:.2f}"] = float(out["coverage"][level][i].item())
                row[f"width_{level:.2f}"] = float(out["width"][level][i].item())
            metrics.append(row)
    return metrics


# ---------------------------
# Aggregation / plotting
# ---------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def sorted_thetas(metrics_rows: Sequence[dict]) -> List[float]:
    vals = sorted({round(float(r["theta"]), 10) for r in metrics_rows})
    return vals


def aggregate_by_theta(metrics_rows: Sequence[dict], fields: Sequence[str]):
    groups = defaultdict(list)
    for r in metrics_rows:
        groups[round(float(r["theta"]), 10)].append(r)
    thetas = sorted(groups.keys())
    out = {"theta": thetas}
    for field in fields:
        out[field] = [float(np.mean([g[field] for g in groups[t]])) for t in thetas]
        out[field + "_std"] = [float(np.std([g[field] for g in groups[t]])) for t in thetas]
    return out


def write_csv(path: str, rows: Sequence[dict], fieldnames: Sequence[str]):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def make_matrix(model_to_theta_values: Dict[str, Dict[float, float]], theta_order: Sequence[float], model_order: Sequence[str]):
    M = np.full((len(model_order), len(theta_order)), np.nan, dtype=float)
    for i, model_name in enumerate(model_order):
        theta_map = model_to_theta_values[model_name]
        for j, theta in enumerate(theta_order):
            M[i, j] = theta_map.get(theta, np.nan)
    return M


def plot_heatmap(matrix: np.ndarray, row_labels: Sequence[str], col_labels: Sequence[str], title: str, save_path: str,
                 cmap: str = "viridis", fmt: str = ".3f", center: Optional[float] = None):
    ensure_dir(os.path.dirname(save_path))
    fig, ax = plt.subplots(figsize=(max(7, 0.7 * len(col_labels)), 2.2 + 0.8 * len(row_labels)))
    if center is None:
        im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    else:
        vmax = np.nanmax(np.abs(matrix - center))
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=center - vmax, vmax=center + vmax)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels([str(c) for c in col_labels], rotation=45, ha="right")
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(title.split(":")[-1].strip(), rotation=-90, va="bottom")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=9, color="white")

    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_context_curve(contexts_pct, model_stats: Dict[str, Dict[str, np.ndarray]], title: str, ylabel: str, save_path: str):
    ensure_dir(os.path.dirname(save_path))
    plt.figure(figsize=(8, 5))
    for model_name, stats in model_stats.items():
        mean = np.asarray(stats["mean"], dtype=float)
        std = np.asarray(stats["std"], dtype=float)
        plt.plot(contexts_pct, mean, marker="o", label=model_name)
        plt.fill_between(contexts_pct, mean - std, mean + std, alpha=0.2)
    plt.xlabel("Context (%)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_coverage(model_to_cov: Dict[str, Dict[float, float]], title: str, save_path: str):
    ensure_dir(os.path.dirname(save_path))
    levels = sorted(next(iter(model_to_cov.values())).keys())
    x = np.array(levels) * 100.0
    plt.figure(figsize=(7, 5))
    plt.plot(x, x / 100.0, linestyle="--", label="Ideal")
    for model_name, cov in model_to_cov.items():
        y = np.array([cov[l] for l in levels])
        plt.plot(x, y, marker="o", label=model_name)
    plt.xlabel("Nominal coverage (%)")
    plt.ylabel("Empirical coverage")
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_widths(model_to_width: Dict[str, Dict[float, float]], title: str, save_path: str):
    ensure_dir(os.path.dirname(save_path))
    levels = sorted(next(iter(model_to_width.values())).keys())
    x = np.array(levels) * 100.0
    plt.figure(figsize=(7, 5))
    for model_name, w in model_to_width.items():
        y = np.array([w[l] for l in levels])
        plt.plot(x, y, marker="o", label=model_name)
    plt.xlabel("Nominal coverage (%)")
    plt.ylabel("Mean interval width")
    plt.grid(True, alpha=0.3)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_robustness(drop_probs: Sequence[float], model_stats: Dict[str, Dict[str, np.ndarray]], title: str, ylabel: str, save_path: str):
    ensure_dir(os.path.dirname(save_path))
    plt.figure(figsize=(8, 5))
    for model_name, stats in model_stats.items():
        mean = np.asarray(stats["mean"], dtype=float)
        std = np.asarray(stats["std"], dtype=float)
        x = np.asarray(drop_probs, dtype=float)
        plt.plot(x, mean, marker="o", label=model_name)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)
    plt.xlabel("Sensor dropout p (test)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def _pick_qualitative_samples(metrics_rows_baseline: Sequence[dict], metrics_rows_recurrent: Sequence[dict], max_per_topology: int = 6):
    by_theta_b = defaultdict(list)
    by_theta_r = defaultdict(list)
    for r in metrics_rows_baseline:
        by_theta_b[round(float(r["theta"]), 10)].append(r)
    for r in metrics_rows_recurrent:
        by_theta_r[round(float(r["theta"]), 10)].append(r)

    selected = []
    for theta in sorted(by_theta_b.keys()):
        b = sorted(by_theta_b[theta], key=lambda x: x["sample_idx"])
        r = {x["sample_idx"]: x for x in by_theta_r[theta]}
        # choose the sample with largest absolute MAE difference for visual contrast
        candidates = []
        for row in b:
            idx = row["sample_idx"]
            if idx in r:
                diff = abs(r[idx]["mae"] - row["mae"])
                candidates.append((diff, idx))
        if candidates:
            idx = sorted(candidates, reverse=True)[0][1]
            selected.append((theta, idx))
    return selected[:max_per_topology]


def plot_trajectory_comparison(
    x: torch.Tensor,
    y: torch.Tensor,
    bundle_base: ModelBundle,
    bundle_rec: ModelBundle,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    x_means_SP: torch.Tensor,
    num_time_points: int,
    num_sensors: int,
    context_frac: float,
    ctx_mode: str,
    target_mode: str,
    mask_fill: str,
    sensor_drop_mode: str,
    sensor_drop_p: float,
    coverage_level: float,
    save_path: str,
    title: str,
):
    ensure_dir(os.path.dirname(save_path))
    x_b = x.unsqueeze(0)
    y_b = y.unsqueeze(0)

    out_b = predictive_pass(
        bundle_base, x_b, y_b, y_mean, y_std, x_means_SP,
        num_time_points, num_sensors, context_frac, ctx_mode, target_mode,
        mask_fill, sensor_drop_mode, sensor_drop_p, [coverage_level], generator=None,
    )
    out_r = predictive_pass(
        bundle_rec, x_b, y_b, y_mean, y_std, x_means_SP,
        num_time_points, num_sensors, context_frac, ctx_mode, target_mode,
        mask_fill, sensor_drop_mode, sensor_drop_p, [coverage_level], generator=None,
    )

    ctx_idx = out_b["ctx_idx"].numpy()
    tar_idx = out_b["tar_idx"].numpy()
    gt = y.numpy()
    pred_b = out_b["pred_mean"][0].numpy()
    std_b = out_b["pred_std"][0].numpy()
    pred_r = out_r["pred_mean"][0].numpy()
    std_r = out_r["pred_std"][0].numpy()

    z = z_from_coverage(coverage_level, bundle_base.device)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # XY trajectory
    axes[0].plot(gt[:, 0], gt[:, 1], label="GT", linewidth=2)
    axes[0].scatter(gt[ctx_idx, 0], gt[ctx_idx, 1], label="Context", s=22)
    axes[0].plot(gt[tar_idx, 0], gt[tar_idx, 1], alpha=0.15)
    axes[0].plot(pred_b[:, 0], pred_b[:, 1], label="ANP", linewidth=1.8)
    axes[0].plot(pred_r[:, 0], pred_r[:, 1], label="RANP", linewidth=1.8)
    axes[0].set_title("XY trajectory")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Coordinates over time (x-dim only for compactness)
    t_tar = tar_idx
    axes[1].plot(np.arange(len(gt)), gt[:, 0], label="GT x", linewidth=2)
    axes[1].scatter(ctx_idx, gt[ctx_idx, 0], label="Context", s=18)
    axes[1].plot(t_tar, pred_b[:, 0], label="ANP x", linewidth=1.6)
    axes[1].fill_between(t_tar, pred_b[:, 0] - z * std_b[:, 0], pred_b[:, 0] + z * std_b[:, 0], alpha=0.15)
    axes[1].plot(t_tar, pred_r[:, 0], label="RANP x", linewidth=1.6)
    axes[1].fill_between(t_tar, pred_r[:, 0] - z * std_r[:, 0], pred_r[:, 0] + z * std_r[:, 0], alpha=0.15)
    axes[1].set_title(f"Coordinate x over time ({int(coverage_level*100)}% interval)")
    axes[1].set_xlabel("Trajectory point")
    axes[1].set_ylabel("x")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


# ---------------------------
# Main
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate baseline ANP vs recurrent ANP across topologies/theta groups")
    p.add_argument("--data-dir", type=str, required=True, help="Processed data dir containing topology_* folders")
    p.add_argument("--baseline-root", type=str, required=True, help="Root dir of baseline ANP experiment")
    p.add_argument("--recurrent-root", type=str, required=True, help="Root dir of recurrent ANP experiment")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--topologies", type=str, default="random")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-sensors", type=int, default=10)
    p.add_argument("--num-time-points", type=int, default=201)
    p.add_argument("--mask-fill", type=str, default="train_mean", choices=["train_mean", "zero"])
    p.add_argument("--sensor-drop-mode", type=str, default="bernoulli", choices=["bernoulli", "k_uniform"])
    p.add_argument("--ctx-mode", type=str, default="first", choices=["first", "random"])
    p.add_argument("--target-mode", type=str, default="all_points", choices=["all_points", "future_only"])
    p.add_argument("--contexts", type=str, default="0.05,0.10,0.15,0.20,0.30,0.50,0.70,0.90")
    p.add_argument("--heatmap-context", type=float, default=0.30)
    p.add_argument("--qualitative-context", type=float, default=0.30)
    p.add_argument("--coverage-levels", type=str, default="0.50,0.80,0.90,0.95")
    p.add_argument("--drop-probs", type=str, default="0.0,0.2,0.5")
    p.add_argument("--eval-seed", type=int, default=1)
    p.add_argument("--qualitative-max-per-topology", type=int, default=6)
    p.add_argument("--qualitative-drop-p", type=float, default=0.0)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    topologies = [t.strip() for t in args.topologies.split(",") if t.strip()]
    context_fracs = [float(x) for x in args.contexts.split(",") if x.strip()]
    coverage_levels = [float(x) for x in args.coverage_levels.split(",") if x.strip()]
    drop_probs = [float(x) for x in args.drop_probs.split(",") if x.strip()]

    ensure_dir(args.output_dir)
    with open(os.path.join(args.output_dir, "eval_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    global_summary_rows = []

    for topology in topologies:
        print(f"\n=== Evaluating topology: {topology} ===")
        topo_out = os.path.join(args.output_dir, f"topology_{topology}")
        ensure_dir(topo_out)

        train_data, _, test_data, metadata = load_topology_split(args.data_dir, topology)
        thetas = metadata["test_thetas"]
        test_ds = EvalDataset(test_data, thetas)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        y_mean, y_std = compute_y_stats(train_data)
        x_means_np = compute_x_sensor_means(train_data, args.num_time_points, args.num_sensors)
        y_mean = y_mean.to(device)
        y_std = y_std.to(device)
        x_means_SP = torch.tensor(x_means_np, dtype=torch.float32, device=device)

        baseline_ckpt = find_checkpoint(args.baseline_root, topology)
        recurrent_ckpt = find_checkpoint(args.recurrent_root, topology)
        baseline = load_model_bundle("ANP", baseline_ckpt, device)
        recurrent = load_model_bundle("RANP", recurrent_ckpt, device)

        # 1) Fixed-context heatmaps (MAE + NLL + delta)
        fixed_metrics = {}
        for bundle in [baseline, recurrent]:
            rows = evaluate_model_over_loader(
                bundle=bundle,
                loader=test_loader,
                y_mean=y_mean,
                y_std=y_std,
                x_means_SP=x_means_SP,
                num_time_points=args.num_time_points,
                num_sensors=args.num_sensors,
                context_frac=args.heatmap_context,
                ctx_mode=args.ctx_mode,
                target_mode=args.target_mode,
                mask_fill=args.mask_fill,
                sensor_drop_mode=args.sensor_drop_mode,
                sensor_drop_p=0.0,
                coverage_levels=coverage_levels,
                seed=args.eval_seed,
            )
            fixed_metrics[bundle.name] = rows
            write_csv(
                os.path.join(topo_out, f"per_sample_{bundle.name.lower()}_ctx{args.heatmap_context:.2f}.csv"),
                rows,
                fieldnames=list(rows[0].keys()) if rows else ["theta", "sample_idx", "mae", "nll"],
            )

        theta_order = sorted_thetas(fixed_metrics["ANP"] + fixed_metrics["RANP"])
        model_order = ["ANP", "RANP"]

        model_to_theta_mae = {}
        model_to_theta_nll = {}
        agg_rows = []
        for model_name in model_order:
            agg = aggregate_by_theta(fixed_metrics[model_name], ["mae", "nll"] + [f"coverage_{l:.2f}" for l in coverage_levels] + [f"width_{l:.2f}" for l in coverage_levels])
            model_to_theta_mae[model_name] = {theta: val for theta, val in zip(agg["theta"], agg["mae"])}
            model_to_theta_nll[model_name] = {theta: val for theta, val in zip(agg["theta"], agg["nll"])}
            for i, theta in enumerate(agg["theta"]):
                row = {"model": model_name, "theta": theta, "mae": agg["mae"][i], "nll": agg["nll"][i]}
                for lvl in coverage_levels:
                    row[f"coverage_{lvl:.2f}"] = agg[f"coverage_{lvl:.2f}"][i]
                    row[f"width_{lvl:.2f}"] = agg[f"width_{lvl:.2f}"][i]
                agg_rows.append(row)
                global_summary_rows.append({"topology": topology, **row})

        write_csv(
            os.path.join(topo_out, f"aggregated_by_theta_ctx{args.heatmap_context:.2f}.csv"),
            agg_rows,
            fieldnames=list(agg_rows[0].keys()),
        )

        mae_mat = make_matrix(model_to_theta_mae, theta_order, model_order)
        nll_mat = make_matrix(model_to_theta_nll, theta_order, model_order)
        delta_mae = mae_mat[1:2, :] - mae_mat[0:1, :]

        plot_heatmap(
            mae_mat, model_order, theta_order,
            title=f"{topology}: MAE by model and theta (ctx={args.heatmap_context:.0%})",
            save_path=os.path.join(topo_out, f"heatmap_mae_ctx{args.heatmap_context:.2f}.png"),
            cmap="viridis", fmt=".3f",
        )
        plot_heatmap(
            nll_mat, model_order, theta_order,
            title=f"{topology}: NLL by model and theta (ctx={args.heatmap_context:.0%})",
            save_path=os.path.join(topo_out, f"heatmap_nll_ctx{args.heatmap_context:.2f}.png"),
            cmap="viridis", fmt=".3f",
        )
        plot_heatmap(
            delta_mae, ["RANP - ANP"], theta_order,
            title=f"{topology}: Delta MAE (ctx={args.heatmap_context:.0%})",
            save_path=os.path.join(topo_out, f"heatmap_delta_mae_ctx{args.heatmap_context:.2f}.png"),
            cmap="coolwarm", fmt="+.3f", center=0.0,
        )

        # 2) Context curves
        mae_curve_stats = {"ANP": {"mean": [], "std": []}, "RANP": {"mean": [], "std": []}}
        nll_curve_stats = {"ANP": {"mean": [], "std": []}, "RANP": {"mean": [], "std": []}}
        ctx_curve_rows = []
        for frac in context_fracs:
            for bundle in [baseline, recurrent]:
                rows = evaluate_model_over_loader(
                    bundle=bundle,
                    loader=test_loader,
                    y_mean=y_mean,
                    y_std=y_std,
                    x_means_SP=x_means_SP,
                    num_time_points=args.num_time_points,
                    num_sensors=args.num_sensors,
                    context_frac=frac,
                    ctx_mode=args.ctx_mode,
                    target_mode=args.target_mode,
                    mask_fill=args.mask_fill,
                    sensor_drop_mode=args.sensor_drop_mode,
                    sensor_drop_p=0.0,
                    coverage_levels=coverage_levels,
                    seed=args.eval_seed,
                )
                agg = aggregate_by_theta(rows, ["mae", "nll"])
                mae_mean = float(np.mean(agg["mae"]))
                mae_std = float(np.std(agg["mae"]))
                nll_mean = float(np.mean(agg["nll"]))
                nll_std = float(np.std(agg["nll"]))
                mae_curve_stats[bundle.name]["mean"].append(mae_mean)
                mae_curve_stats[bundle.name]["std"].append(mae_std)
                nll_curve_stats[bundle.name]["mean"].append(nll_mean)
                nll_curve_stats[bundle.name]["std"].append(nll_std)
                ctx_curve_rows.append({
                    "context_frac": frac,
                    "context_pct": 100.0 * frac,
                    "model": bundle.name,
                    "mae_mean_over_thetas": mae_mean,
                    "mae_std_over_thetas": mae_std,
                    "nll_mean_over_thetas": nll_mean,
                    "nll_std_over_thetas": nll_std,
                })

        write_csv(
            os.path.join(topo_out, "context_curves.csv"),
            ctx_curve_rows,
            fieldnames=list(ctx_curve_rows[0].keys()),
        )
        pct = [100.0 * c for c in context_fracs]
        plot_context_curve(
            pct, mae_curve_stats,
            title=f"{topology}: mean MAE over theta groups vs context",
            ylabel="MAE",
            save_path=os.path.join(topo_out, "curve_mae_vs_context.png"),
        )
        plot_context_curve(
            pct, nll_curve_stats,
            title=f"{topology}: mean NLL over theta groups vs context",
            ylabel="NLL",
            save_path=os.path.join(topo_out, "curve_nll_vs_context.png"),
        )

        # 3) Coverage + interval widths at fixed context
        cov_rows = []
        model_to_cov = {}
        model_to_width = {}
        for model_name in model_order:
            agg = aggregate_by_theta(
                fixed_metrics[model_name],
                [f"coverage_{l:.2f}" for l in coverage_levels] + [f"width_{l:.2f}" for l in coverage_levels],
            )
            model_to_cov[model_name] = {lvl: float(np.mean(agg[f"coverage_{lvl:.2f}"])) for lvl in coverage_levels}
            model_to_width[model_name] = {lvl: float(np.mean(agg[f"width_{lvl:.2f}"])) for lvl in coverage_levels}
            for lvl in coverage_levels:
                cov_rows.append({
                    "model": model_name,
                    "level": lvl,
                    "coverage_mean_over_thetas": model_to_cov[model_name][lvl],
                    "width_mean_over_thetas": model_to_width[model_name][lvl],
                })
        write_csv(os.path.join(topo_out, "coverage_width_summary.csv"), cov_rows, fieldnames=list(cov_rows[0].keys()))
        plot_coverage(model_to_cov, f"{topology}: empirical coverage (ctx={args.heatmap_context:.0%})", os.path.join(topo_out, "coverage_plot.png"))
        plot_widths(model_to_width, f"{topology}: interval width (ctx={args.heatmap_context:.0%})", os.path.join(topo_out, "interval_width_plot.png"))

        # 4) Robustness to sensor dropout
        rob_rows = []
        mae_rob_stats = {"ANP": {"mean": [], "std": []}, "RANP": {"mean": [], "std": []}}
        nll_rob_stats = {"ANP": {"mean": [], "std": []}, "RANP": {"mean": [], "std": []}}
        for p_drop in drop_probs:
            for bundle in [baseline, recurrent]:
                rows = evaluate_model_over_loader(
                    bundle=bundle,
                    loader=test_loader,
                    y_mean=y_mean,
                    y_std=y_std,
                    x_means_SP=x_means_SP,
                    num_time_points=args.num_time_points,
                    num_sensors=args.num_sensors,
                    context_frac=args.heatmap_context,
                    ctx_mode=args.ctx_mode,
                    target_mode=args.target_mode,
                    mask_fill=args.mask_fill,
                    sensor_drop_mode=args.sensor_drop_mode,
                    sensor_drop_p=p_drop,
                    coverage_levels=coverage_levels,
                    seed=args.eval_seed,
                )
                agg = aggregate_by_theta(rows, ["mae", "nll"])
                mae_mean = float(np.mean(agg["mae"]))
                mae_std = float(np.std(agg["mae"]))
                nll_mean = float(np.mean(agg["nll"]))
                nll_std = float(np.std(agg["nll"]))
                mae_rob_stats[bundle.name]["mean"].append(mae_mean)
                mae_rob_stats[bundle.name]["std"].append(mae_std)
                nll_rob_stats[bundle.name]["mean"].append(nll_mean)
                nll_rob_stats[bundle.name]["std"].append(nll_std)
                rob_rows.append({
                    "drop_p": p_drop,
                    "model": bundle.name,
                    "mae_mean_over_thetas": mae_mean,
                    "mae_std_over_thetas": mae_std,
                    "nll_mean_over_thetas": nll_mean,
                    "nll_std_over_thetas": nll_std,
                })
        write_csv(os.path.join(topo_out, "robustness_curves.csv"), rob_rows, fieldnames=list(rob_rows[0].keys()))
        plot_robustness(drop_probs, mae_rob_stats, f"{topology}: robustness vs sensor dropout", "MAE", os.path.join(topo_out, "robustness_mae.png"))
        plot_robustness(drop_probs, nll_rob_stats, f"{topology}: robustness vs sensor dropout", "NLL", os.path.join(topo_out, "robustness_nll.png"))

        # 5) Qualitative trajectories at fixed context
        qual_dir = os.path.join(topo_out, "qualitative_trajectories")
        ensure_dir(qual_dir)
        qual_base_rows = evaluate_model_over_loader(
            bundle=baseline,
            loader=test_loader,
            y_mean=y_mean,
            y_std=y_std,
            x_means_SP=x_means_SP,
            num_time_points=args.num_time_points,
            num_sensors=args.num_sensors,
            context_frac=args.qualitative_context,
            ctx_mode=args.ctx_mode,
            target_mode=args.target_mode,
            mask_fill=args.mask_fill,
            sensor_drop_mode=args.sensor_drop_mode,
            sensor_drop_p=args.qualitative_drop_p,
            coverage_levels=coverage_levels,
            seed=args.eval_seed,
        )
        qual_rec_rows = evaluate_model_over_loader(
            bundle=recurrent,
            loader=test_loader,
            y_mean=y_mean,
            y_std=y_std,
            x_means_SP=x_means_SP,
            num_time_points=args.num_time_points,
            num_sensors=args.num_sensors,
            context_frac=args.qualitative_context,
            ctx_mode=args.ctx_mode,
            target_mode=args.target_mode,
            mask_fill=args.mask_fill,
            sensor_drop_mode=args.sensor_drop_mode,
            sensor_drop_p=args.qualitative_drop_p,
            coverage_levels=coverage_levels,
            seed=args.eval_seed,
        )
        selected = _pick_qualitative_samples(qual_base_rows, qual_rec_rows, args.qualitative_max_per_topology)
        for theta, sample_idx in selected:
            x_np, y_np = test_data[sample_idx]
            plot_trajectory_comparison(
                x=torch.tensor(x_np, dtype=torch.float32),
                y=torch.tensor(y_np, dtype=torch.float32),
                bundle_base=baseline,
                bundle_rec=recurrent,
                y_mean=y_mean,
                y_std=y_std,
                x_means_SP=x_means_SP,
                num_time_points=args.num_time_points,
                num_sensors=args.num_sensors,
                context_frac=args.qualitative_context,
                ctx_mode=args.ctx_mode,
                target_mode=args.target_mode,
                mask_fill=args.mask_fill,
                sensor_drop_mode=args.sensor_drop_mode,
                sensor_drop_p=args.qualitative_drop_p,
                coverage_level=max(coverage_levels),
                save_path=os.path.join(qual_dir, f"theta_{theta:.3f}_sample_{sample_idx}.png"),
                title=f"{topology} | theta={theta} | sample={sample_idx} | ctx={args.qualitative_context:.0%}",
            )

        # short text summary per topology
        summary_txt = os.path.join(topo_out, "summary.txt")
        best_theta_improv = []
        for theta in theta_order:
            b = model_to_theta_mae["ANP"][theta]
            r = model_to_theta_mae["RANP"][theta]
            best_theta_improv.append((b - r, theta, b, r))
        best_theta_improv.sort(reverse=True)
        with open(summary_txt, "w") as f:
            f.write(f"Topology: {topology}\n")
            f.write(f"Baseline ckpt: {baseline.checkpoint_path}\n")
            f.write(f"Recurrent ckpt: {recurrent.checkpoint_path}\n")
            f.write(f"Heatmap context: {args.heatmap_context}\n")
            f.write(f"Target mode: {args.target_mode}\n\n")
            f.write("Theta-wise MAE (ANP vs RANP)\n")
            f.write("=" * 60 + "\n")
            for theta in theta_order:
                f.write(f"theta={theta:<8} ANP={model_to_theta_mae['ANP'][theta]:.4f} | RANP={model_to_theta_mae['RANP'][theta]:.4f} | delta={model_to_theta_mae['RANP'][theta]-model_to_theta_mae['ANP'][theta]:+.4f}\n")
            f.write("\nBest improvements (ANP - RANP)\n")
            for imp, theta, b, r in best_theta_improv[:5]:
                f.write(f"theta={theta}: improvement={imp:.4f} (ANP {b:.4f} -> RANP {r:.4f})\n")

    if global_summary_rows:
        write_csv(
            os.path.join(args.output_dir, "global_summary_by_topology_theta.csv"),
            global_summary_rows,
            fieldnames=list(global_summary_rows[0].keys()),
        )

    print(f"\nDone. Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
