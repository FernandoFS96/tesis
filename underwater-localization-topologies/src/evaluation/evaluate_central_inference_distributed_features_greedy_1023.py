#!/usr/bin/env python3
"""
Central inference + distributed acquisition/features (node = sensor)

Adds:
  - Greedy sensor pruning (sequential backward elimination) to estimate
    how to retain good performance.

Fixes vs original:
  - Save per-topology tables correctly (sensor_importance / perm_importance)
  - Always save central_inference_distributed_features.csv (even if perm_importance enabled)

Greedy pruning:
  Starts with all sensors, then at each step evaluates removing each remaining sensor,
  removes the least harmful one, repeats until min_sensors remain.
  Evaluations per topology for S=10 -> 3: 10+9+...+4 = 49.

Usage example:
    1. Greedy pruning with min 3 sensors for a total of 49 evaluations:

    python evaluate_central_inference_distributed_features_greedy_1023.py \
  --data_dir /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_low_variance \
  --anp_result_dir /home/fernando/tesis/underwater-localization-topologies/src/training/results/ANP_topologies_masked/masked_dropbernoulli_p0.2_train_mean_first \
  --output_dir /home/fernando/tesis/underwater-localization-topologies/results/eval_masked_central_features \
  --topologies aligned,ellipsoidal,random \
  --context_percent 30 \
  --ctx_sample_mode first \
  --num_time_points 201 \
  --num_sensors 10 \
  --mc_samples 1 \
  --fill_mode train_mean \
  --greedy_prune \
  --greedy_min_sensors 3

  2. Exhaustive evaluation of all subsets of size 1 to 10 (1023 evaluations):
  
  python evaluate_central_inference_distributed_features_greedy_1023.py \
  --data_dir /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_low_variance \
  --anp_result_dir /home/fernando/tesis/underwater-localization-topologies/src/training/results/ANP_topologies_masked/masked_dropbernoulli_p0.2_train_mean_first \
  --output_dir /home/fernando/tesis/underwater-localization-topologies/results/eval_masked_central_features \
  --topologies aligned,ellipsoidal,random \
  --context_percent 30 \
  --ctx_sample_mode first \
  --num_time_points 201 \
  --num_sensors 10 \
  --mc_samples 1 \
  --fill_mode train_mean \
  --exhaustive_subsets \
  --subset_min_k 1 --subset_max_k 10

"""

import argparse
import math
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import itertools

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.models.anp import LatentModel


# --------------------------
# Timing
# --------------------------
class Timer:
    def __init__(self, device: torch.device):
        self.device = device
        self.t0: Optional[float] = None
        self.dt = 0.0

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        if self.t0 is None:
            raise RuntimeError("Timer used without being started.")
        self.dt = time.perf_counter() - self.t0


# --------------------------
# Mask utilities (for masked ANP)
# --------------------------
def make_sensor_mask_feat(T: int, num_sensors: int, active_sensors: List[int], device: torch.device) -> torch.Tensor:
    """
    mask_feat: (1, T, S) float32, mask[s]=1 if sensor s active
    """
    m = torch.zeros((1, num_sensors), dtype=torch.float32, device=device)
    m[0, active_sensors] = 1.0
    return m.view(1, 1, num_sensors).expand(1, T, num_sensors)

def append_sensor_mask(x: torch.Tensor, mask_feat: torch.Tensor) -> torch.Tensor:
    """
    x: (1,T,Dx), mask_feat: (1,T,S)  -> (1,T,Dx+S)
    """
    return torch.cat([x, mask_feat], dim=-1)


# --------------------------
# Data loading
# --------------------------
def load_topology_test(data_dir: Path, topology: str):
    topo_dir = data_dir / f"topology_{topology}"
    test_path = topo_dir / "test_data.pkl"
    meta_path = topo_dir / "metadata.pkl"
    with open(test_path, "rb") as f:
        test_data = pickle.load(f)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    theta_groups: Dict[float, List] = {}
    for sample, theta in zip(test_data, meta["test_thetas"]):
        theta_groups.setdefault(theta, []).append(sample)
    return theta_groups, sorted(theta_groups.keys())

def load_y_stats(data_dir: Path, topology: str, device: torch.device):
    topo_dir = data_dir / f"topology_{topology}"
    train_path = topo_dir / "train_data.pkl"
    with open(train_path, "rb") as f:
        train_data = pickle.load(f)
    Y = np.concatenate([y for _, y in train_data], axis=0)
    y_mean = torch.tensor(Y.mean(axis=0), dtype=torch.float32, device=device)
    y_std = torch.tensor(Y.std(axis=0) + 1e-6, dtype=torch.float32, device=device)
    return y_mean, y_std

def load_x_sensor_means(data_dir: Path, topology: str, num_time_points: int, num_sensors: int, device: torch.device):
    """
    Returns list length S, each tensor shape (P,) = mean over train for that sensor block.
    Assumes interleaved layout: sensor s is X_flat[:, s::S] -> (N*T, P)
    """
    topo_dir = data_dir / f"topology_{topology}"
    train_path = topo_dir / "train_data.pkl"
    with open(train_path, "rb") as f:
        train_data = pickle.load(f)

    X = np.concatenate([x for x, _ in train_data], axis=0)  # (N*T, Dx)
    Dx = X.shape[1]
    assert Dx == num_time_points * num_sensors, (Dx, num_time_points * num_sensors)

    X3 = X.reshape(X.shape[0], num_time_points, num_sensors)  # (N*T, P, S)
    mean_PS = X3.mean(axis=0)  # (P,S)
    x_means = [torch.tensor(mean_PS[:, s], dtype=torch.float32, device=device) for s in range(num_sensors)]
    return x_means


# --------------------------
# Normalization
# --------------------------
def normalize_y(y, y_mean, y_std):
    return (y - y_mean.view(1, 1, -1)) / y_std.view(1, 1, -1)

def denormalize_y(y_norm, y_mean, y_std):
    return y_norm * y_std.view(1, 1, -1) + y_mean.view(1, 1, -1)


# --------------------------
# Context sampling
# --------------------------
def sample_context_indices(T: int, n_ctx: int, mode: str, gen: torch.Generator, device: torch.device):
    n_ctx = max(1, min(n_ctx, T))
    if mode == "first":
        return torch.arange(n_ctx, device=device)
    if mode == "random":
        perm = torch.randperm(T, generator=gen, device=device)
        return perm[:n_ctx].sort().values
    raise ValueError(mode)


# --------------------------
# Model loading
# --------------------------
def load_model(anp_result_dir: Path, topology: str, input_dim: int, output_dim: int, device: torch.device):
    # masked training layout
    ckpt_path_masked = anp_result_dir / f"topology_{topology}" / "best_checkpoint.pth.tar"
    # legacy layout
    ckpt_path_legacy = anp_result_dir / f"ANP_{topology}" / "best_checkpoint.pth.tar"

    ckpt_path = ckpt_path_masked if ckpt_path_masked.exists() else ckpt_path_legacy
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found. Tried:\n  {ckpt_path_masked}\n  {ckpt_path_legacy}")

    model = LatentModel(num_hidden=128, input_dim=input_dim, output_dim=output_dim)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    return model.to(device).eval()


# --------------------------
# ANP predict helper
# --------------------------
def anp_predict(model, cx, cy, tx, mc_samples: int, device: torch.device):
    means, vars_ = [], []
    with Timer(device) as tmr:
        for _ in range(max(1, mc_samples)):
            out = model(cx, cy, tx)
            # robust unpack
            if isinstance(out, (tuple, list)):
                if len(out) >= 2 and torch.is_tensor(out[0]) and torch.is_tensor(out[1]):
                    m, v = out[0], out[1]
                elif len(out) >= 1 and isinstance(out[0], (tuple, list)) and len(out[0]) >= 2:
                    m, v = out[0][0], out[0][1]
                else:
                    raise TypeError(f"Unexpected model output tuple structure: {type(out)} len={len(out)}")
            elif isinstance(out, dict):
                m, v = out["mean"], out["var"]
            else:
                raise TypeError(f"Unexpected model output type: {type(out)}")
            means.append(m); vars_.append(v)
    mean = torch.stack(means, dim=0).mean(dim=0)
    var = torch.stack(vars_, dim=0).mean(dim=0)
    return mean, var, tmr.dt


# --------------------------
# Sensor feature block helpers
# --------------------------
def extract_sensor_features(x_full: torch.Tensor, s: int, S: int) -> torch.Tensor:
    """
    x_full: (1,T,Dx) interleaved layout -> local block (1,T,P) = x_full[..., s::S]
    """
    return x_full[..., s::S]

def build_x_parts_with_mask(
    x_full: torch.Tensor,                 # (1,T,Dx)
    active_sensors: List[int],
    num_time_points: int,
    num_sensors: int,
    device: torch.device,
    fill_mode: str,
    x_means: Optional[List[torch.Tensor]],  # list of (P,)
) -> List[torch.Tensor]:
    """
    Returns x_parts list length S with shape (1,T,P) each.
    For inactive sensors:
      - fill_mode='train_mean': fill with per-sensor mean over train
      - fill_mode='zero': fill with zeros
    """
    T = x_full.shape[1]
    parts = []
    active = set(active_sensors)

    for s in range(num_sensors):
        if s in active:
            parts.append(extract_sensor_features(x_full, s, num_sensors))  # (1,T,P)
        else:
            if fill_mode == "zero":
                parts.append(torch.zeros((1, T, num_time_points), dtype=torch.float32, device=device))
            elif fill_mode == "train_mean":
                assert x_means is not None
                mu = x_means[s].view(1, 1, num_time_points).expand(1, T, num_time_points)
                parts.append(mu.clone())
            else:
                raise ValueError(fill_mode)

    return parts

def reconstruct_x_from_sensors(x_parts: List[torch.Tensor], S: int) -> torch.Tensor:
    """
    x_parts[s]: (1,T,P), s=0..S-1 -> reconstruct x_full (1,T,P*S) interleaved
    """
    X = torch.stack(x_parts, dim=0)          # (S,1,T,P)
    X = X.permute(1, 2, 3, 0).contiguous()   # (1,T,P,S)
    return X.view(X.shape[0], X.shape[1], -1)  # (1,T,P*S)

def permute_sensor_block_time(x_block: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """
    x_block: (1, T, P) -> shuffle along time dimension T
    """
    T = x_block.shape[1]
    perm = torch.randperm(T, generator=gen, device=x_block.device)
    return x_block[:, perm, :].contiguous()


# --------------------------
# Greedy pruning (NEW)
# --------------------------
@torch.no_grad()
def mae_for_active_sensors(
    model: torch.nn.Module,
    x_full: torch.Tensor,   # (1,T,Dx)
    y: torch.Tensor,        # (1,T,Dy) original
    y_norm: torch.Tensor,   # (1,T,Dy) normalized
    ctx_idx: torch.Tensor,  # (C,)
    non_ctx_mask: torch.Tensor,  # (T,) bool
    active_sensors: List[int],
    num_time_points: int,
    num_sensors: int,
    device: torch.device,
    fill_mode: str,
    x_means: Optional[List[torch.Tensor]],
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    mc_samples: int,
) -> float:
    """
    Builds x for given active sensor subset (missing sensors filled),
    appends explicit mask, runs ANP, returns MAE on non-context points.
    """
    T = x_full.shape[1]

    # Build masked x parts
    x_parts = build_x_parts_with_mask(
        x_full=x_full,
        active_sensors=active_sensors,
        num_time_points=num_time_points,
        num_sensors=num_sensors,
        device=device,
        fill_mode=fill_mode,
        x_means=x_means,
    )
    x_sub = reconstruct_x_from_sensors(x_parts, num_sensors)  # (1,T,Dx)

    # Append explicit sensor mask (Dx+S)
    mask_feat = make_sensor_mask_feat(T, num_sensors, active_sensors, device)
    x_sub_aug = append_sensor_mask(x_sub, mask_feat)

    cx = x_sub_aug[:, ctx_idx, :]
    cy = y_norm[:, ctx_idx, :]
    mean_norm, _, _ = anp_predict(model, cx, cy, x_sub_aug, mc_samples, device)
    pred = denormalize_y(mean_norm, y_mean, y_std)

    mae = F.l1_loss(pred[:, non_ctx_mask, :], y[:, non_ctx_mask, :], reduction="none").mean().item()
    return float(mae)


@torch.no_grad()
def mae_for_active_sensors_batch(
    model: torch.nn.Module,
    batch_items: List[Dict[str, Any]],
    active_sensors: List[int],
    num_time_points: int,
    num_sensors: int,
    device: torch.device,
    fill_mode: str,
    x_means: Optional[List[torch.Tensor]],
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    mc_samples: int,
    batch_size: int,
) -> torch.Tensor:
    """
    Batched version of mae_for_active_sensors. Returns per-trajectory MAE tensor (B,).
    """
    if not batch_items:
        return torch.empty((0,), device=device)

    if batch_size is None or batch_size <= 0:
        batch_size = len(batch_items)

    T = batch_items[0]["x_full"].shape[1]
    mask_feat = make_sensor_mask_feat(T, num_sensors, active_sensors, device)

    mae_chunks = []
    for start in range(0, len(batch_items), batch_size):
        chunk = batch_items[start:start + batch_size]

        x_sub_list = []
        y_list = []
        y_norm_list = []
        ctx_idx_list = []
        non_ctx_mask_list = []

        for item in chunk:
            x_full = item["x_full"]

            x_parts = build_x_parts_with_mask(
                x_full=x_full,
                active_sensors=active_sensors,
                num_time_points=num_time_points,
                num_sensors=num_sensors,
                device=device,
                fill_mode=fill_mode,
                x_means=x_means,
            )
            x_sub = reconstruct_x_from_sensors(x_parts, num_sensors)
            x_sub_aug = append_sensor_mask(x_sub, mask_feat)

            x_sub_list.append(x_sub_aug)
            y_list.append(item["y"])
            y_norm_list.append(item["y_norm"])
            ctx_idx_list.append(item["ctx_idx"])
            non_ctx_mask_list.append(item["non_ctx_mask"])

        x_sub_batch = torch.cat(x_sub_list, dim=0)
        y_batch = torch.cat(y_list, dim=0)
        y_norm_batch = torch.cat(y_norm_list, dim=0)
        ctx_idx_batch = torch.stack(ctx_idx_list, dim=0)
        non_ctx_mask_batch = torch.stack(non_ctx_mask_list, dim=0)

        batch_idx = torch.arange(x_sub_batch.shape[0], device=device).unsqueeze(1)
        cx = x_sub_batch[batch_idx, ctx_idx_batch, :]
        cy = y_norm_batch[batch_idx, ctx_idx_batch, :]

        mean_norm, _, _ = anp_predict(model, cx, cy, x_sub_batch, mc_samples, device)
        pred = denormalize_y(mean_norm, y_mean, y_std)

        abs_err = torch.abs(pred - y_batch)
        mask = non_ctx_mask_batch.unsqueeze(-1)
        abs_err = abs_err * mask

        denom = mask.sum(dim=1).float() * y_batch.shape[2]
        denom = denom.clamp(min=1.0)
        mae_per = abs_err.sum(dim=(1, 2)) / denom
        mae_chunks.append(mae_per)

    return torch.cat(mae_chunks, dim=0)

def greedy_prune_report(
    model: torch.nn.Module,
    traj_cache: List[Dict[str, Any]],
    num_sensors: int,
    min_sensors: int,
    num_time_points: int,
    device: torch.device,
    fill_mode: str,
    x_means: Optional[List[torch.Tensor]],
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    mc_samples: int,
    eval_batch_size: int,
) -> List[Dict[str, Any]]:
    """
    Greedy sequential backward elimination. Returns history list.
    Each history step includes: active set, removed sensor, mae, candidates list.
    """
    active = list(range(num_sensors))
    history: List[Dict[str, Any]] = []

    def eval_set(active_set: List[int]) -> float:
        if not traj_cache:
            return float("nan")
        mae_vals = mae_for_active_sensors_batch(
            model=model,
            batch_items=traj_cache,
            active_sensors=active_set,
            num_time_points=num_time_points,
            num_sensors=num_sensors,
            device=device,
            fill_mode=fill_mode,
            x_means=x_means,
            y_mean=y_mean,
            y_std=y_std,
            mc_samples=mc_samples,
            batch_size=eval_batch_size,
        )
        return float(mae_vals.mean().item()) if len(mae_vals) else float("nan")

    # baseline
    mae_cur = eval_set(active)
    history.append({
        "step": 0,
        "n_sensors": len(active),
        "active": active.copy(),
        "removed": None,
        "mae": mae_cur,
        "candidates": None,
    })

    step = 1
    while len(active) > min_sensors:
        candidates = []
        for s_remove in active:
            cand_active = [s for s in active if s != s_remove]
            mae_cand = eval_set(cand_active)
            candidates.append((int(s_remove), float(mae_cand), float(mae_cand - mae_cur)))

        # remove least harmful (min delta). Tie-breaker: min mae.
        candidates.sort(key=lambda x: (x[2], x[1]))
        best_remove, best_mae, best_delta = candidates[0]

        # commit removal
        active = [s for s in active if s != best_remove]
        mae_cur = best_mae

        history.append({
            "step": step,
            "n_sensors": len(active),
            "active": active.copy(),
            "removed": int(best_remove),
            "mae": float(mae_cur),
            "candidates": candidates,  # list of tuples
        })
        step += 1

    return history

def write_greedy_txt(
    out_path: Path,
    topology: str,
    ctx_percent: int,
    ctx_sample_mode: str,
    mc_samples: int,
    fill_mode: str,
    min_sensors: int,
    history: List[Dict[str, Any]],
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("Greedy sensor pruning (sequential backward elimination)\n")
        f.write("=" * 90 + "\n")
        f.write(f"topology: {topology}\n")
        f.write(f"context_percent: {ctx_percent}\n")
        f.write(f"ctx_sample_mode: {ctx_sample_mode}\n")
        f.write(f"mc_samples: {mc_samples}\n")
        f.write(f"fill_mode: {fill_mode}\n")
        f.write(f"stop_at_min_sensors: {min_sensors}\n")
        f.write("=" * 90 + "\n\n")

        for h in history:
            f.write(f"STEP {h['step']:02d} | sensors={h['n_sensors']:02d} | MAE={h['mae']:.4f}\n")
            f.write(f"  active: {h['active']}\n")
            if h["removed"] is not None:
                f.write(f"  removed: {h['removed']}\n")
                f.write("  candidates (remove -> MAE, delta vs prev):\n")
                for (s_remove, mae_cand, delta) in h["candidates"]:
                    f.write(f"    - remove {s_remove:2d}: MAE={mae_cand:.4f}, delta={delta:+.4f}\n")
            f.write("\n")


# --------------------------
# Exhaustive subset eval (NEW)
# --------------------------
@torch.no_grad()
def exhaustive_subsets_report(
    model: torch.nn.Module,
    traj_cache: List[Dict[str, Any]],
    num_sensors: int,
    num_time_points: int,
    device: torch.device,
    fill_mode: str,
    x_means: Optional[List[torch.Tensor]],
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    mc_samples: int,
    eval_batch_size: int,
    min_k: int = 1,
    max_k: Optional[int] = None,
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Evaluates every non-empty sensor subset (optionally filtered by size).
    Returns:
      - mae_all: mean MAE using all sensors
      - rows: list of dicts with subset MAE and metadata

    NOTE: This is O((2^S-1) * N_traj) model forward passes.
    """
    if max_k is None:
        max_k = num_sensors
    min_k = max(1, min(min_k, num_sensors))
    max_k = max(1, min(max_k, num_sensors))
    if min_k > max_k:
        raise ValueError("min_k cannot be > max_k")

    def eval_set(active_set: List[int]) -> float:
        if not traj_cache:
            return float("nan")
        mae_vals = mae_for_active_sensors_batch(
            model=model,
            batch_items=traj_cache,
            active_sensors=active_set,
            num_time_points=num_time_points,
            num_sensors=num_sensors,
            device=device,
            fill_mode=fill_mode,
            x_means=x_means,
            y_mean=y_mean,
            y_std=y_std,
            mc_samples=mc_samples,
            batch_size=eval_batch_size,
        )
        return float(mae_vals.mean().item()) if len(mae_vals) else float("nan")

    active_all = list(range(num_sensors))
    mae_all = eval_set(active_all)

    rows: List[Dict[str, Any]] = []
    total_subsets = sum(math.comb(num_sensors, k) for k in range(min_k, max_k + 1))
    with tqdm(total=total_subsets, desc="exhaustive subsets", unit="subset", leave=True) as pbar:
        for k in range(min_k, max_k + 1):
            for comb in itertools.combinations(range(num_sensors), k):
                active = list(comb)
                mask = 0
                for s in active:
                    mask |= (1 << int(s))
                mae = eval_set(active)
                rows.append({
                    "subset_size": int(k),
                    "active": active,
                    "bitmask": int(mask),
                    "mae": float(mae),
                    "delta_vs_all": round(float(mae - mae_all), 4),
                })
                pbar.update(1)

    return float(mae_all), rows


def plot_exhaustive_mae_boxplot(
    rows_sub: List[Dict[str, Any]],
    out_path: Path,
    topology: str,
    ctx_percent: int,
    ctx_sample_mode: str,
    mc_samples: int,
    fill_mode: str,
) -> None:
    """
    Save a boxplot of MAE values grouped by subset size (number of sensors).
    """
    if not rows_sub:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped: Dict[int, List[float]] = {}
    for row in rows_sub:
        k = int(row["subset_size"])
        grouped.setdefault(k, []).append(float(row["mae"]))

    ks = sorted(grouped.keys())
    data = [grouped[k] for k in ks]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, tick_labels=[str(k) for k in ks], showfliers=True)
    ax.set_xlabel("Number of sensors (subset size)")
    ax.set_ylabel("MAE")
    ax.set_title(
        f"MAE distribution by subset size | {topology} | ctx={ctx_percent}% {ctx_sample_mode} | mc={mc_samples} | fill={fill_mode}"
    )
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def round_mae_columns(df):
    for col in df.columns:
        if "mae" in col:
            try:
                df[col] = df[col].round(4)
            except Exception:
                pass
    return df


# --------------------------
# Main
# --------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--anp_result_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--topologies", type=str, default="aligned,ellipsoidal,random")

    p.add_argument("--context_percent", type=int, default=30)
    p.add_argument("--ctx_sample_mode", type=str, default="first", choices=["first", "random"])

    p.add_argument("--num_time_points", type=int, default=201)
    p.add_argument("--num_sensors", type=int, default=10)

    p.add_argument("--mc_samples", type=int, default=1)
    p.add_argument("--seed_eval", type=int, default=0)
    p.add_argument("--max_traj_per_theta", type=int, default=-1)

    p.add_argument("--sensor_importance", action="store_true", help="Run LOSO+LISO sensor ablation and save per-sensor MAE table.")
    p.add_argument("--fill_mode", type=str, default="train_mean", choices=["train_mean", "zero"], help="How to fill missing sensors for LOSO/LISO/greedy/exhaustive.")

    p.add_argument("--perm_importance", action="store_true", help="Run permutation feature importance (PFI) per sensor and save table.")
    p.add_argument("--perm_repeats", type=int, default=5, help="Number of permutations per sensor.")
    p.add_argument("--perm_mode", type=str, default="time", choices=["time"], help="Permutation mode.")
    p.add_argument("--perm_seed", type=int, default=0, help="Base seed for permutation importance.")

    # batch size for greedy/exhaustive eval.
    p.add_argument("--eval_batch_size", type=int, default=0,
                   help="Batch size for MAE eval in greedy/exhaustive. 0 means all trajectories.")

    # greedy pruning
    p.add_argument("--greedy_prune", action="store_true",
                   help="Run greedy sensor pruning (10->...->min) and save .txt report per topology.")
    p.add_argument("--greedy_min_sensors", type=int, default=3,
                   help="Stop greedy pruning when this many sensors remain (default 3).")
    p.add_argument("--greedy_out_prefix", type=str, default="greedy_pruning",
                   help="Prefix for greedy report txt files.")

    # Ehaustive evaluation of all sensor subsets
    p.add_argument("--exhaustive_subsets", action="store_true",
                   help="Evaluate all non-empty sensor subsets (2^S-1) and save a per-topology CSV.")
    p.add_argument("--exhaustive_out_prefix", type=str, default="exhaustive_subsets",
                   help="Prefix for exhaustive subset CSV files.")
    p.add_argument("--subset_min_k", type=int, default=1,
                   help="Minimum subset size k to evaluate (default 1).")
    p.add_argument("--subset_max_k", type=int, default=None,
                   help="Maximum subset size k to evaluate (default: num_sensors).")

    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed_eval)

    topologies = [t.strip() for t in args.topologies.split(",") if t.strip()]
    rows_main = []  # central comparison rows

    for topo in topologies:
        theta_groups, theta_values = load_topology_test(args.data_dir, topo)

        total = sum(len(theta_groups[t]) for t in theta_values)
        print("TEST trajectories total =", total)
        print("Per-theta counts:", {t: len(theta_groups[t]) for t in theta_values})

        x0, y0 = theta_groups[theta_values[0]][0]
        T = x0.shape[0]
        Dx = x0.shape[1]
        Dy = y0.shape[1]

        expected = args.num_time_points * args.num_sensors
        if Dx != expected:
            raise ValueError(f"[{topo}] Dx={Dx} != {expected} (num_time_points*num_sensors)")

        # masked ANP expects Dx+S
        input_dim_model = Dx + args.num_sensors
        model = load_model(args.anp_result_dir, topo, input_dim_model, Dy, device)
        y_mean, y_std = load_y_stats(args.data_dir, topo, device)

        # Means per sensor for filling missing sensors
        x_means = None
        if (args.sensor_importance or args.greedy_prune or args.exhaustive_subsets) and args.fill_mode == "train_mean":
            x_means = load_x_sensor_means(args.data_dir, topo, args.num_time_points, args.num_sensors, device)

        n_ctx = max(1, min(int((args.context_percent / 100.0) * T), T))

        # per-topology accumulators
        total_mae_direct = []
        total_mae_recon = []
        recon_errs = []
        total_time_direct = 0.0
        total_time_recon = 0.0

        mae_all_list = []  # baseline all-sensors MAE per trajectory (for LOSO/LISO/PFI summary)
        pfi_delta_lists = {s: [] for s in range(args.num_sensors)}
        pfi_mae_lists   = {s: [] for s in range(args.num_sensors)}
        loso_mae_lists  = {s: [] for s in range(args.num_sensors)}
        liso_mae_lists  = {s: [] for s in range(args.num_sensors)}

        # cache trajectories for greedy mode (only needed if greedy_prune)
        traj_cache: List[Dict[str, Any]] = []

        # comm calculation
        comm_features_floats_per_traj = args.num_sensors * T * args.num_time_points
        comm_outputs_floats_per_traj = (args.num_sensors - 1) * 2 * T * Dy

        # evaluate
        for theta in theta_values:
            samples = theta_groups[theta]
            if args.max_traj_per_theta > 0:
                samples = samples[:args.max_traj_per_theta]

            for (x_np, y_np) in tqdm(samples, desc=f"{topo} theta={theta:.3f}", leave=False):
                x_full = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(0)  # (1,T,Dx)
                y = torch.tensor(y_np, dtype=torch.float32, device=device).unsqueeze(0)       # (1,T,Dy)
                y_norm = normalize_y(y, y_mean, y_std)

                # distributed acquisition -> reconstruct
                x_parts = [extract_sensor_features(x_full, s, args.num_sensors) for s in range(args.num_sensors)]
                x_full_recon = reconstruct_x_from_sensors(x_parts, args.num_sensors)

                recon_err = torch.max(torch.abs(x_full - x_full_recon)).item()
                if recon_err > 1e-4:
                    raise ValueError(f"Reconstruction error too high: {recon_err}")

                # context selection
                ctx_idx = sample_context_indices(T, n_ctx, args.ctx_sample_mode, gen, device)
                non_ctx_mask = torch.ones(T, dtype=torch.bool, device=device)
                non_ctx_mask[ctx_idx] = False

                active_all = list(range(args.num_sensors))
                mask_all = make_sensor_mask_feat(T, args.num_sensors, active_all, device)
                x_full_aug = append_sensor_mask(x_full, mask_all)
                x_full_recon_aug = append_sensor_mask(x_full_recon, mask_all)

                # ---- A) Central direct ----
                cx0 = x_full_aug[:, ctx_idx, :]
                cy0 = y_norm[:, ctx_idx, :]
                mean0, _, dt0 = anp_predict(model, cx0, cy0, x_full_aug, args.mc_samples, device)
                pred0 = denormalize_y(mean0, y_mean, y_std)
                mae0 = F.l1_loss(pred0[:, non_ctx_mask, :], y[:, non_ctx_mask, :], reduction="none").mean().item()

                # ---- B) Central from recon(features) ----
                cx1 = x_full_recon_aug[:, ctx_idx, :]
                cy1 = y_norm[:, ctx_idx, :]
                mean1, _, dt1 = anp_predict(model, cx1, cy1, x_full_recon_aug, args.mc_samples, device)
                pred1 = denormalize_y(mean1, y_mean, y_std)
                mae1 = F.l1_loss(pred1[:, non_ctx_mask, :], y[:, non_ctx_mask, :], reduction="none").mean().item()

                total_mae_direct.append(mae0)
                total_mae_recon.append(mae1)
                total_time_direct += dt0
                total_time_recon += dt1
                recon_errs.append(recon_err)

                # baseline all sensors (same as recon, but keep explicit for consistency)
                x_parts_all = [extract_sensor_features(x_full, s, args.num_sensors) for s in range(args.num_sensors)]
                x_all = reconstruct_x_from_sensors(x_parts_all, args.num_sensors)
                x_all_aug = append_sensor_mask(x_all, mask_all)

                cx = x_all_aug[:, ctx_idx, :]
                cy = y_norm[:, ctx_idx, :]
                mean_norm, _, _ = anp_predict(model, cx, cy, x_all_aug, args.mc_samples, device)
                pred = denormalize_y(mean_norm, y_mean, y_std)
                mae_all = F.l1_loss(pred[:, non_ctx_mask, :], y[:, non_ctx_mask, :], reduction="none").mean().item()
                mae_all_list.append(mae_all)

                # cache for greedy pruning (store minimal info)
                if args.greedy_prune or args.exhaustive_subsets:
                    traj_cache.append({
                        "x_full": x_full.detach(),
                        "y": y.detach(),
                        "y_norm": y_norm.detach(),
                        "ctx_idx": ctx_idx.detach(),
                        "non_ctx_mask": non_ctx_mask.detach(),
                    })

                # LOSO/LISO
                if args.sensor_importance:
                    for s in range(args.num_sensors):
                        # LOSO: all except s
                        active_loso = [k for k in range(args.num_sensors) if k != s]
                        x_parts_loso = build_x_parts_with_mask(
                            x_full, active_loso,
                            args.num_time_points, args.num_sensors,
                            device, args.fill_mode, x_means
                        )
                        x_loso = reconstruct_x_from_sensors(x_parts_loso, args.num_sensors)
                        mask_loso = make_sensor_mask_feat(T, args.num_sensors, active_loso, device)
                        x_loso_aug = append_sensor_mask(x_loso, mask_loso)

                        cx_loso = x_loso_aug[:, ctx_idx, :]
                        cy_loso = y_norm[:, ctx_idx, :]
                        m_loso, _, _ = anp_predict(model, cx_loso, cy_loso, x_loso_aug, args.mc_samples, device)
                        pred_loso = denormalize_y(m_loso, y_mean, y_std)
                        mae_loso = F.l1_loss(pred_loso[:, non_ctx_mask, :], y[:, non_ctx_mask, :], reduction="none").mean().item()
                        loso_mae_lists[s].append(mae_loso)

                        # LISO: only s
                        active_liso = [s]
                        x_parts_liso = build_x_parts_with_mask(
                            x_full, active_liso,
                            args.num_time_points, args.num_sensors,
                            device, args.fill_mode, x_means
                        )
                        x_liso = reconstruct_x_from_sensors(x_parts_liso, args.num_sensors)
                        mask_liso = make_sensor_mask_feat(T, args.num_sensors, active_liso, device)
                        x_liso_aug = append_sensor_mask(x_liso, mask_liso)

                        cx_liso = x_liso_aug[:, ctx_idx, :]
                        cy_liso = y_norm[:, ctx_idx, :]
                        m_liso, _, _ = anp_predict(model, cx_liso, cy_liso, x_liso_aug, args.mc_samples, device)
                        pred_liso = denormalize_y(m_liso, y_mean, y_std)
                        mae_liso = F.l1_loss(pred_liso[:, non_ctx_mask, :], y[:, non_ctx_mask, :], reduction="none").mean().item()
                        liso_mae_lists[s].append(mae_liso)

                # Permutation importance
                if args.perm_importance:
                    for s in range(args.num_sensors):
                        for r in range(args.perm_repeats):
                            x_parts_perm = [p.clone() for p in x_parts_all]
                            gen_sr = torch.Generator(device=device)
                            gen_sr.manual_seed(args.perm_seed + 1000 * s + 10 * r)
                            x_parts_perm[s] = permute_sensor_block_time(x_parts_perm[s], gen_sr)

                            x_perm = reconstruct_x_from_sensors(x_parts_perm, args.num_sensors)
                            x_perm_aug = append_sensor_mask(x_perm, mask_all)

                            cx_p = x_perm_aug[:, ctx_idx, :]
                            cy_p = y_norm[:, ctx_idx, :]
                            m_p, _, _ = anp_predict(model, cx_p, cy_p, x_perm_aug, args.mc_samples, device)
                            pred_p = denormalize_y(m_p, y_mean, y_std)
                            mae_p = F.l1_loss(pred_p[:, non_ctx_mask, :], y[:, non_ctx_mask, :], reduction="none").mean().item()

                            pfi_mae_lists[s].append(mae_p)
                            pfi_delta_lists[s].append(mae_p - mae_all)

        # ----- summary -----
        mae_direct = float(np.mean(total_mae_direct)) if total_mae_direct else float("nan")
        mae_recon  = float(np.mean(total_mae_recon))  if total_mae_recon  else float("nan")
        recon_max  = float(np.max(recon_errs)) if recon_errs else float("nan")

        n_traj = len(total_mae_direct)
        comm_features_kib = (comm_features_floats_per_traj * 4 * n_traj) / 1024.0
        comm_outputs_kib = (comm_outputs_floats_per_traj * 4 * n_traj) / 1024.0

        print("\n" + "=" * 85)
        print(f"Topology {topo} | ctx={args.context_percent}% ({args.ctx_sample_mode}) | central inference comparison")
        print(f"  Centralized direct           MAE={mae_direct:8.4f} time(serial)={total_time_direct:7.3f}s")
        print(f"  Central from recon(features) MAE={mae_recon:8.4f}  time(serial)={total_time_recon:7.3f}s")
        print(f"  Recon max|x-x_recon| = {recon_max:.3e}")
        print(f"  comm(features)={comm_features_kib:,.1f} KiB comm(outputs mu,var)={comm_outputs_kib:,.1f} KiB")
        print("=" * 85)

        rows_main.append({
            "topology": topo,
            "context_percent": args.context_percent,
            "ctx_sample_mode": args.ctx_sample_mode,
            "mc_samples": args.mc_samples,
            "mae_centralized_direct": mae_direct,
            "mae_central_from_recon": mae_recon,
            "time_direct_total_s": total_time_direct,
            "time_recon_total_s": total_time_recon,
            "recon_max_abs_diff": recon_max,
            "n_trajectories": n_traj,
            "comm_features_total_kib": comm_features_kib,
            "comm_outputs_total_kib": comm_outputs_kib,
            "num_time_points": args.num_time_points,
            "num_sensors": args.num_sensors,
        })

        # ----- Save per-topology sensor importance -----
        import pandas as pd

        if args.sensor_importance:
            mae_all_mean = float(np.mean(mae_all_list)) if mae_all_list else float("nan")
            table_rows = []
            for s in range(args.num_sensors):
                mae_loso = float(np.mean(loso_mae_lists[s])) if loso_mae_lists[s] else float("nan")
                mae_liso = float(np.mean(liso_mae_lists[s])) if liso_mae_lists[s] else float("nan")
                table_rows.append({
                    "topology": topo,
                    "sensor": s,
                    "fill_mode": args.fill_mode,
                    "ctx_percent": args.context_percent,
                    "ctx_sample_mode": args.ctx_sample_mode,
                    "mc_samples": args.mc_samples,
                    "mae_all": mae_all_mean,
                    "mae_loso": mae_loso,
                    "delta_mae_loso": mae_loso - mae_all_mean,
                    "mae_liso": mae_liso,
                })
            df_imp = pd.DataFrame(table_rows)
            df_imp = round_mae_columns(df_imp)
            out_csv = outdir / f"sensor_importance_topology_{topo}.csv"
            df_imp.to_csv(out_csv, index=False)
            print(f"\nSaved sensor importance table: {out_csv}\n")

        # ----- Save per-topology PFI -----
        if args.perm_importance:
            mae_all_mean = float(np.mean(mae_all_list)) if mae_all_list else float("nan")
            rows_pfi = []
            for s in range(args.num_sensors):
                deltas = np.array(pfi_delta_lists[s], dtype=float)
                maes   = np.array(pfi_mae_lists[s], dtype=float)
                rows_pfi.append({
                    "topology": topo,
                    "sensor": s,
                    "ctx_percent": args.context_percent,
                    "ctx_sample_mode": args.ctx_sample_mode,
                    "mc_samples": args.mc_samples,
                    "perm_repeats": args.perm_repeats,
                    "perm_mode": args.perm_mode,
                    "mae_all": mae_all_mean,
                    "mae_perm_mean": float(np.mean(maes)) if len(maes) else float("nan"),
                    "delta_mae_mean": float(np.mean(deltas)) if len(deltas) else float("nan"),
                    "delta_mae_std": float(np.std(deltas)) if len(deltas) else float("nan"),
                    "n_eval": int(len(deltas)),
                })
            df_pfi = pd.DataFrame(rows_pfi)
            df_pfi = round_mae_columns(df_pfi)
            out_csv = outdir / f"sensor_permutation_importance_topology_{topo}.csv"
            df_pfi.to_csv(out_csv, index=False)
            print(f"\nSaved PFI table: {out_csv}\n")

        # ----- NEW: Greedy pruning report -----
        if args.greedy_prune:
            if args.greedy_min_sensors < 1 or args.greedy_min_sensors > args.num_sensors:
                raise ValueError("--greedy_min_sensors must be in [1, num_sensors].")

            history = greedy_prune_report(
                model=model,
                traj_cache=traj_cache,
                num_sensors=args.num_sensors,
                min_sensors=args.greedy_min_sensors,
                num_time_points=args.num_time_points,
                device=device,
                fill_mode=args.fill_mode,
                x_means=x_means,
                y_mean=y_mean,
                y_std=y_std,
                mc_samples=args.mc_samples,
                eval_batch_size=args.eval_batch_size,
            )

            out_txt = outdir / f"{args.greedy_out_prefix}_topology_{topo}.txt"
            write_greedy_txt(
                out_path=out_txt,
                topology=topo,
                ctx_percent=args.context_percent,
                ctx_sample_mode=args.ctx_sample_mode,
                mc_samples=args.mc_samples,
                fill_mode=args.fill_mode,
                min_sensors=args.greedy_min_sensors,
                history=history,
            )
            print(f"[greedy_prune] Saved report: {out_txt}")

        # ----- NEW: Exhaustive subset evaluation (2^S-1) -----
        if args.exhaustive_subsets:
            t_exh0 = time.perf_counter()
            mae_all_exh, rows_sub = exhaustive_subsets_report(
                model=model,
                traj_cache=traj_cache,
                num_sensors=args.num_sensors,
                num_time_points=args.num_time_points,
                device=device,
                fill_mode=args.fill_mode,
                x_means=x_means,
                y_mean=y_mean,
                y_std=y_std,
                mc_samples=args.mc_samples,
                eval_batch_size=args.eval_batch_size,
                min_k=args.subset_min_k,
                max_k=args.subset_max_k,
            )
            dt_exh = time.perf_counter() - t_exh0

            df_sub = pd.DataFrame(rows_sub)
            if len(df_sub):
                df_sub.insert(0, "topology", topo)
                df_sub.insert(1, "fill_mode", args.fill_mode)
                df_sub.insert(2, "ctx_percent", args.context_percent)
                df_sub.insert(3, "ctx_sample_mode", args.ctx_sample_mode)
                df_sub.insert(4, "mc_samples", args.mc_samples)
                df_sub["active"] = df_sub["active"].apply(lambda xs: "|".join(map(str, xs)))
                df_sub = round_mae_columns(df_sub)

            out_csv = outdir / f"{args.exhaustive_out_prefix}_topology_{topo}.csv"
            df_sub.to_csv(out_csv, index=False)

            out_plot = outdir / f"exhaustive_subsets_mae_boxplot_topology_{topo}.png"
            plot_exhaustive_mae_boxplot(
                rows_sub=rows_sub,
                out_path=out_plot,
                topology=topo,
                ctx_percent=args.context_percent,
                ctx_sample_mode=args.ctx_sample_mode,
                mc_samples=args.mc_samples,
                fill_mode=args.fill_mode,
            )

            # Best subset per size k
            if len(df_sub):
                grp = df_sub.groupby("subset_size")["mae"]
                df_best = df_sub.loc[grp.idxmin()].sort_values("subset_size")
                df_best = df_best.rename(columns={"mae": "mae_best"})
                df_best = df_best.merge(
                    grp.agg(mae_min="min", mae_max="max", mae_mean="mean").reset_index(),
                    on="subset_size",
                    how="left",
                )
                #df_best["mae"] = df_best["mae_mean"]
                df_best = round_mae_columns(df_best)
                out_best = outdir / f"{args.exhaustive_out_prefix}_best_per_k_topology_{topo}.csv"
                df_best.to_csv(out_best, index=False)
            else:
                out_best = None

            print(
                f"[exhaustive_subsets] Saved: {out_csv}"
                + (f" and {out_best}" if out_best is not None else "")
                + f" | mae_all={mae_all_exh:.4f} | elapsed={dt_exh:.1f}s"
            )

    # ----- Always save main CSV -----
    import pandas as pd
    df_main = pd.DataFrame(rows_main)
    df_main = round_mae_columns(df_main)
    csv_path = outdir / "central_inference_distributed_features.csv"
    df_main.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}\n")


if __name__ == "__main__":
    main()
