"""
finetune_decoder_ood.py
=======================
Compares two decoder fine-tuning strategies when adapting a pre-trained
ANP (highvar) to a new deployment domain (lowvar):

  full_decoder : fine-tune all layers of the Decoder
  last_layer   : fine-tune only the output heads
                 (mean_projection + log_var_projection)

Both encoders are frozen in both cases.

Sweep
-----
For each strategy × each n in --sweep-ns, a fresh model copy is fine-tuned
on n random lowvar training trajectories, then evaluated on the lowvar test
set. Records MAE (m) and wall-clock training time (s).

Outputs (in --output-dir)
-------------------------
  comparison_sweep.csv           MAE + training time per (mode, n)
  comparison_sweep_mae.png       MAE vs n_trajectories
  comparison_sweep_time.png      Training time vs n_trajectories
  comparison_sweep_combined.png  Three-panel combined figure

Usage
-----
  cd /home/fernando/tesis/underwater-localization-topologies
  python -m src.evaluation.finetune_decoder_ood \
    --lowvar-ckpt  /home/fernando/tesis/underwater-localization-topologies/src/training/results/ANP_topologies_masked/lowvar/masked_dropbernoulli_p0.2_train_mean_first/topology_ellipsoidal/best_checkpoint.pth.tar \
    --highvar-ckpt /home/fernando/tesis/underwater-localization-topologies/src/training/results/ANP_topologies_masked/highvar/masked_dropbernoulli_p0.2_train_mean_first/topology_ellipsoidal/best_checkpoint.pth.tar \
    --lowvar-data-dir  /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_low_variance \
    --highvar-data-dir /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_high_variance \
    --topology ellipsoidal \
    --epochs 500 --lr 1e-4 \
    --output-dir results/finetune_decoder_ood \
    --sweep-ns 10,20,50,100,200,300,0
"""

from __future__ import annotations

import argparse
import copy
import csv
import pickle
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.anp import LatentModel
from src.utils.nav_dataset import NavigationTrajectoryDataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVAL_SEED = 18

# Fine-tuning modes
FT_MODES: List[str] = ["full_decoder", "last_layer"]
FT_MODE_LABELS: Dict[str, str] = {
    "full_decoder": "FT full decoder",
    "last_layer":   "FT last layer",
}

# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _detect_input_dim(ckpt_path: Path) -> int:
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    preferred = [
        "context_projection.linear_layer.weight",
        "decoder.target_projection.linear_layer.weight",
        "target_projection.linear_layer.weight",
    ]
    for sub in preferred:
        for k, v in state.items():
            if sub in k and v.ndim == 2:
                return int(v.shape[1])
    raise RuntimeError(f"Cannot detect input_dim from {ckpt_path}")


def load_anp(ckpt_path: Path, device: torch.device) -> LatentModel:
    input_dim = _detect_input_dim(ckpt_path)
    model = LatentModel(num_hidden=128, input_dim=input_dim, output_dim=3)
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    return model.to(device)


# Decoder layer name prefixes for selective freezing
_DECODER_LAST_LAYER_PREFIXES = (
    "decoder.mean_projection",
    "decoder.log_var_projection",
)


def _apply_freeze(model: LatentModel, verbose: bool = True) -> int:
    """Helper: print trainable count and return it."""
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"[freeze] trainable: {n_trainable:,} / {n_total:,} "
              f"({100.*n_trainable/n_total:.1f} %)")
    return n_trainable


def freeze_encoders(model: LatentModel, verbose: bool = True) -> int:
    """Freeze LatentEncoder + DeterministicEncoder (full decoder trained)."""
    for name, p in model.named_parameters():
        if name.startswith("latent_encoder") or name.startswith("deterministic_encoder"):
            p.requires_grad_(False)
    return _apply_freeze(model, verbose)


def freeze_for_last_layer_only(model: LatentModel, verbose: bool = True) -> int:
    """
    Freeze LatentEncoder, DeterministicEncoder, AND all Decoder layers
    except mean_projection and log_var_projection.
    Only the two output heads of the Decoder remain trainable.
    """
    for name, p in model.named_parameters():
        if name.startswith("latent_encoder") or name.startswith("deterministic_encoder"):
            p.requires_grad_(False)
        elif name.startswith("decoder") and not name.startswith(_DECODER_LAST_LAYER_PREFIXES):
            p.requires_grad_(False)
    return _apply_freeze(model, verbose)


def apply_ft_mode(model: LatentModel, ft_mode: str, verbose: bool = True) -> int:
    """Dispatch to the correct freeze function based on ft_mode."""
    if ft_mode == "last_layer":
        return freeze_for_last_layer_only(model, verbose)
    else:  # full_decoder
        return freeze_encoders(model, verbose)

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_data(data_dir: Path, topology: str) -> Tuple[List, List, List, Dict]:
    tdir = Path(data_dir) / f"topology_{topology}"
    def _load(name):
        with open(tdir / name, "rb") as f:
            return pickle.load(f)
    return _load("train_data.pkl"), _load("val_data.pkl"), _load("test_data.pkl"), _load("metadata.pkl")


def compute_y_stats(data: List, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    Y = np.concatenate([y for _, y in data], axis=0)
    return (
        torch.tensor(Y.mean(0), dtype=torch.float32, device=device),
        torch.tensor(Y.std(0) + 1e-6, dtype=torch.float32, device=device),
    )


def compute_x_sensor_means(
    data: List, num_time_points: int, num_sensors: int
) -> np.ndarray:
    """Returns (S, P) array of per-sensor feature means."""
    X = np.concatenate([x for x, _ in data], axis=0)   # (N*T, Dx)
    P, S = num_time_points, num_sensors
    X3 = X.reshape(X.shape[0], P, S)                   # (N*T, P, S)
    return X3.mean(axis=0).T                            # (S, P)


def group_by_theta(test_data: List, metadata: Dict) -> Dict[float, List]:
    groups: Dict[float, List] = {}
    for sample, theta in zip(test_data, metadata["test_thetas"]):
        groups.setdefault(float(theta), []).append(sample)
    return groups


def norm_y(y: torch.Tensor, m: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return (y - m.view(1, 1, -1)) / s.view(1, 1, -1)


def denorm_y(yn: torch.Tensor, m: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return yn * s.view(1, 1, -1) + m.view(1, 1, -1)

# ---------------------------------------------------------------------------
# Sensor masking  (same logic as training)
# ---------------------------------------------------------------------------

def sample_sensor_mask(
    B: int, S: int, p_drop: float, device: torch.device
) -> torch.Tensor:
    keep = torch.rand(B, S, device=device) > p_drop
    all_off = ~keep.any(dim=1)
    if all_off.any():
        idx = torch.randint(0, S, (int(all_off.sum().item()),), device=device)
        keep[all_off, idx] = True
    return keep.float()


def apply_masking(
    x: torch.Tensor,            # (B, T, Dx)   Dx = P * S
    sensor_mask: torch.Tensor,  # (B, S)
    x_means: torch.Tensor,      # (S, P)  on device
    P: int, S: int,
) -> torch.Tensor:              # (B, T, Dx + S)
    B, T, Dx = x.shape
    x4   = x.view(B, T, P, S)
    mu   = x_means.T.view(1, 1, P, S).to(x.device, x.dtype)
    mu   = mu.expand(B, T, P, S)
    m    = sensor_mask.view(B, 1, 1, S)
    x_m  = (x4 * m + mu * (1.0 - m)).reshape(B, T, Dx)
    mask_feat = sensor_mask.view(B, 1, S).expand(B, T, S)
    return torch.cat([x_m, mask_feat], dim=-1)


def _augment_x_allsensors(x: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    """Append all-ones mask if the raw x is missing the mask features."""
    expected: Optional[int] = None
    for name, p in model.named_parameters():
        if "context_projection.linear_layer.weight" in name and p.ndim == 2:
            expected = int(p.shape[1])
            break
    if expected is None or x.shape[-1] >= expected:
        return x
    n_extra = expected - x.shape[-1]
    ones = torch.ones(x.shape[0], x.shape[1], n_extra, dtype=x.dtype, device=x.device)
    return torch.cat([x, ones], dim=-1)


def first_ctx_idx(T: int, ctx_pct: int, device: torch.device) -> torch.Tensor:
    n = max(1, min(int(ctx_pct / 100 * T), T - 1))
    return torch.arange(n, device=device, dtype=torch.long)

# ---------------------------------------------------------------------------
# Fine-tuning loop
# ---------------------------------------------------------------------------

def finetune(
    model:         LatentModel,
    train_data:    List,
    val_data:      List,
    y_mean:        torch.Tensor,
    y_std:         torch.Tensor,
    x_means_SP:    np.ndarray,   # (S, P)
    device:        torch.device,
    num_time_points: int,
    num_sensors:   int,
    ctx_pct:       int,
    lr:            float,
    epochs:        int,
    batch_size:    int,
    patience:      int,
    p_drop:        float,
    output_dir:    Optional[Path],
    verbose:       bool = True,
) -> Tuple[List[float], List[float], List[float], List[float], float]:
    """
    Fine-tune the decoder of `model` in-place.
    Returns (train_nll, val_nll, train_mae, val_mae, train_time_seconds).
    """
    x_means_t = torch.tensor(x_means_SP, dtype=torch.float32, device=device)
    P = num_time_points
    S = num_sensors

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience // 2, min_lr=1e-6
    )

    train_ds = NavigationTrajectoryDataset(train_data)
    val_ds   = NavigationTrajectoryDataset(val_data)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_nll  = float("inf")
    best_val_mae  = float("inf")
    best_state    = None
    patience_ctr  = 0

    hist_train_nll: List[float] = []
    hist_val_nll:   List[float] = []
    hist_train_mae: List[float] = []
    hist_val_mae:   List[float] = []

    train_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        # ---- Training ----
        model.train()
        epoch_nll  = 0.0
        epoch_mae  = 0.0
        n_batches  = 0

        for x_raw, y_raw in train_loader:
            x_raw = x_raw.to(device)   # (B, T, Dx)
            y_raw = y_raw.to(device)   # (B, T, 3)
            B, T, _ = x_raw.shape

            # sensor masking
            mask = sample_sensor_mask(B, S, p_drop, device)
            x    = apply_masking(x_raw, mask, x_means_t, P, S)   # (B, T, Dx+S)

            # normalize y
            y    = norm_y(y_raw, y_mean, y_std)

            ctx  = first_ctx_idx(T, ctx_pct, device)
            cx, cy = x[:, ctx, :], y[:, ctx, :]

            # forward with target_y → NLL loss (KL contribution is zero because
            # beta=0.0, and encoder gradients are frozen anyway)
            mean_n, var_n, loss_full, kl, nll = model(cx, cy, x, y, beta=0.0)
            # nll here is over all T points including context; use it directly
            loss = nll

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                mean_phys = denorm_y(mean_n, y_mean, y_std)
                mae = torch.abs(mean_phys - y_raw).mean().item()

            epoch_nll += loss.item()
            epoch_mae += mae
            n_batches  += 1

        hist_train_nll.append(epoch_nll / n_batches)
        hist_train_mae.append(epoch_mae / n_batches)

        # ---- Validation ----
        model.eval()
        val_nll_sum = 0.0
        val_mae_sum = 0.0
        val_n       = 0

        with torch.no_grad():
            for x_raw, y_raw in val_loader:
                x_raw = x_raw.to(device)
                y_raw = y_raw.to(device)
                B, T, _ = x_raw.shape

                # all-sensors mask at eval (no dropout)
                x = _augment_x_allsensors(x_raw, model)
                y = norm_y(y_raw, y_mean, y_std)

                ctx  = first_ctx_idx(T, ctx_pct, device)
                cx, cy = x[:, ctx, :], y[:, ctx, :]

                mean_n, var_n, _, _, nll = model(cx, cy, x, y, beta=0.0)
                mean_phys = denorm_y(mean_n, y_mean, y_std)
                mae = torch.abs(mean_phys - y_raw).mean().item()

                val_nll_sum += nll.item()
                val_mae_sum += mae
                val_n       += 1

        val_nll = val_nll_sum / max(val_n, 1)
        val_mae = val_mae_sum / max(val_n, 1)
        hist_val_nll.append(val_nll)
        hist_val_mae.append(val_mae)

        scheduler.step(val_nll)

        # Early stopping
        if val_mae < best_val_mae - 1e-5:
            best_val_mae = val_mae
            best_state   = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"  epoch {epoch:4d}  train_nll={hist_train_nll[-1]:.4f} "
                  f" val_nll={val_nll:.4f}  val_mae={val_mae:.4f} m"
                  f"  patience={patience_ctr}/{patience}")

        if patience_ctr >= patience:
            if verbose:
                print(f"[early stop] epoch {epoch}")
            break

    train_elapsed = time.perf_counter() - train_start

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save checkpoint
    if output_dir is not None and verbose:
        ckpt_path = output_dir / "best_finetuned.pth.tar"
        torch.save({"model": model.state_dict()}, ckpt_path)
        print(f"[✓] Saved fine-tuned checkpoint → {ckpt_path}")

    return hist_train_nll, hist_val_nll, hist_train_mae, hist_val_mae, train_elapsed

# ---------------------------------------------------------------------------
# Comparison sweep  (ft_mode × n_trajectories)
# ---------------------------------------------------------------------------

def run_comparison_sweep(
    hv_original_state: dict,
    lv_train:          List,
    lv_val:            List,
    lv_groups:         Dict[float, List],
    lv_y_mean:         torch.Tensor,
    lv_y_std:          torch.Tensor,
    lv_x_means:        np.ndarray,
    device:            torch.device,
    sweep_ns:          List[int],
    ft_modes:          List[str],
    seed:              int,
    finetune_kw:       dict,
) -> Dict[str, Dict[int, Dict[str, object]]]:
    """
    For each ft_mode in ft_modes and each n in sweep_ns, fine-tune a fresh
    copy of the highvar model and record mean MAE + training time.

    Returns:
      {ft_mode: {n: {"mae": float, "train_time_s": float}}}
    """
    from src.models.anp import LatentModel as _LM

    rng        = np.random.default_rng(seed)
    n_positive = sorted(n for n in sweep_ns if n > 0)
    max_n      = min(max(n_positive), len(lv_train)) if n_positive else 0
    pool_idx   = rng.permutation(len(lv_train))[:max_n]

    # detect input_dim from state dict
    key       = next(k for k in hv_original_state
                     if "context_projection.linear_layer.weight" in k)
    input_dim = hv_original_state[key].shape[1]
    ctx_pct   = finetune_kw["ctx_pct"]

    results: Dict[str, Dict[int, Dict[str, object]]] = {}

    for ft_mode in ft_modes:
        results[ft_mode] = {}
        print(f"\n[comparison sweep] ft_mode={ft_mode}")
        for n in sorted(set(sweep_ns)):
            k       = len(lv_train) if n == 0 else min(n, len(lv_train))
            ft_data = lv_train if n == 0 else [lv_train[i] for i in pool_idx[:k]]
            label   = "all" if n == 0 else str(n)
            print(f"  n={label:>4s} ({len(ft_data)} traj) … ", end="", flush=True)

            # Fresh model copy
            model_n = _LM(num_hidden=128, input_dim=input_dim, output_dim=3)
            model_n.load_state_dict(copy.deepcopy(hv_original_state))
            model_n = model_n.to(device)
            apply_ft_mode(model_n, ft_mode, verbose=False)

            _, _, _, _, train_time = finetune(
                model=model_n,
                train_data=ft_data,
                val_data=lv_val,
                y_mean=lv_y_mean,
                y_std=lv_y_std,
                x_means_SP=lv_x_means,
                output_dir=None,
                verbose=False,
                **finetune_kw,
            )

            mae = eval_raw_mae(model_n, lv_groups, lv_y_mean, lv_y_std, ctx_pct, device)
            results[ft_mode][n] = {"mae": mae, "train_time_s": train_time}
            print(f"mae={mae:.2f} m  time={train_time:.1f}s")

    return results


def save_comparison_sweep_csv(
    sweep_results: Dict[str, Dict[int, Dict[str, object]]],
    oracle_mae:    float,
    hv_raw_mae:    float,
    output_dir:    Path,
) -> None:
    path = output_dir / "comparison_sweep.csv"
    gap  = hv_raw_mae - oracle_mae
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ft_mode", "ft_mode_label", "n_finetune", "n_label",
                    "train_time_s", "mae_m", "pct_gap_closed"])
        for ft_mode, ns_dict in sweep_results.items():
            lbl = FT_MODE_LABELS.get(ft_mode, ft_mode)
            for n in sorted(ns_dict.keys()):
                entry   = ns_dict[n]
                n_label = "all" if n == 0 else str(n)
                mae     = float(entry["mae"])  # type: ignore
                pct     = 100.0 * (hv_raw_mae - mae) / max(gap, 1e-6)
                w.writerow([ft_mode, lbl, n, n_label,
                            f"{float(entry['train_time_s']):.2f}",
                            f"{mae:.4f}", f"{pct:.1f}"])
    print(f"[✓] Saved {path}")


def plot_comparison_sweep(
    sweep_results: Dict[str, Dict[int, Dict[str, object]]],
    oracle_mae:    float,
    hv_raw_mae:    float,
    output_dir:    Path,
) -> None:
    """
    Three-panel figure:
      Left   — MAE vs n_trajectories per ft_mode
      Middle — Training time (s) vs n_trajectories per ft_mode
      Right  — MAE vs Training time (Pareto-style scatter)
    """
    ft_modes     = list(sweep_results.keys())
    mode_colors  = {"full_decoder": "#27ae60", "last_layer": "#8e44ad"}
    mode_markers = {"full_decoder": "o",        "last_layer": "D"}

    all_ns: List[int] = []
    for mode in ft_modes:
        all_ns.extend(sweep_results[mode].keys())
    ns_pos    = sorted(set(n for n in all_ns if n != 0))
    ns_sorted = ns_pos + ([0] if 0 in all_ns else [])
    x_labels  = ["all" if n == 0 else str(n) for n in ns_sorted]
    x_pos     = np.arange(len(ns_sorted))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5))

    for ft_mode in ft_modes:
        color     = mode_colors.get(ft_mode, "#555555")
        marker    = mode_markers.get(ft_mode, "s")
        lbl       = FT_MODE_LABELS.get(ft_mode, ft_mode)
        d         = sweep_results[ft_mode]
        mae_vals  = np.array([float(d[n]["mae"]) for n in ns_sorted])          # type: ignore
        time_vals = np.array([float(d[n]["train_time_s"]) for n in ns_sorted]) # type: ignore

        ax1.plot(x_pos, mae_vals,  color=color, marker=marker, lw=2, markersize=7, label=lbl)
        ax2.plot(x_pos, time_vals, color=color, marker=marker, lw=2, markersize=7, label=lbl)
        ax3.scatter(time_vals, mae_vals, color=color, marker=marker, s=70, label=lbl, zorder=3)
        for i, n in enumerate(ns_sorted):
            ax3.annotate(x_labels[i], (time_vals[i], mae_vals[i]),
                         textcoords="offset points", xytext=(5, 3), fontsize=7)

    for ax in (ax1, ax3):
        ax.axhline(oracle_mae, color="#2c3e50", ls="-",  lw=1.5, alpha=0.7,
                   label=f"Oracle ceiling ({oracle_mae:.2f} m)")
        ax.axhline(hv_raw_mae, color="#e74c3c", ls="--", lw=1.5, alpha=0.7,
                   label=f"HV raw baseline ({hv_raw_mae:.2f} m)")

    ax1.set_xticks(x_pos);  ax1.set_xticklabels(x_labels, fontsize=9)
    ax1.set_xlabel("Number of fine-tuning trajectories", fontsize=10)
    ax1.set_ylabel("MAE (m)", fontsize=10)
    ax1.set_title("MAE vs fine-tuning trajectories", fontsize=10)
    ax1.legend(fontsize=8);  ax1.grid(alpha=0.35)
    ax1.set_xlim(-0.5, len(ns_sorted) - 0.5)

    ax2.set_xticks(x_pos);  ax2.set_xticklabels(x_labels, fontsize=9)
    ax2.set_xlabel("Number of fine-tuning trajectories", fontsize=10)
    ax2.set_ylabel("Training time (s)", fontsize=10)
    ax2.set_title("Training time vs fine-tuning trajectories", fontsize=10)
    ax2.legend(fontsize=8);  ax2.grid(alpha=0.35)
    ax2.set_xlim(-0.5, len(ns_sorted) - 0.5)

    ax3.set_xlabel("Training time (s)", fontsize=10)
    ax3.set_ylabel("MAE (m)", fontsize=10)
    ax3.set_title("MAE vs training time (lower-left = better)", fontsize=10)
    ax3.legend(fontsize=8);  ax3.grid(alpha=0.35)

    plt.suptitle(
        f"Fine-tuning comparison: full decoder vs last layer only\n"
        f"(oracle={oracle_mae:.2f} m, HV raw={hv_raw_mae:.2f} m)",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = output_dir / "comparison_sweep_combined.png"
    plt.savefig(path, dpi=160);  plt.close()
    print(f"[✓] Saved {path}")

    # Individual panels
    _plot_comparison_panel(sweep_results, oracle_mae, hv_raw_mae,
                           ns_sorted, x_labels, metric="mae", output_dir=output_dir)
    _plot_comparison_panel(sweep_results, oracle_mae, hv_raw_mae,
                           ns_sorted, x_labels, metric="time", output_dir=output_dir)


def _plot_comparison_panel(
    sweep_results: Dict[str, Dict[int, Dict[str, object]]],
    oracle_mae:    float,
    hv_raw_mae:    float,
    ns_sorted:     List[int],
    x_labels:      List[str],
    metric:        str,
    output_dir:    Path,
) -> None:
    ft_modes     = list(sweep_results.keys())
    mode_colors  = {"full_decoder": "#27ae60", "last_layer": "#8e44ad"}
    mode_markers = {"full_decoder": "o",       "last_layer": "D"}
    x_pos = np.arange(len(ns_sorted))
    fig, ax = plt.subplots(figsize=(9, 5))

    for ft_mode in ft_modes:
        color  = mode_colors.get(ft_mode, "#555555")
        marker = mode_markers.get(ft_mode, "s")
        lbl    = FT_MODE_LABELS.get(ft_mode, ft_mode)
        d      = sweep_results[ft_mode]
        if metric == "mae":
            vals = np.array([float(d[n]["mae"]) for n in ns_sorted])          # type: ignore
        else:
            vals = np.array([float(d[n]["train_time_s"]) for n in ns_sorted]) # type: ignore
        ax.plot(x_pos, vals, color=color, marker=marker, lw=2, markersize=7, label=lbl)

    if metric == "mae":
        ax.axhline(oracle_mae, color="#2c3e50", ls="-",  lw=1.5, alpha=0.7,
                   label=f"Oracle ({oracle_mae:.2f} m)")
        ax.axhline(hv_raw_mae, color="#e74c3c", ls="--", lw=1.5, alpha=0.7,
                   label=f"HV raw ({hv_raw_mae:.2f} m)")
        ax.set_ylabel("MAE (m)", fontsize=11)
        ax.set_title("MAE vs fine-tuning trajectories — full decoder vs last layer",
                     fontsize=10)
        fname = "comparison_sweep_mae.png"
    else:
        ax.set_ylabel("Training time (s)", fontsize=11)
        ax.set_title("Training time vs fine-tuning trajectories — full decoder vs last layer",
                     fontsize=10)
        fname = "comparison_sweep_time.png"

    ax.set_xticks(x_pos);  ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_xlabel("Number of fine-tuning trajectories", fontsize=11)
    ax.legend(fontsize=9);  ax.grid(alpha=0.35)
    ax.set_xlim(-0.5, len(ns_sorted) - 0.5)
    plt.tight_layout()
    path = output_dir / fname
    plt.savefig(path, dpi=160);  plt.close()
    print(f"[✓] Saved {path}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_raw_mae(
    model:   LatentModel,
    groups:  Dict[float, List],
    y_mean:  torch.Tensor,
    y_std:   torch.Tensor,
    ctx_pct: int,
    device:  torch.device,
) -> float:
    """Mean MAE (m) across all theta groups, raw forward pass only."""
    model.eval()
    all_mae: List[float] = []
    for samples in groups.values():
        for x_np, y_np in samples:
            T  = x_np.shape[0]
            x  = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(0)
            y  = torch.tensor(y_np, dtype=torch.float32, device=device).unsqueeze(0)
            x  = _augment_x_allsensors(x, model)
            ctx = first_ctx_idx(T, ctx_pct, device)
            nc  = np.ones(T, bool)
            nc[ctx.cpu().numpy()] = False
            y_norm     = norm_y(y, y_mean, y_std)
            cx, cy     = x[:, ctx, :], y_norm[:, ctx, :]
            mean_n, *_ = model(cx, cy, x)
            p_phys     = denorm_y(mean_n, y_mean, y_std).squeeze(0).cpu().numpy()
            y_gt       = y.squeeze(0).cpu().numpy()
            all_mae.append(float(np.mean(np.abs(p_phys[nc] - y_gt[nc]))))
    return float(np.mean(all_mae)) if all_mae else float("nan")


# argparse + main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare full-decoder vs last-layer fine-tuning for ANP OoD adaptation.")
    p.add_argument("--lowvar-ckpt",      type=Path, required=True)
    p.add_argument("--highvar-ckpt",     type=Path, required=True)
    p.add_argument("--lowvar-data-dir",  type=Path, required=True)
    p.add_argument("--highvar-data-dir", type=Path, required=True)
    p.add_argument("--topology", type=str, default="ellipsoidal", choices=["ellipsoidal", "aligned", "random"])
    p.add_argument("--num-sensors",     type=int,   default=10)
    p.add_argument("--num-time-points", type=int,   default=201)
    p.add_argument("--sensor-drop-p",   type=float, default=0.2)
    p.add_argument("--epochs",     type=int,   default=500)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--batch-size", type=int,   default=8)
    p.add_argument("--patience",   type=int,   default=100)
    p.add_argument("--context",    type=int,   default=40, help="Context percentage [0-100].")
    p.add_argument("--seed",       type=int,   default=EVAL_SEED)
    p.add_argument("--output-dir", type=Path, default=Path("results/finetune_decoder_ood"))
    p.add_argument("--sweep-ns",   type=str, default="10,20,50,100,200,300,0", help="Comma-separated n values (0 = all training data).")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    set_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load models
    print("\n[load] oracle (lowvar) model …")
    oracle_model = load_anp(args.lowvar_ckpt, device)
    oracle_model.eval()

    print("[load] highvar model …")
    hv_model = load_anp(args.highvar_ckpt, device)
    hv_model.eval()

    # 2. Load data
    print("\n[data] lowvar …")
    lv_train, lv_val, lv_test, lv_meta = load_data(args.lowvar_data_dir, args.topology)

    print("[data] highvar (normalization stats only) …")
    hv_train, _, _, _ = load_data(args.highvar_data_dir, args.topology)

    lv_y_mean, lv_y_std = compute_y_stats(lv_train, device)
    hv_y_mean, hv_y_std = compute_y_stats(hv_train, device)
    lv_x_means = compute_x_sensor_means(lv_train, args.num_time_points, args.num_sensors)

    lv_groups = group_by_theta(lv_test, lv_meta)
    print(f"[data] test θ values: {sorted(lv_groups.keys())}")

    # 3. Baselines (raw MAE only)
    print("\n[eval] oracle model …")
    oracle_mae = eval_raw_mae(oracle_model, lv_groups, lv_y_mean, lv_y_std,
                              args.context, device)
    print(f"       oracle MAE = {oracle_mae:.4f} m")

    print("[eval] highvar baseline …")
    hv_raw_mae = eval_raw_mae(hv_model, lv_groups, hv_y_mean, hv_y_std,
                              args.context, device)
    print(f"       HV raw MAE = {hv_raw_mae:.4f} m  "
          f"(gap = {hv_raw_mae - oracle_mae:.4f} m)")

    hv_original_state = copy.deepcopy(hv_model.state_dict())

    # 4. Comparison sweep
    sweep_ns = [int(x.strip()) for x in args.sweep_ns.split(",") if x.strip()]
    print(f"\n[comparison sweep] modes={FT_MODES}  n values={sweep_ns}")

    ft_kw = dict(
        device=device,
        num_time_points=args.num_time_points,
        num_sensors=args.num_sensors,
        ctx_pct=args.context,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        p_drop=args.sensor_drop_p,
    )

    comp_res = run_comparison_sweep(
        hv_original_state=hv_original_state,
        lv_train=lv_train,
        lv_val=lv_val,
        lv_groups=lv_groups,
        lv_y_mean=lv_y_mean,
        lv_y_std=lv_y_std,
        lv_x_means=lv_x_means,
        device=device,
        sweep_ns=sweep_ns,
        ft_modes=FT_MODES,
        seed=args.seed,
        finetune_kw=ft_kw,
    )

    # 5. Outputs
    save_comparison_sweep_csv(comp_res, oracle_mae, hv_raw_mae, args.output_dir)
    plot_comparison_sweep(comp_res, oracle_mae, hv_raw_mae, args.output_dir)

    print(f"\n[done] all outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
