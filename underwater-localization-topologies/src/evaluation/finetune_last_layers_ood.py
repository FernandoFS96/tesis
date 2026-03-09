"""
finetune_last_layers_ood.py
===========================
Adapts a pre-trained ANP (highvar) to a new deployment domain (lowvar) by
fine-tuning ONLY the LAST LAYER of each of the three sub-networks:

  LatentEncoder        → mu  +  log_var                (2 Linear heads)
  DeterministicEncoder → cross_attentions[-1]           (last cross-attn block)
  Decoder              → mean_projection + log_var_projection  (2 Linear heads)

Strategy (comparison with finetune_decoder_ood.py)
---------------------------------------------------
  finetune_decoder_ood   : freeze both encoders  → train entire Decoder (~30.9%)
  HERE                   : freeze most of every sub-network → train only the
                           output heads of each one               (~2 – 3 %)

Rationale: distributional shift (highvar → lowvar) primarily distorts the
*output* mappings of each sub-network.  Touching only the final projections
(and the last cross-attention for the DeterministicEncoder) should be enough
to re-calibrate predictions while preserving shared internal representations
and keeping the fine-tuned parameter count very small.

Fine-tuning details
-------------------
  - Data     : lowvar train split (--finetune-n random trajectories)
  - Val      : lowvar val split (early stopping)
  - Loss     : ANP NLL only  (beta=0)
  - Sensor masking applied identically to original training (bernoulli p=0.2,
    fill=train_mean from lowvar data)
  - Context  : "first" mode, same --context percentage as original training
  - Saves best checkpoint → <output-dir>/best_finetuned.pth.tar

Trainable sub-sets (parameter counts for num_hidden=128, input_dim=2020):
  latent_encoder.mu                              16,512
  latent_encoder.log_var                         16,512
  deterministic_encoder.cross_attentions[-1]     82,304
  decoder.mean_projection                         1,155
  decoder.log_var_projection                      1,155
  ─────────────────────────────────────────────────────
  Total trainable                               ~117,638  (~2.6 % of model)

Evaluation
----------
  oracle_raw              → lowvar model on lowvar test (ceiling)
  hv_raw                  → highvar model, no adaptation  (OoD baseline)
  hv_rts_var              → HV + RTS (R=σ²)
  hv_ar_rts_var           → HV + AR + RTS (R=σ²)
  ft_ll_raw               → fine-tuned last-layers, no post-processing
  ft_ll_rts_var           → fine-tuned + RTS (R=σ²)
  ft_ll_ar_rts_var        → fine-tuned + AR + RTS (R=σ²)

Outputs (in --output-dir)
-------------------------
  best_finetuned.pth.tar   checkpoint
  finetune_curves.png      NLL and MAE learning curves
  mae_comparison.csv       per-theta MAE for all series
  mae_vs_theta.png         line plot (analogous to finetune_decoder_ood)
  latency_comparison.csv
  latency_comparison.png
  efficiency_curve.png     (if --efficiency-sweep)
  efficiency_sweep.csv     (if --efficiency-sweep)

Usage
-----
  cd /home/fernando/tesis/underwater-localization-topologies
  python -m src.evaluation.finetune_last_layers_ood \
    --lowvar-ckpt /home/fernando/tesis/underwater-localization-topologies/src/training/results/ANP_topologies_masked/lowvar/masked_dropbernoulli_p0.2_train_mean_first/topology_ellipsoidal/best_checkpoint.pth.tar \
    --highvar-ckpt /home/fernando/tesis/underwater-localization-topologies/src/training/results/ANP_topologies_masked/highvar/masked_dropbernoulli_p0.2_train_mean_first/topology_ellipsoidal/best_checkpoint.pth.tar \
    --lowvar-data-dir  /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_low_variance \
    --highvar-data-dir /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_high_variance \
    --topology ellipsoidal \
    --finetune-n 100 \
    --epochs 300 \
    --lr 1e-4 \
    --output-dir results/finetune_last_layers_ood
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.anp import LatentModel
from src.utils.nav_dataset import NavigationTrajectoryDataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVAL_SEED = 18

EVAL_METHODS = [
    "raw",
    "kalman_rts_var",
    "ar_kalman_rts_var",
]

METHOD_LABELS = {
    "raw":               "Raw",
    "kalman_rts_var":    "RTS (R=σ²)",
    "ar_kalman_rts_var": "AR+RTS (R=σ²)",
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


def freeze_all_but_last_layers(model: LatentModel) -> int:
    """
    Freeze ALL parameters, then unfreeze ONLY the last layer of each
    sub-network:

      LatentEncoder        : mu, log_var
      DeterministicEncoder : cross_attentions[-1]  (last cross-attention block)
      Decoder              : mean_projection, log_var_projection

    Returns the number of trainable parameters.
    """
    # 1. Freeze everything
    for p in model.parameters():
        p.requires_grad_(False)

    # 2. Unfreeze: LatentEncoder output projections
    for p in model.latent_encoder.mu.parameters():
        p.requires_grad_(True)
    for p in model.latent_encoder.log_var.parameters():
        p.requires_grad_(True)

    # 3. Unfreeze: last cross-attention block of DeterministicEncoder
    for p in model.deterministic_encoder.cross_attentions[-1].parameters():
        p.requires_grad_(True)

    # 4. Unfreeze: Decoder output projections
    for p in model.decoder.mean_projection.parameters():
        p.requires_grad_(True)
    for p in model.decoder.log_var_projection.parameters():
        p.requires_grad_(True)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"[freeze] trainable: {n_trainable:,} / {n_total:,} "
          f"({100.0 * n_trainable / n_total:.1f} %)")

    # Breakdown
    def _count(mod): return sum(p.numel() for p in mod.parameters())
    print(f"   latent_encoder   mu + log_var         : "
          f"{_count(model.latent_encoder.mu) + _count(model.latent_encoder.log_var):,}")
    print(f"   det_encoder      cross_attentions[-1] : "
          f"{_count(model.deterministic_encoder.cross_attentions[-1]):,}")
    print(f"   decoder          mean + log_var proj  : "
          f"{_count(model.decoder.mean_projection) + _count(model.decoder.log_var_projection):,}")
    return n_trainable

# ---------------------------------------------------------------------------
# Data helpers  (identical to finetune_decoder_ood.py)
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
    X = np.concatenate([x for x, _ in data], axis=0)
    P, S = num_time_points, num_sensors
    X3 = X.reshape(X.shape[0], P, S)
    return X3.mean(axis=0).T


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
    x: torch.Tensor,
    sensor_mask: torch.Tensor,
    x_means: torch.Tensor,
    P: int, S: int,
) -> torch.Tensor:
    B, T, Dx = x.shape
    x4   = x.view(B, T, P, S)
    mu   = x_means.T.view(1, 1, P, S).to(x.device, x.dtype)
    mu   = mu.expand(B, T, P, S)
    m    = sensor_mask.view(B, 1, 1, S)
    x_m  = (x4 * m + mu * (1.0 - m)).reshape(B, T, Dx)
    mask_feat = sensor_mask.view(B, 1, S).expand(B, T, S)
    return torch.cat([x_m, mask_feat], dim=-1)


def _augment_x_allsensors(x: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
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
    model:           LatentModel,
    train_data:      List,
    val_data:        List,
    y_mean:          torch.Tensor,
    y_std:           torch.Tensor,
    x_means_SP:      np.ndarray,
    device:          torch.device,
    num_time_points: int,
    num_sensors:     int,
    ctx_pct:         int,
    lr:              float,
    epochs:          int,
    batch_size:      int,
    patience:        int,
    p_drop:          float,
    output_dir:      Optional[Path],
    verbose:         bool = True,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Fine-tune the last layers of model in-place.
    Returns (train_nll, val_nll, train_mae, val_mae) history lists.
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

    best_val_nll = float("inf")
    best_state   = None
    patience_ctr = 0

    hist_train_nll: List[float] = []
    hist_val_nll:   List[float] = []
    hist_train_mae: List[float] = []
    hist_val_mae:   List[float] = []

    for epoch in range(1, epochs + 1):
        # ---- Training ----
        model.train()
        epoch_nll = 0.0
        epoch_mae = 0.0
        n_batches = 0

        for x_raw, y_raw in train_loader:
            x_raw = x_raw.to(device)
            y_raw = y_raw.to(device)
            B, T, _ = x_raw.shape

            mask = sample_sensor_mask(B, S, p_drop, device)
            x    = apply_masking(x_raw, mask, x_means_t, P, S)

            y    = norm_y(y_raw, y_mean, y_std)
            ctx  = first_ctx_idx(T, ctx_pct, device)
            cx, cy = x[:, ctx, :], y[:, ctx, :]

            # beta=0: NLL only.  Even though we now have trainable encoder
            # params, the KL term would require a valid prior from a fully
            # frozen reference — omitting it keeps comparison fair with
            # finetune_decoder_ood.
            mean_n, var_n, loss_full, kl, nll = model(cx, cy, x, y, beta=0.0)
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

                x = _augment_x_allsensors(x_raw, model)
                y = norm_y(y_raw, y_mean, y_std)

                ctx  = first_ctx_idx(x_raw.shape[1], ctx_pct, device)
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

        if val_nll < best_val_nll - 1e-5:
            best_val_nll = val_nll
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

    if best_state is not None:
        model.load_state_dict(best_state)

    if output_dir is not None and verbose:
        ckpt_path = output_dir / "best_finetuned.pth.tar"
        torch.save({"model": model.state_dict()}, ckpt_path)
        print(f"[✓] Saved fine-tuned checkpoint → {ckpt_path}")

    return hist_train_nll, hist_val_nll, hist_train_mae, hist_val_mae

# ---------------------------------------------------------------------------
# Data-efficiency sweep
# ---------------------------------------------------------------------------

def _apply_last_layer_freeze(model: LatentModel) -> None:
    """Helper: apply the same freeze strategy as freeze_all_but_last_layers
    without printing. Used in sweep."""
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.latent_encoder.mu.parameters():
        p.requires_grad_(True)
    for p in model.latent_encoder.log_var.parameters():
        p.requires_grad_(True)
    for p in model.deterministic_encoder.cross_attentions[-1].parameters():
        p.requires_grad_(True)
    for p in model.decoder.mean_projection.parameters():
        p.requires_grad_(True)
    for p in model.decoder.log_var_projection.parameters():
        p.requires_grad_(True)


def _run_single_sweep_point(
    n:                  int,
    ft_data:            List,
    hv_original_state:  dict,
    lv_val:             List,
    lv_groups:          Dict[float, List],
    lv_y_mean:          torch.Tensor,
    lv_y_std:           torch.Tensor,
    lv_x_means:         np.ndarray,
    device:             torch.device,
    finetune_kw:        dict,
    eval_kw:            dict,
) -> Dict[str, float]:
    from src.models.anp import LatentModel as _LM
    key = next(k for k in hv_original_state
               if "context_projection.linear_layer.weight" in k)
    input_dim = hv_original_state[key].shape[1]
    model_n = _LM(num_hidden=128, input_dim=input_dim, output_dim=3)
    model_n.load_state_dict(copy.deepcopy(hv_original_state))
    model_n = model_n.to(device)
    _apply_last_layer_freeze(model_n)

    finetune(
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

    theta_res, _ = eval_model_on_groups(
        model_n, lv_groups, lv_y_mean, lv_y_std, **eval_kw
    )
    method_means: Dict[str, float] = {}
    for method in eval_kw["methods"]:
        vals = [theta_res[t][method] for t in theta_res if method in theta_res[t]]
        method_means[method] = float(np.mean(vals)) if vals else float("nan")
    return method_means


def run_efficiency_sweep(
    hv_original_state:  dict,
    lv_train:           List,
    lv_val:             List,
    lv_groups:          Dict[float, List],
    lv_y_mean:          torch.Tensor,
    lv_y_std:           torch.Tensor,
    lv_x_means:         np.ndarray,
    device:             torch.device,
    sweep_ns:           List[int],
    seed:               int,
    finetune_kw:        dict,
    eval_kw:            dict,
) -> Dict[int, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    n_positive = sorted(n for n in sweep_ns if n > 0)
    max_n = min(max(n_positive), len(lv_train)) if n_positive else 0
    pool_idx = rng.permutation(len(lv_train))[:max_n]

    results: Dict[int, Dict[str, float]] = {}
    for n in sorted(set(sweep_ns)):
        k = len(lv_train) if n == 0 else min(n, len(lv_train))
        ft_data = lv_train if n == 0 else [lv_train[i] for i in pool_idx[:k]]
        label   = "all" if n == 0 else str(n)
        print(f"  [sweep] n={label:>4s}  ({len(ft_data)} traj) … ", end="", flush=True)

        method_means = _run_single_sweep_point(
            n=n, ft_data=ft_data,
            hv_original_state=hv_original_state,
            lv_val=lv_val, lv_groups=lv_groups,
            lv_y_mean=lv_y_mean, lv_y_std=lv_y_std, lv_x_means=lv_x_means,
            device=device, finetune_kw=finetune_kw, eval_kw=eval_kw,
        )
        results[n] = method_means
        print(f"raw={method_means.get('raw', float('nan')):.2f} m  "
              f"FT+AR+RTS={method_means.get('ar_kalman_rts_var', float('nan')):.2f} m")

    return results


def save_efficiency_csv(
    sweep_results:  Dict[int, Dict[str, float]],
    oracle_mae:     float,
    hv_raw_mae:     float,
    output_dir:     Path,
) -> None:
    path = output_dir / "efficiency_sweep.csv"
    methods = list(next(iter(sweep_results.values())).keys())
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_finetune", "n_label"] + methods + ["pct_gap_closed_ar_rts"])
        gap = hv_raw_mae - oracle_mae
        for n in sorted(sweep_results.keys()):
            label = "all" if n == 0 else str(n)
            row   = sweep_results[n]
            ar_mae = row.get("ar_kalman_rts_var", float("nan"))
            pct    = 100.0 * (hv_raw_mae - ar_mae) / max(gap, 1e-6)
            w.writerow(
                [n, label] +
                [f"{row.get(m, float('nan')):.4f}" for m in methods] +
                [f"{pct:.1f}"]
            )
    print(f"[✓] Saved {path}")


def plot_efficiency_curve(
    sweep_results:  Dict[int, Dict[str, float]],
    oracle_mae:     float,
    hv_raw_mae:     float,
    output_dir:     Path,
) -> None:
    ns_sorted = sorted((n for n in sweep_results if n != 0)) + \
                ([0] if 0 in sweep_results else [])
    x_labels = ["all" if n == 0 else str(n) for n in ns_sorted]
    x_pos    = np.arange(len(ns_sorted))

    def get_vals(method):
        return np.array([sweep_results[n].get(method, float("nan")) for n in ns_sorted])

    raw_v = get_vals("raw")
    rts_v = get_vals("kalman_rts_var")
    ar_v  = get_vals("ar_kalman_rts_var")
    gap   = hv_raw_mae - oracle_mae

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    ax1.axhline(oracle_mae,  color="#2c3e50", ls="-",  lw=1.5, alpha=0.7,
                label=f"Oracle ceiling ({oracle_mae:.2f} m)")
    ax1.axhline(hv_raw_mae, color="#e74c3c", ls="--", lw=1.5, alpha=0.7,
                label=f"HV raw baseline ({hv_raw_mae:.2f} m)")
    ax1.fill_between([-0.5, len(ns_sorted) - 0.5], oracle_mae, hv_raw_mae,
                     alpha=0.04, color="#e74c3c")

    ax1.plot(x_pos, raw_v, color="#27ae60", ls="-",  marker="o", lw=2,
             markersize=7, label="FT last-layers - Raw")
    ax1.plot(x_pos, rts_v, color="#1abc9c", ls="-.", marker="^", lw=2,
             markersize=7, label="FT last-layers - RTS(R=σ²)")
    ax1.plot(x_pos, ar_v,  color="#6c3483", ls=":",  marker="D", lw=2,
             markersize=7, label="FT last-layers - AR+RTS(R=σ²)")

    pct_closed = 100.0 * (hv_raw_mae - ar_v) / max(gap, 1e-6)
    ax2.plot(x_pos, pct_closed, color="#6c3483", ls=":", marker="D",
             lw=2, markersize=7, alpha=0.0)
    ax2.set_ylabel("% gap closed  [FT+AR+RTS(R=σ²)]", fontsize=9, color="#6c3483")
    ax2.tick_params(axis="y", labelcolor="#6c3483")
    ax2.set_ylim(
        100.0 * (hv_raw_mae - ax1.get_ylim()[1]) / max(gap, 1e-6),
        100.0 * (hv_raw_mae - ax1.get_ylim()[0]) / max(gap, 1e-6),
    )

    for i, (n, pct) in enumerate(zip(ns_sorted, pct_closed)):
        label = "all" if n == 0 else str(n)
        ax1.annotate(
            f"{pct:.0f}%",
            xy=(x_pos[i], ar_v[i]),
            xytext=(0, 10), textcoords="offset points",
            ha="center", fontsize=8, color="#6c3483", fontweight="bold",
        )

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, fontsize=9)
    ax1.set_xlabel("Number of fine-tuning trajectories", fontsize=11)
    ax1.set_ylabel("MAE (m)", fontsize=11)
    ax1.set_title(
        "Data-efficiency curve — Last-layer fine-tuning on lowvar data\n"
        "(% = gap closed by FT+AR+RTS(R=σ²) vs oracle)",
        fontsize=10,
    )
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(alpha=0.35)
    ax1.set_xlim(-0.5, len(ns_sorted) - 0.5)
    plt.tight_layout()
    path = output_dir / "efficiency_curve.png"
    plt.savefig(path, dpi=160); plt.close()
    print(f"[✓] Saved {path}")


# ---------------------------------------------------------------------------
# Post-processing kernels  (identical to finetune_decoder_ood.py)
# ---------------------------------------------------------------------------

def _cv_matrices(dt: float, sigma_a: float):
    F = np.array([[1, 0, dt, 0], [0, 1, 0, dt],
                  [0, 0, 1,  0], [0, 0, 0,  1]], dtype=np.float64)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
    q = sigma_a ** 2
    dt2, dt3, dt4 = dt**2, dt**3, dt**4
    Q1 = np.array([[dt4/4, dt3/2], [dt3/2, dt2]]) * q
    Q  = np.zeros((4, 4))
    Q[np.ix_([0, 2], [0, 2])] = Q1
    Q[np.ix_([1, 3], [1, 3])] = Q1
    return F, H, Q


def _kalman_cv(z_xy, R_xy, dt, sigma_a):
    T = z_xy.shape[0]
    F, H, Q = _cv_matrices(dt, sigma_a)
    x = np.zeros(4); x[:2] = z_xy[0]
    P = np.eye(4)
    xf, Pf       = np.zeros((T, 4)), np.zeros((T, 4, 4))
    xp_a, Pp_a   = np.zeros((T, 4)), np.zeros((T, 4, 4))
    I = np.eye(4)
    for t in range(T):
        xp = F @ x;  Pp = F @ P @ F.T + Q
        xp_a[t] = xp;  Pp_a[t] = Pp
        R   = np.eye(2) if R_xy is None else R_xy[t].astype(np.float64)
        inn = z_xy[t] - H @ xp
        S   = H @ Pp @ H.T + R
        K   = Pp @ H.T @ np.linalg.inv(S)
        x   = xp + K @ inn
        P   = (I - K @ H) @ Pp @ (I - K @ H).T + K @ R @ K.T
        xf[t] = x;  Pf[t] = P
    return xf, Pf, xp_a, Pp_a


def _rts(xf, Pf, xp_a, Pp_a, dt, sigma_a):
    T = xf.shape[0]
    F, _, _ = _cv_matrices(dt, sigma_a)
    xs, Ps = xf.copy(), Pf.copy()
    for k in range(T - 2, -1, -1):
        C     = Pf[k] @ F.T @ np.linalg.inv(Pp_a[k + 1])
        xs[k] = xf[k] + C @ (xs[k + 1] - xp_a[k + 1])
        Ps[k] = Pf[k] + C @ (Ps[k + 1] - Pp_a[k + 1]) @ C.T
    return xs, Ps


def _build_R(var_xy: np.ndarray, eps: float = 0.01) -> np.ndarray:
    T = var_xy.shape[0]
    R = np.zeros((T, 2, 2))
    R[:, 0, 0] = np.maximum(var_xy[:, 0], eps)
    R[:, 1, 1] = np.maximum(var_xy[:, 1], eps)
    return R


def rts_var(z_xy: np.ndarray, var_xy: np.ndarray, dt: float, sigma_a: float) -> np.ndarray:
    R = _build_R(var_xy)
    xf, Pf, xp_a, Pp_a = _kalman_cv(z_xy, R, dt, sigma_a)
    xs, _ = _rts(xf, Pf, xp_a, Pp_a, dt, sigma_a)
    return xs[:, :2].astype(np.float32)


# ---------------------------------------------------------------------------
# AR rollout  (identical to finetune_decoder_ood.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _ar_rollout(
    model:       torch.nn.Module,
    x:           torch.Tensor,
    y:           torch.Tensor,
    ctx_idx:     torch.Tensor,
    y_mean:      torch.Tensor,
    y_std:       torch.Tensor,
    block_k:     int   = 5,
    var_thresh:  float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = x.device
    _, T, _ = x.shape
    y_norm = norm_y(y, y_mean, y_std)
    ctx_np = ctx_idx.cpu().numpy().astype(int)
    non_ctx = [t for t in range(T) if t not in set(ctx_np.tolist())]
    ctx_sorted = np.sort(ctx_np)
    non_ctx.sort(key=lambda t: (int(np.min(np.abs(ctx_sorted - t))), t))

    roll_x = x[:, ctx_idx, :]
    roll_y = y_norm[:, ctx_idx, :]
    mu_z, _, _ = model.latent_encoder(roll_x, roll_y)      # type: ignore

    y_pred = torch.zeros_like(y_norm)
    y_var  = torch.zeros_like(y_norm)
    y_pred[:, ctx_idx, :] = roll_y

    K = max(1, block_k)
    for start in range(0, len(non_ctx), K):
        idxs   = non_ctx[start:start + K]
        idxs_t = torch.tensor(idxs, dtype=torch.long, device=device)
        tx     = x[:, idxs_t, :]
        r      = model.deterministic_encoder(roll_x, roll_y, tx)   # type: ignore
        z      = mu_z.unsqueeze(1).expand(1, tx.shape[1], -1)
        mn, vr = model.decoder(r, z, tx)                           # type: ignore
        y_pred[:, idxs_t, :] = mn
        y_var[:, idxs_t, :]  = vr
        v_xy   = vr[:, :, :2].mean(2).squeeze(0)
        accept = v_xy <= var_thresh
        if not accept.any():
            accept[torch.argmin(v_xy)] = True
        if accept.any():
            roll_x = torch.cat([roll_x, tx[:, accept, :]], dim=1)
            roll_y = torch.cat([roll_y, mn[:, accept, :]], dim=1)

    y_pred_phys = denorm_y(y_pred, y_mean, y_std)
    y_var_phys  = y_var * (y_std.view(1, 1, -1) ** 2)
    return y_pred_phys, y_var_phys


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_model_on_groups(
    model:        LatentModel,
    lv_groups:    Dict[float, List],
    y_mean:       torch.Tensor,
    y_std:        torch.Tensor,
    ctx_pct:      int,
    device:       torch.device,
    dt:           float,
    sigma_a:      float,
    ar_block_k:   int,
    ar_var_thresh: float,
    methods:      List[str],
) -> Tuple[Dict[float, Dict[str, float]], Dict[float, Dict[str, float]]]:
    """
    Returns:
      mae_results     : {theta: {method: mean_mae_m}}
      latency_results : {theta: {method: mean_seconds_per_trajectory}}
    """
    results:   Dict[float, Dict[str, float]] = {}
    latencies: Dict[float, Dict[str, float]] = {}
    model.eval()

    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    for theta in sorted(lv_groups.keys()):
        samples   = lv_groups[theta]
        mae_lists: Dict[str, List[float]] = {m: [] for m in methods}
        lat_lists: Dict[str, List[float]] = {m: [] for m in methods}

        for x_np, y_np in samples:
            T = x_np.shape[0]
            x = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(0)
            y = torch.tensor(y_np, dtype=torch.float32, device=device).unsqueeze(0)
            x = _augment_x_allsensors(x, model)

            ctx  = first_ctx_idx(T, ctx_pct, device)
            nc   = np.ones(T, bool)
            nc[ctx.cpu().numpy()] = False

            y_norm = norm_y(y, y_mean, y_std)
            cx, cy = x[:, ctx, :], y_norm[:, ctx, :]

            _sync()
            t0 = time.perf_counter()
            mean_n, var_n, *_ = model(cx, cy, x)
            _sync()
            t_raw = time.perf_counter() - t0

            p_phys = denorm_y(mean_n, y_mean, y_std).squeeze(0).cpu().numpy()
            v_phys = (var_n * (y_std.view(1, 1, -1) ** 2)).squeeze(0).cpu().numpy()
            y_gt   = y.squeeze(0).cpu().numpy()

            def mae(p): return float(np.mean(np.abs(p[nc] - y_gt[nc])))

            if "raw" in methods:
                mae_lists["raw"].append(mae(p_phys))
                lat_lists["raw"].append(t_raw)

            if "kalman_rts_var" in methods:
                t0 = time.perf_counter()
                xy_rts = rts_var(p_phys[:, :2], v_phys[:, :2], dt, sigma_a)
                lat_lists["kalman_rts_var"].append(time.perf_counter() - t0)
                p_rts = p_phys.copy(); p_rts[:, :2] = xy_rts
                mae_lists["kalman_rts_var"].append(mae(p_rts))

            if "ar_kalman_rts_var" in methods:
                _sync()
                t0 = time.perf_counter()
                ar_phys, ar_var = _ar_rollout(
                    model, x, y, ctx, y_mean, y_std,
                    block_k=ar_block_k, var_thresh=ar_var_thresh,
                )
                _sync()
                t_ar = time.perf_counter() - t0
                p_ar = ar_phys.squeeze(0).cpu().numpy()
                v_ar = ar_var.squeeze(0).cpu().numpy()
                t0   = time.perf_counter()
                xy_ar_rts = rts_var(p_ar[:, :2], v_ar[:, :2], dt, sigma_a)
                lat_lists["ar_kalman_rts_var"].append(t_ar + (time.perf_counter() - t0))
                p_ar_rts = p_ar.copy(); p_ar_rts[:, :2] = xy_ar_rts
                mae_lists["ar_kalman_rts_var"].append(mae(p_ar_rts))

        results[theta]   = {m: float(np.mean(v)) for m, v in mae_lists.items() if v}
        latencies[theta] = {m: float(np.mean(v)) for m, v in lat_lists.items()  if v}

    return results, latencies

# ---------------------------------------------------------------------------
# Latency helpers
# ---------------------------------------------------------------------------

def _latency_mean_ms(
    latency_results: Dict[float, Dict[str, float]],
    method: str,
) -> float:
    vals = [latency_results[t][method] for t in latency_results
            if method in latency_results[t]]
    return float(np.mean(vals)) * 1e3 if vals else float("nan")


def save_latency_csv(
    latency_by_model: Dict[str, Dict[float, Dict[str, float]]],
    output_dir: Path,
) -> None:
    path = output_dir / "latency_comparison.csv"
    thetas = sorted(next(iter(latency_by_model.values())).keys())
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "method", "label"] +
                   [f"theta_{t:.1f}_ms" for t in thetas] + ["mean_ms"])
        for model_label, theta_dict in latency_by_model.items():
            for method in EVAL_METHODS:
                vals_ms = [
                    theta_dict.get(t, {}).get(method, float("nan")) * 1e3
                    for t in thetas
                ]
                mean_ms = float(np.nanmean(vals_ms))
                w.writerow(
                    [model_label, method, METHOD_LABELS[method]] +
                    [f"{v:.3f}" for v in vals_ms] +
                    [f"{mean_ms:.3f}"]
                )
    print(f"[✓] Saved {path}")


def plot_latency_comparison(
    latency_by_model: Dict[str, Dict[float, Dict[str, float]]],
    output_dir: Path,
) -> None:
    model_labels = list(latency_by_model.keys())
    n_models     = len(model_labels)
    model_colors = ["#3498db", "#e74c3c", "#27ae60", "#9b59b6"]

    data: Dict[str, Dict[str, float]] = {}
    for ml, theta_dict in latency_by_model.items():
        data[ml] = {}
        for method in EVAL_METHODS:
            vals = [theta_dict[t].get(method, float("nan")) * 1e3 for t in theta_dict]
            data[ml][method] = float(np.nanmean(vals))

    x     = np.arange(len(EVAL_METHODS))
    width = 0.8 / n_models
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for i, (ml, color) in enumerate(zip(model_labels, model_colors)):
        offsets = (i - (n_models - 1) / 2) * width
        abs_ms  = [data[ml].get(m, float("nan")) for m in EVAL_METHODS]
        ax1.bar(x + offsets, abs_ms, width=width * 0.9,
                color=color, alpha=0.8, label=ml, edgecolor="black", lw=0.4)
        raw_ms   = data[ml].get("raw", 0.0)
        overhead = [max(0.0, v - raw_ms) if m != "raw" else 0.0
                    for m, v in zip(EVAL_METHODS, abs_ms)]
        ax2.bar(x + offsets, overhead, width=width * 0.9,
                color=color, alpha=0.8, label=ml, edgecolor="black", lw=0.4)

    for ax, title, ylabel in [
        (ax1, "Absolute inference latency per trajectory", "Latency (ms)"),
        (ax2, "Post-processing overhead vs raw forward pass", "Extra latency (ms)"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m] for m in EVAL_METHODS],
                           rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.35)

    plt.tight_layout()
    path = output_dir / "latency_comparison.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"[✓] Saved {path}")


# ---------------------------------------------------------------------------
# Plots & outputs
# ---------------------------------------------------------------------------

def plot_finetune_curves(
    train_nll: List[float], val_nll: List[float],
    train_mae: List[float], val_mae: List[float],
    output_dir: Path,
) -> None:
    epochs = np.arange(1, len(train_nll) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, train_nll, label="train NLL", color="#3498db")
    ax1.plot(epochs, val_nll,   label="val NLL",   color="#e74c3c")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("NLL")
    ax1.set_title("NLL Learning Curves"); ax1.legend(); ax1.grid(alpha=0.35)
    ax2.plot(epochs, train_mae, label="train MAE (m)", color="#3498db")
    ax2.plot(epochs, val_mae,   label="val MAE (m)",   color="#e74c3c")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("MAE (m)")
    ax2.set_title("MAE Learning Curves"); ax2.legend(); ax2.grid(alpha=0.35)
    plt.tight_layout()
    path = output_dir / "finetune_curves.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"[✓] Saved {path}")


def save_comparison_csv(
    all_series: Dict[str, Dict[float, Dict[str, float]]],
    thetas:     List[float],
    output_dir: Path,
) -> None:
    path = output_dir / "mae_comparison.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["series", "theta", "method", "mae_m"])
        for series_label, theta_dict in all_series.items():
            for theta in thetas:
                for method, mae in theta_dict.get(theta, {}).items():
                    w.writerow([series_label, f"{theta:.1f}", method, f"{mae:.4f}"])
    print(f"[✓] Saved {path}")


def plot_mae_comparison(
    oracle_res: Dict[float, Dict[str, float]],
    hv_res:     Dict[float, Dict[str, float]],
    ft_res:     Dict[float, Dict[str, float]],
    output_dir: Path,
) -> None:
    thetas = sorted(oracle_res.keys())
    x = np.array(thetas)

    def vals(res, method):
        return np.array([res.get(t, {}).get(method, float("nan")) for t in thetas])

    oracle_v = vals(oracle_res, "raw")
    hv_raw_v = vals(hv_res,     "raw")
    hv_rts_v = vals(hv_res,     "kalman_rts_var")
    hv_ar_v  = vals(hv_res,     "ar_kalman_rts_var")
    ft_raw_v = vals(ft_res,     "raw")
    ft_rts_v = vals(ft_res,     "kalman_rts_var")
    ft_ar_v  = vals(ft_res,     "ar_kalman_rts_var")

    series = [
        ("Oracle (lowvar) - Raw",         oracle_v,  "#2c3e50", "-",   "o"),
        ("HV - Raw (OoD baseline)",        hv_raw_v,  "#e74c3c", "--",  "s"),
        ("HV - RTS(R=σ²)",                 hv_rts_v,  "#e67e22", "-.",  "^"),
        ("HV - AR+RTS(R=σ²)",              hv_ar_v,   "#c0392b", ":",   "D"),
        ("FT last-layers - Raw",           ft_raw_v,  "#27ae60", "-",   "o"),
        ("FT last-layers - RTS(R=σ²)",     ft_rts_v,  "#1abc9c", "-.",  "^"),
        ("FT last-layers - AR+RTS(R=σ²)",  ft_ar_v,   "#6c3483", ":",   "D"),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(x, oracle_v, hv_raw_v, alpha=0.06, color="#e74c3c")

    for label, v, color, ls, marker in series:
        ax.plot(x, v, color=color, ls=ls, marker=marker,
                markersize=6, lw=2, label=label)

    for i, theta in enumerate(thetas):
        gap_total  = hv_raw_v[i] - oracle_v[i]
        gap_closed = hv_raw_v[i] - ft_ar_v[i]
        if gap_total > 0 and not np.isnan(gap_closed):
            pct = 100.0 * gap_closed / gap_total
            ax.annotate(f"{pct:.0f}%", xy=(theta, ft_ar_v[i]),
                        xytext=(0, -14), textcoords="offset points",
                        ha="center", fontsize=8, color="#6c3483", fontweight="bold")

    ax.set_xlabel("θ (channel variability)", fontsize=11)
    ax.set_ylabel("MAE (m)", fontsize=11)
    ax.set_title(
        "MAE vs θ — HV baseline / Last-layer fine-tuning / Post-processing\n"
        "(% = gap closed by FT+AR+RTS(R=σ²) vs oracle)",
        fontsize=10,
    )
    ax.set_xticks(thetas)
    ax.set_xticklabels([f"{t:.1f}" for t in thetas])
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.35)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    path = output_dir / "mae_vs_theta.png"
    plt.savefig(path, dpi=160); plt.close()
    print(f"[✓] Saved {path}")


def print_summary(
    oracle_res: Dict[float, Dict[str, float]],
    hv_res:     Dict[float, Dict[str, float]],
    ft_res:     Dict[float, Dict[str, float]],
) -> None:
    def mean_all(res, method):
        v = [res[t][method] for t in res
             if method in res[t] and not np.isnan(res[t][method])]
        return float(np.mean(v)) if v else float("nan")

    oracle_raw = mean_all(oracle_res, "raw")
    hv_raw     = mean_all(hv_res,     "raw")
    hv_rts     = mean_all(hv_res,     "kalman_rts_var")
    hv_ar_rts  = mean_all(hv_res,     "ar_kalman_rts_var")
    ft_raw     = mean_all(ft_res,     "raw")
    ft_rts     = mean_all(ft_res,     "kalman_rts_var")
    ft_ar_rts  = mean_all(ft_res,     "ar_kalman_rts_var")
    gap = hv_raw - oracle_raw

    def pct(mae): return 100.0 * (hv_raw - mae) / max(gap, 1e-6)

    print("\n" + "=" * 70)
    print("Fine-tuning Summary: Last Layers of All Sub-networks")
    print("(mean MAE across all θ)")
    print("=" * 70)
    print(f"  Oracle raw (ceiling)              : {oracle_raw:.2f} m")
    print(f"  HV raw     (OoD baseline)         : {hv_raw:.2f} m   (gap = {gap:.2f} m)")
    print(f"  HV best pp — RTS(R=σ²)            : {hv_rts:.2f} m  ({pct(hv_rts):.1f}% gap closed)")
    print(f"  HV best pp — AR+RTS(R=σ²)         : {hv_ar_rts:.2f} m  ({pct(hv_ar_rts):.1f}% gap closed)")
    print("-" * 70)
    print(f"  FT last-layers — raw              : {ft_raw:.2f} m   ({pct(ft_raw):.1f}% gap closed)")
    print(f"  FT last-layers — RTS(R=σ²)        : {ft_rts:.2f} m  ({pct(ft_rts):.1f}% gap closed)")
    print(f"  FT last-layers — AR+RTS(R=σ²)     : {ft_ar_rts:.2f} m  ({pct(ft_ar_rts):.1f}% gap closed)")
    print("=" * 70)

# ---------------------------------------------------------------------------
# argparse + main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune last layers of all ANP sub-networks for OoD adaptation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--lowvar-ckpt",      type=Path, required=True)
    p.add_argument("--highvar-ckpt",     type=Path, required=True)
    p.add_argument("--lowvar-data-dir",  type=Path, required=True)
    p.add_argument("--highvar-data-dir", type=Path, required=True)
    p.add_argument("--topology",   type=str, default="ellipsoidal",
                   choices=["ellipsoidal", "aligned", "random"])
    p.add_argument("--num-sensors",     type=int,   default=10)
    p.add_argument("--num-time-points", type=int,   default=201)
    p.add_argument("--sensor-drop-p",  type=float,  default=0.2)
    p.add_argument("--finetune-n",  type=int,   default=100)
    p.add_argument("--epochs",      type=int,   default=300)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--batch-size",  type=int,   default=16)
    p.add_argument("--patience",    type=int,   default=50)
    p.add_argument("--context",     type=int,   default=30)
    p.add_argument("--dt",          type=float, default=1.0)
    p.add_argument("--sigma-a",     type=float, default=1.0)
    p.add_argument("--ar-block-k",     type=int,   default=5)
    p.add_argument("--ar-var-thresh",  type=float, default=0.01)
    p.add_argument("--seed",        type=int,   default=EVAL_SEED)
    p.add_argument("--output-dir",  type=Path,  default=Path("results/finetune_last_layers_ood"))
    p.add_argument("--efficiency-sweep", action="store_true")
    p.add_argument("--sweep-ns", type=str, default="10,20,50,100,200,0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load models
    # ------------------------------------------------------------------ #
    print("\n[load] oracle (lowvar) model …")
    oracle_model = load_anp(args.lowvar_ckpt, device)
    oracle_model.eval()

    print("[load] highvar model (to be fine-tuned) …")
    hv_model = load_anp(args.highvar_ckpt, device)

    # ------------------------------------------------------------------ #
    # 2. Load data
    # ------------------------------------------------------------------ #
    print("\n[data] lowvar …")
    lv_train, lv_val, lv_test, lv_meta = load_data(args.lowvar_data_dir, args.topology)

    print("[data] highvar (normalization stats only) …")
    hv_train, _, _, _ = load_data(args.highvar_data_dir, args.topology)

    lv_y_mean, lv_y_std = compute_y_stats(lv_train, device)
    hv_y_mean, hv_y_std = compute_y_stats(hv_train, device)
    lv_x_means = compute_x_sensor_means(lv_train, args.num_time_points, args.num_sensors)

    if args.finetune_n > 0 and args.finetune_n < len(lv_train):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(lv_train), size=args.finetune_n, replace=False)
        ft_train = [lv_train[i] for i in idx]
    else:
        ft_train = lv_train
    print(f"[data] fine-tuning on {len(ft_train)} / {len(lv_train)} lowvar train trajectories")

    lv_groups = group_by_theta(lv_test, lv_meta)
    thetas    = sorted(lv_groups.keys())
    print(f"[data] test θ values: {thetas}")

    eval_kw = dict(
        ctx_pct=args.context, device=device,
        dt=args.dt, sigma_a=args.sigma_a,
        ar_block_k=args.ar_block_k, ar_var_thresh=args.ar_var_thresh,
        methods=EVAL_METHODS,
    )

    # ------------------------------------------------------------------ #
    # 3. Baseline evaluation (before fine-tuning)
    # ------------------------------------------------------------------ #
    print("\n[eval] oracle model …")
    oracle_res, oracle_lats = eval_model_on_groups(
        oracle_model, lv_groups, lv_y_mean, lv_y_std, **eval_kw
    )

    print("[eval] highvar baseline (no fine-tuning) …")
    hv_res, hv_lats = eval_model_on_groups(
        hv_model, lv_groups, hv_y_mean, hv_y_std, **eval_kw
    )

    hv_original_state = copy.deepcopy(hv_model.state_dict())

    # ------------------------------------------------------------------ #
    # 4. Fine-tune: last layers of all three sub-networks
    # ------------------------------------------------------------------ #
    print("\n[fine-tune] freezing all but last layers …")
    freeze_all_but_last_layers(hv_model)

    print(f"[fine-tune] training for up to {args.epochs} epochs "
          f"(lr={args.lr}, patience={args.patience}) …\n")
    tr_nll, vl_nll, tr_mae, vl_mae = finetune(
        model=hv_model,
        train_data=ft_train,
        val_data=lv_val,
        y_mean=lv_y_mean,
        y_std=lv_y_std,
        x_means_SP=lv_x_means,
        device=device,
        num_time_points=args.num_time_points,
        num_sensors=args.num_sensors,
        ctx_pct=args.context,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        p_drop=args.sensor_drop_p,
        output_dir=args.output_dir,
    )

    # ------------------------------------------------------------------ #
    # 5. Post fine-tuning evaluation
    # ------------------------------------------------------------------ #
    print("\n[eval] fine-tuned model …")
    hv_model.eval()
    ft_res, ft_lats = eval_model_on_groups(
        hv_model, lv_groups, lv_y_mean, lv_y_std, **eval_kw
    )

    # ------------------------------------------------------------------ #
    # 6. Outputs
    # ------------------------------------------------------------------ #
    plot_finetune_curves(tr_nll, vl_nll, tr_mae, vl_mae, args.output_dir)
    plot_mae_comparison(oracle_res, hv_res, ft_res, args.output_dir)
    save_comparison_csv(
        {"oracle": oracle_res, "highvar_baseline": hv_res, "finetuned_last_layers": ft_res},
        thetas,
        args.output_dir,
    )
    print_summary(oracle_res, hv_res, ft_res)
    save_latency_csv(
        {"HV baseline": hv_lats, "FT last-layers": ft_lats},
        args.output_dir,
    )
    plot_latency_comparison(
        {"HV baseline": hv_lats, "FT last-layers": ft_lats},
        args.output_dir,
    )

    # ------------------------------------------------------------------ #
    # 7. Data-efficiency sweep (optional)
    # ------------------------------------------------------------------ #
    if args.efficiency_sweep:
        sweep_ns = [int(x.strip()) for x in args.sweep_ns.split(",") if x.strip()]
        print(f"\n[efficiency sweep] n values: {sweep_ns}")

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

        sweep_res = run_efficiency_sweep(
            hv_original_state=hv_original_state,
            lv_train=lv_train,
            lv_val=lv_val,
            lv_groups=lv_groups,
            lv_y_mean=lv_y_mean,
            lv_y_std=lv_y_std,
            lv_x_means=lv_x_means,
            device=device,
            sweep_ns=sweep_ns,
            seed=args.seed,
            finetune_kw=ft_kw,
            eval_kw=eval_kw,
        )

        def _mean_all(res, method):
            v = [res[t][method] for t in res
                 if method in res[t] and not np.isnan(res[t][method])]
            return float(np.mean(v)) if v else float("nan")

        oracle_mean = _mean_all(oracle_res, "raw")
        hv_mean     = _mean_all(hv_res,     "raw")

        save_efficiency_csv(sweep_res, oracle_mean, hv_mean, args.output_dir)
        plot_efficiency_curve(sweep_res, oracle_mean, hv_mean, args.output_dir)

    print(f"\n[done] all outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
