#!/usr/bin/env python3
"""
explore_lora_ood.py
===================
Explores Low-Rank Adaptation (LoRA) configurations for OoD fine-tuning of
ANP and RANP models (highvar → lowvar domain adaptation).

Motivation
----------
Full fine-tuning of decoder_det_full showed overfitting despite achieving
only 55% gap closure, while decoder_det_last (touching far fewer params)
reached 99%.  LoRA allows testing whether the layers that overfitted
(DeterministicEncoder self-attention) can be beneficially adapted with a
constrained low-rank update, recovering expressiveness without destroying
transferable representations.

LoRA mechanics
--------------
For a frozen weight matrix W (d_out × d_in), LoRA adds a trainable update:
    output = x @ W.T + x @ A.T @ B.T * (alpha / r)
where A ∈ ℝ^{r × d_in}, B ∈ ℝ^{d_out × r}, rank r << min(d_in, d_out).
Only A and B are trained; W stays frozen.

This is applied selectively to nn.Linear layers within chosen modules.
For LSTM layers (RANP TemporalEncoder) LoRA is applied via a parallel
low-rank correction added to the output of the RNN at each timestep,
since PyTorch's fused LSTM does not expose individual gate matrices as
nn.Linear. This is noted explicitly in the results.

LoRA targets per model
----------------------
ANP:
  lora_det_full   – LoRA on all Linear layers of DeterministicEncoder + full Decoder
  lora_det_last   – LoRA on last cross-attn of DeterministicEncoder + full Decoder
                    (comparison baseline: mirrors the best full-FT strategy)

RANP:
  lora_anp_base   – LoRA on full Decoder + full DeterministicEncoder (no RNN touch)
  lora_rnn_out    – LoRA on full Decoder + parallel low-rank correction on RNN output
  lora_full       – LoRA on full Decoder + DeterministicEncoder + RNN output correction

LoRA configurations swept
-------------------------
  ranks  : 4, 8, 16
  alphas : rank × 1.0,  rank × 2.0   (i.e. alpha/r = 1 or 2)
  → 3 × 2 = 6 configs per target per model

Usage
-----
  cd <project-root>
  python explore_lora_ood.py \
  --optuna-root /home/fernando/tesis/underwater-localization-topologies/src/training/results/optuna \
  --lowvar-data /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_low_variance \
  --topologies  ellipsoidal,random,aligned \
  --model-types anp \
  --lora-targets lora_det_full,lora_det_last \
  --ranks 4,8,16 --alpha-ratios 1.0,2.0 \
  --n-traj 100,200,all --device cuda \
  --skip-existing


  python explore_lora_ood.py \
  --optuna-root /home/fernando/tesis/underwater-localization-topologies/src/training/results/optuna \
  --lowvar-data /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_low_variance \
  --topologies  ellipsoidal,random,aligned \
  --model-types ranp \
  --lora-targets lora_anp_base,lora_rnn_out,lora_full \
  --ranks 4,8,16 --alpha-ratios 1.0,2.0 \
  --n-traj 100,200,all --device cuda \
  --skip-existing

Notes
-----
- Same data pipeline and evaluation protocol (inverse_context_holdout) as
  finetune_ood.py, ensuring direct comparability of results.
- Baselines (OoD and oracle) are recomputed and saved per (topology, model).
- --skip-existing works at the (target, rank, alpha, n_traj) level.
- Use --n-seeds for multi-seed averaging (same mechanic as finetune_ood.py).
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
from tqdm.auto import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.load_optuna_model import (
    load_optuna_best_model,
    resolve_optuna_best_model_dir,
)

# =============================================================================
# Constants (must match finetune_ood.py)
# =============================================================================

NUM_SENSORS     = 10
NUM_TIME_POINTS = 201
OUTPUT_DIM      = 3

# LoRA targets available per model type
ANP_LORA_TARGETS  = ["lora_det_full", "lora_det_last"]
RANP_LORA_TARGETS = ["lora_anp_base", "lora_rnn_out", "lora_full"]
ALL_LORA_TARGETS  = ANP_LORA_TARGETS + RANP_LORA_TARGETS

TARGET_LABELS = {
    "lora_det_full": "LoRA: Dec. Full + Det.Enc. Full",
    "lora_det_last": "LoRA: Dec. Full + Det.Enc. last",
    "lora_anp_base": "LoRA (RANP): Dec. Full + Full Det.Enc.",
    "lora_rnn_out":  "LoRA (RANP): RNN output + Dec. Full",
    "lora_full":     "LoRA (RANP): RNN output + Det.Enc. + Dec. Full",
}

TARGET_COLORS = {
    "lora_det_full": "#2980b9",
    "lora_det_last": "#27ae60",
    "lora_anp_base": "#e67e22",
    "lora_rnn_out":  "#8e44ad",
    "lora_full":     "#c0392b",
}

# --- Global plotting controls (match finetune_ood.py) ---
PLOT_AXIS_LABEL_SIZE = 20
PLOT_TICK_LABEL_SIZE = 18
PLOT_LEGEND_SIZE     = 14
PLOT_TEXT_SIZE       = 18
PLOT_TITLE_SIZE      = 20
# If False, plot titles will be omitted
PLOT_SHOW_TITLES     = False

# =============================================================================
# LoRA implementation
# =============================================================================

class LoRALinear(nn.Module):
    """
    Wraps a frozen nn.Linear with a trainable low-rank update.

    output = x @ W.T  +  x @ A.T @ B.T * (alpha / r)

    Only A and B are trainable.  W is frozen (requires_grad=False).
    """

    def __init__(
        self,
        linear:  nn.Linear,
        rank:    int,
        alpha:   float,
    ):
        super().__init__()
        d_out, d_in = linear.weight.shape

        # freeze original weight
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad_(False)

        self.rank  = rank
        self.scale = alpha / rank

        # LoRA matrices — standard init (A ~ N(0,σ), B = 0)
        self.lora_A = nn.Parameter(torch.empty(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        lora = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
        return base + lora

    @property
    def weight(self):
        """Expose .weight for compatibility with code that reads it."""
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def extra_repr(self) -> str:
        d_out, d_in = self.linear.weight.shape
        return (
            f"in={d_in}, out={d_out}, "
            f"rank={self.rank}, scale={self.scale:.3f}, "
            f"lora_params={self.rank*(d_in+d_out):,}"
        )


class RNNLoRACorrection(nn.Module):
    """
    Parallel low-rank correction applied to the output sequence of the RNN.

    h_corrected[t] = h[t]  +  x[t] @ A.T @ B.T * (alpha / r)

    This avoids patching the fused LSTM kernel while still adapting how
    the input signal x influences the hidden representation h.
    A ∈ ℝ^{r × d_input}, B ∈ ℝ^{d_hidden × r}
    """

    def __init__(
        self,
        d_input:  int,
        d_hidden: int,
        rank:     int,
        alpha:    float,
    ):
        super().__init__()
        self.rank  = rank
        self.scale = alpha / rank

        self.lora_A = nn.Parameter(torch.empty(rank, d_input))
        self.lora_B = nn.Parameter(torch.zeros(d_hidden, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))

    def forward(
        self,
        x_seq: torch.Tensor,   # (B, T, d_input)
        h_seq: torch.Tensor,   # (B, T, d_hidden)
    ) -> torch.Tensor:
        correction = F.linear(F.linear(x_seq, self.lora_A), self.lora_B) * self.scale
        return h_seq + correction

    def extra_repr(self) -> str:
        d_in  = self.lora_A.shape[1]
        d_out = self.lora_B.shape[0]
        return (
            f"in={d_in}, hidden={d_out}, "
            f"rank={self.rank}, scale={self.scale:.3f}"
        )


def _apply_lora_to_linears(module: nn.Module, rank: int, alpha: float) -> int:
    """
    Recursively replace all nn.Linear children with LoRALinear.
    Returns number of layers replaced.
    """
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank, alpha))
            replaced += 1
        else:
            replaced += _apply_lora_to_linears(child, rank, alpha)
    return replaced


def count_trainable(model: nn.Module) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return trainable, total


# =============================================================================
# LoRA application strategies
# =============================================================================

def apply_lora_strategy(
    model:       nn.Module,
    model_type:  str,
    lora_target: str,
    rank:        int,
    alpha:       float,
) -> Optional[RNNLoRACorrection]:
    """
    Freeze the whole model, then apply LoRA to the chosen modules.

    Returns an RNNLoRACorrection instance if the strategy requires one
    (RANP rnn_out / lora_full), else None.

    The caller must:
      1. Add the returned correction module to the model (if not None).
      2. Only optimize parameters with requires_grad=True.
    """
    # Step 1: freeze everything
    for p in model.parameters():
        p.requires_grad_(False)

    rnn_correction: Optional[RNNLoRACorrection] = None

    # ── ANP strategies ────────────────────────────────────────────────────────
    if lora_target == "lora_det_full":
        # LoRA on every Linear in DeterministicEncoder + unfreeze full Decoder
        n = _apply_lora_to_linears(model.deterministic_encoder, rank, alpha)
        for p in model.decoder.parameters(): # type: ignore
            p.requires_grad_(True)

    elif lora_target == "lora_det_last":
        # LoRA on last cross-attention of DeterministicEncoder + full Decoder
        n = _apply_lora_to_linears(
            model.deterministic_encoder.cross_attentions[-1], rank, alpha  # type: ignore
        )
        for p in model.decoder.parameters():  # type: ignore
            p.requires_grad_(True)

    # ── RANP strategies ───────────────────────────────────────────────────────
    elif lora_target == "lora_anp_base":
        # LoRA on DeterministicEncoder + full Decoder (no RNN touch, same as ANP)
        _apply_lora_to_linears(model.deterministic_encoder, rank, alpha)
        for p in model.decoder.parameters():  # type: ignore
            p.requires_grad_(True)

    elif lora_target == "lora_rnn_out":
        # Parallel low-rank correction on RNN output + full Decoder
        d_input  = model.temporal_encoder.input_proj.in_features  # type: ignore
        d_hidden = model.temporal_encoder.input_proj.out_features  # type: ignore
        rnn_correction = RNNLoRACorrection(d_input, d_hidden, rank, alpha)
        for p in model.decoder.parameters(): # type: ignore
            p.requires_grad_(True)

    elif lora_target == "lora_full":
        # LoRA on DeterministicEncoder + RNN correction + full Decoder
        _apply_lora_to_linears(model.deterministic_encoder, rank, alpha)
        d_input  = model.temporal_encoder.input_proj.in_features # type: ignore
        d_hidden = model.temporal_encoder.input_proj.out_features # type: ignore
        rnn_correction = RNNLoRACorrection(d_input, d_hidden, rank, alpha)
        for p in model.decoder.parameters(): # type: ignore
            p.requires_grad_(True)

    else:
        raise ValueError(
            f"Unknown lora_target '{lora_target}'. "
            f"Valid: {ALL_LORA_TARGETS}"
        )

    return rnn_correction


def validate_target_for_model(lora_target: str, model_type: str) -> None:
    if model_type == "anp" and lora_target in RANP_LORA_TARGETS:
        raise ValueError(
            f"Target '{lora_target}' is RANP-only "
            f"(model_type='{model_type}' given)."
        )
    if model_type == "ranp" and lora_target in ANP_LORA_TARGETS:
        raise ValueError(
            f"Target '{lora_target}' is ANP-only "
            f"(model_type='{model_type}' given)."
        )


# =============================================================================
# Data utilities  (identical to finetune_ood.py)
# =============================================================================

def load_topology_data(data_dir: str | Path, topology: str):
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
    Y    = np.concatenate([y for _, y in data], axis=0)
    mean = torch.tensor(Y.mean(0), dtype=torch.float32)
    std  = torch.tensor(Y.std(0) + 1e-6, dtype=torch.float32)
    return mean, std


def compute_x_means(data, num_time_points: int, num_sensors: int) -> torch.Tensor:
    X  = np.concatenate([x for x, _ in data], axis=0)
    X3 = X.reshape(X.shape[0], num_time_points, num_sensors)
    return torch.tensor(X3.mean(0).T, dtype=torch.float32)


def apply_mask_and_append(
    x_batch:     torch.Tensor,
    sensor_mask: torch.Tensor,
    x_means_SP:  torch.Tensor,
    num_time_points: int,
    num_sensors: int,
) -> torch.Tensor:
    B, T, Dx = x_batch.shape
    P, S     = num_time_points, num_sensors
    x4  = x_batch.view(B, T, P, S)
    mu  = x_means_SP.T.view(1, 1, P, S).to(x_batch.device, dtype=x_batch.dtype)
    m   = sensor_mask.view(B, 1, 1, S)
    x4m = x4 * m + mu * (1.0 - m)
    x_masked  = x4m.reshape(B, T, Dx)
    mask_feat = sensor_mask.view(B, 1, S).expand(B, T, S)
    return torch.cat([x_masked, mask_feat], dim=-1)


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
    n_holdout     = max(1, int(round(holdout_frac * total_points)))
    holdout_start = total_points - n_holdout
    max_ctx       = max(1, holdout_start)
    ctx_size      = max(1, min(max_ctx, int(round(context_frac * max_ctx))))
    ctx_start     = holdout_start - ctx_size
    ctx_idx       = torch.arange(ctx_start, holdout_start, device=device)
    tar_idx       = torch.arange(holdout_start, total_points, device=device)
    return ctx_idx, tar_idx


def subsample_data(data: list, n: int | str, seed: int = 0) -> list:
    if n == "all" or int(n) >= len(data):
        return data
    rng     = np.random.default_rng(seed)
    indices = rng.choice(len(data), size=int(n), replace=False)
    return [data[i] for i in sorted(indices)]


# =============================================================================
# Forward pass dispatcher (ANP / RANP, with optional RNN correction)
# =============================================================================

def model_forward(
    model:          nn.Module,
    model_type:     str,
    x_aug:          torch.Tensor,
    ctx_idx:        torch.Tensor,
    ctx_y:          torch.Tensor,
    tar_idx:        torch.Tensor,
    tar_y:          Optional[torch.Tensor] = None,
    beta:           float = 1.0,
    rnn_correction: Optional[RNNLoRACorrection] = None,
):
    """
    Unified forward for ANP and RANP, supporting an optional RNNLoRACorrection.

    For RANP with rnn_correction, we call the TemporalEncoder manually,
    apply the correction, then pass the patched h_seq to the rest of the model
    via a monkey-patched forward that skips the internal temporal_encoder call.
    """
    if model_type == "anp":
        ctx_x = x_aug[:, ctx_idx, :]
        tar_x = x_aug[:, tar_idx, :]
        return model(ctx_x, ctx_y, tar_x, tar_y, beta=beta)

    elif model_type == "ranp":
        if rnn_correction is None:
            return model(
                x_seq           = x_aug,
                context_indices = ctx_idx,
                context_y       = ctx_y,
                target_indices  = tar_idx,
                target_y        = tar_y,
                beta            = beta,
            )
        else:
            # Manual forward with RNN correction injected
            h_seq = model.temporal_encoder(x_aug)          # (B, T, H)
            h_seq = rnn_correction(x_aug, h_seq)            # apply LoRA correction

            ctx_x      = h_seq[:, ctx_idx, :]
            tar_x      = h_seq[:, tar_idx, :]
            num_targets = tar_x.size(1)

            prior_mu, prior_var, prior = model.latent_encoder(ctx_x, ctx_y)

            if tar_y is not None:
                post_mu, post_var, posterior = model.latent_encoder(tar_x, tar_y)
                z = posterior
            else:
                z = prior

            z = z.unsqueeze(1).repeat(1, num_targets, 1)
            r = model.deterministic_encoder(ctx_x, ctx_y, tar_x)
            y_pred_mean, y_pred_var = model.decoder(r, z, tar_x)

            if tar_y is not None:
                nll  = (0.5 * torch.log(2 * torch.pi * y_pred_var)
                        + 0.5 * (tar_y - y_pred_mean) ** 2 / y_pred_var).mean()
                kl   = model.kl_div(prior_mu, prior_var, post_mu, post_var)
                loss = nll + beta * kl
            else:
                kl = loss = nll = None

            return y_pred_mean, y_pred_var, loss, kl, nll

    else:
        raise ValueError(f"Unknown model_type: '{model_type}'")


# =============================================================================
# Training loop
# =============================================================================

def run_lora_finetune(
    model:          nn.Module,
    model_type:     str,
    lora_target:    str,
    rank:           int,
    alpha:          float,
    train_data:     list,
    val_data:       list,
    y_mean:         torch.Tensor,
    y_std:          torch.Tensor,
    x_means_SP:     torch.Tensor,
    n_traj:         int | str,
    lr:             float,
    epochs:         int,
    patience:       int,
    batch_size:     int,
    device:         str | torch.device,
    save_dir:       Path,
    holdout_frac:   float = 0.2,
    es_ctx_frac:    float = 0.4,
    seed:           int   = 0,
) -> Dict:
    """
    Apply LoRA to *model* according to *lora_target*, then fine-tune.

    Returns metadata dict with best_val_mae, total_time_s, n_epochs,
    n_trainable_params, n_traj_used.
    Best checkpoint written to save_dir/lora_checkpoint.pth.tar.
    Epoch log written to save_dir/lora_log.csv.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── apply LoRA ────────────────────────────────────────────────────────────
    rnn_correction = apply_lora_strategy(model, model_type, lora_target, rank, alpha)
    model.to(device)
    if rnn_correction is not None:
        rnn_correction = rnn_correction.to(device)

    # collect trainable params (LoRA matrices + unfrozen decoder)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if rnn_correction is not None:
        trainable_params += list(rnn_correction.parameters())

    n_trainable = sum(p.numel() for p in trainable_params)
    n_total     = sum(p.numel() for p in model.parameters())
    if rnn_correction is not None:
        n_total += sum(p.numel() for p in rnn_correction.parameters())

    print(
        f"    LoRA trainable: {n_trainable:,} / {n_total:,}  "
        f"({100*n_trainable/max(1,n_total):.2f}%)"
    )

    # ── data ─────────────────────────────────────────────────────────────────
    ft_data      = subsample_data(train_data, n_traj, seed=seed)
    train_loader = make_dataloader(ft_data, batch_size, shuffle=True)
    val_loader   = make_dataloader(val_data, batch_size, shuffle=False)

    y_mean     = y_mean.to(device)
    y_std      = y_std.to(device)
    x_means_SP = x_means_SP.to(device)

    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=1e-5)

    best_val_mae  = float("inf")
    patience_ctr  = 0
    best_state    = copy.deepcopy(model.state_dict())
    best_rnn_corr = copy.deepcopy(rnn_correction.state_dict()) if rnn_correction else None
    log_rows: List = []
    t_start   = time.time()

    for epoch in range(epochs):

        # ── train ─────────────────────────────────────────────────────────────
        model.train()
        if rnn_correction is not None:
            rnn_correction.train()

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            B, T, _ = x_batch.shape

            sensor_mask = torch.ones(B, NUM_SENSORS, device=device)
            x_aug = apply_mask_and_append(
                x_batch, sensor_mask, x_means_SP, NUM_TIME_POINTS, NUM_SENSORS
            )

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
                model, model_type, x_aug, ctx_idx, ctx_y, tar_idx, tar_y,
                beta=1.0, rnn_correction=rnn_correction,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

        # ── validation ────────────────────────────────────────────────────────
        model.eval()
        if rnn_correction is not None:
            rnn_correction.eval()

        val_mae_acc, n_val = 0.0, 0
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
                    T, holdout_frac, es_ctx_frac, device
                )
                ctx_y = y_norm[:, ctx_idx, :]

                y_pred_norm, *_ = model_forward(
                    model, model_type, x_aug, ctx_idx, ctx_y, tar_idx,
                    rnn_correction=rnn_correction,
                )
                y_pred   = y_pred_norm * y_std + y_mean
                mae      = F.l1_loss(y_pred, y_batch[:, tar_idx, :], reduction="mean").item()
                val_mae_acc += mae
                n_val       += 1

        val_mae = val_mae_acc / max(1, n_val)
        log_rows.append([epoch + 1, val_mae, time.time() - t_start])

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_ctr = 0
            best_state   = copy.deepcopy(model.state_dict())
            if rnn_correction is not None:
                best_rnn_corr = copy.deepcopy(rnn_correction.state_dict())
            torch.save(
                {
                    "model":          best_state,
                    "rnn_correction": best_rnn_corr,
                    "lora_target":    lora_target,
                    "rank":           rank,
                    "alpha":          alpha,
                },
                save_dir / "lora_checkpoint.pth.tar",
            )
        else:
            patience_ctr += 1

        if patience_ctr >= patience:
            print(
                f"    Early stop  epoch={epoch+1}  "
                f"best_val_mae={best_val_mae:.4f} m"
            )
            break

    total_time = time.time() - t_start
    model.load_state_dict(best_state)
    if rnn_correction is not None and best_rnn_corr is not None:
        rnn_correction.load_state_dict(best_rnn_corr)

    with open(save_dir / "lora_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "val_mae_m", "elapsed_s"])
        w.writerows(log_rows)

    return {
        "best_val_mae":       best_val_mae,
        "total_time_s":       total_time,
        "n_epochs":           len(log_rows),
        "n_traj_used":        len(ft_data),
        "n_trainable_params": n_trainable,
        "rnn_correction":     rnn_correction,
    }


# =============================================================================
# Evaluation (identical protocol to finetune_ood.py)
# =============================================================================

def evaluate_model(
    model:          nn.Module,
    model_type:     str,
    loader:         DataLoader,
    y_mean:         torch.Tensor,
    y_std:          torch.Tensor,
    x_means_SP:     torch.Tensor,
    context_fracs:  List[float],
    device:         str | torch.device,
    holdout_frac:   float = 0.2,
    rnn_correction: Optional[RNNLoRACorrection] = None,
) -> Dict[float, float]:
    y_mean     = y_mean.to(device)
    y_std      = y_std.to(device)
    x_means_SP = x_means_SP.to(device)

    mae_sums  = {f: 0.0 for f in context_fracs}
    n_batches = 0

    model.eval()
    if rnn_correction is not None:
        rnn_correction.eval()

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
                    model, model_type, x_aug, ctx_idx, ctx_y, tar_idx,
                    rnn_correction=rnn_correction,
                )
                y_pred = y_pred_norm * y_std + y_mean
                mae    = F.l1_loss(
                    y_pred, y_batch[:, tar_idx, :], reduction="mean"
                ).item()
                mae_sums[frac] += mae

            n_batches += 1

    return {f: mae_sums[f] / max(1, n_batches) for f in context_fracs}


# =============================================================================
# Plotting
# =============================================================================

def _cfg_label(rank: int, alpha: float) -> str:
    return f"r={rank} α={alpha:.0f}"


def _cfg_color(rank: int, alpha_ratio: float, ranks: List[int], alpha_ratios: List[float]) -> str:
    """Generate a color from a 2D grid: rank → hue, alpha_ratio → lightness."""
    base_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    ri = ranks.index(rank) if rank in ranks else 0
    ai = alpha_ratios.index(alpha_ratio) if alpha_ratio in alpha_ratios else 0
    # use alpha_ratio to shift lightness slightly
    base = base_colors[ri % len(base_colors)]
    return base if ai == 0 else base.replace("#", "#") + "aa"  # slight alpha


def plot_lora_mae_vs_ntraj(
    results:       Dict,
    ood_by_frac:   Dict[float, float],
    oracle_by_frac: Dict[float, float],
    lora_target:   str,
    n_traj_values: List,
    context_frac:  float,
    ranks:         List[int],
    alpha_ratios:  List[float],
    save_path:     Path,
) -> None:
    fig, ax = plt.subplots(figsize=(16, 8))
    ticks  = list(range(len(n_traj_values)))
    labels = [str(n) for n in n_traj_values]

    ood    = ood_by_frac[context_frac]
    oracle = oracle_by_frac[context_frac]
    ax.axhline(ood,    color="#e74c3c", ls="--", lw=1.8, label="OoD baseline")
    ax.axhline(oracle, color="#2c3e50", ls="--", lw=1.8, label="Oracle")
    ax.fill_between([-0.3, len(n_traj_values) - 0.7], oracle, ood,
                    alpha=0.07, color="#e74c3c")

    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_idx  = 0
    for rank in ranks:
        for ar in alpha_ratios:
            alpha = rank * ar
            y = [
                results.get((lora_target, rank, alpha, str(n)), {}).get(
                    context_frac, float("nan")
                )
                for n in n_traj_values
            ]
            ax.plot(ticks, y, marker="o",
                    color=prop_cycle[color_idx % len(prop_cycle)],
                    label=_cfg_label(rank, alpha),
                    linewidth=2, markersize=6)
            color_idx += 1

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlabel("N trajectories for fine-tuning", fontsize=PLOT_AXIS_LABEL_SIZE)
    ax.set_ylabel("MAE (m)", fontsize=PLOT_AXIS_LABEL_SIZE)
    if PLOT_SHOW_TITLES:
        ax.set_title(
            f"LoRA MAE vs data budget\n"
            f"target={TARGET_LABELS[lora_target]} | ctx={int(context_frac*100)}%",
            fontsize=PLOT_TITLE_SIZE,
        )
    ax.legend(fontsize=PLOT_LEGEND_SIZE, ncol=2)
    ax.grid(alpha=0.3)
    ax.tick_params(axis="both", labelsize=PLOT_TICK_LABEL_SIZE)
    # Adjust y-axis bottom to be closer to the Oracle (with a small margin)
    try:
        y_min_vals: list = []
        for rank in ranks:
            for ar in alpha_ratios:
                alpha = rank * ar
                for n in n_traj_values:
                    mae = results.get((lora_target, rank, alpha, str(n)), {}).get(
                        context_frac, float("nan")
                    )
                    if np.isfinite(mae):
                        y_min_vals.append(mae)
        
        if y_min_vals:
            min_all    = min(y_min_vals)
            gap        = ood - oracle
            bottom_ref = min(min_all, oracle)
            margin     = max(0.05 * gap, 0.1) if gap > 0 else 0.1
            y_bottom   = max(0.0, bottom_ref - margin)
            ax.set_ylim(bottom=y_bottom)
    except Exception:
        ax.set_ylim(bottom=0)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  [plot] {save_path}")


def plot_lora_gap_closed(
    results:       Dict,
    ood_by_frac:   Dict[float, float],
    oracle_by_frac: Dict[float, float],
    lora_target:   str,
    n_traj_values: List,
    context_frac:  float,
    ranks:         List[int],
    alpha_ratios:  List[float],
    save_path:     Path,
) -> None:
    ood    = ood_by_frac[context_frac]
    oracle = oracle_by_frac[context_frac]
    gap    = ood - oracle
    if gap <= 0:
        return

    fig, ax = plt.subplots(figsize=(16, 8))
    ticks  = list(range(len(n_traj_values)))
    labels = [str(n) for n in n_traj_values]

    ax.axhline(100.0, color="#2c3e50", ls="--", lw=1.5, label="Oracle (100%)")
    ax.axhline(  0.0, color="#e74c3c", ls="--", lw=1.5, label="OoD baseline (0%)")

    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_idx  = 0
    for rank in ranks:
        for ar in alpha_ratios:
            alpha = rank * ar
            y = []
            for n in n_traj_values:
                mae = results.get((lora_target, rank, alpha, str(n)), {}).get(
                    context_frac, float("nan")
                )
                pct = 100.0 * (ood - mae) / gap if np.isfinite(mae) else float("nan")
                y.append(pct)
            ax.plot(ticks, y, marker="o",
                    color=prop_cycle[color_idx % len(prop_cycle)],
                    label=_cfg_label(rank, alpha),
                    linewidth=2, markersize=6)
            color_idx += 1

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlabel("N trajectories", fontsize=PLOT_AXIS_LABEL_SIZE)
    ax.set_ylabel("% OoD gap closed", fontsize=PLOT_AXIS_LABEL_SIZE)
    if PLOT_SHOW_TITLES:
        ax.set_title(
            f"LoRA gap closure\n"
            f"target={TARGET_LABELS[lora_target]} | ctx={int(context_frac*100)}%",
            fontsize=PLOT_TITLE_SIZE,
        )
    ax.legend(fontsize=PLOT_LEGEND_SIZE, ncol=2)
    ax.grid(alpha=0.3)
    ax.tick_params(axis="both", labelsize=PLOT_TICK_LABEL_SIZE)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  [plot] {save_path}")


def plot_lora_heatmap(
    results:       Dict,
    ood_by_frac:   Dict[float, float],
    oracle_by_frac: Dict[float, float],
    lora_target:   str,
    n_traj_values: List,
    context_frac:  float,
    ranks:         List[int],
    alpha_ratios:  List[float],
    save_path:     Path,
) -> None:
    """Heatmap of % gap closed — rows=rank, cols=n_traj, panels per alpha_ratio."""
    ood    = ood_by_frac[context_frac]
    oracle = oracle_by_frac[context_frac]
    gap    = ood - oracle
    if gap <= 0:
        return

    n_panels = len(alpha_ratios)
    # Use GridSpec to allocate space for heatmaps and colorbar separately
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(max(7, 2.5*len(n_traj_values)) * n_panels + 1.5, 4))
    gs = gridspec.GridSpec(1, n_panels + 1, figure=fig, width_ratios=[1.0]*n_panels + [0.03],
                           wspace=0.18, hspace=0.3)

    axes = [fig.add_subplot(gs[0, i]) for i in range(n_panels)]
    cbar_ax = fig.add_subplot(gs[0, n_panels])

    im_last = None  # store last image for shared colorbar
    for panel_idx, ar in enumerate(alpha_ratios):
        ax  = axes[panel_idx]
        mat = np.full((len(ranks), len(n_traj_values)), np.nan)
        for ri, rank in enumerate(ranks):
            alpha = rank * ar
            for ni, n in enumerate(n_traj_values):
                mae = results.get((lora_target, rank, alpha, str(n)), {}).get(
                    context_frac, float("nan")
                )
                if np.isfinite(mae):
                    mat[ri, ni] = 100.0 * (ood - mae) / gap

        im_last = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(n_traj_values)))
        ax.set_xticklabels([str(n) for n in n_traj_values], fontsize=PLOT_TICK_LABEL_SIZE)
        ax.set_yticks(range(len(ranks)))
        ax.set_yticklabels([f"r={r}" for r in ranks], fontsize=PLOT_TICK_LABEL_SIZE, rotation=45)
        ax.set_xlabel("N trajectories", fontsize=PLOT_AXIS_LABEL_SIZE)
        if PLOT_SHOW_TITLES:
            ax.set_title(f"α/r = {ar:.1f}", fontsize=PLOT_TITLE_SIZE)

        for ri in range(len(ranks)):
            for ni in range(len(n_traj_values)):
                txt = f"{mat[ri,ni]:.0f}%" if np.isfinite(mat[ri,ni]) else "—"
                color = "black" if 20 < mat[ri,ni] < 80 else "white" \
                        if np.isfinite(mat[ri,ni]) else "gray"
                ax.text(ni, ri, txt, ha="center", va="center",
                    fontsize=PLOT_TEXT_SIZE, color=color)

    # single shared colorbar in dedicated axes
    if im_last is not None:
        cbar = fig.colorbar(im_last, cax=cbar_ax, label="% gap closed")
        try:
            cbar.ax.tick_params(labelsize=PLOT_TICK_LABEL_SIZE)
            cbar.set_label('% gap closed', fontsize=PLOT_AXIS_LABEL_SIZE)
        except Exception:
            pass

    axes[0].set_ylabel("LoRA rank", fontsize=PLOT_AXIS_LABEL_SIZE)
    if PLOT_SHOW_TITLES:
        fig.suptitle(
            f"LoRA gap closure heatmap — {TARGET_LABELS[lora_target]} | ctx={int(context_frac*100)}%",
            fontsize=PLOT_TITLE_SIZE,
        )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [plot] {save_path}")


def plot_lora_pareto(
    results:       Dict,
    time_results:  Dict,
    ood_by_frac:   Dict[float, float],
    oracle_by_frac: Dict[float, float],
    lora_target:   str,
    n_traj_values: List,
    context_frac:  float,
    ranks:         List[int],
    alpha_ratios:  List[float],
    save_path:     Path,
) -> None:
    ood    = ood_by_frac[context_frac]
    oracle = oracle_by_frac[context_frac]

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axhline(ood,    color="#e74c3c", ls="--", lw=1.2, alpha=0.75, label="OoD baseline")
    ax.axhline(oracle, color="#2c3e50", ls="--", lw=1.2, alpha=0.75, label="Oracle")

    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_idx  = 0
    for rank in ranks:
        for ar in alpha_ratios:
            alpha = rank * ar
            xs, ys, lbls = [], [], []
            for n in n_traj_values:
                key = (lora_target, rank, alpha, str(n))
                t   = time_results.get(key, float("nan"))
                m   = results.get(key, {}).get(context_frac, float("nan"))
                if np.isfinite(t) and np.isfinite(m):
                    xs.append(t)
                    ys.append(m)
                    lbls.append(str(n))
            if not xs:
                continue
            color = prop_cycle[color_idx % len(prop_cycle)]
            ax.plot(xs, ys, "-o", color=color,
                    label=_cfg_label(rank, alpha),
                    linewidth=1.5, markersize=7)
            for x, y, lbl in zip(xs, ys, lbls):
                ax.annotate(lbl, (x, y), textcoords="offset points",
                            xytext=(4, 4), fontsize=PLOT_TEXT_SIZE, alpha=0.85)
            color_idx += 1

    ax.set_xlabel("Fine-tuning time (s)", fontsize=PLOT_AXIS_LABEL_SIZE)
    ax.set_ylabel(f"MAE (m) — ctx={int(context_frac*100)}%", fontsize=PLOT_AXIS_LABEL_SIZE)
    if PLOT_SHOW_TITLES:
        ax.set_title(
            f"LoRA Pareto: time vs. quality\n{TARGET_LABELS[lora_target]}",
            fontsize=PLOT_TITLE_SIZE,
        )
    ax.legend(fontsize=PLOT_LEGEND_SIZE, ncol=2)
    ax.grid(alpha=0.3)
    ax.tick_params(axis="both", labelsize=PLOT_TICK_LABEL_SIZE)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  [plot] {save_path}")


# =============================================================================
# Main experiment orchestration
# =============================================================================

def run_experiment(
    topology:      str,
    model_type:    str,
    args:          argparse.Namespace,
    context_fracs: List[float],
    ranks:         List[int],
    alpha_ratios:  List[float],
) -> None:
    print(f"\n{'='*72}")
    print(f"  Topology : {topology}    Model : {model_type.upper()}")
    print(f"{'='*72}")

    out_base = Path(args.output_dir) / f"topology_{topology}" / f"model_{model_type}"
    out_base.mkdir(parents=True, exist_ok=True)

    # ── 1. Load target domain data ────────────────────────────────────────────
    print("\n[data] Loading low-variance data (target domain)…")
    lv_train, lv_val, lv_test = load_topology_data(args.lowvar_data, topology)
    eval_data = lv_test if lv_test is not None else lv_val
    print(f"  train={len(lv_train)}  val={len(lv_val)}  "
          f"{'test='+str(len(lv_test)) if lv_test else 'test=N/A'}")

    y_mean_lv, y_std_lv = compute_y_stats(lv_train)
    x_means_lv = compute_x_means(lv_train, NUM_TIME_POINTS, NUM_SENSORS)
    val_loader  = make_dataloader(lv_val,   args.batch_size)
    eval_loader = make_dataloader(eval_data, args.batch_size)

    # ── 2. Load models ────────────────────────────────────────────────────────
    print("\n[model] Loading HIGH-VAR source model…")
    hv_study = f"{model_type}_masked_highvar_{topology}_{args.study_version}"
    hv_dir   = resolve_optuna_best_model_dir(
        results_dir=args.optuna_root, study_name=hv_study,
        model_type=model_type, version=args.study_version,
    )
    hv_model, _, hv_meta = load_optuna_best_model(
        best_model_dir=hv_dir, topology=topology, model_type=model_type,
        num_sensors=NUM_SENSORS, num_time_points=NUM_TIME_POINTS,
        output_dim=OUTPUT_DIM, device=args.device,
    )
    print(f"  Loaded: {hv_dir}  (trial={hv_meta['trial_number'] if hv_meta else '?'})")

    print("[model] Loading LOW-VAR oracle model…")
    lv_study = f"{model_type}_masked_lowvar_{topology}_{args.study_version}"
    lv_dir   = resolve_optuna_best_model_dir(
        results_dir=args.optuna_root, study_name=lv_study,
        model_type=model_type, version=args.study_version,
    )
    lv_model, _, _ = load_optuna_best_model(
        best_model_dir=lv_dir, topology=topology, model_type=model_type,
        num_sensors=NUM_SENSORS, num_time_points=NUM_TIME_POINTS,
        output_dim=OUTPUT_DIM, device=args.device,
    )

    # ── 3. Baselines ──────────────────────────────────────────────────────────
    print("\n[baseline] OoD baseline…")
    ood_by_frac = evaluate_model(
        hv_model, model_type, eval_loader,
        y_mean_lv, y_std_lv, x_means_lv,
        context_fracs, args.device, args.holdout_frac,
    )
    print(f"  mean MAE = {np.mean(list(ood_by_frac.values())):.4f} m")

    print("[baseline] Oracle…")
    oracle_by_frac = evaluate_model(
        lv_model, model_type, eval_loader,
        y_mean_lv, y_std_lv, x_means_lv,
        context_fracs, args.device, args.holdout_frac,
    )
    print(f"  mean MAE = {np.mean(list(oracle_by_frac.values())):.4f} m")

    with open(out_base / "baselines.json", "w") as f:
        json.dump({
            "ood_baseline":  {str(k): v for k, v in ood_by_frac.items()},
            "oracle":        {str(k): v for k, v in oracle_by_frac.items()},
        }, f, indent=2)

    # ── 4. Select targets for this model ─────────────────────────────────────
    lora_targets = [
        t for t in args.lora_targets
        if (model_type == "anp"  and t in ANP_LORA_TARGETS) or
           (model_type == "ranp" and t in RANP_LORA_TARGETS)
    ]
    if not lora_targets:
        print(f"  No valid LoRA targets for model_type={model_type}, skipping.")
        return

    print(f"\n[lora] Targets for {model_type.upper()}: {lora_targets}")

    n_traj_values = args.n_traj
    results:      Dict = {}   # (target, rank, alpha, n_str) → {frac: mae}
    time_results: Dict = {}   # (target, rank, alpha, n_str) → seconds
    summary_rows: List = []

    for lora_target in lora_targets:
        print(f"\n  ── Target: {lora_target} ──")

        total_cfgs = len(ranks) * len(alpha_ratios) * len(n_traj_values)
        cfg_bar = tqdm(
            total=total_cfgs,
            desc=f"{model_type}:{lora_target}",
            unit="cfg",
            leave=True,
        )

        for rank in ranks:
            for ar in alpha_ratios:
                alpha = rank * ar

                for n in n_traj_values:
                    n_str   = str(n)
                    cfg_key = (lora_target, rank, alpha, n_str)
                    cfg_dir = (
                        out_base / "checkpoints" / lora_target
                        / f"rank_{rank}_alpha_{alpha:.0f}" / f"n_traj_{n_str}"
                    )
                    ckpt = cfg_dir / "lora_checkpoint.pth.tar"

                    cfg_bar.set_postfix(rank=rank, alpha=f"{alpha:.0f}", n=n_str)

                    if args.skip_existing and ckpt.exists():
                        eval_json = cfg_dir / "eval_results.json"
                        if eval_json.exists():
                            with open(eval_json) as ef:
                                saved = json.load(ef)
                            results[cfg_key]      = {float(k): v for k, v in saved["mae_by_frac"].items()}
                            time_results[cfg_key] = saved["total_time_s"]
                            print(f"    [skip] {lora_target}  r={rank}  α={alpha:.0f}  n={n_str}")
                            cfg_bar.update(1)
                            continue

                    print(f"    r={rank}  α={alpha:.0f}  n={n_str}")

                    # fresh copy of source model for each config
                    model_copy = copy.deepcopy(hv_model)
                    model_copy.to(args.device)

                    ft_meta = run_lora_finetune(
                        model        = model_copy,
                        model_type   = model_type,
                        lora_target  = lora_target,
                        rank         = rank,
                        alpha        = alpha,
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
                        save_dir     = cfg_dir,
                        holdout_frac = args.holdout_frac,
                        es_ctx_frac  = args.es_context_frac,
                        seed         = args.seed,
                    )

                    cfg_bar.set_postfix(
                        rank=rank,
                        alpha=f"{alpha:.0f}",
                        n=n_str,
                        time_s=f"{ft_meta['total_time_s']:.1f}",
                    )
                    cfg_bar.update(1)

                    rnn_corr = ft_meta.pop("rnn_correction")

                    model_copy.eval()
                    ft_eval = evaluate_model(
                        model_copy, model_type, eval_loader,
                        y_mean_lv, y_std_lv, x_means_lv,
                        context_fracs, args.device, args.holdout_frac,
                        rnn_correction=rnn_corr,
                    )

                    results[cfg_key]      = ft_eval
                    time_results[cfg_key] = ft_meta["total_time_s"]

                    with open(cfg_dir / "eval_results.json", "w") as ef:
                        json.dump({
                            "mae_by_frac":  {str(k): v for k, v in ft_eval.items()},
                            "total_time_s": ft_meta["total_time_s"],
                        }, ef, indent=2)

                    mean_ood    = np.mean(list(ood_by_frac.values()))
                    mean_oracle = np.mean(list(oracle_by_frac.values()))
                    mean_ft     = np.mean(list(ft_eval.values()))
                    gap         = mean_ood - mean_oracle
                    pct_closed  = 100.0 * (mean_ood - mean_ft) / max(gap, 1e-6)

                    print(
                        f"      → MAE={mean_ft:.4f} m  |  "
                        f"gap={pct_closed:.1f}%  |  "
                        f"time={ft_meta['total_time_s']:.1f}s  |  "
                        f"epochs={ft_meta['n_epochs']}"
                    )

                    row = {
                        "topology":           topology,
                        "model_type":         model_type,
                        "lora_target":        lora_target,
                        "rank":               rank,
                        "alpha":              alpha,
                        "alpha_ratio":        ar,
                        "n_traj":             n_str,
                        "n_traj_used":        ft_meta["n_traj_used"],
                        "n_trainable_params": ft_meta["n_trainable_params"],
                        "total_time_s":       round(ft_meta["total_time_s"], 2),
                        "n_epochs":           ft_meta["n_epochs"],
                        "best_val_mae":       round(ft_meta["best_val_mae"], 6),
                        "test_mean_mae":      round(mean_ft,  6),
                        "ood_baseline_mean":  round(mean_ood, 6),
                        "oracle_mean":        round(mean_oracle, 6),
                        "gap_closed_pct":     round(pct_closed, 2),
                        **{
                            f"test_mae_ctx{int(f*100)}": round(ft_eval[f], 6)
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

        cfg_bar.close()

    # ── 5. Save CSV ───────────────────────────────────────────────────────────
    if summary_rows:
        csv_path = out_base / "lora_summary.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\n[csv] Saved → {csv_path}")

    # ── 6. Plots ──────────────────────────────────────────────────────────────
    plots_dir  = out_base / "plots"
    main_frac  = args.context_frac

    for lora_target in lora_targets:
        t_ranks  = [t for t in targets_present(results, lora_target)]
        for frac in context_fracs:
            pct = int(frac * 100)
            plot_lora_mae_vs_ntraj(
                results, ood_by_frac, oracle_by_frac,
                lora_target, n_traj_values, frac, ranks, alpha_ratios,
                plots_dir / lora_target / f"mae_vs_ntraj_ctx{pct}.png",
            )
            plot_lora_gap_closed(
                results, ood_by_frac, oracle_by_frac,
                lora_target, n_traj_values, frac, ranks, alpha_ratios,
                plots_dir / lora_target / f"gap_closed_ctx{pct}.png",
            )
            plot_lora_heatmap(
                results, ood_by_frac, oracle_by_frac,
                lora_target, n_traj_values, frac, ranks, alpha_ratios,
                plots_dir / lora_target / f"heatmap_ctx{pct}.png",
            )
        plot_lora_pareto(
            results, time_results, ood_by_frac, oracle_by_frac,
            lora_target, n_traj_values, main_frac, ranks, alpha_ratios,
            plots_dir / lora_target / f"pareto_ctx{int(main_frac*100)}.png",
        )


def targets_present(results: Dict, lora_target: str) -> List[int]:
    return sorted(set(rank for (t, rank, *_) in results if t == lora_target))


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Explore LoRA configurations for OoD fine-tuning of ANP/RANP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # paths
    p.add_argument("--optuna-root",   required=True,
                   help="Root of Optuna results (contains anp/ and ranp/ subdirs).")
    p.add_argument("--lowvar-data",   required=True,
                   help="Root of low-variance processed data.")
    p.add_argument("--output-dir",    default="results/explore_lora")

    # scope
    p.add_argument("--topologies",    default="ellipsoidal",
                   help="Comma-separated: aligned,ellipsoidal,random.")
    p.add_argument("--model-types",   default="anp",
                   help="Comma-separated: anp,ranp.")
    p.add_argument("--study-version", default="v2")

    # LoRA config sweep
    p.add_argument(
        "--lora-targets",
        default=",".join(ANP_LORA_TARGETS + RANP_LORA_TARGETS),
        help=(
            "Comma-separated LoRA targets. "
            "ANP-only: lora_det_full, lora_det_last. "
            "RANP-only: lora_anp_base, lora_rnn_out, lora_full."
        ),
    )
    p.add_argument("--ranks",        default="4,8,16",
                   help="Comma-separated LoRA ranks to sweep.")
    p.add_argument("--alpha-ratios", default="1.0,2.0",
                   help="Comma-separated alpha/rank ratios (alpha = rank × ratio).")

    # data
    p.add_argument("--n-traj",       default="100,200,all",
                   help="Comma-separated data budgets.")

    # training
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--epochs",       type=int,   default=500)
    p.add_argument("--patience",     type=int,   default=50)
    p.add_argument("--batch-size",   type=int,   default=8)
    p.add_argument("--seed",         type=int,   default=0, help="Random seed (single seed; for multi-seed use the full finetune_ood.py).")

    # evaluation
    p.add_argument("--holdout-frac",   type=float, default=0.2)
    p.add_argument("--es-context-frac",type=float, default=0.4)
    p.add_argument("--context-frac",   type=float, default=0.3,
                   help="Primary context fraction for summary plots.")
    p.add_argument("--context-fracs",  default="0.1,0.2,0.3,0.4,0.5,0.6")

    # misc
    p.add_argument("--device",       default="cuda")
    p.add_argument("--skip-existing",action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    args.topologies   = [t.strip() for t in args.topologies.split(",")   if t.strip()]
    args.model_types  = [m.strip() for m in args.model_types.split(",")  if m.strip()]
    args.lora_targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()]

    ranks        = [int(r.strip())   for r in args.ranks.split(",")       if r.strip()]
    alpha_ratios = [float(a.strip()) for a in args.alpha_ratios.split(",") if a.strip()]

    raw_n = [x.strip() for x in args.n_traj.split(",") if x.strip()]
    args.n_traj = ["all" if x.lower() == "all" else int(x) for x in raw_n]

    context_fracs = [float(x) for x in args.context_fracs.split(",") if x.strip()]
    if args.context_frac not in context_fracs:
        context_fracs = sorted(set(context_fracs + [args.context_frac]))

    # validate targets
    for t in args.lora_targets:
        if t not in ALL_LORA_TARGETS:
            raise ValueError(f"Unknown lora_target '{t}'. Valid: {ALL_LORA_TARGETS}")

    n_configs = len(ranks) * len(alpha_ratios) * len(args.n_traj)

    print("=" * 72)
    print("  explore_lora_ood.py — LoRA configuration sweep")
    print("=" * 72)
    print(f"  Topologies    : {args.topologies}")
    print(f"  Model types   : {args.model_types}")
    print(f"  LoRA targets  : {args.lora_targets}")
    print(f"  Ranks         : {ranks}")
    print(f"  Alpha ratios  : {alpha_ratios}")
    print(f"  Data budgets  : {args.n_traj}")
    print(f"  Configs/target: {n_configs}")
    print(f"  Ctx fracs     : {context_fracs}")
    print(f"  Device        : {args.device}")
    print(f"  Output dir    : {args.output_dir}")
    print()

    for topology in args.topologies:
        for model_type in args.model_types:
            try:
                run_experiment(
                    topology      = topology,
                    model_type    = model_type,
                    args          = args,
                    context_fracs = context_fracs,
                    ranks         = ranks,
                    alpha_ratios  = alpha_ratios,
                )
            except FileNotFoundError as exc:
                print(f"\n[skip] {topology}/{model_type}: {exc}\n")

    print(f"\n{'='*72}")
    print(f"  Done.  Results in: {args.output_dir}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
