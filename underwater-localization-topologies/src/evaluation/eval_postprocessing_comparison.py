#!/usr/bin/env python3
"""
eval_postprocessing_comparison.py
================================
Benchmark of causal trajectory postprocessors applied to ANP/RANP predictions.

What this script does
---------------------
1) Loads one or more models (ANP or RANP) from either:
     - a direct checkpoint (--ckpt), or
     - an Optuna best_model directory (--optuna-best-model-dir), selecting the
          topology-specific checkpoint automatically, or
      - an Optuna root directory (--optuna-root-dir), auto-discovering all
          studies/best_model folders recursively.

2) Evaluates across topologies:
    - default: aligned, ellipsoidal, random
    - optional: only one topology with --single-topology --topology <name>
    - in auto mode, if topology is inferred from study name, it evaluates that
        topology for that model by default.

3) Runs protocol-aware evaluation with strict separation between protocols:
     - holdout: context from the first points; target is the final holdout tail
     - inverse_holdout: context from the block immediately before holdout tail;
         target remains the same final holdout tail
     - both_holdouts: runs both protocols independently

4) Performs inference over the full trajectory length so postprocessors operate
     on complete temporal sequences, while MAE is computed only on protocol target
     points via an explicit target mask.

5) Tunes postprocessor hyperparameters with random search on validation data,
     then evaluates the selected configuration on the test split.

6) In auto mode, infers metadata from study names/paths when possible:
    - model type (anp/ranp)
    - version (v1, v2, ...)
    - data variant (lowvar/highvar)
    - topology (aligned/random/ellipsoidal)
   and selects the data directory via --data-dir-lowvar / --data-dir-highvar.

Postprocessors compared
-----------------------
1. Raw          - no postprocessing (baseline)
2. EMA          - exponential moving average                      [alpha]
3. EMA-Var      - variance-weighted adaptive EMA                 [sigma_ref, alpha_min, alpha_max]
4. EKF          - Kalman filter, constant-velocity model         [sigma_a, R_scale, dt, init_P]
5. UKF          - Unscented Kalman filter (CV)                   [sigma_a, R_scale, dt, init_P, ukf_alpha]
6. Mahalanobis  - gated KF with outlier rejection                [mahal_thresh, sigma_a, R_scale, dt, init_P]
7. BiasAR       - decaying bias correction from context residuals [rho]
8. AR-p         - AR(p) residual correction on context residuals [p, ridge]

All methods are strictly causal (use only information up to current timestep).
EKF/UKF/Mahalanobis consume model-predicted variance as measurement noise.

Outputs
-------
Results are saved under:
    --output-dir/version_<vX>/topology_<topology>/<lowvar|highvar>/<model_name>/protocol_<protocol>/

Per protocol and topology, artifacts include:
    - comparison_report_<topology>_<protocol>.txt
    - mae_boxplot_<topology>_<protocol>.png
    - pareto_mae_latency_<topology>_<protocol>.png
    - traj_plots_<topology>_<protocol>/
    - protocol-specific inference caches (_cache_val_*.pkl, _cache_test_*.pkl)

This guarantees holdout and inverse_holdout outputs are never mixed, enabling
direct side-by-side Pareto and report comparisons.

Usage examples
--------------
Run full evaluation (all topologies, both protocols) from Optuna best_model:
    cd /home/fernando/tesis/underwater-localization-topologies/src/evaluation

    python eval_postprocessing_comparison.py \
      --optuna-best-model-dir /home/fernando/tesis/underwater-localization-topologies/src/training/results/optuna/anp/v1/anp_masked_lowvar_ellipsoidal_v1/best_model \
      --model-type anp \
      --data-dir /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_low_variance \
      --output-dir /home/fernando/tesis/underwater-localization-topologies/src/evaluation/results/postprocessing/full_run_anp_lowvar \
      --ctx-frac 0.3 \
      --holdout-frac 0.2 \
      --n-hparam-trials 50 \
      --seed 18

Run one topology only:
    python eval_postprocessing_comparison.py \
            --optuna-best-model-dir <.../best_model> \
            --model-type ranp \
            --data-dir <.../data_processed_topologies_*> \
            --single-topology --topology ellipsoidal

Additional models can be evaluated with --extra-configs as a JSON list of
ModelConfig-compatible dictionaries.

Run automatic discovery from Optuna root (all discovered models):
        python eval_postprocessing_comparison.py \
            --optuna-root-dir /home/fernando/tesis/underwater-localization-topologies/src/training/results/optuna \
            --versions v2 \
            --data-dir-lowvar /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_low_variance \
            --data-dir-highvar /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_high_variance \
            --output-dir /home/fernando/tesis/underwater-localization-topologies/src/evaluation/results/postprocessing/full_auto_optuna \
            --ctx-frac 0.3 \
            --holdout-frac 0.2 \
            --n-hparam-trials 50 \
            --seed 18
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ── make project root importable ─────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import src.models.anp as anp_module
import src.models.r_anp as ranp_module
from src.utils.load_optuna_model import load_optuna_best_model
from src.utils.nav_dataset import NavigationTrajectoryDataset

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════
NUM_SENSORS    = 10
NUM_TIME_PTS   = 201
INPUT_DIM_BASE = NUM_TIME_PTS * NUM_SENSORS          # 2010  (raw acoustic features)
INPUT_DIM      = INPUT_DIM_BASE + NUM_SENSORS        # 2020  (+ mask bits)
OUTPUT_DIM     = 3                                   # x, z, (depth/heading)
XZ_DIMS        = slice(0, 2)                         # dimensions we filter in x-z plane


# ═══════════════════════════════════════════════════════════════════════════════
# Model configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    """Container for a model variant to evaluate."""
    name: str
    ckpt_path: str = "" # path to best_checkpoint.pth.tar
    optuna_best_model_dir: str = "" # path to .../best_model
    model_type: str = "anp" # "anp" | "ranp"
    num_hidden: int = 128
    # RANP-specific
    rnn_type: str = "lstm"
    rnn_layers: int = 1
    rnn_dropout: float = 0.0
    # Optional metadata used by auto-discovery mode
    version: str = ""
    data_variant: str = ""      # "lowvar" | "highvar"
    preferred_topology: str = "" # "aligned" | "ellipsoidal" | "random"


def _infer_num_hidden(hparams_path: str, fallback: int = 128) -> int:
    p = Path(hparams_path)
    if p.exists():
        with open(p) as f:
            hp = json.load(f)
        return int(hp.get("num_hidden", fallback))
    return fallback


def _resolve_optuna_study_dir(optuna_dir_hint: str) -> Tuple[str, str]:
    """Resolve Optuna study directory and infer study name.
    
    optuna_dir_hint can point to:
      - .../best_model (most specific)
      - .../study_name
      - .../study_name/best_model (also works)
      - .../version/study_name
      - etc.
    
    Returns (best_model_dir, study_name).
    """
    hint_path = Path(optuna_dir_hint).resolve()
    
    # If it ends with /best_model, use it directly
    if hint_path.name == "best_model" and hint_path.exists():
        study_name = hint_path.parent.name
        return str(hint_path), study_name
    
    # If best_model exists as a subdirectory, use it
    best_model_candidate = hint_path / "best_model"
    if best_model_candidate.exists():
        study_name = hint_path.name
        return str(best_model_candidate), study_name
    
    # Otherwise assume hint_path is the study directory
    study_name = hint_path.name
    best_model_dir = hint_path / "best_model"
    if best_model_dir.exists():
        return str(best_model_dir), study_name
    
    # Fallback: assume hint_path is already best_model
    return str(hint_path), study_name


def _infer_model_metadata_from_study_name(
    study_name: str,
    fallback_model_type: str = "anp",
) -> Tuple[str, str, str, str]:
    """Infer (model_type, version, data_variant, topology) from study name/path.

    Typical names: anp_masked_lowvar_ellipsoidal_v1, ranp_*_highvar_random_v2
    """
    s = study_name.lower()

    model_type = "ranp" if "ranp" in s else ("anp" if "anp" in s else fallback_model_type)

    # Underscore-aware token match: captures v1 in names like ..._v1 and ..._v1_...
    m_ver = re.search(r"(?:^|_)(v\d+)(?:$|_)", s)
    version = m_ver.group(1) if m_ver else ""

    data_variant = ""
    if "lowvar" in s or "low_variance" in s:
        data_variant = "lowvar"
    elif "highvar" in s or "high_variance" in s:
        data_variant = "highvar"

    topology = ""
    for topo in ("aligned", "ellipsoidal", "random"):
        if topo in s:
            topology = topo
            break

    return model_type, version, data_variant, topology


def _discover_optuna_model_configs(optuna_root_dir: str) -> List[ModelConfig]:
    """Discover all Optuna studies under root and return ModelConfig entries.

    Expected layout (flexible):
      <root>/<model_type>/<version>/<study_name>/best_model
    but any nested location containing a best_model dir is accepted.
    """
    root = Path(optuna_root_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Optuna root does not exist: {root}")

    best_model_dirs = sorted(p for p in root.rglob("best_model") if p.is_dir())
    cfgs: List[ModelConfig] = []
    seen_names = set()

    for best_dir in best_model_dirs:
        study_dir = best_dir.parent
        study_name = study_dir.name

        # Infer model type from path first, then study name
        path_parts = [part.lower() for part in best_dir.parts]
        path_model_type = ""
        if "ranp" in path_parts:
            path_model_type = "ranp"
        elif "anp" in path_parts:
            path_model_type = "anp"

        model_type, version, data_variant, topology = _infer_model_metadata_from_study_name(
            study_name,
            fallback_model_type=path_model_type or "anp",
        )

        # Build a stable readable name and avoid duplicates
        base_name_parts = [model_type]
        if version:
            base_name_parts.append(version)
        base_name_parts.append(study_name)
        cfg_name = "_".join(base_name_parts)

        if cfg_name in seen_names:
            k = 2
            while f"{cfg_name}_{k}" in seen_names:
                k += 1
            cfg_name = f"{cfg_name}_{k}"
        seen_names.add(cfg_name)

        cfgs.append(
            ModelConfig(
                name=cfg_name,
                optuna_best_model_dir=str(best_dir),
                model_type=model_type,
                version=version,
                data_variant=data_variant,
                preferred_topology=topology,
            )
        )

    if not cfgs:
        raise FileNotFoundError(f"No 'best_model' directories found under: {root}")

    print(f"[auto ] discovered {len(cfgs)} Optuna model(s) under {root}")
    return cfgs


def _resolve_data_dir_for_model(args: argparse.Namespace, cfg: ModelConfig) -> str:
    """Select data directory for a model according to inferred data variant."""
    if cfg.data_variant == "lowvar" and args.data_dir_lowvar:
        return args.data_dir_lowvar
    if cfg.data_variant == "highvar" and args.data_dir_highvar:
        return args.data_dir_highvar
    if args.data_dir:
        return args.data_dir

    wanted = cfg.data_variant or "(unknown variant)"
    raise ValueError(
        "No data directory available for model "
        f"'{cfg.name}' (variant={wanted}). Provide --data-dir or "
        "variant-specific --data-dir-lowvar / --data-dir-highvar."
    )


def _infer_data_variant_from_data_dir(data_dir: str) -> str:
    """Infer lowvar/highvar from data dir name as fallback for output grouping."""
    s = str(data_dir).lower()
    if "lowvar" in s or "low_variance" in s:
        return "lowvar"
    if "highvar" in s or "high_variance" in s:
        return "highvar"
    return "unknown"


def load_model(cfg: ModelConfig, device: torch.device, topology: str) -> torch.nn.Module:
    """Load ANP or RANP from Optuna best_model dir or direct checkpoint."""
    if cfg.optuna_best_model_dir:
        model, hparams, _ = load_optuna_best_model(
            best_model_dir=cfg.optuna_best_model_dir,
            topology=topology,
            model_type=cfg.model_type,
            num_sensors=NUM_SENSORS,
            num_time_points=NUM_TIME_PTS,
            output_dim=OUTPUT_DIM,
            device=device,
        )
        cfg.num_hidden = int(hparams.get("num_hidden", cfg.num_hidden))
        ckpt_path = Path(cfg.optuna_best_model_dir) / f"topology_{topology}" / "best_checkpoint.pth.tar"
        print(f"[  OK  ] {cfg.model_type.upper()} '{cfg.name}' loaded from {ckpt_path} (Optuna)")
        return model

    ckpt_path = Path(cfg.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if cfg.model_type == "anp":
        model = anp_module.LatentModel(
            num_hidden=cfg.num_hidden,
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
        ).to(device)
    elif cfg.model_type == "ranp":
        model = ranp_module.LatentModel(
            num_hidden=cfg.num_hidden,
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            rnn_type=cfg.rnn_type,
            rnn_layers=cfg.rnn_layers,
            rnn_dropout=cfg.rnn_dropout,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type!r}")

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"[  OK  ] {cfg.model_type.upper()} '{cfg.name}' loaded from {ckpt_path}")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_split(data_dir: str, topology: str, split: str) -> Tuple[list, dict]:
    """Load one split (train/val/test) + metadata for a topology."""
    topo_dir = Path(data_dir) / f"topology_{topology}"
    data_path = topo_dir / f"{split}_data.pkl"
    meta_path = topo_dir / "metadata.pkl"
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return data, meta


def compute_y_stats(train_data: list, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    Y = np.concatenate([y for _, y in train_data], axis=0)
    y_mean = torch.tensor(Y.mean(axis=0), dtype=torch.float32, device=device)
    y_std  = torch.tensor(Y.std(axis=0)  + 1e-6, dtype=torch.float32, device=device)
    return y_mean, y_std


def augment_x_with_full_mask(x_batch: torch.Tensor) -> torch.Tensor:
    """Append all-ones sensor-mask features (no dropout at test time).

    x_batch : (B, T, Dx)
    returns  : (B, T, Dx + S)
    """
    B, T, _ = x_batch.shape
    mask_feat = torch.ones(B, T, NUM_SENSORS, device=x_batch.device, dtype=x_batch.dtype)
    return torch.cat([x_batch, mask_feat], dim=-1)


def group_by_theta(data: list, thetas: list) -> Dict[float, list]:
    groups: Dict[float, list] = {}
    for sample, theta in zip(data, thetas):
        groups.setdefault(float(theta), []).append(sample)
    return groups


def normalize_y(y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (y - mean.view(1, 1, -1)) / std.view(1, 1, -1)


def denormalize_y(y_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return y_norm * std.view(1, 1, -1) + mean.view(1, 1, -1)


# ═══════════════════════════════════════════════════════════════════════════════
# Inference & prediction bundles
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PredBundle:
    """Pre-computed model predictions for one trajectory."""
    mean_real:    np.ndarray   # (T, 3) de-normalised mean
    var_real:     np.ndarray   # (T, 3) de-normalised variance (sigma^2)
    gt_real:      np.ndarray   # (T, 3) ground truth
    ctx_mask:     np.ndarray   # (T,)   bool - True for context points
    target_mask:  np.ndarray   # (T,)   bool - True for evaluation target points
    theta:        float = 0.0
    infer_time_s: float = 0.0  # model forward-pass wall time for this trajectory (s)


def _build_eval_indices(
    total_points: int,
    context_frac: float,
    eval_protocol: str,
    holdout_frac: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create context and target indices for holdout / inverse_holdout protocols."""
    n_holdout = max(1, int(round(holdout_frac * total_points)))
    holdout_start = total_points - n_holdout

    if eval_protocol == "holdout":
        max_ctx = max(1, holdout_start - 1)
        ctx_size = max(1, min(max_ctx, int(round(context_frac * total_points))))
        ctx_idx = torch.arange(ctx_size, device=device)
        tar_idx = torch.arange(holdout_start, total_points, device=device)
        return ctx_idx, tar_idx

    if eval_protocol == "inverse_holdout":
        max_ctx = max(1, holdout_start)
        ctx_size = max(1, min(max_ctx, int(round(context_frac * total_points))))
        ctx_start = holdout_start - ctx_size
        ctx_idx = torch.arange(ctx_start, holdout_start, device=device)
        tar_idx = torch.arange(holdout_start, total_points, device=device)
        return ctx_idx, tar_idx

    raise ValueError(f"Unknown eval_protocol: {eval_protocol!r}")


@torch.no_grad()
def run_inference(model: torch.nn.Module, model_type: str, data: list, thetas: list, y_mean: torch.Tensor, y_std:  torch.Tensor, ctx_frac: float, eval_protocol: str, holdout_frac: float, device: torch.device,
) -> List[PredBundle]:
    """Run model inference on all trajectories, return PredBundle list."""
    bundles: List[PredBundle] = []

    ds = NavigationTrajectoryDataset(data)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    y_std_np = y_std.cpu().numpy()

    for (x_raw, y_gt), theta in zip(loader, thetas):
        x_raw = x_raw.to(device)   # (1, T, Dx)
        y_gt  = y_gt.to(device)    # (1, T, 3)
        T = x_raw.shape[1]

        # Build augmented input (append all-ones mask)
        x_aug = augment_x_with_full_mask(x_raw)   # (1, T, INPUT_DIM)

        # Context / evaluation-target split according to selected protocol.
        # Important: model prediction is run over the full trajectory (all T points) so postprocessors operate on a full sequence. 
        # The evaluation target is represented separately via target_mask.
        ctx_idx, eval_tar_idx = _build_eval_indices(
            total_points=T,
            context_frac=ctx_frac,
            eval_protocol=eval_protocol,
            holdout_frac=holdout_frac,
            device=device,
        )
        pred_idx = torch.arange(T, device=device)

        y_norm = normalize_y(y_gt, y_mean, y_std)

        ctx_y  = y_norm[:, ctx_idx, :]   # (1, Nc, 3)
        tar_y  = y_norm[:, pred_idx, :]   # kept for shape symmetry

        _t_infer_start = time.perf_counter()
        if model_type == "anp":
            ctx_x = x_aug[:, ctx_idx, :]
            tar_x = x_aug[:, pred_idx, :]
            mean_norm, var_norm, *_ = model(ctx_x, ctx_y, tar_x)
        elif model_type == "ranp":
            mean_norm, var_norm, *_ = model(x_seq=x_aug,
                context_indices=ctx_idx,
                context_y=ctx_y,
                target_indices=pred_idx,
                target_y=None,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type!r}")
        _infer_time = time.perf_counter() - _t_infer_start

        mean_real = denormalize_y(mean_norm, y_mean, y_std)[0].cpu().numpy()  # (T, 3)
        # var_norm is in normalised space; convert to real units: var_real = var_norm * y_std^2
        var_real  = (var_norm[0].cpu().numpy() * (y_std_np ** 2)) # (T, 3)

        gt_np     = y_gt[0].cpu().numpy() # (T, 3)

        ctx_mask_np = np.zeros(T, dtype=bool)
        target_mask_np = np.zeros(T, dtype=bool)
        ctx_mask_np[ctx_idx.cpu().numpy()] = True
        target_mask_np[eval_tar_idx.cpu().numpy()] = True

        bundles.append(PredBundle(mean_real=mean_real,
            var_real=var_real,
            gt_real=gt_np,
            ctx_mask=ctx_mask_np,
            target_mask=target_mask_np,
            theta=float(theta) if not isinstance(theta, float) else theta,
            infer_time_s=_infer_time,
        ))

    return bundles


# ═══════════════════════════════════════════════════════════════════════════════
# Postprocessor base class
# ═══════════════════════════════════════════════════════════════════════════════

class PostProcessor(ABC):
    """Base class for all online (causal) postprocessors.

    Subclasses must implement :meth:`apply`.
    """

    @abstractmethod
    def apply(self, bundle: PredBundle) -> np.ndarray:
        """Apply postprocessing, return filtered trajectory (T, 2) for x-z plane.

        Context points are overwritten with ground truth (position is known from initial conditioning, as in deployment).
        """

    def apply_timed(self, bundle: PredBundle) -> Tuple[np.ndarray, float]:
        """Return (filtered_xy, elapsed_seconds)."""
        t0 = time.perf_counter()
        out = self.apply(bundle)
        return out, time.perf_counter() - t0

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def params(self) -> dict: ...


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Raw (baseline)
# ═══════════════════════════════════════════════════════════════════════════════

class RawPostProcessor(PostProcessor):
    @property
    def name(self) -> str:
        return "Raw"

    @property
    def params(self) -> dict:
        return {}

    def apply(self, bundle: PredBundle) -> np.ndarray:
        out = bundle.mean_real[:, :2].copy()
        out[bundle.ctx_mask] = bundle.gt_real[bundle.ctx_mask, :2]
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# 2. EMA — exponential moving average
# ═══════════════════════════════════════════════════════════════════════════════

class EMAPostProcessor(PostProcessor):
    """Simple causal EMA: filtered[t] = alpha * raw[t] + (1-alpha) * filtered[t-1].

    alpha close to 1 → trust new prediction; alpha close to 0 → heavy smoothing.
    At context points, filtered value is forced to ground truth.
    """

    def __init__(self, alpha: float = 0.7):
        self._alpha = float(alpha)

    @property
    def name(self) -> str:
        return "EMA"

    @property
    def params(self) -> dict:
        return {"alpha": self._alpha}

    def apply(self, bundle: PredBundle) -> np.ndarray:
        raw  = bundle.mean_real[:, :2]
        ctx  = bundle.ctx_mask
        gt   = bundle.gt_real[:, :2]
        T    = raw.shape[0]

        out = np.empty_like(raw)
        # Initialise from first context point
        out[0] = gt[0] if ctx[0] else raw[0]

        for t in range(1, T):
            if ctx[t]:
                out[t] = gt[t]
            else:
                out[t] = self._alpha * raw[t] + (1.0 - self._alpha) * out[t - 1]

        return out.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. EMA-Var — variance-weighted adaptive EMA
# ═══════════════════════════════════════════════════════════════════════════════

class EMAVarPostProcessor(PostProcessor):
    """Variance-weighted EMA with velocity extrapolation to avoid positional drift.

    At each target step t:
      1. Compute a velocity-extrapolated "predicted" position from the last two filtered estimates:  predicted[t] = out[t-1] + (out[t-1] - out[t-2])
      2. Blend the raw ANP prediction with the predicted position using an adaptive weight that scales with model confidence:

         alpha(t) = clip( sigma_ref / (sigma_xz(t) + sigma_ref), alpha_min, alpha_max )

         where sigma_xz(t) = sqrt( mean(var_real[t, 0:2]) )  (std, not variance).

         Small sigma  → model is confident → alpha high → trust new prediction.
         Large sigma  → model is uncertain → alpha low  → rely on vel-extrapolated state.

    The velocity extrapolation step prevents the estimate from freezing when the model is uncertain (alpha ≈ 0), which would cause unbounded positional drift.

    Parameters
    ----------
    sigma_ref  : reference std (physical units) that controls the transition point
    alpha_min  : lower bound on alpha  (always update at least a little)
    alpha_max  : upper bound on alpha  (never fully discard predicted state)
    """

    def __init__(self, sigma_ref: float = 2.0, alpha_min: float = 0.1, alpha_max: float = 0.95):
        self._sigma_ref  = float(sigma_ref)
        self._alpha_min  = float(alpha_min)
        self._alpha_max  = float(alpha_max)

    @property
    def name(self) -> str:
        return "EMA-Var"

    @property
    def params(self) -> dict:
        return {
            "sigma_ref": self._sigma_ref,
            "alpha_min": self._alpha_min,
            "alpha_max": self._alpha_max,
        }

    def apply(self, bundle: PredBundle) -> np.ndarray:
        raw  = bundle.mean_real[:, :2]
        ctx  = bundle.ctx_mask
        gt   = bundle.gt_real[:, :2]
        var  = bundle.var_real[:, :2]     # (T, 2)  variance in real units
        T    = raw.shape[0]

        out = np.empty_like(raw)
        out[0] = gt[0] if ctx[0] else raw[0]
        if T > 1:
            out[1] = gt[1] if ctx[1] else raw[1]

        for t in range(2, T):
            if ctx[t]:
                out[t] = gt[t]
            else:
                # Velocity-extrapolated predicted position (prevents drift)
                vel       = out[t - 1] - out[t - 2]
                predicted = out[t - 1] + vel

                # Adaptive blend weight from model std (not variance)
                sigma_xz  = float(np.sqrt(var[t].mean()))
                alpha_t   = self._sigma_ref / (sigma_xz + self._sigma_ref)
                alpha_t   = float(np.clip(alpha_t, self._alpha_min, self._alpha_max))

                out[t] = alpha_t * raw[t] + (1.0 - alpha_t) * predicted

        return out.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# KF / EKF helpers (constant-velocity 2-D model)
# ═══════════════════════════════════════════════════════════════════════════════
# State: [x, z, vx, vz]  (4D)
# Transition: F (constant velocity)
# Measurement matrix: H  (observe position only)

def _cv_matrices(dt: float, sigma_a: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (F, H, Q) for a constant-velocity 2-D process."""
    F = np.array(
        [[1, 0, dt, 0],
         [0, 1, 0,  dt],
         [0, 0, 1,  0],
         [0, 0, 0,  1]], dtype=np.float64
    )
    H = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0]], dtype=np.float64
    )
    q    = sigma_a ** 2
    dt2, dt3, dt4 = dt**2, dt**3, dt**4
    Q1d  = np.array([[dt4 / 4, dt3 / 2],
                     [dt3 / 2, dt2    ]], dtype=np.float64) * q
    Q    = np.zeros((4, 4), dtype=np.float64)
    Q[np.ix_([0, 2], [0, 2])] = Q1d
    Q[np.ix_([1, 3], [1, 3])] = Q1d
    return F, H, Q


def _R_from_var(var_xz: np.ndarray, R_scale: float) -> np.ndarray:
    """Build 2x2 diagonal R matrix from xz variance (s^2) scaled by R_scale."""
    return np.diag(var_xz.astype(np.float64)) * R_scale


def _kf_predict(x: np.ndarray, P: np.ndarray,
                F: np.ndarray, Q: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray]:
    x_p = F @ x
    P_p = F @ P @ F.T + Q
    return x_p, P_p


def _kf_update(x_p: np.ndarray, P_p: np.ndarray,
               z: np.ndarray, H: np.ndarray, R: np.ndarray
               ) -> Tuple[np.ndarray, np.ndarray]:
    innov = z - H @ x_p
    S     = H @ P_p @ H.T + R
    K     = P_p @ H.T @ np.linalg.inv(S)
    x_u   = x_p + K @ innov
    KH    = K @ H
    I     = np.eye(P_p.shape[0])
    P_u   = (I - KH) @ P_p @ (I - KH).T + K @ R @ K.T  # Joseph form
    return x_u, P_u


# ═══════════════════════════════════════════════════════════════════════════════
# 4. EKF — Extended Kalman Filter (standard KF for the linear CV model)
# ═══════════════════════════════════════════════════════════════════════════════

class EKFPostProcessor(PostProcessor):
    """Online constant-velocity Kalman filter applied to ANP predictions.

    The ANP's per-step variance sigma² is used directly as measurement noise R:
        R_t = diag(var_xz[t]) * R_scale

    This is the key advantage over a classical Kalman filter with fixed R.

    At context points the filter is updated with ground truth (R = 0 / hard constraint) so the vehicle's known initial positions are honoured.

    Parameters
    ----------
    sigma_a  : process noise (acceleration std in physical units / s^2)
    R_scale  : scalar multiplier on the per-step variance → R_t = diag(var) * R_scale
    dt       : timestep (seconds), default 1.0
    init_P   : initial state covariance diagonal value
    """

    def __init__(self, sigma_a: float = 0.5, R_scale: float = 1.0,
                 dt: float = 1.0, init_P: float = 10.0):
        self._sigma_a = float(sigma_a)
        self._R_scale = float(R_scale)
        self._dt      = float(dt)
        self._init_P  = float(init_P)

    @property
    def name(self) -> str:
        return "EKF"

    @property
    def params(self) -> dict:
        return {
            "sigma_a": self._sigma_a,
            "R_scale": self._R_scale,
            "dt":      self._dt,
            "init_P":  self._init_P,
        }

    def apply(self, bundle: PredBundle) -> np.ndarray:
        raw = bundle.mean_real[:, :2]
        var = bundle.var_real[:,  :2]
        ctx = bundle.ctx_mask
        gt  = bundle.gt_real[:, :2]
        T   = raw.shape[0]

        F, H, Q = _cv_matrices(self._dt, self._sigma_a)

        # Initialise state from first available context point (or raw prediction)
        init_pos = gt[0].astype(np.float64) if ctx[0] else raw[0].astype(np.float64)
        x = np.array([init_pos[0], init_pos[1], 0.0, 0.0], dtype=np.float64)
        P = np.eye(4, dtype=np.float64) * self._init_P

        out = np.empty((T, 2), dtype=np.float64)
        out[0] = x[:2]

        for t in range(1, T):
            x, P = _kf_predict(x, P, F, Q)

            if ctx[t]:
                # Hard update with ground truth (near-zero measurement noise)
                R  = np.eye(2, dtype=np.float64) * 1e-6
                z  = gt[t].astype(np.float64)
            else:
                R  = _R_from_var(var[t], self._R_scale)
                R  = np.maximum(R, np.eye(2) * 1e-6)
                z  = raw[t].astype(np.float64)

            x, P   = _kf_update(x, P, z, H, R)
            out[t] = x[:2]

        return out.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# UKF helpers — Merwe scaled sigma points
# ═══════════════════════════════════════════════════════════════════════════════
# Note: for the linear CV model, UKF and standard KF yield identical results.
# The UKF implementation is to support possible nonlinear dynamics extensions.

def _ukf_weights(n: int, alpha: float = 1.0, kappa: float = 0.0, beta: float = 2.0
                 ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute Merwe scaled sigma-point weights.

    Returns (Wm, Wc, lambda_) where:
        Wm : (2n+1,) mean weights
        Wc : (2n+1,) covariance weights
    """
    lam    = alpha**2 * (n + kappa) - n
    c_val  = n + lam
    Wm     = np.full(2 * n + 1, 1.0 / (2.0 * c_val))
    Wc     = Wm.copy()
    Wm[0]  = lam / c_val
    Wc[0]  = lam / c_val + (1.0 - alpha**2 + beta)
    return Wm, Wc, lam


def _sigma_points(mean: np.ndarray, P: np.ndarray, lam: float) -> np.ndarray:
    """Generate 2n+1 Merwe sigma points around mean given covariance P."""
    n  = len(mean)
    c  = n + lam
    # Add a small nugget for numerical stability
    try:
        L = np.linalg.cholesky(c * P + np.eye(n) * 1e-9)
    except np.linalg.LinAlgError:
        P_sym = 0.5 * (P + P.T)
        L = np.linalg.cholesky(c * P_sym + np.eye(n) * 1e-6)

    sigmas        = np.empty((2 * n + 1, n))
    sigmas[0]     = mean
    for i in range(n):
        sigmas[i + 1]     = mean + L[:, i]
        sigmas[n + i + 1] = mean - L[:, i]
    return sigmas


def _ukf_predict(x: np.ndarray, P: np.ndarray, F: np.ndarray, Q: np.ndarray, Wm: np.ndarray, Wc: np.ndarray, lam: float
                 ) -> Tuple[np.ndarray, np.ndarray]:
    n      = len(x)
    sigmas = _sigma_points(x, P, lam)                    # (2n+1, n)
    sigmas_p = sigmas @ F.T                               # propagate: F is linear

    x_p = Wm @ sigmas_p                                   # (n,)
    diff = sigmas_p - x_p                                 # (2n+1, n)
    P_p = (Wc[:, None] * diff).T @ diff + Q              # (n, n)
    return x_p, P_p


def _ukf_update(x_p: np.ndarray, P_p: np.ndarray, z: np.ndarray, H: np.ndarray, R: np.ndarray, Wm: np.ndarray, Wc: np.ndarray, lam: float
                ) -> Tuple[np.ndarray, np.ndarray]:
    n      = len(x_p)
    sigmas = _sigma_points(x_p, P_p, lam)                # (2n+1, n)
    z_sig  = sigmas @ H.T                                 # (2n+1, 2)

    z_p    = Wm @ z_sig                                   # (2,) predicted measurement
    dz     = z_sig - z_p
    S      = (Wc[:, None] * dz).T @ dz + R               # innovation cov
    Pxz    = (Wc[:, None] * (sigmas - x_p)).T @ dz       # cross-covariance

    K      = Pxz @ np.linalg.inv(S)
    x_u    = x_p + K @ (z - z_p)
    P_u    = P_p - K @ S @ K.T
    return x_u, P_u


# ═══════════════════════════════════════════════════════════════════════════════
# 5. UKF — Unscented Kalman Filter (CV model)
# ═══════════════════════════════════════════════════════════════════════════════

class UKFPostProcessor(PostProcessor):
    """Unscented Kalman Filter with constant-velocity dynamics.

    Uses the same ANP variance → R strategy as EKFPostProcessor.
    For the linear CV model the results are numerically identical to EKF; the sigma-point machinery is for easy extension to nonlinear motion models in the future.

    Parameters
    ----------
    sigma_a   : process noise (acceleration std)
    R_scale   : multiplier on per-step ANP variance → R_t
    dt        : timestep
    init_P    : initial state covariance diagonal
    ukf_alpha : sigma-point spread parameter (alpha=1 → standard symmetric UT)
    """

    def __init__(self, sigma_a: float = 0.5, R_scale: float = 1.0, dt: float = 1.0, init_P: float = 10.0, ukf_alpha: float = 1.0):
        self._sigma_a   = float(sigma_a)
        self._R_scale   = float(R_scale)
        self._dt        = float(dt)
        self._init_P    = float(init_P)
        self._ukf_alpha = float(ukf_alpha)

        n = 4  # state dimension for CV model
        self._Wm, self._Wc, self._lam = _ukf_weights( n, alpha=self._ukf_alpha, kappa=0.0, beta=2.0
        )

    @property
    def name(self) -> str:
        return "UKF"

    @property
    def params(self) -> dict:
        return {
            "sigma_a":   self._sigma_a,
            "R_scale":   self._R_scale,
            "dt":        self._dt,
            "init_P":    self._init_P,
            "ukf_alpha": self._ukf_alpha,
        }

    def apply(self, bundle: PredBundle) -> np.ndarray:
        raw = bundle.mean_real[:, :2]
        var = bundle.var_real[:,  :2]
        ctx = bundle.ctx_mask
        gt  = bundle.gt_real[:, :2]
        T   = raw.shape[0]

        F, H, Q  = _cv_matrices(self._dt, self._sigma_a)
        Wm, Wc   = self._Wm, self._Wc
        lam      = self._lam

        init_pos = gt[0].astype(np.float64) if ctx[0] else raw[0].astype(np.float64)
        x = np.array([init_pos[0], init_pos[1], 0.0, 0.0], dtype=np.float64)
        P = np.eye(4, dtype=np.float64) * self._init_P

        out    = np.empty((T, 2), dtype=np.float64)
        out[0] = x[:2]

        for t in range(1, T):
            x, P = _ukf_predict(x, P, F, Q, Wm, Wc, lam)

            if ctx[t]:
                R = np.eye(2, dtype=np.float64) * 1e-6
                z = gt[t].astype(np.float64)
            else:
                R = _R_from_var(var[t], self._R_scale)
                R = np.maximum(R, np.eye(2) * 1e-6)
                z = raw[t].astype(np.float64)

            x, P   = _ukf_update(x, P, z, H, R, Wm, Wc, lam)
            out[t] = x[:2]

        return out.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Mahalanobis rejection — Gated Kalman Filter
# ═══════════════════════════════════════════════════════════════════════════════

class MahalanobisPostProcessor(PostProcessor):
    """Gated Kalman filter: outlier predictions rejected via Mahalanobis distance.

    At each target timestep:
      1. Predict state forward with CV dynamics.
      2. Compute Mahalanobis distance between raw ANP prediction and predicted position using the innovation covariance S = H*P*H^T + R.
      3. If distance > mahal_thresh: output predicted position (reject raw pred).
         If distance <= mahal_thresh: standard KF update with raw ANP prediction.

    This makes the filter robust to large, spurious acoustic measurement errors without completely discarding the uncertainty information from the model.

    Parameters
    ----------
    mahal_thresh : Mahalanobis distance threshold for rejection (chi-squared 95% for 2-DOF ≈ 2.45, 99% ≈ 3.03)
    sigma_a      : process noise
    R_scale      : multiplier on per-step ANP variance → R_t
    dt           : timestep
    init_P       : initial state covariance diagonal
    """

    def __init__(self, mahal_thresh: float = 3.0, sigma_a: float = 0.5, R_scale: float = 1.0, dt: float = 1.0, init_P: float = 10.0):
        self._mahal_thresh = float(mahal_thresh)
        self._sigma_a      = float(sigma_a)
        self._R_scale      = float(R_scale)
        self._dt           = float(dt)
        self._init_P       = float(init_P)

    @property
    def name(self) -> str:
        return "Mahalanobis"

    @property
    def params(self) -> dict:
        return {
            "mahal_thresh": self._mahal_thresh,
            "sigma_a":      self._sigma_a,
            "R_scale":      self._R_scale,
            "dt":           self._dt,
            "init_P":       self._init_P,
        }

    def _mahal_dist(self, z: np.ndarray, z_pred: np.ndarray, S: np.ndarray) -> float:
        diff  = (z - z_pred).reshape(-1, 1)
        try:
            d2 = float((diff.T @ np.linalg.inv(S) @ diff).item())
        except np.linalg.LinAlgError:
            return float("inf")
        return float(np.sqrt(max(d2, 0.0)))

    def apply(self, bundle: PredBundle) -> np.ndarray:
        raw = bundle.mean_real[:, :2]
        var = bundle.var_real[:,  :2]
        ctx = bundle.ctx_mask
        gt  = bundle.gt_real[:, :2]
        T   = raw.shape[0]

        F, H, Q = _cv_matrices(self._dt, self._sigma_a)

        init_pos = gt[0].astype(np.float64) if ctx[0] else raw[0].astype(np.float64)
        x = np.array([init_pos[0], init_pos[1], 0.0, 0.0], dtype=np.float64)
        P = np.eye(4, dtype=np.float64) * self._init_P

        out         = np.empty((T, 2), dtype=np.float64)
        out[0]      = x[:2]
        n_rejected  = 0

        for t in range(1, T):
            x, P = _kf_predict(x, P, F, Q)

            x_pred_pos = (H @ x)   # (2,) predicted measurement

            if ctx[t]:
                # Always accept ground-truth context positions
                R  = np.eye(2, dtype=np.float64) * 1e-6
                z  = gt[t].astype(np.float64)
                x, P   = _kf_update(x, P, z, H, R)
                out[t] = x[:2]
            else:
                R   = _R_from_var(var[t], self._R_scale)
                R   = np.maximum(R, np.eye(2) * 1e-6)
                z   = raw[t].astype(np.float64)
                S   = H @ P @ H.T + R
                d   = self._mahal_dist(z, x_pred_pos, S)

                if d > self._mahal_thresh:
                    # Outlier: output predicted position, skip KF update
                    out[t]  = x_pred_pos
                    n_rejected += 1
                else:
                    x, P    = _kf_update(x, P, z, H, R)
                    out[t]  = x[:2]

        return out.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. BiasAR — Decaying bias correction from context residuals
# ═══════════════════════════════════════════════════════════════════════════════

class BiasARPostProcessor(PostProcessor):
    """Decaying bias correction using the mean residual observed during context.

    During the context window GT is known, so the model's prediction errors (residuals) are directly observable.
    Their mean is used as a trajectory-specific bias estimate and subtracted from subsequent predictions with exponential decay:

        r_bias   = mean( raw[ctx] - gt[ctx] )
        output_t = raw_t - rho^(t - last_ctx_step) * r_bias

    rho = 1 → constant correction (bias persists throughout trajectory).
    rho = 0 → correction applied only at step 1 then vanishes immediately.

    Parameters
    ----------
    rho : decay factor in [0, 1].
    """

    def __init__(self, rho: float = 0.9):
        self._rho = float(rho)

    @property
    def name(self) -> str:
        return "BiasAR"

    @property
    def params(self) -> dict:
        return {"rho": self._rho}

    def apply(self, bundle: PredBundle) -> np.ndarray:
        raw = bundle.mean_real[:, :2]
        ctx = bundle.ctx_mask
        gt  = bundle.gt_real[:, :2]
        T   = raw.shape[0]

        ctx_idx  = np.where(ctx)[0]
        out      = np.empty((T, 2), dtype=np.float64)

        if len(ctx_idx) > 0:
            r_bias   = (raw[ctx_idx] - gt[ctx_idx]).mean(axis=0)  # (2,)
            last_ctx = int(ctx_idx[-1])
        else:
            r_bias   = np.zeros(2)
            last_ctx = -1

        for t in range(T):
            if ctx[t]:
                out[t] = gt[t]
            else:
                step   = t - last_ctx   # >= 1 for the first prediction step
                out[t] = raw[t] - (self._rho ** step) * r_bias

        return out.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. AR-p — AR(p) residual correction fitted on context window
# ═══════════════════════════════════════════════════════════════════════════════

class ARpPostProcessor(PostProcessor):
    """Autoregressive residual correction of order p.

    Fitting phase (context window, GT known):
        r_t = raw_t - gt_t
        Fit AR(p) via ridge OLS independently per output dimension:
            r_t = phi_1*r_{t-1} + ... + phi_p*r_{t-p}

    Prediction phase (autoregressive, strictly causal):
        r_hat_t  = phi_1*r_{t-1} + ... + phi_p*r_{t-p}
        output_t = raw_t - r_hat_t
        residual buffer updated with r_hat_t (GT not available)

    Falls back to mean-bias correction when n_ctx <= p.

    Parameters
    ----------
    p     : AR order.
    ridge : L2 regularisation coefficient for OLS fitting.
    """

    def __init__(self, p: int = 3, ridge: float = 1.0):
        self._p     = int(p)
        self._ridge = float(ridge)

    @property
    def name(self) -> str:
        return "AR-p"

    @property
    def params(self) -> dict:
        return {"p": self._p, "ridge": self._ridge}

    def _fit_ar_1d(self, residuals: np.ndarray) -> np.ndarray:
        """Fit AR(p) coefficients for one dimension via ridge OLS."""
        n, p = len(residuals), self._p
        if n <= p:
            return np.zeros(p)
        rows = n - p
        X = np.empty((rows, p), dtype=np.float64)
        for i in range(rows):
            X[i] = residuals[i:i + p][::-1]   # [r_{i+p-1}, ..., r_i]
        y = residuals[p:].astype(np.float64)
        A = X.T @ X + self._ridge * np.eye(p)
        b = X.T @ y
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.zeros(p)

    def apply(self, bundle: PredBundle) -> np.ndarray:
        raw = bundle.mean_real[:, :2]
        ctx = bundle.ctx_mask
        gt  = bundle.gt_real[:, :2]
        T   = raw.shape[0]
        p   = self._p

        ctx_idx = np.where(ctx)[0]
        n_ctx   = len(ctx_idx)
        out     = np.empty((T, 2), dtype=np.float64)

        # Observable residuals on context window
        r_ctx = (raw[ctx_idx] - gt[ctx_idx]).astype(np.float64)  # (n_ctx, 2)

        # Stability clamp: AR recursion can diverge if roots are outside the unit circle. We clip predicted residuals to ±max_r to prevent float overflow.
        # max_r is set to 5× the observed residual std (or a 1 m floor).
        if n_ctx > 1:
            max_r = np.maximum(5.0 * r_ctx.std(axis=0), 1.0)
        else:
            max_r = np.full(2, 1e3)

        # Fit AR(p) per dimension
        phi = np.stack(
            [self._fit_ar_1d(r_ctx[:, d]) for d in range(2)], axis=1
        )  # (p, 2)

        # Initialise residual buffer with last p context residuals
        r_buf = np.zeros((p, 2), dtype=np.float64)
        if n_ctx > 0:
            take = min(n_ctx, p)
            r_buf[p - take:] = r_ctx[-take:]

        # Context → GT
        for t in ctx_idx:
            out[t] = gt[t]

        # Prediction phase: autoregressive residual correction
        for t in range(T):
            if ctx[t]:
                continue
            # r_hat = sum_k phi[k, d] * r_buf[p-1-k, d]  (r_buf[-1] = most recent)
            r_hat  = (phi * r_buf[::-1]).sum(axis=0)  # (2,)
            # Clamp to prevent explosive divergence of unstable AR processes
            r_hat  = np.clip(r_hat, -max_r, max_r)
            out[t] = raw[t] - r_hat
            # Shift buffer and store clipped residual for next step
            r_buf       = np.roll(r_buf, -1, axis=0)
            r_buf[-1]   = r_hat

        return out.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Hyperparameter search
# ═══════════════════════════════════════════════════════════════════════════════

def _mae_xz(filtered_xy: np.ndarray, bundle: PredBundle) -> float:
    """MAE over evaluation target points, x-z-plane only."""
    tgt_mask = bundle.target_mask
    return float(np.mean(np.abs(filtered_xy[tgt_mask] - bundle.gt_real[tgt_mask, :2])))


def _mae_full(filtered_xy: np.ndarray, bundle: PredBundle) -> float:
    """MAE over evaluation target points, all 3 output dims (x-z filtered, 3rd dim raw)."""
    tgt_mask = bundle.target_mask
    y_filt   = bundle.mean_real[tgt_mask].copy()
    y_filt[:, :2] = filtered_xy[tgt_mask]
    return float(np.mean(np.abs(y_filt - bundle.gt_real[tgt_mask])))


def _mean_bundle_mae(pp: PostProcessor, bundles: List[PredBundle]) -> float:
    """Average MAE (x-z) across a list of bundles."""
    maes = [_mae_xz(pp.apply(b), b) for b in bundles]
    return float(np.mean(maes))


def random_search_hparams( pp_class,
    param_grid: Dict[str, list],
    val_bundles: List[PredBundle],
    n_trials: int = 50,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[dict, float]:
    """Random search over param_grid, evaluate on val_bundles.

    Returns (best_params, best_val_mae).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # Build all combinations (cap if too large)
    import itertools
    all_keys = list(param_grid.keys())
    all_vals = list(param_grid.values())
    all_combos = list(itertools.product(*all_vals))

    if len(all_combos) <= n_trials:
        candidates = all_combos
    else:
        idx = rng.choice(len(all_combos), size=n_trials, replace=False)
        candidates = [all_combos[i] for i in idx]

    best_params = None
    best_mae    = float("inf")

    for combo in candidates:
        kw = dict(zip(all_keys, combo))
        try:
            pp  = pp_class(**kw)
            mae = _mean_bundle_mae(pp, val_bundles)
        except Exception:
            continue
        if mae < best_mae:
            best_mae    = mae
            best_params = kw

    if best_params is None:
        # Fallback: first combo
        best_params = dict(zip(all_keys, candidates[0]))
        best_mae    = _mean_bundle_mae(pp_class(**best_params), val_bundles)

    return best_params, best_mae


# ═══════════════════════════════════════════════════════════════════════════════
# Parameter grids
# ═══════════════════════════════════════════════════════════════════════════════

def _build_param_grids(dt: float) -> Dict[str, Dict[str, list]]:
    """Return parameter grids for each postprocessor class."""
    sigma_a_vals = [0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
    R_scale_vals = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    return {
        "EMA": {
            "alpha": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95],
        },
        "EMA-Var": {
            # sigma_ref in physical std units (metres); spans from very tight to very loose
            "sigma_ref":  [0.5, 1.0, 2.0, 4.0, 7.0, 10.0, 20.0],
            "alpha_min":  [0.05, 0.1, 0.2],
            "alpha_max":  [0.7, 0.8, 0.9, 0.95],
        },
        "EKF": {
            "sigma_a": sigma_a_vals,
            "R_scale": R_scale_vals,
            "dt":      [dt],
            "init_P":  [1.0, 10.0, 100.0],
        },
        "UKF": {
            "sigma_a":   sigma_a_vals,
            "R_scale":   R_scale_vals,
            "dt":        [dt],
            "init_P":    [1.0, 10.0, 100.0],
            "ukf_alpha": [0.5, 1.0],
        },
        "Mahalanobis": {
            "mahal_thresh": [1.5, 2.0, 2.45, 3.0, 3.5, 5.0],
            "sigma_a":      sigma_a_vals,
            "R_scale":      R_scale_vals,
            "dt":           [dt],
            "init_P":       [1.0, 10.0, 100.0],
        },
        "BiasAR": {
            "rho": [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
        },
        "AR-p": {
            "p":     [1, 2, 3, 5, 8],
            "ridge": [0.01, 0.1, 1.0, 10.0, 100.0],
        },
    }


_PP_CLASSES = {
    "EMA":          EMAPostProcessor,
    "EMA-Var":      EMAVarPostProcessor,
    "EKF":          EKFPostProcessor,
    "UKF":          UKFPostProcessor,
    "Mahalanobis":  MahalanobisPostProcessor,
    "BiasAR":       BiasARPostProcessor,
    "AR-p":         ARpPostProcessor,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalResult:
    name:                str
    params:              dict
    maes_xz:             List[float]   # per trajectory, x-z plane
    maes_full:           List[float]   # per trajectory, all 3 dims
    latencies_post_s:    List[float]   # postprocessing overhead only (s)
    latencies_infer_s:   List[float]   # model inference time per trajectory (s)
    val_mae_xz:          float = float("nan")  # from hparam search


def evaluate_postprocessor(
    pp: PostProcessor,
    test_bundles: List[PredBundle],
    val_mae_xz: float = float("nan"),
) -> EvalResult:
    maes_xz          = []
    maes_full        = []
    latencies_post   = []
    latencies_infer  = []

    for bundle in test_bundles:
        fxy, elapsed_post = pp.apply_timed(bundle)
        maes_xz.append(_mae_xz(fxy, bundle))
        maes_full.append(_mae_full(fxy, bundle))
        latencies_post.append(elapsed_post)
        # infer_time_s may be 0.0 for bundles loaded from an old cache
        latencies_infer.append(getattr(bundle, "infer_time_s", 0.0))

    return EvalResult(
        name=pp.name,
        params=pp.params,
        maes_xz=maes_xz,
        maes_full=maes_full,
        latencies_post_s=latencies_post,
        latencies_infer_s=latencies_infer,
        val_mae_xz=val_mae_xz,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════════

def _stats(vals: List[float]) -> Tuple[float, float, float]:
    """(mean, std, median)."""
    a = np.array(vals)
    return float(a.mean()), float(a.std()), float(np.median(a))


def save_txt_report(
    results: List[EvalResult],
    output_path: str,
    model_cfg: ModelConfig,
    topology: str,
    eval_protocol: str,
    holdout_frac: float,
    ctx_frac: float,
    n_test: int,
    n_val:  int,
) -> None:
    lines = []
    SEP = "=" * 80
    lines.append(SEP)
    lines.append("POSTPROCESSING COMPARISON — TRAJECTORY LOCALIZATION")
    lines.append(SEP)
    lines.append(f"  Model       : {model_cfg.name}  ({model_cfg.model_type.upper()})")
    ckpt_info = model_cfg.ckpt_path if model_cfg.ckpt_path else f"Optuna dir: {model_cfg.optuna_best_model_dir}"
    lines.append(f"  Checkpoint  : {ckpt_info}")
    lines.append(f"  Topology    : {topology}")
    lines.append(f"  Protocol    : {eval_protocol}")
    lines.append(f"  Holdout     : {holdout_frac * 100:.0f}%  (final trajectory tail)")
    lines.append(f"  Context     : {ctx_frac * 100:.0f}%  (first points)")
    lines.append(f"  Val / Test  : {n_val} / {n_test} trajectories")
    lines.append(f"  Date        : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # ── MAE table ────────────────────────────────────────────────────────────
    lines.append("MAE (evaluation target points)")
    lines.append("-" * 80)
    hdr = f"{'Method':<18} {'Mean':>10} {'Std':>10} {'Median':>10} {'Val MAE':>10}"
    lines.append(hdr)
    lines.append("-" * 80)

    raw_result = next(r for r in results if r.name == "Raw")
    raw_mean   = np.mean(raw_result.maes_xz)

    for r in results:
        m, s, med = _stats(r.maes_xz)
        delta_str  = (f"  ({(m - raw_mean):+.4f})" if r.name != "Raw" else "  (baseline)")
        val_str    = f"{r.val_mae_xz:.4f}" if not np.isnan(r.val_mae_xz) else "    -"
        lines.append(
            f"{r.name:<18} {m:>10.4f} {s:>10.4f} {med:>10.4f} {val_str:>10}"
            + delta_str
        )
    lines.append("")

    # ── Latency table ────────────────────────────────────────────────────────
    lines.append("Latency per trajectory (ms)   [Infer = model forward pass | PP = postprocessing overhead]")
    lines.append("-" * 80)
    hdr2 = f"{'Method':<18} {'Infer (ms)':>12} {'PP (ms)':>12} {'Total (ms)':>12} {'PP p95 (ms)':>13}"
    lines.append(hdr2)
    lines.append("-" * 80)
    for r in results:
        infer_ms = np.array(r.latencies_infer_s) * 1e3
        post_ms  = np.array(r.latencies_post_s)  * 1e3
        total_ms = infer_ms + post_ms
        lines.append(
            f"{r.name:<18}"
            f" {float(infer_ms.mean()):>12.3f}"
            f" {float(post_ms.mean()):>12.3f}"
            f" {float(total_ms.mean()):>12.3f}"
            f" {float(np.percentile(post_ms, 95)):>13.3f}"
        )
    lines.append("")

    # ── Best hyperparameters ─────────────────────────────────────────────────
    lines.append("Best hyperparameters (from val-set random search)")
    lines.append("-" * 80)
    for r in results:
        if r.params:
            lines.append(f"  {r.name:<16}: {json.dumps(r.params)}")
    lines.append(SEP)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[report] saved → {output_path}")


def save_mae_boxplot(results: List[EvalResult], output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(max(8, len(results) * 1.5), 5))
    data    = [r.maes_xz for r in results]
    labels  = [r.name for r in results]
    bp      = ax.boxplot(data, patch_artist=True, notch=False, vert=True,
                         medianprops={"color": "black", "linewidth": 2})
    colors  = matplotlib.colormaps["tab10"](np.linspace(0, 0.9, len(results)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticks(range(1, len(results) + 1))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("MAE (m)", fontsize=11)
    ax.set_title("Postprocessing comparison — MAE distribution (test set)", fontsize=12)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[plot ] boxplot → {output_path}")


def save_pareto_plot(results: List[EvalResult], output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = matplotlib.colormaps["tab10"](np.linspace(0, 0.9, len(results)))

    for r, c in zip(results, colors):
        mae_mean   = float(np.mean(r.maes_xz))
        infer_mean = float(np.mean(r.latencies_infer_s)) * 1e3   # ms
        post_mean  = float(np.mean(r.latencies_post_s))  * 1e3   # ms
        total_mean = infer_mean + post_mean
        ax.scatter(total_mean, mae_mean, color=c, s=120, zorder=5, label=r.name)
        label_txt = f"{r.name} ({total_mean:.1f} ms)"
        ax.annotate(label_txt, (total_mean, mae_mean),
                    textcoords="offset points", xytext=(6, 3), fontsize=9)

    ax.set_xlabel("Mean total latency (infer + PP overhead, ms per trajectory)", fontsize=11)
    ax.set_ylabel("Mean MAE (m)", fontsize=11)
    ax.set_title("MAE vs total latency (Pareto view)", fontsize=12)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[plot ] pareto  → {output_path}")


def save_qualitative_plots(
    results: List[EvalResult],
    test_bundles: List[PredBundle],
    pp_list: List[PostProcessor],
    output_dir: str,
    n_samples: int = 4,
    rng: Optional[np.random.Generator] = None,
) -> None:
    if rng is None:
        rng = np.random.default_rng(0)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    n_samples = min(n_samples, len(test_bundles))
    idxs      = rng.choice(len(test_bundles), size=n_samples, replace=False)

    colors = matplotlib.colormaps["tab10"](np.linspace(0, 0.9, len(pp_list)))

    for i, idx in enumerate(idxs):
        bundle = test_bundles[idx]
        fig, ax = plt.subplots(figsize=(9, 4))

        # Ground truth
        gt = bundle.gt_real[:, :2]
        ax.plot(gt[:, 0], gt[:, 1], "k-", lw=2, label="GT", zorder=10)

        for pp, color in zip(pp_list, colors):
            fxy = pp.apply(bundle)
            ls  = "-" if pp.name == "Raw" else "--"
            ax.plot(fxy[:, 0], fxy[:, 1], ls, color=color,
                    lw=1.5, alpha=0.8, label=pp.name)

        # Mark context / target
        ctx_idx = np.where(bundle.ctx_mask)[0]
        ax.scatter(gt[ctx_idx, 0], gt[ctx_idx, 1],
                   marker="o", c="red", s=25, zorder=9, label="Context (GT)")

        ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
        ax.set_title(f"Trajectory #{idx}  θ={bundle.theta:.2f}", fontsize=11)
        ax.legend(fontsize=8, loc="upper right", ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = os.path.join(output_dir, f"traj_{i:03d}_idx{idx}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    print(f"[plot ] qualitative plots → {output_dir}")


def _pareto_non_dominated_mask(lat_ms: np.ndarray, mae: np.ndarray) -> np.ndarray:
    """Return boolean mask of non-dominated points for minimization on both axes."""
    n = len(lat_ms)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominated = (
            (lat_ms <= lat_ms[i])
            & (mae <= mae[i])
            & ((lat_ms < lat_ms[i]) | (mae < mae[i]))
        )
        dominated[i] = False
        if np.any(dominated):
            keep[i] = False
    return keep


def save_aggregate_pareto(points: List[dict], output_path: str, title: str) -> None:
    """Save aggregate Pareto scatter for one (version, topology, data, protocol) group."""
    if not points:
        return

    lat_ms = np.array([p["latency_total_ms"] for p in points], dtype=np.float64)
    mae = np.array([p["mae_mean"] for p in points], dtype=np.float64)
    frontier = _pareto_non_dominated_mask(lat_ms, mae)

    methods = sorted({p["method"] for p in points})
    method_colors = {
        m: c for m, c in zip(
            methods,
            matplotlib.colormaps["tab10"](np.linspace(0, 0.9, max(len(methods), 2)))
        )
    }
    marker_by_model = {"anp": "o", "ranp": "s"}

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, p in enumerate(points):
        marker = marker_by_model.get(p["model_type"], "^")
        color = method_colors.get(p["method"], "gray")
        ax.scatter(
            p["latency_total_ms"],
            p["mae_mean"],
            s=130 if frontier[i] else 90,
            marker=marker,
            color=color,
            edgecolors="black" if frontier[i] else "none",
            linewidths=0.7 if frontier[i] else 0.0,
            alpha=0.9,
            zorder=6 if frontier[i] else 4,
        )

    frontier_points = [points[i] for i in np.where(frontier)[0]]
    frontier_points = sorted(frontier_points, key=lambda x: (x["latency_total_ms"], x["mae_mean"]))
    if frontier_points:
        ax.plot(
            [p["latency_total_ms"] for p in frontier_points],
            [p["mae_mean"] for p in frontier_points],
            color="black",
            linestyle="-",
            linewidth=1.3,
            alpha=0.8,
            label="Pareto frontier",
            zorder=5,
        )
        for p in frontier_points:
            ax.annotate(
                f"{p['model_type']}-{p['version']}-{p['method']}",
                (p["latency_total_ms"], p["mae_mean"]),
                textcoords="offset points",
                xytext=(5, 3),
                fontsize=8,
            )

    method_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=m,
                   markerfacecolor=method_colors[m], markersize=8)
        for m in methods
    ]
    model_handles = [
        plt.Line2D([0], [0], marker=marker_by_model[k], color="black", label=k.upper(),
                   linestyle="None", markersize=7)
        for k in sorted(marker_by_model.keys())
    ]
    ax.legend(handles=method_handles + model_handles, fontsize=8, loc="upper right", ncol=2)

    ax.set_xlabel("Mean total latency (infer + PP, ms per trajectory)")
    ax.set_ylabel("Mean MAE (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[plot ] aggregate pareto → {output_path}")


def save_aggregate_points_csv(points: List[dict], output_path: str) -> None:
    """Save aggregate Pareto source points with frontier flag."""
    if not points:
        return
    lat_ms = np.array([p["latency_total_ms"] for p in points], dtype=np.float64)
    mae = np.array([p["mae_mean"] for p in points], dtype=np.float64)
    frontier = _pareto_non_dominated_mask(lat_ms, mae)

    rows = []
    for i, p in enumerate(points):
        rows.append([
            p["version"],
            p["topology"],
            p["data_variant"],
            p["protocol"],
            p["model_name"],
            p["model_type"],
            p["method"],
            p["mae_mean"],
            p["latency_total_ms"],
            int(frontier[i]),
        ])

    save_csv(
        Path(output_path),
        rows,
        [
            "version", "topology", "data_variant", "protocol",
            "model_name", "model_type", "method",
            "mae_mean", "latency_total_ms", "is_pareto_frontier",
        ],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Online postprocessing comparison for ANP/RANP trajectory predictions.")
    # ── model ──
    p.add_argument("--ckpt", default=None, help="Path to best_checkpoint.pth.tar")
    p.add_argument("--optuna-best-model-dir", default=None, help="Path to Optuna study directory or best_model subdirectory. Can point to .../study_name, .../study_name/best_model, or nested structures like .../model_type/version/study_name.")
    p.add_argument("--optuna-root-dir", default=None, help="Root directory containing many Optuna studies. The script auto-discovers all best_model folders and evaluates all discovered models.")
    p.add_argument("--versions", default="all", help="Version filter in auto mode, e.g. 'v1', 'v2' or 'v1,v2' (default: all).")
    p.add_argument("--model-type", default="anp", choices=["anp", "ranp"], help="Model architecture")
    p.add_argument("--model-name", default=None, help="Human-readable model name (default: inferred from --ckpt)")
    p.add_argument("--rnn-type", default="lstm", choices=["lstm", "gru"], help="(RANP only) RNN cell type")
    p.add_argument("--rnn-layers", type=int, default=1, help="(RANP only) number of RNN layers")
    # ── data ──
    p.add_argument("--data-dir", default=None, help="Default data directory (used for all models unless variant-specific dirs are provided).")
    p.add_argument("--data-dir-lowvar", default=None, help="Data directory for low-variance models (auto mode).")
    p.add_argument("--data-dir-highvar", default=None, help="Data directory for high-variance models (auto mode).")
    p.add_argument("--topology", default="ellipsoidal", choices=["ellipsoidal", "random", "aligned"], help="Topology used only when --single-topology is set")
    p.add_argument("--single-topology", action="store_true", help="Evaluate only --topology (default evaluates aligned, ellipsoidal, random)")
    p.add_argument("--ctx-frac", type=float, default=0.3, help="Fraction of trajectory used as context (first points)")
    p.add_argument("--holdout-frac", type=float, default=0.2, help="Fraction reserved as target holdout tail (default 0.2 => 10/50 points)")
    p.add_argument("--eval-protocol", default="both_holdouts", choices=["holdout", "inverse_holdout", "both_holdouts"], help="Evaluation protocol. 'both_holdouts' runs and stores both separately.")
    p.add_argument("--dt", type=float, default=1.0, help="Timestep in seconds (used by KF / EKF / UKF)")
    # ── output ──
    p.add_argument("--output-dir", default="results/postprocessing", help="Directory where reports and plots are saved")
    p.add_argument("--n-qual-plots", type=int, default=4, help="Number of qualitative trajectory plots to generate")
    # ── search ──
    p.add_argument("--n-hparam-trials", type=int, default=50, help="Random search trials per postprocessor method")
    # ── misc ──
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on")
    p.add_argument("--seed", type=int, default=18, help="Random seed")
    p.add_argument("--extra-configs", default=None, help="JSON string: list of extra ModelConfig dicts to also evaluate")
    p.add_argument("--no-cache", action="store_true", help="Force re-running inference even if cached bundles exist")
    return p


def main() -> None:
    args   = build_arg_parser().parse_args()
    device = torch.device(args.device)
    rng    = np.random.default_rng(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.ckpt and not args.optuna_best_model_dir and not args.optuna_root_dir:
        raise ValueError("Provide one of: --ckpt, --optuna-best-model-dir, --optuna-root-dir")

    # ── resolve Optuna paths and infer model name ────────────────────────────
    if args.optuna_best_model_dir:
        optuna_best_model_dir, study_name = _resolve_optuna_study_dir(args.optuna_best_model_dir)
        args.optuna_best_model_dir = optuna_best_model_dir
        if args.model_name is None:
            args.model_name = study_name

    if args.model_name is None:
        if args.ckpt:
            args.model_name = Path(args.ckpt).parents[2].name  # study dir name
        elif args.optuna_best_model_dir:
            args.model_name = "anp_model"  # fallback
        else:
            args.model_name = "auto_discovered_models"

    # ── infer num_hidden from hparams.json if present ───────────────────────
    num_hidden = 128
    if args.ckpt:
        hparams_candidate = Path(args.ckpt).parent.parent / "hparams.json"
        num_hidden = _infer_num_hidden(str(hparams_candidate))
    elif args.optuna_best_model_dir:
        # For Optuna, search for hparams.json in the study directory (parent of best_model)
        study_dir = Path(args.optuna_best_model_dir).parent
        hparams_candidate = study_dir / "hparams.json"
        num_hidden = _infer_num_hidden(str(hparams_candidate))

    # ── primary model config ─────────────────────────────────────────────────
    model_cfg = ModelConfig(
        name=args.model_name,
        ckpt_path=args.ckpt or "",
        optuna_best_model_dir=args.optuna_best_model_dir or "",
        model_type=args.model_type,
        num_hidden=num_hidden,
        rnn_type=args.rnn_type,
        rnn_layers=args.rnn_layers,
    )

    # ── parse extra model configs (for future extensibility) ────────────────
    extra_cfgs: List[ModelConfig] = []
    if args.extra_configs:
        for d in json.loads(args.extra_configs):
            extra_cfgs.append(ModelConfig(**d))

    if args.optuna_root_dir:
        all_cfgs = _discover_optuna_model_configs(args.optuna_root_dir)
        if args.versions.strip().lower() != "all":
            versions_filter = {v.strip().lower() for v in args.versions.split(",") if v.strip()}
            before = len(all_cfgs)
            all_cfgs = [cfg for cfg in all_cfgs if cfg.version and cfg.version.lower() in versions_filter]
            print(f"[auto ] version filter={sorted(versions_filter)} -> {len(all_cfgs)}/{before} model(s)")
            if not all_cfgs:
                raise ValueError(
                    "No models matched --versions filter. "
                    "Check available versions and study naming under --optuna-root-dir."
                )
        # merge optional extra configs on top of discovered ones
        all_cfgs.extend(extra_cfgs)
    else:
        all_cfgs = [model_cfg] + extra_cfgs

    default_topologies = ["aligned", "ellipsoidal", "random"]
    eval_protocols = (
        ["holdout", "inverse_holdout"]
        if args.eval_protocol == "both_holdouts"
        else [args.eval_protocol]
    )
    aggregate_points: Dict[Tuple[str, str, str, str], List[dict]] = {}

    # ── evaluate each model config ───────────────────────────────────────────
    for cfg in all_cfgs:
        model_data_dir = _resolve_data_dir_for_model(args, cfg)
        output_data_variant = cfg.data_variant or _infer_data_variant_from_data_dir(model_data_dir)
        output_version = cfg.version if cfg.version else "vunknown"

        if args.single_topology:
            model_topologies = [args.topology]
        elif cfg.preferred_topology:
            model_topologies = [cfg.preferred_topology]
        else:
            model_topologies = default_topologies

        model_tag = (
            f"type={cfg.model_type}"
            + (f" version={cfg.version}" if cfg.version else "")
            + (f" data={cfg.data_variant}" if cfg.data_variant else "")
            + (f" topo={cfg.preferred_topology}" if cfg.preferred_topology else "")
        )
        print(f"\n[model] {cfg.name} ({model_tag})")

        for topology in model_topologies:
            # ── load data per topology ───────────────────────────────────────
            print(f"\n[data ] loading topology={topology} from {model_data_dir}")
            train_data, _   = load_split(model_data_dir, topology, "train")
            val_data, meta  = load_split(model_data_dir, topology, "val")
            test_data, meta = load_split(model_data_dir, topology, "test")
            y_mean, y_std   = compute_y_stats(train_data, device)
            val_thetas      = meta.get("val_thetas",  [0.0] * len(val_data))
            test_thetas     = meta.get("test_thetas", [0.0] * len(test_data))
            print(f"[data ] val={len(val_data)}  test={len(test_data)}")

            for eval_protocol in eval_protocols:
                print(f"\n{'─'*86}")
                print(f"  Model={cfg.name} | topology={topology} | protocol={eval_protocol}")
                print(f"{'─'*86}")

                model_out_dir = os.path.join(
                    args.output_dir,
                    f"version_{output_version}",
                    f"topology_{topology}",
                    output_data_variant,
                    cfg.name,
                    f"protocol_{eval_protocol}",
                )
                Path(model_out_dir).mkdir(parents=True, exist_ok=True)

                # keep cache keys protocol/topology-specific to avoid accidental mixing
                cache_tag = (
                    f"ctx{int(args.ctx_frac*100):03d}_"
                    f"hold{int(args.holdout_frac*100):03d}_"
                    f"{eval_protocol}"
                )
                val_cache  = Path(model_out_dir) / f"_cache_val_{cache_tag}.pkl"
                test_cache = Path(model_out_dir) / f"_cache_test_{cache_tag}.pkl"

                if val_cache.exists() and test_cache.exists() and not args.no_cache:
                    print(f"[infer] loading cached bundles from {model_out_dir} ...")
                    with open(val_cache, "rb") as f:
                        val_bundles = pickle.load(f)
                    with open(test_cache, "rb") as f:
                        test_bundles = pickle.load(f)
                    print(f"[infer] val={len(val_bundles)}  test={len(test_bundles)}  (from cache)")
                else:
                    model = load_model(cfg, device, topology=topology)
                    print("[infer] running inference on val set ...")
                    val_bundles = run_inference(
                        model, cfg.model_type, val_data, val_thetas,
                        y_mean, y_std, args.ctx_frac, eval_protocol, args.holdout_frac, device,
                    )
                    print("[infer] running inference on test set ...")
                    test_bundles = run_inference(
                        model, cfg.model_type, test_data, test_thetas,
                        y_mean, y_std, args.ctx_frac, eval_protocol, args.holdout_frac, device,
                    )
                    with open(val_cache, "wb") as f:
                        pickle.dump(val_bundles, f)
                    with open(test_cache, "wb") as f:
                        pickle.dump(test_bundles, f)
                    print(f"[infer] bundles cached → {model_out_dir}")

                # ── hparam search + build best postprocessors ───────────────
                param_grids = _build_param_grids(args.dt)
                best_pps:      List[PostProcessor] = [RawPostProcessor()]
                best_val_maes: List[float]         = [float("nan")]

                for pp_key, pp_class in _PP_CLASSES.items():
                    grid = param_grids[pp_key]
                    print(f"[hpopt] {pp_key}  ({args.n_hparam_trials} trials) ...")
                    best_params, best_val_mae = random_search_hparams(
                        pp_class, grid, val_bundles,
                        n_trials=args.n_hparam_trials,
                        rng=rng,
                    )
                    best_pps.append(pp_class(**best_params))
                    best_val_maes.append(best_val_mae)
                    print(f"       → best params: {best_params}  val MAE={best_val_mae:.4f}")

                # ── final evaluation on test set ────────────────────────────
                print("[eval ] running final evaluation on test set ...")
                eval_results: List[EvalResult] = []
                for pp, val_mae in zip(best_pps, best_val_maes):
                    res       = evaluate_postprocessor(pp, test_bundles, val_mae_xz=val_mae)
                    m         = np.mean(res.maes_xz)
                    lat_infer = np.mean(res.latencies_infer_s) * 1e3
                    lat_post  = np.mean(res.latencies_post_s)  * 1e3
                    lat_total = lat_infer + lat_post
                    print(f"        {res.name:<16} MAE={m:.4f}  infer={lat_infer:.2f} ms  pp={lat_post:.3f} ms  total={lat_total:.2f} ms")
                    eval_results.append(res)

                    agg_key = (output_version, topology, output_data_variant, eval_protocol)
                    aggregate_points.setdefault(agg_key, []).append(
                        {
                            "version": output_version,
                            "topology": topology,
                            "data_variant": output_data_variant,
                            "protocol": eval_protocol,
                            "model_name": cfg.name,
                            "model_type": cfg.model_type,
                            "method": res.name,
                            "mae_mean": float(m),
                            "latency_total_ms": float(lat_total),
                        }
                    )

                # ── reports (suffix includes topology + protocol) ───────────
                suffix      = f"{topology}_{eval_protocol}"
                txt_path    = os.path.join(model_out_dir, f"comparison_report_{suffix}.txt")
                box_path    = os.path.join(model_out_dir, f"mae_boxplot_{suffix}.png")
                pareto_path = os.path.join(model_out_dir, f"pareto_mae_latency_{suffix}.png")
                qual_dir    = os.path.join(model_out_dir, f"traj_plots_{suffix}")

                save_txt_report(
                    eval_results, txt_path, cfg, topology, eval_protocol, args.holdout_frac,
                    args.ctx_frac, len(test_data), len(val_data),
                )
                save_mae_boxplot(eval_results, box_path)
                save_pareto_plot(eval_results, pareto_path)
                save_qualitative_plots(
                    eval_results, test_bundles, best_pps, qual_dir,
                    n_samples=args.n_qual_plots, rng=rng,
                )

                print(f"\n[done ] outputs saved to {model_out_dir}")

    # ── aggregate Pareto plots per version / topology / data / protocol ─────
    for (version, topology, data_variant, protocol), points in sorted(aggregate_points.items()):
        out_dir = Path(args.output_dir) / f"version_{version}" / f"topology_{topology}" / data_variant / "aggregate" / f"protocol_{protocol}"
        title = (
            f"Aggregate Pareto | version={version} | topology={topology} | "
            f"data={data_variant} | protocol={protocol}"
        )
        save_aggregate_pareto(points, str(out_dir / "pareto_all_models.png"), title)
        save_aggregate_points_csv(points, str(out_dir / "pareto_points.csv"))

    print("\n[done ] all models evaluated.")


if __name__ == "__main__":
    main()
