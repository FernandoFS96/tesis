#!/usr/bin/env python3
"""
Evalúa postprocesado de trayectorias ANP con filtros, y (opcionalmente) con AR-rollout + filtros.

Incluye:
- Evaluación estándar con un único postprocesado (--filter none|alpha_beta|kalman_cv|kalman_rts)
- Modo comparación (--filter compare) que:
  - calcula: raw, alpha_beta(raw), kalman_cv(raw), kalman_rts(raw)
  - y opcionalmente también: ar_raw, alpha_beta(ar_raw), kalman_cv(ar_raw), kalman_rts(ar_raw)
  - guarda un resumen en un TXT (--out-txt),
  - opcionalmente genera plots cualitativos por theta (4 trayectorias al azar) *SOLO* para el stream raw,
  - opcionalmente genera un boxplot con todas las distribuciones de MAE (incluyendo las 8 si AR está activo).

Notas:
- El AR-rollout implementa despliegue autoregresivo en test-time (sin reentrenar), inspirado en AR deployment para CNP/NP.
  Véase: Bruinsma et al., "Autoregressive Conditional Neural Processes" (2023). (solo referencia conceptual)

Ejemplo:
python eval_anp_postprocess_filters_ar.py \
  --data-dir /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_low_variance \
  --anp-dir  /home/fernando/tesis/underwater-localization-topologies/src/training/results/ANP_topologies/low_variance/ctx_first \
  --topology random \
  --context 30 \
  --dt 1.0 \
  --filter compare \
  --out-txt comparativa_postprocesado_ar8.txt \
  --make-boxplot --boxplot-path mae_boxplot_ar8.png --boxplot-showfliers \
  --compare-include-ar \
  --ar-order closest --ar-block-k 5 --use-mu-as-z --ar-var-thresh 0.01 --ar-force-accept --pareto-hull
"""

from __future__ import annotations

import argparse
from os import times
import pickle
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from matplotlib import lines
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

# Ajusta imports si repo usa otra ruta
from src.models.anp import LatentModel
from src.utils.nav_dataset import NavigationTrajectoryDataset


# ---------------------------
# Utilidades I/O y seeds
# ---------------------------

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def _sync_if_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()

def timed_block(device: torch.device):
    """
    Uso:
      t0 = timed_block(device)   # start
      ... code ...
      dt = timed_block(device) - t0   # elapsed seconds
    """
    _sync_if_cuda(device)
    return time.perf_counter()


def load_topology_test_grouped(data_dir: Path, topology: str) -> Tuple[Dict[float, List], List[float]]:
    """Agrupa test_data por theta usando metadata['test_thetas']."""
    topology_dir = Path(data_dir) / f"topology_{topology}"
    test_path = topology_dir / "test_data.pkl"
    metadata_path = topology_dir / "metadata.pkl"
    if not test_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Missing test_data/metadata in {topology_dir}")

    with open(test_path, "rb") as f:
        test_data = pickle.load(f)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    theta_groups: Dict[float, List] = {}
    for sample, theta in zip(test_data, metadata["test_thetas"]):
        theta_groups.setdefault(float(theta), []).append(sample)

    theta_values = sorted(theta_groups.keys())
    return theta_groups, theta_values


def get_y_stats_from_train(data_dir: Path, topology: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calcula stats en train_data.pkl para normalización."""
    topology_dir = Path(data_dir) / f"topology_{topology}"
    train_path = topology_dir / "train_data.pkl"
    if not train_path.exists():
        raise FileNotFoundError(f"train_data.pkl no encontrado: {train_path}")

    with open(train_path, "rb") as f:
        train_data = pickle.load(f)

    # train_data: list[(X:(T,D), Y:(T,3))]
    Y = np.concatenate([y for _, y in train_data], axis=0)
    y_mean = torch.tensor(Y.mean(axis=0), dtype=torch.float32, device=device)
    y_std = torch.tensor(Y.std(axis=0) + 1e-6, dtype=torch.float32, device=device)
    return y_mean, y_std


def normalize_y(y: torch.Tensor, y_mean: torch.Tensor, y_std: torch.Tensor) -> torch.Tensor:
    return (y - y_mean.view(1, 1, -1)) / y_std.view(1, 1, -1)


def denormalize_y(y_norm: torch.Tensor, y_mean: torch.Tensor, y_std: torch.Tensor) -> torch.Tensor:
    return y_norm * y_std.view(1, 1, -1) + y_mean.view(1, 1, -1)


def sample_context_indices(total_points: int, n_context: int, g: torch.Generator, device: torch.device) -> torch.Tensor:
    """Subconjunto aleatorio determinista y ordenado."""
    perm = torch.randperm(total_points, generator=g, device=device)
    return perm[:n_context].sort().values


def sample_context_indices_ordered(total_points: int, n_context: int) -> torch.Tensor:
    """Subconjunto ordenado equiespaciado (determinista)."""
    indices = np.linspace(0, total_points - 1, n_context, dtype=int)
    return torch.tensor(indices, dtype=torch.long)


def load_anp_model(
    anp_dir: Path,
    topology: str,
    input_dim: int,
    output_dim: int,
    device: torch.device,
) -> torch.nn.Module:
    """Carga best_checkpoint.pth.tar."""
    ckpt_path = Path(anp_dir) / f"ANP_{topology}" / "best_checkpoint.pth.tar"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ANP checkpoint not found: {ckpt_path}")

    sensor_emb_dim = 64
    n_sensors = 10
    sensor_feature_dim = 401

    model = LatentModel(num_hidden=128, input_dim=input_dim, output_dim=output_dim)

    checkpoint = torch.load(ckpt_path, map_location=device)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state)
    return model.to(device).eval()


# -------------------------
# Autoregressive rollout (test-time)
# -------------------------

def get_rollout_order(T: int, ctx_idx_np: np.ndarray, mode: str, rng: Optional[np.random.Generator] = None) -> List[int]:
    """Lista de timesteps a predecir (excluye ctx), según el modo."""
    ctx_set = set(int(i) for i in ctx_idx_np.tolist())
    candidates = [t for t in range(T) if t not in ctx_set]

    mode = mode.lower()
    if mode == "time":
        return candidates

    if mode == "closest":
        ctx_sorted = np.sort(ctx_idx_np.astype(int))

        def dist_to_ctx(t: int) -> int:
            return int(np.min(np.abs(ctx_sorted - t)))

        candidates.sort(key=lambda t: (dist_to_ctx(t), t))
        return candidates

    if mode == "random":
        if rng is None:
            rng = np.random.default_rng(0)
        rng.shuffle(candidates)
        return candidates

    raise ValueError(f"Unknown ar-order mode: {mode}")


@torch.no_grad()
def anp_ar_rollout_one(
    model: torch.nn.Module,
    x: torch.Tensor,     # (1,T,D_in_flat)
    y: torch.Tensor,     # (1,T,Dy)  (real units)
    ctx_idx: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    use_mu_as_z: bool = True,
    order_mode: str = "closest",
    block_k: int = 5,
    order_seed: int = 0,
    var_thresh: float = 0.01,
    force_accept: bool = True,
    max_added: int = -1,
    gate_debug: bool = False,
) -> torch.Tensor:
    """
    Devuelve y_pred_ar en unidades reales: (1,T,Dy)

    - Predice targets en bloques de K (block_k) en el orden indicado.
    - Añade al contexto SOLO predicciones "aceptadas" por gating (varianza baja).
    """
    device = x.device
    B, T, _ = x.shape
    assert B == 1, "Helper pensado para un solo sample."
    added_so_far = 0

    # Normaliza GT para context_y
    y_norm = normalize_y(y, y_mean, y_std)  # (1,T,Dy)

    # Wrapper para distributed: fuse(x) y base_anp
    if hasattr(model, "base_anp") and hasattr(model, "_fuse"):
        base = model.base_anp
        def fuse(x_flat):  # (B,N,Dflat) -> (B,N,emb_dim) # type: ignore
            return model._fuse(x_flat)  # type: ignore
    else:
        base = model
        def fuse(x_flat):
            return x_flat

    ctx_idx_np = ctx_idx.detach().cpu().numpy().astype(int)

    # Contexto inicial (usa GT en contexto)
    cx_raw = x[:, ctx_idx, :]       # (1,Nctx,Dflat)
    cy     = y_norm[:, ctx_idx, :]  # (1,Nctx,Dy)
    cx = fuse(cx_raw)               # (1,Nctx,Din_for_base)

    # z fijo desde latent encoder
    mu, log_sigma, z_sample = base.latent_encoder(cx, cy)  # type: ignore
    z_fixed = mu if use_mu_as_z else z_sample              # (1,H)

    # predicción en espacio normalizado
    y_pred_norm = torch.zeros_like(y_norm)
    y_pred_norm[:, ctx_idx, :] = cy

    # rolling context
    roll_cx = cx.clone()
    roll_cy = cy.clone()

    rng = None
    if order_mode.lower() == "random":
        rng = np.random.default_rng(order_seed)
    order = get_rollout_order(T, ctx_idx_np, order_mode, rng=rng)

    K = max(1, int(block_k))
    for start in range(0, len(order), K):
        idxs = order[start:start + K]
        idxs_t = torch.tensor(idxs, dtype=torch.long, device=device)

        tx_raw = x[:, idxs_t, :]    # (1,K,Dflat)
        tx = fuse(tx_raw)           # (1,K,Din_for_base)

        r = base.deterministic_encoder(roll_cx, roll_cy, tx)             # type: ignore
        z = z_fixed.unsqueeze(1).expand(1, tx.shape[1], -1)              # (1,K,H)
        mean_k, var_k = base.decoder(r, z, tx)                           # type: ignore

        # escribe predicciones (siempre)
        y_pred_norm[:, idxs_t, :] = mean_k

        # gating por varianza XY (en espacio normalizado)
        v_xy = var_k[:, :, :2].mean(dim=2).squeeze(0)  # (K,)

        if gate_debug:
            v_np = v_xy.detach().cpu().numpy()
            p50 = float(np.percentile(v_np, 50))
            p80 = float(np.percentile(v_np, 80))
            p95 = float(np.percentile(v_np, 95))
            print(f"[gate] block v_xy: p50={p50:.4e}, p80={p80:.4e}, p95={p95:.4e}, min={v_np.min():.4e}, max={v_np.max():.4e}")

        if var_thresh is not None and var_thresh > 0:
            accept = (v_xy <= var_thresh)
        else:
            accept = torch.ones((tx.shape[1],), device=device, dtype=torch.bool)

        if force_accept and (not accept.any()):
            j = torch.argmin(v_xy)
            accept[j] = True

        if max_added is not None and max_added > 0:
            remaining = max_added - added_so_far
            if remaining <= 0:
                accept[:] = False
            else:
                if int(accept.sum().item()) > remaining:
                    idx_acc = torch.where(accept)[0]
                    vals = v_xy[idx_acc]
                    keep = idx_acc[torch.argsort(vals)[:remaining]]
                    accept[:] = False
                    accept[keep] = True

        # añade SOLO aceptados al contexto
        if accept.any():
            tx_acc = tx[:, accept, :]
            mean_acc = mean_k[:, accept, :]
            roll_cx = torch.cat([roll_cx, tx_acc], dim=1)
            roll_cy = torch.cat([roll_cy, mean_acc], dim=1)
            added_so_far += int(accept.sum().item())

    y_pred = denormalize_y(y_pred_norm, y_mean, y_std)
    return y_pred


# ---------------------------
# Postprocesado (alpha-beta / Kalman / RTS)
# ---------------------------

def alpha_beta_filter_2d(z_xy: np.ndarray, dt: float, alpha: float, beta: float, R_xy: Optional[np.ndarray] = None) -> np.ndarray:
    """Filtro alpha-beta 2D (g-h) sencillo. R_xy no se usa aquí; se mantiene por firma."""
    T = z_xy.shape[0]
    if T == 0:
        return z_xy.astype(np.float32)

    x = z_xy[0, 0]
    y = z_xy[0, 1]
    vx = 0.0
    vy = 0.0

    out = np.zeros((T, 2), dtype=np.float32)
    out[0] = [x, y]

    for k in range(1, T):
        # predict
        x_pred = x + vx * dt
        y_pred = y + vy * dt

        # residual
        rx = z_xy[k, 0] - x_pred
        ry = z_xy[k, 1] - y_pred

        # update
        x = x_pred + alpha * rx
        y = y_pred + alpha * ry
        vx = vx + (beta / dt) * rx
        vy = vy + (beta / dt) * ry

        out[k] = [x, y]

    return out


def _cv_matrices_2d(dt: float, sigma_a: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Modelo CV 2D con estado [px, py, vx, vy].
    Devuelve (F, H, Q).
    """
    F = np.array(
        [[1.0, 0.0, dt,  0.0],
         [0.0, 1.0, 0.0, dt ],
         [0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64
    )
    H = np.array(
        [[1.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 0.0]],
        dtype=np.float64
    )

    q = sigma_a ** 2
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2
    Q_1d = np.array([[dt4 / 4.0, dt3 / 2.0],
                     [dt3 / 2.0, dt2]], dtype=np.float64) * q

    Q = np.zeros((4, 4), dtype=np.float64)
    Q[np.ix_([0, 2], [0, 2])] = Q_1d
    Q[np.ix_([1, 3], [1, 3])] = Q_1d
    return F, H, Q


def kalman_filter_cv_2d(z_xy: np.ndarray, R_xy: Optional[np.ndarray], dt: float, sigma_a: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Filtro Kalman CV 2D.
    z_xy: (T,2) en unidades reales.
    R_xy: (T,2,2) o None (si None, usa identidad).
    """
    T = z_xy.shape[0]
    if T == 0:
        x = np.zeros((0, 4), dtype=np.float64)
        P = np.zeros((0, 4, 4), dtype=np.float64)
        return x, P, x, P

    F, H, Q = _cv_matrices_2d(dt, sigma_a)

    x = np.zeros(4, dtype=np.float64)
    x[0:2] = z_xy[0].astype(np.float64)
    P = np.eye(4, dtype=np.float64) * 1.0

    x_filt = np.zeros((T, 4), dtype=np.float64)
    P_filt = np.zeros((T, 4, 4), dtype=np.float64)
    x_pred = np.zeros((T, 4), dtype=np.float64)
    P_pred = np.zeros((T, 4, 4), dtype=np.float64)

    I = np.eye(4, dtype=np.float64)

    for t in range(T):
        # predict
        xp = F @ x
        Pp = F @ P @ F.T + Q

        x_pred[t] = xp
        P_pred[t] = Pp

        # update
        if R_xy is None:
            R = np.eye(2, dtype=np.float64)
        else:
            R = R_xy[t].astype(np.float64)

        z = z_xy[t].astype(np.float64)
        innov = z - (H @ xp)
        S = H @ Pp @ H.T + R
        K = (Pp @ H.T) @ np.linalg.inv(S)

        x = xp + K @ innov

        # Joseph form
        KH = K @ H
        P = (I - KH) @ Pp @ (I - KH).T + K @ R @ K.T

        x_filt[t] = x
        P_filt[t] = P

    return x_filt, P_filt, x_pred, P_pred


def rts_smoother(
    x_filt: np.ndarray,
    P_filt: np.ndarray,
    x_pred: np.ndarray,
    P_pred: np.ndarray,
    dt: float,
    sigma_a: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rauch-Tung-Striebel (RTS) fixed-interval smoother."""
    T = x_filt.shape[0]
    if T == 0:
        return x_filt, P_filt

    F, _, _ = _cv_matrices_2d(dt, sigma_a)

    x_smooth = x_filt.copy()
    P_smooth = P_filt.copy()

    for k in range(T - 2, -1, -1):
        Ppk1 = P_pred[k + 1]
        Ck = P_filt[k] @ F.T @ np.linalg.inv(Ppk1)
        x_smooth[k] = x_filt[k] + Ck @ (x_smooth[k + 1] - x_pred[k + 1])
        P_smooth[k] = P_filt[k] + Ck @ (P_smooth[k + 1] - Ppk1) @ Ck.T

    return x_smooth, P_smooth


def postprocess_filter(
    z_xy: np.ndarray,
    R_xy: Optional[np.ndarray],
    method: str,
    dt: float,
    alpha: float = 0.85,
    beta: float = 0.005,
    sigma_a: float = 1.0,
) -> np.ndarray:
    """Aplica el método de postprocesado seleccionado."""
    method = method.lower()
    if method == "none":
        return z_xy.astype(np.float32)

    if method == "alpha_beta":
        return alpha_beta_filter_2d(z_xy=z_xy, dt=dt, alpha=alpha, beta=beta, R_xy=None)

    if method == "kalman_cv":
        x_filt, _, _, _ = kalman_filter_cv_2d(z_xy=z_xy, R_xy=None, dt=dt, sigma_a=sigma_a)
        return x_filt[:, :2].astype(np.float32)

    if method == "kalman_rts":
        x_filt, P_filt, x_pred, P_pred = kalman_filter_cv_2d(z_xy=z_xy, R_xy=None, dt=dt, sigma_a=sigma_a)
        x_sm, _ = rts_smoother(x_filt=x_filt, P_filt=P_filt, x_pred=x_pred, P_pred=P_pred, dt=dt, sigma_a=sigma_a)
        return x_sm[:, :2].astype(np.float32)

    raise ValueError(f"Unknown filter method: {method}")


# ---------------------------
# Evaluación (raw-only): ANP -> filtro -> MAE
# ---------------------------

@torch.no_grad()
def eval_one_theta_group_multi(
    samples: List,
    anp_model: torch.nn.Module,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    device: torch.device,
    context_percent: int,
    eval_seed: int,
    filter_methods: List[str],
    dt: float,
    use_gt_context_in_filter: bool,
    context_R_eps: float,
    alpha: float,
    beta: float,
    sigma_a: float,
    return_per_sample: bool = False,
) -> Tuple[float, Dict[str, float], Optional[List[float]], Optional[Dict[str, List[float]]]]:
    """
    Devuelve:
      mae_raw (float),
      mae_filt_dict: {method: mae_filtered_mean}

    Si return_per_sample=True, además:
      maes_raw_list: List[float] (por trayectoria)
      maes_filt_lists: Dict[str, List[float]] (por trayectoria)
    """
    ds = NavigationTrajectoryDataset(samples)
    loader = DataLoader(ds, batch_size=8, shuffle=False)

    g = torch.Generator(device=device)
    g.manual_seed(eval_seed)

    maes_raw: List[float] = []
    maes_filt: Dict[str, List[float]] = {m: [] for m in filter_methods}

    for x, y in loader:
        x = x.to(device)  # (B,T,D)
        y = y.to(device)  # (B,T,3)
        B, T, _ = x.shape

        n_context = int((context_percent / 100) * T)
        n_context = max(1, min(n_context, T - 1))
        ctx_idx = sample_context_indices_ordered(T, n_context)

        non_ctx_mask = torch.ones(T, dtype=torch.bool, device=device)
        non_ctx_mask[ctx_idx] = False

        y_norm = normalize_y(y, y_mean, y_std)
        cx = x[:, ctx_idx, :]
        cy = y_norm[:, ctx_idx, :]
        pred_mean_norm, pred_var_norm, *_ = anp_model(cx, cy, x)  # (B,T,3)

        pred_mean = denormalize_y(pred_mean_norm, y_mean, y_std)  # (B,T,3)

        ctx_idx_np = ctx_idx.detach().cpu().numpy()
        non_ctx_mask_np = non_ctx_mask.detach().cpu().numpy()
        ctx_set = set(ctx_idx_np.tolist())

        for b in range(B):
            y_true_np = y[b].detach().cpu().numpy()
            y_pred_np = pred_mean[b].detach().cpu().numpy()

            mae_raw = float(np.mean(np.abs(y_pred_np[non_ctx_mask_np, :] - y_true_np[non_ctx_mask_np, :])))
            maes_raw.append(mae_raw)

            z_xy = y_pred_np[:, :2].copy()
            if use_gt_context_in_filter:
                z_xy[ctx_idx_np, :] = y_true_np[ctx_idx_np, :2]

            for method in filter_methods:
                x_filt_xy = postprocess_filter(
                    z_xy=z_xy,
                    R_xy=None,
                    method=method,
                    dt=dt,
                    alpha=alpha,
                    beta=beta,
                    sigma_a=sigma_a,
                )
                y_filt = y_pred_np.copy()
                y_filt[:, :2] = x_filt_xy

                mae_f = float(np.mean(np.abs(y_filt[non_ctx_mask_np, :] - y_true_np[non_ctx_mask_np, :])))
                maes_filt[method].append(mae_f)

    mae_raw = float(np.mean(maes_raw)) if maes_raw else float("nan")
    mae_filt_dict = {m: float(np.mean(v)) if v else float("nan") for m, v in maes_filt.items()}

    if return_per_sample:
        return mae_raw, mae_filt_dict, maes_raw, maes_filt

    return mae_raw, mae_filt_dict, None, None


@torch.no_grad()
def eval_one_theta_group(
    samples: List,
    anp_model: torch.nn.Module,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    device: torch.device,
    context_percent: int,
    eval_seed: int,
    filter_method: str,
    dt: float,
    use_gt_context_in_filter: bool,
    context_R_eps: float,
    alpha: float,
    beta: float,
    sigma_a: float,
) -> Tuple[float, float]:
    """Wrapper: eval de un único método usando la implementación multi (raw-only)."""
    mae_raw, d, _, _ = eval_one_theta_group_multi(
        samples=samples,
        anp_model=anp_model,
        y_mean=y_mean,
        y_std=y_std,
        device=device,
        context_percent=context_percent,
        eval_seed=eval_seed,
        filter_methods=[filter_method],
        dt=dt,
        use_gt_context_in_filter=use_gt_context_in_filter,
        context_R_eps=context_R_eps,
        alpha=alpha,
        beta=beta,
        sigma_a=sigma_a,
        return_per_sample=False,
    )
    return mae_raw, d[filter_method]


# ---------------------------
# Evaluación (compare 4 u 8 métodos)
# ---------------------------

COMPARE_KEYS_4 = ["raw", "alpha_beta", "kalman_cv", "kalman_rts"]
COMPARE_KEYS_8 = ["raw", "alpha_beta", "kalman_cv", "kalman_rts",
                  "ar_raw", "ar_alpha_beta", "ar_kalman_cv", "ar_kalman_rts"]


@torch.no_grad()
def eval_one_theta_group_compare(
    samples: List,
    anp_model: torch.nn.Module,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    device: torch.device,
    context_percent: int,
    eval_seed: int,
    dt: float,
    use_gt_context_in_filter: bool,
    context_R_eps: float,
    alpha: float,
    beta: float,
    sigma_a: float,
    include_ar: bool,
    # AR params
    ar_use_mu_as_z: bool,
    ar_order: str,
    ar_block_k: int,
    ar_order_seed: int,
    ar_var_thresh: float,
    ar_force_accept: bool,
    ar_max_added: int,
    ar_gate_debug: bool,
    return_per_sample: bool = True,
) -> Tuple[Dict[str, float], Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Devuelve:
      maes_mean: Dict[key, float]
      maes_lists: Dict[key, List[float]]  (por trayectoria)
      times: Dict[key, List[float]] (tiempos por trayectoria)
    """
    keys = COMPARE_KEYS_8 if include_ar else COMPARE_KEYS_4
    maes: Dict[str, List[float]] = {k: [] for k in keys}
    times: Dict[str, List[float]] = {k: [] for k in keys}  # seconds per trajectory

    ds = NavigationTrajectoryDataset(samples)
    loader = DataLoader(ds, batch_size=8, shuffle=False)

    sample_counter = 0

    for x, y in loader:
        x = x.to(device)  # (B,T,D)
        y = y.to(device)  # (B,T,3)
        B, T, _ = x.shape

        n_context = int((context_percent / 100) * T)
        n_context = max(1, min(n_context, T - 1))
        ctx_idx = sample_context_indices_ordered(T, n_context).to(device)

        non_ctx_mask = torch.ones(T, dtype=torch.bool, device=device)
        non_ctx_mask[ctx_idx] = False
        non_ctx_mask_np = non_ctx_mask.detach().cpu().numpy()
        ctx_idx_np = ctx_idx.detach().cpu().numpy()

        # raw parallel ANP (1 forward por batch)
        y_norm = normalize_y(y, y_mean, y_std)
        cx = x[:, ctx_idx, :]
        cy = y_norm[:, ctx_idx, :]
        t0 = timed_block(device)
        pred_mean_norm, pred_var_norm, *_ = anp_model(cx, cy, x)
        t1 = timed_block(device)
        batch_forward_s = t1 - t0
        per_sample_forward_s = batch_forward_s / float(B)

        pred_mean = denormalize_y(pred_mean_norm, y_mean, y_std)

        for b in range(B):
            y_true_np = y[b].detach().cpu().numpy()
            y_pred_np = pred_mean[b].detach().cpu().numpy()

            times["raw"].append(per_sample_forward_s)

            # ---- stream RAW ----
            maes["raw"].append(float(np.mean(np.abs(y_pred_np[non_ctx_mask_np, :] - y_true_np[non_ctx_mask_np, :]))))

            z_xy = y_pred_np[:, :2].copy()
            if use_gt_context_in_filter:
                z_xy[ctx_idx_np, :] = y_true_np[ctx_idx_np, :2]

            # alpha_beta / kalman_cv / kalman_rts sobre RAW
            for m in ["alpha_beta", "kalman_cv", "kalman_rts"]:
                t0 = time.perf_counter()

                x_filt_xy = postprocess_filter(
                    z_xy=z_xy, R_xy=None, method=m, dt=dt, alpha=alpha, beta=beta, sigma_a=sigma_a
                )
                dt_filter = time.perf_counter() - t0
                times[m].append(per_sample_forward_s + dt_filter)
                y_filt = y_pred_np.copy()
                y_filt[:, :2] = x_filt_xy
                maes[m].append(float(np.mean(np.abs(y_filt[non_ctx_mask_np, :] - y_true_np[non_ctx_mask_np, :]))))

            # ---- stream AR (opcional) ----
            if include_ar:
                # predicción AR (por trayectoria)
                order_seed = ar_order_seed + sample_counter
                t0 = timed_block(device)
                y_ar = anp_ar_rollout_one(
                    model=anp_model,
                    x=x[b:b+1],
                    y=y[b:b+1],
                    ctx_idx=ctx_idx,
                    y_mean=y_mean,
                    y_std=y_std,
                    use_mu_as_z=ar_use_mu_as_z,
                    order_mode=ar_order,
                    block_k=ar_block_k,
                    order_seed=order_seed,
                    var_thresh=ar_var_thresh,
                    force_accept=ar_force_accept,
                    max_added=ar_max_added,
                    gate_debug=ar_gate_debug,
                )
                t1 = timed_block(device)
                ar_s = t1 - t0
                times["ar_raw"].append(ar_s)
                y_ar_np = y_ar.squeeze(0).detach().cpu().numpy()

                maes["ar_raw"].append(float(np.mean(np.abs(y_ar_np[non_ctx_mask_np, :] - y_true_np[non_ctx_mask_np, :]))))

                z_xy_ar = y_ar_np[:, :2].copy()
                if use_gt_context_in_filter:
                    z_xy_ar[ctx_idx_np, :] = y_true_np[ctx_idx_np, :2]

                for m, key in [("alpha_beta", "ar_alpha_beta"), ("kalman_cv", "ar_kalman_cv"), ("kalman_rts", "ar_kalman_rts")]:
                    t0 = time.perf_counter()
                    x_filt_xy = postprocess_filter(
                        z_xy=z_xy_ar, R_xy=None, method=m, dt=dt, alpha=alpha, beta=beta, sigma_a=sigma_a
                    )
                    dt_filter = time.perf_counter() - t0
                    y_filt = y_ar_np.copy()
                    y_filt[:, :2] = x_filt_xy
                    maes[key].append(float(np.mean(np.abs(y_filt[non_ctx_mask_np, :] - y_true_np[non_ctx_mask_np, :]))))
                    times[key].append(ar_s + dt_filter)

            sample_counter += 1

    maes_mean = {k: float(np.mean(v)) if len(v) else float("nan") for k, v in maes.items()}
    return maes_mean, maes, times


# ---------------------------
# Cualitativos y boxplot
# (cualitativos: solo RAW stream)
# ---------------------------

def _add_figure_legend(fig, handles, labels):
    """Leyenda a nivel de figura; fallback si 'outside' no existe."""
    try:
        fig.legend(handles, labels, loc="outside right")
    except Exception:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5))


@torch.no_grad()
def predict_and_postprocess_single(
    x_np: np.ndarray,
    y_np: np.ndarray,
    anp_model: torch.nn.Module,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    device: torch.device,
    context_percent: int,
    eval_seed: int,
    methods: List[str],
    dt: float,
    use_gt_context_in_filter: bool,
    context_R_eps: float,
    alpha: float,
    beta: float,
    sigma_a: float,
) -> Dict:
    T = x_np.shape[0]
    x = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(0)
    y = torch.tensor(y_np, dtype=torch.float32, device=device).unsqueeze(0)

    g = torch.Generator(device=device)
    g.manual_seed(eval_seed)

    n_context = int((context_percent / 100) * T)
    n_context = max(1, min(n_context, T - 1))
    ctx_idx = sample_context_indices(T, n_context, g, device=device)
    ctx_idx_np = ctx_idx.detach().cpu().numpy()

    non_ctx_mask = torch.ones(T, dtype=torch.bool, device=device)
    non_ctx_mask[ctx_idx] = False
    non_ctx_mask_np = non_ctx_mask.detach().cpu().numpy()

    y_norm = normalize_y(y, y_mean, y_std)
    cx = x[:, ctx_idx, :]
    cy = y_norm[:, ctx_idx, :]
    pred_mean_norm, pred_var_norm, *_ = anp_model(cx, cy, x)

    pred_mean = denormalize_y(pred_mean_norm, y_mean, y_std)
    y_pred_np = pred_mean.squeeze(0).detach().cpu().numpy()

    z_xy = y_pred_np[:, :2].copy()
    if use_gt_context_in_filter:
        z_xy[ctx_idx_np, :] = y_np[ctx_idx_np, :2]

    preds = {"raw": y_pred_np}
    for m in methods:
        x_filt_xy = postprocess_filter(
            z_xy=z_xy,
            R_xy=None,
            method=m,
            dt=dt,
            alpha=alpha,
            beta=beta,
            sigma_a=sigma_a,
        )
        y_filt = y_pred_np.copy()
        y_filt[:, :2] = x_filt_xy
        preds[m] = y_filt

    return {"gt": y_np, "preds": preds, "ctx_idx": ctx_idx_np, "non_ctx_mask": non_ctx_mask_np}


def save_qualitative_plot_for_theta(
    theta: float,
    samples: List,
    anp_model: torch.nn.Module,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    device: torch.device,
    context_percent: int,
    base_eval_seed: int,
    methods: List[str],
    dt: float,
    use_gt_context_in_filter: bool,
    context_R_eps: float,
    alpha: float,
    beta: float,
    sigma_a: float,
    n_traj: int,
    pick_seed: int,
    out_path: Path,
):
    rng = np.random.default_rng(pick_seed + int(round(theta * 1000)))
    n_traj = min(n_traj, len(samples))
    idxs = rng.choice(len(samples), size=n_traj, replace=False)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axs = axs.ravel()

    handles_labels = None

    for k, sample_idx in enumerate(idxs):
        ax = axs[k]
        x_np, y_np = samples[int(sample_idx)]

        pack = predict_and_postprocess_single(
            x_np=x_np,
            y_np=y_np,
            anp_model=anp_model,
            y_mean=y_mean,
            y_std=y_std,
            device=device,
            context_percent=context_percent,
            eval_seed=base_eval_seed + int(sample_idx),
            methods=methods,
            dt=dt,
            use_gt_context_in_filter=use_gt_context_in_filter,
            context_R_eps=context_R_eps,
            alpha=alpha,
            beta=beta,
            sigma_a=sigma_a,
        )

        gt = pack["gt"]
        preds = pack["preds"]
        ctx_idx = pack["ctx_idx"]

        ax.plot(gt[:, 0], gt[:, 1], linewidth=2, label="GT")
        ax.scatter(gt[ctx_idx, 0], gt[ctx_idx, 1], s=18, label="Context", zorder=3)
        ax.plot(preds["raw"][:, 0], preds["raw"][:, 1], linestyle="--", label="Raw")
        for m in methods:
            ax.plot(preds[m][:, 0], preds[m][:, 1], label=m)

        ax.set_title(f"θ={theta:.1f} | traj #{int(sample_idx)}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.25)

        if handles_labels is None:
            handles_labels = ax.get_legend_handles_labels()

    for j in range(n_traj, 4):
        axs[j].axis("off")

    if handles_labels is not None:
        handles, labels = handles_labels
        _add_figure_legend(fig, handles, labels)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_mae_boxplot(mae_dict: Dict[str, List[float]], out_path: Path, title: str, showfliers: bool, order: Optional[List[str]] = None):
    labels = order if order is not None else list(mae_dict.keys())
    data = [mae_dict[k] for k in labels]

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    try:
        ax.boxplot(data, tick_labels=labels, showfliers=showfliers)
    except TypeError:
        ax.boxplot(data, labels=labels, showfliers=showfliers)

    ax.set_ylabel("MAE (solo NO-contexto)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------
# TXT comparativa (4 u 8)
# ---------------------------

def _write_compare_txt_ar8(
    out_path: Path,
    args,
    theta_values: List[float],
    per_theta: Dict[str, List[float]],
    keys: List[str],
    per_theta_time: Dict[str, List[float]],
):
    g_raw = float(np.mean(per_theta["raw"]))
    lines: List[str] = []
    lines.append("Postprocess comparison report")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Topology={args.topology} | context={args.context}% | dt={args.dt} | eval_seed={args.eval_seed} | device={args.device}")
    lines.append(f"use_gt_context={args.use_gt_context} | context_R_eps={args.context_R_eps}")
    lines.append(f"alpha_beta: alpha={args.ab_alpha} beta={args.ab_beta} | kalman: sigma_a={args.kf_sigma_a}")
    if args.compare_include_ar:
        lines.append(f"AR: order={args.ar_order} | block_k={args.ar_block_k} | use_mu_as_z={args.use_mu_as_z} | var_thresh={args.ar_var_thresh} | force_accept={args.ar_force_accept} | max_added={args.ar_max_added}")
    lines.append("")

    lines.append("=== Global means (NO-context MAE) ===")
    for k in keys:
        gk = float(np.mean(per_theta[k]))
        impr = g_raw - gk
        imprp = (100.0 * impr / g_raw) if g_raw != 0 else float('nan')
        lines.append(f"Global mean {k:<13s}: {gk:.4f} | improv_vs_raw={impr:+.4f} | improv%={imprp:+.2f}%")
    lines.append("")

    lines.append("=== Per-theta ===")
    header = f"{'theta':>6} " + " ".join([f"{k:>14s}" for k in keys]) + " " + " ".join([f"{('impr_'+k):>14s}" for k in keys if k != 'raw'])
    lines.append(header)

    for i, th in enumerate(theta_values):
        row = f"{th:6.1f} "
        for k in keys:
            row += f"{per_theta[k][i]:14.4f} "
        for k in keys:
            if k == "raw":
                continue
            row += f"{(per_theta['raw'][i] - per_theta[k][i]):14.4f} "
        lines.append(row)

    lines.append("")
    lines.append("=== Global mean time (ms/trajectory) ===")
    t_raw = float(np.mean(per_theta_time["raw"])) * 1000.0
    for k in keys:
        tk = float(np.mean(per_theta_time[k])) * 1000.0
        rel = (tk / t_raw) if t_raw > 0 else float("nan")
        lines.append(f"Global mean time {k:<13s}: {tk:8.2f} ms | x{rel:5.2f} vs raw")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def pareto_mask_2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Pareto (minimización) para 2 objetivos:
      x = tiempo (ms)
      y = MAE
    Devuelve mask booleano con los puntos NO dominados.
    """
    n = len(x)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        # j domina a i si es <= en ambos y < en al menos uno
        for j in range(n):
            if i == j:
                continue
            if (x[j] <= x[i] and y[j] <= y[i]) and (x[j] < x[i] or y[j] < y[i]):
                mask[i] = False
                break
    return mask


def save_pareto_mae_vs_time(
    mae_dict: Dict[str, List[float]],
    time_dict: Dict[str, List[float]],
    out_path: Path,
    title: str,
    order: List[str],
    use_median: bool = False,
    annotate: bool = True,
    show_supported_hull: bool = False
):
    """
    Dibuja Pareto MAE vs tiempo (ms/trajectory).
    - mae_dict[k] lista de MAE por trayectoria (o por-theta, pero mejor por trayectoria)
    - time_dict[k] lista de tiempos por trayectoria (segundos)
    """
    labels = order
    mae_vals = []
    time_ms = []

    for k in labels:
        maes = np.asarray(mae_dict[k], dtype=float)
        ts = np.asarray(time_dict[k], dtype=float) * 1000.0  # -> ms

        if use_median:
            mae_vals.append(float(np.median(maes)))
            time_ms.append(float(np.median(ts)))
        else:
            mae_vals.append(float(np.mean(maes)))
            time_ms.append(float(np.mean(ts)))

    mae_vals = np.asarray(mae_vals)
    time_ms = np.asarray(time_ms)

    # Pareto no dominado (ya lo tienes)
    front = pareto_mask_2d(time_ms, mae_vals)
    idx_front = np.where(front)[0]
    idx_front = idx_front[np.argsort(time_ms[idx_front])]

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.scatter(time_ms, mae_vals)
    ax.scatter(time_ms[front], mae_vals[front])
    ax.plot(time_ms[idx_front], mae_vals[idx_front], label="Pareto (no dominado)")

    # --- Supported hull (lower convex hull) ---
    if show_supported_hull and len(time_ms) >= 2:
        pts = np.column_stack([time_ms, mae_vals])
        idx_hull = lower_hull_indices(pts)
        ax.plot(time_ms[idx_hull], mae_vals[idx_hull], linestyle="--", label="Supported hull (lower convex hull)")

    if annotate:
        for i, k in enumerate(labels):
            ax.annotate(k, (time_ms[i], mae_vals[i]))

    ax.set_xlabel("Tiempo medio (ms/trajectory)")
    ax.set_ylabel("MAE medio" + (" (mediana)" if use_median else ""))
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def lower_hull_indices(points_xy: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Devuelve índices del LOWER convex hull para puntos 2D (x,y) en orden creciente de x.
    Útil para minimización (queremos la envolvente "inferior").

    points_xy: (N,2) con columnas [x, y]
    """
    pts = np.asarray(points_xy, dtype=float)
    n = pts.shape[0]
    if n <= 1:
        return np.arange(n)

    # ordenar por x y luego por y
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts_sorted = pts[order]

    def cross(o, a, b):
        # producto cruzado (OA x OB)
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for i, p in enumerate(pts_sorted):
        while len(lower) >= 2:
            o = pts_sorted[lower[-2]]
            a = pts_sorted[lower[-1]]
            b = p
            # Para lower hull, quitamos giros no "clockwise" (cross <= 0) según convención.
            # eps evita problemas numéricos / colinealidad.
            if cross(o, a, b) <= eps:
                lower.pop()
            else:
                break
        lower.append(i)

    # mapear a índices originales
    hull_sorted_idx = np.array(lower, dtype=int)
    hull_original_idx = order[hull_sorted_idx]
    # asegurar orden por x creciente
    hull_original_idx = hull_original_idx[np.argsort(pts[hull_original_idx, 0])]
    return hull_original_idx

# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--anp-dir", type=str, required=True)
    parser.add_argument("--topology", type=str, required=True, choices=["ellipsoidal", "random", "aligned"])
    parser.add_argument("--context", type=int, default=40)
    parser.add_argument("--eval-seed", type=int, default=0)

    parser.add_argument("--filter", type=str, default="none",
                        choices=["none", "alpha_beta", "kalman_cv", "kalman_rts", "compare"])
    parser.add_argument("--out-txt", type=str, default="postprocess_comparison.txt",
                        help="(solo con --filter compare) archivo txt de comparativa")

    parser.add_argument("--dt", type=float, default=1.0, help="dt del filtro")
    parser.add_argument("--use-gt-context", action="store_true", help="Usa GT en puntos de contexto como medidas del filtro")
    parser.add_argument("--context-R-eps", type=float, default=1e-4, help="Ruido de medida para puntos de contexto (si use-gt-context)")

    parser.add_argument("--max-per-theta", type=int, default=-1, help="Limita nº de trayectorias por theta (debug)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    # hyperparams filtros
    parser.add_argument("--ab-alpha", type=float, default=0.85)
    parser.add_argument("--ab-beta", type=float, default=0.005)
    parser.add_argument("--kf-sigma-a", type=float, default=0.11, help="Std de aceleración para Q")

    # compare options
    parser.add_argument("--compare-include-ar", action="store_true",
                        help="En --filter compare, añade también ar_raw y ar+filtros (8 resultados en total).")

    # AR params (defaults = tu configuración óptima)
    parser.add_argument("--ar-order", type=str, default="closest", choices=["time", "closest", "random"])
    parser.add_argument("--ar-block-k", type=int, default=5)
    parser.add_argument("--ar-order-seed", type=int, default=0)
    parser.add_argument("--use-mu-as-z", action="store_true", help="En AR rollout, usa mu como z fijo (en vez de sample).")
    parser.add_argument("--ar-var-thresh", type=float, default=0.01, help="Umbral de varianza (XY, norm-space) para gating; <=0 desactiva gating.")
    parser.add_argument("--ar-force-accept", action="store_true", help="Si no pasa ninguno el umbral en un bloque, acepta el de menor varianza.")
    parser.add_argument("--ar-max-added", type=int, default=-1, help="Máximo nº de pseudo-puntos añadidos al contexto (-1 sin límite).")
    parser.add_argument("--ar-gate-debug", action="store_true", help="Imprime percentiles de varianza XY por bloque (debug).")

    # cualitativos (solo RAW)
    parser.add_argument("--make-qual-plots", action="store_true",
                        help="(mejor con --filter compare) plots por theta con 4 trayectorias al azar (solo raw y filtros)")
    parser.add_argument("--qual-n", type=int, default=4)
    parser.add_argument("--qual-seed", type=int, default=123)
    parser.add_argument("--qual-dir", type=str, default="qualitative_plots")

    # boxplot
    parser.add_argument("--make-boxplot", action="store_true", help="(mejor con --filter compare) boxplot con MAE de todas las trayectorias")
    parser.add_argument("--boxplot-path", type=str, default="mae_boxplot.png")
    parser.add_argument("--boxplot-showfliers", action="store_true", help="muestra outliers como puntos (si no, los oculta)")

    parser.add_argument("--pareto-hull", action="store_true", help="Dibuja la envolvente convexa inferior (supported hull) en el Pareto.")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        ("cuda" if args.device == "cuda" else "cpu")
    )
    print(f"Device: {device}")
    set_all_seeds(args.eval_seed)

    theta_groups, theta_values = load_topology_test_grouped(Path(args.data_dir), args.topology)
    print(f"Theta values: {theta_values}")

    first = next(iter(theta_groups.values()))[0]
    input_dim = first[0].shape[-1]
    output_dim = first[1].shape[-1]
    print(f"Input dim={input_dim}, Output dim={output_dim}")

    y_mean, y_std = get_y_stats_from_train(Path(args.data_dir), args.topology, device=device)
    print(f"y_mean={y_mean.detach().cpu().numpy()}, y_std={y_std.detach().cpu().numpy()}")

    anp = load_anp_model(Path(args.anp_dir), args.topology, input_dim, output_dim, device=device)
    print("Loaded ANP.")

    # ---- modo normal (raw-only) ----
    if args.filter != "compare":
        global_raw: List[float] = []
        global_filt: List[float] = []

        for theta in theta_values:
            samples = theta_groups[theta]
            if args.max_per_theta > 0:
                samples = samples[:args.max_per_theta]

            mae_raw, mae_f = eval_one_theta_group(
                samples=samples,
                anp_model=anp,
                y_mean=y_mean,
                y_std=y_std,
                device=device,
                context_percent=args.context,
                eval_seed=args.eval_seed,
                filter_method=args.filter,
                dt=args.dt,
                use_gt_context_in_filter=args.use_gt_context,
                context_R_eps=args.context_R_eps,
                alpha=args.ab_alpha,
                beta=args.ab_beta,
                sigma_a=args.kf_sigma_a,
            )

            global_raw.append(mae_raw)
            global_filt.append(mae_f)
            print(f"[θ={theta:.1f}] MAE raw={mae_raw:.4f} | MAE filt={mae_f:.4f}")

        print("\n=== Summary ===")
        print(f"Topology={args.topology} | context={args.context}% | filter={args.filter} | dt={args.dt}")
        print(f"Global mean raw : {float(np.mean(global_raw)):.4f}")
        print(f"Global mean filt: {float(np.mean(global_filt)):.4f}")
        return

    # ---- compare mode ----
    include_ar = bool(args.compare_include_ar)
    keys = COMPARE_KEYS_8 if include_ar else COMPARE_KEYS_4

    per_theta: Dict[str, List[float]] = {k: [] for k in keys}
    mae_all: Dict[str, List[float]] = {k: [] for k in keys}
    per_theta_time: Dict[str, List[float]] = {k: [] for k in keys}
    time_all: Dict[str, List[float]] = {k: [] for k in keys}

    for theta in theta_values:
        samples = theta_groups[theta]
        if args.max_per_theta > 0:
            samples = samples[:args.max_per_theta]

        mae_mean, mae_lists, times_lists = eval_one_theta_group_compare(
            samples=samples,
            anp_model=anp,
            y_mean=y_mean,
            y_std=y_std,
            device=device,
            context_percent=args.context,
            eval_seed=args.eval_seed,
            dt=args.dt,
            use_gt_context_in_filter=args.use_gt_context,
            context_R_eps=args.context_R_eps,
            alpha=args.ab_alpha,
            beta=args.ab_beta,
            sigma_a=args.kf_sigma_a,
            include_ar=include_ar,
            ar_use_mu_as_z=bool(args.use_mu_as_z),
            ar_order=args.ar_order,
            ar_block_k=args.ar_block_k,
            ar_order_seed=args.ar_order_seed,
            ar_var_thresh=args.ar_var_thresh,
            ar_force_accept=args.ar_force_accept,
            ar_max_added=args.ar_max_added,
            ar_gate_debug=args.ar_gate_debug,
            return_per_sample=True,
        )

        print(f"[θ={theta:.1f}]")
        for k in keys:
            per_theta[k].append(mae_mean[k])
            mae_all[k].extend(mae_lists[k])
            per_theta_time[k].append(float(np.mean(times_lists[k])))
            time_all[k].extend(times_lists[k])
            if k == "raw":
                continue
            print(f"    - {k:13s} MAE={mae_mean[k]:.4f} | improv_vs_raw={mae_mean['raw'] - mae_mean[k]:+.4f}")
        print(f"    - {'raw':13s} MAE={mae_mean['raw']:.4f}")

    out_txt = Path(args.out_txt)
    _write_compare_txt_ar8(out_txt, args, theta_values, per_theta, keys, per_theta_time)
    print(f"\n[OK] Guardado comparativa en: {out_txt.resolve()}")

    # Pareto MAE vs tiempo (ms/trajectory)
    save_pareto_mae_vs_time(
        mae_dict=mae_all,
        time_dict=time_all,
        out_path=Path("pareto_mae_vs_time.png"),
        title=f"Pareto MAE vs Time | topology={args.topology} | context={args.context}% | include_ar={include_ar}",
        order=keys,
        use_median=False,
        annotate=True,
        show_supported_hull=args.pareto_hull,
    )
    print("[OK] Pareto guardado en: pareto_mae_vs_time.png")

    # boxplot MAE por trayectoria
    if args.make_boxplot:
        save_mae_boxplot(
            mae_dict=mae_all,
            out_path=Path(args.boxplot_path),
            title=f"MAE distribution | topology={args.topology} | context={args.context}% | include_ar={include_ar}",
            showfliers=args.boxplot_showfliers,
            order=keys,
        )
        print(f"[OK] Boxplot guardado en: {Path(args.boxplot_path).resolve()}")

    # cualitativos: solo raw + filtros clásicos
    if args.make_qual_plots:
        qdir = Path(args.qual_dir)
        qdir.mkdir(parents=True, exist_ok=True)
        methods = ["alpha_beta", "kalman_cv", "kalman_rts"]

        for theta in theta_values:
            samples = theta_groups[theta]
            if args.max_per_theta > 0:
                samples = samples[:args.max_per_theta]

            out_fig = qdir / f"theta_{theta:.1f}.png"
            save_qualitative_plot_for_theta(
                theta=theta,
                samples=samples,
                anp_model=anp,
                y_mean=y_mean,
                y_std=y_std,
                device=device,
                context_percent=args.context,
                base_eval_seed=args.eval_seed,
                methods=methods,
                dt=args.dt,
                use_gt_context_in_filter=args.use_gt_context,
                context_R_eps=args.context_R_eps,
                alpha=args.ab_alpha,
                beta=args.ab_beta,
                sigma_a=args.kf_sigma_a,
                n_traj=args.qual_n,
                pick_seed=args.qual_seed,
                out_path=out_fig,
            )

        print(f"[OK] Plots cualitativos guardados en: {qdir.resolve()}")


if __name__ == "__main__":
    main()
