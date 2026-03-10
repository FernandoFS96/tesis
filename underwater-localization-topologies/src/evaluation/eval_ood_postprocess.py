"""
eval_ood_postprocess.py
=======================
Cierra el gap Out-of-Distribution entre el modelo entrenado en datos de alta
varianza (highvar) y el oráculo entrenado en baja varianza (lowvar), aplicando
técnicas de postprocesado al flujo de predicciones del modelo degradado.

Escenario OoD evaluado:
  Oracle  : modelo lowvar  evaluado en datos lowvar test  → ~5 m MAE (techo)
  Baseline: modelo highvar evaluado en datos lowvar test  → ~11 m MAE (partida)

Métodos de postprocesado aplicados al modelo highvar:
  raw            - sin postprocesado
  alpha_beta     - filtro g-h (alpha-beta)
  kalman_cv_I    - Kalman velocidad constante, R = identidad
  kalman_cv_var  - Kalman velocidad constante, R = varianza ANP  
  kalman_rts_I   - suavizador RTS offline, R = identidad
  kalman_rts_var - suavizador RTS offline, R = varianza ANP      
  ar_raw         - rollout autoregresivo test-time (AR)
  ar_kalman_rts_I   - AR + RTS, R = identidad
  ar_kalman_rts_var - AR + RTS, R = varianza ANP                  

Salidas (en --output-dir):
  mae_summary.csv
  mae_summary.txt
  mae_boxplot.png
  mae_heatmap.png (MAE grid: métodos x theta, oracle como fila de referencia)
  latency_per_method.csv
  latency_plot.png (latencia absoluta + overhead vs raw ANP)
  trajectories/theta_{theta:.1f}.png  (una figura 2x2 por valor de theta)

Uso:
  python eval_ood_postprocess.py \
    --lowvar-ckpt /home/fernando/tesis/underwater-localization-topologies/src/training/results/ANP_topologies_masked/lowvar/masked_dropbernoulli_p0.2_train_mean_first/topology_ellipsoidal/best_checkpoint.pth.tar \
    --highvar-ckpt /home/fernando/tesis/underwater-localization-topologies/src/training/results/ANP_topologies_masked/highvar/masked_dropbernoulli_p0.2_train_mean_first/topology_ellipsoidal/best_checkpoint.pth.tar \
    --lowvar-data-dir /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_low_variance \
    --highvar-data-dir /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_high_variance \
    --topology ellipsoidal \
    --context 30 \
    --output-dir results/eval_ood_postprocess
"""

from __future__ import annotations

import argparse
import pickle
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.anp import LatentModel
from src.utils.nav_dataset import NavigationTrajectoryDataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVAL_SEED = 18
ALL_METHODS = [
    "raw",
    "alpha_beta",
    "kalman_cv_I",
    "kalman_cv_var",
    "kalman_cv_calib",      # calibrated R + velocity init from context
    "kalman_rts_I",
    "kalman_rts_var",
    "ar_raw",
    "ar_uncert",            # AR ordered by ascending predicted variance
    "ar_kalman_rts_I",
    "ar_kalman_rts_var",
]

METHOD_LABELS = {
    "raw":               "Raw",
    "alpha_beta":        "Alpha-Beta",
    "kalman_cv_I":       "Kal CV (R=I)",
    "kalman_cv_var":     "Kal CV (R=\u03c3\u00b2)",
    "kalman_cv_calib":   "Kal CV (R=calib)",
    "kalman_rts_I":      "RTS (R=I)",
    "kalman_rts_var":    "RTS (R=\u03c3\u00b2)",
    "ar_raw":            "AR raw",
    "ar_uncert":         "AR (by \u03c3\u00b2)",
    "ar_kalman_rts_I":   "AR+RTS (R=I)",
    "ar_kalman_rts_var": "AR+RTS (R=\u03c3\u00b2)",
}

METHOD_COLORS = {
    "raw":               "#e74c3c",
    "alpha_beta":        "#e67e22",
    "kalman_cv_I":       "#f1c40f",
    "kalman_cv_var":     "#2ecc71",
    "kalman_cv_calib":   "#16a085",
    "kalman_rts_I":      "#1abc9c",
    "kalman_rts_var":    "#27ae60",
    "ar_raw":            "#3498db",
    "ar_uncert":         "#2980b9",
    "ar_kalman_rts_I":   "#9b59b6",
    "ar_kalman_rts_var": "#6c3483",
}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _detect_input_dim(ckpt_path: Path) -> int:
    """
    Infer input_dim from the checkpoint.

    In LatentModel the deterministic encoder's context_projection and
    target_projection process only x (no y concatenated), so their weight
    shape is [num_hidden, input_dim].  The input_projection instead processes
    concat(x, y) → [num_hidden, input_dim + output_dim], which is 3 too large.
    We therefore look specifically for context_projection / target_projection
    weights first, then fall back to decoder.target_projection.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # Preferred: keys whose name contains context_projection or
    # decoder's target_projection (pure-x projections)
    preferred_substrings = [
        "context_projection.linear_layer.weight",
        "decoder.target_projection.linear_layer.weight",
        "target_projection.linear_layer.weight",
    ]
    for sub in preferred_substrings:
        for k, v in state.items():
            if sub in k and v.ndim == 2:
                return int(v.shape[1])

    raise RuntimeError(
        f"Cannot detect input_dim from {ckpt_path}. "
        "Keys available: " + ", ".join(list(state.keys())[:10])
    )


def load_anp(ckpt_path: Path, device: torch.device) -> LatentModel:
    """Load LatentModel; auto-detect input_dim from checkpoint."""
    input_dim = _detect_input_dim(ckpt_path)
    model = LatentModel(num_hidden=128, input_dim=input_dim, output_dim=3)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    return model.to(device).eval()


def load_data(data_dir: Path, topology: str) -> Tuple[List, List, Dict]:
    tdir = Path(data_dir) / f"topology_{topology}"
    with open(tdir / "test_data.pkl", "rb") as f:
        test_data = pickle.load(f)
    with open(tdir / "train_data.pkl", "rb") as f:
        train_data = pickle.load(f)
    with open(tdir / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return test_data, train_data, metadata


def compute_y_stats(
    train_data: List, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    Y = np.concatenate([y for _, y in train_data], axis=0)
    y_mean = torch.tensor(Y.mean(axis=0), dtype=torch.float32, device=device)
    y_std  = torch.tensor(Y.std(axis=0)  + 1e-6, dtype=torch.float32, device=device)
    return y_mean, y_std


def group_by_theta(
    test_data: List, metadata: Dict
) -> Dict[float, List]:
    groups: Dict[float, List] = {}
    for sample, theta in zip(test_data, metadata["test_thetas"]):
        groups.setdefault(float(theta), []).append(sample)
    return groups


def norm_y(
    y: torch.Tensor, y_mean: torch.Tensor, y_std: torch.Tensor
) -> torch.Tensor:
    return (y - y_mean.view(1, 1, -1)) / y_std.view(1, 1, -1)


def denorm_y(
    yn: torch.Tensor, y_mean: torch.Tensor, y_std: torch.Tensor
) -> torch.Tensor:
    return yn * y_std.view(1, 1, -1) + y_mean.view(1, 1, -1)


def first_ctx_idx(T: int, ctx_percent: int, device: torch.device) -> torch.Tensor:
    n = max(1, min(int(ctx_percent / 100 * T), T - 1))
    return torch.arange(n, device=device, dtype=torch.long)


def _augment_x_allsensors(x: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    """
    Masked ANP models were trained with an explicit all-sensors binary mask
    appended to x: x_aug = [x_raw | ones_mask]  shape (B,T,Dx+N_sensors).
    Raw test-data files only contain x_raw (Dx features).  This helper appends
    an all-ones mask tensor of the correct width when needed.
    """
    expected: Optional[int] = None
    for name, param in model.named_parameters():
        if "context_projection.linear_layer.weight" in name and param.ndim == 2:
            expected = int(param.shape[1])
            break
    if expected is None or x.shape[-1] >= expected:
        return x  # already the right size (or unmasked model)
    n_extra = expected - x.shape[-1]
    ones = torch.ones(x.shape[0], x.shape[1], n_extra, dtype=x.dtype, device=x.device)
    return torch.cat([x, ones], dim=-1)


# ---------------------------------------------------------------------------
# Post-processing kernels
# ---------------------------------------------------------------------------

def _ab_filter(z_xy: np.ndarray, dt: float, alpha: float, beta: float) -> np.ndarray:
    """Alpha-beta (g-h) 2D filter."""
    T = z_xy.shape[0]
    out = np.zeros((T, 2), dtype=np.float32)
    x, y = float(z_xy[0, 0]), float(z_xy[0, 1])
    vx = vy = 0.0
    out[0] = [x, y]
    for k in range(1, T):
        xp = x + vx * dt;  yp = y + vy * dt
        rx = z_xy[k, 0] - xp;  ry = z_xy[k, 1] - yp
        x = xp + alpha * rx;   y = yp + alpha * ry
        vx += (beta / dt) * rx; vy += (beta / dt) * ry
        out[k] = [x, y]
    return out


def _cv_matrices(dt: float, sigma_a: float):
    F = np.array([[1, 0, dt, 0], [0, 1, 0, dt],
                  [0, 0, 1,  0], [0, 0, 0,  1]], dtype=np.float64)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
    q = sigma_a ** 2
    dt2, dt3, dt4 = dt**2, dt**3, dt**4
    Q1 = np.array([[dt4/4, dt3/2], [dt3/2, dt2]], dtype=np.float64) * q
    Q = np.zeros((4, 4), dtype=np.float64)
    Q[np.ix_([0, 2], [0, 2])] = Q1
    Q[np.ix_([1, 3], [1, 3])] = Q1
    return F, H, Q


def _kalman_cv(
    z_xy: np.ndarray,
    R_xy: Optional[np.ndarray],   # (T,2,2) or None → identity
    dt: float,
    sigma_a: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    T = z_xy.shape[0]
    F, H, Q = _cv_matrices(dt, sigma_a)
    x = np.zeros(4, dtype=np.float64); x[:2] = z_xy[0]
    P = np.eye(4, dtype=np.float64)
    xf, Pf = np.zeros((T, 4)), np.zeros((T, 4, 4))
    xp_all, Pp_all = np.zeros((T, 4)), np.zeros((T, 4, 4))
    I = np.eye(4, dtype=np.float64)
    for t in range(T):
        xp = F @ x;  Pp = F @ P @ F.T + Q
        xp_all[t] = xp;  Pp_all[t] = Pp
        R = np.eye(2) if R_xy is None else R_xy[t].astype(np.float64)
        inn = z_xy[t] - H @ xp
        S = H @ Pp @ H.T + R
        K = Pp @ H.T @ np.linalg.inv(S)
        x = xp + K @ inn
        KH = K @ H
        P = (I - KH) @ Pp @ (I - KH).T + K @ R @ K.T
        xf[t] = x;  Pf[t] = P
    return xf, Pf, xp_all, Pp_all


def _rts(xf, Pf, xp_all, Pp_all, dt, sigma_a):
    T = xf.shape[0]
    F, _, _ = _cv_matrices(dt, sigma_a)
    xs, Ps = xf.copy(), Pf.copy()
    for k in range(T - 2, -1, -1):
        C = Pf[k] @ F.T @ np.linalg.inv(Pp_all[k + 1])
        xs[k] = xf[k] + C @ (xs[k + 1] - xp_all[k + 1])
        Ps[k] = Pf[k] + C @ (Ps[k + 1] - Pp_all[k + 1]) @ C.T
    return xs, Ps


def _build_R(var_xy_np: np.ndarray, eps: float = 0.01) -> np.ndarray:
    """
    Build per-step (T,2,2) diagonal measurement noise from ANP variance.
    var_xy_np: (T,2)  - predicted variance in physical units for x,y.
    """
    T = var_xy_np.shape[0]
    R = np.zeros((T, 2, 2), dtype=np.float64)
    R[:, 0, 0] = np.maximum(var_xy_np[:, 0], eps)
    R[:, 1, 1] = np.maximum(var_xy_np[:, 1], eps)
    return R


def _kalman_cv_calib(
    z_xy:      np.ndarray,   # (T,2)  raw ANP predictions  (physical units)
    var_xy:    np.ndarray,   # (T,2)  ANP predicted variance (physical units)
    ctx_idx:   np.ndarray,   # (N_ctx,)  integer indices of context points
    gt_ctx_xy: np.ndarray,   # (N_ctx,2) ground-truth positions at context points
    dt:        float,
    sigma_a:   float,
    eps:       float = 0.01,
) -> np.ndarray:             # (T,2)
    """
    Kalman CV filter with two online improvements over kalman_cv_var:
      1. R re-calibrated from context residuals: gamma = mean(err²) / mean(σ²)
         This corrects the scale of the ANP's predicted variance, which may be
         poorly calibrated under domain shift.
      2. Initial velocity estimated from the last two context points (causal).
    Both improvements require only context observations → fully online / causal.
    """
    # Sort context by time
    order  = np.argsort(ctx_idx)
    ctx_s  = ctx_idx[order]
    gt_s   = gt_ctx_xy[order]           # gt at ctx, time-sorted

    # 1. Per-axis calibration factor
    pred_ctx = z_xy[ctx_s]              # (N_ctx, 2)
    err_sq   = (pred_ctx - gt_s) ** 2
    var_ctx  = var_xy[ctx_s]
    mean_var = np.mean(var_ctx, axis=0)
    gamma    = np.where(
        mean_var > eps,
        np.mean(err_sq, axis=0) / np.maximum(mean_var, eps),
        1.0,
    )
    gamma    = np.clip(gamma, 0.1, 100.0)
    R_calib  = _build_R(var_xy * gamma, eps=eps)

    # 2. Initial state: position + velocity from context
    T = z_xy.shape[0]
    F, H, Q = _cv_matrices(dt, sigma_a)
    x_s = np.zeros(4, dtype=np.float64)
    x_s[:2] = gt_s[-1] if len(ctx_s) >= 1 else z_xy[0]
    if len(ctx_s) >= 2:
        dt_steps = max(int(ctx_s[-1]) - int(ctx_s[-2]), 1)
        x_s[2]   = (gt_s[-1, 0] - gt_s[-2, 0]) / (dt * dt_steps)
        x_s[3]   = (gt_s[-1, 1] - gt_s[-2, 1]) / (dt * dt_steps)

    P  = np.eye(4, dtype=np.float64)
    xf = np.zeros((T, 4))
    I  = np.eye(4, dtype=np.float64)

    for t in range(T):
        xp = F @ x_s;  Pp = F @ P @ F.T + Q
        R  = R_calib[t]
        inn = z_xy[t] - H @ xp
        S   = H @ Pp @ H.T + R
        K   = Pp @ H.T @ np.linalg.inv(S)
        x_s = xp + K @ inn
        KH  = K @ H
        P   = (I - KH) @ Pp @ (I - KH).T + K @ R @ K.T
        xf[t] = x_s

    return xf[:, :2].astype(np.float32)


def apply_postprocess(
    z_xy:     np.ndarray,    # (T,2) positions in physical units
    var_xy:   np.ndarray,    # (T,2) predicted variance in physical units
    method:   str,
    dt:       float,
    alpha:    float = 0.85,
    beta:     float = 0.005,
    sigma_a:  float = 1.0,
) -> np.ndarray:             # (T,2)
    if method == "alpha_beta":
        return _ab_filter(z_xy, dt, alpha, beta)

    R_I   = None
    R_var = _build_R(var_xy)

    if method == "kalman_cv_I":
        xf, _, _, _ = _kalman_cv(z_xy, R_I,   dt, sigma_a)
        return xf[:, :2].astype(np.float32)
    if method == "kalman_cv_var":
        xf, _, _, _ = _kalman_cv(z_xy, R_var, dt, sigma_a)
        return xf[:, :2].astype(np.float32)
    if method == "kalman_rts_I":
        xf, Pf, xp, Pp = _kalman_cv(z_xy, R_I, dt, sigma_a)
        xs, _ = _rts(xf, Pf, xp, Pp, dt, sigma_a)
        return xs[:, :2].astype(np.float32)
    if method == "kalman_rts_var":
        xf, Pf, xp, Pp = _kalman_cv(z_xy, R_var, dt, sigma_a)
        xs, _ = _rts(xf, Pf, xp, Pp, dt, sigma_a)
        return xs[:, :2].astype(np.float32)

    raise ValueError(f"apply_postprocess: unknown method '{method}'")


# ---------------------------------------------------------------------------
# AR rollout (test-time autoregressive)
# ---------------------------------------------------------------------------

def _ar_rollout(
    model:         torch.nn.Module,
    x:             torch.Tensor,                   # (1,T,D)
    y:             torch.Tensor,                   # (1,T,3) real units
    ctx_idx:       torch.Tensor,
    y_mean:        torch.Tensor,
    y_std:         torch.Tensor,
    order:         str                    = "closest",
    block_k:       int                    = 5,
    var_thresh:    float                  = 0.01,
    init_var_norm: Optional[torch.Tensor] = None,
    force_accept:  bool                   = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (y_pred_phys, y_var_phys), both (1,T,3) in physical units.
    Variance for context points is ~0 (set to a small placeholder).
    order: "closest" | "time" | "uncertainty" | anything else → random.
    """
    device = x.device
    B, T, _ = x.shape
    assert B == 1

    y_norm = norm_y(y, y_mean, y_std)                      # (1,T,3)
    ctx_np = ctx_idx.detach().cpu().numpy().astype(int)
    ctx_set = set(ctx_np.tolist())

    # ordered list of targets
    non_ctx = [t for t in range(T) if t not in ctx_set]
    if order == "closest":
        ctx_sorted = np.sort(ctx_np)
        non_ctx.sort(key=lambda t: (int(np.min(np.abs(ctx_sorted - t))), t))
    elif order == "time":
        pass  # already sorted
    elif order == "uncertainty":
        if init_var_norm is not None:
            var_mean_np = init_var_norm[0, :, :2].mean(-1).detach().cpu().numpy()
        else:
            _, iv_norm, *_ = model(x[:, ctx_idx, :], y_norm[:, ctx_idx, :], x)
            var_mean_np = iv_norm[0, :, :2].mean(-1).detach().cpu().numpy()
        non_ctx.sort(key=lambda t: float(var_mean_np[t]))
    else:
        np.random.shuffle(non_ctx)

    # rolling context
    roll_x = x[:, ctx_idx, :]
    roll_y = y_norm[:, ctx_idx, :]

    mu_fixed, _, _ = model.latent_encoder(roll_x, roll_y)   # type: ignore

    y_pred_norm = torch.zeros_like(y_norm)
    y_var_norm  = torch.zeros_like(y_norm)
    y_pred_norm[:, ctx_idx, :] = roll_y

    K = max(1, block_k)
    for start in range(0, len(non_ctx), K):
        idxs = non_ctx[start:start + K]
        idxs_t = torch.tensor(idxs, dtype=torch.long, device=device)

        tx = x[:, idxs_t, :]
        r  = model.deterministic_encoder(roll_x, roll_y, tx)       # type: ignore
        z  = mu_fixed.unsqueeze(1).expand(1, tx.shape[1], -1)
        mean_k, var_k = model.decoder(r, z, tx)                    # type: ignore

        y_pred_norm[:, idxs_t, :] = mean_k
        y_var_norm[:, idxs_t, :]  = var_k

        v_xy = var_k[:, :, :2].mean(dim=2).squeeze(0)
        accept = v_xy <= var_thresh
        if force_accept and not accept.any():
            accept[torch.argmin(v_xy)] = True

        if accept.any():
            roll_x = torch.cat([roll_x, tx[:, accept, :]], dim=1)
            roll_y = torch.cat([roll_y, mean_k[:, accept, :]], dim=1)

    y_pred_phys = denorm_y(y_pred_norm, y_mean, y_std)
    # denormalize variance: Var[y * std + mean] = Var[y] * std^2
    y_var_phys = y_var_norm * (y_std.view(1, 1, -1) ** 2)
    return y_pred_phys, y_var_phys


# ---------------------------------------------------------------------------
# Core evaluation for a single theta group
# ---------------------------------------------------------------------------

_FILTER_METHODS = [
    "alpha_beta", "kalman_cv_I", "kalman_cv_var",
    "kalman_rts_I", "kalman_rts_var",
]
_AR_FILTER_METHODS = [
    "ar_kalman_rts_I", "ar_kalman_rts_var",
]


@torch.no_grad()
def eval_theta_group(
    samples:     List,
    model:       torch.nn.Module,
    y_mean:      torch.Tensor,
    y_std:       torch.Tensor,
    ctx_pct:     int,
    device:      torch.device,
    dt:          float,
    alpha:       float,
    beta:        float,
    sigma_a:     float,
    ar_block_k:  int,
    ar_var_thresh: float,
    return_per_sample: bool = False,
) -> Tuple[Dict[str, float], Optional[Dict[str, List[float]]], Dict[str, float]]:
    """
    Evaluates all post-processing methods for one theta group.

    Returns:
      maes_mean       : method → mean MAE (m)
      maes_lists      : method → [per-traj MAE]  (if return_per_sample)
      latencies_mean  : method → mean seconds per trajectory
                        For filter methods, this is the POST-PROCESSING overhead only
                        (i.e. excluding the base ANP forward pass).
                        For AR-based methods, it is the total AR rollout time
                        (plus any filter overhead on top).
                        "raw" contains the base ANP forward pass time.
    """
    maes:      Dict[str, List[float]] = {m: [] for m in ALL_METHODS}
    latencies: Dict[str, List[float]] = {m: [] for m in ALL_METHODS}

    ds     = NavigationTrajectoryDataset(samples)
    loader = DataLoader(ds, batch_size=1, shuffle=False)   # batch=1 for AR

    for (x, y) in loader:
        x = x.to(device); y = y.to(device)
        x = _augment_x_allsensors(x, model)               # append mask=1 for masked ANPs
        B, T, _ = x.shape                                 # B==1 (DataLoader batch=1)

        ctx        = first_ctx_idx(T, ctx_pct, device)
        ctx_np_arr = ctx.cpu().numpy().astype(int)
        nc_mask_np = np.ones(T, bool)
        nc_mask_np[ctx_np_arr] = False

        y_norm = norm_y(y, y_mean, y_std)
        cx, cy = x[:, ctx, :], y_norm[:, ctx, :]

        # ---- Standard forward pass (timed) ----
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        mean_norm, var_norm, *_ = model(cx, cy, x)        # (1,T,3)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_raw = time.perf_counter() - t0

        mean_phys = denorm_y(mean_norm, y_mean, y_std)
        # variance in physical units
        var_phys  = var_norm * (y_std.view(1, 1, -1) ** 2)

        y_np   = y.squeeze(0).cpu().numpy()               # (T,3)
        p_np   = mean_phys.squeeze(0).cpu().numpy()       # (T,3)
        v_np   = var_phys.squeeze(0).cpu().numpy()        # (T,3)

        def _mae(pred_np: np.ndarray) -> float:
            return float(np.mean(np.abs(pred_np[nc_mask_np, :] - y_np[nc_mask_np, :])))

        maes["raw"].append(_mae(p_np))
        latencies["raw"].append(t_raw)

        for m in _FILTER_METHODS:
            t0 = time.perf_counter()
            xy_pp = apply_postprocess(
                z_xy=p_np[:, :2], var_xy=v_np[:, :2],
                method=m, dt=dt, alpha=alpha, beta=beta, sigma_a=sigma_a,
            )
            latencies[m].append(time.perf_counter() - t0)  # filter overhead only
            p_pp = p_np.copy(); p_pp[:, :2] = xy_pp
            maes[m].append(_mae(p_pp))

        # --- kalman_cv_calib: calibrated R + velocity init from context ---
        t0 = time.perf_counter()
        xy_calib = _kalman_cv_calib(
            p_np[:, :2], v_np[:, :2], ctx_np_arr, y_np[ctx_np_arr, :2], dt, sigma_a,
        )
        latencies["kalman_cv_calib"].append(time.perf_counter() - t0)
        p_calib = p_np.copy(); p_calib[:, :2] = xy_calib
        maes["kalman_cv_calib"].append(_mae(p_calib))

        # ---- AR rollout (timed) ----
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        ar_phys, ar_var_phys = _ar_rollout(
            model=model, x=x, y=y,
            ctx_idx=ctx, y_mean=y_mean, y_std=y_std,
            block_k=ar_block_k, var_thresh=ar_var_thresh,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_ar = time.perf_counter() - t0

        p_ar = ar_phys.squeeze(0).cpu().numpy()           # (T,3)
        v_ar = ar_var_phys.squeeze(0).cpu().numpy()       # (T,3)
        maes["ar_raw"].append(_mae(p_ar))
        latencies["ar_raw"].append(t_ar)

        # --- AR by uncertainty: sort targets ascending by predicted variance ---
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        ar_uncert_phys, _ = _ar_rollout(
            model=model, x=x, y=y,
            ctx_idx=ctx, y_mean=y_mean, y_std=y_std,
            order="uncertainty", init_var_norm=var_norm,
            block_k=ar_block_k, var_thresh=ar_var_thresh,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies["ar_uncert"].append(time.perf_counter() - t0)
        maes["ar_uncert"].append(_mae(ar_uncert_phys.squeeze(0).cpu().numpy()))

        for ar_key, filt in [("ar_kalman_rts_I", "kalman_rts_I"),
                              ("ar_kalman_rts_var", "kalman_rts_var")]:
            t0 = time.perf_counter()
            xy_pp = apply_postprocess(
                z_xy=p_ar[:, :2], var_xy=v_ar[:, :2],
                method=filt, dt=dt, alpha=alpha, beta=beta, sigma_a=sigma_a,
            )
            latencies[ar_key].append(t_ar + (time.perf_counter() - t0))  # AR + filter
            p_pp = p_ar.copy(); p_pp[:, :2] = xy_pp
            maes[ar_key].append(_mae(p_pp))

    maes_mean  = {k: float(np.mean(v)) if v else float("nan") for k, v in maes.items()}
    lats_mean  = {k: float(np.mean(v)) if v else float("nan") for k, v in latencies.items()}
    if return_per_sample:
        return maes_mean, maes, lats_mean
    return maes_mean, None, lats_mean


# ---------------------------------------------------------------------------
# Single-trajectory prediction (for qualitative plots)
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_single(
    x_np:    np.ndarray,
    y_np:    np.ndarray,
    model:   torch.nn.Module,
    y_mean:  torch.Tensor,
    y_std:   torch.Tensor,
    ctx_pct: int,
    device:  torch.device,
    dt:      float,
    alpha:   float,
    beta:    float,
    sigma_a: float,
    ar_block_k:    int,
    ar_var_thresh: float,
) -> Dict:
    """
    Returns dict with keys: gt, ctx_idx, nc_mask, and per-method predictions.
    Each prediction is (T,3) ndarray in physical units.
    """
    T = x_np.shape[0]
    x = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(0)
    y = torch.tensor(y_np, dtype=torch.float32, device=device).unsqueeze(0)
    x = _augment_x_allsensors(x, model)                   # append mask=1 for masked ANPs

    ctx   = first_ctx_idx(T, ctx_pct, device)
    ctx_np = ctx.cpu().numpy().astype(int)
    nc_mask = np.ones(T, bool); nc_mask[ctx_np] = False

    y_norm = norm_y(y, y_mean, y_std)
    cx, cy = x[:, ctx, :], y_norm[:, ctx, :]
    mean_norm, var_norm, *_ = model(cx, cy, x)
    mean_phys = denorm_y(mean_norm, y_mean, y_std)
    var_phys  = var_norm * (y_std.view(1, 1, -1) ** 2)

    p_np = mean_phys.squeeze(0).cpu().numpy()
    v_np = var_phys.squeeze(0).cpu().numpy()

    preds: Dict[str, np.ndarray] = {"raw": p_np}
    for m in _FILTER_METHODS:
        xy_pp = apply_postprocess(
            z_xy=p_np[:, :2], var_xy=v_np[:, :2],
            method=m, dt=dt, alpha=alpha, beta=beta, sigma_a=sigma_a,
        )
        pp = p_np.copy(); pp[:, :2] = xy_pp
        preds[m] = pp

    # kalman_cv_calib
    xy_calib = _kalman_cv_calib(
        p_np[:, :2], v_np[:, :2], ctx_np, y_np[ctx_np, :2], dt, sigma_a,
    )
    pp = p_np.copy(); pp[:, :2] = xy_calib
    preds["kalman_cv_calib"] = pp

    ar_phys, ar_var = _ar_rollout(
        model=model, x=x, y=y, ctx_idx=ctx,
        y_mean=y_mean, y_std=y_std,
        block_k=ar_block_k, var_thresh=ar_var_thresh,
    )
    p_ar = ar_phys.squeeze(0).cpu().numpy()
    v_ar = ar_var.squeeze(0).cpu().numpy()
    preds["ar_raw"] = p_ar
    ar_uncert_phys, _ = _ar_rollout(
        model=model, x=x, y=y, ctx_idx=ctx,
        y_mean=y_mean, y_std=y_std,
        order="uncertainty", init_var_norm=var_norm,
        block_k=ar_block_k, var_thresh=ar_var_thresh,
    )
    preds["ar_uncert"] = ar_uncert_phys.squeeze(0).cpu().numpy()
    for ar_key, filt in [("ar_kalman_rts_I", "kalman_rts_I"),
                          ("ar_kalman_rts_var", "kalman_rts_var")]:
        xy_pp = apply_postprocess(
            z_xy=p_ar[:, :2], var_xy=v_ar[:, :2],
            method=filt, dt=dt, alpha=alpha, beta=beta, sigma_a=sigma_a,
        )
        pp = p_ar.copy(); pp[:, :2] = xy_pp
        preds[ar_key] = pp

    return {"gt": y_np, "ctx_idx": ctx_np, "nc_mask": nc_mask, "preds": preds}


# ---------------------------------------------------------------------------
# Summary / CSV
# ---------------------------------------------------------------------------

def build_summary_csv(
    all_results: Dict[str, Dict[float, Dict[str, float]]],  # model_key → theta → method → MAE
    output_dir: Path,
) -> None:
    """Save per-theta x per-method MAE table."""
    import csv
    path = output_dir / "mae_summary.csv"
    # collect all thetas
    thetas = sorted({t for r in all_results.values() for t in r})
    methods = ALL_METHODS

    rows = []
    # Header
    header = ["model", "theta"] + methods
    rows.append(header)
    for model_key, theta_dict in all_results.items():
        for theta in thetas:
            if theta not in theta_dict:
                continue
            method_maes = theta_dict[theta]
            row = [model_key, f"{theta:.1f}"] + [
                f"{method_maes.get(m, float('nan')):.4f}" for m in methods
            ]
            rows.append(row)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"[✓] Saved {path}")


def build_summary_txt(
    all_results:  Dict[str, Dict[float, Dict[str, float]]],
    oracle_key:   str,
    degraded_key: str,
    output_dir:   Path,
) -> None:
    lines = []
    lines.append("=" * 72)
    lines.append("OoD Post-Processing Evaluation Summary")
    lines.append("=" * 72)

    # Overall mean (all thetas)
    def mean_all(model_key: str, method: str) -> float:
        vals = [v[method] for v in all_results[model_key].values()
                if method in v and not np.isnan(v[method])]
        return float(np.mean(vals)) if vals else float("nan")

    lines.append("\n[Overall mean MAE across all θ values]\n")
    oracle_raw  = mean_all(oracle_key,   "raw")
    degraded_raw = mean_all(degraded_key, "raw")
    lines.append(f"  Oracle   ({oracle_key})  - raw:  {oracle_raw:.2f} m  (ceiling)")
    lines.append(f"  Degraded ({degraded_key}) - raw:  {degraded_raw:.2f} m  (baseline OoD)")
    lines.append(f"  OoD gap (baseline)  :  {degraded_raw - oracle_raw:.2f} m\n")

    for m in ALL_METHODS:
        if m == "raw":
            continue
        mae_m = mean_all(degraded_key, m)
        gap   = mae_m - oracle_raw
        pct   = 100.0 * (degraded_raw - mae_m) / max(degraded_raw - oracle_raw, 1e-6)
        lines.append(
            f"  {METHOD_LABELS[m]:<26}:  {mae_m:.2f} m   "
            f"(gap={gap:+.2f} m, {pct:.1f}% closed)"
        )

    lines.append("\n" + "-" * 72)
    lines.append("[Per-θ breakdown — degraded model only]\n")
    thetas = sorted(all_results[degraded_key].keys())
    col_w = 13
    header = f"{'θ':>5}" + "".join(f"  {METHOD_LABELS[m][:col_w]:>{col_w}}" for m in ALL_METHODS)
    lines.append(header)
    for theta in thetas:
        td = all_results[degraded_key][theta]
        row = f"{theta:>5.1f}"
        for m in ALL_METHODS:
            v = td.get(m, float("nan"))
            row += f"  {v:>{col_w}.2f}"
        lines.append(row)

    txt = "\n".join(lines)
    path = output_dir / "mae_summary.txt"
    path.write_text(txt)
    print(f"[✓] Saved {path}")
    print("\n" + txt)


# ---------------------------------------------------------------------------
# Heatmap  (methods x theta)
# ---------------------------------------------------------------------------

def make_heatmap(
    all_results: Dict[str, Dict[float, Dict[str, float]]],
    oracle_key:  str,
    degraded_key: str,
    output_dir:  Path,
) -> None:
    """
    Single figure: rows = [oracle raw] + [highvar x each method],
    columns = theta values.  Cell colour and annotation = mean MAE (m).
    """
    thetas  = sorted(all_results[degraded_key].keys())
    # row labels / data
    row_keys:   List[str]       = []
    row_labels: List[str]       = []
    grid_data:  List[List[float]] = []

    # Oracle row (raw only)
    row_keys.append("oracle_raw")
    row_labels.append(f"Oracle (lowvar)\nraw")
    grid_data.append([
        all_results[oracle_key].get(t, {}).get("raw", float("nan")) for t in thetas
    ])

    # Highvar rows (one per method)
    for m in ALL_METHODS:
        row_keys.append(f"hv_{m}")
        row_labels.append(f"HV – {METHOD_LABELS[m]}")
        grid_data.append([
            all_results[degraded_key].get(t, {}).get(m, float("nan")) for t in thetas
        ])

    mat = np.array(grid_data, dtype=np.float32)  # (n_rows, n_thetas)
    n_rows, n_cols = mat.shape

    fig_h = max(5, 0.55 * n_rows)
    fig, ax = plt.subplots(figsize=(max(6, 1.8 * n_cols), fig_h))

    vmin = float(np.nanmin(mat))
    vmax = float(np.nanmax(mat))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", vmin=vmin, vmax=vmax)

    # Cell text annotations
    for r in range(n_rows):
        for c in range(n_cols):
            val = mat[r, c]
            if np.isnan(val):
                txt = "–"
            else:
                txt = f"{val:.2f}"
            # choose black or white text for readability
            normalised = (val - vmin) / max(vmax - vmin, 1e-6)
            txt_color = "white" if normalised > 0.65 else "black"
            ax.text(c, r, txt, ha="center", va="center",
                    fontsize=8, color=txt_color, fontweight="bold")

    # Separate oracle row with a thick horizontal line
    ax.axhline(0.5, color="white", lw=2.5)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([f"θ={t:.1f}" for t in thetas], fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title("MAE (m) — Oracle vs HV model + post-processing  (lowvar test data)",
                 fontsize=10, pad=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("MAE (m)", fontsize=8)

    plt.tight_layout()
    path = output_dir / "mae_heatmap.png"
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"[✓] Saved {path}")


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------

def save_latency_csv(
    latencies_by_theta: Dict[float, Dict[str, float]],  # theta → method → mean_s
    output_dir: Path,
) -> None:
    import csv
    path = output_dir / "latency_per_method.csv"
    thetas = sorted(latencies_by_theta.keys())
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "label"] +
                   [f"theta_{t:.1f}_s" for t in thetas] +
                   ["mean_s", "mean_ms"])
        for m in ALL_METHODS:
            vals = [latencies_by_theta[t].get(m, float("nan")) for t in thetas]
            mean_s = float(np.nanmean(vals))
            w.writerow([m, METHOD_LABELS[m]] +
                       [f"{v:.5f}" for v in vals] +
                       [f"{mean_s:.5f}", f"{mean_s*1000:.2f}"])
    print(f"[✓] Saved {path}")


def make_latency_plot(
    latencies_by_theta: Dict[float, Dict[str, float]],  # theta → method → mean_s
    output_dir: Path,
) -> None:
    """
    Two-panel figure:
      Left : absolute mean latency per method (ms) — bar chart.
      Right: extra latency vs raw ANP forward pass (ms) — bar chart.
    For filter methods the extra latency is their overhead alone;
    for AR-based methods it replaces the forward pass entirely.
    """
    thetas = sorted(latencies_by_theta.keys())

    # Mean across thetas
    means: Dict[str, float] = {}
    for m in ALL_METHODS:
        vals = [latencies_by_theta[t].get(m, float("nan")) for t in thetas]
        means[m] = float(np.nanmean(vals))

    raw_s   = means["raw"]
    labels  = [METHOD_LABELS[m] for m in ALL_METHODS]
    abs_ms  = [means[m] * 1e3 for m in ALL_METHODS]

    # Extra latency: for filters it's their overhead; for AR it's total AR minus raw;
    # for AR+filter it's (AR+filter) minus raw
    extras_ms: List[float] = []
    for m in ALL_METHODS:
        if m == "raw":
            extras_ms.append(0.0)
        elif m.startswith("ar_"):
            extras_ms.append(max(0.0, (means[m] - raw_s) * 1e3))
        else:
            extras_ms.append(means[m] * 1e3)   # filter overhead already excludes raw

    colors = [METHOD_COLORS[m] for m in ALL_METHODS]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(ALL_METHODS))
    w = 0.65

    # --- Absolute ---
    bars = ax1.bar(x, abs_ms, width=w, color=colors, alpha=0.8, edgecolor="black", lw=0.5)
    ax1.axhline(raw_s * 1e3, color="black", ls="--", lw=1, label=f"Raw baseline ({raw_s*1e3:.1f} ms)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("Latency (ms) per trajectory")
    ax1.set_title("Absolute latency per method")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.35)
    for bar, v in zip(bars, abs_ms):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{v:.1f}", ha="center", va="bottom", fontsize=7)

    # --- Extra ---
    bars2 = ax2.bar(x, extras_ms, width=w, color=colors, alpha=0.8, edgecolor="black", lw=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("Extra latency vs raw ANP (ms)")
    ax2.set_title("Post-processing overhead (vs raw ANP forward pass)")
    ax2.grid(axis="y", alpha=0.35)
    for bar, v in zip(bars2, extras_ms):
        if v > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f"{v:.1f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    path = output_dir / "latency_plot.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[✓] Saved {path}")


# ---------------------------------------------------------------------------
# MAE vs theta line plot (best methods comparison)
# ---------------------------------------------------------------------------

def make_mae_vs_theta_plot(
    all_results:  Dict[str, Dict[float, Dict[str, float]]],
    oracle_key:   str,
    degraded_key: str,
    output_dir:   Path,
) -> None:
    """
    Line plot: MAE (m) vs θ for the four key series:
      - Oracle raw              (reference ceiling)
      - HV raw                  (OoD baseline)
      - HV RTS (R=σ²)           (best single-pass filter)
      - HV AR+RTS (R=σ²)        (best overall)
    Shaded gap between oracle and each HV series for visual clarity.
    """
    thetas = sorted(all_results[degraded_key].keys())
    x = np.array(thetas)

    series = {
        "oracle_raw":      ("Oracle (lowvar) \u2013 Raw",    "#2c3e50", "-",  "o"),
        "hv_raw":          ("HV \u2013 Raw (OoD baseline)",  "#e74c3c", "--", "s"),
        "hv_kalman_calib": ("HV \u2013 Kal CV (R=calib)",   "#16a085", "-.", "^"),
        "hv_rts_var":      ("HV \u2013 RTS (R=\u03c3\u00b2)",        "#27ae60", "-.", "D"),
        "hv_ar_uncert":    ("HV \u2013 AR (by \u03c3\u00b2)",         "#2980b9", ":",  "p"),
        "hv_ar_rts_var":   ("HV \u2013 AR+RTS (R=\u03c3\u00b2)",     "#6c3483", ":",  "*"),
    }

    def get_vals(model_key: str, method: str) -> np.ndarray:
        return np.array([
            all_results[model_key].get(t, {}).get(method, float("nan"))
            for t in thetas
        ])

    oracle_vals          = get_vals(oracle_key,   "raw")
    hv_raw_vals          = get_vals(degraded_key, "raw")
    hv_kalman_calib_vals = get_vals(degraded_key, "kalman_cv_calib")
    hv_rts_vals          = get_vals(degraded_key, "kalman_rts_var")
    hv_ar_uncert_vals    = get_vals(degraded_key, "ar_uncert")
    hv_ar_rts_vals       = get_vals(degraded_key, "ar_kalman_rts_var")

    data = {
        "oracle_raw":      oracle_vals,
        "hv_raw":          hv_raw_vals,
        "hv_kalman_calib": hv_kalman_calib_vals,
        "hv_rts_var":      hv_rts_vals,
        "hv_ar_uncert":    hv_ar_uncert_vals,
        "hv_ar_rts_var":   hv_ar_rts_vals,
    }

    fig, ax = plt.subplots(figsize=(10, 5))

    # Shaded gap: hv_raw → oracle (red fill), improvements narrow this
    ax.fill_between(x, oracle_vals, hv_raw_vals,
                    alpha=0.07, color="#e74c3c", label="_nolegend_")
    ax.fill_between(x, oracle_vals, hv_ar_rts_vals,
                    alpha=0.09, color="#6c3483", label="_nolegend_")

    for key, (label, color, ls, marker) in series.items():
        ax.plot(x, data[key], color=color, ls=ls, marker=marker,
                markersize=7, lw=2, label=label)

    # Annotate % gap closed for best AR method at each theta
    for i, theta in enumerate(thetas):
        gap_total = hv_raw_vals[i] - oracle_vals[i]
        best_ar_v = float(np.nanmin([hv_ar_uncert_vals[i], hv_ar_rts_vals[i]]))
        gap_closed = hv_raw_vals[i] - best_ar_v
        if gap_total > 0 and not np.isnan(gap_closed):
            pct = 100.0 * gap_closed / gap_total
            ax.annotate(
                f"{pct:.0f}%",
                xy=(theta, best_ar_v),
                xytext=(0, -14), textcoords="offset points",
                ha="center", fontsize=8, color="#6c3483",
                fontweight="bold",
            )

    ax.set_xlabel("\u03b8 (channel variability)", fontsize=11)
    ax.set_ylabel("MAE (m)", fontsize=11)
    ax.set_title(
        "MAE vs \u03b8 \u2014 Oracle / HV raw / Post-processing improvements\n"
        "(% = gap closed by best AR method vs oracle)",
        fontsize=10,
    )
    ax.set_xticks(thetas)
    ax.set_xticklabels([f"{t:.1f}" for t in thetas])
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.35)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = output_dir / "mae_vs_theta_best_methods.png"
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"[✓] Saved {path}")


# ---------------------------------------------------------------------------
# Box-plot
# ---------------------------------------------------------------------------

def make_boxplot(
    all_per_sample: Dict[str, Dict[float, Dict[str, List[float]]]],  # model_key → theta → method → [maes]
    oracle_key:     str,
    degraded_key:   str,
    output_dir:     Path,
) -> None:
    """
    One box per method (degraded model) + one box for oracle.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    # Collect all per-sample MAEs across thetas for each method
    def collect(model_key: str, method: str) -> List[float]:
        vals: List[float] = []
        for theta_dict in all_per_sample.get(model_key, {}).values():
            vals.extend(theta_dict.get(method, []))
        return vals

    oracle_vals = collect(oracle_key, "raw")

    # x positions
    labels  = ["Oracle\n(lowvar)"] + [METHOD_LABELS[m] for m in ALL_METHODS]
    data    = [oracle_vals]        + [collect(degraded_key, m) for m in ALL_METHODS]
    colors  = ["#2c3e50"]          + [METHOD_COLORS[m] for m in ALL_METHODS]

    bps = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
    for patch, color in zip(bps["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in bps["medians"]:
        median.set_color("black")

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("MAE (m)")
    ax.set_title("OoD Post-Processing: MAE Distribution (highvar model on lowvar data)")
    ax.grid(axis="y", alpha=0.35)

    path = output_dir / "mae_boxplot.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[✓] Saved {path}")


# ---------------------------------------------------------------------------
# Trajectory plots
# ---------------------------------------------------------------------------

# Which methods to show on trajectory plots (to keep readable)
_TRAJ_SHOW = ["raw", "kalman_rts_var", "ar_kalman_rts_var"]
_TRAJ_COLORS = {"raw": "#e74c3c", "kalman_rts_var": "#27ae60", "ar_kalman_rts_var": "#6c3483"}
_TRAJ_STYLES = {"raw": "--",      "kalman_rts_var": "-.",        "ar_kalman_rts_var": ":"}


def _plot_traj_ax(
    ax:          Axes,
    pack_oracle: Dict,
    pack_hv:     Dict,
    title:       str,
) -> None:
    y_gt       = pack_oracle["gt"]
    ctx_np     = pack_oracle["ctx_idx"]

    # Ground truth
    ax.plot(y_gt[:, 0], y_gt[:, 1], "k-", lw=1.5, label="Ground truth", zorder=5)
    ax.plot(y_gt[ctx_np, 0], y_gt[ctx_np, 1], "k-", lw=3, alpha=0.35, label="Context")
    ax.scatter([y_gt[0, 0]], [y_gt[0, 1]], c="green", s=50, marker="o", zorder=10)
    ax.scatter([y_gt[-1, 0]], [y_gt[-1, 1]], c="black", s=50, marker="s", zorder=10)

    # Oracle prediction (thin green)
    p_or = pack_oracle["preds"]["raw"]
    ax.plot(p_or[:, 0], p_or[:, 1], color="green", lw=1, ls="-", alpha=0.7,
            label=f"Oracle raw ({_mae_str(p_or, y_gt, pack_oracle['nc_mask'])})")

    # HV methods
    for m in _TRAJ_SHOW:
        p = pack_hv["preds"].get(m)
        if p is None:
            continue
        mae_s = _mae_str(p, y_gt, pack_hv["nc_mask"])
        ax.plot(p[:, 0], p[:, 1],
                color=_TRAJ_COLORS[m], ls=_TRAJ_STYLES[m], lw=1.2,
                label=f"HV {METHOD_LABELS[m]} ({mae_s})")

    ax.set_title(title, fontsize=9)
    ax.axis("equal")
    ax.tick_params(labelsize=7)


def _mae_str(pred: np.ndarray, gt: np.ndarray, nc_mask: np.ndarray) -> str:
    if nc_mask.sum() == 0:
        return "n/a"
    return f"{np.mean(np.abs(pred[nc_mask, :] - gt[nc_mask, :])):.1f} m"


def save_trajectory_plot(
    theta:         float,
    samples:       List,
    oracle_model:  torch.nn.Module,
    hv_model:      torch.nn.Module,
    oracle_y_mean: torch.Tensor,
    oracle_y_std:  torch.Tensor,
    hv_y_mean:     torch.Tensor,
    hv_y_std:      torch.Tensor,
    ctx_pct:       int,
    device:        torch.device,
    dt:            float,
    alpha:         float,
    beta:          float,
    sigma_a:       float,
    ar_block_k:    int,
    ar_var_thresh: float,
    n_traj:        int,
    pick_seed:     int,
    out_path:      Path,
) -> None:
    rng = np.random.default_rng(pick_seed + int(round(theta * 1000)))
    n_traj = min(n_traj, len(samples))
    idxs   = rng.choice(len(samples), size=n_traj, replace=False)

    fig, axs = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    axs_flat = axs.ravel()

    for k, si in enumerate(idxs):
        x_np, y_np = samples[int(si)]

        pack_or = predict_single(
            x_np, y_np, oracle_model, oracle_y_mean, oracle_y_std,
            ctx_pct, device, dt, alpha, beta, sigma_a, ar_block_k, ar_var_thresh,
        )
        pack_hv = predict_single(
            x_np, y_np, hv_model, hv_y_mean, hv_y_std,
            ctx_pct, device, dt, alpha, beta, sigma_a, ar_block_k, ar_var_thresh,
        )

        _plot_traj_ax(
            ax=axs_flat[k], pack_oracle=pack_or, pack_hv=pack_hv,
            title=f"Trajectory {k+1}",
        )
        axs_flat[k].legend(fontsize=6, loc="upper left")

    fig.suptitle(f"θ = {theta:.1f} — Oracle (green) vs HV model post-processing", fontsize=11)
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"[✓] Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate post-processing to reduce OoD MAE gap."
    )
    p.add_argument("--lowvar-ckpt",   type=Path, required=True,
                   help="Checkpoint of lowvar ANP (oracle).")
    p.add_argument("--highvar-ckpt",  type=Path, required=True,
                   help="Checkpoint of highvar ANP (OoD model to improve).")
    p.add_argument("--lowvar-data-dir",  type=Path, required=True,
                   help="Processed data directory for lowvar (test target).")
    p.add_argument("--highvar-data-dir", type=Path, required=True,
                   help="Processed data directory for highvar "
                        "(used only for normalization stats of the highvar model).")
    p.add_argument("--topology",  type=str, default="ellipsoidal",
                   choices=["ellipsoidal", "aligned", "random"])
    p.add_argument("--context",   type=int, default=30,
                   help="Context percentage [0-100].")
    p.add_argument("--dt",        type=float, default=1.0)
    p.add_argument("--alpha",     type=float, default=0.85)
    p.add_argument("--beta",      type=float, default=0.005)
    p.add_argument("--sigma-a",   type=float, default=1.0)
    p.add_argument("--ar-block-k",     type=int,   default=5)
    p.add_argument("--ar-var-thresh",  type=float, default=0.01)
    p.add_argument("--n-traj-plots",   type=int,   default=4)
    p.add_argument("--seed",      type=int, default=EVAL_SEED)
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/eval_ood_postprocess"))
    p.add_argument("--no-traj-plots", action="store_true",
                   help="Skip qualitative trajectory plots.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = args.output_dir / "trajectories"
    if not args.no_traj_plots:
        traj_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    print("[loading] oracle (lowvar) model …")
    oracle_model = load_anp(args.lowvar_ckpt, device)
    print("[loading] degraded (highvar) model …")
    hv_model = load_anp(args.highvar_ckpt, device)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("[data] loading lowvar test+train data …")
    lv_test, lv_train, lv_meta = load_data(args.lowvar_data_dir, args.topology)
    print("[data] loading highvar train data (for normalization stats) …")
    _, hv_train, _ = load_data(args.highvar_data_dir, args.topology)

    oracle_y_mean, oracle_y_std = compute_y_stats(lv_train, device)
    hv_y_mean,     hv_y_std     = compute_y_stats(hv_train, device)

    # Group by theta (lowvar test data for both models)
    lv_groups = group_by_theta(lv_test, lv_meta)
    print(f"[data] theta values in lowvar test: {sorted(lv_groups.keys())}")

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------
    # Structure: {model_key: {theta: {method: mean_mae}}}
    all_results:     Dict[str, Dict[float, Dict[str, float]]]         = {"oracle": {}, "highvar": {}}
    all_per_sample:  Dict[str, Dict[float, Dict[str, List[float]]]]   = {"oracle": {}, "highvar": {}}
    # latency only tracked for highvar model (deployment scenario)
    hv_latencies_by_theta: Dict[float, Dict[str, float]] = {}  # theta → method → mean_s

    kw = dict(
        ctx_pct=args.context, dt=args.dt,
        alpha=args.alpha, beta=args.beta, sigma_a=args.sigma_a,
        ar_block_k=args.ar_block_k, ar_var_thresh=args.ar_var_thresh,
        device=device,
    )

    for theta in sorted(lv_groups.keys()):
        samples = lv_groups[theta]
        print(f"\n[eval] θ={theta:.1f}  n={len(samples)} trajectories")

        # Oracle
        maes_or, lists_or, _ = eval_theta_group(
            samples=samples, model=oracle_model,
            y_mean=oracle_y_mean, y_std=oracle_y_std,
            return_per_sample=True, **kw,
        )
        all_results["oracle"][theta]    = maes_or
        all_per_sample["oracle"][theta] = lists_or or {}

        # Highvar (OoD model)
        maes_hv, lists_hv, lats_hv = eval_theta_group(
            samples=samples, model=hv_model,
            y_mean=hv_y_mean, y_std=hv_y_std,
            return_per_sample=True, **kw,
        )
        all_results["highvar"][theta]    = maes_hv
        all_per_sample["highvar"][theta] = lists_hv or {}
        hv_latencies_by_theta[theta]     = lats_hv

        # Quick per-theta print
        oracle_raw = maes_or["raw"]
        hv_raw     = maes_hv["raw"]
        best_m     = min(ALL_METHODS, key=lambda m: maes_hv.get(m, 999))
        best_mae   = maes_hv[best_m]
        print(
            f"  oracle={oracle_raw:.2f} m  |  hv_raw={hv_raw:.2f} m  |"
            f"  best_pp={METHOD_LABELS[best_m]} → {best_mae:.2f} m"
        )

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    print("\n[output] building summary …")
    build_summary_csv(all_results, args.output_dir)
    build_summary_txt(all_results, "oracle", "highvar", args.output_dir)
    make_boxplot(all_per_sample, "oracle", "highvar", args.output_dir)
    make_heatmap(all_results, "oracle", "highvar", args.output_dir)
    make_mae_vs_theta_plot(all_results, "oracle", "highvar", args.output_dir)
    save_latency_csv(hv_latencies_by_theta, args.output_dir)
    make_latency_plot(hv_latencies_by_theta, args.output_dir)

    if not args.no_traj_plots:
        print("\n[output] generating trajectory plots …")
        for theta in sorted(lv_groups.keys()):
            save_trajectory_plot(
                theta=theta,
                samples=lv_groups[theta],
                oracle_model=oracle_model,
                hv_model=hv_model,
                oracle_y_mean=oracle_y_mean,
                oracle_y_std=oracle_y_std,
                hv_y_mean=hv_y_mean,
                hv_y_std=hv_y_std,
                ctx_pct=args.context,
                device=device,
                dt=args.dt,
                alpha=args.alpha,
                beta=args.beta,
                sigma_a=args.sigma_a,
                ar_block_k=args.ar_block_k,
                ar_var_thresh=args.ar_var_thresh,
                n_traj=args.n_traj_plots,
                pick_seed=args.seed,
                out_path=traj_dir / f"theta_{theta:.1f}.png",
            )

    print("\n[done] all outputs written to", args.output_dir)


if __name__ == "__main__":
    main()
