"""
Evalúa postprocesado de trayectorias ANP con filtros.

- Carga test_data.pkl + metadata.pkl y agrupa por theta.
- Carga train_data.pkl para y_mean/y_std.
- Carga ANP checkpoint: ANP_<topology>/best_checkpoint.pth.tar
- Predice mean/var (normalizado) -> denormaliza a metros.
- Construye una "secuencia de medida" z_t para el filtro:
    - por defecto: en puntos de contexto usa GT (observación)
    - en el resto usa la media del ANP
  y define R_t:
    - contexto: ruido muy pequeño
    - no-contexto: varianza del ANP (en metros^2)
- Calcula MAE (solo NO-contexto) antes y después del filtro.


Uso:

    - With alpha-beta filter:
    python eval_anp_postprocess_filters.py \
      --data-dir /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_low_variance \
      --anp-dir  /home/fernando/tesis/underwater-localization-topologies/src/training/results/ANP_topologies/low_variance \
      --topology random \
      --context 40 \
      --filter alpha_beta \
      --ab-alpha 0.85 --ab-beta 0.005

    With alpha-beta adaptative filter:
    python eval_anp_postprocess_filters.py \
      --data-dir /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_low_variance \
      --anp-dir  /home/fernando/tesis/underwater-localization-topologies/src/training/results/ANP_topologies/low_variance \
      --topology random \
      --context 40 \
      --filter alpha_beta \
      --ab-alpha 0.85 --ab-beta 0.005 \
      --ab-adaptive

    - With Kalman CV filter:
    python eval_anp_postprocess_filters.py \
      --data-dir /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_low_variance \
      --anp-dir  /home/fernando/tesis/underwater-localization-topologies/src/training/results/ANP_topologies/low_variance \
      --topology random \
      --context 40 \
      --filter kalman_cv \
      --kf-sigma-a 1.0

    - With Kalman RTS smoother:      
    python eval_anp_postprocess_filters.py \
      --data-dir /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_low_variance \
      --anp-dir  /home/fernando/tesis/underwater-localization-topologies/src/training/results/ANP_topologies/low_variance \
      --topology random \
      --context 40 \
      --filter kalman_rts \
      --kf-sigma-a 1.0
"""

import argparse
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Importa tu modelo ANP (ajusta si tu repo usa otra ruta)
from src.models.anp import LatentModel, DistributedLatentModel
from src.utils.nav_dataset import NavigationTrajectoryDataset

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_topology_test_grouped(data_dir: Path, topology: str) -> Tuple[Dict[float, List], List[float]]:
    """Como TopologyEvaluator.load_topology_data(): agrupa test_data por theta usando metadata['test_thetas']."""
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
        theta_groups.setdefault(theta, []).append(sample)

    theta_values = sorted(theta_groups.keys())
    return theta_groups, theta_values

def get_y_stats_from_train(data_dir: Path, topology: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Como TopologyEvaluator.get_y_stats(): calcula stats en train_data.pkl para normalización."""
    topology_dir = Path(data_dir) / f"topology_{topology}"
    train_path = topology_dir / "train_data.pkl"
    if not train_path.exists():
        raise FileNotFoundError(f"train_data.pkl no encontrado: {train_path}")

    with open(train_path, "rb") as f:
        train_data = pickle.load(f)

    Y = np.concatenate([y for _, y in train_data], axis=0)  # (N*T, 3)
    y_mean = torch.tensor(Y.mean(axis=0), dtype=torch.float32, device=device)
    y_std  = torch.tensor(Y.std(axis=0) + 1e-6, dtype=torch.float32, device=device)
    return y_mean, y_std

def normalize_y(y: torch.Tensor, y_mean: torch.Tensor, y_std: torch.Tensor) -> torch.Tensor:
    return (y - y_mean.view(1, 1, -1)) / y_std.view(1, 1, -1)

def denormalize_y(y_norm: torch.Tensor, y_mean: torch.Tensor, y_std: torch.Tensor) -> torch.Tensor:
    return y_norm * y_std.view(1, 1, -1) + y_mean.view(1, 1, -1)

def sample_context_indices(total_points: int, n_context: int, g: torch.Generator, device: torch.device) -> torch.Tensor:
    """Subconjunto aleatorio determinista y ordenado (como tu evaluador)."""
    perm = torch.randperm(total_points, generator=g, device=device)
    return perm[:n_context].sort().values

def load_anp_model(anp_dir: Path, topology: str, input_dim: int, output_dim: int, device: torch.device, distributed: bool = False) -> torch.nn.Module:
    """Como TopologyEvaluator.load_anp_model(): carga best_checkpoint.pth.tar."""
    ckpt_path = Path(anp_dir) / f"ANP_{topology}" / "best_checkpoint.pth.tar"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ANP checkpoint not found: {ckpt_path}")

    sensor_emb_dim = 64
    n_sensors = 10
    sensor_feature_dim = 401

    if distributed:
        base = LatentModel(num_hidden=128, input_dim=sensor_emb_dim, output_dim=output_dim)
        model = DistributedLatentModel(
            base_anp=base,
            n_sensors=n_sensors,
            in_dim_per_sensor=sensor_feature_dim,
            emb_dim=sensor_emb_dim,
            fusion="mean",
        )
    else:
        model = LatentModel(num_hidden=128, input_dim=input_dim, output_dim=output_dim)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()

# ---------------------------
# Postprocesado
# ---------------------------

def alpha_beta_filter_2d(
    z_xy: np.ndarray,
    dt: float,
    alpha: float = 0.85,
    beta: float = 0.005,
    x0: Optional[np.ndarray] = None,
    v0: Optional[np.ndarray] = None,
    adaptive_from_R: bool = False,
    R_xy: Optional[np.ndarray] = None,
    min_w: float = 0.05,
) -> np.ndarray:
    """
    Alpha-Beta filter (g-h) en 2D para posición (x,y) con modelo de velocidad constante.

    Ecuaciones (por eje):
      x_pred = x + v*dt
      v_pred = v
      r      = z - x_pred
      x      = x_pred + alpha*r
      v      = v_pred + (beta/dt)*r

    Si adaptive_from_R=True y R_xy está disponible (T,2,2),
    se escala alpha/beta por un peso w_t en función de la fiabilidad de la medida:
      w_t = 1 / (1 + tr(R_t)/tr_ref)
    (medidas con mayor varianza => menor w_t => suaviza más).
    """
    assert z_xy.ndim == 2 and z_xy.shape[1] == 2, "z_xy debe ser (T,2)"
    T = z_xy.shape[0]
    if T == 0:
        return z_xy

    if dt <= 0:
        raise ValueError("dt debe ser > 0")

    # Inicialización
    if x0 is None:
        x = z_xy[0].astype(np.float64).copy()
    else:
        x = np.asarray(x0, dtype=np.float64).copy()

    if v0 is None:
        if T >= 2:
            v = ((z_xy[1] - z_xy[0]) / dt).astype(np.float64)
        else:
            v = np.zeros(2, dtype=np.float64)
    else:
        v = np.asarray(v0, dtype=np.float64).copy()

    x_filt = np.zeros((T, 2), dtype=np.float64)
    x_filt[0] = x

    # Referencia para adaptación con R
    tr_ref = None
    if adaptive_from_R and R_xy is not None:
        # tr(R_t) típico (evita extremos)
        traces = np.clip(np.trace(R_xy, axis1=1, axis2=2), 1e-12, np.inf)
        tr_ref = float(np.median(traces))

    for t in range(1, T):
        # Predicción
        x_pred = x + v * dt
        v_pred = v

        # Residual (innovación)
        r = z_xy[t].astype(np.float64) - x_pred

        # (Opcional) escalado de ganancias usando R_t
        a_t = alpha
        b_t = beta
        if adaptive_from_R and R_xy is not None and tr_ref is not None and tr_ref > 0:
            tr = float(np.trace(R_xy[t]))
            w = 1.0 / (1.0 + (tr / tr_ref))
            w = float(np.clip(w, min_w, 1.0))
            a_t = alpha * w
            b_t = beta * w

        # Estabilidad típica: 0<alpha<1 y 0<beta<=2 (práctica común) :contentReference[oaicite:1]{index=1}
        a_t = float(np.clip(a_t, 1e-6, 0.999999))
        b_t = float(np.clip(b_t, 1e-6, 2.0))

        # Corrección
        x = x_pred + a_t * r
        v = v_pred + (b_t / dt) * r

        x_filt[t] = x

    return x_filt.astype(np.float32)

import numpy as np
from typing import Optional, Tuple

def _cv_matrices_2d(dt: float, sigma_a: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Constant Velocity (CV) 2D:
      state x = [px, py, vx, vy]^T
      z = [px, py]^T

    Process noise: white acceleration with intensity sigma_a^2.
    Q block (per axis) = sigma_a^2 * [[dt^4/4, dt^3/2],
                                     [dt^3/2, dt^2]]
    (Extendido a 2D con bloques independientes).
    """
    if dt <= 0:
        raise ValueError("dt debe ser > 0")
    sa2 = float(sigma_a) ** 2

    F = np.array([
        [1.0, 0.0, dt,  0.0],
        [0.0, 1.0, 0.0, dt ],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)

    H = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ], dtype=np.float64)

    q11 = (dt**4) / 4.0
    q12 = (dt**3) / 2.0
    q22 = (dt**2)

    Q = sa2 * np.array([
        [q11, 0.0, q12, 0.0],
        [0.0, q11, 0.0, q12],
        [q12, 0.0, q22, 0.0],
        [0.0, q12, 0.0, q22],
    ], dtype=np.float64)

    return F, H, Q


def kalman_filter_cv_2d(
    z_xy: np.ndarray,
    R_xy: Optional[np.ndarray],
    dt: float,
    sigma_a: float = 1.0,
    P0_pos_scale: float = 10.0,
    P0_vel_scale: float = 100.0,
    r_eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Filtrado Kalman CV 2D con R_t variable.
    Devuelve:
      x_filt (T,4), P_filt (T,4,4), x_pred (T,4), P_pred (T,4,4)
    """
    assert z_xy.ndim == 2 and z_xy.shape[1] == 2, "z_xy debe ser (T,2)"
    T = z_xy.shape[0]
    if T == 0:
        return (np.zeros((0,4)), np.zeros((0,4,4)), np.zeros((0,4)), np.zeros((0,4,4)))

    F, H, Q = _cv_matrices_2d(dt, sigma_a)
    I = np.eye(4, dtype=np.float64)

    # Inicialización: posición = primera medida; velocidad ~ diferencia inicial
    x0 = np.zeros(4, dtype=np.float64)
    x0[0:2] = z_xy[0].astype(np.float64)
    if T >= 2:
        x0[2:4] = ((z_xy[1] - z_xy[0]) / dt).astype(np.float64)
    else:
        x0[2:4] = 0.0

    # P0: usa R0 si existe; si no, escalas razonables
    if R_xy is not None:
        R0 = R_xy[0].astype(np.float64)
        p0x = float(R0[0,0]) * P0_pos_scale
        p0y = float(R0[1,1]) * P0_pos_scale
    else:
        p0x = p0y = 1.0 * P0_pos_scale

    P = np.diag([p0x, p0y, P0_vel_scale, P0_vel_scale]).astype(np.float64)

    x_filt = np.zeros((T, 4), dtype=np.float64)
    P_filt = np.zeros((T, 4, 4), dtype=np.float64)
    x_pred = np.zeros((T, 4), dtype=np.float64)
    P_pred = np.zeros((T, 4, 4), dtype=np.float64)

    x = x0

    for t in range(T):
        # Predict
        if t == 0:
            xp = x
            Pp = P
        else:
            xp = F @ x
            Pp = F @ P @ F.T + Q

        x_pred[t] = xp
        P_pred[t] = Pp

        # Measurement noise
        if R_xy is None:
            R = np.eye(2, dtype=np.float64)
        else:
            R = R_xy[t].astype(np.float64)

        # Asegura SPD
        R = R + r_eps * np.eye(2, dtype=np.float64)

        # Update
        z = z_xy[t].astype(np.float64)
        y = z - (H @ xp)                                # innovación
        S = H @ Pp @ H.T + R
        # K = Pp H^T S^-1
        K = (Pp @ H.T) @ np.linalg.inv(S)

        x = xp + K @ y

        # Joseph form para estabilidad numérica:
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
    """
    Rauch-Tung-Striebel (RTS) fixed-interval smoother:
    backward pass sobre el filtro forward.
    """
    T = x_filt.shape[0]
    if T == 0:
        return x_filt, P_filt

    F, _, _ = _cv_matrices_2d(dt, sigma_a)

    x_smooth = x_filt.copy()
    P_smooth = P_filt.copy()

    for k in range(T - 2, -1, -1):
        # Ck = P_filt[k] F^T (P_pred[k+1])^-1
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
    use_rts: bool = False,
) -> np.ndarray:
    """
    z_xy: (T,2) medidas (posiciones) para filtrar
    R_xy: (T,2,2) cov de medida por tiempo (o None)
    returns: x_filt_xy (T,2)
    """
    method = method.lower()
    if method == "none":
        return z_xy

    # ---- stubs ----
    if method == "alpha_beta":
        return alpha_beta_filter_2d(z_xy=z_xy,
            dt=dt,alpha=alpha,beta=beta,adaptive_from_R=(R_xy is not None),R_xy=R_xy,
        )
    if method == "kalman_cv":
        x_filt, P_filt, x_pred, P_pred = kalman_filter_cv_2d(
            z_xy=z_xy, R_xy=R_xy, dt=dt, sigma_a=sigma_a
        )
        return x_filt[:, :2].astype(np.float32)
    
    if method == "kalman_rts":
        x_filt, P_filt, x_pred, P_pred = kalman_filter_cv_2d(
            z_xy=z_xy, R_xy=R_xy, dt=dt, sigma_a=sigma_a
        )
        x_sm, P_sm = rts_smoother(
            x_filt=x_filt, P_filt=P_filt, x_pred=x_pred, P_pred=P_pred, dt=dt, sigma_a=sigma_a
        )
        return x_sm[:, :2].astype(np.float32)

    raise ValueError(f"Unknown filter method: {method}")


# ---------------------------
# Evaluación: ANP -> medidas z_t -> filtro -> MAE
# ---------------------------

@torch.no_grad()
def eval_one_theta_group(
    samples: List,
    anp_model: torch.nn.Module,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    device: torch.device,
    topology: str,
    context_percent: int,
    eval_seed: int,
    filter_method: str,
    dt: float,
    use_gt_context_in_filter: bool,
    context_R_eps: float,
    alpha: float,
    beta: float,
    adaptive: bool,
    sigma_a: float,
) -> Tuple[float, float]:
    """
    Devuelve (mae_raw, mae_filtered) promediado en el grupo.
    MAE calculado SOLO en no-contexto (misma lógica de tu evaluador).
    """
    ds = NavigationTrajectoryDataset(samples)
    loader = DataLoader(ds, batch_size=8, shuffle=False)

    g = torch.Generator(device=device)
    g.manual_seed(eval_seed)

    maes_raw = []
    maes_filt = []

    for x, y in loader:
        x = x.to(device)  # (B,T,D)
        y = y.to(device)  # (B,T,3)
        B, T, _ = x.shape

        n_context = int((context_percent / 100) * T)
        n_context = max(1, min(n_context, T - 1))
        ctx_idx = sample_context_indices(T, n_context, g, device=device)

        non_ctx_mask = torch.ones(T, dtype=torch.bool, device=device)
        non_ctx_mask[ctx_idx] = False

        # ANP: context_y en espacio normalizado
        y_norm = normalize_y(y, y_mean, y_std)
        cx = x[:, ctx_idx, :]
        cy = y_norm[:, ctx_idx, :]

        # Predicción ANP en todos los targets
        pred_mean_norm, pred_var_norm, *_ = anp_model(cx, cy, x)  # (B,T,3) y var

        # Denormaliza mean a metros
        pred_mean = denormalize_y(pred_mean_norm, y_mean, y_std)  # (B,T,3)

        # MAE raw (solo no-contexto)
        mae_raw_b = F.l1_loss(pred_mean[:, non_ctx_mask, :], y[:, non_ctx_mask, :],
                              reduction="none").mean(dim=[1, 2])
        maes_raw.extend(mae_raw_b.detach().cpu().numpy().tolist())

        # ---- Prepara medidas z_t y R_t para filtro (por batch, una a una) ----
        # std_real = sqrt(var_norm) * y_std  (como tu plot_axiswise_ci)
        pred_std_real = torch.sqrt(pred_var_norm) * y_std.view(1, 1, -1)  # (B,T,3)
        pred_var_real = pred_std_real ** 2  # (B,T,3) en metros^2

        for b in range(B):
            y_true_np = y[b].detach().cpu().numpy()          # (T,3)
            y_pred_np = pred_mean[b].detach().cpu().numpy()  # (T,3)
            v_pred_np = pred_var_real[b].detach().cpu().numpy()  # (T,3)

            # Nos centramos en XY
            z_xy = y_pred_np[:, :2].copy()

            # Si queremos, “clamp” del contexto a GT (observación disponible)
            if use_gt_context_in_filter:
                z_xy[ctx_idx.detach().cpu().numpy(), :] = y_true_np[ctx_idx.detach().cpu().numpy(), :2]

            # Construye R_t (T,2,2): contexto pequeño; no-contexto desde var ANP
            R_xy = np.zeros((T, 2, 2), dtype=np.float32)
            for t in range(T):
                if t in set(ctx_idx.detach().cpu().numpy().tolist()):
                    R_xy[t] = np.diag([context_R_eps, context_R_eps])
                else:
                    R_xy[t] = np.diag([float(v_pred_np[t, 0]), float(v_pred_np[t, 1])])

            # Filtra
            x_filt_xy = postprocess_filter(
                z_xy=z_xy,
                R_xy=(R_xy if adaptive else None),
                method=filter_method,
                dt=dt,
                alpha=alpha,
                beta=beta,
                sigma_a=sigma_a,
            )

            # Reconstruye (T,3) para comparar MAE como en tu código (incluye 3 dims)
            y_filt = y_pred_np.copy()
            y_filt[:, :2] = x_filt_xy

            # MAE filtered (solo no-contexto)
            mask_np = non_ctx_mask.detach().cpu().numpy()
            mae_f = np.mean(np.abs(y_filt[mask_np, :] - y_true_np[mask_np, :]))
            maes_filt.append(float(mae_f))

    return float(np.mean(maes_raw)), float(np.mean(maes_filt))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--anp-dir", type=str, required=True)
    parser.add_argument("--topology", type=str, required=True, choices=["ellipsoidal", "random", "aligned"])
    parser.add_argument("--context", type=int, default=40)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--filter", type=str, default="none", choices=["none", "alpha_beta", "kalman_cv", "kalman_rts"])
    parser.add_argument("--dt", type=float, default=1.0, help="dt del filtro (si no lo sabes, deja 1.0 y tunearemos Q)")
    parser.add_argument("--use-gt-context", action="store_true", help="Usa GT en puntos de contexto como medidas del filtro")
    parser.add_argument("--context-R-eps", type=float, default=1e-4, help="Ruido de medida para puntos de contexto (si use-gt-context)")
    parser.add_argument("--distributed", action="store_true", help="Cargar ANP en modo DistributedLatentModel")
    parser.add_argument("--max-per-theta", type=int, default=-1, help="Limita nº de trayectorias por theta (debug)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--ab-alpha", type=float, default=0.85)
    parser.add_argument("--ab-beta", type=float, default=0.005)
    parser.add_argument("--ab-adaptive", action="store_true", help="Escala alpha/beta con R_t (varianza ANP) para suavizar más cuando ANP es inseguro")
    parser.add_argument("--kf-sigma-a", type=float, default=1.0, help="Std de aceleración (m/s^2 en unidades de dt) para Q")

    args = parser.parse_args()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        ("cuda" if args.device == "cuda" else "cpu")
    )
    print(f"Device: {device}")
    set_all_seeds(18)

    # Carga test agrupado
    theta_groups, theta_values = load_topology_test_grouped(Path(args.data_dir), args.topology)
    print(f"Theta values: {theta_values}")

    # Dimensiones del primer sample
    first = next(iter(theta_groups.values()))[0]
    input_dim = first[0].shape[-1]
    output_dim = first[1].shape[-1]
    print(f"Input dim={input_dim}, Output dim={output_dim}")

    # y_mean/y_std desde train
    y_mean, y_std = get_y_stats_from_train(Path(args.data_dir), args.topology, device=device)
    print(f"y_mean={y_mean.detach().cpu().numpy()}, y_std={y_std.detach().cpu().numpy()}")

    # ANP
    anp = load_anp_model(Path(args.anp_dir), args.topology, input_dim, output_dim, device=device, distributed=args.distributed)
    print("Loaded ANP.")

    # Eval por theta + global
    rows = []
    global_raw = []
    global_filt = []

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
            topology=args.topology,
            context_percent=args.context,
            eval_seed=args.eval_seed,
            filter_method=args.filter,
            dt=args.dt,
            use_gt_context_in_filter=args.use_gt_context,
            context_R_eps=args.context_R_eps,
            alpha = args.ab_alpha,
            beta = args.ab_beta,
            adaptive = args.ab_adaptive,
            sigma_a = args.kf_sigma_a,
        )
        rows.append((theta, mae_raw, mae_f))
        global_raw.append(mae_raw)
        global_filt.append(mae_f)

        print(f"[θ={theta:.1f}] MAE raw={mae_raw:.4f} | MAE filt={mae_f:.4f}")


    print("\n=== Summary ===")
    print(f"Topology={args.topology} | context={args.context}% | filter={args.filter} | dt={args.dt}")
    print(f"Global mean raw : {float(np.mean(global_raw)):.4f}")
    print(f"Global mean filt: {float(np.mean(global_filt)):.4f}")


if __name__ == "__main__":
    main()