# ------------------------------------------------------------
#  plot_anp_uncertainty.py
# ------------------------------------------------------------
"""
Draws ANP mean ± 1 σ against ground-truth (x,y) for N random trajectories.

Usage
-----
python plot_anp_uncertainty.py \
  --anp-ckpt /home/fernando/tesis/ANP/training/anp_runs/no_sensor_mask/anp_best_ep9933.pt \
  --drs-ckpt /home/fernando/tesis/ANP/training/drs_runs/drs_general_best_ep7958.pt \
  --csv /home/fernando/tesis/ANP/data/tasks/3x3x3/20m-64T_8mx4m/dataset_without_sensor_1_3.csv \
  --device cuda \
  --n-trajs 10 \
  --ctx-frac 0.3 \
  --ctx-mode random

  
python plot_anp_uncertainty.py \
  --anp-ckpt /home/fernando/tesis/ANP/training/anp_runs/no_sensor_mask/extrapolation/anp_best_ep5324.pt \
  --drs-ckpt /home/fernando/tesis/ANP/training/drs_runs/drs_general_best_ep7958.pt \
  --csv /home/fernando/tesis/ANP/data/tasks/3x3x3/20m-64T_8mx4m/dataset_without_sensor_1_3.csv \
  --device cuda \
  --n-trajs 10 \
  --ctx-frac 0.3 \
  --ctx-mode first

Outputs
-------
PNG files `traj_<k>.png` saved under a sub-folder whose name encodes
the dataset stem plus the sensor count, e.g.:
  20m-64T_8mx4m-4sensors/traj_03.png
"""

from __future__ import annotations
import argparse, random, re
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

ROOT = Path(__file__).resolve().parent.parent.parent
import sys; sys.path.append(str(ROOT))

from utils.data_loading import SENSOR_COLS, Y_COLS, parse_meta
from utils.load_anp_from_ckpt import load_anp

# ----------------------------------------------------------------------
def infer_sensor_count(csv_name: str) -> int:
    """Return 4, 5, 6 based on file name pattern."""
    if "without_sensor_1_3_5" in csv_name:     return 3
    if "without_sensor_1_3" in csv_name:     return 4
    if "without_sensor_1"   in csv_name:     return 5
    return 6

# ----------------------------------------------------------------------
def load_trajectories(csv_path: Path, *, use_meta=False):
    df = pd.read_csv(csv_path)
    for c in SENSOR_COLS:
        if c not in df.columns:
            df[c] = 0.0
    depth, length, width = parse_meta(csv_path.parent.name)
    meta = torch.tensor([depth, length, width], dtype=torch.float32)

    trajs = []
    for _, g in df.groupby("traj_id"):
        sensors = torch.tensor(g[SENSOR_COLS].values, dtype=torch.float32)
        coords  = torch.tensor(g[Y_COLS].values, dtype=torch.float32)
        if use_meta:
            sensors = torch.cat([sensors, meta.repeat(sensors.size(0), 1)], -1)
        trajs.append((sensors, coords))
    return trajs

# ----------------------------------------------------------------------
@torch.inference_mode()
def plot_traj(anp,
              sensors: torch.Tensor,
              coords:  torch.Tensor,
              out_file: Path,
              device: str = "cpu",
              ctx_frac: float = 0.3,
              ctx_mode: str  = "random",
              teacher: bool  = False,
              drs=None):
    """
    Predict a whole trajectory with a *subset* of context points.

    Parameters
    ----------
    anp       : loaded ANP model.
    sensors   : (L, x_dim) full sensor sequence ─ on CPU.
    coords    : (L, 2)     full GT positions   ─ on CPU.
    out_file  : where to save the PNG.
    device    : "cpu" | "cuda".
    ctx_frac  : fraction of points used as context (0‥1].
    ctx_mode  : "first" - first n_ctx points<br>
               "random" - random unique indices.
    teacher   : if True, replace ANP μ at context points with GT.
    """
    L = sensors.size(0)
    n_ctx = max(1, int(ctx_frac * L))

    if ctx_mode == "first":
        ctx_idx = torch.arange(n_ctx)
    elif ctx_mode == "random":
        ctx_idx = torch.randperm(L)[:n_ctx].sort().values
    else:
        raise ValueError("ctx_mode must be 'first' or 'random'")

    # -------------------------------- context / target split ----------
    x_c = sensors[ctx_idx]              # (n_ctx, x_dim)
    y_c = coords[ctx_idx]               # (n_ctx, 2)
    x_t = sensors                       # predict every point

    # Batch dimension (B = 1) and send to device
    x_c = x_c.unsqueeze(0).to(device)
    y_c = y_c.unsqueeze(0).to(device)
    x_t = x_t.unsqueeze(0).to(device)

    ctx_mask = torch.zeros_like(x_c[..., 0], dtype=torch.bool)  # (1, n_ctx)
    mu, sigma = anp.predict(x_c, y_c, x_t, ctx_mask=ctx_mask, n_samples=1)
    mu, sigma = mu.squeeze(0).cpu(), sigma.squeeze(0).cpu()

    sigma = 3 * sigma  # scale σ to match the plot style
    
    if teacher:
        mu[ctx_idx]    = coords[ctx_idx]          # ground-truth mean
        sigma[ctx_idx] = 1e-4                     # virtually zero std

    # ---------- DRS_general prediction (if provided) ----------
    if drs is not None:
        with torch.inference_mode():
            # (L, x_dim) → (L, 2)
            drs_pred = drs(sensors.to(device)).cpu()
    else:
        drs_pred = None

    ctx_idx_set = set(ctx_idx.tolist())
    target_idx = torch.tensor([i for i in range(L) if i not in ctx_idx_set])
    coords_cpu = coords            # already on CPU
    t = range(L)

    # --------------------------- figure -------------------------------
    fig, (ax_x, ax_y) = plt.subplots(1, 2, figsize=(10, 5), sharex=True)
    # --- X ------------------------------------------------------------
    ax_x.plot(t, coords_cpu[:, 0], label="GT", lw=1.2)
    ax_x.scatter(ctx_idx, coords_cpu[ctx_idx, 0], color="red",
                 zorder=5, s=18, label="Context")
    #ax_x.plot(t, mu[:, 0], label="ANP μ", lw=1.2)
    ax_x.plot(target_idx, mu[target_idx, 0], label="ANP μ", lw=1, color="tab:red")
    if drs_pred is not None:
        ax_x.plot(t, drs_pred[:, 0], lw=1,
                  label="DRS μ", color="tab:green")
    #ax_x.fill_between(t, mu[:, 0] - sigma[:, 0], mu[:, 0] + sigma[:, 0],
    #                  color="tab:blue", alpha=0.25, label="±1 σ")
    ax_x.fill_between(
        target_idx, 
        mu[target_idx, 0] - sigma[target_idx, 0], 
        mu[target_idx, 0] + sigma[target_idx, 0],
        color="tab:red", alpha=0.25, label="±3 σ"
    )
    ax_x.set_title("X coordinate", fontsize = 20); ax_x.set_xlabel("sample", fontsize = 18); ax_x.set_ylabel("m", fontsize = 18)
    ax_x.legend(frameon=False, fontsize=14)

    # --- Y ------------------------------------------------------------
    ax_y.plot(t, coords_cpu[:, 1], lw=1.2)
    ax_y.scatter(ctx_idx, coords_cpu[ctx_idx, 1], color="red", zorder=5, s=18)
    #ax_y.plot(t, mu[:, 1], lw=1.2)
    ax_y.plot(target_idx, mu[target_idx, 1], lw=1, color="tab:red")
    if drs_pred is not None:
        ax_y.plot(t, drs_pred[:, 1], lw=1,
                  color="tab:green")
    #ax_y.fill_between(t, mu[:, 1] - sigma[:, 1], mu[:, 1] + sigma[:, 1],
    #                  color="tab:orange", alpha=0.25, label="±1 σ")
    ax_y.fill_between(
        target_idx,
        mu[target_idx, 1] - sigma[target_idx, 1],
        mu[target_idx, 1] + sigma[target_idx, 1],
        color="tab:red", alpha=0.25, label="±3 σ"
    )
    ax_y.set_title("Y coordinate", fontsize = 20); ax_y.set_xlabel("sample", fontsize = 18)

    # Cambia el tamaño de los ticks de los ejes
    ax_x.tick_params(axis='both', labelsize=16)
    ax_y.tick_params(axis='both', labelsize=16)

    fig.tight_layout()
    plt.savefig(out_file, dpi=600)
    plt.close(fig)

# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Plot ANP μ±σ on random trajectories")
    ap.add_argument("--seed", type=int, default=18, help="RNG seed for trajectory & context selection.")
    ap.add_argument("--anp-ckpt", required=True)
    ap.add_argument("--drs-ckpt", type=Path, help="Path to DRS_general checkpoint (.pt) to plot alongside ANP.")
    ap.add_argument("--csv",      required=True)
    ap.add_argument("--n-trajs",  type=int, default=10)
    ap.add_argument("--device",   choices=["cpu","cuda"], default="cuda")
    ap.add_argument("--use-meta", action="store_true")
    ap.add_argument("--ctx-frac", type=float, default=0.2, help="Fraction of points used as context (0-1].")
    ap.add_argument("--ctx-mode", choices=["first", "random"], default="random", help="How to pick the context indices.")
    ap.add_argument("--teacher", action="store_true", help="Teacher-force: replace μ at context points with GT.")
    args = ap.parse_args()

    device = args.device
    csv_path = Path(args.csv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- output dir --------------------------------------
    sensor_cnt = infer_sensor_count(csv_path.name)
    stem = csv_path.parent.name  # e.g. 7-5m-64T_2mx1m
    out_dir = Path(__file__).resolve().parent / f"{stem}-{sensor_cnt}sensors"
    out_dir.mkdir(exist_ok=True)

    # --- load model --------------------------------------
    anp = load_anp(args.anp_ckpt,
                   data_root=csv_path.parent.parent,  # any root works
                   device=device, use_meta=args.use_meta)
    anp.eval()

    # --- load DRS_general model if specified ----------------
    drs = None
    if args.drs_ckpt is not None:
        from model.anp_improved import MLP          # same helper you used before
        # x_dim: use ANP config if available, else infer from CSV
        x_dim = anp.cfg.x_dim
        drs = MLP(x_dim, hidden=128, out_dim=2, n_layers=3)
        drs.load_state_dict(torch.load(args.drs_ckpt, map_location="cpu")["model_state"])
        drs.to(device).eval()
        print(f"✓ Loaded DRS_general from {args.drs_ckpt}")

    # --- load trajectories -------------------------------
    trajs = load_trajectories(csv_path, use_meta=args.use_meta)
    sel   = random.sample(trajs, k=min(args.n_trajs, len(trajs)))

    if args.ctx_mode == "first":
        out_dir = out_dir / "extrapolation"
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = out_dir / "interpolation"
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving plots to {out_dir} …")
    for k, (sensors, coords) in enumerate(sel, start=1):
        out_file = out_dir / f"traj_{k:02d}.png"
        plot_traj(anp, sensors, coords, out_file,
          device=device,
          ctx_frac=args.ctx_frac,
          ctx_mode=args.ctx_mode,
          teacher=args.teacher,
          drs=drs)

    print("✓ Done.")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
