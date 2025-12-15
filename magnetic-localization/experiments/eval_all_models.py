# ============================================================
# eval_all_models.py   (Depth ✕ Size ✕ Sensors — 14 x 12 tests)
# ============================================================
"""
One-shot evaluator that unifies the three task scripts:

    • Depth      →  3 ID + 1 OOD
    • Hull size  →  3 ID + 1 OOD
    • Sensor #   →  3 ID + 1 OOD
                   ─────────────────
                     9 ID + 3 OOD  → 12 data sets

Models loaded
─────────────
  9 x single-task MLPs
      MLP_7-5m,  MLP_10m,  MLP_20m
      MLP_2x1m,  MLP_4x2m, MLP_8x4m
      MLP_6s,    MLP_5s,   MLP_4s
  3 x task-specific DRS
      DRS_depth, DRS_size, DRS_sensors
  1 x global DRS (DRS_general)
  1 x Attentive Neural Process (ANP)
  ───────────────────────────────────
      14 models  →  14 x 12 MAE matrix

Output
──────
  • mae_all_tests.csv   (14 x 12)
  • heatmap_all_tests.png   (optional visual aid)

Usage example
─────────────

python eval_all_models.py \
   --tasks-root       /home/fernando/tesis/ANP/data/tasks \
   --depth-mlp-dir    /home/fernando/tesis/ANP/training/drs_depth_runs \
   --size-mlp-dir     /home/fernando/tesis/ANP/training/drs_size_runs \
   --sensor-mlp-dir   /home/fernando/tesis/ANP/training/drs_sensor_runs \
   --anp-ckpt         /home/fernando/tesis/ANP/training/anp_runs/no_sensor_mask/anp_best_ep9933.pt \
   --drs-general-ckpt /home/fernando/tesis/ANP/training/drs_runs/drs_general_best_ep7958.pt \
   --device           cuda \
   --batch-size       10

python eval_all_models.py \
   --tasks-root       /home/fernando/tesis/ANP/data/tasks \
   --depth-mlp-dir    /home/fernando/tesis/ANP/training/drs_depth_runs \
   --size-mlp-dir     /home/fernando/tesis/ANP/training/drs_size_runs \
   --sensor-mlp-dir   /home/fernando/tesis/ANP/training/drs_sensor_runs \
   --anp-ckpt         /home/fernando/tesis/ANP/training/anp_runs/no_sensor_mask/extrapolation/anp_best.pt \
   --drs-general-ckpt /home/fernando/tesis/ANP/training/drs_runs/drs_general_best_ep7958.pt \
   --device           cuda \
   --batch-size       10 \
   --extrapolation-eval --ctx-size 20
"""
from __future__ import annotations
import argparse, csv, itertools, random
from pathlib import Path
from collections import OrderedDict
from functools import partial

import torch, pandas as pd, numpy as np, matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# ---------- project imports ----------
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from utils.data_loading import (
    SENSOR_COLS, Y_COLS, episodic_collate, parse_meta
)
from model.anp_improved import MLP
from utils.load_anp_from_ckpt import load_anp
from utils.eval_helpers import mae_anp_fulltraj

# ══════════════════════════════════════════════════════════════
#                 Helper functions (shared logic)
# ══════════════════════════════════════════════════════════════
def load_flat_csv(csv_path: Path, *, use_meta=False):
    """Stack all points from one CSV into two big tensors (X, Y)."""
    df = pd.read_csv(csv_path)
    for c in SENSOR_COLS:                           # pad missing cols if needed
        if c not in df.columns:
            df[c] = 0.0
    depth, length, width = parse_meta(csv_path.parent.name)
    meta = torch.tensor([depth, length, width], dtype=torch.float32)

    xs, ys = [], []
    for _, g in df.groupby("traj_id"):
        s = torch.tensor(g[SENSOR_COLS].values, dtype=torch.float32)
        if use_meta:
            s = torch.cat([s, meta.repeat(s.size(0), 1)], -1)
        p = torch.tensor(g[Y_COLS].values, dtype=torch.float32)
        xs.append(s);  ys.append(p)
    return torch.cat(xs), torch.cat(ys)

def mae_mlp(model, x, y, *, batch, device):
    model.eval()
    loader = DataLoader(TensorDataset(x, y),
                        batch_size=batch,
                        shuffle=False,
                        pin_memory=(device == "cuda"))
    total_abs = total_n = 0
    with torch.inference_mode():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            total_abs += (model(xb) - yb).abs().sum().item()
            total_n   += yb.numel()
    return total_abs / total_n

def make_heatmap(df: pd.DataFrame, png_path: Path):
    fig, ax = plt.subplots(figsize=(1 + len(df.columns)*1.3, 1 + len(df)*0.5))
    im = ax.imshow(df.values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(df.columns)), labels=df.columns, rotation=60)
    ax.set_yticks(np.arange(len(df.index)),   labels=df.index)
    for i, j in itertools.product(range(len(df.index)), range(len(df.columns))):
        ax.text(j, i, f"{df.iat[i, j]:.2f}",
                ha="center", va="center",
                color=("white" if df.iat[i, j] < df.values.max()/2 else "black"),
                fontsize=7)
    plt.colorbar(im, ax=ax, shrink=.75, label="MAE (m)")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()

# ══════════════════════════════════════════════════════════════
#                            Main
# ══════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description="Unified evaluator for depth/size/sensor tasks (14 x 12)"
    )
    ap.add_argument("--tasks-root", required=True,
        help="Folder that contains 3x3x3 and 1x1x1 task sub-trees as before")
    ap.add_argument("--depth-mlp-dir",   required=True)
    ap.add_argument("--size-mlp-dir",    required=True)
    ap.add_argument("--sensor-mlp-dir",  required=True)
    ap.add_argument("--anp-ckpt",        required=True)
    ap.add_argument("--drs-general-ckpt", required=True)
    ap.add_argument("--device", choices=["cpu","cuda"], default="cuda")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--use-meta", action="store_true")
    ap.add_argument("--out-dir", default="all_eval_results")
    # interpolation vs extrapolation
    ap.add_argument("--extrapolation-eval", action="store_true",
        help="Force sequential contexts (extrapolation mode)")
    ap.add_argument("--ctx-size", type=int,
        help="Fixed # context points (only with --extrapolation-eval)")
    ap.add_argument("--full-traj", action="store_true",
        help="Evaluate ANP on full trajectory (ignores episodic_collate)")
    args = ap.parse_args()
    device = args.device

    # ═════════════════════════════ data-set catalogue ═════════════════════════════
    root3 = Path(args.tasks_root) / "3x3x3"   # in-distribution tree (9 CSVs)
    root1 = Path(args.tasks_root) / "1x1x1"   # each OOD CSV lives here
    csv_map = OrderedDict([
        # Depth (8x4 hull, 4 sensors)
        ("DEPTH_7-5m", root3/"7-5m-64T_8mx4m/dataset_without_sensor_1_3.csv"),
        ("DEPTH_10m",  root3/"10m-64T_8mx4m/dataset_without_sensor_1_3.csv"),
        ("DEPTH_20m",  root3/"20m-64T_8mx4m/dataset_without_sensor_1_3.csv"),
        ("DEPTH_30m_OOD", root1/"30m-64T_8mx4m/dataset_without_sensor_1_3.csv"),
        # Hull size (depth 20 m, 4 sensors)
        ("SIZE_2x1m", root3/"20m-64T_2mx1m/dataset_without_sensor_1_3.csv"),
        ("SIZE_4x2m", root3/"20m-64T_4mx2m/dataset_without_sensor_1_3.csv"),
        ("SIZE_8x4m", root3/"20m-64T_8mx4m/dataset_without_sensor_1_3.csv"),
        ("SIZE_20x10m_OOD", root1/"20m-64T_20mx10m/dataset_without_sensor_1_3.csv"),
        # Sensors (8x4 hull, depth 20 m)
        ("SENS_6s", root3/"20m-64T_8mx4m/dataset.csv"),
        ("SENS_5s", root3/"20m-64T_8mx4m/dataset_without_sensor_1.csv"),
        ("SENS_4s", root3/"20m-64T_8mx4m/dataset_without_sensor_1_3.csv"),
        ("SENS_3s_OOD", root1/"20m-64T_8mx4m/dataset_without_sensor_1_3_5.csv"),
    ])
    for lbl, p in csv_map.items():
        if not p.exists():
            raise FileNotFoundError(p)

    # ═══════════════════════ load ANP & helper collate ═══════════════════════
    anp = load_anp(
        args.anp_ckpt,
        data_root=root3,        # any valid data root works for config
        device=device,
        use_meta=args.use_meta
    )
    x_dim = anp.cfg.x_dim
    collate_fn = partial(
        episodic_collate,
        ctx_mode=("sequential" if args.extrapolation_eval else "random"),
        fixed_ctx_size=(args.ctx_size if args.extrapolation_eval else None)
    )

    # ═══════════════════════ helper to load one MLP ckpt ══════════════════════
    def _load_mlp(ckpt_path: Path):
        m = MLP(x_dim, hidden=128, out_dim=2, n_layers=3)
        m.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model_state"])
        return m.to(device).eval()

    # ---- depth models -------------------------------------------------------
    depth_ckpts = {
        "MLP_7-5m":  next((Path(args.depth_mlp_dir)/"7-5m").glob("MLP_*_best_ep*.pt")),
        "MLP_10m":   next((Path(args.depth_mlp_dir)/"10m").glob("MLP_*_best_ep*.pt")),
        "MLP_20m":   next((Path(args.depth_mlp_dir)/"20m").glob("MLP_*_best_ep*.pt")),
        "DRS_depth": next((Path(args.depth_mlp_dir)/"combined").glob("DRS_depth_*.pt")),
    }
    # ---- size models --------------------------------------------------------
    size_ckpts = {
        "MLP_2x1m": next((Path(args.size_mlp_dir)/"2x1m").glob("MLP_*_best_ep*.pt")),
        "MLP_4x2m": next((Path(args.size_mlp_dir)/"4x2m").glob("MLP_*_best_ep*.pt")),
        "MLP_8x4m": next((Path(args.size_mlp_dir)/"8x4m").glob("MLP_*_best_ep*.pt")),
        "DRS_size": next((Path(args.size_mlp_dir)/"combined").glob("DRS_size_*.pt")),
    }
    # ---- sensor models ------------------------------------------------------
    sensor_ckpts = {
        "MLP_6s":      next((Path(args.sensor_mlp_dir)/"6s").glob("MLP_*_best_ep*.pt")),
        "MLP_5s":      next((Path(args.sensor_mlp_dir)/"5s").glob("MLP_*_best_ep*.pt")),
        "MLP_4s":      next((Path(args.sensor_mlp_dir)/"4s").glob("MLP_*_best_ep*.pt")),
        "DRS_sensors": next((Path(args.sensor_mlp_dir)/"combined").glob("DRS_sensors_*.pt")),
    }

    # ---- global DRS ---------------------------------------------------------
    drs_general = _load_mlp(Path(args.drs_general_ckpt))
    model_dict = {"ANP": anp, "DRS_general": drs_general}
    # add all other ckpts
    for name, ck in {**depth_ckpts, **size_ckpts, **sensor_ckpts}.items():
        model_dict[name] = _load_mlp(ck)

    # ═══════════════════════ evaluation loop ═══════════════════════
    results = {m: {} for m in model_dict}          # nested dict: model → dataset → MAE

    for ds_lbl, csv_path in csv_map.items():
        # ---------- load flat tensors once ----------
        x_flat, y_flat = load_flat_csv(csv_path, use_meta=args.use_meta)

        # ---------- run every deterministic MLP ----------
        for m_name, model in model_dict.items():
            if m_name == "ANP":          # skip for now
                continue
            results[m_name][ds_lbl] = mae_mlp(
                model, x_flat, y_flat,
                batch=args.batch_size,
                device=device
            )

        # ---------- ANP ----------
        if args.full_traj:
            # per-trajectory full-context evaluation
            class _DS(torch.utils.data.Dataset):
                def __init__(self, p):
                    df = pd.read_csv(p); self.samples=[]
                    for c in SENSOR_COLS:
                        if c not in df.columns: df[c]=0.0
                    d,l,w = parse_meta(p.parent.name)
                    meta = torch.tensor([d,l,w], dtype=torch.float32)
                    for _,g in df.groupby("traj_id"):
                        s = torch.tensor(g[SENSOR_COLS].values, dtype=torch.float32)
                        if args.use_meta: s = torch.cat([s, meta.repeat(s.size(0),1)], -1)
                        y = torch.tensor(g[Y_COLS].values, dtype=torch.float32)
                        self.samples.append((s,y))
                def __len__(self): return len(self.samples)
                def __getitem__(self,i): return self.samples[i]
            ds_t = _DS(csv_path)
            vals = [mae_anp_fulltraj(anp, s, y, device) for s,y in ds_t]
            results["ANP"][ds_lbl] = sum(vals)/len(vals)
        else:
            # episodic_collate evaluation
            class _TrajDS(torch.utils.data.Dataset):
                def __init__(self,p):
                    df=pd.read_csv(p); self.samples=[]
                    for c in SENSOR_COLS:
                        if c not in df.columns: df[c]=0.0
                    d,l,w=parse_meta(p.parent.name)
                    meta=torch.tensor([d,l,w],dtype=torch.float32)
                    for _,g in df.groupby("traj_id"):
                        s=torch.tensor(g[SENSOR_COLS].values,dtype=torch.float32)
                        if args.use_meta: s=torch.cat([s,meta.repeat(s.size(0),1)],-1)
                        y=torch.tensor(g[Y_COLS].values,dtype=torch.float32)
                        self.samples.append((s,y))
                def __len__(self): return len(self.samples)
                def __getitem__(self,i): return self.samples[i]
            loader = DataLoader(
                _TrajDS(csv_path),
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                pin_memory=(device=="cuda")
            )
            tot_err = tot_pts = 0
            with torch.inference_mode():
                for x_c,y_c,x_t,y_t,cm,tm in loader:
                    x_c,y_c,x_t,y_t=[t.to(device) for t in (x_c,y_c,x_t,y_t)]
                    cm,tm=cm.to(device),tm.to(device)
                    dist,_ = anp.forward(x_c,y_c,x_t,ctx_mask=cm,tgt_mask=tm)
                    ae = (dist.mean - y_t).abs().sum(-1)
                    ae = ae.masked_fill(tm,0.0)
                    tot_err += ae.sum().item()
                    tot_pts += (~tm).sum().item()
            results["ANP"][ds_lbl] = tot_err / max(tot_pts,1)

        print(f"✓ {ds_lbl}")

    # ═══════════════ persist CSV & optional heatmap ═══════════════
    out_dir = Path(args.out_dir)
    if args.extrapolation_eval:
        out_dir = out_dir/"extrapolation"
    else:
        out_dir = out_dir/"interpolation"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results).T[df_order := list(csv_map.keys())]  # ensure column order
    csv_path = out_dir/"mae_all_tests.csv"
    df.to_csv(csv_path, float_format="%.6f")
    print(f"\nResults saved to {csv_path}")

    try:
        make_heatmap(df, out_dir/"heatmap_all_tests.png")
        print("Heat-map generated.")
    except Exception as e:
        print(f"(Heat-map skipped: {e})")

if __name__ == "__main__":
    main()
