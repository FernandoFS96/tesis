# ============================================================
# evaluate_models.py  (ANP vs. DRS_general baseline)
# ============================================================
"""
Compares an Attentive Neural Process checkpoint against the
deterministic DRS_general MLP on every folder-level dataset.

Two evaluation modes are available:

* Default  : random context/target split  (interpolation test)
* --extrapolation-eval : context = first k points (extrapolation test)

ID and OOD datasets can be evaluated by passing the flag

    --data-root /home/fernando/tesis/ANP/data/tasks/3x3x3

    --data-root /home/fernando/tesis/ANP/data/tasks/1x1x1


The script produces:
  • mae_per_folder_ID.csv / mae_per_folder_OOD.csv
  • heat-maps (saved under results folder)

Example
-------

python evaluate_models.py \
  --data-root /home/fernando/tesis/ANP/data/tasks/3x3x3 \
  --anp-ckpt  /home/fernando/tesis/ANP/training/anp_runs/no_sensor_mask/anp_best_ep9933.pt \
  --drs-general-ckpt /home/fernando/tesis/ANP/training/drs_runs/drs_general_best_ep7958.pt \
  --device cuda \
  
python evaluate_models.py \
  --data-root /home/fernando/tesis/ANP/data/tasks/3x3x3 \
  --anp-ckpt  /home/fernando/tesis/ANP/training/anp_runs/no_sensor_mask/extrapolation/anp_best.pt \
  --drs-general-ckpt /home/fernando/tesis/ANP/training/drs_runs/drs_general_best_ep7958.pt \
  --device cuda \
  --extrapolation-eval \
  --ctx-size 20 \
"""
from __future__ import annotations

import argparse, random
from functools import partial
from pathlib import Path
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
import sys; sys.path.append(str(ROOT))

from utils.data_loading import (
    SENSOR_COLS, Y_COLS, parse_meta,
    episodic_collate, MagneticTrajectoryDataset
)
from model.anp_improved import MLP
from utils.load_anp_from_ckpt import load_anp

# ----------------------------------------------------------------------
def load_flat(csv_path: Path, *, use_meta=False):
    """Return (x,y) tensors for a whole CSV."""
    df = pd.read_csv(csv_path)
    for c in SENSOR_COLS:
        if c not in df.columns:
            df[c] = 0.0
    d, l, w = parse_meta(csv_path.parent.name)
    meta = torch.tensor([d, l, w], dtype=torch.float32)

    xs, ys = [], []
    for _, g in df.groupby("traj_id"):
        s = torch.tensor(g[SENSOR_COLS].values, dtype=torch.float32)
        if use_meta:
            s = torch.cat([s, meta.repeat(s.size(0), 1)], dim=-1)
        c = torch.tensor(g[Y_COLS].values, dtype=torch.float32)
        xs.append(s); ys.append(c)
    return torch.cat(xs), torch.cat(ys)

# ----------------------------------------------------------------------
def mae_mlp(model, x, y, batch, device):
    model.eval(); tot_abs = tot_n = 0
    loader = DataLoader(TensorDataset(x, y), batch_size=batch,
                        shuffle=False, pin_memory=(device == "cuda"))
    with torch.inference_mode():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            tot_abs += (model(xb) - yb).abs().sum().item()
            tot_n   += yb.numel()
    return tot_abs / tot_n

# ----------------------------------------------------------------------
def heatmap(df: pd.DataFrame, png: Path):
    models, cols, data = df.index, df.columns, df.values
    fig, ax = plt.subplots(figsize=(1 + len(cols) * 0.8, 3))
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(cols)), labels=cols, rotation=60)
    ax.set_yticks(np.arange(len(models)), labels=models)
    for i in range(len(models)):
        for j in range(len(cols)):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                    color="white" if data[i, j] < data.max() / 2 else "black",
                    fontsize=8)
    plt.colorbar(im, ax=ax, shrink=.75, label="MAE (m)")
    plt.tight_layout(); plt.savefig(png, dpi=300); plt.close()

# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="ANP vs. DRS_general evaluator")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--anp-ckpt",  required=True)
    ap.add_argument("--drs-general-ckpt", required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--use-meta", action="store_true")
    ap.add_argument("--out-dir", default="eval_results")
    ap.add_argument("--ood", type=Path, help="Optional OOD CSV to append as extra column.")
    ap.add_argument("--extrapolation-eval", action="store_true", help="Use sequential context (first-k) during evaluation.")
    ap.add_argument("--ctx-size", type=int, help="(Opcional) si --extrapolation-eval, fija el número de puntos de contexto")
    args = ap.parse_args(); device = args.device
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Collect CSVs (one per folder)
    # ------------------------------------------------------------------
    root = Path(args.data_root)
    csvs = OrderedDict()
    for csv_path in root.rglob("dataset*.csv"):
        key = csv_path.parent.name  # folder as label
        csvs[key] = csv_path
    if args.ood:
        if not args.ood.exists():
            raise RuntimeError(f"OOD CSV {args.ood} not found")
        csvs["OOD"] = args.ood

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    anp = load_anp(args.anp_ckpt, data_root=root,
                   device=device, use_meta=args.use_meta)
    anp.eval()

    x_dim = anp.cfg.x_dim
    drs = MLP(x_dim, hidden=128, out_dim=2, n_layers=3)
    drs.load_state_dict(torch.load(args.drs_general_ckpt,
                                   map_location="cpu")["model_state"])
    drs.to(device).eval()

    # ------------------------------------------------------------------
    # Collate choice according to flag
    # ------------------------------------------------------------------
    collate_fn = partial(
        episodic_collate,
        ctx_mode="sequential" if args.extrapolation_eval else "random",
        fixed_ctx_size=args.ctx_size if args.extrapolation_eval else None
    )

    # ------------------------------------------------------------------
    # Evaluate per folder
    # ------------------------------------------------------------------
    rows = []
    for label, csv_path in csvs.items():
        x, y = load_flat(csv_path, use_meta=args.use_meta)

        res = {
            "DRS_general": mae_mlp(drs, x, y, args.batch_size, device)
        }

        # ---- ANP evaluation with chosen collate mode ----
        class SingleCSVTraj(torch.utils.data.Dataset):
            def __init__(self, path, *, use_meta=False):
                self.samples = []
                df = pd.read_csv(path)
                for c in SENSOR_COLS:
                    if c not in df.columns:
                        df[c] = 0.0
                d, l, w = parse_meta(path.parent.name)
                meta = torch.tensor([d, l, w], dtype=torch.float32)
                for _, g in df.groupby("traj_id"):
                    s = torch.tensor(g[SENSOR_COLS].values, dtype=torch.float32)
                    if use_meta:
                        s = torch.cat([s, meta.repeat(s.size(0), 1)], -1)
                    c = torch.tensor(g[Y_COLS].values, dtype=torch.float32)
                    self.samples.append((s, c))
            def __len__(self): return len(self.samples)
            def __getitem__(self, idx): return self.samples[idx]

        loader = DataLoader(SingleCSVTraj(csv_path, use_meta=args.use_meta),
                            batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn,
                            pin_memory=(device == "cuda"))

        mae_anp, tot_pts = 0.0, 0
        with torch.inference_mode():
            for x_c, y_c, x_t, y_t, cm, tm in loader:
                x_c, y_c, x_t, y_t = [t.to(device) for t in (x_c, y_c, x_t, y_t)]
                cm, tm = cm.to(device), tm.to(device)
                dist, _ = anp.forward(x_c, y_c, x_t,
                                      ctx_mask=cm, tgt_mask=tm)
                ae = (dist.mean - y_t).abs().sum(-1)
                ae = ae.masked_fill(tm, 0.0)
                mae_anp += ae.sum().item()
                tot_pts += (~tm).sum().item()
        res["ANP"] = mae_anp / tot_pts

        rows.append({"folder": label, **res})
        print(label, {k: f"{v:.4f}" for k, v in res.items()})

    # ------------------------------------------------------------------
    # Save CSV and heat-map
    # ------------------------------------------------------------------
    df = pd.DataFrame(rows).set_index("folder").T
    tag = "extrap" if args.extrapolation_eval else "interp"
    csv_path = out_dir / f"mae_{tag}.csv"
    df.T.to_csv(csv_path, float_format="%.6f")
    print(f"✓ CSV {csv_path}")

    png_path = out_dir / f"heatmap_{tag}.png"
    heatmap(df, png_path)
    print(f"✓ Heat-map {png_path}")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
