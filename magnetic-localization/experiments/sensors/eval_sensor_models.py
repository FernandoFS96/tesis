# ============================================================
# eval_sensor_models.py  (with optional extrapolation evaluation)
# ============================================================
"""
Evalúa 6 modelos sobre los datasets de sensores (6, 5, 4) en
20m-64T_8mx4m y, opcionalmente, un CSV OOD:

  • MLP_6s, MLP_5s, MLP_4s
  • DRS_sensors
  • DRS_general, ANP

Genera:
  • mae_sensor_tests.csv          ── o  mae_sensor_tests_OOD.csv
  • heatmap_sensor_tests.png      ── o  heatmap_sensor_tests_OOD.png

Uso:
    python eval_sensor_models.py \
      --data-root /home/fernando/tesis/ANP/data/tasks/3x3x3/20m-64T_8mx4m \
      --mlp-dir   /home/fernando/tesis/ANP/training/drs_sensor_runs \
      --anp-ckpt  /home/fernando/tesis/ANP/training/anp_runs/no_sensor_mask/anp_best_ep9933.pt \
      --drs-general-ckpt /home/fernando/tesis/ANP/training/drs_runs/drs_general_best_ep7958.pt \
      --device cuda --batch-size 10 \
      --ood-csv /home/fernando/tesis/ANP/data/tasks/1x1x1/20m-64T_8mx4m/dataset_without_sensor_1_3_5.csv 

    python eval_sensor_models.py \
      --data-root /home/fernando/tesis/ANP/data/tasks/3x3x3/20m-64T_8mx4m \
      --mlp-dir   /home/fernando/tesis/ANP/training/drs_sensor_runs \
      --anp-ckpt  /home/fernando/tesis/ANP/training/anp_runs/no_sensor_mask/extrapolation/anp_best.pt \
      --drs-general-ckpt /home/fernando/tesis/ANP/training/drs_runs/drs_general_best_ep7958.pt \
      --ood-csv /home/fernando/tesis/ANP/data/tasks/1x1x1/20m-64T_8mx4m/dataset_without_sensor_1_3_5.csv \
      --device cuda --batch-size 10 --extrapolation-eval --ctx-size 20
"""
from __future__ import annotations
import argparse, random
from pathlib import Path
from collections import OrderedDict
from functools import partial

import torch, pandas as pd, numpy as np, matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

import sys
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))

from utils.data_loading import SENSOR_COLS, Y_COLS, episodic_collate, parse_meta
from model.anp_improved import MLP
from utils.load_anp_from_ckpt import load_anp
from utils.eval_helpers import mae_anp_fulltraj

# ───────────────────── helpers ─────────────────────
def load_flat(csv_path: Path, use_meta=False):
    df = pd.read_csv(csv_path)
    for c in SENSOR_COLS:
        if c not in df.columns:
            df[c] = 0.0
    d, l, w = parse_meta(csv_path.parent.name)
    meta = torch.tensor([d, l, w], dtype=torch.float32)

    xs, ys = [], []
    for _, traj in df.groupby("traj_id"):
        s = torch.tensor(traj[SENSOR_COLS].values, dtype=torch.float32)
        if use_meta:
            s = torch.cat([s, meta.repeat(s.size(0),1)], dim=-1)
        c = torch.tensor(traj[Y_COLS].values, dtype=torch.float32)
        xs.append(s); ys.append(c)
    return torch.cat(xs), torch.cat(ys)

def mae_mlp(model, x, y, batch, device):
    model.eval()
    tot_abs = tot_n = 0
    loader = DataLoader(TensorDataset(x, y),
                        batch_size=batch,
                        shuffle=False,
                        pin_memory=(device=="cuda"))
    with torch.inference_mode():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            tot_abs += (model(xb) - yb).abs().sum().item()
            tot_n   += yb.numel()
    return tot_abs / tot_n

def heatmap(df: pd.DataFrame, png: Path):
    models, cols, data = df.index, df.columns, df.values
    fig, ax = plt.subplots(figsize=(1+len(cols)*1.2, 3))
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(cols)), labels=cols, rotation=60)
    ax.set_yticks(np.arange(len(models)), labels=models)
    for i in range(len(models)):
        for j in range(len(cols)):
            ax.text(j, i, f"{data[i,j]:.2f}",
                    ha="center", va="center",
                    color="white" if data[i,j]<data.max()/2 else "black",
                    fontsize=8)
    plt.colorbar(im, ax=ax, shrink=.75, label="MAE (m)")
    plt.tight_layout()
    plt.savefig(png, dpi=300)
    plt.close()

# ───────────────────────── main ─────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Eval sensor-test (6×3 [+ OOD + extrapolation])")
    ap.add_argument("--data-root", required=True,
                    help="…/data/tasks/3x3x3/20m-64T_8mx4m")
    ap.add_argument("--mlp-dir",   required=True,
                    help="drs_sensor_runs con subdirs 6s/ 5s/ 4s/ combined/")
    ap.add_argument("--anp-ckpt",  required=True)
    ap.add_argument("--drs-general-ckpt", required=True)
    ap.add_argument("--device", choices=["cpu","cuda"], default="cuda")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--use-meta", action="store_true")
    ap.add_argument("--out-dir", default="sensor_eval_results")
    ap.add_argument("--ood-csv", type=Path,
                    help="CSV extra OOD (e.g. .../dataset_without_sensor_1_3_5.csv)")
    ap.add_argument("--full-traj", action="store_true",
                    help="MAE del ANP sobre toda la trayectoria.")
    ap.add_argument("--extrapolation-eval", action="store_true",
                    help="Usar contextos secuenciales (primeros k puntos).")
    ap.add_argument("--ctx-size", type=int,
                    help="Fija el número de puntos de contexto (con extrapolation-eval).")
    args = ap.parse_args()
    device = args.device

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- CSVs in-distribution ----
    base = Path(args.data_root)
    csvs = OrderedDict([
        ("6s", base/"dataset.csv"),
        ("5s", base/"dataset_without_sensor_1.csv"),
        ("4s", base/"dataset_without_sensor_1_3.csv"),
    ])

    # ---- OOD opcional ----
    if args.ood_csv:
        if not args.ood_csv.exists():
            raise RuntimeError(f"El OOD CSV '{args.ood_csv}' no existe")
        csvs["3s"] = args.ood_csv

    for label, path in csvs.items():
        if not path.exists():
            raise RuntimeError(f"Falta CSV {path}")

    # ---- cargar modelos ----
    anp = load_anp(args.anp_ckpt,
                   data_root=args.data_root,
                   device=device,
                   use_meta=args.use_meta)
    x_dim = anp.cfg.x_dim

    mlp_paths = {
        "MLP_6s":      next((Path(args.mlp_dir)/"6s").glob("MLP_*_best_ep*.pt")),
        "MLP_5s":      next((Path(args.mlp_dir)/"5s").glob("MLP_*_best_ep*.pt")),
        "MLP_4s":      next((Path(args.mlp_dir)/"4s").glob("MLP_*_best_ep*.pt")),
        "DRS_sensors": next((Path(args.mlp_dir)/"combined").glob("DRS_sensors_best_ep*.pt")),
    }
    mlps = {}
    for name, ckpt in mlp_paths.items():
        m = MLP(x_dim, hidden=128, out_dim=2, n_layers=3)
        m.load_state_dict(torch.load(ckpt, map_location="cpu")["model_state"])
        mlps[name] = m.to(device).eval()

    drs_general = MLP(x_dim, hidden=128, out_dim=2, n_layers=3)
    drs_general.load_state_dict(torch.load(args.drs_general_ckpt,
                                           map_location="cpu")["model_state"])
    drs_general.to(device).eval()

    # ---- preparar collate_fn ----
    collate_fn = partial(
        episodic_collate,
        ctx_mode="sequential" if args.extrapolation_eval else "random",
        fixed_ctx_size=(args.ctx_size if args.extrapolation_eval else None)
    )

    # ---- evaluación ----
    rows = []
    for folder, csv_path in csvs.items():
        # 1) MLPs + DRS_general
        x_flat, y_flat = load_flat(csv_path, use_meta=args.use_meta)
        res = {}
        for name, m in mlps.items():
            res[name] = mae_mlp(m, x_flat, y_flat, args.batch_size, device)
        res["DRS_general"] = mae_mlp(drs_general, x_flat, y_flat, args.batch_size, device)

        # 2) ANP
        class SingleCSVTraj(torch.utils.data.Dataset):
            def __init__(self, path, *, use_meta=False):
                import pandas as pd, torch
                self.samples = []
                df = pd.read_csv(path)
                for c in SENSOR_COLS:
                    if c not in df.columns:
                        df[c] = 0.0
                d, l, w = parse_meta(path.parent.name)
                meta = torch.tensor([d, l, w], dtype=torch.float32)
                for _, traj in df.groupby("traj_id"):
                    s = torch.tensor(traj[SENSOR_COLS].values, dtype=torch.float32)
                    if use_meta:
                        s = torch.cat([s, meta.repeat(s.size(0),1)], -1)
                    c = torch.tensor(traj[Y_COLS].values, dtype=torch.float32)
                    self.samples.append((s, c))
            def __len__(self): return len(self.samples)
            def __getitem__(self, i): return self.samples[i]

        if args.full_traj:
            ds = SingleCSVTraj(csv_path, use_meta=args.use_meta)
            vals = [mae_anp_fulltraj(anp, s, c, device) for s, c in ds]
            res["ANP"] = sum(vals) / len(vals)
        else:
            loader = DataLoader(
                SingleCSVTraj(csv_path, use_meta=args.use_meta),
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                pin_memory=(device=="cuda")
            )
            total_err = total_pts = 0.0
            with torch.inference_mode():
                for x_c, y_c, x_t, y_t, cm, tm in loader:
                    x_c, y_c, x_t, y_t = [t.to(device) for t in (x_c, y_c, x_t, y_t)]
                    cm, tm = cm.to(device), tm.to(device)
                    dist, _ = anp.forward(x_c, y_c, x_t, ctx_mask=cm, tgt_mask=tm)
                    ae = (dist.mean - y_t).abs().sum(-1)
                    ae = ae.masked_fill(tm, 0.0)
                    total_err += ae.sum().item()
                    total_pts += (~tm).sum().item()
            res["ANP"] = total_err / max(total_pts, 1)

        rows.append({"folder": folder, **res})
        print(folder, {k: f"{v:.4f}" for k, v in res.items()})

    # ---- guardar resultados & heatmap ----
    df = pd.DataFrame(rows).set_index("folder").T

    # if --extrapolation-eval present, save to extrapolation subdir
    if args.extrapolation_eval:
        out_dir = out_dir / "extrapolation"
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = out_dir / "interpolation"
        out_dir.mkdir(parents=True, exist_ok=True)

    if args.ood_csv:
        out_csv = out_dir/"mae_sensor_tests_OOD.csv"
        out_png = out_dir/"heatmap_sensor_tests_OOD.png"
    else:
        out_csv = out_dir/"mae_sensor_tests.csv"
        out_png = out_dir/"heatmap_sensor_tests.png"

    df.T.to_csv(out_csv, float_format="%.6f")
    print(f"✓ CSV {out_csv}")
    heatmap(df, out_png)
    print("✓ Sensor-test completado")

if __name__ == "__main__":
    main()
