# ============================================================
# eval_depth_models.py  (with optional extrapolation evaluation)
# ============================================================
"""
Evalúa 6 modelos (3 MLP individuales, DRS_depth, DRS_general, ANP)
en los 3 datasets “8 m x 4 m, 4 sensores” y genera:

  • mae_depth_tests.csv
  • heatmap_depth_tests.png

Opcionalmente, se puede evaluar el ANP en modo extrapolación,
usando siempre los primeros k puntos de contexto (secuencial).

Uso:
    # Evaluación estándar (contexto aleatorio/interpolación)
    python eval_depth_models.py \
      --data-root /home/fernando/tesis/ANP/data/tasks/3x3x3 \
      --mlp-dir   /home/fernando/tesis/ANP/training/drs_depth_runs \
      --anp-ckpt  /home/fernando/tesis/ANP/training/anp_runs/no_sensor_mask/anp_best_ep9933.pt \
      --drs-general-ckpt /home/fernando/tesis/ANP/training/drs_runs/drs_general_best_ep7958.pt \
      --ood-csv /home/fernando/tesis/ANP/data/tasks/1x1x1/30m-64T_8mx4m/dataset_without_sensor_1_3.csv \
      --device cuda --batch-size 10
      
    # Evaluación en modo extrapolación
    python eval_depth_models.py \
      --data-root /home/fernando/tesis/ANP/data/tasks/3x3x3 \
      --mlp-dir   /home/fernando/tesis/ANP/training/drs_depth_runs \
      --anp-ckpt  /home/fernando/tesis/ANP/training/anp_runs/no_sensor_mask/extrapolation/anp_best.pt \
      --drs-general-ckpt /home/fernando/tesis/ANP/training/drs_runs/drs_general_best_ep7958.pt \
      --ood-csv /home/fernando/tesis/ANP/data/tasks/1x1x1/30m-64T_8mx4m/dataset_without_sensor_1_3.csv \
      --device cuda --batch-size 10 --extrapolation-eval --ctx-size 30
"""
from __future__ import annotations
import argparse, time, random
from pathlib import Path
from collections import OrderedDict
from functools import partial

import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import sys
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))

from utils.data_loading import SENSOR_COLS, Y_COLS, episodic_collate, parse_meta
from model.anp_improved import MLP
from utils.load_anp_from_ckpt import load_anp
from utils.eval_helpers import mae_anp_fulltraj

# ---------- Data helpers ----------
def load_flat_points(csv_path: Path, use_meta=False):
    import pandas as pd, torch
    df = pd.read_csv(csv_path)
    for c in SENSOR_COLS:
        if c not in df.columns:
            df[c] = 0.0
    depth, length, width = parse_meta(csv_path.parent.name)
    meta_tensor = torch.tensor([depth, length, width], dtype=torch.float32)

    xs, ys = [], []
    for _, traj_df in df.groupby("traj_id"):
        sensors = torch.tensor(traj_df[SENSOR_COLS].values, dtype=torch.float32)
        if use_meta:
            sensors = torch.cat(
                [sensors, meta_tensor.repeat(sensors.size(0), 1)],
                dim=-1
            )
        coords = torch.tensor(traj_df[Y_COLS].values, dtype=torch.float32)
        xs.append(sensors); ys.append(coords)
    return torch.cat(xs), torch.cat(ys)

def mae_mlp(model, x, y, batch, device):
    model.eval()
    total_abs = total_n = 0
    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=batch,
        shuffle=False,
        pin_memory=(device == "cuda")
    )
    with torch.inference_mode():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            total_abs += (model(xb) - yb).abs().sum().item()
            total_n   += yb.numel()
    return total_abs / total_n

# ---------- Heat-map ----------
def heatmap(df: pd.DataFrame, out_png: Path):
    models, cols, data = df.index, df.columns, df.values
    fig, ax = plt.subplots(figsize=(1 + len(cols) * 1.2, 3))
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(cols)), labels=cols, rotation=60)
    ax.set_yticks(np.arange(len(models)), labels=models)
    for i in range(len(models)):
        for j in range(len(cols)):
            ax.text(
                j, i, f"{data[i, j]:.2f}",
                ha="center", va="center",
                color="white" if data[i, j] < data.max() / 2 else "black",
                fontsize=8
            )
    plt.colorbar(im, ax=ax, shrink=.75, label="MAE (m)")
    plt.tight_layout()
    # ajustar nombre según OOD
    if "OOD" in out_png.name:
        out_png = out_png.with_name("heatmap_depth_tests_OOD.png")
    else:
        out_png = out_png.with_name("heatmap_depth_tests.png")
    plt.savefig(out_png, dpi=300)
    plt.close()

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Eval. depth-tests 6×3")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--mlp-dir", required=True, help="Carpeta con subdirs 7-5m/ 10m/ 20m/ combined/")
    ap.add_argument("--anp-ckpt", required=True)
    ap.add_argument("--drs-general-ckpt", required=True)
    ap.add_argument("--device", choices=["cpu","cuda"], default="cuda")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--use-meta", action="store_true")
    ap.add_argument("--out-dir", default="depth_eval_results")
    ap.add_argument("--ood-csv", type=Path, help="CSV extra para evaluación OOD")
    ap.add_argument("--full-traj", action="store_true", help="MAE ANP sobre toda la trayectoria")
    # --- NEW: extrapolation eval mode ---
    ap.add_argument("--extrapolation-eval", action="store_true", help="Usar contextos secuenciales (primeros k puntos)")
    ap.add_argument("--ctx-size", type=int, help="(Opcional) si --extrapolation-eval, fija el número de puntos de contexto")
    args = ap.parse_args()
    device = args.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- localizar CSVs ----
    csvs = OrderedDict([
        ("7-5m", Path(args.data_root)/"7-5m-64T_8mx4m/dataset_without_sensor_1_3.csv"),
        ("10m",  Path(args.data_root)/"10m-64T_8mx4m/dataset_without_sensor_1_3.csv"),
        ("20m",  Path(args.data_root)/"20m-64T_8mx4m/dataset_without_sensor_1_3.csv"),
    ])
    if args.ood_csv:
        if not args.ood_csv.exists():
            raise RuntimeError(f"Falta CSV {args.ood_csv}")
        label = args.ood_csv.parent.name.split("-")[0]
        csvs[label] = args.ood_csv
    for k, p in csvs.items():
        if not p.exists():
            raise RuntimeError(f"Falta CSV {p}")

    # ---- cargar modelos ----
    anp = load_anp(args.anp_ckpt,
                   data_root=args.data_root,
                   device=device,
                   use_meta=args.use_meta)
    x_dim = anp.cfg.x_dim

    # cargar MLPs de profundidad
    mlp_paths = {
        "MLP_7-5m":  next((Path(args.mlp_dir)/"7-5m").glob("MLP_*_best_ep*.pt")),
        "MLP_10m":   next((Path(args.mlp_dir)/"10m").glob("MLP_*_best_ep*.pt")),
        "MLP_20m":   next((Path(args.mlp_dir)/"20m").glob("MLP_*_best_ep*.pt")),
        "DRS_depth": next((Path(args.mlp_dir)/"combined").glob("DRS_depth_best_ep*.pt")),
    }
    mlps = {}
    for name, ckpt in mlp_paths.items():
        m = MLP(x_dim, hidden=128, out_dim=2, n_layers=3)
        m.load_state_dict(torch.load(ckpt, map_location="cpu")["model_state"])
        mlps[name] = m.to(device).eval()

    drs_general = MLP(x_dim, hidden=128, out_dim=2, n_layers=3)
    drs_general.load_state_dict(
        torch.load(args.drs_general_ckpt, map_location="cpu")["model_state"]
    )
    drs_general.to(device).eval()

    # ---- preparar collate_fn según modo  ----
    collate_fn = partial(
        episodic_collate,
        ctx_mode="sequential" if args.extrapolation_eval else "random",
        fixed_ctx_size=args.ctx_size if args.extrapolation_eval else None
    )

    # ---- evaluar ----
    rows = []
    for folder, csv_path in csvs.items():
        # MLPs
        x_flat, y_flat = load_flat_points(csv_path, use_meta=args.use_meta)
        res = {}
        for name, m in mlps.items():
            res[name] = mae_mlp(m, x_flat, y_flat,
                                args.batch_size, device)
        res["DRS_general"] = mae_mlp(drs_general,
                                     x_flat, y_flat,
                                     args.batch_size, device)

        # ANP
        print(f"Evaluando ANP en {folder}…")
        class SingleCSVDataset(torch.utils.data.Dataset):
            def __init__(self, path, *, use_meta=False):
                import pandas as pd, torch
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
                self.x_dim = self.samples[0][0].shape[1]
            def __len__(self): return len(self.samples)
            def __getitem__(self, idx): return self.samples[idx]

        if args.full_traj:
            # Evaluación full trajectory
            mae_vals = []
            ds = SingleCSVDataset(csv_path, use_meta=args.use_meta)
            for sensors, coords in ds:
                mae_vals.append(
                    mae_anp_fulltraj(anp, sensors, coords, device)
                )
            res["ANP"] = sum(mae_vals) / len(mae_vals)
        else:
            # Evaluación con episodic_collate
            ds = SingleCSVDataset(csv_path, use_meta=args.use_meta)
            loader = DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                pin_memory=(device == "cuda")
            )
            total_err = total_pts = 0.0
            with torch.inference_mode():
                for x_c, y_c, x_t, y_t, cm, tm in loader:
                    x_c, y_c, x_t, y_t = [t.to(device) for t in (x_c, y_c, x_t, y_t)]
                    cm, tm = cm.to(device), tm.to(device)
                    dist, _ = anp.forward(
                        x_c, y_c, x_t,
                        ctx_mask=cm, tgt_mask=tm
                    )
                    ae = (dist.mean - y_t).abs().sum(-1)
                    ae = ae.masked_fill(tm, 0.0)
                    total_err += ae.sum().item()
                    total_pts += (~tm).sum().item()
            res["ANP"] = total_err / max(total_pts, 1)

        rows.append({"folder": folder, **res})
        print(folder, {k: f"{v:.4f}" for k, v in res.items()})

    # ---- guardar CSV ----
    df = pd.DataFrame(rows).set_index("folder").T

    # if --extrapolation-eval present, save to extrapolation subdir
    if args.extrapolation_eval:
        out_dir = out_dir / "extrapolation"
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = out_dir / "interpolation"
        out_dir.mkdir(parents=True, exist_ok=True)

    if args.ood_csv:
        out_csv = out_dir / "mae_depth_tests_OOD.csv"
    else:
        out_csv = out_dir / "mae_depth_tests.csv"
    df.T.to_csv(out_csv, float_format="%.6f")
    print(f"✓ CSV {out_csv}")

    # ---- generar heatmap ----
    if args.ood_csv:
        out_png = out_dir / "heatmap_depth_tests_OOD.png"
    else:
        out_png = out_dir / "heatmap_depth_tests.png"
    print(f"Generando heatmap en {out_png}…")
    heatmap(df, out_png)

    print("✓ Experimento depth-tests completado")

if __name__ == "__main__":
    main()
