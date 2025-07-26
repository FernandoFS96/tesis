# ============================================================
# train_size_mlps.py
# ============================================================
"""
Entrena 4 MLP para el experimento “Size tests (profundidad 20 m, 4 sensores)”:

  • 3 modelos individuales: 2x1 m, 4x2 m, 8x4 m
  • 1 modelo combinado   : DRS_size  (las 3 carpetas juntas)

Cada modelo usa la arquitectura base del DRS_general.
Registra barra tqdm, early-stop con paciencia, curva MAE, summary CSV y
checkpoint del mejor MAE.

Use: 
    python train_size_mlps.py \
  --data-root /home/fernando/tesis/ANP/data/tasks/3x3x3 \
  --out-dir  /home/fernando/tesis/ANP/training/drs_size_runs \
  --device cuda --batch-size 500 --patience 250
"""
from __future__ import annotations
import argparse, signal, time, csv
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
ROOT = Path(__file__).resolve().parent.parent.parent   # .../ANP
sys.path.append(str(ROOT))

from utils.data_loading import SENSOR_COLS, Y_COLS, parse_meta
from model.anp_improved import MLP

# ───────────────────────── Dataset (un único CSV) ─────────────────────────
class SingleCSVTrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path: str | Path, *, use_meta: bool = False):
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)

        # Rellenar sensores faltantes con 0
        for c in SENSOR_COLS:
            if c not in df.columns:
                df[c] = 0.0

        depth, length, width = parse_meta(csv_path.parent.name)
        meta = torch.tensor([depth, length, width], dtype=torch.float32)

        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for _, traj in df.groupby("traj_id"):
            s = torch.tensor(traj[SENSOR_COLS].values, dtype=torch.float32)
            if use_meta:
                s = torch.cat([s, meta.repeat(s.size(0), 1)], dim=-1)
            c = torch.tensor(traj[Y_COLS].values, dtype=torch.float32)
            self.samples.append((s, c))

        self.x_dim = self.samples[0][0].shape[1]

    def __len__(self):  return len(self.samples)
    def __getitem__(self, idx):  return self.samples[idx]

def flatten_to_points(samples):
    xs, ys = [], []
    for s, c in samples:
        xs.append(s); ys.append(c)
    return torch.cat(xs), torch.cat(ys)

# ─────────────────────── Entrenamiento con tqdm ───────────────────────────
def train_mlp(model: nn.Module,
              train_ds, val_ds,
              batch_size: int, device: str,
              epochs: int, lr: float,
              ckpt_dir: Path, patience: int,
              model_name: str):

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              pin_memory=(device == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              pin_memory=(device == "cuda"))

    optim  = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    best_mae = float("inf"); best_ckpt: Path | None = None
    epochs_no_improve = 0
    train_hist: List[float] = []; val_hist: List[float] = []
    t0 = time.time(); interrupted = False

    def _handler(_, __):
        nonlocal interrupted
        interrupted = True
        print("\nCTRL-C recibido: terminaré la época y pararé.")
    signal.signal(signal.SIGINT, _handler)

    bar = tqdm(range(1, epochs + 1), unit="epoch", desc=model_name)
    for ep in bar:
        # ---- TRAIN ----
        model.train(); total_abs = total_n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device_type=device):
                y_hat = model(x)
                loss = nn.functional.l1_loss(y_hat, y)
            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update(); optim.zero_grad(set_to_none=True)
            total_abs += loss.item() * y.numel();  total_n += y.numel()
        train_mae = total_abs / total_n;  train_hist.append(train_mae)

        # ---- VAL ----
        model.eval(); total_abs = total_n = 0
        with torch.inference_mode():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                total_abs += (model(x) - y).abs().sum().item()
                total_n   += y.numel()
        val_mae = total_abs / total_n;  val_hist.append(val_mae)

        # ---- Early-stop / ckpt ----
        if val_mae < best_mae - 1e-6:
            best_mae = val_mae; epochs_no_improve = 0
            if best_ckpt and best_ckpt.exists(): best_ckpt.unlink()
            best_ckpt = ckpt_dir / f"{model_name}_best_ep{ep:04d}.pt"
            torch.save({"epoch": ep,
                        "model_state": model.state_dict(),
                        "optim_state": optim.state_dict(),
                        "best_mae": best_mae}, best_ckpt)
        else:
            epochs_no_improve += 1

        bar.set_postfix(train=f"{train_mae:.4f}",
                        val=f"{val_mae:.4f}",
                        best=f"{best_mae:.4f}",
                        wait=f"{epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience or interrupted:
            break
    bar.close()

    # ---- curva MAE ----
    plt.figure()
    plt.plot(train_hist, label="Train"); plt.plot(val_hist, label="Val")
    plt.legend(); plt.xlabel("Epoch"); plt.ylabel("MAE"); plt.tight_layout()
    plt.savefig(ckpt_dir / "mae_curve.png"); plt.close()

    # ---- resumen CSV ----
    summary = ckpt_dir / "training_summary.csv"
    write_head = not summary.exists()
    with summary.open("a", newline="") as f:
        w = csv.writer(f)
        if write_head:
            w.writerow(["run_datetime","model","epochs_ran","best_mae",
                        "train_time_sec","batch_size","lr","hidden_dim",
                        "n_layers","patience"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), model_name, ep,
                    round(best_mae,6), int(time.time()-t0),
                    batch_size, lr, patience])

    print(f"✓ {model_name} entrenado | mejor MAE {best_mae:.4f} | ckpt {best_ckpt}")

# ─────────────────────────── CLI principal ────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Entrena MLPs – experimento Size (20 m, 4 sensores)")
    ap.add_argument("--data-root", required=True,
                    help="…/data/tasks/3x3x3")
    ap.add_argument("--out-dir", default="drs_size_runs")
    ap.add_argument("--device", choices=["cpu","cuda"], default="cuda")
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=10000)
    ap.add_argument("--patience", type=int, default=500)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--use-meta", action="store_true")
    args = ap.parse_args(); device=args.device
    out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)

    # ---- CSVs objetivo (profundidad 20 m, sensores=4) ----
    sizes = {
        "2x1m": Path(args.data_root)/"20m-64T_2mx1m/dataset_without_sensor_1_3.csv",
        "4x2m": Path(args.data_root)/"20m-64T_4mx2m/dataset_without_sensor_1_3.csv",
        "8x4m": Path(args.data_root)/"20m-64T_8mx4m/dataset_without_sensor_1_3.csv",
    }
    for k,p in sizes.items():
        if not p.exists():
            raise RuntimeError(f"Falta CSV {p}")
    print("Datasets seleccionados:")
    for k,p in sizes.items(): print(f"  {k}: {p}")

    ckpts = {}  # para guardar las rutas de checkpoints

    # ---- 3 MLP individuales ----
    for size_key, csv_path in sizes.items():
        ds = SingleCSVTrajectoryDataset(csv_path, use_meta=args.use_meta)
        val_len = int(0.2*len(ds)); train_len = len(ds) - val_len
        train_traj, val_traj = random_split(ds, [train_len,val_len],
                                            generator=torch.Generator().manual_seed(18))
        x_train,y_train = flatten_to_points(train_traj)
        x_val,  y_val   = flatten_to_points(val_traj)
        train_ds = TensorDataset(x_train,y_train); val_ds = TensorDataset(x_val,y_val)

        model = MLP(ds.x_dim, hidden=128, out_dim=2, n_layers=3).to(device)
        ckpt_dir = out_root / size_key; ckpt_dir.mkdir(exist_ok=True)
        train_mlp(model, train_ds, val_ds,
                  args.batch_size, device,
                  args.epochs, args.lr,
                  ckpt_dir, args.patience,
                  model_name=f"MLP_{size_key}")
        ckpts[size_key] = next(ckpt_dir.glob("MLP_*_best_ep*.pt"))

    # ---- MLP combinado (DRS_size) ----
    combined_samples = []
    for csv_path in sizes.values():
        combined_samples += SingleCSVTrajectoryDataset(csv_path,
                                                       use_meta=args.use_meta).samples
    x_all, y_all = flatten_to_points(combined_samples)
    comb_ds = TensorDataset(x_all, y_all)
    val_len = int(0.2*len(comb_ds)); train_len = len(comb_ds) - val_len
    train_ds, val_ds = random_split(comb_ds, [train_len,val_len],
                                    generator=torch.Generator().manual_seed(18))

    model = MLP(x_all.shape[1], hidden=128, out_dim=2, n_layers=3).to(device)
    ckpt_dir = out_root / "combined"; ckpt_dir.mkdir(exist_ok=True)
    train_mlp(model, train_ds, val_ds,
              args.batch_size, device,
              args.epochs, args.lr,
              ckpt_dir, args.patience,
              model_name="DRS_size")
    ckpts["combined"] = next(ckpt_dir.glob("DRS_size_best_ep*.pt"))

    print("\nCheckpoint finales:")
    for k,v in ckpts.items(): print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
