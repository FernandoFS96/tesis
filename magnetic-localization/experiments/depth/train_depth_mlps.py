# ============================================================
# train_depth_mlps.py
# ============================================================
"""
Entrena 4 modelos MLP para el experimento “Depth tests 8 m × 4 m, 4 sensores”:

  • 3 MLP individuales: 7-5 m, 10 m y 20 m de profundidad.
  • 1 MLP combinado  : DRS_depth (los 3 datasets juntos).

Cada modelo usa la misma arquitectura base que DRS_general y
registra curva MAE + training_summary.csv + checkpoint con early-stop.

Use:
    python train_depth_mlps.py \
  --data-root /home/fernando/tesis/ANP/data/tasks/3x3x3 \
  --out-dir  /home/fernando/tesis/ANP/training/drs_depth_runs \
  --device cuda --batch-size 500 --patience 200
"""
from __future__ import annotations
import argparse, time, signal, csv
from pathlib import Path
from typing import List
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────
#  Utilidades comunes
# ──────────────────────────────────────────────────────────────
import sys
ROOT = Path(__file__).resolve().parent.parent.parent   # .../ANP
sys.path.append(str(ROOT))

from utils.data_loading import SENSOR_COLS, Y_COLS, parse_meta
from model.anp_improved import MLP

# ---------- Dataset para un único CSV ----------
class SingleCSVTrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path: str | Path, *, use_meta: bool = False):
        csv_path = Path(csv_path)
        folder   = csv_path.parent.name

        df = pd.read_csv(csv_path)
        missing = [c for c in SENSOR_COLS if c not in df.columns]
        for col in missing:
            df[col] = 0.0

        self.samples, self.folder = [], folder
        depth, length, width = parse_meta(folder)
        meta_tensor = torch.tensor([depth, length, width], dtype=torch.float32)

        for _, traj_df in df.groupby("traj_id"):
            sensors = torch.as_tensor(traj_df[SENSOR_COLS].values, dtype=torch.float32)
            coords  = torch.as_tensor(traj_df[Y_COLS].values,    dtype=torch.float32)
            if use_meta:
                sensors = torch.cat([sensors, meta_tensor.repeat(sensors.size(0), 1)], dim=-1)
            self.samples.append((sensors, coords))

        self.x_dim = self.samples[0][0].shape[1]

    def __len__(self):             return len(self.samples)
    def __getitem__(self, idx):    return self.samples[idx]

# ---------- Flatten a nivel punto ----------
def flatten_to_points(samples):
    xs, ys = [], []
    for sensors, coords in samples:
        xs.append(sensors); ys.append(coords)
    return torch.cat(xs), torch.cat(ys)

# ---------- Early-stop MLP trainer -----------------
def train_mlp(model: nn.Module,
              train_ds, val_ds,
              batch_size: int, device: str,
              epochs: int, lr: float,
              ckpt_dir: Path, patience: int = 300,
              model_name: str = "MLP"):

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              pin_memory=(device == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              pin_memory=(device == "cuda"))

    optim  = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    best_mae = float("inf"); best_ckpt = None
    epochs_no_improve = 0
    train_hist, val_hist = [], []
    t0 = time.time()

    def _handler(_, __):
        nonlocal interrupted
        interrupted = True
        print("\nCTRL-C recibido: terminaré esta época y pararé.")
    interrupted = False
    signal.signal(signal.SIGINT, _handler)

    # ───────── bucle ─────────
    bar = tqdm(range(1, epochs + 1), unit="epoch", desc=model_name)
    for ep in bar:
        # ----- TRAIN -----
        model.train()
        total_abs = total_n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device_type=device):
                y_hat = model(x)
                loss = nn.functional.l1_loss(y_hat, y)      # MAE
            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update(); optim.zero_grad(set_to_none=True)

            total_abs += loss.item() * y.numel()
            total_n   += y.numel()
        train_mae = total_abs / total_n
        train_hist.append(train_mae)

        # ----- VAL -----
        model.eval(); total_abs = total_n = 0
        with torch.inference_mode():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                total_abs += (y_hat - y).abs().sum().item()
                total_n   += y.numel()
        val_mae = total_abs / total_n
        val_hist.append(val_mae)

        # ----- Early-stop & ckpt -----
        if val_mae < best_mae - 1e-6:          # mejora real (> tol)
            best_mae = val_mae
            epochs_no_improve = 0
            if best_ckpt and best_ckpt.exists():
                best_ckpt.unlink()
            best_ckpt = ckpt_dir / f"{model_name}_best_ep{ep:04d}.pt"
            torch.save({"epoch": ep,
                        "model_state": model.state_dict(),
                        "optim_state": optim.state_dict(),
                        "best_mae": best_mae}, best_ckpt)
        else:
            epochs_no_improve += 1

        # ----- Actualiza barra -----
        bar.set_postfix(train=f"{train_mae:.4f}",
                        val=f"{val_mae:.4f}",
                        best=f"{best_mae:.4f}",
                        wait=f"{epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience or interrupted:
            break
    bar.close()

    # ----- curva MAE + summary CSV (igual que antes) -----
    plt.figure(); plt.plot(train_hist,label="Train"); plt.plot(val_hist,label="Val")
    plt.legend(); plt.xlabel("Epoch"); plt.ylabel("MAE"); plt.tight_layout()
    plt.savefig(ckpt_dir / "mae_curve.png"); plt.close()

    summary_path = ckpt_dir / "training_summary.csv"
    write_head = not summary_path.exists()
    with open(summary_path,"a",newline="") as f:
        w = csv.writer(f)
        if write_head:
            w.writerow(["run_datetime","model","epochs_ran","best_mae","train_time_sec",
                        "batch_size","lr","hidden_dim","n_layers","patience"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), model_name, ep,
                    round(best_mae,6), int(time.time()-t0),
                    train_loader.batch_size, lr, patience])

    print(f"✓ {model_name} entrenado | mejor MAE {best_mae:.4f} | ckpt {best_ckpt}")

# ---------- Main CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Entrena MLPs para depth-test 8×4 m")
    ap.add_argument("--data-root", required=True,
                    help="…/data/tasks/3x3x3")
    ap.add_argument("--out-dir",   default="drs_depth_runs")
    ap.add_argument("--device",    choices=["cpu","cuda"], default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs",     type=int, default=10000)
    ap.add_argument("--patience",   type=int, default=200)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--use-meta", action="store_true")
    args = ap.parse_args(); device=args.device
    out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)

    # ---- localizar los 3 CSV target ----
    depths = {"7-5m": None, "10m": None, "20m": None}
    for folder in depths:
        glob_path = Path(args.data_root)/f"{folder}-64T_8mx4m/dataset_without_sensor_1_3.csv"
        if not glob_path.exists():
            raise RuntimeError(f"Falta CSV {glob_path}")
        depths[folder] = glob_path
    print("\n".join([f"{k}: {v}" for k,v in depths.items()]))

    # ---- entrenar MLP por CSV ----
    ckpts = {}
    for depth_key, csv_path in depths.items():
        ds = SingleCSVTrajectoryDataset(csv_path, use_meta=args.use_meta)
        val_len = int(0.2*len(ds)); train_len=len(ds)-val_len
        train_traj, val_traj = random_split(ds, [train_len,val_len],
                                            generator=torch.Generator().manual_seed(18))
        x_train,y_train = flatten_to_points(train_traj)
        x_val,  y_val   = flatten_to_points(val_traj)
        train_ds = TensorDataset(x_train,y_train); val_ds = TensorDataset(x_val,y_val)

        model = MLP(ds.x_dim, hidden=128, out_dim=2, n_layers=3).to(device)
        ckpt_dir = out_root / depth_key; ckpt_dir.mkdir(exist_ok=True)
        train_mlp(model, train_ds, val_ds,
                  args.batch_size, device,
                  args.epochs, args.lr,
                  ckpt_dir, args.patience,
                  model_name=f"MLP_{depth_key}")
        ckpts[depth_key] = next(ckpt_dir.glob("MLP_*_best_ep*.pt"))

    # ---- entrenar DRS_depth combinado ----
    comb_samples=[]
    for csv_path in depths.values():
        comb_samples += SingleCSVTrajectoryDataset(csv_path, use_meta=args.use_meta).samples
    x_all,y_all=flatten_to_points(comb_samples)
    comb_ds = torch.utils.data.TensorDataset(x_all,y_all)
    val_len=int(0.2*len(comb_ds)); train_len=len(comb_ds)-val_len
    train_ds, val_ds = random_split(comb_ds,[train_len,val_len],
                                    generator=torch.Generator().manual_seed(18))

    model = MLP(x_all.shape[1], hidden=128, out_dim=2, n_layers=3).to(device)
    ckpt_dir = out_root/"combined"; ckpt_dir.mkdir(exist_ok=True)
    train_mlp(model, train_ds, val_ds,
              args.batch_size, device,
              args.epochs, args.lr,
              ckpt_dir, args.patience,
              model_name="DRS_depth")
    ckpts["combined"]=next(ckpt_dir.glob("DRS_depth_best_ep*.pt"))

    print("\nTodos los modelos entrenados:")
    for k,v in ckpts.items(): print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
