"""
train_mlp_drs.py
================
Entrena un modelo MLP (DRS_general) para predecir (x,y) a partir de
lecturas de sensores magnéticos.  Comparte el *mismo* conjunto de datos
y lógica de partición 80/20 que el script `train_anp_magnetic.py`.

Durante el entrenamiento se muestra una barra de progreso y se guarda
automáticamente el mejor checkpoint según la MAE de validación.

Uso:
-----
# Sin metadata
python train_mlp_drs.py --data-root /home/fernando/tesis/ANP/data/tasks/3x3x3 \
                        --epochs 10000 --batch-size 512 --patience 1000 --device cuda

# Con metadata en las entradas
python train_mlp_drs.py --data-root /home/fernando/tesis/ANP/data/tasks/3x3x3 \
                        --use-meta --epochs 10000 --batch-size 512 --patience 1000 --device cuda
"""
from __future__ import annotations
import argparse, signal, sys
from pathlib import Path
from typing import Tuple, List
from datetime import datetime
import csv, time
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
#  Utilidades de datos
# ----------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parent.parent))   # ANP root
from utils.data_loading import MagneticTrajectoryDataset, SENSOR_COLS, Y_COLS

# ----------------------------------------------------------------------
#  Dataset “flatten” a nivel de punto
# ----------------------------------------------------------------------
class MagneticPointDataset(Dataset):
    """Convierte cada punto de trayectoria en una muestra independiente."""
    def __init__(self, root: str | Path, *, use_meta: bool = False):
        base_ds = MagneticTrajectoryDataset(root, use_meta=use_meta)
        # Aplanamos todas las trayectorias
        xs, ys = [], []
        for sensors, coords in base_ds.samples:
            xs.append(sensors) # (L, x_dim)
            ys.append(coords) # (L, 2)
        self.x = torch.cat(xs, dim=0)
        self.y = torch.cat(ys, dim=0)
        self.x_dim = self.x.shape[1]
        print(f"[PointDataset] {len(self)} puntos | x_dim = {self.x_dim}")

    def __len__(self):  return self.x.shape[0]
    def __getitem__(self, idx):  return self.x[idx], self.y[idx]

# ----------------------------------------------------------------------
#  Modelo (MLP idéntico al usado internamente por el ANP)
# ----------------------------------------------------------------------
from model.anp_improved import MLP

# ----------------------------------------------------------------------
#  Métrica y utilidades
# ----------------------------------------------------------------------
def evaluate_mae(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    total_abs, total_n = 0.0, 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            total_abs += (y_pred - y).abs().sum().item()
            total_n   += y.numel()
    return total_abs / max(total_n, 1)

def save_mae_plot(train_hist: List[float], val_hist: List[float], out: Path):
    plt.figure()
    plt.plot(train_hist, label="Train MAE")
    plt.plot(val_hist, label="Val MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE (m)")
    plt.legend(); plt.tight_layout(); plt.savefig(out); plt.close()

# ----------------------------------------------------------------------
#  Bucle de entrenamiento
# ----------------------------------------------------------------------
def train(model: nn.Module,
          train_loader: DataLoader,
          val_loader: DataLoader,
          epochs: int,
          device: str,
          lr: float,
          ckpt_dir: Path, 
          patience: int = 500, 
          model_name: str = "DRS_general"):

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    optim  = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    best_mae = float("inf")
    best_ckpt: Path | None = None
    train_hist, val_hist = [], []
    interrupted = False
    epochs_no_improve = 0

    def _handler(_, __):
        nonlocal interrupted
        interrupted = True
        print("\nCTRL-C recibido; terminaré la época y pararé…")
    signal.signal(signal.SIGINT, _handler)

    bar = tqdm(range(1, epochs + 1), unit="epoch")
    t0 = time.time()
    for ep in bar:
        # ---- entrenamiento ----
        model.train()
        total_abs, total_n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device_type=device):
                y_pred = model(x)
                loss = nn.functional.l1_loss(y_pred, y)  # MAE
            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update(); optim.zero_grad(set_to_none=True)

            total_abs += loss.item() * y.numel()
            total_n   += y.numel()

        train_mae = total_abs / max(total_n, 1)
        train_hist.append(train_mae)

        # ---- validación ----
        val_mae = evaluate_mae(model, val_loader, device)
        val_hist.append(val_mae)

        bar.set_description(f"Ep {ep:03d}")
        bar.set_postfix(train=f"{train_mae:.4f}", val=f"{val_mae:.4f}", best=f"{best_mae:.4f}")

        # ---- actualizar early-stop ----
        if val_mae < best_mae - 1e-6:
            best_mae = val_mae
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # ---- barra ----
        bar.set_description(f"Ep {ep:03d}")
        bar.set_postfix(train=f"{train_mae:.4f}",
                        val=f"{val_mae:.4f}",
                        best=f"{best_mae:.4f}",
                        wait=f"{epochs_no_improve}/{patience}")

        # ---- early stop ----
        if epochs_no_improve >= patience:
            print(f"Early stop: {patience} épocas sin mejora")
            break

        if interrupted:
            break
    bar.close()
    save_mae_plot(train_hist, val_hist, ckpt_dir / "mae_curve.png")
    # -------- summary CSV --------
    elapsed = time.time() - t0
    summary_path = ckpt_dir / "training_summary.csv"
    header = ["run_datetime", "model", "epochs_ran", "best_epoch",
              "best_val_mae", "train_time_sec", "patience"]

    row = [datetime.now().isoformat(sep=" ", timespec="seconds"),
           model_name, ep, best_ckpt.stem.split("ep")[-1],
           round(best_mae, 6), int(elapsed), patience]

    write_header = not summary_path.exists()
    with open(summary_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

    print(f"Curva MAE y resumen guardados en {ckpt_dir}\n"
          f"Mejor checkpoint: {best_ckpt} | MAE {best_mae:.4f} | "
          f"Tiempo total {elapsed/60:.1f} min")

# ----------------------------------------------------------------------
#  CLI principal
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Entrenamiento MLP DRS_general")
    ap.add_argument("--data-root", default="/home/fernando/tesis/ANP/data/tasks")
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--use-meta", action="store_true", help="Añadir (profundidad,longitud,anchura) a las entradas")
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--n-layers",  type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ckpt-dir", default="./drs_runs")
    ap.add_argument("--patience", type=int, default=500, help="Épocas sin mejora antes de parar (early stop)")
    args = ap.parse_args()

    # ------- datos --------
    # 1) Dataset de trayectorias completo
    traj_ds = MagneticTrajectoryDataset(args.data_root, use_meta=args.use_meta)

    # 2) Mismo split 80/20 por TRAYECTORIAS
    val_len  = int(0.2 * len(traj_ds))
    train_len = len(traj_ds) - val_len
    train_traj, val_traj = random_split(
            traj_ds, [train_len, val_len],
            generator=torch.Generator().manual_seed(18))

    # 3) Aplanar cada parte a nivel de PUNTO
    def flatten(traj_subset):
        xs, ys = [], []
        for sensors, coords in traj_subset:
            xs.append(sensors)
            ys.append(coords)
        return torch.cat(xs), torch.cat(ys)

    x_train, y_train = flatten(train_traj)
    x_val,   y_val   = flatten(val_traj)

    # 4) Convertir a TensorDataset/DataLoader como prefieras
    train_ds = torch.utils.data.TensorDataset(x_train, y_train)
    val_ds   = torch.utils.data.TensorDataset(x_val,   y_val)


    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, drop_last=False)

    # ------- modelo -------
    model = MLP(traj_ds.x_dim, args.hidden_dim, 2, n_layers=args.n_layers).to(args.device)
    print(model)

    # ------- entrenamiento -
    train(model, train_loader, val_loader, args.epochs, args.device, args.lr, Path(args.ckpt_dir), args.patience, model_name="DRS_general")

if __name__ == "__main__":
    main()
