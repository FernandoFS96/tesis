"""
training/train_anp_magnetic.py
====================================================
Entrena un Attentive Neural Process (ANP) sobre trayectorias magnéticas,
usando un split 80/20 por trayectorias completas y calculando la métrica
**MAE** en entrenamiento y validación. Solo se guarda un checkpoint cuando
el MAE de validación mejora (el anterior se elimina).

Se puedes elegir entrenar el modelo **con o sin apagado de sensores** en las entradas,
usando el flag `--sensor-mask`:

  • Si añades `--sensor-mask`, se activará el enmascarado/apagado de sensores durante el entrenamiento.
  • Si NO añades `--sensor-mask`, los datos se usarán completos (sin apagado).

Ejemplos de uso:
-----------------
  # Entrenamiento **sin apagado** de sensores (por defecto)
  python train_anp_magnetic.py --data-root /home/fernando/tesis/ANP/data/tasks/3x3x3 --epochs 10000 --batch-size 10 --patience 2000 --device cuda

  # Entrenamiento **con apagado** de sensores (robustez a fallos)
  python train_anp_magnetic.py --data-root /home/fernando/tesis/ANP/data/trajectories --epochs 10000 --batch-size 20 --patience 2000 --device cuda --sensor-mask
"""

from __future__ import annotations

import argparse
import signal
from pathlib import Path
from typing import Tuple, List
from datetime import datetime
import csv, time
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))  # para importar ANP
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from functools import partial
from utils.data_loading import MagneticTrajectoryDataset, episodic_collate
from model.anp_improved import ANP, ANPConfig


# -----------------------------------------------------------------------------
#  Paso de optimización (calcula también MAE)
# -----------------------------------------------------------------------------

def loss_step(
    model: ANP,
    batch: Tuple[torch.Tensor, ...],
    scaler: torch.amp.GradScaler,
    optim: torch.optim.Optimizer,
    device: str,
    clip_grad: float = 1.0,
):
    x_c, y_c, x_t, y_t, ctx_mask, tgt_mask = (t.to(device) for t in batch)

    with torch.autocast(device_type=device, dtype=torch.float16 if device == "cuda" else torch.float32):
        dist, kl = model.forward(x_c, y_c, x_t, y_t,
                                 ctx_mask=ctx_mask, tgt_mask=tgt_mask)
        log_p = dist.log_prob(y_t).sum(-1)
        log_p = log_p.masked_fill(tgt_mask, 0.0)
        denom = (~tgt_mask).sum() + 1e-6
        log_p = log_p.sum() / denom
        loss = -(log_p - kl)

    # MAE
    abs_err = (dist.mean - y_t).abs().sum(-1)
    abs_err = abs_err.masked_fill(tgt_mask, 0.0)
    n_pts = (~tgt_mask).sum().item()
    mae = abs_err.sum().item() / max(n_pts, 1)

    # Backprop
    scaler.scale(loss).backward()
    scaler.unscale_(optim)
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
    scaler.step(optim)
    scaler.update()
    optim.zero_grad(set_to_none=True)

    return mae, n_pts

# -----------------------------------------------------------------------------
#  Evaluación (solo MAE)
# -----------------------------------------------------------------------------
@torch.inference_mode()
def evaluate_mae(model: ANP, loader: DataLoader, device: str) -> float:
    model.eval()
    total_abs, total_pts = 0.0, 0
    for x_c, y_c, x_t, y_t, ctx_m, tgt_m in loader:
        x_c, y_c, x_t, y_t = [t.to(device) for t in (x_c, y_c, x_t, y_t)]
        ctx_m, tgt_m = ctx_m.to(device), tgt_m.to(device)
        dist, _ = model.forward(x_c, y_c, x_t,
                                 ctx_mask=ctx_m, tgt_mask=tgt_m)
        abs_err = (dist.mean - y_t).abs().sum(-1)
        abs_err = abs_err.masked_fill(tgt_m, 0.0)
        total_abs += abs_err.sum().item()
        total_pts += (~tgt_m).sum().item()
    return total_abs / max(total_pts, 1)

# -----------------------------------------------------------------------------
#  Gráfico de evolución MAE
# -----------------------------------------------------------------------------

def save_mae_plot(train_hist: List[float], val_hist: List[float], out_path: Path):
    plt.figure()
    plt.plot(train_hist, label="Train MAE")
    plt.plot(val_hist, label="Val MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE (m)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# -----------------------------------------------------------------------------
#  Bucle de entrenamiento con validación + tqdm
# -----------------------------------------------------------------------------

def train(
    model: ANP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: str,
    lr: float,
    ckpt_dir: Path, 
    patience: int = 1000, 
    model_name: str = "ANP"
):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    best_mae = float("inf")
    best_ckpt: Path | None = None
    interrupted = False
    epochs_no_improve = 0

    train_hist: List[float] = []
    val_hist: List[float]   = []

    def _handler(_, __):
        nonlocal interrupted
        interrupted = True
        print("\nCTRL-C recibido; terminaré la época y pararé…")

    signal.signal(signal.SIGINT, _handler)

    bar = tqdm(range(1, epochs + 1), unit="epoch")
    t0 = time.time()
    for ep in bar:
        # ---------------- entrenamiento ----------------
        model.train()
        abs_accum, pts_accum = 0.0, 0
        for batch in train_loader:
            mae_batch, n_pts = loss_step(model, batch, scaler, optim, device)
            abs_accum += mae_batch * n_pts
            pts_accum += n_pts
        train_mae = abs_accum / max(pts_accum, 1)
        train_hist.append(train_mae)

        # ---------------- validación -------------------
        val_mae = evaluate_mae(model, val_loader, device)
        val_hist.append(val_mae)

        # ---------------- checkpoint -------------------
        if val_mae < best_mae - 1e-6:
            if best_ckpt and best_ckpt.exists():
                best_ckpt.unlink()
            best_mae = val_mae
            best_ckpt = ckpt_dir / f"anp_best_ep{ep:03d}.pt"
            torch.save({
                "model_state": model.state_dict(),
                "optim_state": optim.state_dict(),
                "scaler_state": scaler.state_dict(),
                "epoch": ep,
                "best_mae": best_mae,
            }, best_ckpt)
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

    # Guardar curva MAE
    plot_path = ckpt_dir / "mae_curve.png"
    save_mae_plot(train_hist, val_hist, plot_path)

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

# -----------------------------------------------------------------------------
#  CLI principal
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Entrenamiento ANP con MAE y validación 80/20")
    ap.add_argument("--data-root", default="/home/fernando/tesis/ANP/data/trajectories")
    ap.add_argument("--epochs", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--use-meta", action="store_true")
    ap.add_argument("--sensor-mask", action="store_true", help="Activar apagado de sensores")
    ap.add_argument("--extrapolation", action="store_true", help="If set, use sequential left-to-right context for extrapolation training.")
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--latent-dim", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--lr", type=float, default=9e-5)
    ap.add_argument("--ckpt-dir", default="./anp_runs")
    ap.add_argument("--patience", type=int, default=1000, help="Épocas sin mejora antes de parar (early stop)")
    args = ap.parse_args()

    # add mask to ckpt dir if sensor mask is used
    if args.sensor_mask:
        args.ckpt_dir = Path(args.ckpt_dir) / "sensor_mask"
    else:
        args.ckpt_dir = Path(args.ckpt_dir) / "no_sensor_mask"

    # add extrapolation or interpolation to ckpt dir
    if args.extrapolation:
        args.ckpt_dir = args.ckpt_dir / "extrapolation"
    else:
        args.ckpt_dir = args.ckpt_dir / "interpolation"

    # Dataset completo y split 80/20 ------------------------------------------------
    full_ds = MagneticTrajectoryDataset(args.data_root, use_meta=args.use_meta)
    val_len = int(0.2 * len(full_ds))
    train_len = len(full_ds) - val_len
    train_ds, val_ds = random_split(full_ds, [train_len, val_len],
                                    generator=torch.Generator().manual_seed(18))

    # DataLoader para entrenamiento y validación --------------------------------
    collate_fn = partial(
        episodic_collate,
        ctx_mode="sequential" if args.extrapolation else "random",
        # keep default min/max for interpolation; they are ignored in sequential
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Modelo -----------------------------------------------------------------------
    cfg = ANPConfig(
        x_dim=full_ds.x_dim,
        y_dim=2,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        n_heads=args.n_heads,
        sensor_mask=args.sensor_mask
    )
    model = ANP(cfg).to(args.device)
    print(cfg)

    train(model, train_loader, val_loader, args.epochs, args.device,
          args.lr, Path(args.ckpt_dir), args.patience, model_name="ANP_Magnetic_simplfied")


if __name__ == "__main__":
    main()
