'''
train_anp_sensor_local.py

Trains one independent ANP per sensor (distributed / federated baseline).
Each sensor-model only sees the P time-point features that belong to its own
sensor, completely unaware of the other S-1 sensors.

This generates the "naïve distributed" reference point to compare against the
centralised model (input_dim = P*S, all sensors concatenated).

Data layout assumption (same as train_anp_topologies_masked.py):
  X  shape  (T, Dx)   with  Dx = P * S
  After reshape to (T, P, S):  sensor s  ←→  X_reshaped[:, :, s]   shape (T, P)

Usage examples:
  # All topologies, all sensors
  python train_anp_sensor_local.py \
    --data-dir  /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_low_variance \
    --num-sensors 10 --num-time-points 201 \
    --batch-size 16 --epochs 3000 --patience 100 \
    --topologies random,ellipsoidal,aligned \

  # Only topology "random", sensors 0 and 3
  python train_anp_sensor_local.py \
    --data-dir  /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_low_variance \
    --topologies random \
    --sensors 3,9 \
    --num-sensors 10 --num-time-points 201 \
    --batch-size 16 --epochs 3000 --patience 100
'''

import csv
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from src.models.anp import LatentModel
from src.utils.plots import plot_training_metrics


# ─────────────────────────────────────────────
# Dataset: wraps (X_full, Y) and extracts sensor s
# ─────────────────────────────────────────────

class SingleSensorDataset(Dataset):
    """Converts full-input data (T, P*S) → single-sensor view (T, P)."""

    def __init__(self, data, sensor_idx: int, num_time_points: int, num_sensors: int):
        self.sensor_idx = sensor_idx
        self.P = num_time_points
        self.S = num_sensors
        self.samples = []
        for x, y in data:
            # x: (T, P*S)
            x_t = torch.tensor(x, dtype=torch.float32)
            y_t = torch.tensor(y, dtype=torch.float32)
            # reshape to (T, P, S), take sensor s → (T, P)
            x_s = x_t.view(-1, self.P, self.S)[:, :, sensor_idx]  # (T, P)
            self.samples.append((x_s, y_t))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ─────────────────────────────────────────────
# Stats helpers
# ─────────────────────────────────────────────

def compute_y_stats(train_data):
    Y = np.concatenate([y for _, y in train_data], axis=0)
    y_mean = torch.tensor(Y.mean(axis=0), dtype=torch.float32)
    y_std  = torch.tensor(Y.std(axis=0) + 1e-6, dtype=torch.float32)
    return y_mean, y_std


def kl_beta(epoch, warmup_epochs=500):
    return min(1.0, float(epoch) / float(max(1, warmup_epochs)))


def sample_context_indices(total_points, context_size, mode="first", device="cpu", generator=None):
    if mode == "first":
        return torch.arange(context_size, device=device)
    if mode == "random":
        perm = torch.randperm(total_points, device=device, generator=generator)
        return perm[:context_size].sort().values
    raise ValueError(f"Unknown context sampling mode: {mode}")


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

def load_topology_data(data_dir, topology):
    topology_dir = os.path.join(data_dir, f"topology_{topology}")
    train_path    = os.path.join(topology_dir, "train_data.pkl")
    val_path      = os.path.join(topology_dir, "val_data.pkl")
    metadata_path = os.path.join(topology_dir, "metadata.pkl")

    if not all(os.path.exists(p) for p in [train_path, val_path, metadata_path]):
        print(f"Warning: Missing data files for topology {topology}")
        return None, None, None

    with open(train_path, "rb") as f:
        train_data = pickle.load(f)
    with open(val_path, "rb") as f:
        val_data = pickle.load(f)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    return train_data, val_data, metadata


# ─────────────────────────────────────────────
# Metrics persistence
# ─────────────────────────────────────────────

def save_all_metrics(train_loss, val_loss, train_mae, val_mae, experiment_dir,
                     train_nll=None, val_nll=None, train_kl=None, val_kl=None,
                     train_beta=None,
                     train_var_min=None, train_var_mean=None, train_var_max=None,
                     val_var_min=None,   val_var_mean=None,   val_var_max=None):
    metrics = dict(
        train_loss=train_loss, val_loss=val_loss,
        train_mae=train_mae,   val_mae=val_mae,
        train_nll=train_nll,   val_nll=val_nll,
        train_kl=train_kl,     val_kl=val_kl,
        train_beta=train_beta,
        train_var_min=train_var_min, train_var_mean=train_var_mean, train_var_max=train_var_max,
        val_var_min=val_var_min,     val_var_mean=val_var_mean,     val_var_max=val_var_max,
    )
    with open(os.path.join(experiment_dir, "metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)


# ─────────────────────────────────────────────
# Training loop (single sensor)
# ─────────────────────────────────────────────

def train_single_sensor(
    train_data, val_data,
    sensor_idx: int,
    save_dir: str,
    topology_name: str,
    num_sensors: int,
    num_time_points: int,
    batch_size: int = 8,
    epochs: int = 3000,
    patience: int = 200,
    device: str = "cuda",
    ctx_sample_mode: str = "first",
    kl_warmup_epochs: int = 500,
    num_hidden: int = 128,
    lr: float = 9e-4,
    weight_decay: float = 1e-4,
    save_checkpoints: bool = True,
):
    os.makedirs(save_dir, exist_ok=True)

    P = num_time_points
    input_dim  = P           # single-sensor ANP only sees P features
    output_dim = train_data[0][1].shape[-1]

    print(f"\n  [sensor {sensor_idx}] topology={topology_name}  "
          f"input_dim={input_dim}  output_dim={output_dim}  "
          f"train={len(train_data)}  val={len(val_data)}")

    # ── datasets ──
    train_ds = SingleSensorDataset(train_data, sensor_idx, P, num_sensors)
    val_ds   = SingleSensorDataset(val_data,   sensor_idx, P, num_sensors)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # ── y-normalisation stats (computed from full data, same target) ──
    y_mean, y_std = compute_y_stats(train_data)
    y_mean = y_mean.to(device)
    y_std  = y_std.to(device)

    # ── model ──
    model = LatentModel(num_hidden=num_hidden, input_dim=input_dim, output_dim=output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ── logging lists ──
    train_loss_list, val_loss_list = [], []
    train_mae_list,  val_mae_list  = [], []
    train_nll_list,  val_nll_list  = [], []
    train_kl_list,   val_kl_list   = [], []
    train_beta_list  = []
    train_var_min_list, train_var_mean_list, train_var_max_list = [], [], []
    val_var_min_list,   val_var_mean_list,   val_var_max_list   = [], [], []

    val_fracs = [0.1, 0.3, 0.5]

    best_val_mae    = float("inf")
    early_stop_ctr  = 0
    t_init          = time.time()

    pbar = tqdm(
        range(epochs),
        desc=f"  sensor {sensor_idx:02d}",
        unit="ep",
        ncols=160,
        leave=False,
    )

    for epoch in pbar:
        # ────── Train ──────
        model.train()
        t_loss = t_nll = t_kl = t_mae = 0.0
        t_vmin = t_vmean = t_vmax = 0.0
        beta = kl_beta(epoch, warmup_epochs=kl_warmup_epochs)

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)   # (B, T, P)
            y_batch = y_batch.to(device)    # (B, T, 3)
            B, T, _ = x_batch.shape

            # dynamic context size
            min_ctx = max(1, int(0.05 * T))
            max_ctx = min(int(0.95 * T), T - 1)
            ctx_size = torch.randint(min_ctx, max_ctx + 1, (1,), device=device).item() \
                if max_ctx > min_ctx else min_ctx

            ctx_idx = sample_context_indices(T, ctx_size, mode=ctx_sample_mode, device=device)
            tar_idx = torch.arange(T, device=device)

            y_norm = (y_batch - y_mean) / y_std

            ctx_x = x_batch[:, ctx_idx, :]
            ctx_y = y_norm[:, ctx_idx, :]
            tar_x = x_batch[:, tar_idx, :]
            tar_y = y_norm[:, tar_idx, :]

            y_hat, y_var, loss, kl, nll = model(ctx_x, ctx_y, tar_x, tar_y, beta=beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                non_ctx = torch.ones(T, dtype=torch.bool, device=device)
                non_ctx[ctx_idx] = False
                y_pred = y_hat * y_std + y_mean
                mae = F.l1_loss(y_pred[:, non_ctx, :], y_batch[:, non_ctx, :]).item()
                t_vmin  += y_var.min().item()
                t_vmean += y_var.mean().item()
                t_vmax  += y_var.max().item()

            t_loss += loss.item()
            t_nll  += nll.item()
            t_kl   += kl.item()
            t_mae  += mae

        n = len(train_loader)
        train_loss_list.append(t_loss / n)
        train_mae_list.append(t_mae / n)
        train_nll_list.append(t_nll / n)
        train_kl_list.append(t_kl / n)
        train_var_min_list.append(t_vmin / n)
        train_var_mean_list.append(t_vmean / n)
        train_var_max_list.append(t_vmax / n)
        train_beta_list.append(beta)

        # ────── Val ──────
        g = torch.Generator(device=device)
        g.manual_seed(1)
        model.eval()
        v_loss = v_nll = v_kl = v_mae = 0.0
        v_vmin = v_vmean = v_vmax = 0.0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                B, T, _ = x_batch.shape
                y_norm = (y_batch - y_mean) / y_std

                bloss = bmae = 0.0
                for frac in val_fracs:
                    ctx_size = max(1, min(T - 1, int(round(frac * T))))
                    ctx_idx  = sample_context_indices(T, ctx_size, mode=ctx_sample_mode,
                                                      device=device, generator=g)
                    tar_idx  = torch.arange(T, device=device)
                    non_ctx  = torch.ones(T, dtype=torch.bool, device=device)
                    non_ctx[ctx_idx] = False

                    ctx_x = x_batch[:, ctx_idx, :]
                    ctx_y = y_norm[:, ctx_idx,  :]
                    tar_x = x_batch[:, tar_idx, :]
                    tar_y = y_norm[:, tar_idx,  :]

                    y_hat, y_var, loss, kl, nll = model(ctx_x, ctx_y, tar_x, tar_y, beta=1.0)

                    y_pred = y_hat * y_std + y_mean
                    mae = F.l1_loss(y_pred[:, non_ctx, :], y_batch[:, non_ctx, :]).item()
                    bloss += loss.item()
                    bmae  += mae

                    v_nll  += nll.item()
                    v_kl   += kl.item()
                    v_vmin  += y_var.min().item()
                    v_vmean += y_var.mean().item()
                    v_vmax  += y_var.max().item()

                v_loss += bloss / len(val_fracs)
                v_mae  += bmae  / len(val_fracs)

        nv  = len(val_loader)
        nvf = nv * len(val_fracs)
        val_loss_list.append(v_loss / nv)
        val_mae_list.append(v_mae / nv)
        val_nll_list.append(v_nll / nvf)
        val_kl_list.append(v_kl  / nvf)
        val_var_min_list.append(v_vmin  / nvf)
        val_var_mean_list.append(v_vmean / nvf)
        val_var_max_list.append(v_vmax  / nvf)

        cur_val_mae = v_mae / nv
        if cur_val_mae < best_val_mae:
            best_val_mae   = cur_val_mae
            early_stop_ctr = 0
            if save_checkpoints:
                torch.save(
                    {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "sensor_idx": sensor_idx, "topology": topology_name,
                     "input_dim": input_dim, "output_dim": output_dim,
                     "num_hidden": num_hidden,
                     "num_time_points": P, "num_sensors": num_sensors},
                    os.path.join(save_dir, "best_checkpoint.pth.tar"),
                )
        else:
            early_stop_ctr += 1

        if early_stop_ctr >= patience:
            pbar.write(f"    Early stopping at epoch {epoch+1}")
            break

        pbar.set_postfix({
            "tMAE": f"{train_mae_list[-1]:.2f}",
            "vMAE": f"{cur_val_mae:.2f}",
            "best": f"{best_val_mae:.2f}",
            "ES":   early_stop_ctr,
        })

    # ── persist ──
    if save_checkpoints:
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "sensor_idx": sensor_idx, "topology": topology_name,
             "input_dim": input_dim, "output_dim": output_dim,
             "num_hidden": num_hidden,
             "num_time_points": P, "num_sensors": num_sensors},
            os.path.join(save_dir, "last_checkpoint.pth.tar"),
        )

    save_all_metrics(
        train_loss_list, val_loss_list, train_mae_list, val_mae_list, save_dir,
        train_nll=train_nll_list, val_nll=val_nll_list,
        train_kl=train_kl_list,   val_kl=val_kl_list,
        train_beta=train_beta_list,
        train_var_min=train_var_min_list, train_var_mean=train_var_mean_list,
        train_var_max=train_var_max_list,
        val_var_min=val_var_min_list, val_var_mean=val_var_mean_list,
        val_var_max=val_var_max_list,
    )

    metrics_file = os.path.join(save_dir, "metrics.pkl")
    output_plot  = os.path.join(save_dir, "training_curves.png")
    plot_training_metrics(metrics_file, output_plot)

    # CSV training log
    csv_path = os.path.join(save_dir, "training_log.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_nll", "train_kl", "beta",
                    "train_var_mean", "train_mae",
                    "val_loss", "val_nll", "val_kl",
                    "val_var_mean", "val_mae"])
        for e in range(len(train_loss_list)):
            w.writerow([
                e + 1,
                train_loss_list[e], train_nll_list[e], train_kl_list[e], train_beta_list[e],
                train_var_mean_list[e], train_mae_list[e],
                val_loss_list[e], val_nll_list[e], val_kl_list[e],
                val_var_mean_list[e], val_mae_list[e],
            ])

    elapsed = (time.time() - t_init) / 60
    with open(os.path.join(save_dir, "training_summary.txt"), "w") as f:
        f.write(f"Sensor-local ANP  —  topology: {topology_name},  sensor: {sensor_idx}\n")
        f.write("=" * 60 + "\n")
        f.write(f"  input_dim       : {input_dim}  (= P = num_time_points)\n")
        f.write(f"  output_dim      : {output_dim}\n")
        f.write(f"  train samples   : {len(train_data)}\n")
        f.write(f"  val samples     : {len(val_data)}\n")
        f.write(f"  best val MAE    : {best_val_mae:.6f}\n")
        f.write(f"  epochs trained  : {epoch + 1}\n")
        f.write(f"  training time   : {elapsed:.2f} min\n")

    return best_val_mae


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train one ANP per sensor (distributed baseline)")
    parser.add_argument("--data-dir",         type=str,   required=True)
    parser.add_argument("--save-dir",         type=str,   default=None,
                        help="Base output dir. Default: results/ANP_sensor_local")
    parser.add_argument("--topologies",       type=str,   default="aligned,ellipsoidal,random")
    parser.add_argument("--sensors",          type=str,   default=None,
                        help="Comma-separated sensor indices to train. Default: all sensors.")
    parser.add_argument("--num-sensors",      type=int,   default=10)
    parser.add_argument("--num-time-points",  type=int,   default=201)
    parser.add_argument("--batch-size",       type=int,   default=8)
    parser.add_argument("--epochs",           type=int,   default=3000)
    parser.add_argument("--patience",         type=int,   default=200)
    parser.add_argument("--device",           type=str,   default="cuda")
    parser.add_argument("--ctx-sample-mode",  type=str,   default="first",
                        choices=["first", "random"])
    parser.add_argument("--kl-warmup-epochs", type=int,   default=500)
    parser.add_argument("--num-hidden",       type=int,   default=128)
    parser.add_argument("--lr",               type=float, default=9e-4)

    args = parser.parse_args()

    topologies = [t.strip() for t in args.topologies.split(",") if t.strip()]
    all_sensors = list(range(args.num_sensors))
    sensors = (
        [int(s.strip()) for s in args.sensors.split(",") if s.strip()]
        if args.sensors else all_sensors
    )

    base = args.save_dir or os.path.join(os.getcwd(), "results", "ANP_sensor_local")

    global_results: dict = {}   # {(topo, sensor): best_mae}

    for topo in topologies:
        print(f"\n{'='*60}")
        print(f"Topology: {topo}")
        print(f"{'='*60}")

        train_data, val_data, _ = load_topology_data(args.data_dir, topo)
        if train_data is None:
            continue

        # Quick dimension check
        x0 = train_data[0][0]
        assert x0.shape[-1] == args.num_time_points * args.num_sensors, (
            f"Expected Dx={args.num_time_points * args.num_sensors}, "
            f"got {x0.shape[-1]}"
        )

        for s in sensors:
            save_dir = os.path.join(base, f"topology_{topo}", f"sensor_{s:02d}")
            best_mae = train_single_sensor(
                train_data        = train_data,
                val_data          = val_data,
                sensor_idx        = s,
                save_dir          = save_dir,
                topology_name     = topo,
                num_sensors       = args.num_sensors,
                num_time_points   = args.num_time_points,
                batch_size        = args.batch_size,
                epochs            = args.epochs,
                patience          = args.patience,
                device            = args.device,
                ctx_sample_mode   = args.ctx_sample_mode,
                kl_warmup_epochs  = args.kl_warmup_epochs,
                num_hidden        = args.num_hidden,
                lr                = args.lr,
            )
            global_results[(topo, s)] = best_mae

    # ── global summary ──
    summary_path = os.path.join(base, "summary_all_sensors.txt")
    with open(summary_path, "w") as f:
        f.write("Sensor-local ANP — best validation MAE per (topology, sensor)\n")
        f.write("=" * 60 + "\n")
        for topo in topologies:
            f.write(f"\nTopology: {topo}\n")
            for s in sensors:
                key = (topo, s)
                if key in global_results:
                    f.write(f"  sensor {s:02d}  :  {global_results[key]:.6f}\n")

    print(f"\nDone. Summary saved to {summary_path}")
    print("\nBest val MAE per (topology, sensor):")
    for topo in topologies:
        for s in sensors:
            key = (topo, s)
            if key in global_results:
                print(f"  {topo:<12}  sensor {s:02d}  :  {global_results[key]:.6f}")


if __name__ == "__main__":
    main()
