'''
Docstring for src.training.train_anp_topologies_masked

This script trains ANP models with sensor masking for each topology and logs detailed diagnostics.

Usage:
Using bernoulli dropout with 20% drop probability and filling masked sensors with training mean:
python train_anp_topologies_masked.py \
  --data-dir /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_low_variance \
  --batch-size 8 \
  --epochs 5000 \
  --ctx-sample-mode first \
  --patience 500 \
  --num-sensors 10 \
  --num-time-points 201 \
  --sensor-drop-mode bernoulli \
  --sensor-drop-p 0.2 \
  --mask-fill train_mean \
  --topologies aligned,ellipsoidal,random \

  Using k-uniform dropout with random k:
python train_anp_topologies_masked.py \
  --data-dir /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_low_variance \
  --batch-size 8 \
  --epochs 5000 \
  --ctx-sample-mode first \
  --patience 500 \
  --num-sensors 10 \
  --num-time-points 201 \
  --sensor-drop-mode k_uniform \
  --mask-fill train_mean \
  --topologies aligned,ellipsoidal,random
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
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.models.anp import LatentModel
from src.utils.nav_dataset import NavigationTrajectoryDataset
from src.utils.plots import plot_training_metrics


# ---------------------------
# Utils / stats
# ---------------------------

def compute_y_stats(train_data):
    Y = np.concatenate([y for _, y in train_data], axis=0)  # (N*T, 3)
    y_mean = torch.tensor(Y.mean(axis=0), dtype=torch.float32)
    y_std  = torch.tensor(Y.std(axis=0) + 1e-6, dtype=torch.float32)
    return y_mean, y_std


def compute_x_sensor_means(train_data, num_time_points: int, num_sensors: int):
    """
    train_data: list of (X: (T, Dx), Y: (T, 3))
    Dx must be P*S, layout interleaved by sensor consistent with:
      sensor s features are X[:, s::S]   (shape (T,P))
    Returns: mean per sensor shape (S,P) as numpy
    """
    X = np.concatenate([x for x, _ in train_data], axis=0)  # (N*T, Dx)
    Dx = X.shape[1]
    P, S = num_time_points, num_sensors
    assert Dx == P * S, f"Dx={Dx} but expected P*S={P*S}"

    X3 = X.reshape(X.shape[0], P, S)  # (N*T, P, S) assuming interleaved layout
    mean_PS = X3.mean(axis=0)         # (P, S)
    return mean_PS.T                  # (S, P)


def kl_beta(epoch, warmup_epochs=500):
    return min(1.0, float(epoch) / float(max(1, warmup_epochs)))


def sample_context_indices(total_points, context_size, mode="first", device="cpu", generator=None):
    if mode == "first":
        return torch.arange(context_size, device=device)
    if mode == "random":
        perm = torch.randperm(total_points, device=device, generator=generator)
        return perm[:context_size].sort().values
    raise ValueError(f"Unknown context sampling mode: {mode}")


# ---------------------------
# Masking / dropout
# ---------------------------

def sample_sensor_mask(B: int, S: int, mode: str, p_drop: float, device: torch.device):
    """
    Returns float mask (B,S) in {0,1}, where 1=available.
    Ensures at least one sensor available per sample.
    """
    if mode == "bernoulli":
        keep = (torch.rand(B, S, device=device) > p_drop)
    elif mode == "k_uniform":
        keep = torch.zeros(B, S, dtype=torch.bool, device=device)
        for b in range(B):
            k = torch.randint(1, S + 1, (1,), device=device).item()
            idx = torch.randperm(S, device=device)[:k]
            keep[b, idx] = True
    else:
        raise ValueError(f"Unknown sensor_drop_mode: {mode}")

    # ensure at least one sensor on
    all_off = ~keep.any(dim=1)
    if all_off.any():
        idx = torch.randint(0, S, (all_off.sum().item(),), device=device)
        keep[all_off, idx] = True

    return keep.float()


def apply_sensor_dropout_and_append_mask(
    x_batch: torch.Tensor,          # (B,T,Dx)
    sensor_mask: torch.Tensor,      # (B,S) float
    x_means_SP: torch.Tensor,       # (S,P) float
    num_time_points: int,
    num_sensors: int,
    fill: str = "train_mean",       # train_mean|zero
):
    """
    Masks full sensors in x_batch and appends explicit per-sensor mask features.
    Output: (B,T,Dx+S)
    """
    B, T, Dx = x_batch.shape
    P, S = num_time_points, num_sensors
    assert Dx == P * S, f"x_batch Dx={Dx} but expected {P*S}"

    x4 = x_batch.view(B, T, P, S)  # (B,T,P,S)

    if fill == "zero":
        fill_val = torch.zeros((B, T, P, S), device=x_batch.device, dtype=x_batch.dtype)
    elif fill == "train_mean":
        # x_means_SP: (S,P) -> (P,S) then to (1,1,P,S)
        mu = x_means_SP.T.view(1, 1, P, S).to(device=x_batch.device, dtype=x_batch.dtype)
        fill_val = mu.expand(B, T, P, S)
    else:
        raise ValueError(f"Unknown mask_fill: {fill}")

    m = sensor_mask.view(B, 1, 1, S)  # (B,1,1,S)
    x4_masked = x4 * m + fill_val * (1.0 - m)
    x_masked = x4_masked.reshape(B, T, Dx)

    # append explicit mask features per time step
    mask_feat = sensor_mask.view(B, 1, S).expand(B, T, S)
    x_aug = torch.cat([x_masked, mask_feat], dim=-1)  # (B,T,Dx+S)
    return x_aug


def load_topology_data(data_dir, topology):
    topology_dir = os.path.join(data_dir, f"topology_{topology}")
    train_path = os.path.join(topology_dir, "train_data.pkl")
    val_path   = os.path.join(topology_dir, "val_data.pkl")
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


# ---------------------------
# Saving / plotting (igual que tu script)
# ---------------------------

def save_all_metrics(train_loss, val_loss, train_mae, val_mae,
                     experiment_dir,
                     train_nll=None, val_nll=None,
                     train_kl=None,  val_kl=None,
                     train_beta=None,
                     train_var_min=None, train_var_mean=None, train_var_max=None,
                     val_var_min=None,   val_var_mean=None,   val_var_max=None,
                     train_nll_nonctx=None, val_nll_nonctx=None):
    metrics = {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_mae': train_mae,
        'val_mae': val_mae,
        'train_nll': train_nll,
        'val_nll': val_nll,
        'train_kl': train_kl,
        'val_kl': val_kl,
        'train_beta': train_beta,
        'train_var_min': train_var_min,
        'train_var_mean': train_var_mean,
        'train_var_max': train_var_max,
        'val_var_min': val_var_min,
        'val_var_mean': val_var_mean,
        'val_var_max': val_var_max,
        'train_nll_nonctx': train_nll_nonctx,
        'val_nll_nonctx': val_nll_nonctx,
    }
    with open(os.path.join(experiment_dir, "metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)


def plot_anp_diagnostics(save_dir,
                         train_nll, val_nll,
                         train_kl, val_kl,
                         betas,
                         train_var_mean, val_var_mean,
                         train_var_min=None, train_var_max=None,
                         val_var_min=None,   val_var_max=None,
                         train_nll_nonctx=None, val_nll_nonctx=None):

    epochs = np.arange(1, len(train_nll) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_nll, label="train NLL")
    plt.plot(epochs, val_nll,   label="val NLL")
    if train_nll_nonctx is not None and val_nll_nonctx is not None:
        plt.plot(epochs, train_nll_nonctx, label="train NLL (non-ctx)", linestyle="--")
        plt.plot(epochs, val_nll_nonctx,   label="val NLL (non-ctx)",   linestyle="--")
    plt.plot(epochs, train_kl,  label="train KL")
    plt.plot(epochs, val_kl,    label="val KL")
    plt.plot(epochs, betas,     label="beta", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("ANP diagnostics: NLL / KL / beta")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_diagnostics_nll_kl_beta.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_var_mean, label="train var mean")
    plt.plot(epochs, val_var_mean,   label="val var mean")
    if train_var_min is not None and train_var_max is not None:
        plt.plot(epochs, train_var_min, label="train var min", linestyle="--")
        plt.plot(epochs, train_var_max, label="train var max", linestyle="--")
    if val_var_min is not None and val_var_max is not None:
        plt.plot(epochs, val_var_min, label="val var min", linestyle=":")
        plt.plot(epochs, val_var_max, label="val var max", linestyle=":")
    plt.xlabel("Epoch")
    plt.ylabel("Variance (normalized space)")
    plt.title("ANP diagnostics: predicted variance stats")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_diagnostics_variance.png"), dpi=150)
    plt.close()


# ---------------------------
# Training core
# ---------------------------

def train_anp_topology_masked(
    train_data, val_data, save_dir, topology_name,
    batch_size=8, epochs=5000, patience=200, device="cuda",
    ctx_sample_mode="first",
    num_sensors=10,
    num_time_points=201,
    sensor_drop_mode="bernoulli",
    sensor_drop_p=0.2,
    mask_fill="train_mean",
    mask_in_val=False,
    kl_warmup_epochs=500,
    num_hidden=128,
    lr=8e-4,
    weight_decay=1e-4,
    trial=None,
    report_every=25,
):
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nTraining MASKED ANP for topology: {topology_name}")
    print(f"  Training set size: {len(train_data)} trajectories")
    print(f"  Validation set size: {len(val_data)} trajectories")
    print(f"  X shape: {train_data[0][0].shape}, Y shape: {train_data[0][1].shape}")
    print(f"  Masking: drop_mode={sensor_drop_mode}, p_drop={sensor_drop_p}, fill={mask_fill}, mask_in_val={mask_in_val}")
    print(f"  Sensors: S={num_sensors}, time_points P={num_time_points}")

    x0, y0 = train_data[0]
    input_dim_old = x0.shape[-1]        # Dx = P*S
    output_dim = y0.shape[-1]
    assert input_dim_old == num_time_points * num_sensors, (
        input_dim_old, num_time_points * num_sensors
    )
    input_dim_new = input_dim_old + num_sensors  # append mask

    # datasets / loaders
    train_dataset = NavigationTrajectoryDataset(train_data)
    val_dataset = NavigationTrajectoryDataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # stats
    y_mean, y_std = compute_y_stats(train_data)
    y_mean = y_mean.to(device)
    y_std  = y_std.to(device)

    x_means_np = compute_x_sensor_means(train_data, num_time_points, num_sensors)  # (S,P)
    x_means_SP = torch.tensor(x_means_np, dtype=torch.float32, device=device)

    # model
    model = LatentModel(num_hidden=num_hidden, input_dim=input_dim_new, output_dim=output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # logs
    best_val_mae = float("inf")
    early_stop_counter = 0

    train_loss_list, val_loss_list = [], []
    train_mae_list,  val_mae_list  = [], []
    train_nll_list,  val_nll_list  = [], []
    train_kl_list,   val_kl_list   = [], []
    train_beta_list  = []
    train_var_min_list, train_var_mean_list, train_var_max_list = [], [], []
    val_var_min_list,   val_var_mean_list,   val_var_max_list   = [], [], []
    train_nll_nonctx_list, val_nll_nonctx_list = [], []

    # fixed context fractions for validation
    val_fracs = [0.1, 0.3, 0.5]

    t_init = time.time()
    pbar = tqdm(range(epochs), desc=f"[ANP-MASKED-{topology_name}]", unit="epoch", ncols=200)

    for epoch in pbar:
        # -------------------
        # Train
        # -------------------
        model.train()
        train_loss, train_mae = 0.0, 0.0
        train_nll, train_kl = 0.0, 0.0
        train_var_min, train_var_mean, train_var_max = 0.0, 0.0, 0.0
        train_nll_nonctx = 0.0
        train_k_active = 0.0  # avg active sensors (debug)

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            beta = kl_beta(epoch, warmup_epochs=kl_warmup_epochs)

            B = x_batch.size(0)
            T = x_batch.size(1)

            # sample sensor mask per sample in batch
            sensor_mask = sample_sensor_mask(B, num_sensors, sensor_drop_mode, sensor_drop_p, device=device)
            train_k_active += sensor_mask.sum(dim=1).mean().item()

            # apply dropout + append explicit mask
            x_batch_aug = apply_sensor_dropout_and_append_mask(
                x_batch, sensor_mask, x_means_SP,
                num_time_points=num_time_points, num_sensors=num_sensors,
                fill=mask_fill
            )  # (B,T,Dx+S)

            # dynamic context size
            total_points = T
            min_context = max(1, int(0.05 * total_points))
            max_context = min(int(0.95 * total_points), total_points - 1)
            context_size = torch.randint(min_context, max_context + 1, (1,), device=device).item() \
                if max_context > min_context else min_context

            context_indices = sample_context_indices(
                total_points, context_size, mode=ctx_sample_mode, device=device
            )
            target_indices = torch.arange(total_points, device=device)

            # normalize Y
            y_batch_raw = y_batch
            y_batch_norm = (y_batch - y_mean) / y_std

            context_x = x_batch_aug[:, context_indices, :]
            context_y = y_batch_norm[:, context_indices, :]
            target_x  = x_batch_aug[:, target_indices, :]
            target_y  = y_batch_norm[:, target_indices, :]

            # forward
            y_pred_mean_norm, y_pred_var_norm, loss, kl, nll = model(
                context_x, context_y, target_x, target_y, beta=beta
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # diagnostics
            with torch.no_grad():
                train_var_min  += y_pred_var_norm.min().item()
                train_var_mean += y_pred_var_norm.mean().item()
                train_var_max  += y_pred_var_norm.max().item()
                train_nll += nll.item()
                train_kl  += kl.item()

                non_ctx_mask = torch.ones(total_points, dtype=torch.bool, device=device)
                non_ctx_mask[context_indices] = False

                nll_pointwise = 0.5 * torch.log(2 * torch.pi * y_pred_var_norm) \
                                + 0.5 * ((target_y - y_pred_mean_norm) ** 2) / y_pred_var_norm
                train_nll_nonctx += nll_pointwise[:, non_ctx_mask, :].mean().item()

                y_pred_mean = y_pred_mean_norm * y_std + y_mean
                mae = F.l1_loss(
                    y_pred_mean[:, non_ctx_mask, :],
                    y_batch_raw[:, non_ctx_mask, :],
                    reduction="mean"
                ).item()

            train_loss += loss.item()
            train_mae  += mae

        # avg over batches
        train_loss /= len(train_loader)
        train_mae  /= len(train_loader)
        train_k_active /= len(train_loader)

        train_loss_list.append(train_loss)
        train_mae_list.append(train_mae)

        train_nll /= len(train_loader)
        train_kl  /= len(train_loader)
        train_var_min  /= len(train_loader)
        train_var_mean /= len(train_loader)
        train_var_max  /= len(train_loader)
        train_nll_nonctx /= len(train_loader)

        train_nll_list.append(train_nll)
        train_kl_list.append(train_kl)
        train_var_min_list.append(train_var_min)
        train_var_mean_list.append(train_var_mean)
        train_var_max_list.append(train_var_max)
        train_nll_nonctx_list.append(train_nll_nonctx)
        train_beta_list.append(beta)

        # -------------------
        # Val
        # -------------------
        g = torch.Generator(device=device)
        g.manual_seed(1)
        model.eval()
        val_loss, val_mae = 0.0, 0.0
        val_nll, val_kl = 0.0, 0.0
        val_var_min, val_var_mean, val_var_max = 0.0, 0.0, 0.0
        val_nll_nonctx = 0.0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                B = x_batch.size(0)
                T = x_batch.size(1)

                # always append mask; optionally apply dropout in val
                if mask_in_val:
                    sensor_mask = sample_sensor_mask(B, num_sensors, sensor_drop_mode, sensor_drop_p, device=device)
                else:
                    sensor_mask = torch.ones((B, num_sensors), device=device)

                x_batch_aug = apply_sensor_dropout_and_append_mask(
                    x_batch, sensor_mask, x_means_SP,
                    num_time_points=num_time_points, num_sensors=num_sensors,
                    fill=mask_fill
                )

                y_batch_raw = y_batch
                y_batch_norm = (y_batch - y_mean) / y_std
                total_points = T

                batch_loss = 0.0
                batch_mae = 0.0

                for frac in val_fracs:
                    context_size = max(1, min(total_points - 1, int(round(frac * total_points))))
                    ctx_idx = sample_context_indices(
                        total_points, context_size, mode=ctx_sample_mode, device=device, generator=g
                    )
                    tar_idx = torch.arange(total_points, device=device)

                    context_x = x_batch_aug[:, ctx_idx, :]
                    context_y = y_batch_norm[:, ctx_idx, :]
                    target_x  = x_batch_aug[:, tar_idx, :]
                    target_y  = y_batch_norm[:, tar_idx, :]

                    y_pred_mean_norm, y_pred_var_norm, loss, kl, nll = model(
                        context_x, context_y, target_x, target_y, beta=1.0
                    )

                    non_ctx_mask = torch.ones(total_points, dtype=torch.bool, device=device)
                    non_ctx_mask[ctx_idx] = False

                    y_pred_mean = y_pred_mean_norm * y_std + y_mean
                    mae = F.l1_loss(
                        y_pred_mean[:, non_ctx_mask, :],
                        y_batch_raw[:, non_ctx_mask, :],
                        reduction="mean"
                    ).item()

                    batch_loss += loss.item()
                    batch_mae  += mae

                    val_nll += nll.item()
                    val_kl  += kl.item()
                    val_var_min  += y_pred_var_norm.min().item()
                    val_var_mean += y_pred_var_norm.mean().item()
                    val_var_max  += y_pred_var_norm.max().item()

                    nll_pointwise = 0.5 * torch.log(2 * torch.pi * y_pred_var_norm) \
                                    + 0.5 * ((target_y - y_pred_mean_norm) ** 2) / y_pred_var_norm
                    val_nll_nonctx += nll_pointwise[:, non_ctx_mask, :].mean().item()

                val_loss += (batch_loss / len(val_fracs))
                val_mae  += (batch_mae  / len(val_fracs))

        val_loss /= len(val_loader)
        val_mae  /= len(val_loader)

        val_loss_list.append(val_loss)
        val_mae_list.append(val_mae)

        den = len(val_loader) * len(val_fracs)
        val_nll /= den
        val_kl  /= den
        val_var_min  /= den
        val_var_mean /= den
        val_var_max  /= den
        val_nll_nonctx /= den

        val_nll_list.append(val_nll)
        val_kl_list.append(val_kl)
        val_var_min_list.append(val_var_min)
        val_var_mean_list.append(val_var_mean)
        val_var_max_list.append(val_var_max)
        val_nll_nonctx_list.append(val_nll_nonctx)

        # Optuna pruning
        if trial is not None and (epoch % report_every == 0):
            import optuna
            trial.report(val_mae, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # early stopping
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            early_stop_counter = 0
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
                       os.path.join(save_dir, 'best_checkpoint.pth.tar'))
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

        pbar.set_postfix({
            'Loss': f"{train_loss:.2f}",
            'NLL': f"{train_nll:.2f}",
            'KL': f"{train_kl:.2f}",
            #'β': f"{beta:.2f}",
            #'k_on': f"{train_k_active:.2f}/{num_sensors}",
            #'varμ': f"{train_var_mean:.2e}",
            'MAE': f"{train_mae:.2f}",
            'Val MAE': f"{val_mae:.2f}",
            'Best': f"{best_val_mae:.2f}",
            'ES': f"{early_stop_counter}"
        })

    # save final
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
               os.path.join(save_dir, 'last_checkpoint.pth.tar'))

    save_all_metrics(
        train_loss_list, val_loss_list, train_mae_list, val_mae_list, save_dir,
        train_nll=train_nll_list, val_nll=val_nll_list,
        train_kl=train_kl_list, val_kl=val_kl_list,
        train_beta=train_beta_list,
        train_var_min=train_var_min_list, train_var_mean=train_var_mean_list, train_var_max=train_var_max_list,
        val_var_min=val_var_min_list, val_var_mean=val_var_mean_list, val_var_max=val_var_max_list,
        train_nll_nonctx=train_nll_nonctx_list, val_nll_nonctx=val_nll_nonctx_list
    )

    plot_anp_diagnostics(
        save_dir,
        train_nll_list, val_nll_list,
        train_kl_list, val_kl_list,
        train_beta_list,
        train_var_mean_list, val_var_mean_list,
        train_var_min=train_var_min_list, train_var_max=train_var_max_list,
        val_var_min=val_var_min_list, val_var_max=val_var_max_list,
        train_nll_nonctx=train_nll_nonctx_list, val_nll_nonctx=val_nll_nonctx_list
    )

    # CSV log
    csv_path = os.path.join(save_dir, "training_log.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "epoch",
            "train_loss","train_nll","train_nll_nonctx","train_kl","beta","train_var_min","train_var_mean","train_var_max","train_mae",
            "val_loss","val_nll","val_nll_nonctx","val_kl","val_var_min","val_var_mean","val_var_max","val_mae"
        ])
        for e in range(len(train_loss_list)):
            w.writerow([
                e+1,
                train_loss_list[e], train_nll_list[e], train_nll_nonctx_list[e], train_kl_list[e], train_beta_list[e],
                train_var_min_list[e], train_var_mean_list[e], train_var_max_list[e], train_mae_list[e],
                val_loss_list[e], val_nll_list[e], val_nll_nonctx_list[e], val_kl_list[e],
                val_var_min_list[e], val_var_mean_list[e], val_var_max_list[e], val_mae_list[e],
            ])

    # plot training curves
    metrics_file = os.path.join(save_dir, 'metrics.pkl')
    output_plot = os.path.join(save_dir, 'training_curves.png')
    plot_training_metrics(metrics_file, output_plot)

    # summary
    with open(os.path.join(save_dir, 'training_summary.txt'), 'w') as f:
        f.write(f"ANP MASKED Training Summary - Topology: {topology_name}\n")
        f.write("="*60 + "\n")
        f.write(f"Training samples: {len(train_data)} trajectories\n")
        f.write(f"Validation samples: {len(val_data)} trajectories\n")
        f.write(f"Best validation MAE: {best_val_mae:.6f}\n")
        f.write(f"Final epoch: {min(epoch+1, epochs)}\n")
        f.write(f"Early stopping counter: {early_stop_counter}/{patience}\n")
        f.write(f"Training time: {(time.time() - t_init)/60:.2f} minutes\n")
        f.write("\nMasking config:\n")
        f.write(f"  num_sensors: {num_sensors}\n")
        f.write(f"  num_time_points: {num_time_points}\n")
        f.write(f"  sensor_drop_mode: {sensor_drop_mode}\n")
        f.write(f"  sensor_drop_p: {sensor_drop_p}\n")
        f.write(f"  mask_fill: {mask_fill}\n")
        f.write(f"  mask_in_val: {mask_in_val}\n")

    print(f"  Best validation MAE: {best_val_mae:.6f}")
    return best_val_mae


# ---------------------------
# Main: train one model per topology
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Base output directory. If None, uses <cwd>/results/ANP_topologies_masked/<run_name>")
    parser.add_argument("--topologies", type=str, default="aligned,ellipsoidal,random")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--patience", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ctx-sample-mode", type=str, default="first", choices=["first", "random"])

    # masking params
    parser.add_argument("--num-sensors", type=int, default=10)
    parser.add_argument("--num-time-points", type=int, default=201)
    parser.add_argument("--sensor-drop-mode", type=str, default="bernoulli", choices=["bernoulli", "k_uniform"])
    parser.add_argument("--sensor-drop-p", type=float, default=0.2)
    parser.add_argument("--mask-fill", type=str, default="train_mean", choices=["train_mean", "zero"])
    parser.add_argument("--mask-in-val", action="store_true")
    parser.add_argument("--kl-warmup-epochs", type=int, default=500)

    args = parser.parse_args()

    topologies = [t.strip() for t in args.topologies.split(",") if t.strip()]
    run_name = f"masked_drop{args.sensor_drop_mode}_p{args.sensor_drop_p}_{args.mask_fill}_{args.ctx_sample_mode}"

    if args.save_dir is None:
        base = os.path.join(os.getcwd(), "results", "ANP_topologies_masked", run_name)
    else:
        base = os.path.join(args.save_dir, run_name)

    os.makedirs(base, exist_ok=True)

    results = {}

    for topo in topologies:
        train_data, val_data, _ = load_topology_data(args.data_dir, topo)
        if train_data is None:
            continue

        save_dir = os.path.join(base, f"topology_{topo}")
        best_mae = train_anp_topology_masked(
            train_data=train_data,
            val_data=val_data,
            save_dir=save_dir,
            topology_name=topo,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            device=args.device,
            ctx_sample_mode=args.ctx_sample_mode,
            num_sensors=args.num_sensors,
            num_time_points=args.num_time_points,
            sensor_drop_mode=args.sensor_drop_mode,
            sensor_drop_p=args.sensor_drop_p,
            mask_fill=args.mask_fill,
            mask_in_val=args.mask_in_val,
            kl_warmup_epochs=args.kl_warmup_epochs,
        )
        results[topo] = best_mae

    # write global summary
    with open(os.path.join(base, "summary_all_topologies.txt"), "w") as f:
        f.write("Best validation MAE per topology (MASKED training)\n")
        f.write("="*60 + "\n")
        for topo, mae in results.items():
            f.write(f"{topo:<12} : {mae:.6f}\n")

    print("\nDone. Results:")
    for topo, mae in results.items():
        print(f"  {topo:<12} : {mae:.6f}")


if __name__ == "__main__":
    main()
