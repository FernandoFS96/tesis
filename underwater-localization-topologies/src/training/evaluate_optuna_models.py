"""
evaluate_optuna_models.py
=========================
Example script showing how to use load_optuna_model.py to evaluate ANP and RANP models trained with Optuna HPO.

Loads both models, runs evaluation on the val set for each context fraction [0.2, 0.4, 0.6], and reports per-fraction and mean MAE.

Usage
-----
Run from the project root (underwater-localization-topologies/):

    python -m src.training.evaluate_optuna_models \
        --topology ellipsoidal \
        --data-dir data/data/data_processed_topologies_low_variance \
        --anp-dir  src/training/results/optuna/anp_masked_lowvar_ellipsoidal_v1/best_model \
        --ranp-dir src/training/results/optuna/ranp_masked_lowvar_ellipsoidal_v1/best_model \
        --device   cuda

Either --anp-dir or --ranp-dir can be omitted if only one model is available.
"""

from __future__ import annotations

import argparse
import os
import sys
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# -------------------------------------
# Helpers shared with training scripts
# -------------------------------------

def _load_topology_data(data_dir: str, topology: str):
    topology_dir = os.path.join(data_dir, f"topology_{topology}")
    train_path   = os.path.join(topology_dir, "train_data.pkl")
    val_path     = os.path.join(topology_dir, "val_data.pkl")
    test_path    = os.path.join(topology_dir, "test_data.pkl")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(
            f"Data not found for topology '{topology}' in {data_dir}.\n"
            f"Expected: {train_path}, {val_path}"
        )
    with open(train_path, "rb") as f:
        train_data = pickle.load(f)
    with open(val_path, "rb") as f:
        val_data = pickle.load(f)
    test_data = None
    if os.path.exists(test_path):
        with open(test_path, "rb") as f:
            test_data = pickle.load(f)
    return train_data, val_data, test_data


def _compute_y_stats(train_data):
    Y = np.concatenate([y for _, y in train_data], axis=0)
    y_mean = torch.tensor(Y.mean(axis=0), dtype=torch.float32)
    y_std  = torch.tensor(Y.std(axis=0) + 1e-6, dtype=torch.float32)
    return y_mean, y_std


def _compute_x_sensor_means(train_data, num_time_points: int, num_sensors: int):
    X = np.concatenate([x for x, _ in train_data], axis=0)
    X3 = X.reshape(X.shape[0], num_time_points, num_sensors)
    return X3.mean(axis=0).T  # (S, P)


def _apply_mask_and_append(x_batch, sensor_mask, x_means_SP,
                            num_time_points, num_sensors):
    """Mask dropped sensors and append the per-sensor binary mask feature.

    x_batch    : (B, T, Dx),  Dx = P*S
    sensor_mask: (B, S) float, 1=available
    x_means_SP : (S, P) tensor
    Returns    : (B, T, Dx+S)
    """
    B, T, Dx = x_batch.shape
    P, S = num_time_points, num_sensors
    assert Dx == P * S

    x4 = x_batch.view(B, T, P, S)
    # fill dropped sensors with their training mean
    mu = x_means_SP.T.view(1, 1, P, S).to(x_batch.device, dtype=x_batch.dtype)
    m  = sensor_mask.view(B, 1, 1, S)
    x4_masked = x4 * m + mu * (1.0 - m)
    x_masked  = x4_masked.reshape(B, T, Dx)

    mask_feat = sensor_mask.view(B, 1, S).expand(B, T, S)
    return torch.cat([x_masked, mask_feat], dim=-1)  # (B, T, Dx+S)


def _make_dataloader(data, batch_size: int):
    xs = torch.tensor(np.stack([x for x, _ in data]), dtype=torch.float32)
    ys = torch.tensor(np.stack([y for _, y in data]), dtype=torch.float32)
    return DataLoader(TensorDataset(xs, ys), batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    model_type: str,
    val_loader,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    x_means_SP: torch.Tensor,
    num_time_points: int,
    num_sensors: int,
    context_fracs: list,
    device: str | torch.device,
) -> dict[float, float]:
    """Evaluate *model* on *val_loader* for each context fraction.

    Returns a dict mapping context_frac → mean MAE (metres).
    """
    y_mean = y_mean.to(device)
    y_std  = y_std.to(device)
    x_means_SP = x_means_SP.to(device)

    mae_sums  = {f: 0.0 for f in context_fracs}
    n_batches = 0

    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)   # (B, T, Dx)
            y_batch = y_batch.to(device)   # (B, T, 3)
            B, T, _ = x_batch.shape

            # Use all sensors (no dropout at evaluation time)
            sensor_mask = torch.ones(B, num_sensors, device=device)
            x_aug = _apply_mask_and_append(
                x_batch, sensor_mask, x_means_SP, num_time_points, num_sensors
            )  # (B, T, Dx+S)

            y_norm = (y_batch - y_mean) / y_std

            for frac in context_fracs:
                ctx_size = max(1, min(T - 1, int(round(frac * T))))
                ctx_idx  = torch.arange(ctx_size, device=device)
                tar_idx  = torch.arange(ctx_size, T, device=device)

                context_y = y_norm[:, ctx_idx, :]
                target_y  = y_norm[:, tar_idx, :]

                if model_type == "anp":
                    context_x = x_aug[:, ctx_idx, :]
                    target_x  = x_aug[:, tar_idx, :]
                    y_pred_norm, *_ = model(context_x, context_y, target_x)
                elif model_type == "ranp":
                    y_pred_norm, *_ = model(
                        x_seq=x_aug,
                        context_indices=ctx_idx,
                        context_y=context_y,
                        target_indices=tar_idx,
                    )
                else:
                    raise ValueError(f"Unknown model_type: {model_type}")

                y_pred = y_pred_norm * y_std + y_mean
                mae = F.l1_loss(y_pred, y_batch[:, tar_idx, :], reduction="mean").item()
                mae_sums[frac] += mae

            n_batches += 1

    return {f: mae_sums[f] / n_batches for f in context_fracs}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate Optuna best ANP/RANP models")
    parser.add_argument("--topology",  default="ellipsoidal",
                        help="Topology name (e.g. ellipsoidal, aligned, random)")
    parser.add_argument("--data-dir",
                        default="data/data/data_processed_topologies_low_variance",
                        help="Path to the directory containing topology_<name>/ folders")
    parser.add_argument("--anp-dir",   default=None,
                        help="Path to ANP best_model/ dir (omit to skip ANP eval)")
    parser.add_argument("--ranp-dir",  default=None,
                        help="Path to RANP best_model/ dir (omit to skip RANP eval)")
    parser.add_argument("--device",    default="cpu",
                        help="Torch device: cpu | cuda | cuda:0 ...")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-sensors",     type=int, default=10)
    parser.add_argument("--num-time-points", type=int, default=201)
    args = parser.parse_args()

    if args.anp_dir is None and args.ranp_dir is None:
        parser.error("Provide at least one of --anp-dir or --ranp-dir")

    # ---- data ----------------------------------------------------------------
    print(f"Loading data for topology '{args.topology}' from {args.data_dir} ...")
    train_data, val_data, test_data = _load_topology_data(args.data_dir, args.topology)
    print(f"  train: {len(train_data)} trajectories | val: {len(val_data)} trajectories", end="")
    if test_data is not None:
        print(f" | test: {len(test_data)} trajectories")
    else:
        print(" | test: not found (skipping)")

    y_mean, y_std   = _compute_y_stats(train_data)
    x_means_SP      = torch.tensor(
        _compute_x_sensor_means(train_data, args.num_time_points, args.num_sensors),
        dtype=torch.float32,
    )
    val_loader  = _make_dataloader(val_data,  args.batch_size)
    test_loader = _make_dataloader(test_data, args.batch_size) if test_data is not None else None

    context_fracs = [0.2, 0.4, 0.6]

    # ---- load & evaluate each model -----------------------------------------
    from src.utils.load_optuna_model import load_optuna_best_model

    results = {}

    for label, model_dir in [("ANP", args.anp_dir), ("RANP", args.ranp_dir)]:
        if model_dir is None:
            continue

        print(f"\n{'='*60}")
        print(f"Loading {label} model from: {model_dir}")
        try:
            model, hparams, meta = load_optuna_best_model(
                best_model_dir=model_dir,
                topology=args.topology,
                model_type="auto",
                num_sensors=args.num_sensors,
                num_time_points=args.num_time_points,
                output_dim=3,
                device=args.device,
            )
        except FileNotFoundError as exc:
            print(f"  Skipping {label}: {exc}")
            continue

        model_type = "ranp" if label == "RANP" else "anp"
        n_params = sum(p.numel() for p in model.parameters())
        trial_num = meta["trial_number"] if meta else "?"
        trial_mae = meta.get("value", "?") if meta else "?"
        print(f"  Trial: {trial_num}  |  Optuna MAE: {trial_mae}  |  Params: {n_params:,}")
        print(f"  Hparams: {hparams}")

        eval_sets = [("val", val_loader)]
        if test_loader is not None:
            eval_sets.append(("test", test_loader))

        for split_name, loader in eval_sets:
            print(f"\nEvaluating {label} on {split_name} set ...")
            mae_by_frac = evaluate_model(
                model=model,
                model_type=model_type,
                val_loader=loader,
                y_mean=y_mean,
                y_std=y_std,
                x_means_SP=x_means_SP,
                num_time_points=args.num_time_points,
                num_sensors=args.num_sensors,
                context_fracs=context_fracs,
                device=args.device,
            )
            mean_mae = np.mean(list(mae_by_frac.values()))
            results[f"{label}_{split_name}"] = {"by_frac": mae_by_frac, "mean": mean_mae}

            print(f"\n  {label} [{split_name}] Results (topology={args.topology}):")
            for frac, mae in mae_by_frac.items():
                print(f"    ctx={int(frac*100):3d}%  →  MAE = {mae:.4f} m")
            print(f"    Mean MAE (avg over fracs) = {mean_mae:.4f} m")

    # ---- comparison summary --------------------------------------------------
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("Summary comparison:")
        print(f"  {'Model+Split':<14} | " + " | ".join(f"ctx={int(f*100)}%" for f in context_fracs) + " | Mean MAE")
        print(f"  {'-'*14}-+-" + "-+-".join("-"*7 for _ in context_fracs) + "-+----------")
        for key, res in results.items():
            row = f"  {key:<14} | "
            row += " | ".join(f"{res['by_frac'][f]:.4f}" for f in context_fracs)
            row += f" | {res['mean']:.4f} m"
            print(row)


if __name__ == "__main__":
    main()
