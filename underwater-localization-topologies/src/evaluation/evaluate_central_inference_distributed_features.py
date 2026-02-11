#!/usr/bin/env python3
"""
Central inference + distributed acquisition/features (node = sensor)

Each sensor/node provides its local feature block x^(s)(t) of length num_time_points (201).
A fusion center reconstructs x_full(t) in R^(num_time_points*num_sensors) and runs ONE ANP inference.

Also reports:
- MAE (global) on non-context targets
- comm_features: KiB to send local features to the fusion center
- comm_outputs: KiB to send predictive params (mu,var) if you wanted output-level fusion
  (useful to compare comm budgets)

This matches an edge acquisition + central analytics architecture. :contentReference[oaicite:1]{index=1}
"""

import argparse
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.models.anp import LatentModel


class Timer:
    def __init__(self, device: torch.device):
        self.device = device
        self.t0: Optional[float] = None
        self.dt = 0.0

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        if self.t0 is None:
            raise RuntimeError("Timer used without being started.")
        self.dt = time.perf_counter() - self.t0


def load_topology_test(data_dir: Path, topology: str):
    topo_dir = data_dir / f"topology_{topology}"
    test_path = topo_dir / "test_data.pkl"
    meta_path = topo_dir / "metadata.pkl"
    with open(test_path, "rb") as f:
        test_data = pickle.load(f)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    theta_groups: Dict[float, List] = {}
    for sample, theta in zip(test_data, meta["test_thetas"]):
        theta_groups.setdefault(theta, []).append(sample)
    return theta_groups, sorted(theta_groups.keys())


def load_y_stats(data_dir: Path, topology: str, device: torch.device):
    topo_dir = data_dir / f"topology_{topology}"
    train_path = topo_dir / "train_data.pkl"
    with open(train_path, "rb") as f:
        train_data = pickle.load(f)
    Y = np.concatenate([y for _, y in train_data], axis=0)
    y_mean = torch.tensor(Y.mean(axis=0), dtype=torch.float32, device=device)
    y_std = torch.tensor(Y.std(axis=0) + 1e-6, dtype=torch.float32, device=device)
    return y_mean, y_std


def normalize_y(y, y_mean, y_std):
    return (y - y_mean.view(1, 1, -1)) / y_std.view(1, 1, -1)


def denormalize_y(y_norm, y_mean, y_std):
    return y_norm * y_std.view(1, 1, -1) + y_mean.view(1, 1, -1)


def sample_context_indices(T: int, n_ctx: int, mode: str, gen: torch.Generator, device: torch.device):
    n_ctx = max(1, min(n_ctx, T))
    if mode == "first":
        return torch.arange(n_ctx, device=device)
    if mode == "random":
        perm = torch.randperm(T, generator=gen, device=device)
        return perm[:n_ctx].sort().values
    raise ValueError(mode)


def load_model(anp_result_dir: Path, topology: str, input_dim: int, output_dim: int, device: torch.device):
    ckpt_path = anp_result_dir / f"ANP_{topology}" / "best_checkpoint.pth.tar"
    model = LatentModel(num_hidden=128, input_dim=input_dim, output_dim=output_dim)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    return model.to(device).eval()


def anp_predict(model, cx, cy, tx, mc_samples: int, device: torch.device):
    means, vars_ = [], []
    with Timer(device) as tmr:
        for _ in range(max(1, mc_samples)):
            out = model(cx, cy, tx)
            # robust unpack
            if isinstance(out, (tuple, list)):
                if len(out) >= 2 and torch.is_tensor(out[0]) and torch.is_tensor(out[1]):
                    m, v = out[0], out[1]
                elif len(out) >= 1 and isinstance(out[0], (tuple, list)) and len(out[0]) >= 2:
                    m, v = out[0][0], out[0][1]
                else:
                    raise TypeError(f"Unexpected model output tuple structure: {type(out)} len={len(out)}")
            elif isinstance(out, dict):
                m, v = out["mean"], out["var"]
            else:
                raise TypeError(f"Unexpected model output type: {type(out)}")
            means.append(m); vars_.append(v)
    mean = torch.stack(means, dim=0).mean(dim=0)
    var = torch.stack(vars_, dim=0).mean(dim=0)
    return mean, var, tmr.dt


def extract_sensor_features(x_full: torch.Tensor, s: int, S: int) -> torch.Tensor:
    """
    Given x_full: (1,T,Dx) with interleaved layout, return local sensor features:
      x^(s): (1,T,num_time_points) as x_full[..., s::S]
    """
    return x_full[..., s::S]


def reconstruct_x_from_sensors(x_parts: List[torch.Tensor], S: int) -> torch.Tensor:
    """
    Given list of S tensors x_parts[s] each (1,T,num_time_points), reconstruct x_full (1,T,Dx)
    with interleaved layout.
    """
    # stack -> (S,1,T,num_time_points) then permute to (1,T,num_time_points,S) then reshape
    X = torch.stack(x_parts, dim=0)          # (S,1,T,P)
    X = X.permute(1, 2, 3, 0).contiguous()   # (1,T,P,S)
    return X.view(X.shape[0], X.shape[1], -1)  # (1,T,P*S)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--anp_result_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--topologies", type=str, default="aligned,ellipsoidal,random")

    p.add_argument("--context_percent", type=int, default=30)
    p.add_argument("--ctx_sample_mode", type=str, default="first", choices=["first", "random"])

    p.add_argument("--num_time_points", type=int, default=201)
    p.add_argument("--num_sensors", type=int, default=10)

    p.add_argument("--mc_samples", type=int, default=1)
    p.add_argument("--seed_eval", type=int, default=0)
    p.add_argument("--max_traj_per_theta", type=int, default=-1)

    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed_eval)

    topologies = [t.strip() for t in args.topologies.split(",") if t.strip()]
    rows = []

    for topo in topologies:
        theta_groups, theta_values = load_topology_test(args.data_dir, topo)
        x0, y0 = theta_groups[theta_values[0]][0]
        T = x0.shape[0]
        Dx = x0.shape[1]
        Dy = y0.shape[1]

        expected = args.num_time_points * args.num_sensors
        if Dx != expected:
            raise ValueError(f"[{topo}] Dx={Dx} != {expected} (num_time_points*num_sensors)")

        model = load_model(args.anp_result_dir, topo, Dx, Dy, device)
        y_mean, y_std = load_y_stats(args.data_dir, topo, device)

        n_ctx = max(1, min(int((args.context_percent / 100.0) * T), T))

        total_mae = []
        total_time = 0.0

        # comm: sending features (float32) from S sensors to fusion center for all T points
        # per trajectory: S * T * num_time_points floats
        comm_features_floats_per_traj = args.num_sensors * T * args.num_time_points

        # comm: sending outputs (mu,var) from S sensors (if doing output-level fusion)
        # per trajectory: (S-1) * 2 * T * Dy floats to a fusion center (same convention as your old script)
        comm_outputs_floats_per_traj = (args.num_sensors - 1) * 2 * T * Dy

        for theta in theta_values:
            samples = theta_groups[theta]
            if args.max_traj_per_theta > 0:
                samples = samples[:args.max_traj_per_theta]

            for (x_np, y_np) in tqdm(samples, desc=f"{topo} theta={theta:.3f}", leave=False):
                x_full = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(0)  # (1,T,Dx)
                y = torch.tensor(y_np, dtype=torch.float32, device=device).unsqueeze(0)       # (1,T,Dy)
                y_norm = normalize_y(y, y_mean, y_std)

                # distributed acquisition: each sensor "has" only its local part
                x_parts = [extract_sensor_features(x_full, s, args.num_sensors) for s in range(args.num_sensors)]
                # fusion center reconstructs x_full_recon
                x_full_recon = reconstruct_x_from_sensors(x_parts, args.num_sensors)

                # context selection (same indices for all sensors and central)
                ctx_idx = sample_context_indices(T, n_ctx, args.ctx_sample_mode, gen, device)
                non_ctx_mask = torch.ones(T, dtype=torch.bool, device=device)
                non_ctx_mask[ctx_idx] = False

                cx = x_full_recon[:, ctx_idx, :]
                cy = y_norm[:, ctx_idx, :]

                mean_norm, var_norm, dt = anp_predict(model, cx, cy, x_full_recon, args.mc_samples, device)
                total_time += dt

                pred = denormalize_y(mean_norm, y_mean, y_std)
                mae = F.l1_loss(pred[:, non_ctx_mask, :], y[:, non_ctx_mask, :], reduction="none").mean().item()
                total_mae.append(mae)

        mae_overall = float(np.mean(total_mae)) if total_mae else float("nan")

        # total comm in KiB across all evaluated trajectories
        n_traj = len(total_mae)
        comm_features_kib = (comm_features_floats_per_traj * 4 * n_traj) / 1024.0
        comm_outputs_kib = (comm_outputs_floats_per_traj * 4 * n_traj) / 1024.0

        print("\n" + "="*95)
        print(f"Topology {topo} | ctx={args.context_percent}% ({args.ctx_sample_mode}) | central inference, distributed features")
        print(f"  MAE={mae_overall:8.4f}  time(serial)={total_time:7.3f}s")
        print(f"  comm(features)={comm_features_kib:,.1f} KiB   comm(outputs mu,var)={comm_outputs_kib:,.1f} KiB")
        print("="*95)

        rows.append({
            "topology": topo,
            "context_percent": args.context_percent,
            "ctx_sample_mode": args.ctx_sample_mode,
            "mc_samples": args.mc_samples,
            "mae_overall": mae_overall,
            "time_serial_total_s": total_time,
            "n_trajectories": n_traj,
            "comm_features_total_kib": comm_features_kib,
            "comm_outputs_total_kib": comm_outputs_kib,
            "num_time_points": args.num_time_points,
            "num_sensors": args.num_sensors,
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    csv_path = outdir / "central_inference_distributed_features.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}\n")


if __name__ == "__main__":
    main()
