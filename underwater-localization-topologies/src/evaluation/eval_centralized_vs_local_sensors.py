'''
eval_centralized_vs_local_sensors.py

Measures the performance GAP between:

  1. CENTRALIZED  — one ANP with input_dim = P*S (all sensors).
  2. LOCAL ENSEMBLE — S independent ANPs each with input_dim = P (one sensor),
                      whose predicted means are averaged at inference time.
  3. (optional) PER-SENSOR individually — MAE of each local model in isolation.

Metrics are computed on the test set for each topology independently,
sweeping over context fractions.

Output (per topology):
  • gap_results.csv          — numeric table
  • gap_heatmap.png          — heatmap  (rows=context_frac, cols=method)
  • gap_bar.png              — bar chart for a quick visual comparison

Usage example:
  python src/evaluation/eval_centralized_vs_local_sensors.py \
    --data-dir     data/data/data_processed_topologies_low_variance \
    --central-dir  results/ANP_topologies/low_variance/ctx_first \
    --local-dir    results/ANP_sensor_local \
    --output-dir   results/eval_central_vs_local_sensors \
    --topologies   ellipsoidal,aligned,random \
    --num-sensors  10 \
    --num-time-points 201 \
    --context-fracs 0.1,0.3,0.5 \
    --ctx-sample-mode first \
    --num-hidden 128
'''

import argparse
import os
import sys
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.anp import LatentModel


# ──────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────

def load_topology_data(data_dir: str, topology: str):
    """Loads (train_data, test_data, metadata). Falls back to val_data if test_data missing."""
    base = os.path.join(data_dir, f"topology_{topology}")
    train_path    = os.path.join(base, "train_data.pkl")
    test_path     = os.path.join(base, "test_data.pkl")
    val_path      = os.path.join(base, "val_data.pkl")
    metadata_path = os.path.join(base, "metadata.pkl")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train_data.pkl not found in {base}")

    eval_path = test_path if os.path.exists(test_path) else val_path
    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"Neither test_data.pkl nor val_data.pkl found in {base}")

    with open(train_path, "rb")    as f: train_data = pickle.load(f)
    with open(eval_path, "rb")     as f: eval_data  = pickle.load(f)
    with open(metadata_path, "rb") as f: metadata   = pickle.load(f)

    used = "test_data.pkl" if os.path.exists(test_path) else "val_data.pkl"
    print(f"  [{topology}] Evaluation data: {used}  ({len(eval_data)} trajectories)")
    return train_data, eval_data, metadata


def compute_y_stats(train_data) -> Tuple[torch.Tensor, torch.Tensor]:
    Y = np.concatenate([y for _, y in train_data], axis=0)
    y_mean = torch.tensor(Y.mean(axis=0), dtype=torch.float32)
    y_std  = torch.tensor(Y.std(axis=0) + 1e-6, dtype=torch.float32)
    return y_mean, y_std


def sample_context_indices(T: int, ctx_size: int, mode: str, device: torch.device,
                           generator=None) -> torch.Tensor:
    if mode == "first":
        return torch.arange(ctx_size, device=device)
    if mode == "random":
        perm = torch.randperm(T, device=device, generator=generator)
        return perm[:ctx_size].sort().values
    raise ValueError(f"Unknown context sampling mode: {mode}")


# ──────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────

def load_centralized_model(
    central_dir: str, topology: str, num_hidden: int,
    input_dim: int, output_dim: int, device: torch.device
) -> Optional[LatentModel]:
    """Loads best_checkpoint.pth.tar from <central_dir>/topology_<topo>/"""
    ckpt_path = os.path.join(central_dir, f"topology_{topology}", "best_checkpoint.pth.tar")
    if not os.path.exists(ckpt_path):
        print(f"  [WARNING] Centralized checkpoint not found: {ckpt_path}")
        return None

    model = LatentModel(num_hidden=num_hidden, input_dim=input_dim, output_dim=output_dim).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  Loaded centralised model  ({input_dim} inputs)  ← {ckpt_path}")
    return model


def load_local_sensor_models(
    local_dir: str, topology: str,
    sensor_indices: List[int],
    input_dim_per_sensor: int,
    output_dim: int,
    num_hidden: int,
    device: torch.device
) -> Dict[int, LatentModel]:
    """Loads one model per sensor from <local_dir>/topology_<topo>/sensor_<s>/best_checkpoint.pth.tar"""
    models: Dict[int, LatentModel] = {}
    for s in sensor_indices:
        ckpt_path = os.path.join(
            local_dir, f"topology_{topology}", f"sensor_{s:02d}", "best_checkpoint.pth.tar"
        )
        if not os.path.exists(ckpt_path):
            print(f"  [WARNING] Local sensor {s} checkpoint not found: {ckpt_path}")
            continue
        m = LatentModel(num_hidden=num_hidden,
                        input_dim=input_dim_per_sensor,
                        output_dim=output_dim).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        m.load_state_dict(ckpt["model"])
        m.eval()
        models[s] = m

    print(f"  Loaded {len(models)}/{len(sensor_indices)} local sensor models  "
          f"({input_dim_per_sensor} inputs each)")
    return models


# ──────────────────────────────────────────────────────────────────
# Inference helpers
# ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_centralized(
    model: LatentModel,
    x: torch.Tensor,   # (1, T, P*S)
    y_norm: torch.Tensor,  # (1, T, out)
    ctx_idx: torch.Tensor,
) -> torch.Tensor:
    """Returns predicted mean (1, T, out) in normalised space."""
    tar_idx = torch.arange(x.size(1), device=x.device)
    ctx_x, ctx_y = x[:, ctx_idx, :], y_norm[:, ctx_idx, :]
    tar_x = x[:, tar_idx, :]
    y_hat, _, _, _, _ = model(ctx_x, ctx_y, tar_x, target_y=None)
    return y_hat  # (1, T, out)


@torch.no_grad()
def predict_ensemble(
    sensor_models: Dict[int, LatentModel],
    x: torch.Tensor,   # (1, T, P*S)
    y_norm: torch.Tensor,  # (1, T, out)
    ctx_idx: torch.Tensor,
    P: int,
    S: int,
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    """
    Each local model sees only the P features of its sensor.
    Returns:
      ensemble_mean  (1, T, out)  — simple average of individual means
      per_sensor_means  {s: (1, T, out)}
    """
    # build sensor-specific views once
    x_per_sensor: Dict[int, torch.Tensor] = {}
    x4 = x.view(1, -1, P, S)  # (1, T, P, S)
    for s in sensor_models:
        x_per_sensor[s] = x4[:, :, :, s]  # (1, T, P)

    per_sensor_means: Dict[int, torch.Tensor] = {}
    for s, m in sensor_models.items():
        xs = x_per_sensor[s]
        y_norm_ctx = y_norm[:, ctx_idx, :]
        xs_ctx = xs[:, ctx_idx, :]
        xs_tar = xs[:, :, :]
        tar_idx = torch.arange(xs.size(1), device=xs.device)
        y_hat, _, _, _, _ = m(xs_ctx, y_norm_ctx, xs_tar, target_y=None)
        per_sensor_means[s] = y_hat  # (1, T, out)

    # simple average (PoM: pool of means)
    stacked = torch.stack(list(per_sensor_means.values()), dim=0)  # (S, 1, T, out)
    ensemble_mean = stacked.mean(dim=0)                             # (1, T, out)
    return ensemble_mean, per_sensor_means


# ──────────────────────────────────────────────────────────────────
# Core evaluation
# ──────────────────────────────────────────────────────────────────

def evaluate_topology(
    topology: str,
    eval_data: list,
    train_data: list,
    central_model: Optional[LatentModel],
    sensor_models: Dict[int, LatentModel],
    context_fracs: List[float],
    ctx_sample_mode: str,
    P: int,
    S: int,
    device: torch.device,
    seed: int = 42,
) -> Dict:
    """
    Returns a dict keyed by context_frac with sub-dicts:
      mae_central, mae_ensemble,
      mae_sensor_<s> for each loaded sensor.
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    y_mean, y_std = compute_y_stats(train_data)
    y_mean = y_mean.to(device)
    y_std  = y_std.to(device)

    results: Dict[float, Dict] = {}

    for frac in context_fracs:
        acc_central  = 0.0
        acc_ensemble = 0.0
        acc_sensor   = {s: 0.0 for s in sensor_models}
        n_traj       = 0

        for x_np, y_np in eval_data:
            x = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(0)  # (1,T,Dx)
            y = torch.tensor(y_np, dtype=torch.float32, device=device).unsqueeze(0)  # (1,T,out)
            T = x.size(1)

            ctx_size = max(1, min(T - 1, int(round(frac * T))))
            ctx_idx  = sample_context_indices(T, ctx_size, ctx_sample_mode, device, generator=g)
            non_ctx  = torch.ones(T, dtype=torch.bool, device=device)
            non_ctx[ctx_idx] = False

            y_norm = (y - y_mean) / y_std

            # ── centralized ──
            if central_model is not None:
                y_hat_norm = predict_centralized(central_model, x, y_norm, ctx_idx)
                y_hat      = y_hat_norm * y_std + y_mean
                acc_central += F.l1_loss(y_hat[:, non_ctx, :], y[:, non_ctx, :]).item()

            # ── ensemble & per-sensor ──
            if sensor_models:
                ens_mean, ps_means = predict_ensemble(
                    sensor_models, x, y_norm, ctx_idx, P=P, S=S
                )
                ens_mean_raw = ens_mean * y_std + y_mean
                acc_ensemble += F.l1_loss(ens_mean_raw[:, non_ctx, :], y[:, non_ctx, :]).item()

                for s, y_hat_s_norm in ps_means.items():
                    y_hat_s = y_hat_s_norm * y_std + y_mean
                    acc_sensor[s] += F.l1_loss(y_hat_s[:, non_ctx, :], y[:, non_ctx, :]).item()

            n_traj += 1

        row = {
            "mae_central":  acc_central  / n_traj if central_model else float("nan"),
            "mae_ensemble": acc_ensemble / n_traj if sensor_models else float("nan"),
        }
        for s in sensor_models:
            row[f"mae_sensor_{s:02d}"] = acc_sensor[s] / n_traj

        results[frac] = row

    return results


# ──────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────

def plot_gap_heatmap(results: Dict[float, Dict], topology: str, output_dir: str):
    """Heatmap: rows = context_frac, columns = method."""
    fracs   = sorted(results.keys())
    methods = ["mae_central", "mae_ensemble"]

    data_matrix = []
    labels_col  = ["Centralized", "Local Ensemble"]

    for frac in fracs:
        row = results[frac]
        data_matrix.append([row.get("mae_central", np.nan),
                             row.get("mae_ensemble", np.nan)])

    arr = np.array(data_matrix)
    row_labels = [f"ctx={int(f*100)}%" for f in fracs]

    fig, ax = plt.subplots(figsize=(max(5, len(labels_col) * 1.2), max(3, len(fracs) * 0.8 + 1)))
    sns.heatmap(
        arr, annot=True, fmt=".3f", cmap="YlOrRd",
        xticklabels=labels_col, yticklabels=row_labels,
        ax=ax, linewidths=0.5, cbar_kws={"label": "MAE"},
    )
    ax.set_title(f"MAE — topology: {topology}")
    ax.set_xlabel("Method")
    ax.set_ylabel("Context fraction")
    plt.tight_layout()
    path = os.path.join(output_dir, f"gap_heatmap_{topology}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved heatmap → {path}")


def plot_gap_bar(results: Dict[float, Dict], topology: str, output_dir: str,
                 sensor_indices: List[int]):
    """Bar chart: one group per context fraction, bars = centralized / ensemble / per-sensor."""
    fracs = sorted(results.keys())
    x     = np.arange(len(fracs))
    width = 0.8

    # build groups: centralized, ensemble, sensor_0 … sensor_S-1
    groups = {"Centralized":    [results[f]["mae_central"]  for f in fracs],
              "Local ensemble": [results[f]["mae_ensemble"] for f in fracs]}
    for s in sorted(sensor_indices):
        key = f"mae_sensor_{s:02d}"
        if key in results[fracs[0]]:
            groups[f"Sensor {s}"] = [results[f][key] for f in fracs]

    n_groups = len(groups)
    w        = width / n_groups
    offsets  = np.linspace(-(width - w) / 2, (width - w) / 2, n_groups)

    fig, ax = plt.subplots(figsize=(max(8, len(fracs) * 2), 5))
    for (label, vals), offset in zip(groups.items(), offsets):
        # highlight main methods
        color  = None
        hatch  = None
        lw     = 1.0
        alpha  = 0.85
        if label == "Centralized":
            color = "#2196F3"; lw = 1.5
        elif label == "Local ensemble":
            color = "#FF5722"; lw = 1.5
        else:
            alpha = 0.55

        ax.bar(x + offset, vals, width=w, label=label,
               color=color, hatch=hatch, linewidth=lw, alpha=alpha,
               edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([f"ctx={int(f*100)}%" for f in fracs])
    ax.set_ylabel("MAE")
    ax.set_title(f"Centralized vs local-sensor ensemble — topology: {topology}")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    path = os.path.join(output_dir, f"gap_bar_{topology}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved bar chart → {path}")


def plot_gap_line(results: Dict[float, Dict], topology: str, output_dir: str,
                  sensor_indices: List[int]):
    """Line plot: MAE vs context fraction for centralized, ensemble, and per-sensor."""
    fracs = sorted(results.keys())
    frac_pct = [int(f * 100) for f in fracs]

    fig, ax = plt.subplots(figsize=(7, 4))

    central_vals  = [results[f]["mae_central"]  for f in fracs]
    ensemble_vals = [results[f]["mae_ensemble"] for f in fracs]

    ax.plot(frac_pct, central_vals,  marker="o", lw=2.0,  label="Centralized",    color="#2196F3")
    ax.plot(frac_pct, ensemble_vals, marker="s", lw=2.0,  label="Local ensemble", color="#FF5722")

    for s in sorted(sensor_indices):
        key = f"mae_sensor_{s:02d}"
        if key in results[fracs[0]]:
            vals = [results[f][key] for f in fracs]
            ax.plot(frac_pct, vals, marker=".", lw=0.8, alpha=0.45, label=f"Sensor {s}")

    # shade gap
    ax.fill_between(frac_pct, central_vals, ensemble_vals,
                    alpha=0.12, color="gray", label="Gap")

    ax.set_xlabel("Context fraction (%)")
    ax.set_ylabel("MAE")
    ax.set_title(f"Performance gap — topology: {topology}")
    ax.grid(alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    path = os.path.join(output_dir, f"gap_line_{topology}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved line plot → {path}")


# ──────────────────────────────────────────────────────────────────
# CSV export
# ──────────────────────────────────────────────────────────────────

def save_results_csv(all_results: Dict[str, Dict], output_dir: str, sensor_indices: List[int]):
    path = os.path.join(output_dir, "gap_results.csv")
    # collect all column names
    sample_frac_res = next(iter(next(iter(all_results.values())).values()))
    extra_cols = [k for k in sample_frac_res if k not in ("mae_central", "mae_ensemble")]

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["topology", "context_frac", "mae_central", "mae_ensemble",
                    "gap_abs", "gap_rel_pct"] + extra_cols)
        for topo, frac_dict in all_results.items():
            for frac, row in sorted(frac_dict.items()):
                central  = row["mae_central"]
                ensemble = row["mae_ensemble"]
                gap_abs  = ensemble - central if not (np.isnan(central) or np.isnan(ensemble)) else np.nan
                gap_rel  = (gap_abs / central * 100) if (not np.isnan(gap_abs) and central > 0) else np.nan
                w.writerow([
                    topo, f"{frac:.2f}",
                    f"{central:.6f}", f"{ensemble:.6f}",
                    f"{gap_abs:.6f}" if not np.isnan(gap_abs) else "nan",
                    f"{gap_rel:.2f}" if not np.isnan(gap_rel) else "nan",
                ] + [f"{row.get(c, float('nan')):.6f}" for c in extra_cols])
    print(f"\n  Saved CSV → {path}")
    return path


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate performance gap: centralized ANP vs naïve local-sensor ensemble"
    )
    parser.add_argument("--data-dir",         type=str, required=True,
                        help="Root data directory containing topology_<name>/ folders")
    parser.add_argument("--central-dir",      type=str, required=True,
                        help="Base dir that contains topology_<name>/best_checkpoint.pth.tar "
                             "for the centralized model")
    parser.add_argument("--local-dir",        type=str, required=True,
                        help="Base dir from train_anp_sensor_local.py "
                             "(topology_<name>/sensor_<s>/best_checkpoint.pth.tar)")
    parser.add_argument("--output-dir",       type=str, default="results/eval_central_vs_local_sensors")
    parser.add_argument("--topologies",       type=str, default="aligned,ellipsoidal,random")
    parser.add_argument("--sensors",          type=str, default=None,
                        help="Comma-separated indices. Default: 0..num-sensors-1")
    parser.add_argument("--num-sensors",      type=int,   default=10)
    parser.add_argument("--num-time-points",  type=int,   default=201)
    parser.add_argument("--context-fracs",    type=str,   default="0.1,0.3,0.5")
    parser.add_argument("--ctx-sample-mode",  type=str,   default="first",
                        choices=["first", "random"])
    parser.add_argument("--num-hidden",       type=int,   default=128)
    parser.add_argument("--output-dim",       type=int,   default=3)
    parser.add_argument("--device",           type=str,   default=None,
                        help="cuda or cpu. Default: auto-detect.")
    parser.add_argument("--seed",             type=int,   default=42)

    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    topologies    = [t.strip() for t in args.topologies.split(",") if t.strip()]
    context_fracs = [float(f.strip()) for f in args.context_fracs.split(",") if f.strip()]
    P = args.num_time_points
    S = args.num_sensors

    sensor_indices = (
        [int(s.strip()) for s in args.sensors.split(",") if s.strip()]
        if args.sensors else list(range(S))
    )

    input_dim_central = P * S   # full-input centralised model
    input_dim_local   = P       # per-sensor local model

    os.makedirs(args.output_dir, exist_ok=True)

    all_results: Dict[str, Dict] = {}

    for topo in topologies:
        print(f"\n{'='*60}")
        print(f"Topology: {topo}")
        print(f"{'='*60}")

        try:
            train_data, eval_data, _ = load_topology_data(args.data_dir, topo)
        except FileNotFoundError as e:
            print(f"  SKIP — {e}")
            continue

        topo_out = os.path.join(args.output_dir, f"topology_{topo}")
        os.makedirs(topo_out, exist_ok=True)

        # ── load models ──
        central_model = load_centralized_model(
            args.central_dir, topo,
            num_hidden=args.num_hidden,
            input_dim=input_dim_central,
            output_dim=args.output_dim,
            device=device,
        )

        sensor_models = load_local_sensor_models(
            args.local_dir, topo,
            sensor_indices=sensor_indices,
            input_dim_per_sensor=input_dim_local,
            output_dim=args.output_dim,
            num_hidden=args.num_hidden,
            device=device,
        )

        if central_model is None and not sensor_models:
            print(f"  SKIP — no models loaded for topology {topo}")
            continue

        # ── evaluate ──
        print(f"  Evaluating on {len(eval_data)} trajectories ...")
        results = evaluate_topology(
            topology       = topo,
            eval_data      = eval_data,
            train_data     = train_data,
            central_model  = central_model,
            sensor_models  = sensor_models,
            context_fracs  = context_fracs,
            ctx_sample_mode= args.ctx_sample_mode,
            P              = P,
            S              = S,
            device         = device,
            seed           = args.seed,
        )
        all_results[topo] = results

        # ── print table ──
        print(f"\n  {'ctx':>6}  {'Central':>10}  {'Ensemble':>10}  {'Gap':>8}  {'Gap%':>7}")
        print(f"  {'-'*50}")
        for frac in sorted(results.keys()):
            row = results[frac]
            c   = row["mae_central"]
            e   = row["mae_ensemble"]
            g   = e - c if not (np.isnan(c) or np.isnan(e)) else float("nan")
            gp  = g / c * 100 if (not np.isnan(g) and c > 0) else float("nan")
            print(f"  {int(frac*100):>5}%  {c:>10.4f}  {e:>10.4f}  {g:>8.4f}  {gp:>6.1f}%")

        # ── plots ──
        plot_gap_heatmap(results, topo, topo_out)
        plot_gap_bar(results, topo, topo_out, list(sensor_models.keys()))
        plot_gap_line(results, topo, topo_out, list(sensor_models.keys()))

        # ── per-topology CSV ──
        csv_topo = os.path.join(topo_out, "gap_results.csv")
        extra_cols = [k for k in results[context_fracs[0]]
                      if k not in ("mae_central", "mae_ensemble")]
        with open(csv_topo, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["context_frac", "mae_central", "mae_ensemble",
                        "gap_abs", "gap_rel_pct"] + extra_cols)
            for frac in sorted(results.keys()):
                row      = results[frac]
                central  = row["mae_central"]
                ensemble = row["mae_ensemble"]
                gap_abs  = ensemble - central
                gap_rel  = gap_abs / central * 100 if central > 0 else float("nan")
                w.writerow([
                    f"{frac:.2f}", f"{central:.6f}", f"{ensemble:.6f}",
                    f"{gap_abs:.6f}", f"{gap_rel:.2f}",
                ] + [f"{row.get(c, float('nan')):.6f}" for c in extra_cols])
        print(f"  Saved per-topology CSV → {csv_topo}")

    # ── global CSV ──
    if all_results:
        save_results_csv(all_results, args.output_dir, sensor_indices)

    # ── global summary printout ──
    print("\n" + "="*60)
    print("SUMMARY — MAE gap (ensemble − centralized)")
    print("="*60)
    print(f"  {'topology':<14} {'ctx':>6}  {'Central':>10}  {'Ensemble':>10}  {'Gap%':>7}")
    for topo, frac_dict in all_results.items():
        for frac in sorted(frac_dict.keys()):
            row = frac_dict[frac]
            c   = row["mae_central"]
            e   = row["mae_ensemble"]
            gp  = (e - c) / c * 100 if (not np.isnan(c) and c > 0) else float("nan")
            print(f"  {topo:<14} {int(frac*100):>5}%  {c:>10.4f}  {e:>10.4f}  {gp:>6.1f}%")

    print(f"\nAll outputs saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
