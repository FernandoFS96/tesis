"""
eval_ood_sensor_restriction.py

Evaluates masked ANP models under two stress axes:
  1. Out-of-Distribution (OoD) acoustic channel variability
  2. Sensor restriction at deployment time

Experiments
-----------
  1.1  Bidirectional OoD matrix:
            rows = {low-var model, high-var model}
            cols = {low-var test data, high-var test data}
            metric = mean MAE over all thetas in each domain

  1.2  Per-theta MAE degradation curve:
            both models evaluated on every individual theta level
            (uses test_thetas from metadata.pkl to group test samples)

  1.3  Context-fraction sweep (OoD mitigator analysis):
            context_frac in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
            both models x both domains, all sensors available

  1.4º  Trajectory prediction plots per theta:
            for each theta, plot GT + predictions from all available models
            (uses test_thetas from metadata.pkl to group test samples)

  2.1  Sensor restriction modes on ellipsoidal topology:
            a) Bernoulli dropout: p_drop sweep
            b) k-uniform: fixed k_active sweep
            c) Cluster dropout: cluster_size sweep (uses circular index distance)
            evaluated on the model's in-distribution test data

Usage
-----
python eval_ood_sensor_restriction.py \
  --lowvar-ckpt  /home/fernando/tesis/underwater-localization-topologies/src/training/results/ANP_topologies_masked/lowvar/masked_dropbernoulli_p0.2_train_mean_first/topology_ellipsoidal/best_checkpoint.pth.tar \
  --highvar-ckpt /home/fernando/tesis/underwater-localization-topologies/src/training/results/ANP_topologies_masked/highvar/masked_dropbernoulli_p0.2_train_mean_first/topology_ellipsoidal/best_checkpoint.pth.tar \
  --lowvar-data-dir  /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_low_variance \
  --highvar-data-dir /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_high_variance \
  --topology ellipsoidal \
  --output-dir results/eval_ood_sensor_restriction \
  --experiments 1.1,1.2,1.3,1.4,2.1
"""

import os
import sys
import csv
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.anp import LatentModel
from src.utils.nav_dataset import NavigationTrajectoryDataset

# =============================================================================
# Constants
# =============================================================================
NUM_SENSORS = 10
NUM_TIME_POINTS = 201
OUTPUT_DIM = 3
NUM_HIDDEN = 128
INPUT_DIM_BASE = NUM_TIME_POINTS * NUM_SENSORS # 2010
INPUT_DIM_AUG = INPUT_DIM_BASE + NUM_SENSORS # 2020  (base + mask features)
MASK_FILL = "train_mean"
EVAL_SEED = 18
CONTEXT_FRAC_DEFAULT = 0.3
# Sweeps
CONTEXT_FRACS = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]
BERNOULLI_P_DROPS = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8] # 0.0 = all sensors
K_ACTIVE_VALUES = [10, 9, 8, 7, 5, 3, 1] # sensors kept
CLUSTER_SIZES = [1, 2, 3, 4, 5] # sensors REMOVED

# =============================================================================
# I/O helpers
# =============================================================================

def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_csv(path: Path, rows: list, header: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  Saved CSV → {path}")

# =============================================================================
# Model loading
# =============================================================================

def load_masked_anp(ckpt_path: Path, device: torch.device) -> LatentModel:
    """Load a masked-trained ANP (input_dim = INPUT_DIM_AUG)."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model = LatentModel(
        num_hidden=NUM_HIDDEN,
        input_dim=INPUT_DIM_AUG,
        output_dim=OUTPUT_DIM,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  Loaded model from {ckpt_path}")
    return model

# =============================================================================
# Data / statistics
# =============================================================================

def load_topology_split(data_dir: Path, topology: str):
    """Returns (train_data, val_data, test_data, metadata)."""
    tdir = data_dir / f"topology_{topology}"
    train_data = load_pickle(tdir / "train_data.pkl")
    val_data   = load_pickle(tdir / "val_data.pkl")
    test_data  = load_pickle(tdir / "test_data.pkl")
    metadata   = load_pickle(tdir / "metadata.pkl")
    return train_data, val_data, test_data, metadata

def compute_train_stats(train_data, device: torch.device):
    """
    Compute y_mean, y_std and x_means_SP from training data.
    These MUST match the stats used during model training.
    
    Returns:
        y_mean   : (3,)   tensor on device
        y_std    : (3,)   tensor on device
        x_means_SP : (S,P) tensor on device
    """
    S, P = NUM_SENSORS, NUM_TIME_POINTS

    Y = np.concatenate([y for _, y in train_data], axis=0)   # (N*T, 3)
    y_mean = torch.tensor(Y.mean(axis=0), dtype=torch.float32, device=device)
    y_std  = torch.tensor(Y.std(axis=0) + 1e-6, dtype=torch.float32, device=device)

    X = np.concatenate([x for x, _ in train_data], axis=0)   # (N*T, P*S)
    X3 = X.reshape(X.shape[0], P, S)                          # (N*T, P, S)
    mean_PS = X3.mean(axis=0)                                  # (P, S)
    x_means_SP = torch.tensor(mean_PS.T, dtype=torch.float32, device=device)  # (S, P)

    return y_mean, y_std, x_means_SP

def group_test_data_by_theta(test_data, metadata):
    """Returns dict {theta_value: [(X,y), ...]}."""
    groups = {}
    for sample, theta in zip(test_data, metadata["test_thetas"]):
        groups.setdefault(theta, []).append(sample)
    return groups

# =============================================================================
# Sensor masking (evaluation-time)
# =============================================================================

def make_all_sensors_mask(B: int, device: torch.device) -> torch.Tensor:
    """All-ones mask: all S sensors available."""
    return torch.ones(B, NUM_SENSORS, device=device)

def make_bernoulli_mask(B: int, p_drop: float, device: torch.device,
                        rng: torch.Generator) -> torch.Tensor:
    """Independent Bernoulli dropout per sensor."""
    if p_drop == 0.0:
        return make_all_sensors_mask(B, device)
    keep = torch.rand(B, NUM_SENSORS, generator=rng, device=device) > p_drop
    # ensure at least one sensor on
    all_off = ~keep.any(dim=1)
    if all_off.any():
        idx = torch.randint(0, NUM_SENSORS, (all_off.sum().item(),),
                            generator=rng, device=device)
        keep[all_off, idx] = True
    return keep.float()

def make_k_uniform_mask(B: int, k_active: int, device: torch.device,
                        rng: torch.Generator) -> torch.Tensor:
    """Exactly k_active sensors kept (random subset per sample)."""
    k = max(1, min(k_active, NUM_SENSORS))
    keep = torch.zeros(B, NUM_SENSORS, dtype=torch.bool, device=device)
    for b in range(B):
        idx = torch.randperm(NUM_SENSORS, generator=rng, device=device)[:k]
        keep[b, idx] = True
    return keep.float()

def make_cluster_mask(B: int, cluster_size: int, device: torch.device,
                      rng: torch.Generator) -> torch.Tensor:
    """
    Cluster dropout for ellipsoidal topology.
    Sensors are uniformly spaced at angles 2πi/S on an ellipse,
    so circular index distance is the proxy for spatial proximity.
    Fails `cluster_size` sensors: a random center sensor plus its
    (cluster_size - 1) nearest neighbours by circular distance.
    All remaining sensors stay on.
    cluster_size=1 means one random sensor is dropped.
    """
    S = NUM_SENSORS
    cluster_size = max(1, min(cluster_size, S - 1))  # keep at least 1 on

    # circular distances from each possible center
    idx_range = torch.arange(S, device=device)
    keep = torch.ones(B, S, dtype=torch.bool, device=device)

    for b in range(B):
        center = torch.randint(0, S, (1,), generator=rng, device=device).item()
        dist = torch.minimum(
            torch.abs(idx_range - center),
            S - torch.abs(idx_range - center)
        )
        # sort by distance, take the closest cluster_size as the failed cluster
        _, sorted_idx = dist.sort()
        failed = sorted_idx[:cluster_size]
        keep[b, failed] = False

    return keep.float()

def augment_with_mask(
    x_batch: torch.Tensor,      # (B, T, Dx)
    sensor_mask: torch.Tensor,  # (B, S)
    x_means_SP: torch.Tensor,   # (S, P) — from MODEL's training domain
) -> torch.Tensor:
    """
    Replace masked sensors with training-mean fill, then append binary mask
    features → output shape (B, T, Dx + S).
    """
    B, T, Dx = x_batch.shape
    P, S = NUM_TIME_POINTS, NUM_SENSORS
    assert Dx == P * S

    x4 = x_batch.view(B, T, P, S)
    mu  = x_means_SP.T.view(1, 1, P, S).to(device=x_batch.device, dtype=x_batch.dtype)
    fill_val = mu.expand(B, T, P, S)

    m = sensor_mask.view(B, 1, 1, S)
    x4_masked = x4 * m + fill_val * (1.0 - m)
    x_masked  = x4_masked.reshape(B, T, Dx)

    mask_feat = sensor_mask.view(B, 1, S).expand(B, T, S)
    return torch.cat([x_masked, mask_feat], dim=-1)  # (B, T, Dx+S)

# =============================================================================
# Core evaluation routine
# =============================================================================

def evaluate_model(
    model: LatentModel,
    test_data: list,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    x_means_SP: torch.Tensor,
    device: torch.device,
    context_frac: float = CONTEXT_FRAC_DEFAULT,
    sensor_mask_fn=None,          # callable(B, device, rng) -> (B,S) mask
    batch_size: int = 16,
    seed: int = EVAL_SEED,
    eval_on_all_points: bool = False,
) -> float:
    """
    Returns mean MAE (denormalised, metres) over the test set.

    sensor_mask_fn: if None, all sensors are available.

    eval_on_all_points: if True, compute MAE over ALL T target points (including context points). 
        This is the correct mode for a context fraction sweep: the denominator stays fixed across fractions, 
        so differences in MAE reflect purely the benefit of more context, not a shift in which (potentially harder) points are being evaluated.
        If False (default), MAE is computed only on non-context points, which is appropriate when context_frac is fixed across conditions.
    """
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    dataset = NavigationTrajectoryDataset(test_data)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_mae, n_batches = 0.0, 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)   # (B, T, Dx)
            y_batch = y_batch.to(device)   # (B, T, 3)
            B, T, _ = x_batch.shape

            # sensor mask
            if sensor_mask_fn is None:
                sensor_mask = make_all_sensors_mask(B, device)
            else:
                sensor_mask = sensor_mask_fn(B, device, rng)

            x_aug = augment_with_mask(x_batch, sensor_mask, x_means_SP)  # (B,T,Dx+S)

            # context / target split
            n_ctx = max(1, min(T - 1, int(round(context_frac * T))))
            ctx_idx = torch.arange(n_ctx, device=device)          # "first" mode
            tar_idx = torch.arange(T, device=device)

            y_norm = (y_batch - y_mean) / y_std

            ctx_x = x_aug[:, ctx_idx, :]
            ctx_y = y_norm[:, ctx_idx, :]
            tar_x = x_aug[:, tar_idx, :]
            tar_y = y_norm[:, tar_idx, :]

            y_pred_norm, _, _, _, _ = model(ctx_x, ctx_y, tar_x)

            y_pred = y_pred_norm * y_std + y_mean

            if eval_on_all_points:
                # MAE over the entire trajectory — denominator is fixed regardless
                # of context_frac, isolating the pure effect of context size.
                mae = F.l1_loss(y_pred, y_batch, reduction="mean").item()
            else:
                # MAE only on non-context points (default for fixed-context exps).
                non_ctx = torch.ones(T, dtype=torch.bool, device=device)
                non_ctx[ctx_idx] = False
                mae = F.l1_loss(
                    y_pred[:, non_ctx, :],
                    y_batch[:, non_ctx, :],
                    reduction="mean",
                ).item()

            total_mae += mae
            n_batches += 1

    return total_mae / max(n_batches, 1)

def evaluate_by_theta(
    model: LatentModel,
    theta_groups: dict,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    x_means_SP: torch.Tensor,
    device: torch.device,
    context_frac: float = CONTEXT_FRAC_DEFAULT,
    sensor_mask_fn=None,
    batch_size: int = 16,
    seed: int = EVAL_SEED,
) -> dict:
    """Returns {theta: mae} for every theta group."""
    results = {}
    for theta, data in sorted(theta_groups.items()):
        mae = evaluate_model(
            model, data, y_mean, y_std, x_means_SP, device,
            context_frac=context_frac,
            sensor_mask_fn=sensor_mask_fn,
            batch_size=batch_size,
            seed=seed,
        )
        results[theta] = mae
    return results

# =============================================================================
# Experiment 1.1 — Bidirectional OoD matrix
# =============================================================================

def run_exp_11(
    models: dict,        # {"lowvar": model, "highvar": model} or subset
    stats: dict,         # {"lowvar": (y_mean, y_std, x_means_SP), ...}
    test_data: dict,     # {"lowvar": test_list, "highvar": test_list}
    output_dir: Path,
    device: torch.device,
    context_frac: float = CONTEXT_FRAC_DEFAULT,
):
    print("\n[Exp 1.1] Bidirectional OoD matrix")
    exp_dir = output_dir / "exp_1.1_ood_matrix"
    exp_dir.mkdir(parents=True, exist_ok=True)

    domains      = [k for k in ["lowvar", "highvar"] if k in models]
    test_domains = [k for k in ["lowvar", "highvar"] if k in test_data]

    results = {}
    for m_domain in domains:
        model               = models[m_domain]
        y_mean, y_std, xm   = stats[m_domain]
        results[m_domain]   = {}
        for d_domain in test_domains:
            print(f"  model={m_domain}  data={d_domain} ...", end=" ", flush=True)
            mae = evaluate_model(
                model, test_data[d_domain],
                y_mean, y_std, xm,
                device, context_frac=context_frac,
            )
            results[m_domain][d_domain] = mae
            print(f"MAE = {mae:.4f} m")

    # --- CSV ---
    rows = []
    for m_d in domains:
        for d_d in test_domains:
            rows.append([m_d, d_d, results[m_d].get(d_d, float("nan"))])
    save_csv(exp_dir / "ood_matrix.csv",
             rows, ["model_domain", "data_domain", "mae"])

    # --- Heatmap ---
    mat  = np.array([[results[m].get(d, np.nan) for d in test_domains] for m in domains])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mat, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(test_domains)))
    ax.set_yticks(range(len(domains)))
    ax.set_xticklabels([f"data\n{d}" for d in test_domains])
    ax.set_yticklabels([f"model\n{d}" for d in domains])
    for i in range(len(domains)):
        for j in range(len(test_domains)):
            ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                    fontsize=11, color="black")
    plt.colorbar(im, ax=ax, label="MAE (m)")
    ax.set_title(f"OoD MAE matrix (ctx={context_frac*100:.0f}%)")
    plt.tight_layout()
    fig.savefig(exp_dir / "ood_matrix_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved → {exp_dir}")


# =============================================================================
# Experiment 1.2 — Per-theta degradation curve
# =============================================================================

def run_exp_12(
    models: dict,
    stats: dict,
    theta_groups: dict,   # {"lowvar": {theta: data}, "highvar": {theta: data}}
    output_dir: Path,
    device: torch.device,
    context_frac: float = CONTEXT_FRAC_DEFAULT,
):
    print("\n[Exp 1.2] Per-theta degradation curve")
    exp_dir = output_dir / "exp_1.2_theta_curve"
    exp_dir.mkdir(parents=True, exist_ok=True)

    domains     = [k for k in ["lowvar", "highvar"] if k in models]
    all_thetas  = sorted({
        th
        for d in theta_groups.values()
        for th in d.keys()
    })

    rows   = []
    curves = {}   # (m_domain, d_domain) -> {theta: mae}

    for m_domain in domains:
        model             = models[m_domain]
        y_mean, y_std, xm = stats[m_domain]
        for d_domain, tg in theta_groups.items():
            key    = (m_domain, d_domain)
            result = evaluate_by_theta(
                model, tg, y_mean, y_std, xm,
                device, context_frac=context_frac,
            )
            curves[key] = result
            for theta, mae in result.items():
                rows.append([m_domain, d_domain, theta, mae])
            print(f"  model={m_domain} data={d_domain}: "
                  + "  ".join(f"θ={t:.1f}→{v:.3f}" for t, v in sorted(result.items())))

    save_csv(exp_dir / "theta_degradation.csv",
             rows, ["model_domain", "data_domain", "theta", "mae"])

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    linestyles = {"lowvar": "-", "highvar": "--"}
    colors     = {"lowvar": "#2196F3", "highvar": "#F44336"}
    domain_labels = {"lowvar": "low-var data", "highvar": "high-var data"}

    for m_domain in domains:
        for d_domain in sorted(theta_groups.keys()):
            key = (m_domain, d_domain)
            if key not in curves:
                continue
            ordered = sorted(curves[key].items())
            ths  = [t for t, _ in ordered]
            maes = [v for _, v in ordered]
            ax.plot(ths, maes,
                    linestyle=linestyles.get(d_domain, "-"),
                    color=colors.get(m_domain, "gray"),
                    marker="o",
                    label=f"model={m_domain} / {domain_labels.get(d_domain, d_domain)}")

    ax.set_xlabel("Theta (channel variability)")
    ax.set_ylabel("MAE (m)")
    ax.set_title(f"Per-theta degradation (ctx={context_frac*100:.0f}%)")
    ax.axvline(x=0.35, color="gray", linestyle=":", linewidth=1.5, label="low/high-var boundary")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(exp_dir / "theta_degradation_curve.png", dpi=150)
    plt.close(fig)
    print(f"  Saved → {exp_dir}")

# =============================================================================
# Experiment 1.3 — Context fraction sweep
# =============================================================================

def run_exp_13(
    models: dict,
    stats: dict,
    test_data: dict,
    output_dir: Path,
    device: torch.device,
    context_fracs: list = CONTEXT_FRACS,
):
    print("\n[Exp 1.3] Context-fraction sweep (OoD mitigator)")
    exp_dir = output_dir / "exp_1.3_context_sweep"
    exp_dir.mkdir(parents=True, exist_ok=True)

    domains      = [k for k in ["lowvar", "highvar"] if k in models]
    test_domains = [k for k in ["lowvar", "highvar"] if k in test_data]

    rows   = []
    curves = {}   # (m_domain, d_domain) -> [mae per frac]

    for m_domain in domains:
        model             = models[m_domain]
        y_mean, y_std, xm = stats[m_domain]
        for d_domain in test_domains:
            key  = (m_domain, d_domain)
            maes = []
            for frac in tqdm(context_fracs,
                             desc=f"  ctx sweep model={m_domain} data={d_domain}",
                             leave=False):
                mae = evaluate_model(
                    model, test_data[d_domain],
                    y_mean, y_std, xm,
                    device, context_frac=frac,
                    eval_on_all_points=True, # fixed denominator for fair comparison
                )
                maes.append(mae)
                rows.append([m_domain, d_domain, frac, mae])
            curves[key] = maes
            print(f"  model={m_domain} data={d_domain}: "
                  + "  ".join(f"{f:.0%}→{v:.3f}" for f, v in zip(context_fracs, maes)))

    save_csv(exp_dir / "context_sweep.csv",
             rows, ["model_domain", "data_domain", "context_frac", "mae"])

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    linestyles = {"lowvar": "-", "highvar": "--"}
    colors     = {"lowvar": "#2196F3", "highvar": "#F44336"}

    for m_domain in domains:
        for d_domain in test_domains:
            key = (m_domain, d_domain)
            if key not in curves:
                continue
            ax.plot(
                [f * 100 for f in context_fracs],
                curves[key],
                linestyle=linestyles.get(d_domain, "-"),
                color=colors.get(m_domain, "gray"),
                marker="o",
                label=f"model={m_domain} / data={d_domain}",
            )

    ax.set_xlabel("Context fraction (%)")
    ax.set_ylabel("MAE (m) — all T points")
    ax.set_title("MAE vs context size — OoD mitigator analysis\n(MAE computed over all trajectory points, denominator fixed)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(exp_dir / "context_sweep.png", dpi=150)
    plt.close(fig)
    print(f"  Saved → {exp_dir}")

# =============================================================================
# Experiment 2.1 — Sensor restriction modes
# =============================================================================

def run_exp_21(
    models: dict,
    stats: dict,
    test_data: dict,
    output_dir: Path,
    device: torch.device,
    context_frac: float = CONTEXT_FRAC_DEFAULT,
    bernoulli_p_drops: list  = BERNOULLI_P_DROPS,
    k_active_values: list    = K_ACTIVE_VALUES,
    cluster_sizes: list      = CLUSTER_SIZES,
):
    """
    Sensor restriction sweep for ALL available models × their in-distribution
    test data.

    Modes:
      a) Bernoulli: p_drop ∈ bernoulli_p_drops
      b) k-uniform: k_active ∈ k_active_values
      c) Cluster: cluster_size (sensors removed) ∈ cluster_sizes
    """
    print("\n[Exp 2.1] Sensor restriction sweep")
    exp_dir = output_dir / "exp_2.1_sensor_restriction"
    exp_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    curves   = {}  # (model_domain, mode, param_label) -> mae

    for m_domain in [k for k in ["lowvar", "highvar"] if k in models]:
        model             = models[m_domain]
        y_mean, y_std, xm = stats[m_domain]
        td                = test_data.get(m_domain)
        if td is None:
            print(f"  Warning: no test data for {m_domain}, skipping restriction sweep")
            continue

        print(f"\n  Model: {m_domain}")

        # ---- a) Bernoulli ----
        for p_drop in bernoulli_p_drops:
            fn = (lambda B, dev, rng, _p=p_drop:
                  make_bernoulli_mask(B, _p, dev, rng))
            mae = evaluate_model(
                model, td, y_mean, y_std, xm,
                device, context_frac=context_frac,
                sensor_mask_fn=fn,
            )
            label = f"p={p_drop:.1f}"
            curves[(m_domain, "bernoulli", p_drop)] = mae
            all_rows.append([m_domain, "bernoulli", label, p_drop, mae])
            print(f"    bernoulli p_drop={p_drop:.1f}  MAE={mae:.4f} m")

        # ---- b) k-uniform ----
        for k_act in k_active_values:
            fn = (lambda B, dev, rng, _k=k_act:
                  make_k_uniform_mask(B, _k, dev, rng))
            mae = evaluate_model(
                model, td, y_mean, y_std, xm,
                device, context_frac=context_frac,
                sensor_mask_fn=fn,
            )
            label = f"k={k_act}"
            curves[(m_domain, "k_uniform", k_act)] = mae
            all_rows.append([m_domain, "k_uniform", label, k_act, mae])
            print(f"    k_uniform k_active={k_act}     MAE={mae:.4f} m")

        # ---- c) Cluster ----
        for cs in cluster_sizes:
            fn = (lambda B, dev, rng, _cs=cs:
                  make_cluster_mask(B, _cs, dev, rng))
            mae = evaluate_model(
                model, td, y_mean, y_std, xm,
                device, context_frac=context_frac,
                sensor_mask_fn=fn,
            )
            label = f"cluster={cs}"
            curves[(m_domain, "cluster", cs)] = mae
            all_rows.append([m_domain, "cluster", label, cs, mae])
            print(f"    cluster  size={cs}             MAE={mae:.4f} m")

    save_csv(exp_dir / "sensor_restriction.csv",
             all_rows,
             ["model_domain", "mode", "param_label", "param_value", "mae"])

    # --- Plots: one per model domain ---
    domains_present = sorted({r[0] for r in all_rows})
    for m_domain in domains_present:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Sensor restriction — model={m_domain} (ctx={context_frac*100:.0f}%)")

        # (a) Bernoulli
        ax = axes[0]
        ps   = bernoulli_p_drops
        maes = [curves.get((m_domain, "bernoulli", p), np.nan) for p in ps]
        ax.plot(ps, maes, "o-", color="#E91E63")
        ax.set_xlabel("Drop probability (Bernoulli)")
        ax.set_ylabel("MAE (m)")
        ax.set_title("(a) Bernoulli dropout")
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()   # 0=all sensors on the left

        # (b) k-uniform
        ax = axes[1]
        ks   = k_active_values
        maes = [curves.get((m_domain, "k_uniform", k), np.nan) for k in ks]
        ax.plot(ks, maes, "s-", color="#FF9800")
        ax.set_xlabel("Active sensors (k-uniform)")
        ax.set_title("(b) k-uniform dropout")
        ax.grid(True, alpha=0.3)

        # (c) Cluster
        ax = axes[2]
        css  = cluster_sizes
        maes = [curves.get((m_domain, "cluster", cs), np.nan) for cs in css]
        ax.plot(css, maes, "^-", color="#9C27B0")
        ax.set_xlabel("Cluster size removed")
        ax.set_title("(c) Cluster dropout (ellipsoidal)")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(exp_dir / f"sensor_restriction_{m_domain}.png", dpi=150)
        plt.close(fig)

    # --- Combined comparison plot (if both domains available) ---
    if len(domains_present) == 2:
        _plot_combined_restriction(curves, domains_present,
                                   bernoulli_p_drops, k_active_values, cluster_sizes,
                                   context_frac, exp_dir)

    print(f"  Saved → {exp_dir}")

def _plot_combined_restriction(curves, domains, p_drops, k_actives, cluster_sizes,
                                context_frac, exp_dir):
    """Overlaid comparison between two models for all three restriction modes."""
    colors = {"lowvar": "#2196F3", "highvar": "#F44336"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Sensor restriction: model comparison (ctx={context_frac*100:.0f}%)")

    for m_domain in domains:
        c = colors.get(m_domain, "gray")

        # (a) Bernoulli
        ps   = p_drops
        maes = [curves.get((m_domain, "bernoulli", p), np.nan) for p in ps]
        axes[0].plot(ps, maes, "o-", color=c, label=f"model={m_domain}")

        # (b) k-uniform
        ks   = k_actives
        maes = [curves.get((m_domain, "k_uniform", k), np.nan) for k in ks]
        axes[1].plot(ks, maes, "s-", color=c, label=f"model={m_domain}")

        # (c) Cluster
        css  = cluster_sizes
        maes = [curves.get((m_domain, "cluster", cs), np.nan) for cs in css]
        axes[2].plot(css, maes, "^-", color=c, label=f"model={m_domain}")

    axes[0].set_xlabel("Drop probability"); axes[0].set_ylabel("MAE (m)")
    axes[0].set_title("(a) Bernoulli"); axes[0].grid(True, alpha=0.3)
    axes[0].invert_xaxis(); axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Active sensors (k)"); axes[1].set_title("(b) k-uniform")
    axes[1].grid(True, alpha=0.3); axes[1].legend(fontsize=8)

    axes[2].set_xlabel("Cluster size removed"); axes[2].set_title("(c) Cluster")
    axes[2].grid(True, alpha=0.3); axes[2].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(exp_dir / "sensor_restriction_combined.png", dpi=150)
    plt.close(fig)

# =============================================================================
# Experiment 1.4 — Trajectory prediction plots per theta
# =============================================================================

def _predict_trajectory(
    model: LatentModel,
    x: np.ndarray,           # (T, Dx)
    y: np.ndarray,           # (T, 3)
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    x_means_SP: torch.Tensor,
    device: torch.device,
    context_frac: float,
) -> tuple:                  # returns (y_pred: ndarray (T,3), n_ctx: int)
    """Run a single trajectory through the model and return denormalised predictions."""
    x_t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)  # (1,T,Dx)
    y_t = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(0)  # (1,T,3)
    T   = x_t.shape[1]

    sensor_mask = make_all_sensors_mask(1, device)                 # all sensors on
    x_aug = augment_with_mask(x_t, sensor_mask, x_means_SP)       # (1,T,Dx+S)

    n_ctx   = max(1, min(T - 1, int(round(context_frac * T))))
    ctx_idx = torch.arange(n_ctx, device=device)
    tar_idx = torch.arange(T,     device=device)

    ctx_x = x_aug[:, ctx_idx, :]
    ctx_y = ((y_t - y_mean) / y_std)[:, ctx_idx, :]
    tar_x = x_aug[:, tar_idx, :]

    with torch.no_grad():
        y_pred_norm, _, _, _, _ = model(ctx_x, ctx_y, tar_x)

    y_pred = (y_pred_norm * y_std + y_mean).squeeze(0).cpu().numpy()  # (T, 3)
    return y_pred, n_ctx


def run_exp_15(
    models: dict,         # {"lowvar": model, "highvar": model}
    stats: dict,          # {"lowvar": (y_mean, y_std, x_means_SP), ...}
    theta_groups: dict,   # {"lowvar": {theta: [samples]}, "highvar": {theta: [samples]}}
    output_dir: Path,
    device: torch.device,
    context_frac: float = CONTEXT_FRAC_DEFAULT,
    n_traj: int = 4,
    seed: int = EVAL_SEED,
):
    """
    For each theta level, plot n_traj randomly selected trajectories.
    Each subplot shows ground truth + predictions from all available models.
    Produces one figure per theta, saved as theta_{theta:.1f}.png.
    """
    print("\n[Exp 1.4] Trajectory prediction plots per theta")
    exp_dir = output_dir / "exp_1.4_trajectory_plots"
    exp_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # Build a flat {theta: [samples]} dict spanning both domains
    all_theta_groups: dict = {}
    for domain_groups in theta_groups.values():
        for theta, samples in domain_groups.items():
            all_theta_groups.setdefault(theta, []).extend(samples)

    model_colors = {"lowvar": "#2196F3", "highvar": "#F44336"}
    model_labels = {"lowvar": "ANP low-var", "highvar": "ANP high-var"}
    domain_names = sorted(models.keys())   # models available

    for theta in sorted(all_theta_groups.keys()):
        pool = all_theta_groups[theta]
        chosen_idx = rng.choice(len(pool),
                                size=min(n_traj, len(pool)),
                                replace=False)
        chosen = [pool[i] for i in chosen_idx]

        ncols = 2
        nrows = (len(chosen) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(7 * ncols, 6 * nrows),
                                 squeeze=False)
        fig.suptitle(
            f"Trajectory predictions — θ={theta:.1f}  (context={context_frac*100:.0f}%)",
            fontsize=14, y=1.01,
        )

        for k, (x_np, y_np) in enumerate(chosen):
            ax  = axes[k // ncols][k % ncols]
            row = k // ncols
            col = k % ncols

            # Compute n_ctx for reference lines
            T     = y_np.shape[0]
            n_ctx = max(1, min(T - 1, int(round(context_frac * T))))

            # Ground truth
            ax.plot(y_np[:, 0], y_np[:, 1],
                    color="black", linewidth=1.8, zorder=3, label="Ground truth")
            # Context region
            ax.plot(y_np[:n_ctx, 0], y_np[:n_ctx, 1],
                    color="black", linewidth=4.0, alpha=0.35, zorder=2,
                    label=f"Context ({context_frac*100:.0f}%)")
            # Start / end markers
            ax.plot(y_np[0,  0], y_np[0,  1], "go", markersize=8,  zorder=5, label="Start")
            ax.plot(y_np[-1, 0], y_np[-1, 1], "ks", markersize=8,  zorder=5, label="End")

            # Model predictions
            for m_domain in domain_names:
                y_mean, y_std, xm = stats[m_domain]
                y_pred, _         = _predict_trajectory(
                    models[m_domain], x_np, y_np,
                    y_mean, y_std, xm,
                    device, context_frac,
                )
                # MAE for this trajectory
                mae_traj = float(np.mean(np.abs(y_pred - y_np)))
                ax.plot(
                    y_pred[:, 0], y_pred[:, 1],
                    color=model_colors.get(m_domain, "gray"),
                    linewidth=1.5, linestyle="--", zorder=4,
                    label=f"{model_labels.get(m_domain, m_domain)}  (MAE={mae_traj:.2f} m)",
                )

            ax.set_xlabel("x (m)", fontsize=9)
            ax.set_ylabel("y (m)", fontsize=9)
            ax.set_title(f"Trajectory {k+1}", fontsize=10)
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7.5, loc="best")

        # Hide any unused axes
        for k in range(len(chosen), nrows * ncols):
            axes[k // ncols][k % ncols].set_visible(False)

        plt.tight_layout()
        fname = exp_dir / f"theta_{theta:.1f}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  theta={theta:.1f} → {fname}")

    print(f"  Saved → {exp_dir}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Eval masked ANP: OoD + sensor restriction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--lowvar-ckpt", type=Path, default=None, help="Path to best_checkpoint.pth.tar of the low-variance masked ANP.")
    p.add_argument("--highvar-ckpt", type=Path, default=None, help="Path to best_checkpoint.pth.tar of the high-variance masked ANP.")
    p.add_argument("--lowvar-data-dir", type=Path, default=None, help="Root data dir for low-variance (contains topology_<X>/ subdirs).")
    p.add_argument("--highvar-data-dir", type=Path, default=None, help="Root data dir for high-variance.")
    p.add_argument("--topology", type=str,  default="ellipsoidal", choices=["ellipsoidal", "aligned", "random"])
    p.add_argument("--output-dir", type=Path, default=Path("results/eval_ood_sensor_restriction"))
    p.add_argument("--context-frac", type=float, default=CONTEXT_FRAC_DEFAULT)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=EVAL_SEED)
    p.add_argument("--experiments", type=str, default="1.1,1.2,1.3,2.1", help="Comma-separated list of experiments to run: 1.1 1.2 1.3 1.4 2.1",)
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    print(f"Device: {device}")

    experiments = set(args.experiments.replace(" ", "").split(","))
    print(f"Running experiments: {sorted(experiments)}")

    topo = args.topology
    print(f"Topology: {topo}")

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    models = {}
    stats  = {}

    for domain, ckpt_path, data_dir in [
        ("lowvar",  args.lowvar_ckpt,  args.lowvar_data_dir),
        ("highvar", args.highvar_ckpt, args.highvar_data_dir),
    ]:
        if ckpt_path is None:
            print(f"  Skipping model '{domain}': no checkpoint provided.")
            continue
        if data_dir is None:
            print(f"  Skipping model '{domain}': no data dir provided (needed for stats).")
            continue

        print(f"\nLoading {domain} model…")
        model = load_masked_anp(ckpt_path, device)
        models[domain] = model

        print(f"  Computing training stats for {domain}…")
        train_data, _, _, _ = load_topology_split(data_dir, topo)
        y_mean, y_std, xm   = compute_train_stats(train_data, device)
        stats[domain]        = (y_mean, y_std, xm)
        print(f"  y_mean={y_mean.cpu().numpy()}, y_std={y_std.cpu().numpy()}")

    if not models:
        print("No models loaded. Provide at least one of --lowvar-ckpt / --highvar-ckpt.")
        return

    # ------------------------------------------------------------------
    # Load test data (from each domain, using the provided data dirs)
    # ------------------------------------------------------------------
    test_data    = {}
    theta_groups = {}

    for domain, data_dir in [
        ("lowvar",  args.lowvar_data_dir),
        ("highvar", args.highvar_data_dir),
    ]:
        if data_dir is None:
            continue
        _, _, td, meta = load_topology_split(data_dir, topo)
        test_data[domain]    = td
        theta_groups[domain] = group_test_data_by_theta(td, meta)
        print(f"  Test data {domain}: {len(td)} trajectories | "
              f"thetas = {sorted(theta_groups[domain].keys())}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Run experiments
    # ------------------------------------------------------------------
    ctx = args.context_frac

    if "1.1" in experiments:
        run_exp_11(models, stats, test_data, args.output_dir, device, context_frac=ctx)

    if "1.2" in experiments:
        run_exp_12(models, stats, theta_groups, args.output_dir, device, context_frac=ctx)

    if "1.3" in experiments:
        run_exp_13(models, stats, test_data, args.output_dir, device)

    if "2.1" in experiments:
        run_exp_21(models, stats, test_data, args.output_dir, device, context_frac=ctx)

    if "1.4" in experiments:
        run_exp_15(models, stats, theta_groups, args.output_dir, device, context_frac=ctx)

    print(f"\nAll results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
