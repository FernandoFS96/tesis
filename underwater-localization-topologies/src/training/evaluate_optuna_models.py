"""
evaluate_optuna_models.py
=========================
Example script showing how to use load_optuna_model.py to evaluate ANP and RANP models trained with Optuna HPO.

Loads both models, runs evaluation on the val set for each context fraction [0.2, 0.4, 0.6], and reports per-fraction and mean MAE.

Usage
-----
Run from the project root (underwater-localization-topologies/):

python -m src.training.evaluate_optuna_models \
    --run-all \
    --device cuda \
    --optuna-results-root src/training/results/optuna \
    --data-root data/data \
    --output-dir src/training/results/optuna/models_evaluation \
    --boxplot-split test \
    --context-fracs 0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8\
    --context-frac 0.3

Either --anp-dir or --ranp-dir can be omitted if only one model is available.
"""

from __future__ import annotations

import argparse
import os
import pickle
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader, TensorDataset

DEFAULT_OUTPUT_DIR = "src/training/results/optuna/models_evaluation"

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


def _build_eval_indices(
    total_points: int,
    context_frac: float,
    device: str | torch.device,
    eval_protocol: str,
    holdout_frac: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create context and target indices according to the selected evaluation protocol.

    Protocols:
    - legacy: target is all post-context points [ctx, T).
    - fixed_holdout: target is fixed tail [T-n_holdout, T), and context is capped
      to stay strictly before that tail (no overlap by construction).
    - online_rolling: target is the immediate next point after context (1-step ahead).
    """
    if eval_protocol == "legacy":
        ctx_size = max(1, min(total_points - 1, int(round(context_frac * total_points))))
        ctx_idx = torch.arange(ctx_size, device=device)
        tar_idx = torch.arange(ctx_size, total_points, device=device)
        return ctx_idx, tar_idx

    if eval_protocol == "fixed_holdout":
        n_holdout = max(1, int(round(holdout_frac * total_points)))
        holdout_start = total_points - n_holdout
        # Keep context strictly before holdout_start to avoid any overlap.
        max_ctx = max(1, holdout_start - 1)
        ctx_size = max(1, min(max_ctx, int(round(context_frac * total_points))))
        ctx_idx = torch.arange(ctx_size, device=device)
        tar_idx = torch.arange(holdout_start, total_points, device=device)
        return ctx_idx, tar_idx

    if eval_protocol == "online_rolling":
        # Evaluate pure 1-step-ahead prediction after the context prefix.
        max_ctx = max(1, total_points - 2)
        ctx_size = max(1, min(max_ctx, int(round(context_frac * total_points))))
        ctx_idx = torch.arange(ctx_size, device=device)
        tar_idx = torch.arange(ctx_size, min(total_points, ctx_size + 1), device=device)
        return ctx_idx, tar_idx

    raise ValueError(f"Unknown eval_protocol: {eval_protocol}")


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
    eval_protocol: str,
    holdout_frac: float,
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
                ctx_idx, tar_idx = _build_eval_indices(
                    total_points=T,
                    context_frac=frac,
                    device=device,
                    eval_protocol=eval_protocol,
                    holdout_frac=holdout_frac,
                )

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


def evaluate_model_distribution(
    model,
    model_type: str,
    val_loader,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    x_means_SP: torch.Tensor,
    num_time_points: int,
    num_sensors: int,
    context_frac: float,
    eval_protocol: str,
    holdout_frac: float,
    device: str | torch.device,
) -> list[float]:
    """Return per-trajectory MAE distribution for a fixed context fraction."""
    y_mean = y_mean.to(device)
    y_std = y_std.to(device)
    x_means_SP = x_means_SP.to(device)

    maes = []
    model.eval()

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            B, T, _ = x_batch.shape

            sensor_mask = torch.ones(B, num_sensors, device=device)
            x_aug = _apply_mask_and_append(
                x_batch, sensor_mask, x_means_SP, num_time_points, num_sensors
            )

            y_norm = (y_batch - y_mean) / y_std
            ctx_idx, tar_idx = _build_eval_indices(
                total_points=T,
                context_frac=context_frac,
                device=device,
                eval_protocol=eval_protocol,
                holdout_frac=holdout_frac,
            )

            context_y = y_norm[:, ctx_idx, :]

            if model_type == "anp":
                context_x = x_aug[:, ctx_idx, :]
                target_x = x_aug[:, tar_idx, :]
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
            y_true = y_batch[:, tar_idx, :]
            # Per-trajectory MAE across time and xyz dimensions.
            per_traj = torch.mean(torch.abs(y_pred - y_true), dim=(1, 2))
            maes.extend(per_traj.detach().cpu().tolist())

    return maes


def evaluate_model_rollout5(
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
) -> dict[float, dict[str, float]]:
    """Always-on fixed rollout evaluation.

    For each context fraction, evaluate MAE for the next 5 points after context:
      step1..step5 and their mean.
    Context is capped so that all 5 rollout steps exist (no overlap by design).
    """
    y_mean = y_mean.to(device)
    y_std = y_std.to(device)
    x_means_SP = x_means_SP.to(device)

    sums = {
        f: {"step1": 0.0, "step2": 0.0, "step3": 0.0, "step4": 0.0, "step5": 0.0}
        for f in context_fracs
    }
    n_batches = 0

    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            B, T, _ = x_batch.shape

            sensor_mask = torch.ones(B, num_sensors, device=device)
            x_aug = _apply_mask_and_append(
                x_batch, sensor_mask, x_means_SP, num_time_points, num_sensors
            )
            y_norm = (y_batch - y_mean) / y_std

            # Need at least one context point + 5 future points.
            max_ctx = max(1, T - 6)

            for frac in context_fracs:
                ctx_size = max(1, min(max_ctx, int(round(frac * T))))
                ctx_idx = torch.arange(ctx_size, device=device)
                tar_idx = torch.arange(ctx_size, ctx_size + 5, device=device)

                context_y = y_norm[:, ctx_idx, :]

                if model_type == "anp":
                    context_x = x_aug[:, ctx_idx, :]
                    target_x = x_aug[:, tar_idx, :]
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

                y_pred = y_pred_norm * y_std + y_mean  # (B,5,3)
                y_true = y_batch[:, tar_idx, :]       # (B,5,3)

                abs_err = torch.abs(y_pred - y_true)  # (B,5,3)
                for step_i in range(5):
                    step_key = f"step{step_i + 1}"
                    step_mae = abs_err[:, step_i, :].mean().item()
                    sums[frac][step_key] += step_mae

            n_batches += 1

    out = {}
    for frac in context_fracs:
        step_vals = {
            k: sums[frac][k] / max(1, n_batches)
            for k in ["step1", "step2", "step3", "step4", "step5"]
        }
        step_vals["mean"] = float(np.mean(list(step_vals.values())))
        out[frac] = step_vals

    return out


def _plot_boxplot_by_variance(
    boxplot_store: dict,
    save_path: str,
    context_frac: float,
    split_name: str,
) -> None:
    """Create a 2-panel boxplot (lowvar/highvar) with ANP vs RANP per topology."""
    variances = ["lowvar", "highvar"]
    topologies = ["aligned", "ellipsoidal", "random"]
    model_order = ["anp", "ranp"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    colors = {"anp": "#1f77b4", "ranp": "#ff7f0e"}

    for ax, variance in zip(axes, variances):
        centers = np.arange(len(topologies), dtype=float)
        offsets = {"anp": -0.18, "ranp": 0.18}
        width = 0.32

        for topo_idx, topo in enumerate(topologies):
            for model in model_order:
                values = boxplot_store[variance][topo].get(model, [])
                if len(values) == 0:
                    continue
                pos = centers[topo_idx] + offsets[model]
                bp = ax.boxplot(
                    [values],
                    positions=[pos],
                    widths=width,
                    patch_artist=True,
                    showfliers=False,
                    medianprops={"color": "black", "linewidth": 1.4},
                )
                bp["boxes"][0].set_facecolor(colors[model])
                bp["boxes"][0].set_alpha(0.6)

        ax.set_title(f"{variance} ({split_name})")
        ax.set_xticks(centers)
        ax.set_xticklabels(topologies)
        ax.set_xlabel("Topology")
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("MAE (m)")
    handles = [
        Line2D([0], [0], color=colors["anp"], lw=8, alpha=0.6, label="ANP"),
        Line2D([0], [0], color=colors["ranp"], lw=8, alpha=0.6, label="RANP"),
    ]
    axes[1].legend(handles=handles, loc="upper right")

    fig.suptitle(f"Per-trajectory MAE boxplots at context={int(round(context_frac * 100))}%")
    fig.tight_layout()

    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def _plot_mae_heatmaps_by_split(all_rows: list[dict], save_path: str) -> None:
    """Plot MAE heatmaps with axes topology x (model,variance), split into val/test."""
    if len(all_rows) == 0:
        return

    split_order = ["val", "test"]
    splits = [s for s in split_order if any(r["split"] == s for r in all_rows)]
    if len(splits) == 0:
        return

    topologies = ["aligned", "ellipsoidal", "random"]
    col_pairs = [("anp", "lowvar"), ("ranp", "lowvar"), ("anp", "highvar"), ("ranp", "highvar")]
    col_labels = [f"{m}-{v}" for m, v in col_pairs]

    fig, axes = plt.subplots(
        1,
        len(splits),
        figsize=(7.2 * len(splits), 5.8),
        sharey=True,
        constrained_layout=True,
    )
    if len(splits) == 1:
        axes = [axes]

    vmin = min(r["mean_mae"] for r in all_rows)
    vmax = max(r["mean_mae"] for r in all_rows)

    for ax, split in zip(axes, splits):
        mat = np.full((len(topologies), len(col_pairs)), np.nan, dtype=float)
        for i, topo in enumerate(topologies):
            for j, (model, variance) in enumerate(col_pairs):
                candidates = [
                    r["mean_mae"]
                    for r in all_rows
                    if r["split"] == split and r["topology"] == topo and r["model"] == model and r["variance"] == variance
                ]
                if len(candidates) > 0:
                    mat[i, j] = float(candidates[0])

        im = ax.imshow(mat, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(f"Mean MAE ({split})")
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=30, ha="right")
        ax.set_yticks(np.arange(len(topologies)))
        ax.set_yticklabels(topologies)
        ax.set_xlabel("(model, variance)")

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                txt = "-" if np.isnan(mat[i, j]) else f"{mat[i, j]:.3f}"
                ax.text(j, i, txt, ha="center", va="center", color="white", fontsize=9)

    axes[0].set_ylabel("Topology")
    cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02, shrink=0.95)
    cbar.set_label("MAE (m)")

    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def _plot_delta_heatmaps_by_split(all_rows: list[dict], save_path: str) -> None:
    """Plot delta heatmaps (RANP - ANP) by topology x variance, split into val/test."""
    if len(all_rows) == 0:
        return

    split_order = ["val", "test"]
    splits = [s for s in split_order if any(r["split"] == s for r in all_rows)]
    if len(splits) == 0:
        return

    topologies = ["aligned", "ellipsoidal", "random"]
    variances = ["lowvar", "highvar"]

    # Build all deltas first to set a symmetric color scale around 0.
    all_deltas = []
    for split in splits:
        for topo in topologies:
            for var in variances:
                ranp = [
                    r["mean_mae"]
                    for r in all_rows
                    if r["split"] == split and r["topology"] == topo and r["variance"] == var and r["model"] == "ranp"
                ]
                anp = [
                    r["mean_mae"]
                    for r in all_rows
                    if r["split"] == split and r["topology"] == topo and r["variance"] == var and r["model"] == "anp"
                ]
                if len(ranp) > 0 and len(anp) > 0:
                    all_deltas.append(float(ranp[0] - anp[0]))

    if len(all_deltas) == 0:
        return

    abs_max = max(abs(min(all_deltas)), abs(max(all_deltas)))

    fig, axes = plt.subplots(
        1,
        len(splits),
        figsize=(7.2 * len(splits), 5.8),
        sharey=True,
        constrained_layout=True,
    )
    if len(splits) == 1:
        axes = [axes]

    for ax, split in zip(axes, splits):
        mat = np.full((len(topologies), len(variances)), np.nan, dtype=float)
        for i, topo in enumerate(topologies):
            for j, var in enumerate(variances):
                ranp = [
                    r["mean_mae"]
                    for r in all_rows
                    if r["split"] == split and r["topology"] == topo and r["variance"] == var and r["model"] == "ranp"
                ]
                anp = [
                    r["mean_mae"]
                    for r in all_rows
                    if r["split"] == split and r["topology"] == topo and r["variance"] == var and r["model"] == "anp"
                ]
                if len(ranp) > 0 and len(anp) > 0:
                    mat[i, j] = float(ranp[0] - anp[0])

        im = ax.imshow(mat, cmap="coolwarm", vmin=-abs_max, vmax=abs_max, aspect="auto")
        ax.set_title(f"Delta MAE (RANP - ANP) ({split})")
        ax.set_xticks(np.arange(len(variances)))
        ax.set_xticklabels(variances)
        ax.set_yticks(np.arange(len(topologies)))
        ax.set_yticklabels(topologies)
        ax.set_xlabel("Variance")

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                txt = "-" if np.isnan(mat[i, j]) else f"{mat[i, j]:+.3f}"
                ax.text(j, i, txt, ha="center", va="center", color="black", fontsize=9)

    axes[0].set_ylabel("Topology")
    cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02, shrink=0.95)
    cbar.set_label("Delta MAE (m), negative means RANP better")

    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def _plot_cdf_mae_by_scenario(cdf_records: list[dict], save_path: str, context_frac: float, split_name: str) -> None:
    """Plot CDF of per-trajectory MAE for each scenario on a given split."""
    if len(cdf_records) == 0:
        return

    selected = [r for r in cdf_records if r["split"] == split_name]
    if len(selected) == 0:
        selected = cdf_records

    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    for rec in selected:
        values = np.asarray(rec["values"], dtype=float)
        if values.size == 0:
            continue
        x = np.sort(values)
        y = np.arange(1, x.size + 1) / x.size
        label = f"{rec['model']}-{rec['variance']}-{rec['topology']}"
        ax.plot(x, y, linewidth=1.2, alpha=0.9, label=label)

    ax.set_title(f"CDF of MAE by scenario ({split_name}, ctx={int(round(context_frac * 100))}%)")
    ax.set_xlabel("MAE (m)")
    ax.set_ylabel("CDF")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def _plot_barplot_ci_mean_mae(cdf_records: list[dict], save_path: str, context_frac: float, split_name: str) -> None:
    """Barplot of mean MAE with bootstrap 95% CI for each scenario."""
    if len(cdf_records) == 0:
        return

    selected = [r for r in cdf_records if r["split"] == split_name]
    if len(selected) == 0:
        selected = cdf_records

    labels, means, low_err, up_err = [], [], [], []
    rng = np.random.default_rng(1234)

    def _sort_key(rec: dict):
        return (rec["variance"], rec["topology"], rec["model"])

    for rec in sorted(selected, key=_sort_key):
        values = np.asarray(rec["values"], dtype=float)
        if values.size == 0:
            continue
        mean_val = float(values.mean())

        boots = []
        for _ in range(500):
            sample = rng.choice(values, size=values.size, replace=True)
            boots.append(float(sample.mean()))
        lo = float(np.percentile(boots, 2.5))
        hi = float(np.percentile(boots, 97.5))

        labels.append(f"{rec['model']}-{rec['variance']}-{rec['topology']}")
        means.append(mean_val)
        low_err.append(mean_val - lo)
        up_err.append(hi - mean_val)

    if len(labels) == 0:
        return

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(12, 0.8 * len(labels)), 6), constrained_layout=True)
    ax.bar(
        x,
        means,
        yerr=np.vstack([low_err, up_err]),
        capsize=3,
        alpha=0.85,
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Mean MAE (m)")
    ax.set_title(f"Mean MAE with 95% CI ({split_name}, ctx={int(round(context_frac * 100))}%)")
    ax.grid(axis="y", alpha=0.3)

    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def _plot_scatter_optuna_vs_test(scenario_rows: list[dict], save_path: str, context_fracs: list[float]) -> None:
    """Scatter Optuna MAE (trial_info) vs test MAE for each scenario."""
    valid = [
        r for r in scenario_rows
        if r["split"] == "test" and np.isfinite(r.get("optuna_mae", np.nan))
    ]
    if len(valid) == 0:
        return

    markers = {"anp": "o", "ranp": "s"}
    colors = {"lowvar": "#1f77b4", "highvar": "#ff7f0e", "custom": "#2ca02c"}

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    for rec in valid:
        x = rec["optuna_mae"]
        y = rec["mean_mae"]
        ax.scatter(
            x,
            y,
            marker=markers.get(rec["model"], "o"),
            c=colors.get(rec["variance"], "gray"),
            s=70,
            alpha=0.9,
            edgecolors="black",
            linewidths=0.4,
        )
        ax.annotate(
            f"{rec['model']}-{rec['variance']}-{rec['topology']}",
            (x, y),
            textcoords="offset points",
            xytext=(4, 3),
            fontsize=7,
            alpha=0.85,
        )

    all_vals = [min([v["optuna_mae"] for v in valid]), max([v["optuna_mae"] for v in valid]),
                min([v["mean_mae"] for v in valid]), max([v["mean_mae"] for v in valid])]
    lo, hi = min(all_vals), max(all_vals)
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1)

    ax.set_xlabel("MAE_optuna (trial_info)")
    if len(context_fracs) == 1:
        ax.set_ylabel(f"MAE_test_ctx{int(round(context_fracs[0] * 100))}")
    else:
        ctx_labels = ",".join(str(int(round(f * 100))) for f in context_fracs)
        ax.set_ylabel(f"MAE_test_mean_ctx[{ctx_labels}]")
    ax.set_title("Generalization Check: Optuna MAE vs Test MAE")
    ax.grid(alpha=0.3)

    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def _plot_context_topology_curves(all_rows: list[dict], context_fracs: list[float], save_path: str) -> None:
    """Plot MAE vs context size, faceted by variance (rows) and model (cols).

    - Color encodes topology.
    - Line style encodes split (val/test).
    """
    if len(all_rows) == 0:
        return

    model_order = ["anp", "ranp"]
    variance_order = ["lowvar", "highvar"]
    split_order = ["val", "test"]
    topology_order = ["aligned", "ellipsoidal", "random"]

    x = np.array([100.0 * f for f in context_fracs], dtype=float)
    if x.size == 0:
        return

    colors = {"aligned": "#1f77b4", "ellipsoidal": "#ff7f0e", "random": "#2ca02c"}
    linestyles = {"val": "--", "test": "-"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True, constrained_layout=True)

    for r, variance in enumerate(variance_order):
        for c, model in enumerate(model_order):
            ax = axes[r, c]
            panel_rows = [row for row in all_rows if row["variance"] == variance and row["model"] == model]

            for topo in topology_order:
                for split in split_order:
                    candidates = [
                        row for row in panel_rows
                        if row["topology"] == topo and row["split"] == split
                    ]
                    if len(candidates) == 0:
                        continue
                    row = candidates[0]
                    y = [row.get(f"mae_ctx_{int(round(fr * 100))}", np.nan) for fr in context_fracs]
                    ax.plot(
                        x,
                        y,
                        color=colors[topo],
                        linestyle=linestyles[split],
                        linewidth=1.8,
                        marker="o",
                        markersize=3,
                        alpha=0.9,
                    )

            ax.set_title(f"{model.upper()} | {variance}")
            ax.grid(alpha=0.3)

    for ax in axes[1, :]:
        ax.set_xlabel("Context size (%)")
    for ax in axes[:, 0]:
        ax.set_ylabel("MAE (m)")

    topo_handles = [
        Line2D([0], [0], color=colors[t], lw=2.0, label=f"topology: {t}")
        for t in topology_order
    ]
    split_handles = [
        Line2D([0], [0], color="black", lw=2.0, linestyle=linestyles[s], label=f"split: {s}")
        for s in split_order
    ]
    fig.legend(handles=topo_handles + split_handles, loc="upper right", ncol=1, frameon=False)
    fig.suptitle("MAE vs Context Size by Topology, Model, and Variance", y=1.02)

    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def _default_study_name(model_type: str, variance: str, topology: str) -> str:
    return f"{model_type}_masked_{variance}_{topology}_v1"


def _resolve_best_model_dir(optuna_root: str, model_type: str, variance: str, topology: str) -> str:
    study = _default_study_name(model_type, variance, topology)
    return os.path.join(optuna_root, study, "best_model")


def _resolve_output_path(output_dir: str, path_or_name: str | None, default_filename: str) -> str:
    """Force outputs to be stored under *output_dir*.

    If a custom name/path is provided, only its basename is used.
    """
    filename = default_filename if path_or_name is None else os.path.basename(path_or_name)
    return os.path.join(output_dir, filename)


def _evaluate_one_configuration(
    model_label: str,
    model_dir: str,
    topology: str,
    data_dir: str,
    args,
    context_fracs,
    boxplot_split: str = "test",
    context_frac: float = 0.4,
):
    from src.utils.load_optuna_model import load_optuna_best_model

    print(f"\n{'='*60}")
    print(f"Loading data (topology={topology}) from: {data_dir}")
    train_data, val_data, test_data = _load_topology_data(data_dir, topology)

    print(
        f"  train: {len(train_data)} | val: {len(val_data)}"
        + (f" | test: {len(test_data)}" if test_data is not None else " | test: not found")
    )

    y_mean, y_std = _compute_y_stats(train_data)
    x_means_SP = torch.tensor(
        _compute_x_sensor_means(train_data, args.num_time_points, args.num_sensors),
        dtype=torch.float32,
    )

    val_loader = _make_dataloader(val_data, args.batch_size)
    test_loader = _make_dataloader(test_data, args.batch_size) if test_data is not None else None

    print(f"Loading {model_label} model from: {model_dir}")
    model, hparams, meta = load_optuna_best_model(
        best_model_dir=model_dir,
        topology=topology,
        model_type="auto",
        num_sensors=args.num_sensors,
        num_time_points=args.num_time_points,
        output_dim=3,
        device=args.device,
    )

    n_params = sum(p.numel() for p in model.parameters())
    trial_num = meta["trial_number"] if meta else "?"
    trial_mae = meta.get("value", np.nan) if meta else np.nan
    try:
        trial_mae_float = float(trial_mae)
    except (TypeError, ValueError):
        trial_mae_float = np.nan
    print(f"  Trial: {trial_num} | Optuna MAE: {trial_mae} | Params: {n_params:,}")
    print(f"  Hparams: {hparams}")
    print(f"  Eval protocol: {args.eval_protocol} | holdout_frac: {args.holdout_frac}")

    model_type = "ranp" if model_label.lower() == "ranp" else "anp"
    eval_sets = [("val", val_loader)]
    if test_loader is not None:
        eval_sets.append(("test", test_loader))

    available_splits = {name for name, _ in eval_sets}
    selected_boxplot_split = boxplot_split if boxplot_split in available_splits else "val"

    out_rows = []
    boxplot_values = None
    for split_name, loader in eval_sets:
        print(f"\nEvaluating {model_label} on {split_name}...")
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
            eval_protocol=args.eval_protocol,
            holdout_frac=args.holdout_frac,
            device=args.device,
        )
        mean_mae = float(np.mean(list(mae_by_frac.values())))
        rollout5_by_frac = evaluate_model_rollout5(
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

        row = {
            "model": model_label.lower(),
            "topology": topology,
            "split": split_name,
            "eval_protocol": args.eval_protocol,
            "holdout_frac": args.holdout_frac,
            "mean_mae": mean_mae,
            "optuna_mae": trial_mae_float,
        }
        for frac in context_fracs:
            row[f"mae_ctx_{int(frac * 100)}"] = float(mae_by_frac[frac])
            p = int(frac * 100)
            row[f"mae_roll5_step1_ctx_{p}"] = float(rollout5_by_frac[frac]["step1"])
            row[f"mae_roll5_step2_ctx_{p}"] = float(rollout5_by_frac[frac]["step2"])
            row[f"mae_roll5_step3_ctx_{p}"] = float(rollout5_by_frac[frac]["step3"])
            row[f"mae_roll5_step4_ctx_{p}"] = float(rollout5_by_frac[frac]["step4"])
            row[f"mae_roll5_step5_ctx_{p}"] = float(rollout5_by_frac[frac]["step5"])
            row[f"mae_roll5_mean_ctx_{p}"] = float(rollout5_by_frac[frac]["mean"])
        out_rows.append(row)

        print(f"  {model_label} [{split_name}] mean MAE: {mean_mae:.4f} m")
        for frac in context_fracs:
            print(f"    ctx={int(frac * 100):3d}% -> {mae_by_frac[frac]:.4f} m")
        for frac in context_fracs:
            p = int(frac * 100)
            r = rollout5_by_frac[frac]
            print(
                f"    rollout5 ctx={p:3d}% -> "
                f"s1={r['step1']:.4f}, s2={r['step2']:.4f}, s3={r['step3']:.4f}, "
                f"s4={r['step4']:.4f}, s5={r['step5']:.4f}, mean={r['mean']:.4f}"
            )

        if split_name == selected_boxplot_split:
            boxplot_values = evaluate_model_distribution(
                model=model,
                model_type=model_type,
                val_loader=loader,
                y_mean=y_mean,
                y_std=y_std,
                x_means_SP=x_means_SP,
                num_time_points=args.num_time_points,
                num_sensors=args.num_sensors,
                context_frac=context_frac,
                eval_protocol=args.eval_protocol,
                holdout_frac=args.holdout_frac,
                device=args.device,
            )

    return out_rows, boxplot_values, selected_boxplot_split


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate Optuna best ANP/RANP models")
    parser.add_argument("--topology", default="ellipsoidal", help="Topology name (e.g. ellipsoidal, aligned, random)")
    parser.add_argument("--data-dir", default="data/data/data_processed_topologies_low_variance", help="Path to the directory containing topology_<name>/ folders")
    parser.add_argument("--anp-dir", default=None, help="Path to ANP best_model/ dir (omit to skip ANP eval)")
    parser.add_argument("--ranp-dir", default=None, help="Path to RANP best_model/ dir (omit to skip RANP eval)")
    parser.add_argument("--run-all", action="store_true", help="Evaluate all combinations: model in {anp,ranp}, variance in {lowvar,highvar}, topology in {aligned,ellipsoidal,random}.")
    parser.add_argument("--optuna-results-root", default="src/training/results/optuna", help="Root containing Optuna study folders when --run-all is used.")
    parser.add_argument("--data-root", default="data/data", help="Root containing data_processed_topologies_low_variance and data_processed_topologies_high_variance when --run-all is used.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory where CSV and plots are saved.")
    parser.add_argument("--save-csv", default=None, help="Optional CSV filename override (stored inside --output-dir).")
    parser.add_argument("--save-boxplot", default=None, help="Optional boxplot filename override (stored inside --output-dir).")
    parser.add_argument("--save-heatmap", default=None, help="Optional MAE heatmap filename override (stored inside --output-dir).")
    parser.add_argument("--save-delta-heatmap", default=None, help="Optional delta heatmap filename override (stored inside --output-dir).")
    parser.add_argument("--save-cdf", default=None, help="Optional CDF filename override (stored inside --output-dir).")
    parser.add_argument("--save-ci-barplot", default=None, help="Optional CI barplot filename override (stored inside --output-dir).")
    parser.add_argument("--save-scatter", default=None, help="Optional scatter filename override (stored inside --output-dir).")
    parser.add_argument("--save-context-curves", default=None, help="Optional context-response curves filename override (stored inside --output-dir).")
    parser.add_argument("--boxplot-split", default="test", choices=["val", "test"], help="Which split to use for boxplot distributions.")
    parser.add_argument("--context-fracs", default=None, help="Comma-separated context fractions for table/CSV metrics, e.g. 0.2,0.3,0.4,0.6. If omitted, uses --context-frac.")
    parser.add_argument("--context-frac", type=float, default=0.4, help="Context fraction used by all metrics and plots (default: 0.4).")
    parser.add_argument("--eval-protocol", default="fixed_holdout", choices=["fixed_holdout", "online_rolling", "legacy"], help="Evaluation protocol: fixed_holdout (no overlap), online_rolling (1-step ahead), or legacy (post-context tail).")
    parser.add_argument("--holdout-frac", type=float, default=0.2, help="Fraction reserved as fixed target tail when --eval-protocol=fixed_holdout.")
    parser.add_argument("--device", default="cpu", help="Torch device: cpu | cuda | cuda:0 ...")
    parser.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size.")
    parser.add_argument("--num-sensors", type=int, default=10, help="Number of sensors.")
    parser.add_argument("--num-time-points", type=int, default=201, help="Number of time points.")
    args = parser.parse_args()

    if args.context_fracs is not None:
        context_fracs = [float(x.strip()) for x in args.context_fracs.split(",") if x.strip()]
    else:
        context_fracs = [args.context_frac]
    for frac in context_fracs:
        if not (0.0 < frac < 1.0):
            raise ValueError(f"Invalid context fraction {frac}. Expected values in (0,1).")
    if not (0.0 < args.holdout_frac < 1.0):
        raise ValueError(f"Invalid holdout fraction {args.holdout_frac}. Expected values in (0,1).")

    ctx_pct = int(round(args.context_frac * 100))

    os.makedirs(args.output_dir, exist_ok=True)

    csv_path = _resolve_output_path(
        args.output_dir,
        args.save_csv,
        "all_eval_summary.csv",
    )
    boxplot_path = _resolve_output_path(
        args.output_dir,
        args.save_boxplot,
        f"boxplot_ctx{ctx_pct}_{args.boxplot_split}.png",
    )
    heatmap_path = _resolve_output_path(
        args.output_dir,
        args.save_heatmap,
        f"mae_heatmap_ctx{ctx_pct}_{args.boxplot_split}.png",
    )
    delta_heatmap_path = _resolve_output_path(
        args.output_dir,
        args.save_delta_heatmap,
        f"delta_heatmap_ranp_minus_anp_ctx{ctx_pct}_{args.boxplot_split}.png",
    )
    cdf_path = _resolve_output_path(
        args.output_dir,
        args.save_cdf,
        f"cdf_mae_by_scenario_ctx{ctx_pct}_{args.boxplot_split}.png",
    )
    ci_barplot_path = _resolve_output_path(
        args.output_dir,
        args.save_ci_barplot,
        f"barplot_ci_mean_mae_ctx{ctx_pct}_{args.boxplot_split}.png",
    )
    scatter_path = _resolve_output_path(
        args.output_dir,
        args.save_scatter,
        f"scatter_optuna_vs_test_ctx{ctx_pct}.png",
    )
    context_curves_path = _resolve_output_path(
        args.output_dir,
        args.save_context_curves,
        "context_curves_topology_model_variance.png",
    )

    all_rows = []
    boxplot_store = defaultdict(lambda: defaultdict(dict))
    cdf_records = []

    if args.run_all:
        model_types = ["anp", "ranp"]
        variances = ["lowvar", "highvar"]
        topologies = ["aligned", "ellipsoidal", "random"]

        for model_type in model_types:
            for variance in variances:
                if variance == "lowvar":
                    data_dir = os.path.join(args.data_root, "data_processed_topologies_low_variance")
                else:
                    data_dir = os.path.join(args.data_root, "data_processed_topologies_high_variance")

                for topology in topologies:
                    model_dir = _resolve_best_model_dir(
                        args.optuna_results_root,
                        model_type=model_type,
                        variance=variance,
                        topology=topology,
                    )
                    try:
                        rows, box_values, used_split = _evaluate_one_configuration(
                            model_label=model_type.upper(),
                            model_dir=model_dir,
                            topology=topology,
                            data_dir=data_dir,
                            args=args,
                            context_fracs=context_fracs,
                            boxplot_split=args.boxplot_split,
                            context_frac=args.context_frac,
                        )
                        for row in rows:
                            row["variance"] = variance
                            all_rows.append(row)
                        if box_values is not None:
                            boxplot_store[variance][topology][model_type] = box_values
                            cdf_records.append(
                                {
                                    "model": model_type,
                                    "variance": variance,
                                    "topology": topology,
                                    "split": used_split,
                                    "values": box_values,
                                }
                            )
                    except FileNotFoundError as exc:
                        print(f"\nSkipping {model_type.upper()}-{variance}-{topology}: {exc}")
    else:
        if args.anp_dir is None and args.ranp_dir is None:
            parser.error("Provide at least one of --anp-dir or --ranp-dir, or use --run-all")

        runs = []
        if args.anp_dir is not None:
            runs.append(("ANP", args.anp_dir))
        if args.ranp_dir is not None:
            runs.append(("RANP", args.ranp_dir))

        for label, model_dir in runs:
            try:
                rows, box_values, used_split = _evaluate_one_configuration(
                    model_label=label,
                    model_dir=model_dir,
                    topology=args.topology,
                    data_dir=args.data_dir,
                    args=args,
                    context_fracs=context_fracs,
                    boxplot_split=args.boxplot_split,
                    context_frac=args.context_frac,
                )
                for row in rows:
                    row["variance"] = "custom"
                    all_rows.append(row)
                if box_values is not None:
                    cdf_records.append(
                        {
                            "model": label.lower(),
                            "variance": "custom",
                            "topology": args.topology,
                            "split": used_split,
                            "values": box_values,
                        }
                    )
            except FileNotFoundError as exc:
                print(f"\nSkipping {label}: {exc}")

    if len(all_rows) > 0:
        print(f"\n{'='*90}")
        print("Consolidated summary")
        ctx_headers = [f"ctx{int(round(f * 100)):02d}" for f in context_fracs]
        print("model | variance | topology | split | protocol | " + " | ".join(ctx_headers) + " | mean")
        print("-" * 90)
        for row in all_rows:
            ctx_vals = " | ".join(f"{row[f'mae_ctx_{int(round(f * 100))}']:.4f}" for f in context_fracs)
            print(
                f"{row['model']:<5} | {row['variance']:<8} | {row['topology']:<11} | {row['split']:<4} | {row['eval_protocol']:<12} | "
                f"{ctx_vals} | {row['mean_mae']:.4f}"
            )

    if len(all_rows) > 0:
        import csv

        out_dir = os.path.dirname(csv_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            ctx_fieldnames = [f"mae_ctx_{int(round(f * 100))}" for f in context_fracs]
            rollout_fieldnames = []
            for f in context_fracs:
                p = int(round(f * 100))
                rollout_fieldnames.extend(
                    [
                        f"mae_roll5_step1_ctx_{p}",
                        f"mae_roll5_step2_ctx_{p}",
                        f"mae_roll5_step3_ctx_{p}",
                        f"mae_roll5_step4_ctx_{p}",
                        f"mae_roll5_step5_ctx_{p}",
                        f"mae_roll5_mean_ctx_{p}",
                    ]
                )
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "model",
                    "variance",
                    "topology",
                    "split",
                    "eval_protocol",
                    "holdout_frac",
                    "optuna_mae",
                    *ctx_fieldnames,
                    *rollout_fieldnames,
                    "mean_mae",
                ],
            )
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nSaved CSV summary to: {csv_path}")

    if len(all_rows) > 0:
        if not args.run_all:
            print("\n--save-boxplot is most useful with --run-all; skipping plot in manual mode.")
        else:
            _plot_boxplot_by_variance(
                boxplot_store=boxplot_store,
                save_path=boxplot_path,
                context_frac=args.context_frac,
                split_name=args.boxplot_split,
            )
            print(f"Saved boxplot PNG to: {boxplot_path}")

    if len(all_rows) > 0:
        _plot_mae_heatmaps_by_split(all_rows=all_rows, save_path=heatmap_path)
        print(f"Saved MAE heatmap PNG to: {heatmap_path}")

    if len(all_rows) > 0:
        _plot_delta_heatmaps_by_split(all_rows=all_rows, save_path=delta_heatmap_path)
        print(f"Saved delta heatmap PNG to: {delta_heatmap_path}")

    if len(cdf_records) > 0:
        _plot_cdf_mae_by_scenario(
            cdf_records=cdf_records,
            save_path=cdf_path,
            context_frac=args.context_frac,
            split_name=args.boxplot_split,
        )
        print(f"Saved CDF PNG to: {cdf_path}")

        _plot_barplot_ci_mean_mae(
            cdf_records=cdf_records,
            save_path=ci_barplot_path,
            context_frac=args.context_frac,
            split_name=args.boxplot_split,
        )
        print(f"Saved CI barplot PNG to: {ci_barplot_path}")

    if len(all_rows) > 0:
        _plot_scatter_optuna_vs_test(
            scenario_rows=all_rows,
            save_path=scatter_path,
            context_fracs=context_fracs,
        )
        print(f"Saved Optuna-vs-test scatter PNG to: {scatter_path}")

    if len(all_rows) > 0:
        _plot_context_topology_curves(
            all_rows=all_rows,
            context_fracs=context_fracs,
            save_path=context_curves_path,
        )
        print(f"Saved context-response curves PNG to: {context_curves_path}")


if __name__ == "__main__":
    main()
