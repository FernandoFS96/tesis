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
            ctx_size = max(1, min(T - 1, int(round(context_frac * T))))
            ctx_idx = torch.arange(ctx_size, device=device)
            tar_idx = torch.arange(ctx_size, T, device=device)

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
    trial_mae = meta.get("value", "?") if meta else "?"
    print(f"  Trial: {trial_num} | Optuna MAE: {trial_mae} | Params: {n_params:,}")
    print(f"  Hparams: {hparams}")

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
            device=args.device,
        )
        mean_mae = float(np.mean(list(mae_by_frac.values())))

        row = {
            "model": model_label.lower(),
            "topology": topology,
            "split": split_name,
            "mean_mae": mean_mae,
        }
        for frac in context_fracs:
            row[f"mae_ctx_{int(frac * 100)}"] = float(mae_by_frac[frac])
        out_rows.append(row)

        print(f"  {model_label} [{split_name}] mean MAE: {mean_mae:.4f} m")
        for frac in context_fracs:
            print(f"    ctx={int(frac * 100):3d}% -> {mae_by_frac[frac]:.4f} m")

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
                device=args.device,
            )

    return out_rows, boxplot_values


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
    parser.add_argument("--boxplot-split", default="test", choices=["val", "test"], help="Which split to use for boxplot distributions.")
    parser.add_argument("--context-frac", type=float, default=0.4, help="Context fraction used by all metrics and plots (default: 0.4).")
    parser.add_argument("--device", default="cpu", help="Torch device: cpu | cuda | cuda:0 ...")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-sensors", type=int, default=10)
    parser.add_argument("--num-time-points", type=int, default=201)
    args = parser.parse_args()

    context_fracs = [args.context_frac]
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

    all_rows = []
    boxplot_store = defaultdict(lambda: defaultdict(dict))

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
                        rows, box_values = _evaluate_one_configuration(
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
                rows, _ = _evaluate_one_configuration(
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
            except FileNotFoundError as exc:
                print(f"\nSkipping {label}: {exc}")

    if len(all_rows) > 0:
        print(f"\n{'='*90}")
        print("Consolidated summary")
        print(f"model | variance | topology | split | ctx{ctx_pct:02d} | mean")
        print("-" * 90)
        for row in all_rows:
            print(
                f"{row['model']:<5} | {row['variance']:<8} | {row['topology']:<11} | {row['split']:<4} | "
                f"{row[f'mae_ctx_{ctx_pct}']:.4f} | {row['mean_mae']:.4f}"
            )

    if len(all_rows) > 0:
        import csv

        out_dir = os.path.dirname(csv_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "model",
                    "variance",
                    "topology",
                    "split",
                    f"mae_ctx_{ctx_pct}",
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


if __name__ == "__main__":
    main()
