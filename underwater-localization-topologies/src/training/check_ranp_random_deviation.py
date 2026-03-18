#!/usr/bin/env python3
"""
check_ranp_random_deviation.py

Diagnostic script for the RANP random-topology validation/test deviation.

It performs two blocks of checks:
1) Data checks (val vs test) for low/high variance random topology datasets.
2) Model checks across Optuna trial folders, evaluating each trial checkpoint on
   val/test to quantify generalization gap by context size.

Outputs are written under:
  src/training/results/optuna/models_evaluation/random_ranp_diagnosis/

Example:
  python -m src.training.check_ranp_random_deviation \
    --device cuda \
    --context-fracs 0.1,0.2,0.3,0.4,0.5,0.6
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.utils.load_optuna_model import load_model_from_checkpoint


# Compatibility shim for datasets pickled in environments where internal
# numpy module paths differ (for example numpy 2.x vs 1.x internals).
try:
    import numpy._core  # type: ignore
except Exception:
    import numpy.core as _numpy_core  # type: ignore

    sys.modules["numpy._core"] = _numpy_core


DEFAULT_LOW_DATA_DIR = "data/data/data_processed_topologies_low_variance/topology_random"
DEFAULT_HIGH_DATA_DIR = "data/data/data_processed_topologies_high_variance/topology_random"
DEFAULT_LOW_TRIALS_DIR = "src/training/results/optuna/ranp_masked_lowvar_random_v1"
DEFAULT_HIGH_TRIALS_DIR = "src/training/results/optuna/ranp_masked_highvar_random_v1"
DEFAULT_OUTPUT_DIR = "src/training/results/optuna/models_evaluation/random_ranp_diagnosis"


@dataclass
class SplitData:
    train: list
    val: list
    test: list


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_split_data(topology_dir: Path) -> SplitData:
    return SplitData(
        train=_load_pickle(topology_dir / "train_data.pkl"),
        val=_load_pickle(topology_dir / "val_data.pkl"),
        test=_load_pickle(topology_dir / "test_data.pkl"),
    )


def compute_y_stats(train_data):
    y = np.concatenate([yy for _, yy in train_data], axis=0)
    y_mean = torch.tensor(y.mean(axis=0), dtype=torch.float32)
    y_std = torch.tensor(y.std(axis=0) + 1e-6, dtype=torch.float32)
    return y_mean, y_std


def compute_x_sensor_means(train_data, num_time_points: int, num_sensors: int):
    x = np.concatenate([xx for xx, _ in train_data], axis=0)
    x3 = x.reshape(x.shape[0], num_time_points, num_sensors)
    return x3.mean(axis=0).T


def apply_mask_and_append(x_batch, sensor_mask, x_means_sp, num_time_points, num_sensors):
    bsz, tlen, dx = x_batch.shape
    p, s = num_time_points, num_sensors
    assert dx == p * s

    x4 = x_batch.view(bsz, tlen, p, s)
    mu = x_means_sp.T.view(1, 1, p, s).to(x_batch.device, dtype=x_batch.dtype)
    m = sensor_mask.view(bsz, 1, 1, s)
    x4_masked = x4 * m + mu * (1.0 - m)
    x_masked = x4_masked.reshape(bsz, tlen, dx)
    mask_feat = sensor_mask.view(bsz, 1, s).expand(bsz, tlen, s)
    return torch.cat([x_masked, mask_feat], dim=-1)


def make_loader(data, batch_size: int):
    xs = torch.tensor(np.stack([x for x, _ in data]), dtype=torch.float32)
    ys = torch.tensor(np.stack([y for _, y in data]), dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(xs, ys)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)


def eval_ranp_mae(model, loader, y_mean, y_std, x_means_sp, context_fracs, num_time_points, num_sensors, device):
    model.eval()
    y_mean = y_mean.to(device)
    y_std = y_std.to(device)
    x_means_sp = x_means_sp.to(device)

    sums = {f: 0.0 for f in context_fracs}
    n_batches = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            bsz, tlen, _ = x_batch.shape

            sensor_mask = torch.ones(bsz, num_sensors, device=device)
            x_aug = apply_mask_and_append(x_batch, sensor_mask, x_means_sp, num_time_points, num_sensors)
            y_norm = (y_batch - y_mean) / y_std

            for frac in context_fracs:
                ctx_size = max(1, min(tlen - 1, int(round(frac * tlen))))
                ctx_idx = torch.arange(ctx_size, device=device)
                tar_idx = torch.arange(ctx_size, tlen, device=device)

                context_y = y_norm[:, ctx_idx, :]
                y_pred_norm, *_ = model(
                    x_seq=x_aug,
                    context_indices=ctx_idx,
                    context_y=context_y,
                    target_indices=tar_idx,
                )
                y_pred = y_pred_norm * y_std + y_mean
                mae = torch.mean(torch.abs(y_pred - y_batch[:, tar_idx, :])).item()
                sums[frac] += mae

            n_batches += 1

    return {f: sums[f] / max(1, n_batches) for f in context_fracs}


def persistence_baseline_mae(data, context_frac: float) -> float:
    maes = []
    for _, y in data:
        tlen = y.shape[0]
        ctx_size = max(1, min(tlen - 1, int(round(context_frac * tlen))))
        target = y[ctx_size:, :]
        pred = np.repeat(y[ctx_size - 1:ctx_size, :], repeats=target.shape[0], axis=0)
        maes.append(float(np.mean(np.abs(pred - target))))
    return float(np.mean(maes)) if maes else np.nan


def traj_hash(x: np.ndarray, y: np.ndarray) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(x.tobytes())
    h.update(y.tobytes())
    return h.hexdigest()


def split_overlap(val_data, test_data):
    val_h = {traj_hash(x, y) for x, y in val_data}
    test_h = {traj_hash(x, y) for x, y in test_data}
    inter = len(val_h.intersection(test_h))
    return inter, len(val_h), len(test_h), inter / max(1, len(val_h))


def summarize_array(name: str, arr: np.ndarray):
    return {
        "metric": name,
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def extract_split_features(data):
    x = np.concatenate([xx for xx, _ in data], axis=0)
    y = np.concatenate([yy for _, yy in data], axis=0)

    path_lens = []
    speed_means = []
    speed_p95 = []
    for _, yy in data:
        d = np.diff(yy, axis=0)
        sp = np.linalg.norm(d, axis=1)
        path_lens.append(float(np.sum(sp)))
        speed_means.append(float(np.mean(sp)) if sp.size else 0.0)
        speed_p95.append(float(np.percentile(sp, 95)) if sp.size else 0.0)

    return {
        "x": x,
        "y": y,
        "path_len": np.asarray(path_lens),
        "speed_mean": np.asarray(speed_means),
        "speed_p95": np.asarray(speed_p95),
    }


def run_data_checks(variance_tag: str, topology_dir: Path, context_fracs: list[float]):
    splits = load_split_data(topology_dir)
    val_f = extract_split_features(splits.val)
    test_f = extract_split_features(splits.test)

    records = []
    for key in ["x", "y", "path_len", "speed_mean", "speed_p95"]:
        s_val = summarize_array(f"{key}_val", val_f[key])
        s_test = summarize_array(f"{key}_test", test_f[key])
        pooled_std = max(1e-8, 0.5 * (s_val["std"] + s_test["std"]))
        records.append(
            {
                "variance": variance_tag,
                "feature": key,
                "val_mean": s_val["mean"],
                "test_mean": s_test["mean"],
                "val_std": s_val["std"],
                "test_std": s_test["std"],
                "mean_delta": s_test["mean"] - s_val["mean"],
                "cohen_like_d": (s_test["mean"] - s_val["mean"]) / pooled_std,
                "val_p95": s_val["p95"],
                "test_p95": s_test["p95"],
            }
        )

    inter, n_val, n_test, overlap_ratio = split_overlap(splits.val, splits.test)
    integrity = {
        "variance": variance_tag,
        "n_train": len(splits.train),
        "n_val": len(splits.val),
        "n_test": len(splits.test),
        "val_test_exact_overlap": inter,
        "val_test_overlap_ratio": overlap_ratio,
        "val_x_has_nan": bool(np.isnan(val_f["x"]).any()),
        "test_x_has_nan": bool(np.isnan(test_f["x"]).any()),
        "val_y_has_nan": bool(np.isnan(val_f["y"]).any()),
        "test_y_has_nan": bool(np.isnan(test_f["y"]).any()),
        "val_x_has_inf": bool(np.isinf(val_f["x"]).any()),
        "test_x_has_inf": bool(np.isinf(test_f["x"]).any()),
        "val_y_has_inf": bool(np.isinf(val_f["y"]).any()),
        "test_y_has_inf": bool(np.isinf(test_f["y"]).any()),
    }

    baseline = []
    for frac in context_fracs:
        baseline.append(
            {
                "variance": variance_tag,
                "context_frac": frac,
                "baseline_val_mae": persistence_baseline_mae(splits.val, frac),
                "baseline_test_mae": persistence_baseline_mae(splits.test, frac),
            }
        )

    return records, integrity, baseline, splits


def run_trial_checks(
    variance_tag: str,
    trials_root: Path,
    splits: SplitData,
    context_fracs: list[float],
    num_time_points: int,
    num_sensors: int,
    batch_size: int,
    device: str,
    max_trials: int | None,
):
    y_mean, y_std = compute_y_stats(splits.train)
    x_means = torch.tensor(
        compute_x_sensor_means(splits.train, num_time_points=num_time_points, num_sensors=num_sensors),
        dtype=torch.float32,
    )
    val_loader = make_loader(splits.val, batch_size=batch_size)
    test_loader = make_loader(splits.test, batch_size=batch_size)

    trial_dirs = sorted([d for d in trials_root.iterdir() if d.is_dir() and d.name.startswith("trial_")])
    if max_trials is not None:
        trial_dirs = trial_dirs[:max_trials]

    out = []
    for td in trial_dirs:
        hparams_path = td / "hparams.json"
        ckpt_path = td / "topology_random" / "best_checkpoint.pth.tar"
        log_path = td / "topology_random" / "training_log.csv"
        if not hparams_path.exists() or not ckpt_path.exists():
            continue

        with open(hparams_path, "r") as f:
            hparams = json.load(f)

        model, _, _ = load_model_from_checkpoint(
            checkpoint_path=ckpt_path,
            hparams=hparams,
            model_type="ranp",
            num_sensors=num_sensors,
            num_time_points=num_time_points,
            output_dim=3,
            device=device,
        )

        val_mae = eval_ranp_mae(
            model,
            val_loader,
            y_mean,
            y_std,
            x_means,
            context_fracs,
            num_time_points,
            num_sensors,
            device,
        )
        test_mae = eval_ranp_mae(
            model,
            test_loader,
            y_mean,
            y_std,
            x_means,
            context_fracs,
            num_time_points,
            num_sensors,
            device,
        )

        row = {
            "variance": variance_tag,
            "trial": td.name,
            "num_hidden": hparams.get("num_hidden"),
            "lr": hparams.get("lr"),
            "weight_decay": hparams.get("weight_decay"),
            "sensor_drop_p": hparams.get("sensor_drop_p"),
            "rnn_type": hparams.get("rnn_type"),
            "rnn_layers": hparams.get("rnn_layers"),
            "rnn_dropout": hparams.get("rnn_dropout"),
            "has_training_log": log_path.exists(),
        }
        for frac in context_fracs:
            p = int(round(frac * 100))
            row[f"val_ctx_{p}"] = val_mae[frac]
            row[f"test_ctx_{p}"] = test_mae[frac]
            row[f"gap_ctx_{p}"] = test_mae[frac] - val_mae[frac]
        row["val_mean"] = float(np.mean(list(val_mae.values())))
        row["test_mean"] = float(np.mean(list(test_mae.values())))
        row["gap_mean"] = row["test_mean"] - row["val_mean"]
        out.append(row)

    return out


def plot_baseline_gap(baseline_rows: pd.DataFrame, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, constrained_layout=True)
    for ax, var in zip(axes, ["lowvar", "highvar"]):
        sub = baseline_rows[baseline_rows["variance"] == var].sort_values("context_frac")
        x = 100.0 * sub["context_frac"].to_numpy()
        ax.plot(x, sub["baseline_val_mae"], marker="o", label="baseline val")
        ax.plot(x, sub["baseline_test_mae"], marker="o", label="baseline test")
        ax.set_title(var)
        ax.set_xlabel("Context (%)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Persistence baseline MAE (m)")
    axes[1].legend()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_trial_val_test_scatter(trial_df: pd.DataFrame, context_focus_pct: int, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True, constrained_layout=True)
    for ax, var in zip(axes, ["lowvar", "highvar"]):
        sub = trial_df[trial_df["variance"] == var]
        xv = sub[f"val_ctx_{context_focus_pct}"]
        yv = sub[f"test_ctx_{context_focus_pct}"]
        ax.scatter(xv, yv, alpha=0.9, edgecolors="black", linewidths=0.4)
        if len(xv) > 0:
            lo = min(float(xv.min()), float(yv.min()))
            hi = max(float(xv.max()), float(yv.max()))
            ax.plot([lo, hi], [lo, hi], "--", color="gray")
        ax.set_title(var)
        ax.set_xlabel(f"val MAE ctx{context_focus_pct}")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"test MAE ctx{context_focus_pct}")
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def build_survivor_leaderboards(df_trials: pd.DataFrame, context_focus_pct: int):
    rows = []
    if df_trials.empty:
        return pd.DataFrame(rows)

    focus_gap_col = f"gap_ctx_{context_focus_pct}"
    focus_val_col = f"val_ctx_{context_focus_pct}"
    focus_test_col = f"test_ctx_{context_focus_pct}"

    for var in sorted(df_trials["variance"].unique()):
        sub = df_trials[df_trials["variance"] == var].copy()
        if sub.empty:
            continue

        by_gap = sub.sort_values("gap_mean", ascending=True).reset_index(drop=True)
        for rank, (_, r) in enumerate(by_gap.iterrows(), start=1):
            rows.append(
                {
                    "variance": var,
                    "ranking": "best_gap_mean",
                    "rank": rank,
                    "trial": r["trial"],
                    "val_mean": r.get("val_mean", np.nan),
                    "test_mean": r.get("test_mean", np.nan),
                    "gap_mean": r.get("gap_mean", np.nan),
                    "val_focus": r.get(focus_val_col, np.nan),
                    "test_focus": r.get(focus_test_col, np.nan),
                    "gap_focus": r.get(focus_gap_col, np.nan),
                }
            )

        by_val = sub.sort_values("val_mean", ascending=True).reset_index(drop=True)
        for rank, (_, r) in enumerate(by_val.iterrows(), start=1):
            rows.append(
                {
                    "variance": var,
                    "ranking": "best_val_mean",
                    "rank": rank,
                    "trial": r["trial"],
                    "val_mean": r.get("val_mean", np.nan),
                    "test_mean": r.get("test_mean", np.nan),
                    "gap_mean": r.get("gap_mean", np.nan),
                    "val_focus": r.get(focus_val_col, np.nan),
                    "test_focus": r.get(focus_test_col, np.nan),
                    "gap_focus": r.get(focus_gap_col, np.nan),
                }
            )

    return pd.DataFrame(rows)


def build_survivor_recommendations(df_trials: pd.DataFrame, context_focus_pct: int):
    rows = []
    if df_trials.empty:
        return pd.DataFrame(rows)

    for var in sorted(df_trials["variance"].unique()):
        sub = df_trials[df_trials["variance"] == var].copy()
        if sub.empty:
            continue

        best_gap_row = sub.sort_values("gap_mean", ascending=True).iloc[0]
        best_val_row = sub.sort_values("val_mean", ascending=True).iloc[0]

        rows.append(
            {
                "variance": var,
                "recommended_trial": best_gap_row["trial"],
                "rule": "min_gap_mean",
                "recommended_val_mean": best_gap_row["val_mean"],
                "recommended_test_mean": best_gap_row["test_mean"],
                "recommended_gap_mean": best_gap_row["gap_mean"],
                "best_val_trial": best_val_row["trial"],
                "best_val_val_mean": best_val_row["val_mean"],
                "best_val_test_mean": best_val_row["test_mean"],
                "best_val_gap_mean": best_val_row["gap_mean"],
                "gap_reduction_vs_best_val": best_val_row["gap_mean"] - best_gap_row["gap_mean"],
                "val_cost_vs_best_val": best_gap_row["val_mean"] - best_val_row["val_mean"],
            }
        )

    return pd.DataFrame(rows)


def build_hp_contrast(df_trials: pd.DataFrame):
    rows = []
    hp_cols = [
        "num_hidden",
        "lr",
        "weight_decay",
        "sensor_drop_p",
        "rnn_type",
        "rnn_layers",
        "rnn_dropout",
    ]
    if df_trials.empty:
        return pd.DataFrame(rows)

    for var in sorted(df_trials["variance"].unique()):
        sub = df_trials[df_trials["variance"] == var].copy()
        if sub.empty:
            continue

        best_gap_row = sub.sort_values("gap_mean", ascending=True).iloc[0]
        best_val_row = sub.sort_values("val_mean", ascending=True).iloc[0]
        for hp in hp_cols:
            rows.append(
                {
                    "variance": var,
                    "hyperparam": hp,
                    "best_gap_trial": best_gap_row["trial"],
                    "best_gap_value": best_gap_row.get(hp, np.nan),
                    "best_val_trial": best_val_row["trial"],
                    "best_val_value": best_val_row.get(hp, np.nan),
                    "same_value": best_gap_row.get(hp, None) == best_val_row.get(hp, None),
                }
            )

    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description="Check RANP random topology val/test deviation")
    p.add_argument("--low-data-dir", default=DEFAULT_LOW_DATA_DIR)
    p.add_argument("--high-data-dir", default=DEFAULT_HIGH_DATA_DIR)
    p.add_argument("--low-trials-dir", default=DEFAULT_LOW_TRIALS_DIR)
    p.add_argument("--high-trials-dir", default=DEFAULT_HIGH_TRIALS_DIR)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--context-fracs", default="0.1,0.2,0.3,0.4,0.5,0.6")
    p.add_argument("--context-focus", type=float, default=0.3, help="Context used for key gap plots.")
    p.add_argument("--num-time-points", type=int, default=201)
    p.add_argument("--num-sensors", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-trials", type=int, default=None, help="Optional cap for trial folders per variance.")
    args = p.parse_args()

    context_fracs = [float(x.strip()) for x in args.context_fracs.split(",") if x.strip()]
    for frac in context_fracs:
        if not (0.0 < frac < 1.0):
            raise ValueError(f"Invalid context fraction: {frac}")
    context_focus_pct = int(round(args.context_focus * 100))

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[1/3] Running data-level checks...")
    data_records = []
    integrity_rows = []
    baseline_rows = []

    low_data_records, low_integrity, low_baseline, low_splits = run_data_checks(
        "lowvar", Path(args.low_data_dir), context_fracs
    )
    high_data_records, high_integrity, high_baseline, high_splits = run_data_checks(
        "highvar", Path(args.high_data_dir), context_fracs
    )

    data_records.extend(low_data_records)
    data_records.extend(high_data_records)
    integrity_rows.extend([low_integrity, high_integrity])
    baseline_rows.extend(low_baseline)
    baseline_rows.extend(high_baseline)

    df_data = pd.DataFrame(data_records)
    df_integrity = pd.DataFrame(integrity_rows)
    df_baseline = pd.DataFrame(baseline_rows)

    df_data.to_csv(outdir / "data_split_diagnostics.csv", index=False)
    df_integrity.to_csv(outdir / "data_split_integrity.csv", index=False)
    df_baseline.to_csv(outdir / "baseline_persistence_val_test.csv", index=False)

    print("[2/3] Running trial-level model checks (this may take a while)...")
    low_trials = run_trial_checks(
        variance_tag="lowvar",
        trials_root=Path(args.low_trials_dir),
        splits=low_splits,
        context_fracs=context_fracs,
        num_time_points=args.num_time_points,
        num_sensors=args.num_sensors,
        batch_size=args.batch_size,
        device=args.device,
        max_trials=args.max_trials,
    )
    high_trials = run_trial_checks(
        variance_tag="highvar",
        trials_root=Path(args.high_trials_dir),
        splits=high_splits,
        context_fracs=context_fracs,
        num_time_points=args.num_time_points,
        num_sensors=args.num_sensors,
        batch_size=args.batch_size,
        device=args.device,
        max_trials=args.max_trials,
    )

    df_trials = pd.DataFrame(low_trials + high_trials)
    df_trials.to_csv(outdir / "trial_generalization_diagnostics.csv", index=False)

    survivor_leaderboard = build_survivor_leaderboards(df_trials, context_focus_pct=context_focus_pct)
    survivor_recommendations = build_survivor_recommendations(df_trials, context_focus_pct=context_focus_pct)
    hp_contrast = build_hp_contrast(df_trials)

    survivor_leaderboard.to_csv(outdir / "survivor_trial_leaderboard.csv", index=False)
    survivor_recommendations.to_csv(outdir / "survivor_recommendations.csv", index=False)
    hp_contrast.to_csv(outdir / "survivor_hparam_contrast.csv", index=False)

    print("[3/3] Writing plots and markdown summary...")
    plot_baseline_gap(df_baseline, outdir / "baseline_val_vs_test_by_context.png")
    if not df_trials.empty and f"val_ctx_{context_focus_pct}" in df_trials.columns:
        plot_trial_val_test_scatter(
            trial_df=df_trials,
            context_focus_pct=context_focus_pct,
            save_path=outdir / f"trial_val_vs_test_scatter_ctx{context_focus_pct}.png",
        )

    with open(outdir / "diagnostic_summary.md", "w", newline="") as f:
        f.write("# RANP Random Topology Deviation Check\n\n")
        f.write("## Data Integrity\n")
        f.write(df_integrity.to_markdown(index=False))
        f.write("\n\n## Strongest Val-Test Feature Shifts (by |cohen_like_d|)\n")
        if not df_data.empty:
            top_shift = df_data.reindex(df_data["cohen_like_d"].abs().sort_values(ascending=False).index).head(12)
            f.write(top_shift.to_markdown(index=False))
        else:
            f.write("No data.\n")

        f.write("\n\n## Trial Generalization Gap Overview\n")
        if not df_trials.empty:
            by_var = df_trials.groupby("variance")["gap_mean"].agg(["count", "mean", "std", "min", "max"]).reset_index()
            f.write(by_var.to_markdown(index=False))
        else:
            f.write("No trial diagnostics computed.\n")

        f.write("\n\n## Survivor Recommendations\n")
        if not survivor_recommendations.empty:
            f.write(survivor_recommendations.to_markdown(index=False))
        else:
            f.write("No survivor recommendations available.\n")

        f.write("\n\n## Top Survivor Leaderboard Rows\n")
        if not survivor_leaderboard.empty:
            top_lb = survivor_leaderboard[survivor_leaderboard["rank"] <= 3]
            f.write(top_lb.to_markdown(index=False))
        else:
            f.write("No survivor leaderboard available.\n")

        f.write("\n\n## Hyperparameter Contrast (Best Gap vs Best Val)\n")
        if not hp_contrast.empty:
            f.write(hp_contrast.to_markdown(index=False))
        else:
            f.write("No hyperparameter contrast available.\n")

    print("Done. Outputs written to:")
    print(f"  {outdir}")


if __name__ == "__main__":
    main()
