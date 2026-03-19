#!/usr/bin/env python3
"""
Visual diagnostic for context-target selection effects across evaluation protocols.

The script loads one trajectory and generates slide-ready figures that highlight
how context and target sets change as context grows.

Useage:
    cd /home/fernando/tesis/underwater-localization-topologies/src/evaluation/
    python plot_context_target_relationship.py \
      --data-dir /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_high_variance \
      --topology aligned \
      --split val \
      --trajectory-index 0 \
      --context-fracs 0.2,0.4,0.6,0.8 \
      --focus-context-frac 0.4 \
      --holdout-frac 0.2 \
      --rollout-steps 5 \
      --save-dir /home/fernando/tesis/underwater-localization-topologies/results/evaluation_context_target_plots
"""

import argparse
import os
import pickle
import sys
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np


def _ensure_numpy_pickle_compat() -> None:
    """Alias numpy._core modules for environments where only numpy.core exists."""
    try:
        import numpy.core as np_core
        import numpy.core.multiarray as np_core_multiarray

        sys.modules.setdefault("numpy._core", np_core)
        sys.modules.setdefault("numpy._core.multiarray", np_core_multiarray)
    except Exception:
        pass


def _resolve_topology_dir(data_dir: str, topology: Optional[str]) -> str:
    if os.path.exists(os.path.join(data_dir, "train_data.pkl")):
        return data_dir
    if topology is None:
        raise ValueError(
            "data-dir does not point to a topology folder. "
            "Provide --topology (e.g. aligned, ellipsoidal, random)."
        )
    candidate = os.path.join(data_dir, f"topology_{topology}")
    if not os.path.exists(os.path.join(candidate, "train_data.pkl")):
        raise FileNotFoundError(
            f"Could not find topology data at: {candidate}"
        )
    return candidate


def _load_split(topology_dir: str, split: str):
    split = split.lower()
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of: train, val, test")
    path = os.path.join(topology_dir, f"{split}_data.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing split file: {path}")
    _ensure_numpy_pickle_compat()
    with open(path, "rb") as f:
        return pickle.load(f)


def _build_indices(
    total_points: int,
    context_frac: float,
    holdout_frac: float,
    protocol: str,
) -> tuple[np.ndarray, np.ndarray]:
    if protocol == "legacy":
        ctx_size = max(1, min(total_points - 1, int(round(context_frac * total_points))))
        ctx_idx = np.arange(0, ctx_size)
        tar_idx = np.arange(ctx_size, total_points)
        return ctx_idx, tar_idx

    n_holdout = max(1, int(round(holdout_frac * total_points)))
    holdout_start = total_points - n_holdout

    if protocol == "fixed holdout":
        max_ctx = max(1, holdout_start - 1)
        ctx_size = max(1, min(max_ctx, int(round(context_frac * total_points))))
        ctx_idx = np.arange(0, ctx_size)
        tar_idx = np.arange(holdout_start, total_points)
        return ctx_idx, tar_idx

    if protocol == "inverse holdout":
        max_ctx = max(1, holdout_start)
        ctx_size = max(1, min(max_ctx, int(round(context_frac * total_points))))
        ctx_start = holdout_start - ctx_size
        ctx_idx = np.arange(ctx_start, holdout_start)
        tar_idx = np.arange(holdout_start, total_points)
        return ctx_idx, tar_idx

    raise ValueError(f"Unknown protocol: {protocol}")


def _timeline_plot(
    ax,
    total_points: int,
    ctx_idx: np.ndarray,
    tar_idx: np.ndarray,
    protocol: str,
    context_frac: float,
):
    t = np.arange(total_points)
    ax.scatter(t, np.zeros_like(t), s=10, c="#c8c8c8", alpha=0.8, label="all points")

    if len(ctx_idx) > 0:
        ax.axvspan(ctx_idx[0], ctx_idx[-1], color="#4c78a8", alpha=0.15)
        ax.scatter(ctx_idx, np.zeros_like(ctx_idx) + 0.06, s=18, c="#4c78a8", label="context")
    if len(tar_idx) > 0:
        ax.axvspan(tar_idx[0], tar_idx[-1], color="#e45756", alpha=0.15)
        ax.scatter(tar_idx, np.zeros_like(tar_idx) - 0.06, s=18, c="#e45756", label="target")

    ax.set_ylim(-0.2, 0.2)
    ax.set_yticks([])
    ax.set_xlabel("time index")
    ax.set_title(f"{protocol} | context={context_frac:.0%}")
    ax.grid(alpha=0.2)


def _xy_plot(ax, y: np.ndarray, ctx_idx: np.ndarray, tar_idx: np.ndarray):
    xy = y[:, :2]
    ax.plot(xy[:, 0], xy[:, 1], color="#8a8a8a", linewidth=1.0, alpha=0.8)
    if len(ctx_idx) > 0:
        ax.scatter(xy[ctx_idx, 0], xy[ctx_idx, 1], s=20, c="#4c78a8", label="context")
    if len(tar_idx) > 0:
        ax.scatter(xy[tar_idx, 0], xy[tar_idx, 1], s=20, c="#e45756", label="target")
    ax.scatter(xy[0, 0], xy[0, 1], s=40, c="#2ca02c", marker="^", label="start")
    ax.scatter(xy[-1, 0], xy[-1, 1], s=40, c="#000000", marker="x", label="end")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.2)


def _plot_protocol_progression(
    y: np.ndarray,
    context_fracs: Iterable[float],
    holdout_frac: float,
    protocol: str,
    save_path: str,
):
    context_fracs = list(context_fracs)
    nrows = len(context_fracs)
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(13, 4 * nrows))
    if nrows == 1:
        axes = np.array([axes])

    total_points = y.shape[0]
    for i, frac in enumerate(context_fracs):
        ctx_idx, tar_idx = _build_indices(total_points, frac, holdout_frac, protocol)
        _xy_plot(axes[i, 0], y, ctx_idx, tar_idx)
        axes[i, 0].set_title(f"trajectory view | {protocol} | context={frac:.0%}")
        _timeline_plot(axes[i, 1], total_points, ctx_idx, tar_idx, protocol, frac)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=5, frameon=False)
    fig.suptitle(
        f"Context-target selection progression ({protocol})\n"
        "Blue: context, Red: target",
        y=0.99,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.99])
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_protocol_comparison(
    y: np.ndarray,
    focus_context_frac: float,
    holdout_frac: float,
    save_path: str,
):
    protocols = ["legacy", "fixed holdout", "inverse holdout"]
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(13, 12))

    total_points = y.shape[0]
    for i, protocol in enumerate(protocols):
        ctx_idx, tar_idx = _build_indices(total_points, focus_context_frac, holdout_frac, protocol)
        _xy_plot(axes[i, 0], y, ctx_idx, tar_idx)
        axes[i, 0].set_title(f"trajectory view | {protocol}")
        _timeline_plot(axes[i, 1], total_points, ctx_idx, tar_idx, protocol, focus_context_frac)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=5, frameon=False)
    fig.suptitle(
        "Same context fraction, different protocols\n"
        "This isolates the methodological effect from model behavior",
        y=0.985,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.99])
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_rollout_comparison(
    y: np.ndarray,
    focus_context_frac: float,
    holdout_frac: float,
    rollout_steps: int,
    save_path: str,
):
    protocols = ["legacy", "fixed holdout", "inverse holdout"]
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(13, 12))

    total_points = y.shape[0]
    xy = y[:, :2]

    for i, protocol in enumerate(protocols):
        ctx_idx, tar_idx = _build_indices(total_points, focus_context_frac, holdout_frac, protocol)
        rollout_idx = tar_idx[:rollout_steps]

        # XY view
        ax_xy = axes[i, 0]
        ax_xy.plot(xy[:, 0], xy[:, 1], color="#8a8a8a", linewidth=1.0, alpha=0.8)
        if len(ctx_idx) > 0:
            ax_xy.scatter(xy[ctx_idx, 0], xy[ctx_idx, 1], s=16, c="#4c78a8", label="context")
        if len(tar_idx) > 0:
            ax_xy.scatter(xy[tar_idx, 0], xy[tar_idx, 1], s=10, c="#e45756", alpha=0.18, label="all target")

        if len(rollout_idx) > 0:
            cmap = plt.get_cmap("plasma")(np.linspace(0.15, 0.9, len(rollout_idx)))
            for j, idx in enumerate(rollout_idx):
                ax_xy.scatter(xy[idx, 0], xy[idx, 1], s=44, c=[cmap[j]], edgecolors="black", linewidths=0.6)
                ax_xy.text(xy[idx, 0], xy[idx, 1], str(j + 1), fontsize=8, va="bottom", ha="left")

        ax_xy.scatter(xy[0, 0], xy[0, 1], s=40, c="#2ca02c", marker="^")
        ax_xy.scatter(xy[-1, 0], xy[-1, 1], s=40, c="#000000", marker="x")
        ax_xy.set_xlabel("x")
        ax_xy.set_ylabel("y")
        ax_xy.set_title(f"trajectory view | {protocol} | rollout={rollout_steps}")
        ax_xy.grid(alpha=0.2)

        # Timeline view
        ax_t = axes[i, 1]
        t = np.arange(total_points)
        ax_t.scatter(t, np.zeros_like(t), s=10, c="#c8c8c8", alpha=0.8)
        if len(ctx_idx) > 0:
            ax_t.axvspan(ctx_idx[0], ctx_idx[-1], color="#4c78a8", alpha=0.15)
            ax_t.scatter(ctx_idx, np.zeros_like(ctx_idx) + 0.05, s=16, c="#4c78a8")
        if len(tar_idx) > 0:
            ax_t.axvspan(tar_idx[0], tar_idx[-1], color="#e45756", alpha=0.1)

        if len(rollout_idx) > 0:
            cmap = plt.get_cmap("plasma")(np.linspace(0.15, 0.9, len(rollout_idx)))
            for j, idx in enumerate(rollout_idx):
                ax_t.scatter([idx], [-0.05], s=48, c=[cmap[j]], edgecolors="black", linewidths=0.6)
                ax_t.text(idx, -0.085, str(j + 1), fontsize=8, va="top", ha="center")

        ax_t.set_ylim(-0.2, 0.2)
        ax_t.set_yticks([])
        ax_t.set_xlabel("time index")
        ax_t.set_title(f"timeline | {protocol} | context={focus_context_frac:.0%}")
        ax_t.grid(alpha=0.2)

    fig.suptitle(
        f"Rollout visualization ({rollout_steps} steps) across protocols\n"
        "Number labels indicate step order within the selected target rollout.",
        y=0.985,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.99])
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _parse_fracs(frac_str: str) -> list[float]:
    vals = [float(x.strip()) for x in frac_str.split(",") if x.strip()]
    for v in vals:
        if not (0.0 < v < 1.0):
            raise ValueError(f"Invalid context fraction {v}. Must be in (0,1).")
    return vals


def main():
    parser = argparse.ArgumentParser(description="Plot context-target regions over a single trajectory for presentation/debugging.")
    parser.add_argument("--data-dir", type=str, required=True, help="Either topology_* folder or a parent containing topology_* folders.")
    parser.add_argument("--topology", type=str, default=None, help="Used only if --data-dir is not already a topology_* folder.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--trajectory-index", type=int, default=0)
    parser.add_argument("--context-fracs", type=str, default="0.1,0.2,0.4,0.6,0.8,0.9")
    parser.add_argument("--focus-context-frac", type=float, default=0.4)
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--rollout-steps", type=int, default=5)
    parser.add_argument("--save-dir", type=str, default="results/evaluation_context_target_plots")
    args = parser.parse_args()

    topology_dir = _resolve_topology_dir(args.data_dir, args.topology)
    data = _load_split(topology_dir, args.split)
    if len(data) == 0:
        raise ValueError("Selected split is empty.")

    idx = args.trajectory_index
    if idx < 0 or idx >= len(data):
        raise IndexError(f"trajectory-index {idx} out of range [0, {len(data)-1}]")

    _, y = data[idx]
    y = np.asarray(y)
    if y.ndim != 2 or y.shape[1] < 2:
        raise ValueError("Expected y with shape (T, >=2) to plot XY trajectory.")

    context_fracs = _parse_fracs(args.context_fracs)
    os.makedirs(args.save_dir, exist_ok=True)

    stem = f"{os.path.basename(topology_dir)}_{args.split}_traj{idx:03d}"
    for protocol in ["legacy", "fixed holdout", "inverse holdout"]:
        out_path = os.path.join(args.save_dir, f"{stem}_{protocol}_progression.png")
        _plot_protocol_progression(
            y=y,
            context_fracs=context_fracs,
            holdout_frac=args.holdout_frac,
            protocol=protocol,
            save_path=out_path,
        )

    out_path = os.path.join(args.save_dir, f"{stem}_protocol_comparison_ctx{int(args.focus_context_frac*100):02d}.png")
    _plot_protocol_comparison(
        y=y,
        focus_context_frac=args.focus_context_frac,
        holdout_frac=args.holdout_frac,
        save_path=out_path,
    )

    out_path = os.path.join(
        args.save_dir,
        f"{stem}_rollout{args.rollout_steps}_comparison_ctx{int(args.focus_context_frac*100):02d}.png"
    )
    _plot_rollout_comparison(
        y=y,
        focus_context_frac=args.focus_context_frac,
        holdout_frac=args.holdout_frac,
        rollout_steps=args.rollout_steps,
        save_path=out_path,
    )

    print("Saved plots to:", os.path.abspath(args.save_dir))


if __name__ == "__main__":
    main()
