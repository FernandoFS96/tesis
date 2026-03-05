#!/usr/bin/env python3
"""
plot_pareto_ood.py
------------------
Reads MAE and latency CSV outputs from eval_ood_postprocess.py and
finetune_decoder_ood.py, reconstructs *total* inference latency for every
method, and produces a MAE vs. latency scatter with the Pareto frontier
(minimise both MAE and latency) highlighted.

Latency semantics (documented here for clarity):
  - raw forward pass     : measured directly (total cost)
  - filter-only methods  : measured as OVERHEAD → total = raw + overhead
  - AR-based methods     : measured as TOTAL (AR forward already included)
  - Fine-tuned decoder   : identical inference architecture → same cost as HV

Usage:
    python plot_pareto_ood.py
    python plot_pareto_ood.py --postprocess-dir PATH --finetune-dir PATH --output-dir PATH
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── default result directories ────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).parent
_DEFAULT_PP_DIR  = _SCRIPT_DIR / "results" / "eval_ood_postprocess"
_DEFAULT_FT_DIR  = _SCRIPT_DIR / "results" / "finetune_decoder_ood"
_DEFAULT_OUT_DIR = _SCRIPT_DIR / "results" / "pareto_ood"

# Methods in latency_per_method.csv whose latency column is *overhead only*
# (must add raw forward-pass latency to get true total)
_FILTER_OVERHEAD_METHODS = {
    "alpha_beta",
    "kalman_cv_I", "kalman_cv_var",
    "kalman_rts_I", "kalman_rts_var",
}

# AR-based methods: their latency column already includes the AR forward pass
_AR_TOTAL_METHODS = {"ar_raw", "ar_kalman_rts_I", "ar_kalman_rts_var"}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _method_label(lat_df: pd.DataFrame, method: str, fallback: str) -> str:
    row = lat_df[lat_df["method"] == method]
    return row["label"].iloc[0] if len(row) else fallback


def load_postprocess(pp_dir: Path) -> pd.DataFrame:
    """Load data from eval_ood_postprocess output directory.

    latency_per_method.csv – latency values are in **seconds** (mean_s).
    mae_summary.csv         – columns: model, theta, <method1>, <method2>, …

    Returns a DataFrame with columns:
        method, label, group, mae_mean_m, latency_total_ms
    """
    mae_df = pd.read_csv(pp_dir / "mae_summary.csv")
    lat_df = pd.read_csv(pp_dir / "latency_per_method.csv")

    # raw forward-pass latency (ms)
    raw_ms: float = lat_df.loc[lat_df["method"] == "raw", "mean_s"].iloc[0] * 1000.0

    # Build method → total latency (ms)
    lat_lookup: Dict[str, float] = {}
    for _, row in lat_df.iterrows():
        m = row["method"]
        overhead_ms = row["mean_s"] * 1000.0
        if m == "raw":
            lat_lookup[m] = raw_ms
        elif m in _FILTER_OVERHEAD_METHODS:
            lat_lookup[m] = raw_ms + overhead_ms   # add raw forward pass
        elif m in _AR_TOTAL_METHODS:
            lat_lookup[m] = overhead_ms             # already total
        else:
            lat_lookup[m] = overhead_ms             # fallback

    # MAE CSV: columns = model, theta, <method_cols…>
    method_cols: List[str] = [c for c in mae_df.columns if c not in ("model", "theta")]

    records = []
    for model_name in mae_df["model"].unique():
        sub = mae_df[mae_df["model"] == model_name]
        group = "Oracle" if model_name == "oracle" else "HV baseline"
        prefix = "Oracle" if group == "Oracle" else "HV"
        for mc in method_cols:
            mae_mean = float(sub[mc].mean())
            lat_total = lat_lookup.get(mc, float("nan"))
            label_str = _method_label(lat_df, mc, mc)
            records.append({
                "method": mc,
                "label": f"{prefix} – {label_str}",
                "group": group,
                "mae_mean_m": mae_mean,
                "latency_total_ms": lat_total,
            })

    return pd.DataFrame(records)


def load_finetune(ft_dir: Path) -> pd.DataFrame:
    """Load finetuned-decoder rows from finetune_decoder_ood output directory.

    latency_comparison.csv  – latency values already in **ms** (mean_ms).
    mae_comparison.csv       – columns: series, theta, method, mae_m

    Only returns **FT decoder** rows (HV baseline already in postprocess data).

    Returns a DataFrame with columns:
        method, label, group, mae_mean_m, latency_total_ms
    """
    mae_df = pd.read_csv(ft_dir / "mae_comparison.csv")
    lat_df = pd.read_csv(ft_dir / "latency_comparison.csv")

    # raw forward-pass latency per model name (ms)
    raw_lookup: Dict[str, float] = {
        row["model"]: row["mean_ms"]
        for _, row in lat_df.iterrows()
        if row["method"] == "raw"
    }

    # Build (model, method) → total latency (ms)
    lat_lookup: Dict[Tuple[str, str], float] = {}
    for _, row in lat_df.iterrows():
        m_model, m_method = row["model"], row["method"]
        ms = row["mean_ms"]
        if m_method == "raw":
            lat_lookup[(m_model, m_method)] = ms
        elif m_method in _FILTER_OVERHEAD_METHODS:
            lat_lookup[(m_model, m_method)] = raw_lookup.get(m_model, 0.0) + ms
        elif m_method in _AR_TOTAL_METHODS:
            lat_lookup[(m_model, m_method)] = ms
        else:
            lat_lookup[(m_model, m_method)] = ms

    ft_df = mae_df[mae_df["series"] == "finetuned_decoder"].copy()

    records = []
    for mc in ft_df["method"].unique():
        sub = ft_df[ft_df["method"] == mc]
        mae_mean = float(sub["mae_m"].mean())
        lat_total = lat_lookup.get(("FT decoder", mc), float("nan"))
        lat_row = lat_df[(lat_df["model"] == "FT decoder") & (lat_df["method"] == mc)]
        label_str = lat_row["label"].iloc[0] if len(lat_row) else mc
        records.append({
            "method": mc,
            "label": f"FT decoder – {label_str}",
            "group": "FT decoder",
            "mae_mean_m": mae_mean,
            "latency_total_ms": lat_total,
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Pareto frontier
# ─────────────────────────────────────────────────────────────────────────────

def pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows that are Pareto-optimal (minimise both MAE and latency).

    A point is dominated if another point has ≤ MAE *and* ≤ latency with at
    least one strict inequality.
    """
    pts = df[["mae_mean_m", "latency_total_ms"]].values.astype(float)
    is_pareto = np.ones(len(pts), dtype=bool)
    for i, pt in enumerate(pts):
        if not is_pareto[i]:
            continue
        dominated_by_others = (
            (pts[:, 0] <= pt[0]) & (pts[:, 1] <= pt[1]) &
            ((pts[:, 0] < pt[0]) | (pts[:, 1] < pt[1]))
        )
        dominated_by_others[i] = False
        if dominated_by_others.any():
            is_pareto[i] = False
    return df[is_pareto].sort_values("latency_total_ms").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

_GROUP_STYLE: Dict[str, dict] = {
    "Oracle":      {"color": "#888888", "marker": "D", "zorder": 4, "s": 80},
    "HV baseline": {"color": "#2166ac", "marker": "o", "zorder": 3, "s": 80},
    "FT decoder":  {"color": "#1a9641", "marker": "s", "zorder": 3, "s": 80},
}

# Short method-name suffix → annotation text offset (dx_ms, dy_m)
_ANNOTATION_LABELS = {
    "HV – Raw",
    "HV – RTS (R=σ²)",
    "HV – AR+RTS (R=σ²)",
    "FT decoder – Raw",
    "FT decoder – RTS (R=σ²)",
    "FT decoder – AR+RTS (R=σ²)",
}
_OFFSET_MAP: Dict[str, Tuple[float, float]] = {
    "HV – Raw":                    (2.5,  0.30),
    "HV – RTS (R=σ²)":             (2.5, -0.45),
    "HV – AR+RTS (R=σ²)":          (2.5,  0.30),
    "FT decoder – Raw":             (2.5,  0.30),
    "FT decoder – RTS (R=σ²)":     (2.5, -0.45),
    "FT decoder – AR+RTS (R=σ²)":  (2.5,  0.30),
}


def plot_pareto(df: pd.DataFrame, output_dir: Path) -> None:
    """Create and save the MAE vs. latency Pareto frontier figure."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute frontier excluding Oracle reference points
    non_oracle = df[df["group"] != "Oracle"].copy()
    frontier = pareto_frontier(non_oracle)

    fig, ax = plt.subplots(figsize=(11, 6.5))

    # ── scatter ──────────────────────────────────────────────────────────────
    for group, style in _GROUP_STYLE.items():
        sub = df[df["group"] == group]
        ax.scatter(
            sub["latency_total_ms"], sub["mae_mean_m"],
            s=style["s"],
            color=style["color"],
            marker=style["marker"],
            zorder=style["zorder"],
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
            label=group,
        )

    # ── Pareto frontier step line ─────────────────────────────────────────────
    if len(frontier) >= 2:
        ax.step(
            frontier["latency_total_ms"],
            frontier["mae_mean_m"],
            where="post",
            color="#d73027",
            linewidth=1.8,
            linestyle="--",
            zorder=5,
            label="Pareto frontier",
        )
    # highlight frontier points with a star overlay
    ax.scatter(
        frontier["latency_total_ms"],
        frontier["mae_mean_m"],
        s=150,
        color="#d73027",
        marker="*",
        zorder=6,
        edgecolors="white",
        linewidths=0.5,
    )

    # ── annotations ──────────────────────────────────────────────────────────
    for _, row in df.iterrows():
        if row["label"] not in _ANNOTATION_LABELS:
            continue
        dx, dy = _OFFSET_MAP.get(row["label"], (2.5, 0.25))
        short_name = row["label"].split(" – ", 1)[-1]
        ax.annotate(
            short_name,
            xy=(row["latency_total_ms"], row["mae_mean_m"]),
            xytext=(row["latency_total_ms"] + dx, row["mae_mean_m"] + dy),
            fontsize=7.5,
            color="#333333",
            arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.7),
            ha="left",
        )

    # ── axes labels & formatting ──────────────────────────────────────────────
    ax.set_xlabel("Total inference latency (ms per trajectory)", fontsize=11)
    ax.set_ylabel("Mean MAE (m)  [θ ∈ {0.0, 0.1, 0.2, 0.3}]", fontsize=11)
    ax.set_title(
        "OoD Generalisation: MAE vs. Inference Latency — Pareto Frontier",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_xlim(left=0)

    plt.tight_layout()
    out_png = output_dir / "pareto_ood.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")

    # ── save combined table ───────────────────────────────────────────────────
    out_csv = output_dir / "pareto_all_methods.csv"
    df.sort_values(["group", "latency_total_ms"]).reset_index(drop=True).to_csv(
        out_csv, index=False, float_format="%.4f"
    )
    print(f"Saved: {out_csv}")

    # print frontier summary
    print("\n=== Pareto-optimal points (non-Oracle) ===")
    print(
        frontier[["label", "mae_mean_m", "latency_total_ms"]]
        .to_string(index=False, float_format=lambda x: f"{x:.2f}")
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Pareto frontier: OoD MAE vs. inference latency",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--postprocess-dir", type=Path, default=_DEFAULT_PP_DIR,
        help="Directory containing eval_ood_postprocess.py outputs.",
    )
    parser.add_argument(
        "--finetune-dir", type=Path, default=_DEFAULT_FT_DIR,
        help="Directory containing finetune_decoder_ood.py outputs.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=_DEFAULT_OUT_DIR,
        help="Directory to write pareto_ood.png and pareto_all_methods.csv.",
    )
    args = parser.parse_args()

    print(f"Loading postprocess results from: {args.postprocess_dir}")
    pp_df = load_postprocess(args.postprocess_dir)

    print(f"Loading finetune results from:    {args.finetune_dir}")
    ft_df = load_finetune(args.finetune_dir)

    combined = pd.concat([pp_df, ft_df], ignore_index=True)

    print("\n=== All methods (group | MAE m | latency ms) ===")
    print(
        combined[["group", "label", "mae_mean_m", "latency_total_ms"]]
        .to_string(index=False, float_format=lambda x: f"{x:.2f}")
    )

    plot_pareto(combined, args.output_dir)


if __name__ == "__main__":
    main()
