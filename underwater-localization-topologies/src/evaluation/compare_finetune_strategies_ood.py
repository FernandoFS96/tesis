"""
compare_finetune_strategies_ood.py
===================================
Head-to-head comparison of two fine-tuning strategies:
  • Decoder-only fine-tuning  (finetune_decoder_ood.py)
  • Last-layer fine-tuning    (finetune_last_layers_ood.py)

Reads mae_comparison.csv from both result directories and produces:
  - comparison_mae_vs_theta.png  — line plot per method+strategy
  - comparison_summary.csv       — mean MAE table

Usage
-----
python compare_finetune_strategies_ood.py \
    --decoder-dir     /home/fernando/tesis/underwater-localization-topologies/src/evaluation/results/finetune_decoder_ood \
    --last-layers-dir /home/fernando/tesis/underwater-localization-topologies/src/evaluation/results/finetune_last_layers_ood \
    --output-dir      results/compare_finetune_strategies
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def _load_mae_csv(path: Path) -> Dict[str, Dict[float, float]]:
    """
    Reads mae_comparison.csv produced by save_comparison_csv().
    Returns:  series_name -> {theta: mae_m}
    CSV columns: series, theta, method, mae_m
    """
    result: Dict[str, Dict[float, float]] = defaultdict(dict)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Composite key: "series / method"  (e.g. "finetuned_decoder / ar_kalman_cv_var")
            key   = f"{row['series']} / {row['method']}"
            theta = float(row["theta"])
            mae   = float(row["mae_m"])
            result[key][theta] = mae
    return dict(result)


# ---------------------------------------------------------------------------
# Visual style helpers
# ---------------------------------------------------------------------------

# Per-method linestyle / marker
_METHOD_STYLE: Dict[str, Tuple[str, str]] = {
    "raw":               ("--",  "s"),
    "alpha_beta":        ("-.",  "v"),
    "kalman_cv_var":     ("-.",  "^"),
    "ar_raw":            (":",   "p"),
    "ar_kalman_cv_var":  (":",   "D"),
    "kalman_rts_var":    ("-.",  ">"),
    "ar_kalman_rts_var": (":",   "*"),
}

# Per-strategy colour palette
_STRATEGY_COLORS = {
    "oracle":              "#2c3e50",   # dark slate (same for both)
    "highvar_baseline":    "#e74c3c",   # red
    "finetuned_decoder":   "#2980b9",   # blue
    "finetuned_last_layers": "#27ae60", # green
}

_STRATEGY_LABELS = {
    "oracle":                "Oracle (lowvar)",
    "highvar_baseline":      "HV baseline",
    "finetuned_decoder":     "FT decoder",
    "finetuned_last_layers": "FT last-layers",
}

_METHOD_LABELS = {
    "raw":               "Raw",
    "alpha_beta":        "α-β filter",
    "kalman_cv_var":     "Kalman fwd (R=σ²)",
    "ar_raw":            "AR+Raw",
    "ar_kalman_cv_var":  "AR+Kalman fwd",
    "kalman_rts_var":    "RTS (R=σ²)",
    "ar_kalman_rts_var": "AR+RTS (R=σ²)",
}


def _label_for(series: str, method: str) -> str:
    strat = _STRATEGY_LABELS.get(series, series)
    meth  = _METHOD_LABELS.get(method, method)
    return f"{strat} — {meth}"


# ---------------------------------------------------------------------------
# Build unified table from both CSVs
# ---------------------------------------------------------------------------

def _merge(
    dec_data:  Dict[str, Dict[float, float]],
    ll_data:   Dict[str, Dict[float, float]],
) -> Dict[str, Dict[float, float]]:
    """Merge both dicts, keeping oracle/hv from decoder dir (they should be identical)."""
    merged = {}
    # Oracle and HV baseline — take from decoder dir; skip duplicate from ll_dir
    for key, theta_dict in dec_data.items():
        series, _, method = key.partition(" / ")
        if series in ("oracle", "highvar_baseline", "finetuned_decoder"):
            merged[key] = theta_dict
    for key, theta_dict in ll_data.items():
        series, _, method = key.partition(" / ")
        if series == "finetuned_last_layers":
            merged[key] = theta_dict
    return merged


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _select_highlight_method(data: Dict[str, Dict[float, float]]) -> Optional[str]:
    """Pick the best AR method present in finetuned_decoder keys."""
    candidates = ["ar_kalman_rts_var", "ar_kalman_cv_var", "ar_raw", "raw"]
    existing = set()
    for key in data:
        series, _, method = key.partition(" / ")
        if series == "finetuned_decoder":
            existing.add(method)
    for c in candidates:
        if c in existing:
            return c
    return None


def plot_comparison(
    data:         Dict[str, Dict[float, float]],
    thetas:       List[float],
    highlight_m:  Optional[str],
    output_dir:   Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

    for ax, focus_series, title in [
        (axes[0], "finetuned_decoder",     "Decoder fine-tuning"),
        (axes[1], "finetuned_last_layers", "Last-layer fine-tuning"),
    ]:
        x = np.array(thetas)

        # Oracle ceiling
        oracle_key = f"oracle / raw"
        if oracle_key in data:
            ax.plot(x, [data[oracle_key].get(t, float("nan")) for t in thetas],
                    color=_STRATEGY_COLORS["oracle"], ls="-", marker="o",
                    markersize=5, lw=2, label="Oracle — Raw")

        # HV baseline (raw only to avoid clutter)
        hv_key = "highvar_baseline / raw"
        if hv_key in data:
            hv_v = np.array([data[hv_key].get(t, float("nan")) for t in thetas])
            oracle_v = np.array([data.get("oracle / raw", {}).get(t, float("nan"))
                                 for t in thetas])
            ax.plot(x, hv_v, color=_STRATEGY_COLORS["highvar_baseline"],
                    ls="--", marker="s", markersize=5, lw=1.5, label="HV baseline — Raw")
            ax.fill_between(x, oracle_v, hv_v, alpha=0.05,
                            color=_STRATEGY_COLORS["highvar_baseline"])

        # Fine-tuning series for this panel
        for key, theta_dict in data.items():
            series, _, method = key.partition(" / ")
            if series != focus_series:
                continue
            ls, marker = _METHOD_STYLE.get(method, ("--", "x"))
            color = _STRATEGY_COLORS.get(series, "#555555")
            lw    = 2.5 if method == highlight_m else 1.5
            alpha = 1.0 if method == highlight_m else 0.7
            label = _label_for(series, method)
            v = np.array([theta_dict.get(t, float("nan")) for t in thetas])
            ax.plot(x, v, color=color, ls=ls, marker=marker,
                    markersize=6 if method == highlight_m else 4,
                    lw=lw, alpha=alpha, label=label)

        ax.set_xticks(thetas)
        ax.set_xticklabels([f"{t:.1f}" for t in thetas])
        ax.set_xlabel("θ (channel variability)", fontsize=11)
        ax.set_ylabel("MAE (m)", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.suptitle("Fine-tuning strategy comparison — MAE vs θ", fontsize=13, y=1.01)
    plt.tight_layout()
    path = output_dir / "comparison_mae_vs_theta.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved {path}")


def plot_side_by_side(
    data:        Dict[str, Dict[float, float]],
    thetas:      List[float],
    highlight_m: Optional[str],
    output_dir:  Path,
) -> None:
    """Single panel comparing decoder vs last-layers for each method independently."""
    # Collect unique methods
    methods = sorted({key.partition(" / ")[2] for key in data})
    n_plots = len(methods)
    if n_plots == 0:
        return
    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), sharey=True)
    axes_flat = np.array(axes).flatten() if n_plots > 1 else [axes]

    x = np.array(thetas)
    oracle_v = np.array([data.get("oracle / raw", {}).get(t, float("nan")) for t in thetas])
    hv_v     = np.array([data.get("highvar_baseline / raw", {}).get(t, float("nan"))
                         for t in thetas])

    for ax, method in zip(axes_flat, methods):
        dec_key = f"finetuned_decoder / {method}"
        ll_key  = f"finetuned_last_layers / {method}"
        meth_lbl = _METHOD_LABELS.get(method, method)

        ax.fill_between(x, oracle_v, hv_v, alpha=0.05, color="#e74c3c")
        ax.plot(x, oracle_v, color="#2c3e50", ls="-",  lw=1.5, alpha=0.6, label="Oracle")
        ax.plot(x, hv_v,     color="#e74c3c", ls="--", lw=1.5, alpha=0.6, label="HV baseline")

        ls, marker = _METHOD_STYLE.get(method, ("--", "x"))
        if dec_key in data:
            v = np.array([data[dec_key].get(t, float("nan")) for t in thetas])
            ax.plot(x, v, color=_STRATEGY_COLORS["finetuned_decoder"],
                    ls=ls, marker=marker, markersize=6, lw=2, label="FT decoder")
        if ll_key in data:
            v = np.array([data[ll_key].get(t, float("nan")) for t in thetas])
            ax.plot(x, v, color=_STRATEGY_COLORS["finetuned_last_layers"],
                    ls=ls, marker=marker, markersize=6, lw=2, label="FT last-layers")

        ax.set_title(meth_lbl, fontsize=10)
        ax.set_xticks(thetas)
        ax.set_xticklabels([f"{t:.1f}" for t in thetas], fontsize=7)
        ax.set_xlabel("θ", fontsize=9)
        ax.set_ylabel("MAE (m)", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)

    # Hide empty axes
    for ax in axes_flat[n_plots:]:
        ax.set_visible(False)

    plt.suptitle("Per-method comparison: decoder vs last-layer fine-tuning", fontsize=12)
    plt.tight_layout()
    path = output_dir / "comparison_per_method.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved {path}")


# ---------------------------------------------------------------------------
# Summary CSV + print
# ---------------------------------------------------------------------------

def _mean_theta(theta_dict: Dict[float, float]) -> float:
    v = [m for m in theta_dict.values() if not np.isnan(m)]
    return float(np.mean(v)) if v else float("nan")


def save_summary_csv(
    data:       Dict[str, Dict[float, float]],
    output_dir: Path,
) -> None:
    path = output_dir / "comparison_summary.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["series", "method", "mean_mae_m"])
        for key in sorted(data.keys()):
            series, _, method = key.partition(" / ")
            mean_m = _mean_theta(data[key])
            w.writerow([series, method, f"{mean_m:.4f}"])
    print(f"[✓] Saved {path}")


def print_comparison_table(
    data:       Dict[str, Dict[float, float]],
) -> None:
    print("\n" + "=" * 72)
    print("Strategy comparison  (mean MAE across all θ, metres)")
    print("=" * 72)

    oracle_mean = _mean_theta(data.get("oracle / raw", {}))
    hv_mean     = _mean_theta(data.get("highvar_baseline / raw", {}))
    gap         = hv_mean - oracle_mean

    def pct(m): return 100.0 * (hv_mean - m) / max(gap, 1e-6)

    print(f"  {'Oracle raw (ceiling)':<35}: {oracle_mean:.2f} m")
    print(f"  {'HV raw (OoD baseline)':<35}: {hv_mean:.2f} m  (gap = {gap:.2f} m)")
    print("-" * 72)
    print(f"  {'Key':<35}  {'Mean MAE':>9}  {'% gap closed':>13}")
    print("-" * 72)

    for key in sorted(data.keys()):
        series, _, method = key.partition(" / ")
        if series in ("oracle", "highvar_baseline"):
            continue
        strat_lbl = _STRATEGY_LABELS.get(series, series)
        meth_lbl  = _METHOD_LABELS.get(method, method)
        label     = f"{strat_lbl} — {meth_lbl}"
        m         = _mean_theta(data[key])
        print(f"  {label:<35}  {m:9.2f}  {pct(m):10.1f}%")

    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare decoder-only vs last-layer fine-tuning strategies. "
            "Reads mae_comparison.csv from each result directory."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--decoder-dir",
        type=Path,
        default=Path("results/finetune_decoder_ood"),
        help="Directory produced by finetune_decoder_ood.py.",
    )
    p.add_argument(
        "--last-layers-dir",
        type=Path,
        default=Path("results/finetune_last_layers_ood"),
        help="Directory produced by finetune_last_layers_ood.py.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/compare_finetune_strategies"),
        help="Where to write comparison outputs.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    dec_csv = args.decoder_dir     / "mae_comparison.csv"
    ll_csv  = args.last_layers_dir / "mae_comparison.csv"

    if not dec_csv.exists():
        raise FileNotFoundError(f"Decoder results not found: {dec_csv}")
    if not ll_csv.exists():
        raise FileNotFoundError(f"Last-layers results not found: {ll_csv}")

    print(f"[load] decoder:     {dec_csv}")
    dec_data = _load_mae_csv(dec_csv)
    print(f"[load] last-layers: {ll_csv}")
    ll_data  = _load_mae_csv(ll_csv)

    data   = _merge(dec_data, ll_data)
    thetas = sorted({t for td in data.values() for t in td.keys()})
    print(f"[info] θ values: {thetas}")
    print(f"[info] series × method combinations: {len(data)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    highlight_m = _select_highlight_method(data)
    plot_comparison(data, thetas, highlight_m, args.output_dir)
    plot_side_by_side(data, thetas, highlight_m, args.output_dir)
    save_summary_csv(data, args.output_dir)
    print_comparison_table(data)

    print(f"\n[done] outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
