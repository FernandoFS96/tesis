"""
compare_baselines.py
====================

Aggregate the per-model evaluation outputs (CNP / ANP / RANP / RCNP) into a single comparison table and overlaid figures. 
Two things it produces:

  1. A combined baseline table (printed + baseline_comparison.csv): 
    per-pool MAE, degradation %, and the extrap-interp gap for each model, side by side.

  2. If context-sweep CSVs are present (from eval_np_geometry.py --eval-ctx-sweep),
     an OVERLAID MAE-vs-context figure across models, the decisive diagnostic for whether a model's accuracy depends on temporal context 
     (recurrent models interpolating the trajectory through time) versus per-point acoustic localization. 
     A model whose held-out MAE stays low even at context size 1-2 is doing acoustic localization; one that needs many context points is leaning on temporal smoothness.

INPUTS
------
Point --runs at the directory holding per-model run folders, each containing an `eval/` subdir with `eval_report.txt` (and optionally `per_geometry_mae.csv` and `context_sweep.csv`).
Folder names are used as model labels unless overridden.

USAGE
-----
    # auto-discover run folders under ../runs
    python compare_baselines.py --runs ../runs --out-dir ../runs/_comparison

    # explicit list
    python compare_baselines.py \
        --eval-dirs ../runs/cnp_baseline/eval ../runs/anp_baseline/eval ../runs/ranp_baseline/eval \
        --out-dir ../runs/_comparison

OUTPUTS (under --out-dir)
    baseline_comparison.csv     one row per model
    baseline_comparison.txt     pretty table
    context_sweep_overlay.png   MAE-vs-context, all models (if sweeps exist)
    degradation_overlay.png     degradation%-vs-context, all models (if sweeps exist)
"""

import os, re, csv, glob, argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Parse a single eval_report.txt
# --------------------------------------------------------------------------- #
def parse_eval_report(path):
    """Extract pool MAEs, degradation %, interp/extrap from an eval_report.txt."""
    txt = open(path).read()
    out = {}
    m = re.search(r"^(\w+) baseline", txt, re.M)
    out["model"] = m.group(1).lower() if m else os.path.basename(os.path.dirname(path))
    for pool in ["train", "val", "test"]:
        r = re.search(rf"{pool}\s+pool MAE:\s+([0-9.]+)", txt)
        out[f"{pool}_mae"] = float(r.group(1)) if r else float("nan")
    for pool in ["val", "test"]:
        r = re.search(rf"DEGRADATION {pool} vs train:\s*([+-][0-9.]+)\s*\(([+-][0-9.]+)%\)", txt)
        out[f"deg_{pool}_abs"] = float(r.group(1)) if r else float("nan")
        out[f"deg_{pool}_pct"] = float(r.group(2)) if r else float("nan")
    ri = re.search(r"interp held-out MAE:\s+([0-9.]+)", txt)
    re_ = re.search(r"extrap held-out MAE:\s+([0-9.]+)", txt)
    rg = re.search(r"extrap - interp gap:\s*([+-][0-9.]+)", txt)
    out["interp_mae"] = float(ri.group(1)) if ri else float("nan")
    out["extrap_mae"] = float(re_.group(1)) if re_ else float("nan")
    out["extrap_interp_gap"] = float(rg.group(1)) if rg else float("nan")
    ep = re.search(r"epoch (\d+)", txt)
    out["epoch"] = int(ep.group(1)) if ep else -1
    return out


def parse_context_sweep(path):
    """Read a context_sweep.csv -> list of dict rows."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({k: (float(v) if k != "model" else v) for k, v in r.items()})
    return rows


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def discover_eval_dirs(runs_root):
    dirs = []
    for d in sorted(glob.glob(os.path.join(runs_root, "*"))):
        ev = os.path.join(d, "eval")
        if os.path.isfile(os.path.join(ev, "eval_report.txt")):
            dirs.append(ev)
    return dirs


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=None, help="root holding <model>/eval/ dirs")
    ap.add_argument("--eval-dirs", nargs="*", default=None,
                    help="explicit list of eval dirs (override auto-discovery)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--labels", nargs="*", default=None,
                    help="optional labels matching --eval-dirs order")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    eval_dirs = args.eval_dirs or (discover_eval_dirs(args.runs) if args.runs else [])
    if not eval_dirs:
        raise SystemExit("No eval dirs found. Pass --runs or --eval-dirs.")

    # ---- table ----
    rows = []
    sweeps = {}
    for i, ed in enumerate(eval_dirs):
        rep = os.path.join(ed, "eval_report.txt")
        rec = parse_eval_report(rep)
        if args.labels and i < len(args.labels):
            rec["model"] = args.labels[i]
        rows.append(rec)
        sp = os.path.join(ed, "context_sweep.csv")
        if os.path.isfile(sp):
            sweeps[rec["model"]] = parse_context_sweep(sp)

    cols = ["model", "epoch", "train_mae", "val_mae", "test_mae",
            "deg_val_pct", "deg_test_pct", "interp_mae", "extrap_mae",
            "extrap_interp_gap"]
    csv_path = os.path.join(args.out_dir, "baseline_comparison.csv")
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

    # pretty text table
    def fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else str(v)
    widths = {c: max(len(c), max(len(fmt(r.get(c, ""))) for r in rows)) for c in cols}
    lines = []
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    lines.append(header); lines.append("-" * len(header))
    for r in rows:
        lines.append("  ".join(fmt(r.get(c, "")).ljust(widths[c]) for c in cols))
    table = "\n".join(lines)
    with open(os.path.join(args.out_dir, "baseline_comparison.txt"), "w") as f:
        f.write(table + "\n")
    print(table)
    print(f"\nwrote {csv_path}")

    # ---- overlaid context sweeps ----
    if sweeps:
        # MAE vs context (test pool) for each model
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for model, srows in sweeps.items():
            xs = [r["eval_ctx"] for r in srows]
            ys = [r["test_mae"] for r in srows]
            ax.plot(xs, ys, "o-", label=model)
        ax.set_xlabel("eval context size")
        ax.set_ylabel("test-pool MAE")
        ax.set_title("MAE vs context size (held-out test geometries)\n"
                     "flat-at-low-context => acoustic; steep => temporal reliance")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "context_sweep_overlay.png"), dpi=200)

        # degradation% vs context for each model
        fig2, ax2 = plt.subplots(figsize=(8, 5.5))
        for model, srows in sweeps.items():
            xs = [r["eval_ctx"] for r in srows]
            ys = [r["deg_test_pct"] for r in srows]
            ax2.plot(xs, ys, "o-", label=model)
        ax2.axhline(0, color="k", lw=0.8, alpha=0.5)
        ax2.set_xlabel("eval context size")
        ax2.set_ylabel("test degradation %")
        ax2.set_title("Out-of-position degradation vs context size")
        ax2.legend(); ax2.grid(alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(os.path.join(args.out_dir, "degradation_overlay.png"), dpi=200)
        print("wrote context_sweep_overlay.png and degradation_overlay.png")
    else:
        print("(no context_sweep.csv found in any eval dir, run eval with "
              "--eval-ctx-sweep to enable the temporal-reliance overlay)")


if __name__ == "__main__":
    main()
