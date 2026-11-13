"""compare_np_magnetic.py -- latent vs no-latent comparison of the runs made by
train_np_magnetic.py (training/run_np_magnetic_matrix.sh).

For every arm it averages each test set over eval seeds, then reports the mean
+- std over training seeds. Then, for each latent / no-latent pair, it reports
the per-seed paired difference (latent minus no-latent; negative MAE or NLL
means the latent helps):

  published architecture    anp_orig_fixed - cnp_orig   (the latent, working)
                            anp_orig       - cnp_orig   (the latent as published)
  sensor-agnostic           anp            - cnp        (raw data)
                            anp_norm       - cnp_norm   (normalized data: --normalize,
                                                         run names <arch>_norm)
                            anp_v2         - cnp_v2     (--normalize --dropout 0 --attn-ffn,
                                                         physical variance floor)

The no-context baselines reg (sensor-set regressor) and mlp (the paper's MLP
template) appear in the per-arm table only.

Only seeds present for both arms of a pair are compared.

  python training/compare_np_magnetic.py                      # held-out, both modes
  python training/compare_np_magnetic.py --population full    # eval_all_models / Table A2 style
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
ARMS = ("cnp_orig", "anp_orig_fixed", "anp_orig", "cnp", "anp", "cnp_norm", "anp_norm",
        "cnp_v2", "anp_v2", "mlp", "reg")
PAIRS = (("anp_orig_fixed", "cnp_orig"), ("anp_orig", "cnp_orig"), ("anp", "cnp"),
         ("anp_norm", "cnp_norm"), ("anp_v2", "cnp_v2"))
METRICS = ("mae_l1_single", "mae_l1", "eucl", "nll", "cov1", "cov2")


def load(runs_dir, regime, population):
    """-> {arch: {seed: {(mode, test_set): {metric: value}}}}, group/population labels."""
    name = "test_results.csv" if population == "heldout" else "test_results_full.csv"
    out = defaultdict(dict)
    meta = {}
    for f in sorted(glob.glob(os.path.join(runs_dir, regime, "*", "seed_*", name))):
        acc = defaultdict(lambda: defaultdict(list))
        for r in csv.DictReader(open(f)):
            key = (r["mode"], r["test_set"])
            meta[r["test_set"]] = (r["group"], r["population"])
            for m in METRICS:
                acc[key][m].append(float(r[m]))
        arch, seed = Path(f).parts[-3], int(Path(f).parts[-2].split("_")[1])
        out[arch][seed] = {k: {m: float(np.mean(v)) for m, v in d.items()} for k, d in acc.items()}
    return out, meta


def training_info(runs_dir, regime):
    rows = []
    for ck in sorted(glob.glob(os.path.join(runs_dir, regime, "*", "seed_*", "best.pt"))):
        d = Path(ck).parent
        c = torch.load(ck, map_location="cpu", weights_only=False)
        log = list(csv.DictReader(open(d / "train_log.csv")))
        at_best = next((r for r in log if int(r["epoch"]) == c["epoch"]), log[-1])
        rows.append((d.parts[-2], d.parts[-1], c["epoch"], len(log), c["best_val_mae"],
                     float(at_best["train_kl"])))
    return rows


def fmt(v):
    return f"{np.mean(v):7.3f} +- {np.std(v):5.3f}" if len(v) > 1 else f"{np.mean(v):7.3f}         "


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs-dir", default=str(ROOT / "training" / "np_magnetic_runs"))
    ap.add_argument("--regime", default="interpolation", help="interpolation | extrapolation (training regime)")
    ap.add_argument("--population", choices=("heldout", "full"), default="heldout")
    ap.add_argument("--modes", default="interp,extrap")
    ap.add_argument("--metric", default="mae_l1_single", choices=METRICS,
                    help="metric for the per-arm table (the paired table always shows mae_l1_single and nll)")
    ap.add_argument("--arms", default=None, help="comma-separated subset of arms for the per-arm table")
    ap.add_argument("--out-csv", default=None, help="write the paired differences here")
    a = ap.parse_args()

    res, meta = load(a.runs_dir, a.regime, a.population)
    if not res:
        raise SystemExit(f"no results under {a.runs_dir}/{a.regime}")
    arms = [x for x in ARMS if x in res and (a.arms is None or x in a.arms.split(","))]
    sets = list(meta)

    print(f"runs: {a.runs_dir}/{a.regime}   population: {a.population}")
    print("training (best epoch / epochs run / best val MAE / train KL at best epoch):")
    for arch, seed, be, n, bv, kl in training_info(a.runs_dir, a.regime):
        print(f"  {arch:15s} {seed:7s} best@{be:<5d} of {n:<5d} val {bv:7.3f}  kl {kl:.3e}")

    paired_rows = []
    for mode in a.modes.split(","):
        print(f"\n=== [{mode}] {a.metric}: mean +- std over training seeds (n seeds in brackets)")
        print(f"{'test set':20s} " + " ".join(f"{x:>21s}" for x in arms))
        for s in sets:
            if meta[s][0] == "scenario" and s == next(t for t in sets if meta[t][0] == "scenario"):
                print("-- Table 3 scenarios --")
            cells = []
            for arch in arms:
                v = [res[arch][sd][(mode, s)][a.metric] for sd in res[arch] if (mode, s) in res[arch][sd]]
                cells.append(f"{fmt(v)} [{len(v)}]" if v else f"{'-':>21s}")
            print(f"{s:20s} " + " ".join(f"{c:>21s}" for c in cells))

        for lat, nolat in PAIRS:
            if lat not in res or nolat not in res:
                continue
            seeds = sorted(set(res[lat]) & set(res[nolat]))
            if not seeds:
                continue
            print(f"\n--- [{mode}] {lat} - {nolat}  (seeds {seeds}; negative = latent better)")
            print(f"{'test set':20s} {'d mae_l1_single':>22s} {'latent wins':>11s} {'d nll':>22s} "
                  f"{'cov1 lat/no':>12s}")
            for s in sets:
                k = (mode, s)
                if not all(k in res[lat][sd] and k in res[nolat][sd] for sd in seeds):
                    continue
                dm = [res[lat][sd][k]["mae_l1_single"] - res[nolat][sd][k]["mae_l1_single"] for sd in seeds]
                dn = [res[lat][sd][k]["nll"] - res[nolat][sd][k]["nll"] for sd in seeds]
                c1 = (np.mean([res[lat][sd][k]["cov1"] for sd in seeds]),
                      np.mean([res[nolat][sd][k]["cov1"] for sd in seeds]))
                wins = sum(d < 0 for d in dm)
                print(f"{s:20s} {fmt(dm):>22s} {f'{wins}/{len(seeds)}':>11s} {fmt(dn):>22s} "
                      f"{c1[0]:5.3f}/{c1[1]:5.3f}")
                paired_rows.append({"mode": mode, "pair": f"{lat}-{nolat}", "test_set": s,
                                    "group": meta[s][0], "population": meta[s][1], "n_seeds": len(seeds),
                                    "d_mae_mean": np.mean(dm), "d_mae_std": np.std(dm), "latent_wins": wins,
                                    "d_nll_mean": np.mean(dn), "d_nll_std": np.std(dn),
                                    "cov1_latent": c1[0], "cov1_nolatent": c1[1]})
    if a.out_csv and paired_rows:
        with open(a.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(paired_rows[0]))
            w.writeheader(); w.writerows(paired_rows)
        print(f"\npaired differences -> {a.out_csv}")


if __name__ == "__main__":
    main()
