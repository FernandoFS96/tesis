"""
trivial_baselines.py
====================

Non-learned reference baselines on the protocol of eval_suite.py: the same held-out
pool, the same transits per eval seed, context = the first 10 points of the
transit, targets = the remaining 40, and the same error (Euclidean over every
coordinate of y, averaged per transit, then over transits).

    python scripts/eval/trivial_baselines.py \
        --data-dir data/pfn10kd/hermite_distinct/processed/geometry_split \
        --out-dir output/eval_paper/baselines/b1

Trajectory only (no acoustics):
  ctx_centroid    the mean of the 10 context positions
  last_fix        the last context position
  const_velocity  the last context position moved on at the last context step's velocity
  linear_fit      the least-squares line through the 10 context positions, extrapolated
Acoustics against the context, no learning:
  nn_record       the position of the context point nearest in raw feature space
  nn_rms          the same on each sensor's RMS level (one number per sensor)
"""
import argparse, os, csv, sys
from collections import defaultdict
import numpy as np
import torch as t

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_suite as E

NCTX = 10


def predict(name, X, y, S):
    """X (ppt, F), y (ppt, D) of one transit -> predictions for points NCTX..ppt-1."""
    ppt = y.shape[0]; cy = y[:NCTX]; k = np.arange(NCTX, ppt)
    if name == "ctx_centroid":
        return np.repeat(cy.mean(0, keepdims=True), len(k), 0)
    if name == "last_fix":
        return np.repeat(cy[-1:], len(k), 0)
    if name == "const_velocity":
        return cy[-1] + (k - (NCTX - 1))[:, None] * (cy[-1] - cy[-2])
    if name == "linear_fit":
        A = np.c_[np.ones(NCTX), np.arange(NCTX)]
        coef, *_ = np.linalg.lstsq(A, cy, rcond=None)
        return np.c_[np.ones(len(k)), k] @ coef
    if name in ("nn_record", "nn_rms"):
        f = X if name == "nn_record" else np.sqrt((X.reshape(ppt, -1, S) ** 2).mean(1))
        d = ((f[NCTX:, None, :] - f[None, :NCTX, :]) ** 2).sum(-1)
        return cy[d.argmin(1)]
    raise ValueError(name)


NAMES = ["ctx_centroid", "last_fix", "const_velocity", "linear_fit", "nn_record", "nn_rms"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--eval-seeds", default="0,1,2")
    ap.add_argument("--cap-traj", type=int, default=6)
    args = ap.parse_args()
    data = E.load_data(args.data_dir, t.device("cpu"), load_cap=max(15, args.cap_traj * 3))
    S = data["n_sensors"]

    rows = []
    for es in (int(z) for z in args.eval_seeds.split(",")):
        items = E.subsample_test(data["by_gid"], args.cap_traj, es)
        per = defaultdict(list)          # name -> [(gid, transit error)]
        for X, y, gid, sp in items:
            X, y = X[0].numpy(), y[0].numpy()
            for n in NAMES:
                e = np.linalg.norm(predict(n, X, y, S) - y[NCTX:], axis=1).mean()
                per[n].append((gid, e))
        for n, pe in per.items():
            rows.append((n, "overall", es, float(np.mean([e for _, e in pe]))))
            buck = defaultdict(list)
            for g, e in pe:
                buck[data["labels"].get(g, "?")].append(e)
            for reg, v in buck.items():
                if reg != "?":
                    rows.append((n, reg, es, float(np.mean(v))))
        print(f"eval seed {es}: {len(items)} transits", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "results_per_seed.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "metric", "eval_seed", "value"])
        w.writerows([(n, m, es, f"{v:.4f}") for n, m, es, v in rows])
    agg = defaultdict(list)
    for n, m, es, v in rows:
        agg[(n, m)].append(v)
    with open(os.path.join(args.out_dir, "results.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "metric", "mean", "std", "n"])
        for (n, m), v in agg.items():
            w.writerow([n, m, f"{np.mean(v):.4f}", f"{np.std(v):.4f}", len(v)])
    regs = sorted({m for _, m in agg} - {"overall"})
    print(f"{'baseline':16s}{'overall':>10s}" + "".join(f"{r:>10s}" for r in regs))
    for n in NAMES:
        print(f"{n:16s}{np.mean(agg[(n, 'overall')]):10.2f}"
              + "".join(f"{np.mean(agg[(n, r)]):10.2f}" for r in regs))


if __name__ == "__main__":
    main()
