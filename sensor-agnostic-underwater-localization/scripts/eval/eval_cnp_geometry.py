"""
eval_cnp_geometry.py
====================

Evaluate a trained CNP baseline on the geometry-split data and quantify the
OUT-OF-POSITION DEGRADATION: how much worse the model localizes on held-out
sensor geometries (val/test) than on the geometries it trained on.

This is the number the sensor-displacement project exists to reduce. It is
reported three ways:

  1. Per-pool MAE: train (in-geometry, but unseen trajectories via the legacy
     within-geometry holdout is NOT used here -- instead we report training-pool
     MAE as an upper-reference of "seen geometry") vs val vs test.
  2. By novelty: held-out geometries split into INTERPOLATION (inside the
     training-centroid hull) vs EXTRAPOLATION (outside), from splits.json.
  3. Per-geometry MAE, sorted by centroid distance from the training centroids
     -> the degradation-vs-displacement curve (saved as a figure + CSV).

The "degradation" headline = MAE(held-out) - MAE(train pool), and the
extrapolation - interpolation gap.

USAGE
-----
    python eval_cnp_geometry.py \
        --data-dir ../data/data_random_positions/processed/geometry_split \
        --ckpt     ../runs/cnp_baseline/best.pt \
        --out-dir  ../runs/cnp_baseline/eval \
        --eval-ctx 15 --n-context-draws 5

Outputs:
    eval_report.txt           human-readable summary (headline degradation)
    per_geometry_mae.csv      MAE per geometry + region + centroid distance
    degradation_curve.png     MAE vs centroid distance, interp/extrap colored
"""

import os, sys, json, pickle, argparse
import numpy as np
import torch as t
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "models"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
try:
    from anp import DeterministicModel
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from anp import DeterministicModel

# reuse dataset/collate from the trainer if available, else inline minimal copies
try:
    from train_cnp_geometry import TrajectoryDataset, make_collate
except ImportError:
    class TrajectoryDataset:
        def __init__(self, p):
            self.samples = pickle.load(open(p, "rb"))
            x0 = np.asarray(self.samples[0]["X"]); y0 = np.asarray(self.samples[0]["y"])
            self.ppt, self.feat_dim = x0.shape; self.out_dim = y0.shape[-1]
        def __len__(self): return len(self.samples)
        def __getitem__(self, i):
            s = self.samples[i]
            return (t.as_tensor(np.asarray(s["X"], np.float32)),
                    t.as_tensor(np.asarray(s["y"], np.float32)),
                    int(s.get("geometry_id", -1)), float(s.get("theta", -1.0)))


@t.no_grad()
def per_geometry_mae(model, ds, device, eval_ctx, n_draws, seed=0):
    """Mean per-point Euclidean error, aggregated per geometry_id, averaged over
    n_draws random context selections for stability."""
    from collections import defaultdict
    rng = np.random.default_rng(seed)
    err_sum = defaultdict(float); err_cnt = defaultdict(int)
    model.eval()
    for draw in range(n_draws):
        for i in range(len(ds)):
            X, y, gid, th = ds[i]
            X = X.unsqueeze(0).to(device); y = y.unsqueeze(0).to(device)
            ppt = X.size(1)
            n_ctx = max(1, min(eval_ctx, ppt))
            idx = np.sort(rng.permutation(ppt)[:n_ctx])
            idx = t.as_tensor(idx, device=device)
            cx, cy = X[:, idx, :], y[:, idx, :]
            mean, var, *_ = model(cx, cy, X, None)
            dist = t.sqrt(((mean - y) ** 2).sum(-1) + 1e-12)  # (1, ppt)
            err_sum[gid] += float(dist.sum()); err_cnt[gid] += dist.numel()
    return {g: err_sum[g] / err_cnt[g] for g in err_sum}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--eval-ctx", type=int, default=15)
    ap.add_argument("--n-context-draws", type=int, default=5)
    ap.add_argument("--num-hidden", type=int, default=None,
                    help="override; else read from ckpt config")
    ap.add_argument("--device", default="cuda" if t.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = t.device(args.device)

    # load split labels + metadata
    with open(os.path.join(args.data_dir, "splits.json")) as f:
        split = json.load(f)
    with open(os.path.join(args.data_dir, "metadata.pkl"), "rb") as f:
        meta = pickle.load(f)
    center = np.array(split.get("center_of_centroids", [0.0, 0.0]))
    labels = {int(k): v for k, v in split.get("labels", {}).items()}

    # model
    ck = t.load(args.ckpt, map_location=device)
    feat_dim = ck.get("feat_dim"); out_dim = ck.get("out_dim", 3)
    nh = args.num_hidden or ck.get("config", {}).get("num_hidden", 128)
    model = DeterministicModel(num_hidden=nh, input_dim=feat_dim,
                               output_dim=out_dim).to(device)
    model.load_state_dict(ck["model"])

    # datasets
    pools = {}
    for name in ["train", "val", "test"]:
        p = os.path.join(args.data_dir, f"{name}_data.pkl")
        if os.path.exists(p):
            pools[name] = TrajectoryDataset(p)

    log = []
    def line(s): log.append(s); print(s)

    line("=" * 64)
    line("CNP baseline -- out-of-position degradation")
    line("=" * 64)
    line(f"checkpoint: {args.ckpt}  (epoch {ck.get('epoch','?')})")
    line(f"eval context size: {args.eval_ctx}  draws: {args.n_context_draws}")
    line("")

    pool_mae = {}
    geo_rows = []
    for name, ds in pools.items():
        gmae = per_geometry_mae(model, ds, device, args.eval_ctx,
                                args.n_context_draws)
        # pool-level mean across its geometries
        pool_mae[name] = float(np.mean(list(gmae.values())))
        for gid, m in sorted(gmae.items()):
            region = labels.get(gid, {}).get("region", "train")
            dist = labels.get(gid, {}).get("dist_from_center", 0.0)
            geo_rows.append({"pool": name, "geometry_id": gid, "mae": m,
                             "region": region, "dist": dist})
        line(f"{name:5s} pool MAE: {pool_mae[name]:.4f}  "
             f"({len(gmae)} geometries)")

    # headline degradation
    line("")
    train_ref = pool_mae.get("train", float("nan"))
    for name in ["val", "test"]:
        if name in pool_mae:
            deg = pool_mae[name] - train_ref
            pct = 100.0 * deg / train_ref if train_ref else float("nan")
            line(f"DEGRADATION {name} vs train: "
                 f"{deg:+.4f}  ({pct:+.1f}%)")

    # interpolation vs extrapolation (held-out only)
    held = [r for r in geo_rows if r["region"] in ("interp", "extrap")]
    if held:
        for reg in ["interp", "extrap"]:
            vals = [r["mae"] for r in held if r["region"] == reg]
            if vals:
                line(f"  {reg:6s} held-out MAE: {np.mean(vals):.4f} "
                     f"(n={len(vals)})")
        iv = [r["mae"] for r in held if r["region"] == "interp"]
        ev = [r["mae"] for r in held if r["region"] == "extrap"]
        if iv and ev:
            line(f"  extrap - interp gap: {np.mean(ev) - np.mean(iv):+.4f}")

    # write CSV
    csv_path = os.path.join(args.out_dir, "per_geometry_mae.csv")
    with open(csv_path, "w") as f:
        f.write("pool,geometry_id,region,dist_from_center,mae\n")
        for r in sorted(geo_rows, key=lambda r: (r["pool"], r["dist"])):
            f.write(f"{r['pool']},{r['geometry_id']},{r['region']},"
                    f"{r['dist']:.3f},{r['mae']:.5f}\n")

    # degradation-vs-displacement figure
    fig, ax = plt.subplots(figsize=(7.5, 5))
    colors = {"train": "#2980b9", "interp": "#27ae60", "extrap": "#c0392b"}
    for reg in ["train", "interp", "extrap"]:
        pts = [(r["dist"], r["mae"]) for r in geo_rows if r["region"] == reg]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, c=colors[reg], label=reg, s=60,
                       edgecolor="k", linewidth=0.3)
    ax.axhline(train_ref, color="#2980b9", ls="--", lw=1, alpha=0.6,
               label="train pool mean")
    ax.set_xlabel("sensor-centroid distance from training centroids [m]")
    ax.set_ylabel("localization MAE")
    ax.set_title("Out-of-position degradation (CNP baseline)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "degradation_curve.png"), dpi=200)

    line("")
    line(f"per-geometry CSV : {csv_path}")
    line(f"degradation curve: {os.path.join(args.out_dir, 'degradation_curve.png')}")
    line("")
    line("INTERPRETATION: a large val/test degradation and a positive extrap-")
    line("interp gap mean sensor displacement genuinely hurts the current model")
    line("-- i.e. there is a real problem for the spatial encoder to fix. A near-")
    line("zero degradation would mean the task is too easy / displacement too weak.")
    with open(os.path.join(args.out_dir, "eval_report.txt"), "w") as f:
        f.write("\n".join(log) + "\n")
    print(f"\nreport -> {args.out_dir}/eval_report.txt")


if __name__ == "__main__":
    main()
