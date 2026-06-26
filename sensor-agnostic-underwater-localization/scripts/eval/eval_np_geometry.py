"""
eval_np_geometry.py
===================

Unified evaluation of a trained CNP / ANP / RANP baseline on the geometry-split data. It reads the model type and calling convention from the checkpoint, 
so the same command evaluates any of the three the same way, producing comparable out-of-position degradation numbers.

Reported (identically to the original CNP eval):
  * per-pool MAE (train / val / test)
  * degradation: MAE(val/test) - MAE(train), absolute and %
  * interpolation vs extrapolation held-out MAE + the extrap-interp gap
  * per-geometry MAE CSV, sorted by centroid distance
  * degradation-vs-displacement figure

USAGE
-----
    python eval_np_geometry.py \
        --data-dir ../../data/data_random_positions/hermite_shared/processed/geometry_split \
        --ckpt     ../runs/anp_baseline/best.pt \
        --out-dir  ../runs/anp_baseline/eval \
        --eval-ctx 20 --n-context-draws 5

    Context-size sweep (temporal-reliance diagnostic):
    python eval_np_geometry.py  \
        --data-dir ../../data/data_random_positions/hermite_shared/processed/geometry_split \
        --ckpt     ../runs/anp_baseline/best.pt \
        --out-dir  ../runs/anp_baseline/eval --eval-ctx-sweep "1,2,3,5,10,20" \
        --ctx-sample-mode first

    Shuffle test:
    python eval_np_geometry.py \
        --data-dir ../../data/data_random_positions/hermite_shared/processed/geometry_split \
        --ckpt     ../runs/anp_baseline/best.pt \
        --out-dir  ../runs/anp_baseline/eval_shuf --eval-ctx 20 --shuffle-temporal

To see results for all three models (CNP, ANP, RANP) in one place, run the above three commands for each model.
To compare all three at once, point --ckpt at each best.pt in turn.
"""

import os, sys, json, pickle, argparse
import numpy as np
import torch as t
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

from tqdm import tqdm

# Add the repo root (parent of src/) so that `from src.models.anp import ...` works.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import src.models.anp as anp_mod  # type: ignore
import src.models.r_anp as ranp_mod  # type: ignore


def build_model(name, num_hidden, input_dim, output_dim,
                rnn_type="lstm", rnn_layers=1, rnn_dropout=0.0, dropout=0.1):
    name = name.lower()
    if name == "cnp":
        return anp_mod.DeterministicModel(num_hidden, input_dim, output_dim,
                                          dropout=dropout), "split"
    if name == "anp":
        return anp_mod.LatentModel(num_hidden, input_dim, output_dim,
                                   dropout=dropout), "split"
    if name == "ranp":
        return ranp_mod.LatentModel(num_hidden, input_dim, output_dim,
                                    rnn_type=rnn_type, rnn_layers=rnn_layers,
                                    rnn_dropout=rnn_dropout, dropout=dropout), "indexed"
    if name == "rcnp":
        return ranp_mod.DeterministicModel(num_hidden, input_dim, output_dim,
                                           rnn_type=rnn_type, rnn_layers=rnn_layers,
                                           rnn_dropout=rnn_dropout, dropout=dropout), "indexed"
    raise ValueError(f"unknown model '{name}'")


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
def forward_one(model, conv, X, y, idx, device, shuffle_temporal=False, perm=None):
    """Single-trajectory forward for either convention. X,y: (1, ppt, ·).

    If shuffle_temporal is True, the trajectory point ORDER is permuted before
    the forward pass (and predictions are unpermuted before returning). This is
    a diagnostic: a recurrent model that relies on temporal smoothness will
    degrade sharply under shuffling, whereas a purely acoustic model is
    order-invariant. `perm` (a fixed permutation) can be supplied so the same
    shuffle is applied across draws/models for comparability."""
    ppt = X.size(1)
    if shuffle_temporal:
        if perm is None:
            perm = t.randperm(ppt, device=device)
        inv = t.argsort(perm)
        Xs = X[:, perm, :]; ys = y[:, perm, :]
        # remap context indices into the permuted order
        pos = t.empty(ppt, dtype=t.long, device=device); pos[perm] = t.arange(ppt, device=device)
        idx_s = pos[idx]
        cy = ys[:, idx_s, :]
        if conv == "split":
            cx = Xs[:, idx_s, :]
            mean, var, *rest = model(cx, cy, Xs, None)
        else:
            ti = t.arange(ppt, device=device, dtype=t.long)
            mean, var, *rest = model(Xs, idx_s, cy, ti, None)
        # unpermute predictions back to original order
        return (mean[:, inv, :], var[:, inv, :], *rest)
    cy = y[:, idx, :]
    if conv == "split":
        cx = X[:, idx, :]
        return model(cx, cy, X, None)
    else:
        ti = t.arange(ppt, device=device, dtype=t.long)
        return model(X, idx, cy, ti, None)


@t.no_grad()
def per_geometry_mae(model, conv, ds, device, eval_ctx, n_draws, seed=0,
                     shuffle_temporal=False, y_mean=None, y_std=None,
                     ctx_sample_mode="random"):
    rng = np.random.default_rng(seed)
    err_sum = defaultdict(float); err_cnt = defaultdict(int)
    model.eval()
    ym = y_mean.to(device) if y_mean is not None else None
    ys = y_std.to(device)  if y_std  is not None else None
    for _ in range(n_draws):
        for i in range(len(ds)):
            X, y, gid, th = ds[i]
            X = X.unsqueeze(0).to(device); y = y.unsqueeze(0).to(device)
            y_raw = y
            ppt = X.size(1)
            n_ctx = max(1, min(eval_ctx, ppt))
            if ctx_sample_mode == "first":
                idx = t.arange(n_ctx, device=device, dtype=t.long)
            else:
                idx = t.as_tensor(np.sort(rng.permutation(ppt)[:n_ctx]),
                                  device=device, dtype=t.long)
            y_in = (y - ym) / ys if ym is not None else y
            mean, var, *_ = forward_one(model, conv, X, y_in, idx, device,
                                        shuffle_temporal=shuffle_temporal)
            if ym is not None:
                mean = mean * ys + ym
            dist = t.sqrt(((mean - y_raw) ** 2).sum(-1) + 1e-12)
            err_sum[gid] += float(dist.sum()); err_cnt[gid] += dist.numel()
    return {g: err_sum[g] / err_cnt[g] for g in err_sum}


def pool_means(model, conv, pools, labels, device, eval_ctx, n_draws,
               shuffle_temporal=False, y_mean=None, y_std=None,
               ctx_sample_mode="random"):
    """Returns {pool: mean_mae} and per-geometry rows for one context size."""
    pm = {}; rows = []
    for nm, ds in pools.items():
        gmae = per_geometry_mae(model, conv, ds, device, eval_ctx, n_draws,
                                shuffle_temporal=shuffle_temporal,
                                y_mean=y_mean, y_std=y_std,
                                ctx_sample_mode=ctx_sample_mode)
        pm[nm] = float(np.mean(list(gmae.values())))
        for gid, m in sorted(gmae.items()):
            region = labels.get(gid, {}).get("region", "train")
            dist = labels.get(gid, {}).get("dist_from_center", 0.0)
            rows.append({"pool": nm, "geometry_id": gid, "mae": m,
                         "region": region, "dist": dist})
    return pm, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--eval-ctx", type=int, default=20)
    ap.add_argument("--n-context-draws", type=int, default=5)
    ap.add_argument("--eval-ctx-sweep", default=None,
                    help="comma list of context sizes, e.g. '1,2,3,5,10,20'. "
                         "Runs the per-pool MAE at each and writes a context "
                         "sweep CSV + figure (the temporal-reliance diagnostic).")
    ap.add_argument("--shuffle-temporal", action="store_true",
                    help="permute trajectory point order at eval (diagnostic: "
                         "recurrent models collapse, acoustic models are invariant).")
    ap.add_argument("--ctx-sample-mode", default=None, choices=["first", "random"],
                    help="context sampling mode. If omitted, reads from checkpoint "
                         "config (falls back to 'random' for old checkpoints).")
    ap.add_argument("--device", default="cuda" if t.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = t.device(args.device)

    with open(os.path.join(args.data_dir, "splits.json")) as f:
        split = json.load(f)
    labels = {int(k): v for k, v in split.get("labels", {}).items()}

    ck = t.load(args.ckpt, map_location=device)
    name = ck.get("model_name", "cnp")
    conv = ck.get("convention", "split")
    feat_dim = ck.get("feat_dim"); out_dim = ck.get("out_dim", 3)
    cfg = ck.get("config", {})
    nh = cfg.get("num_hidden", 128)
    model, conv2 = build_model(name, nh, feat_dim, out_dim,
                               rnn_type=cfg.get("rnn_type", "lstm"),
                               rnn_layers=cfg.get("rnn_layers", 1),
                               rnn_dropout=cfg.get("rnn_dropout", 0.0),
                               dropout=cfg.get("dropout", 0.1))
    conv = conv or conv2
    model = model.to(device); model.load_state_dict(ck["model"])

    # normalization stats: auto-loaded from checkpoint (None for old checkpoints)
    y_mean = ck.get("y_mean")
    y_std  = ck.get("y_std")

    # context sampling mode: CLI flag overrides checkpoint config
    ctx_sample_mode = args.ctx_sample_mode or cfg.get("ctx_sample_mode", "random")

    pools = {}
    for nm in ["train", "val", "test"]:
        pth = os.path.join(args.data_dir, f"{nm}_data.pkl")
        if os.path.exists(pth):
            pools[nm] = TrajectoryDataset(pth)

    # ---- Optional context-size sweep (temporal-reliance diagnostic) ----
    if args.eval_ctx_sweep:
        ctx_sizes = [int(c) for c in args.eval_ctx_sweep.split(",") if c.strip()]
        sweep_rows = []
        print(f"[{name}] context sweep: {ctx_sizes} "
              f"(shuffle_temporal={args.shuffle_temporal})")
        for c in tqdm(ctx_sizes, desc=f"{name} ctx-sweep"):
            pm, _ = pool_means(model, conv, pools, labels, device, c,
                               args.n_context_draws,
                               shuffle_temporal=args.shuffle_temporal,
                               y_mean=y_mean, y_std=y_std,
                               ctx_sample_mode=ctx_sample_mode)
            row = {"model": name, "eval_ctx": c,
                   "train": pm.get("train", float("nan")),
                   "val": pm.get("val", float("nan")),
                   "test": pm.get("test", float("nan"))}
            tr = row["train"]
            row["deg_val_pct"] = 100.0*(row["val"]-tr)/tr if tr else float("nan")
            row["deg_test_pct"] = 100.0*(row["test"]-tr)/tr if tr else float("nan")
            sweep_rows.append(row)
        sweep_csv = os.path.join(args.out_dir, "context_sweep.csv")
        with open(sweep_csv, "w") as f:
            f.write("model,eval_ctx,train_mae,val_mae,test_mae,deg_val_pct,deg_test_pct\n")
            for r in sweep_rows:
                f.write(f"{r['model']},{r['eval_ctx']},{r['train']:.5f},"
                        f"{r['val']:.5f},{r['test']:.5f},"
                        f"{r['deg_val_pct']:.3f},{r['deg_test_pct']:.3f}\n")
        # figure: MAE vs context for the three pools
        figs, axs = plt.subplots(1, 2, figsize=(13, 5))
        xs = [r["eval_ctx"] for r in sweep_rows]
        for pool, col in [("train","#2980b9"),("val","#e67e22"),("test","#c0392b")]:
            axs[0].plot(xs, [r[pool] for r in sweep_rows], "o-", color=col, label=pool)
        axs[0].set_xlabel("eval context size"); axs[0].set_ylabel("MAE")
        axs[0].set_title(f"{name.upper()}: MAE vs context size"); axs[0].legend(); axs[0].grid(alpha=0.3)
        axs[1].plot(xs, [r["deg_val_pct"] for r in sweep_rows], "o-", label="val deg %")
        axs[1].plot(xs, [r["deg_test_pct"] for r in sweep_rows], "s-", label="test deg %")
        axs[1].axhline(0, color="k", lw=0.8, alpha=0.5)
        axs[1].set_xlabel("eval context size"); axs[1].set_ylabel("degradation %")
        axs[1].set_title(f"{name.upper()}: out-of-position degradation vs context")
        axs[1].legend(); axs[1].grid(alpha=0.3)
        figs.tight_layout(); figs.savefig(os.path.join(args.out_dir, "context_sweep.png"), dpi=200)
        print(f"[{name}] wrote {sweep_csv} and context_sweep.png")

    log = []
    def line(s): log.append(s); print(s)
    line("=" * 64)
    line(f"{name.upper()} baseline -- out-of-position degradation")
    line("=" * 64)
    line(f"checkpoint: {args.ckpt}  (epoch {ck.get('epoch','?')}, conv={conv})")
    line(f"eval context size: {args.eval_ctx}  draws: {args.n_context_draws}"
         f"  shuffle_temporal={args.shuffle_temporal}"
         f"  ctx_sample_mode={ctx_sample_mode}"
         f"  normalize_y={y_mean is not None}")
    line("")

    pool_mae = {}; geo_rows = []
    for nm, ds in pools.items():
        gmae = per_geometry_mae(model, conv, ds, device, args.eval_ctx,
                                args.n_context_draws,
                                shuffle_temporal=args.shuffle_temporal,
                                y_mean=y_mean, y_std=y_std,
                                ctx_sample_mode=ctx_sample_mode)
        pool_mae[nm] = float(np.mean(list(gmae.values())))
        for gid, m in sorted(gmae.items()):
            region = labels.get(gid, {}).get("region", "train")
            dist = labels.get(gid, {}).get("dist_from_center", 0.0)
            geo_rows.append({"pool": nm, "geometry_id": gid, "mae": m,
                             "region": region, "dist": dist})
        line(f"{nm:5s} pool MAE: {pool_mae[nm]:.4f}  ({len(gmae)} geometries)")

    line("")
    train_ref = pool_mae.get("train", float("nan"))
    for nm in ["val", "test"]:
        if nm in pool_mae:
            deg = pool_mae[nm] - train_ref
            pct = 100.0 * deg / train_ref if train_ref else float("nan")
            line(f"DEGRADATION {nm} vs train: {deg:+.4f}  ({pct:+.1f}%)")

    held = [r for r in geo_rows if r["region"] in ("interp", "extrap")]
    if held:
        for reg in ["interp", "extrap"]:
            vals = [r["mae"] for r in held if r["region"] == reg]
            if vals:
                line(f"  {reg:6s} held-out MAE: {np.mean(vals):.4f} (n={len(vals)})")
        iv = [r["mae"] for r in held if r["region"] == "interp"]
        ev = [r["mae"] for r in held if r["region"] == "extrap"]
        if iv and ev:
            line(f"  extrap - interp gap: {np.mean(ev) - np.mean(iv):+.4f}")

    csv_path = os.path.join(args.out_dir, "per_geometry_mae.csv")
    with open(csv_path, "w") as f:
        f.write("pool,geometry_id,region,dist_from_center,mae\n")
        for r in sorted(geo_rows, key=lambda r: (r["pool"], r["dist"])):
            f.write(f"{r['pool']},{r['geometry_id']},{r['region']},"
                    f"{r['dist']:.3f},{r['mae']:.5f}\n")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    colors = {"train": "#2980b9", "interp": "#27ae60", "extrap": "#c0392b"}
    for reg in ["train", "interp", "extrap"]:
        pts = [(r["dist"], r["mae"]) for r in geo_rows if r["region"] == reg]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, c=colors[reg], label=reg, s=60,
                       edgecolor="k", linewidth=0.3)
    ax.axhline(train_ref, color="#2980b9", ls="--", lw=1, alpha=0.6)
    ax.set_xlabel("sensor-centroid distance from training centroids [m]")
    ax.set_ylabel("localization MAE")
    ax.set_title(f"Out-of-position degradation ({name.upper()} baseline)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "degradation_curve.png"), dpi=200)

    line("")
    line(f"per-geometry CSV : {csv_path}")
    line(f"degradation curve: {os.path.join(args.out_dir, 'degradation_curve.png')}")
    with open(os.path.join(args.out_dir, "eval_report.txt"), "w") as f:
        f.write("\n".join(log) + "\n")
    print(f"\nreport -> {args.out_dir}/eval_report.txt")


if __name__ == "__main__":
    main()