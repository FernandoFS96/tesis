"""
compare_offline_online.py
=========================

Compare an OFFLINE recurrent NP (ranp / rcnp, convention 'indexed') against the
ONLINE streaming variant (online_ranp, convention 'online') under the EXACT same
scenario, so you can read off the price of going online-deployable.

"Same scenario" means: same trajectories, same context-fix indices, same context
sizes, same RNG seed and draw order. The offline model is allowed its full power
(whole-sequence RNN + non-causal context aggregation); the online model only ever
sees context causally, streamed one chunk at a time via model.step /
model.register_context. The gap between the two curves is exactly the
offline->online degradation.

USAGE
-----
    python compare_offline_online.py \
        --data-dir   .../processed/geometry_split \
        --offline-ckpt .../ranp/best.pt \
        --online-ckpt  .../online_ranp/best.pt \
        --out-dir    .../compare \
        --eval-ctx-sweep "1,5,10,20,40" --n-context-draws 5

Outputs: compare_offline_online.csv, compare_offline_online.png, and a printed
summary table.
"""

import os, sys, json, argparse
import numpy as np
import torch as t
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the eval building blocks (build_model, dataset, scorers) verbatim so the
# context-draw logic is byte-for-byte identical between the two models.
sys.path.insert(0, os.path.dirname(__file__))
from eval_np_geometry import (build_model, TrajectoryDataset,  # type: ignore
                              per_geometry_mae, per_geometry_mae_online)


def load_model(ckpt_path, device):
    ck = t.load(ckpt_path, map_location=device)
    name = ck.get("model_name", "cnp")
    conv = ck.get("convention", "split")
    feat = ck.get("feat_dim"); out = ck.get("out_dim", 3)
    cfg = ck.get("config", {})
    model, conv2 = build_model(name, cfg.get("num_hidden", 128), feat, out,
                               rnn_type=cfg.get("rnn_type", "lstm"),
                               rnn_layers=cfg.get("rnn_layers", 1),
                               rnn_dropout=cfg.get("rnn_dropout", 0.0),
                               dropout=cfg.get("dropout", 0.1),
                               max_context=cfg.get("max_context", 128))
    conv = conv or conv2
    model = model.to(device); model.load_state_dict(ck["model"]); model.eval()
    return {"model": model, "conv": conv, "name": name, "cfg": cfg,
            "y_mean": ck.get("y_mean"), "y_std": ck.get("y_std"),
            "chunk_size": int(cfg.get("chunk_size", 8))}


def score_pools(m, pools, device, eval_ctx, n_draws, ctx_sample_mode,
                exclude_ctx, chunk_size):
    """{pool: mean_mae} for one model at one context size."""
    res = {}
    for nm, ds in pools.items():
        if m["conv"] == "online":
            g = per_geometry_mae_online(
                m["model"], ds, device, eval_ctx, n_draws, chunk_size,
                y_mean=m["y_mean"], y_std=m["y_std"],
                ctx_sample_mode=ctx_sample_mode,
                exclude_ctx_from_target=exclude_ctx)
        else:
            g = per_geometry_mae(
                m["model"], m["conv"], ds, device, eval_ctx, n_draws,
                y_mean=m["y_mean"], y_std=m["y_std"],
                ctx_sample_mode=ctx_sample_mode,
                exclude_ctx_from_target=exclude_ctx)
        res[nm] = float(np.mean(list(g.values()))) if g else float("nan")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--offline-ckpt", required=True, help="ranp/rcnp checkpoint")
    ap.add_argument("--online-ckpt", required=True, help="online_ranp checkpoint")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--eval-ctx", type=int, default=20,
                    help="context size for the printed summary table")
    ap.add_argument("--eval-ctx-sweep", default="1,5,10,20,40",
                    help="comma list of context sizes for the comparison curves")
    ap.add_argument("--n-context-draws", type=int, default=5)
    ap.add_argument("--ctx-sample-mode", default="random", choices=["first", "random"],
                    help="how the context fixes are drawn (same for BOTH models).")
    ap.add_argument("--exclude-ctx-from-target", dest="exclude_ctx",
                    default="true", choices=["true", "false"],
                    help="score on the complement of the context (both models).")
    ap.add_argument("--chunk-size", type=int, default=None,
                    help="streaming chunk size for the online model "
                         "(default: read from its checkpoint).")
    ap.add_argument("--device", default="cuda" if t.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = t.device(args.device)
    exclude_ctx = args.exclude_ctx == "true"

    pools = {}
    for nm in ["train", "val", "test"]:
        pth = os.path.join(args.data_dir, f"{nm}_data.pkl")
        if os.path.exists(pth):
            pools[nm] = TrajectoryDataset(pth)

    off = load_model(args.offline_ckpt, device)
    onl = load_model(args.online_ckpt, device)
    chunk_size = args.chunk_size if args.chunk_size is not None else onl["chunk_size"]
    if onl["conv"] != "online":
        print(f"[warn] --online-ckpt is convention '{onl['conv']}', expected "
              f"'online'. Comparison still runs but is not online vs offline.")

    print(f"offline = {off['name']} ({off['conv']})   "
          f"online = {onl['name']} ({onl['conv']}, chunk_size={chunk_size})")
    print(f"scenario: ctx_sample_mode={args.ctx_sample_mode}  "
          f"draws={args.n_context_draws}  exclude_ctx_from_target={exclude_ctx}")

    ctx_sizes = [int(c) for c in args.eval_ctx_sweep.split(",") if c.strip()]
    rows = []  # (ctx, pool, off_mae, on_mae)
    for c in ctx_sizes:
        off_pm = score_pools(off, pools, device, c, args.n_context_draws,
                             args.ctx_sample_mode, exclude_ctx, chunk_size)
        on_pm = score_pools(onl, pools, device, c, args.n_context_draws,
                            args.ctx_sample_mode, exclude_ctx, chunk_size)
        for nm in pools:
            rows.append((c, nm, off_pm[nm], on_pm[nm]))
        print(f"  ctx={c:3d}  " + "  ".join(
            f"{nm}: off={off_pm[nm]:.3f}/on={on_pm[nm]:.3f}" for nm in pools))

    # ---- CSV ---------------------------------------------------------------
    csv_path = os.path.join(args.out_dir, "compare_offline_online.csv")
    with open(csv_path, "w") as f:
        f.write("eval_ctx,pool,offline_mae,online_mae,gap,gap_pct\n")
        for c, nm, o, n in rows:
            gap = n - o
            pct = 100.0 * gap / o if o else float("nan")
            f.write(f"{c},{nm},{o:.5f},{n:.5f},{gap:.5f},{pct:.3f}\n")

    # ---- figure: MAE vs ctx (offline solid / online dashed) + gap% ---------
    colors = {"train": "#2980b9", "val": "#e67e22", "test": "#c0392b"}
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for nm in pools:
        xs = [c for (c, p, *_ ) in rows if p == nm]
        ov = [o for (c, p, o, n) in rows if p == nm]
        nv = [n for (c, p, o, n) in rows if p == nm]
        col = colors.get(nm, None)
        axs[0].plot(xs, ov, "o-", color=col, label=f"{nm} offline")
        axs[0].plot(xs, nv, "s--", color=col, label=f"{nm} online")
        gap_pct = [100.0 * (n - o) / o if o else float("nan")
                   for (o, n) in zip(ov, nv)]
        axs[1].plot(xs, gap_pct, "o-", color=col, label=nm)
    axs[0].set_xlabel("context size (# fixes)"); axs[0].set_ylabel("MAE (physical units)")
    axs[0].set_title("Offline (solid) vs Online (dashed)")
    axs[0].legend(fontsize=8); axs[0].grid(alpha=0.3)
    axs[1].axhline(0, color="k", lw=0.8, alpha=0.5)
    axs[1].set_xlabel("context size (# fixes)"); axs[1].set_ylabel("online - offline  (% of offline MAE)")
    axs[1].set_title("Cost of going online")
    axs[1].legend(); axs[1].grid(alpha=0.3)
    fig.tight_layout()
    png_path = os.path.join(args.out_dir, "compare_offline_online.png")
    fig.savefig(png_path, dpi=200)

    # ---- summary table at args.eval_ctx ------------------------------------
    print("\n" + "=" * 60)
    print(f"SUMMARY @ ctx={args.eval_ctx}  (offline {off['name']} vs online {onl['name']})")
    print("=" * 60)
    off_pm = score_pools(off, pools, device, args.eval_ctx, args.n_context_draws,
                         args.ctx_sample_mode, exclude_ctx, chunk_size)
    on_pm = score_pools(onl, pools, device, args.eval_ctx, args.n_context_draws,
                        args.ctx_sample_mode, exclude_ctx, chunk_size)
    print(f"{'pool':6s} {'offline':>10s} {'online':>10s} {'gap':>10s} {'gap %':>8s}")
    for nm in pools:
        o, n = off_pm[nm], on_pm[nm]
        gap = n - o; pct = 100.0 * gap / o if o else float("nan")
        print(f"{nm:6s} {o:10.4f} {n:10.4f} {gap:+10.4f} {pct:+7.1f}%")
    print(f"\nwrote {csv_path}\nwrote {png_path}")


if __name__ == "__main__":
    main()
