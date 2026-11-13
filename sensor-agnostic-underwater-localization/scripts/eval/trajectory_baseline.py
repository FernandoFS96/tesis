"""
trajectory_baseline.py
======================

The best the trajectory alone can do: a network that never hears the acoustics.
It maps the first --nctx (default 10) positions of a transit to its later positions, trained on
the TRAINING-split trajectories of the benchmark (raw trajectory files, so the
large feature pickles are not loaded), in coordinates relative to the last context
fix, with random rotations and reflections (the trajectory generator is
isotropic). Scored on the protocol of eval_suite.py.

    python scripts/eval/trajectory_baseline.py \
        --data-dir data/pfn10kd/hermite_distinct/processed/geometry_split \
        --out-dir output/eval_paper/baselines/b1
"""
import argparse, glob, json, os, re, sys, csv
from collections import defaultdict
import numpy as np
import torch as t
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_suite as E

SCALE = 100.0


def train_paths(data_dir):
    root = os.path.dirname(os.path.dirname(os.path.normpath(data_dir)))
    train = set(json.load(open(os.path.join(data_dir, "splits.json")))["train"])
    P = []
    for d in glob.glob(os.path.join(root, "position_set_*")):
        if int(re.search(r"position_set_(\d+)", d).group(1)) not in train:
            continue
        f = sorted(glob.glob(os.path.join(d, "channel_option_*", "random", "trajectory", "trajectories.npy")))[0]
        P.append(np.load(f)[:2].transpose(1, 2, 0))           # (n_traj, ppt, 2); paths repeat across theta
    return np.concatenate(P).astype(np.float32)


def rel(P, n):
    """(N, ppt, 2) -> first n positions and the rest, relative to the last context fix, scaled."""
    o = P[:, n - 1:n]
    return (P[:, :n] - o) / SCALE, (P[:, n:] - o) / SCALE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--eval-seeds", default="0,1,2")
    ap.add_argument("--cap-traj", type=int, default=6)
    ap.add_argument("--nctx", type=int, default=10)
    args = ap.parse_args()
    NCTX = args.nctx
    dev = t.device("cuda" if t.cuda.is_available() else "cpu")

    P = t.tensor(train_paths(args.data_dir), device=dev)
    ppt = P.shape[1]
    print(f"{len(P)} training trajectories of {ppt} points", flush=True)
    t.manual_seed(0)
    net = nn.Sequential(nn.Linear(2 * NCTX, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(),
                        nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 2 * (ppt - NCTX))).to(dev)
    opt = t.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sch = t.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=args.steps)
    for i in range(args.steps):
        b = P[t.randint(len(P), (1024,), device=dev)]
        a = t.rand(len(b), device=dev) * 2 * np.pi
        R = t.stack([t.stack([a.cos(), -a.sin()], -1), t.stack([a.sin(), a.cos()], -1)], -2)
        R[:, :, 0] *= t.where(t.rand(len(b), 1, device=dev) < 0.5, -1.0, 1.0)   # reflections
        c, f = rel(b @ R.transpose(1, 2), NCTX)
        loss = (net(c.flatten(1)).view_as(f) - f).norm(dim=-1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()

    data = E.load_data(args.data_dir, t.device("cpu"), load_cap=max(15, args.cap_traj * 3))
    rows = []
    with t.no_grad():
        for es in (int(z) for z in args.eval_seeds.split(",")):
            items = E.subsample_test(data["by_gid"], args.cap_traj, es)
            Y = t.stack([y[0] for _, y, _, _ in items]).to(dev)                 # (N, ppt, D)
            c, _ = rel(Y[..., :2], NCTX)
            pred = net(c.flatten(1)).view(len(Y), -1, 2) * SCALE + Y[:, NCTX - 1:NCTX, :2]
            pred = t.cat([pred, Y[:, NCTX:, 2:]], -1)                          # z as given (constant)
            ept = (pred - Y[:, NCTX:]).norm(dim=-1).cpu().numpy()             # (N, ppt - NCTX)
            e = ept.mean(1)
            rows.append(("traj_net", "overall", es, float(e.mean())))
            # drift blocks of ten target points, as eval_suite's ev_drift
            rows += [("traj_net", f"drift_{k}", es, float(ept[:, 10 * k:10 * k + 10].mean()))
                     for k in range(ept.shape[1] // 10)]
            buck = defaultdict(list)
            for (_, _, g, _), v in zip(items, e):
                buck[data["labels"].get(g, "?")].append(v)
            rows += [("traj_net", r, es, float(np.mean(v))) for r, v in buck.items() if r != "?"]

    os.makedirs(args.out_dir, exist_ok=True)
    f = os.path.join(args.out_dir, "results_traj_net.csv" if NCTX == 10 else f"results_traj_net_c{NCTX}.csv")
    agg = defaultdict(list)
    for n, m, es, v in rows:
        agg[m].append(v)
    with open(f, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["model", "metric", "mean", "std", "n"])
        for m, v in agg.items():
            w.writerow(["traj_net", m, f"{np.mean(v):.4f}", f"{np.std(v):.4f}", len(v)])
    print("traj_net " + "  ".join(f"{m} {np.mean(v):.2f}" for m, v in sorted(agg.items())))


if __name__ == "__main__":
    main()
