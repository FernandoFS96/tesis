"""
train_cnp_geometry.py
=====================

Train a CNP (the DeterministicModel from src/models/anp.py) on the geometry-split
data, for the sensor-displacement robustness study.

WHAT THIS MEASURES
------------------
This is the *decisive* version of the geometry-separability question. Instead of
proxy statistics (T1/T2/T3) we train a real model on the TRAINING geometries only
and later (see eval_cnp_geometry.py) measure how much its localization error
degrades on the held-out VAL/TEST geometries. The size of that degradation is the
gap the spatial-encoder model will try to close.

This is a BASELINE: the plain CNP has NO spatial encoding -- it sees only the
raw acoustic features and never the sensor positions. It therefore measures how
much out-of-position degradation exists with the current architecture, i.e. the
problem we are trying to solve.

DATA CONTRACT (produced by data_process_random_positions.py, mode=geometry)
---------------------------------------------------------------------------
<processed>/geometry_split/
    train_data.pkl / val_data.pkl / test_data.pkl
        list of dicts: {"X":(ppt,1010), "y":(ppt,3),
                        "sensor_pos":(10,3), "geometry_id":int, "theta":float}
    metadata.pkl : {tau, n_sensors, feat_dim, *_geometry_ids, *_thetas, split}

CONTEXT/TARGET SPLIT
--------------------
Each trajectory has ppt=50 points. For every training step we randomly choose a
context subset of the 50 points (size in [ctx_min, ctx_max]); the model predicts
ALL 50 points as targets (standard ANP training). At eval we fix the context
fraction for comparability.

USAGE
-----
    python train_cnp_geometry.py \
        --data-dir ../../data/data_random_positions/processed/geometry_split \
        --out-dir  ../runs/cnp_baseline \
        --epochs 1000 --batch-size 16 --num-hidden 128 --lr 5e-4

Outputs to --out-dir:
    best.pt          (lowest val loss)  + last.pt
    train_log.csv    per-epoch train/val loss, val MAE
    config.json      resolved hyperparameters + data fingerprint
"""

import os, sys, json, pickle, argparse, time
import numpy as np
import torch as t
from torch.utils.data import Dataset, DataLoader

# Add the repo root (parent of src/) so that `from src.models.anp import ...` works.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.models.anp import DeterministicModel  # type: ignore


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class TrajectoryDataset(Dataset):
    """One item = one full trajectory (50 points). Context/target masking is done
    in the collate fn so it can be randomized per batch/epoch."""
    def __init__(self, pkl_path):
        with open(pkl_path, "rb") as f:
            self.samples = pickle.load(f)
        # basic shape introspection
        x0 = np.asarray(self.samples[0]["X"])
        y0 = np.asarray(self.samples[0]["y"])
        self.ppt, self.feat_dim = x0.shape
        self.out_dim = y0.shape[-1]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        X = t.as_tensor(np.asarray(s["X"], dtype=np.float32))   # (ppt, 1010)
        y = t.as_tensor(np.asarray(s["y"], dtype=np.float32))   # (ppt, 3)
        gid = int(s.get("geometry_id", -1))
        th = float(s.get("theta", -1.0))
        return X, y, gid, th


def make_collate(ctx_min, ctx_max, fixed_ctx=None, seed=None):
    """Returns a collate_fn that builds (context_x, context_y, target_x, target_y).
    Targets are ALL points; context is a random subset (or a fixed count)."""
    rng = np.random.default_rng(seed)

    def collate(batch):
        Xs = t.stack([b[0] for b in batch], dim=0)   # (B, ppt, feat)
        ys = t.stack([b[1] for b in batch], dim=0)   # (B, ppt, 3)
        gids = t.tensor([b[2] for b in batch])
        ths = t.tensor([b[3] for b in batch])
        B, ppt, _ = Xs.shape
        if fixed_ctx is not None:
            n_ctx = int(fixed_ctx)
        else:
            n_ctx = int(rng.integers(ctx_min, ctx_max + 1))
        n_ctx = max(1, min(n_ctx, ppt))
        # same context indices across the batch (simple, standard); could be per-item
        idx = rng.permutation(ppt)[:n_ctx]
        idx = t.as_tensor(np.sort(idx))
        context_x = Xs[:, idx, :]
        context_y = ys[:, idx, :]
        target_x = Xs               # predict all points
        target_y = ys
        return context_x, context_y, target_x, target_y, gids, ths
    return collate


# --------------------------------------------------------------------------- #
# Train / eval loops
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    tot_loss, tot_nll, tot_mae, n = 0.0, 0.0, 0.0, 0
    for context_x, context_y, target_x, target_y, _, _ in loader:
        context_x = context_x.to(device); context_y = context_y.to(device)
        target_x = target_x.to(device); target_y = target_y.to(device)
        with t.set_grad_enabled(train):
            mean, var, loss, _, nll = model(context_x, context_y, target_x, target_y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                t.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        b = target_y.size(0)
        tot_loss += loss.item() * b
        tot_nll += nll.item() * b
        # MAE in target units (Euclidean per-point distance, then mean)
        with t.no_grad():
            dist = t.sqrt(((mean - target_y) ** 2).sum(-1) + 1e-12)  # (B, ppt)
            tot_mae += dist.mean().item() * b
        n += b
    return tot_loss / n, tot_nll / n, tot_mae / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True,
                    help="…/processed/geometry_split")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ctx-min", type=int, default=5)
    ap.add_argument("--ctx-max", type=int, default=40)
    ap.add_argument("--val-ctx", type=int, default=20,
                    help="fixed context size for val (comparability)")
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if t.cuda.is_available() else "cpu")
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    t.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    train_ds = TrajectoryDataset(os.path.join(args.data_dir, "train_data.pkl"))
    val_ds   = TrajectoryDataset(os.path.join(args.data_dir, "val_data.pkl"))
    feat_dim, out_dim, ppt = train_ds.feat_dim, train_ds.out_dim, train_ds.ppt
    print(f"feat_dim={feat_dim} out_dim={out_dim} ppt={ppt} "
          f"| train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
        collate_fn=make_collate(args.ctx_min, args.ctx_max, seed=args.seed))
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_collate(args.ctx_min, args.ctx_max,
                                fixed_ctx=args.val_ctx, seed=args.seed + 1))

    device = t.device(args.device)
    model = DeterministicModel(num_hidden=args.num_hidden,
                               input_dim=feat_dim, output_dim=out_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"CNP params: {n_params/1e6:.2f}M  device={device}")

    opt = t.optim.AdamW(model.parameters(), lr=args.lr,
                        weight_decay=args.weight_decay)
    sched = t.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump({**vars(args), "feat_dim": feat_dim, "out_dim": out_dim,
                   "ppt": ppt, "n_params": n_params}, f, indent=2)

    log_path = os.path.join(args.out_dir, "train_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,train_nll,train_mae,val_loss,val_nll,val_mae,lr,sec\n")

    best_val = float("inf")
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_nll, tr_mae = run_epoch(model, train_loader, device, opt)
        va_loss, va_nll, va_mae = run_epoch(model, val_loader, device, None)
        sched.step()
        dt = time.time() - t0
        lr_now = opt.param_groups[0]["lr"]
        with open(log_path, "a") as f:
            f.write(f"{ep},{tr_loss:.6f},{tr_nll:.6f},{tr_mae:.6f},"
                    f"{va_loss:.6f},{va_nll:.6f},{va_mae:.6f},{lr_now:.2e},{dt:.1f}\n")
        if ep % 5 == 0 or ep == 1:
            print(f"ep {ep:3d} | train loss {tr_loss:.4f} mae {tr_mae:.3f} "
                  f"| val loss {va_loss:.4f} mae {va_mae:.3f} | {dt:.1f}s")
        if va_loss < best_val:
            best_val = va_loss
            t.save({"model": model.state_dict(), "config": vars(args),
                    "feat_dim": feat_dim, "out_dim": out_dim, "epoch": ep},
                   os.path.join(args.out_dir, "best.pt"))
    t.save({"model": model.state_dict(), "config": vars(args),
            "feat_dim": feat_dim, "out_dim": out_dim, "epoch": args.epochs},
           os.path.join(args.out_dir, "last.pt"))
    print(f"done. best val loss={best_val:.4f}  -> {args.out_dir}/best.pt")


if __name__ == "__main__":
    main()
