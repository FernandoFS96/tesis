"""
train_np_geometry.py
====================

Unified trainer for the three neural-process baselines on the geometry-split data, selected with --model:

    cnp   -> anp.DeterministicModel      (attentive CNP, no latent, no RNN)
    anp   -> anp.LatentModel             (attentive latent NP)
    ranp  -> r_anp.LatentModel           (recurrent attentive latent NP; LSTM)

All three are trained as BASELINES with NO spatial encoding: 
the model sees only the raw acoustic features, never sensor_pos. 
They measure how much out-of-position degradation exists per architecture - the gap the spatial encoder will close.

WHY TWO CALLING CONVENTIONS (the one thing that matters here)
-------------------------------------------------------------
* cnp / anp  (anp.py): forward(context_x, context_y, target_x, target_y, beta) 
                        -- the caller pre-splits context and target into separate tensors.

* ranp (r_anp.py): forward(x_seq, context_indices, context_y, target_indices, target_y, beta) 
                        -- the caller passes the FULL sequence (B, T, Dx) plus integer index tensors; the model runs its internal LSTM over the whole sequence and splits by index.

The collate fn below produces BOTH forms; `model_forward` dispatches on the model family. 
All three return the same 5-tuple (mean, var, loss, kl, nll), so the loss/optimization code is shared. 
For ANP/RANP (latent models) target_y is passed during training so the posterior + KL term are active; beta controls the KL weight.

DATA CONTRACT (data_process_random_positions.py, mode=geometry)
---------------------------------------------------------------
<processed>/geometry_split/{train,val,test}_data.pkl  -> list of dicts:
    {"X":(ppt,1010), "y":(ppt,3), "sensor_pos":(10,3), "geometry_id":int, "theta":float}

USAGE
-----
    python train_np_geometry.py --model cnp \
        --data-dir ../../data/data_random_positions/processed/geometry_split \
        --out-dir  ../runs/cnp_baseline --epochs 200

    python train_np_geometry.py --model anp \
        --data-dir ../../data/data_random_positions/processed/geometry_split \
        --out-dir  ../runs/anp_baseline --epochs 500

    python train_np_geometry.py --model ranp \
        --data-dir ../../data/data_random_positions/processed/geometry_split \
        --out-dir ../runs/ranp_baseline --epochs 500

Outputs to --out-dir: best.pt, last.pt, train_log.csv, config.json
"""

import os, sys, json, pickle, argparse, time
import numpy as np
import torch as t
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

# --- model imports: support both ../src/models and a local copy -------------
# Add the repo root (parent of src/) so that `from src.models.anp import ...` works.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import src.models.anp as anp_mod  # type: ignore
import src.models.r_anp as ranp_mod  # type: ignore

# --------------------------------------------------------------------------- #
# Model factory
# --------------------------------------------------------------------------- #
def build_model(name, num_hidden, input_dim, output_dim,
                rnn_type="lstm", rnn_layers=1, rnn_dropout=0.0):
    name = name.lower()
    if name == "cnp":
        return anp_mod.DeterministicModel(num_hidden, input_dim, output_dim), "split"
    if name == "anp":
        return anp_mod.LatentModel(num_hidden, input_dim, output_dim), "split"
    if name == "ranp":
        return (ranp_mod.LatentModel(num_hidden, input_dim, output_dim,
                                     rnn_type=rnn_type, rnn_layers=rnn_layers,
                                     rnn_dropout=rnn_dropout),
                "indexed")
    if name == "rcnp":  # bonus: recurrent CNP, same indexed convention
        return (ranp_mod.DeterministicModel(num_hidden, input_dim, output_dim,
                                            rnn_type=rnn_type, rnn_layers=rnn_layers,
                                            rnn_dropout=rnn_dropout),
                "indexed")
    raise ValueError(f"unknown model '{name}' (use cnp|anp|ranp|rcnp)")


def is_latent(name):
    return name.lower() in ("anp", "ranp")  # models with a posterior + KL


# --------------------------------------------------------------------------- #
# Dataset (one item = one trajectory) + collate producing BOTH conventions
# --------------------------------------------------------------------------- #
class TrajectoryDataset(Dataset):
    def __init__(self, pkl_path):
        with open(pkl_path, "rb") as f:
            self.samples = pickle.load(f)
        x0 = np.asarray(self.samples[0]["X"]); y0 = np.asarray(self.samples[0]["y"])
        self.ppt, self.feat_dim = x0.shape
        self.out_dim = y0.shape[-1]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        X = t.as_tensor(np.asarray(s["X"], dtype=np.float32))
        y = t.as_tensor(np.asarray(s["y"], dtype=np.float32))
        return X, y, int(s.get("geometry_id", -1)), float(s.get("theta", -1.0))


def make_collate(ctx_min, ctx_max, fixed_ctx=None, seed=None):
    """Builds a batch carrying BOTH calling conventions:
       - context_x/context_y/target_x/target_y  (for cnp/anp)
       - x_seq + context_indices/target_indices (for ranp)
       Targets are ALL points; context is a random subset (or fixed count)."""
    rng = np.random.default_rng(seed)

    def collate(batch):
        Xs = t.stack([b[0] for b in batch], dim=0)   # (B, ppt, feat)
        ys = t.stack([b[1] for b in batch], dim=0)   # (B, ppt, 3)
        gids = t.tensor([b[2] for b in batch])
        ths = t.tensor([b[3] for b in batch])
        B, ppt, _ = Xs.shape
        n_ctx = int(fixed_ctx) if fixed_ctx is not None else int(rng.integers(ctx_min, ctx_max + 1))
        n_ctx = max(1, min(n_ctx, ppt))
        ctx_idx = t.as_tensor(np.sort(rng.permutation(ppt)[:n_ctx]), dtype=t.long)
        tgt_idx = t.arange(ppt, dtype=t.long)   # predict all points
        return {
            "x_seq": Xs, "y_full": ys,
            "context_indices": ctx_idx, "target_indices": tgt_idx,
            "context_x": Xs[:, ctx_idx, :], "context_y": ys[:, ctx_idx, :],
            "target_x": Xs, "target_y": ys,
            "gids": gids, "thetas": ths,
        }
    return collate


# --------------------------------------------------------------------------- #
# Forward dispatch (the family-dependent call)
# --------------------------------------------------------------------------- #
def model_forward(model, conv, batch, device, beta, with_target_y=True):
    cy = batch["context_y"].to(device)
    ty = batch["target_y"].to(device) if with_target_y else None
    if conv == "split":
        cx = batch["context_x"].to(device)
        tx = batch["target_x"].to(device)
        return model(cx, cy, tx, ty, beta)
    else:  # indexed (ranp)
        x_seq = batch["x_seq"].to(device)
        ci = batch["context_indices"].to(device)
        ti = batch["target_indices"].to(device)
        return model(x_seq, ci, cy, ti, ty, beta)


# --------------------------------------------------------------------------- #
# Epoch loop
# --------------------------------------------------------------------------- #
def run_epoch(model, conv, loader, device, beta, optimizer=None, desc=""):
    train = optimizer is not None
    model.train(train)
    tot_loss = tot_nll = tot_mae = 0.0; n = 0
    bar = tqdm(loader, desc=desc, leave=False)
    for batch in bar:
        ty = batch["target_y"].to(device)
        with t.set_grad_enabled(train):
            mean, var, loss, kl, nll = model_forward(model, conv, batch, device,
                                                     beta, with_target_y=True)
            if train:
                optimizer.zero_grad()
                loss.backward()
                t.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        b = ty.size(0)
        tot_loss += loss.item() * b
        tot_nll += nll.item() * b
        with t.no_grad():
            dist = t.sqrt(((mean - ty) ** 2).sum(-1) + 1e-12)
            mae = dist.mean().item()
            tot_mae += mae * b
        n += b
        bar.set_postfix(loss=f"{loss.item():.3f}", mae=f"{mae:.2f}")
    return tot_loss / n, tot_nll / n, tot_mae / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["cnp", "anp", "ranp", "rcnp"])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ctx-min", type=int, default=5)
    ap.add_argument("--ctx-max", type=int, default=40)
    ap.add_argument("--val-ctx", type=int, default=20)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=1.0, help="KL weight (anp/ranp)")
    ap.add_argument("--rnn-type", default="lstm", choices=["lstm", "gru"])
    ap.add_argument("--rnn-layers", type=int, default=1)
    ap.add_argument("--rnn-dropout", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if t.cuda.is_available() else "cpu")
    ap.add_argument("--num-workers", type=int, default=1)
    args = ap.parse_args()

    t.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    train_ds = TrajectoryDataset(os.path.join(args.data_dir, "train_data.pkl"))
    val_ds = TrajectoryDataset(os.path.join(args.data_dir, "val_data.pkl"))
    feat_dim, out_dim, ppt = train_ds.feat_dim, train_ds.out_dim, train_ds.ppt
    print(f"[{args.model}] feat_dim={feat_dim} out_dim={out_dim} ppt={ppt} "
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
    model, conv = build_model(args.model, args.num_hidden, feat_dim, out_dim,
                              rnn_type=args.rnn_type, rnn_layers=args.rnn_layers,
                              rnn_dropout=args.rnn_dropout)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.model}] convention={conv}  params={n_params/1e6:.2f}M  device={device}")

    opt = t.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = t.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump({**vars(args), "feat_dim": feat_dim, "out_dim": out_dim,
                   "ppt": ppt, "n_params": n_params, "convention": conv}, f, indent=2)

    log_path = os.path.join(args.out_dir, "train_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,train_nll,train_mae,val_loss,val_nll,val_mae,lr,sec\n")

    # latent models use beta; deterministic ignore it (no KL)
    beta = args.beta if is_latent(args.model) else 0.0

    best_val = float("inf")
    epoch_bar = tqdm(range(1, args.epochs + 1), desc=f"{args.model} epochs")
    for ep in epoch_bar:
        t0 = time.time()
        tr = run_epoch(model, conv, train_loader, device, beta, opt,
                       desc=f"ep{ep} train")
        va = run_epoch(model, conv, val_loader, device, beta, None,
                       desc=f"ep{ep} val")
        sched.step()
        dt = time.time() - t0
        lr_now = opt.param_groups[0]["lr"]
        with open(log_path, "a") as f:
            f.write(f"{ep},{tr[0]:.6f},{tr[1]:.6f},{tr[2]:.6f},"
                    f"{va[0]:.6f},{va[1]:.6f},{va[2]:.6f},{lr_now:.2e},{dt:.1f}\n")
        epoch_bar.set_postfix(tr_mae=f"{tr[2]:.2f}", va_mae=f"{va[2]:.2f}",
                              va_loss=f"{va[0]:.3f}")
        if va[0] < best_val:
            best_val = va[0]
            t.save({"model": model.state_dict(), "config": vars(args),
                    "model_name": args.model, "convention": conv,
                    "feat_dim": feat_dim, "out_dim": out_dim, "epoch": ep},
                   os.path.join(args.out_dir, "best.pt"))
    t.save({"model": model.state_dict(), "config": vars(args),
            "model_name": args.model, "convention": conv,
            "feat_dim": feat_dim, "out_dim": out_dim, "epoch": args.epochs},
           os.path.join(args.out_dir, "last.pt"))
    print(f"[{args.model}] done. best val loss={best_val:.4f} -> {args.out_dir}/best.pt")


if __name__ == "__main__":
    main()
