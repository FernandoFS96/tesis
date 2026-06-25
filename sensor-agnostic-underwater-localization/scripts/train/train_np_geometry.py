"""
train_np_geometry.py
====================

Unified trainer for the three neural-process baselines on the geometry-split
data, selected with the Hydra ``model`` group:

    cnp   -> anp.DeterministicModel      (attentive CNP, no latent, no RNN)
    anp   -> anp.LatentModel             (attentive latent NP)
    ranp  -> r_anp.LatentModel           (recurrent attentive latent NP; LSTM)
    rcnp  -> r_anp.DeterministicModel    (bonus: recurrent CNP, indexed conv.)

All baselines are trained with NO spatial encoding: the model sees only the raw
acoustic features, never sensor_pos. They measure how much out-of-position
degradation exists per architecture - the gap the spatial encoder will close.

WHY TWO CALLING CONVENTIONS (the one thing that matters here)
-------------------------------------------------------------
* cnp / anp  (anp.py): forward(context_x, context_y, target_x, target_y, beta)
                        -- the caller pre-splits context and target into
                        separate tensors.

* ranp (r_anp.py): forward(x_seq, context_indices, context_y, target_indices,
                        target_y, beta) -- the caller passes the FULL sequence
                        (B, T, Dx) plus integer index tensors; the model runs
                        its internal LSTM over the whole sequence and splits by
                        index.

The collate fn below produces BOTH forms; `model_forward` dispatches on the
model family. All four return the same 5-tuple (mean, var, loss, kl, nll), so
the loss/optimization code is shared. For ANP/RANP (latent models) target_y is
passed during training so the posterior + KL term are active; beta controls the
KL weight.

DATA CONTRACT (data_process_random_positions.py, mode=geometry)
---------------------------------------------------------------
<processed>/geometry_split/{train,val,test}_data.pkl  -> list of dicts:
    {"X":(ppt,1010), "y":(ppt,3), "sensor_pos":(10,3), "geometry_id":int,
     "theta":float}

CONFIGURATION (Hydra)
---------------------
All hyperparameters live in ``config/`` and are composed by Hydra. Override
anything from the command line, e.g.:

    python train_np_geometry.py model=cnp
    python train_np_geometry.py model=anp data.normalize_y=true \
        data.ctx_sample_mode=first training.epochs=500
    python train_np_geometry.py model=ranp wandb.enabled=false

Outputs (to the Hydra run dir, or ``out_dir`` if set):
    best.pt, last.pt, train_log.csv, config.yaml
"""

import os
import sys
import pickle
import time

import numpy as np
import torch as t
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import wandb

# --- model imports: support both ../src/models and a local copy -------------
# Add the repo root (parent of src/) so that `from src.models.anp import ...`
# works regardless of the (Hydra-changed) current working directory.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)
import src.models.anp as anp_mod  # type: ignore  # noqa: E402
import src.models.r_anp as ranp_mod  # type: ignore  # noqa: E402


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
# Y normalization helpers
# --------------------------------------------------------------------------- #
def compute_y_stats(dataset):
    """Per-dimension mean and std of y across the whole training set."""
    ys = np.concatenate(
        [np.asarray(s["y"], dtype=np.float32) for s in dataset.samples], axis=0
    )
    return (t.tensor(ys.mean(axis=0), dtype=t.float32),
            t.tensor(ys.std(axis=0) + 1e-6, dtype=t.float32))


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


def make_collate(ctx_min, ctx_max, fixed_ctx=None, seed=None, ctx_sample_mode="random"):
    """Builds a batch carrying BOTH calling conventions:
       - context_x/context_y/target_x/target_y  (for cnp/anp)
       - x_seq + context_indices/target_indices (for ranp)
       Targets are ALL points; context is a subset determined by ctx_sample_mode.
       ctx_sample_mode='first'  -> take the first n_ctx points (ordered prefix)
       ctx_sample_mode='random' -> random permutation (original behaviour)"""
    rng = np.random.default_rng(seed)

    def collate(batch):
        Xs = t.stack([b[0] for b in batch], dim=0)   # (B, ppt, feat)
        ys = t.stack([b[1] for b in batch], dim=0)   # (B, ppt, 3)
        gids = t.tensor([b[2] for b in batch])
        ths = t.tensor([b[3] for b in batch])
        B, ppt, _ = Xs.shape
        n_ctx = int(fixed_ctx) if fixed_ctx is not None else int(rng.integers(ctx_min, ctx_max + 1))
        n_ctx = max(1, min(n_ctx, ppt))
        if ctx_sample_mode == "first":
            ctx_idx = t.arange(n_ctx, dtype=t.long)
        else:
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
def run_epoch(model, conv, loader, device, beta, optimizer=None, desc="",
              y_mean=None, y_std=None):
    """y_mean / y_std: if not None, y is normalized before the forward pass and
    denormalized before MAE computation so the metric stays in physical units."""
    train = optimizer is not None
    model.train(train)
    tot_loss = tot_nll = tot_mae = 0.0; n = 0
    ym = y_mean.to(device) if y_mean is not None else None
    ys = y_std.to(device)  if y_std  is not None else None
    bar = tqdm(loader, desc=desc, leave=False)
    for batch in bar:
        ty_raw = batch["target_y"].to(device)   # physical units, for MAE
        if ym is not None:
            batch = {**batch,
                     "context_y": (batch["context_y"].to(device) - ym) / ys,
                     "target_y":  (batch["target_y"].to(device)  - ym) / ys}
        with t.set_grad_enabled(train):
            mean, var, loss, kl, nll = model_forward(model, conv, batch, device,
                                                     beta, with_target_y=True)
            if train:
                optimizer.zero_grad()
                loss.backward()
                t.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        b = ty_raw.size(0)
        tot_loss += loss.item() * b
        tot_nll += nll.item() * b
        with t.no_grad():
            mean_phys = mean * ys + ym if ym is not None else mean
            dist = t.sqrt(((mean_phys - ty_raw) ** 2).sum(-1) + 1e-12)
            mae = dist.mean().item()
            tot_mae += mae * b
        n += b
        bar.set_postfix(loss=f"{loss.item():.3f}", mae=f"{mae:.2f}")
    return tot_loss / n, tot_nll / n, tot_mae / n


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def resolve_device(device_cfg):
    if device_cfg in (None, "auto"):
        return "cuda" if t.cuda.is_available() else "cpu"
    return device_cfg


def flat_ckpt_config(cfg, device):
    """A flat dict stored in the checkpoint so eval_np_geometry.py can read
    num_hidden / rnn_* / ctx_sample_mode etc. via simple ``.get`` calls."""
    m = cfg.model
    return {
        "model": m.name,
        "num_hidden": m.num_hidden,
        "rnn_type": m.get("rnn_type", "lstm"),
        "rnn_layers": m.get("rnn_layers", 1),
        "rnn_dropout": m.get("rnn_dropout", 0.0),
        "data_dir": cfg.data.data_dir,
        "normalize_y": cfg.data.normalize_y,
        "ctx_min": cfg.data.ctx_min,
        "ctx_max": cfg.data.ctx_max,
        "val_ctx": cfg.data.val_ctx,
        "ctx_sample_mode": cfg.data.ctx_sample_mode,
        "epochs": cfg.training.epochs,
        "batch_size": cfg.training.batch_size,
        "lr": cfg.training.lr,
        "weight_decay": cfg.training.weight_decay,
        "beta": cfg.training.beta,
        "seed": cfg.seed,
        "device": device,
        "exp_name": cfg.exp_name,
    }


# --------------------------------------------------------------------------- #
# Main (Hydra entry point)
# --------------------------------------------------------------------------- #
@hydra.main(version_base=None, config_path="../../config", config_name="train")
def main(cfg: DictConfig):
    model_name = cfg.model.name
    seed = cfg.seed
    t.manual_seed(seed); np.random.seed(seed)

    # Output directory: explicit out_dir, otherwise Hydra's per-run dir.
    out_dir = cfg.out_dir or HydraConfig.get().runtime.output_dir
    os.makedirs(out_dir, exist_ok=True)

    device = t.device(resolve_device(cfg.device))

    # ---- wandb -------------------------------------------------------------
    wb = cfg.wandb
    use_wandb = bool(wb.enabled)
    if use_wandb:
        wandb.init(
            project=wb.project_name,
            entity=wb.entity,
            group=wb.group_name,
            name=wb.run_name,
            tags=list(wb.tags) if wb.tags else None,
            mode="offline" if wb.offline else "online",
            job_type="dev" if wb.dev else None,
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=out_dir,
        )

    # ---- data --------------------------------------------------------------
    data_dir = cfg.data.data_dir
    train_ds = TrajectoryDataset(os.path.join(data_dir, "train_data.pkl"))
    val_ds = TrajectoryDataset(os.path.join(data_dir, "val_data.pkl"))
    feat_dim, out_dim, ppt = train_ds.feat_dim, train_ds.out_dim, train_ds.ppt
    print(f"[{model_name}] feat_dim={feat_dim} out_dim={out_dim} ppt={ppt} "
          f"| train={len(train_ds)} val={len(val_ds)}")

    y_mean = y_std = None
    if cfg.data.normalize_y:
        y_mean, y_std = compute_y_stats(train_ds)
        print(f"[{model_name}] y_mean={y_mean.numpy()}  y_std={y_std.numpy()}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True,
        num_workers=cfg.training.num_workers, drop_last=True,
        collate_fn=make_collate(cfg.data.ctx_min, cfg.data.ctx_max, seed=seed,
                                ctx_sample_mode=cfg.data.ctx_sample_mode))
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=cfg.training.num_workers,
        collate_fn=make_collate(cfg.data.ctx_min, cfg.data.ctx_max,
                                fixed_ctx=cfg.data.val_ctx, seed=seed + 1,
                                ctx_sample_mode=cfg.data.ctx_sample_mode))

    # ---- model -------------------------------------------------------------
    model, conv = build_model(
        model_name, cfg.model.num_hidden, feat_dim, out_dim,
        rnn_type=cfg.model.get("rnn_type", "lstm"),
        rnn_layers=cfg.model.get("rnn_layers", 1),
        rnn_dropout=cfg.model.get("rnn_dropout", 0.0))
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{model_name}] convention={conv}  params={n_params/1e6:.2f}M  device={device}")

    opt = t.optim.AdamW(model.parameters(), lr=cfg.training.lr,
                        weight_decay=cfg.training.weight_decay)
    sched = t.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.training.epochs)

    # Resolved config snapshot for reproducibility.
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        OmegaConf.save(config=cfg, f=f)

    log_path = os.path.join(out_dir, "train_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,train_nll,train_mae,val_loss,val_nll,val_mae,lr,sec\n")

    # latent models use beta; deterministic ignore it (no KL)
    beta = cfg.training.beta if is_latent(model_name) else 0.0

    ckpt_cfg = flat_ckpt_config(cfg, str(device))

    best_val = float("inf")
    epoch_bar = tqdm(range(1, cfg.training.epochs + 1), desc=f"{model_name} epochs")
    for ep in epoch_bar:
        t0 = time.time()
        tr = run_epoch(model, conv, train_loader, device, beta, opt,
                       desc=f"ep{ep} train", y_mean=y_mean, y_std=y_std)
        va = run_epoch(model, conv, val_loader, device, beta, None,
                       desc=f"ep{ep} val", y_mean=y_mean, y_std=y_std)
        sched.step()
        dt = time.time() - t0
        lr_now = opt.param_groups[0]["lr"]
        with open(log_path, "a") as f:
            f.write(f"{ep},{tr[0]:.6f},{tr[1]:.6f},{tr[2]:.6f},"
                    f"{va[0]:.6f},{va[1]:.6f},{va[2]:.6f},{lr_now:.2e},{dt:.1f}\n")
        epoch_bar.set_postfix(tr_mae=f"{tr[2]:.2f}", va_mae=f"{va[2]:.2f}",
                              va_loss=f"{va[0]:.3f}")

        if use_wandb:
            wandb.log({
                "epoch": ep,
                "train/loss": tr[0], "train/nll": tr[1], "train/mae": tr[2],
                "val/loss": va[0], "val/nll": va[1], "val/mae": va[2],
                "lr": lr_now, "epoch_time_sec": dt,
            }, step=ep)

        if va[0] < best_val:
            best_val = va[0]
            t.save({"model": model.state_dict(), "config": ckpt_cfg,
                    "model_name": model_name, "convention": conv,
                    "feat_dim": feat_dim, "out_dim": out_dim, "epoch": ep,
                    "y_mean": y_mean, "y_std": y_std},
                   os.path.join(out_dir, "best.pt"))
            if use_wandb:
                wandb.run.summary["best_val_loss"] = best_val
                wandb.run.summary["best_epoch"] = ep

    t.save({"model": model.state_dict(), "config": ckpt_cfg,
            "model_name": model_name, "convention": conv,
            "feat_dim": feat_dim, "out_dim": out_dim, "epoch": cfg.training.epochs,
            "y_mean": y_mean, "y_std": y_std},
           os.path.join(out_dir, "last.pt"))
    print(f"[{model_name}] done. best val loss={best_val:.4f} -> {out_dir}/best.pt")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
