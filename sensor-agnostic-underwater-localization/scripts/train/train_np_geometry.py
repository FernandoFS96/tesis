"""Train the neural-process models (Hydra entry point).

The model is chosen with the `model` config group. Each model config has an
`arch` (the class to build) and a `name` (used for run names and folders).

    arch=cnp          anp.DeterministicModel           attentive CNP
    arch=anp          anp.LatentModel                  attentive latent NP
    arch=pfn          pfn.PFNModel                     prior-data fitted network
    arch=rcnp         r_anp.DeterministicModel         recurrent CNP
    arch=ranp         r_anp.LatentModel                recurrent latent NP
    arch=online_ranp  online_r_anp.OnlineLatentModel   streaming RANP

The plain configs (cnp, anp, ...) use the raw acoustics only; the spatial_*
configs add the spatial front end, which also receives the sensor positions.

Calling conventions:
  split   (cnp, anp, pfn)  forward(context_x, context_y, target_x, target_y, beta)
  indexed (rcnp, ranp)     forward(x_seq, context_indices, context_y,
                                   target_indices, target_y, beta), with the
                           full sequence in x_seq
  online  (online_ranp)    forward_streaming(x_seq, y_seq, ctx_idx, chunk_size)
All of them return (mean, var, loss, kl, nll).

Data: <processed>/geometry_split/{train,val,test}_data.pkl as written by
data/process_data.py, lists of dicts with "X" (points x features), "y"
(points x 3), "sensor_pos" (sensors x 3), "geometry_id" and "theta".

Examples:
    python scripts/train/train_np_geometry.py model=spatial_cnp
    python scripts/train/train_np_geometry.py model=pfn data.normalize_y=true \\
        training.epochs=500

Outputs (Hydra run directory, or out_dir): best.pt, last.pt, train_log.csv,
config.yaml.
"""

import os
import sys
import json
import pickle
import time
from collections import defaultdict

import numpy as np
import torch as t
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import wandb

# Make `src.models` importable whatever the working directory (Hydra may
# change it).
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

# `${repo_root:}` in config/train.yaml resolves to the repository root, so
# outputs and data paths do not depend on the launch directory.
OmegaConf.register_resolver("repo_root", lambda: _PROJECT_ROOT, replace=True)
# Model factory shared with the evaluation scripts.
from src.models.registry import build_model, is_latent  # type: ignore  # noqa: E402
from src.models.anp import physical_var_floor, set_var_floor  # type: ignore  # noqa: E402

import viz  # type: ignore  # noqa: E402  # periodic W&B training visualizations


def should_log_viz(ep, total_epochs, viz_cfg):
    """Whether to log the training figures at this epoch: at each of the
    `milestones`, then every `every_n_epochs`, and always at the last epoch."""
    if ep == total_epochs:
        return True  # always capture the final state
    milestones = list(viz_cfg.get("milestones", []) or [])
    if ep in milestones:
        return True
    every = int(viz_cfg.get("every_n_epochs", 0) or 0)
    last_milestone = max(milestones) if milestones else 0
    return every > 0 and ep > last_milestone and ep % every == 0


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


def compute_x_stats(dataset, max_samples=200):
    """Global mean and std of the acoustic features, for the spatial front end's
    'standardize' mode. Estimated on a subsample."""
    n = min(len(dataset.samples), max_samples)
    xs = np.concatenate(
        [np.asarray(dataset.samples[i]["X"], dtype=np.float32).ravel() for i in range(n)])
    return float(xs.mean()), float(xs.std() + 1e-8)


# --------------------------------------------------------------------------- #
# Dataset (one item = one trajectory) + collate producing both conventions
# --------------------------------------------------------------------------- #
class TrajectoryDataset(Dataset):
    def __init__(self, pkl_path):
        with open(pkl_path, "rb") as f:
            self.samples = pickle.load(f)
        x0 = np.asarray(self.samples[0]["X"]); y0 = np.asarray(self.samples[0]["y"])
        self.ppt, self.feat_dim = x0.shape
        self.out_dim = y0.shape[-1]
        # sensor_pos present only for the geometry split; n_sensors inferred from it.
        sp0 = self.samples[0].get("sensor_pos", None)
        self.has_sensor_pos = sp0 is not None
        self.n_sensors = int(np.asarray(sp0).shape[0]) if sp0 is not None else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        X = t.as_tensor(np.asarray(s["X"], dtype=np.float32))
        y = t.as_tensor(np.asarray(s["y"], dtype=np.float32))
        sp = s.get("sensor_pos", None)
        sensor_pos = (t.as_tensor(np.asarray(sp, dtype=np.float32))
                      if sp is not None else t.zeros(0))
        return X, y, int(s.get("geometry_id", -1)), float(s.get("theta", -1.0)), sensor_pos


def _worker_rng(base_seed):
    """NumPy generator seeded per DataLoader worker, so that workers draw
    different context sizes while staying reproducible for a given seed."""
    wi = t.utils.data.get_worker_info()
    wid = 0 if wi is None else wi.id
    base = 0 if base_seed is None else int(base_seed)
    return np.random.default_rng([base, wid])


def sample_sensor_mask(rng, B, S, mode, p=0.3, min_keep=3):
    """(B, S) bool sensor mask, True for present sensors, used for sensor dropout
    in training. 'bernoulli' drops each sensor with probability p; 'k_uniform'
    keeps a uniform number of sensors in [min_keep, S]. At least min_keep stay."""
    keep = np.ones((B, S), dtype=bool)
    for b in range(B):
        if mode == "bernoulli":
            drop = rng.random(S) < p
            excess = int(drop.sum()) - (S - min_keep)
            if excess > 0:                            # restore some to keep min_keep
                drop[rng.choice(np.flatnonzero(drop), excess, replace=False)] = False
        elif mode == "k_uniform":
            drop = np.zeros(S, dtype=bool)
            drop[rng.permutation(S)[:S - int(rng.integers(min_keep, S + 1))]] = True
        else:
            raise ValueError(f"sensor_drop.mode must be none|bernoulli|k_uniform, got {mode}")
        keep[b] = ~drop
    return t.as_tensor(keep)


def perturb_sensor_pos(rng, sensor_pos, max_m):
    """Train-only sensor-position error: each sample draws an sd from U(0, max_m) and
    adds N(0, sd^2) per sensor and axis to the horizontal positions the model is
    given (labels stay true). One draw per sample, shared by its context and targets."""
    B, S = sensor_pos.shape[:2]
    sd = rng.uniform(0.0, max_m, size=(B, 1, 1))
    sp = sensor_pos.clone()
    sp[..., :2] = sp[..., :2] + t.as_tensor(rng.standard_normal((B, S, 2)) * sd, dtype=sp.dtype)
    return sp


def make_collate(ctx_min, ctx_max, fixed_ctx=None, seed=None, ctx_sample_mode="random",
                 exclude_ctx_from_target=True, per_sample=False, aug_m=0.0,
                 calibration=False, dataset=None, indexed=False, rot_aug=False,
                 sensor_drop=None, pos_err=0.0):
    """Collate function producing both calling conventions:
       - context_x / context_y / target_x / target_y (split models)
       - x_seq with context_indices / target_indices (indexed models)

       ctx_sample_mode: 'first' takes the first n_ctx points, 'random' a random
           subset.
       exclude_ctx_from_target: if True (default) the targets are the points not
           in the context, otherwise all points.
       per_sample: each sample gets its own context size; the context is padded
           and marked with context_mask, and target_mask selects the scored
           targets. Split models only; ignored when fixed_ctx is set.
       calibration: the context comes from another trajectory of the same
           deployment (same geometry and theta), and the targets are all points
           of the sample's own trajectory. Needs `dataset`.
       indexed: in the calibration regime, also build the indexed convention,
           with x_seq = the calibration points followed by the target transit.
       rot_aug: random rotation and reflection of the scene about the array
           centroid (training only).
       sensor_drop: None or (mode, p, min_keep) for sample_sensor_mask (training
           only); the mask is shared by a sample's context and targets.
       pos_err: maximum sd in meters for perturb_sensor_pos (training only)."""
    _rng = {"g": None}   # created lazily, once per DataLoader worker

    # Calibration regime: dataset indices grouped by (geometry_id, theta), so
    # the context can come from another trajectory of the same deployment.
    _donors = None
    if calibration:
        if dataset is None:
            raise ValueError("calibration=True needs `dataset` for donor lookup")
        _groups = {}
        for _i, _s in enumerate(dataset.samples):
            _groups.setdefault((int(_s.get("geometry_id", -1)),
                                round(float(_s.get("theta", -1.0)), 6)), []).append(_i)
        _thin = {k: len(v) for k, v in _groups.items() if len(v) < 2}
        if _thin:
            raise ValueError(
                "calibration context needs >=2 trajectories per (geometry, theta) "
                f"to draw a separate transit; {len(_thin)} group(s) have fewer "
                f"(e.g. {list(_thin.items())[:3]}).")
        _donors = {k: np.asarray(v) for k, v in _groups.items()}

    def collate(batch):
        if _rng["g"] is None:                        # first call in this worker
            _rng["g"] = _worker_rng(seed)
        rng = _rng["g"]                              # per-worker, decorrelated
        Xs = t.stack([b[0] for b in batch], dim=0)   # (B, ppt, feat)
        ys = t.stack([b[1] for b in batch], dim=0)   # (B, ppt, 3)
        gids = t.tensor([b[2] for b in batch])
        ths = t.tensor([b[3] for b in batch])
        sps = [b[4] for b in batch]
        sensor_pos = (t.stack(sps, dim=0)            # (B, n_sensors, 3)
                      if all(sp.numel() > 0 for sp in sps) else None)
        B, ppt, _ = Xs.shape

        # ---- calibration regime: context from another trajectory ------------
        # Built before augmentation, so the context labels get the same shift.
        _cal = None
        if calibration:
            if per_sample and fixed_ctx is None:
                ncs = rng.integers(ctx_min, ctx_max + 1, size=B).clip(1, ppt)
            else:   # validation (fixed_ctx) or batch-shared size
                nf = (int(fixed_ctx) if fixed_ctx is not None
                      else int(rng.integers(ctx_min, ctx_max + 1)))
                ncs = np.full(B, max(1, min(nf, ppt)), dtype=int)
            max_ctx = int(ncs.max())
            cx = t.zeros(B, max_ctx, Xs.size(-1), dtype=Xs.dtype)
            cy = t.zeros(B, max_ctx, ys.size(-1), dtype=ys.dtype)
            cmask = t.zeros(B, max_ctx, dtype=t.bool)
            for b in range(B):
                pool = _donors[(int(gids[b]), round(float(ths[b]), 6))]
                own = ys[b].numpy()
                dX = dy = None
                for _ in range(8):   # the donor must never be the target's own path
                    j = int(pool[rng.integers(len(pool))])
                    cand = np.asarray(dataset.samples[j]["y"], dtype=np.float32)
                    if not np.array_equal(cand, own):
                        dy = cand
                        dX = np.asarray(dataset.samples[j]["X"], dtype=np.float32)
                        break
                if dy is None:
                    # Fallback: scan the group for a different trajectory
                    # (each group holds at least two).
                    for j in pool.tolist():
                        cand = np.asarray(dataset.samples[int(j)]["y"], dtype=np.float32)
                        if not np.array_equal(cand, own):
                            dy = cand
                            dX = np.asarray(dataset.samples[int(j)]["X"], dtype=np.float32)
                            break
                if dy is None:
                    raise RuntimeError(
                        "every trajectory in this (geometry, theta) group is identical "
                        f"to the target, so no calibration donor exists (geometry "
                        f"{int(gids[b])}, theta {float(ths[b]):g})")
                k = int(ncs[b])
                sel = (np.arange(k) if ctx_sample_mode == "first"
                       else np.sort(rng.permutation(dX.shape[0])[:k]))
                cx[b, :k] = t.as_tensor(dX[sel])
                cy[b, :k] = t.as_tensor(dy[sel])
                cmask[b, :k] = True
            _cal = {"x": cx, "y": cy, "mask": cmask}

        # ---- translation augmentation (training only) -------------------------
        # Shift the whole scene (sensor positions and labels) by a random 2-D
        # offset per sample. The channel depends only on the relative
        # source-sensor geometry, so the acoustic features stay valid.
        if aug_m and aug_m > 0:
            if sensor_pos is None:
                raise ValueError("translation augmentation needs sensor_pos in "
                                 "the data (layout splits); got none")
            delta = t.as_tensor(rng.uniform(-aug_m, aug_m, size=(B, 1, 2)),
                                dtype=ys.dtype)
            ys = ys.clone(); ys[..., :2] = ys[..., :2] + delta
            sensor_pos = sensor_pos.clone()
            sensor_pos[..., :2] = sensor_pos[..., :2] + delta
            if _cal is not None:
                # The calibration trajectory belongs to the same scene.
                _cal["y"] = _cal["y"].clone()
                _cal["y"][..., :2] = _cal["y"][..., :2] + delta

        # ---- rotation augmentation (training only) ----------------------------
        # A rigid rotation or reflection of the scene about the array centroid
        # also leaves the acoustic features valid.
        if rot_aug:
            if sensor_pos is None:
                raise ValueError("rotation augmentation needs sensor_pos in the data")
            a = rng.uniform(0.0, 2 * np.pi, size=B)
            f = np.where(rng.random(B) < 0.5, -1.0, 1.0)            # reflect x first
            R = t.as_tensor(np.stack([np.stack([f * np.cos(a), -np.sin(a)], -1),
                                      np.stack([f * np.sin(a), np.cos(a)], -1)], -2),
                            dtype=ys.dtype)                           # (B, 2, 2)
            c = sensor_pos[..., :2].mean(1, keepdim=True)            # (B, 1, 2)

            def _rot(p):
                return c + (p - c) @ R.transpose(1, 2)
            ys = ys.clone(); ys[..., :2] = _rot(ys[..., :2])
            sensor_pos = sensor_pos.clone(); sensor_pos[..., :2] = _rot(sensor_pos[..., :2])
            if _cal is not None:                      # same scene, same rotation
                _cal["y"] = _cal["y"].clone()
                _cal["y"][..., :2] = _rot(_cal["y"][..., :2])

        # ---- calibration regime: return ---------------------------------------
        # Targets are all points of the sample's own trajectory. For indexed
        # models, context and targets form one sequence: the calibration points,
        # then the target transit.
        if _cal is not None and indexed:
            if not bool(_cal["mask"].all()):
                raise ValueError("indexed calibration batches need one context size "
                                 "per batch; set data.context.per_sample=false")
            k = _cal["x"].size(1)
            return {
                "x_seq": t.cat([_cal["x"], Xs], dim=1),
                "context_indices": t.arange(k, dtype=t.long),
                "target_indices": t.arange(k, k + ppt, dtype=t.long),
                "context_y": _cal["y"], "target_y": ys,
                "gids": gids, "thetas": ths, "sensor_pos": sensor_pos,
            }
        if _cal is not None:
            return {
                "context_x": _cal["x"], "context_y": _cal["y"],
                "context_mask": _cal["mask"],
                "target_x": Xs, "target_y": ys,
                "target_mask": t.ones(B, ppt, dtype=t.bool),
                "gids": gids, "thetas": ths, "sensor_pos": sensor_pos,
            }

        # ---- per-sample context sizes (padded and masked) ---------------------
        if per_sample and fixed_ctx is None:
            ncs = rng.integers(ctx_min, ctx_max + 1, size=B).clip(1, ppt)
            max_ctx = int(ncs.max())
            cx = t.zeros(B, max_ctx, Xs.size(-1), dtype=Xs.dtype)
            cy = t.zeros(B, max_ctx, ys.size(-1), dtype=ys.dtype)
            cmask = t.zeros(B, max_ctx, dtype=t.bool)
            tmask = t.ones(B, ppt, dtype=t.bool)     # which points to score
            for b in range(B):
                k = int(ncs[b])
                if ctx_sample_mode == "first":
                    idx = t.arange(k, dtype=t.long)
                else:
                    idx = t.as_tensor(np.sort(rng.permutation(ppt)[:k]), dtype=t.long)
                cx[b, :k] = Xs[b, idx]; cy[b, :k] = ys[b, idx]; cmask[b, :k] = True
                if exclude_ctx_from_target:
                    tmask[b, idx] = False            # don't score a sample's own context
            return {
                "context_x": cx, "context_y": cy, "context_mask": cmask,
                "target_x": Xs, "target_y": ys, "target_mask": tmask,
                "gids": gids, "thetas": ths, "sensor_pos": sensor_pos,
            }

        # ---- standard: one context size shared by the whole batch -------------
        n_ctx = int(fixed_ctx) if fixed_ctx is not None else int(rng.integers(ctx_min, ctx_max + 1))
        n_ctx = max(1, min(n_ctx, ppt))
        if ctx_sample_mode == "first":
            ctx_idx = t.arange(n_ctx, dtype=t.long)
        else:
            ctx_idx = t.as_tensor(np.sort(rng.permutation(ppt)[:n_ctx]), dtype=t.long)
        if exclude_ctx_from_target:
            mask = t.ones(ppt, dtype=t.bool)
            mask[ctx_idx] = False
            tgt_idx = t.nonzero(mask, as_tuple=False).squeeze(-1)  # complement of context
            # Guard: if context covers every point, fall back to all points so
            # the batch is never empty.
            if tgt_idx.numel() == 0:
                tgt_idx = t.arange(ppt, dtype=t.long)
        else:
            tgt_idx = t.arange(ppt, dtype=t.long)   # predict all points
        return {
            "x_seq": Xs, "y_full": ys,
            "context_indices": ctx_idx, "target_indices": tgt_idx,
            "context_x": Xs[:, ctx_idx, :], "context_y": ys[:, ctx_idx, :],
            "target_x": Xs[:, tgt_idx, :], "target_y": ys[:, tgt_idx, :],
            "gids": gids, "thetas": ths, "sensor_pos": sensor_pos,
        }
    if not sensor_drop and not pos_err:
        return collate

    def collate_drop(batch):
        out = collate(batch)
        B, S = out["sensor_pos"].shape[:2]
        if sensor_drop:
            out["sensor_mask"] = sample_sensor_mask(_rng["g"], B, S, *sensor_drop)
        if pos_err:
            out["sensor_pos"] = perturb_sensor_pos(_rng["g"], out["sensor_pos"], pos_err)
        return out
    return collate_drop


def make_online_collate(ctx_min, ctx_max, chunk_size, fixed_ctx=None, seed=None,
                        ctx_sample_mode="random"):
    """Collate for the online (streaming) convention: the full ordered
    trajectory, the timesteps that become context (position fixes, shared across
    the batch) and the chunk size.
       ctx_sample_mode='random' -> fixes spread over the trajectory
       ctx_sample_mode='first'  -> fixes form an initial prefix"""
    _rng = {"g": None}   # created lazily, once per DataLoader worker

    def collate(batch):
        if _rng["g"] is None:                        # first call in this worker
            _rng["g"] = _worker_rng(seed)
        rng = _rng["g"]                              # per-worker, decorrelated
        Xs = t.stack([b[0] for b in batch], dim=0)   # (B, ppt, feat)
        ys = t.stack([b[1] for b in batch], dim=0)   # (B, ppt, 3)
        gids = t.tensor([b[2] for b in batch])
        ths = t.tensor([b[3] for b in batch])
        sps = [b[4] for b in batch]
        sensor_pos = (t.stack(sps, dim=0)
                      if all(sp.numel() > 0 for sp in sps) else None)
        B, ppt, _ = Xs.shape
        n_ctx = int(fixed_ctx) if fixed_ctx is not None else int(rng.integers(ctx_min, ctx_max + 1))
        n_ctx = max(1, min(n_ctx, ppt))
        if ctx_sample_mode == "first":
            ctx_idx = t.arange(n_ctx, dtype=t.long)
        else:  # scattered fixes, kept in temporal (sorted) order
            ctx_idx = t.as_tensor(np.sort(rng.permutation(ppt)[:n_ctx]), dtype=t.long)
        return {
            "x_seq": Xs, "y_seq": ys,
            "ctx_idx": ctx_idx, "chunk_size": int(chunk_size),
            "gids": gids, "thetas": ths, "sensor_pos": sensor_pos,
        }
    return collate


# --------------------------------------------------------------------------- #
# Forward dispatch (the family-dependent call)
# --------------------------------------------------------------------------- #
def model_forward(model, conv, batch, device, beta, with_target_y=True,
                  predict_with_prior=False):
    if conv == "online":
        x_seq = batch["x_seq"].to(device)
        y_seq = batch["y_seq"].to(device)
        ctx_idx = batch["ctx_idx"].to(device)
        chunk = int(batch["chunk_size"])
        return model.forward_streaming(x_seq, y_seq, ctx_idx, chunk, beta,
                                       predict_with_prior=predict_with_prior)
    cy = batch["context_y"].to(device)
    ty = batch["target_y"].to(device) if with_target_y else None
    sp = batch.get("sensor_pos")
    sp = sp.to(device) if sp is not None else None
    if conv == "split":
        cx = batch["context_x"].to(device)
        tx = batch["target_x"].to(device)
        # Masks from per-sample context batching or the calibration regime.
        kw = {}
        cmask = batch.get("context_mask")
        if cmask is not None:
            kw["context_mask"] = cmask.to(device)
        tmask = batch.get("target_mask")
        if tmask is not None:
            kw["target_mask"] = tmask.to(device)
        smask = batch.get("sensor_mask")      # sensor dropout (train only)
        if smask is not None:
            kw["sensor_mask"] = smask.to(device)
        return model(cx, cy, tx, ty, beta, predict_with_prior=predict_with_prior,
                     sensor_pos=sp, **kw)
    else:  # indexed (ranp / rcnp; sensor_pos used only by spatial_r* front ends)
        x_seq = batch["x_seq"].to(device)
        ci = batch["context_indices"].to(device)
        ti = batch["target_indices"].to(device)
        smask = batch.get("sensor_mask")
        kw = {} if smask is None else {"sensor_mask": smask.to(device)}
        return model(x_seq, ci, cy, ti, ty, beta, predict_with_prior=predict_with_prior,
                     sensor_pos=sp, **kw)


# --------------------------------------------------------------------------- #
# Epoch loop
# --------------------------------------------------------------------------- #
def run_epoch(model, conv, loader, device, beta, optimizer=None, desc="",
              y_mean=None, y_std=None):
    """y_mean / y_std: if not None, y is normalized before the forward pass and
    denormalized before MAE computation so the metric stays in physical units."""
    train = optimizer is not None
    model.train(train)
    tot_loss = tot_nll = tot_kl = tot_mae = 0.0; n = 0
    # Per-geometry error accumulators, so the caller can break the metric down by
    # region (train / interp / extrap) using the splits.json labels.
    err_sum = defaultdict(float); err_cnt = defaultdict(int)
    ym = y_mean.to(device) if y_mean is not None else None
    ys = y_std.to(device)  if y_std  is not None else None
    bar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    online = (conv == "online")
    for batch in bar:
        # 'online' carries the full trajectory in y_seq; others split into target_y.
        y_key = "y_seq" if online else "target_y"
        ty_raw = batch[y_key].to(device)        # physical units, for MAE
        gids = batch["gids"].tolist()
        if ym is not None:
            if online:
                batch = {**batch, "y_seq": (batch["y_seq"].to(device) - ym) / ys}
            else:
                batch = {**batch,
                         "context_y": (batch["context_y"].to(device) - ym) / ys,
                         "target_y":  (batch["target_y"].to(device)  - ym) / ys}
        with t.set_grad_enabled(train):
            # Outside training, latent models predict from the prior latent.
            mean, var, loss, kl, nll = model_forward(
                model, conv, batch, device, beta, with_target_y=True,
                predict_with_prior=(not train))
            if train:
                optimizer.zero_grad()
                loss.backward()
                t.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        b = ty_raw.size(0)
        tot_loss += loss.item() * b
        tot_nll += nll.item() * b
        tot_kl += (kl.item() if kl is not None else 0.0) * b  # 0 for deterministic models
        with t.no_grad():
            mean_phys = mean * ys + ym if ym is not None else mean
            dist = t.sqrt(((mean_phys - ty_raw) ** 2).sum(-1) + 1e-12)  # (B, n_tgt)
            # With per-sample batching, score only each sample's targets.
            tmask = batch.get("target_mask")
            if tmask is not None:
                tmask = tmask.to(device).float()
                denom = tmask.sum(dim=1).clamp(min=1)
                per_sample = ((dist * tmask).sum(dim=1) / denom).tolist()
                mae = float(((dist * tmask).sum() / tmask.sum().clamp(min=1)).item())
            else:
                mae = dist.mean().item()
                per_sample = dist.mean(dim=1).tolist()  # mean target error per trajectory
            tot_mae += mae * b
            for g, e in zip(gids, per_sample):
                err_sum[int(g)] += e; err_cnt[int(g)] += 1
        n += b
        bar.set_postfix(loss=f"{loss.item():.3f}", mae=f"{mae:.2f}")
    per_geo_mae = {g: err_sum[g] / err_cnt[g] for g in err_sum}
    # kl is last, so tr[0..3] and va[0..3] keep their meaning.
    return tot_loss / n, tot_nll / n, tot_mae / n, per_geo_mae, tot_kl / n


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def resolve_device(device_cfg):
    if device_cfg in (None, "auto"):
        return "cuda" if t.cuda.is_available() else "cpu"
    return device_cfg


def flat_ckpt_config(cfg, device):
    """Flat dict of model settings stored in the checkpoint and used to rebuild
    the model."""
    m = cfg.model
    ctx = cfg.data.context
    return {
        # "model" holds the architecture, "name" the display name.
        "model": m.get("arch", m.name),
        "name": m.name,
        "num_hidden": m.num_hidden,
        "rnn_type": m.get("rnn_type", "lstm"),
        "rnn_layers": m.get("rnn_layers", 1),
        "rnn_dropout": m.get("rnn_dropout", 0.0),
        "dropout": m.get("dropout", 0.1),
        "data_dir": cfg.data.data_dir,
        "normalize_y": cfg.data.normalize_y,
        "ctx_min": ctx.min,
        "ctx_max": ctx.max,
        "val_ctx": ctx.eval,
        "ctx_sample_mode": ctx.sample_mode,
        "exclude_ctx_from_target": ctx.get("exclude_from_target", True),
        "calibration": bool(ctx.get("calibration", False)),
        "n_layers": m.get("n_layers", 6),             # pfn-only
        "n_heads": m.get("n_heads", 8),               # pfn-only
        "ffn_mult": m.get("ffn_mult", 4),             # pfn-only
        "full_cov": m.get("full_cov", False),
        "attn_ffn": m.get("attn_ffn", False),   # feed-forward sublayer in Attention
        "latent_pooling": m.get("latent_pooling", None),  # None = the model's default
        "frame": m.get("frame", "absolute"),          # centroid = frame.CentroidFrame
        "var_floor_phys": m.get("var_floor_phys", None),  # None = default floor in model units
        "readout": m.get("readout", "joint"),         # pfn-only
        "n_cross_layers": m.get("n_cross_layers", 2), # pfn-only
        "max_context": m.get("max_context", 128),     # online-only
        "chunk_size": cfg.data.get("chunk_size", 8),   # online-only (streaming granularity)
        "epochs": cfg.training.epochs,
        "batch_size": cfg.training.batch_size,
        "lr": cfg.training.lr,
        "weight_decay": cfg.training.weight_decay,
        "beta": cfg.training.beta,
        "kl_warmup_epochs": cfg.training.get("kl_warmup_epochs", 0),
        "seed": cfg.seed,
        "device": device,
        "exp_name": cfg.exp_name,
    }


# --------------------------------------------------------------------------- #
# Main (Hydra entry point)
# --------------------------------------------------------------------------- #
@hydra.main(version_base=None, config_path="../../config", config_name="train")
def main(cfg: DictConfig):
    model_name = cfg.model.name                    # display name (runs/dirs/prints)
    arch = str(cfg.model.get("arch", model_name))  # architecture (build/dispatch)
    seed = cfg.seed
    t.manual_seed(seed); np.random.seed(seed)

    # Output directory: explicit out_dir, otherwise Hydra's per-run dir.
    out_dir = HydraConfig.get().runtime.output_dir
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
            config=dict(OmegaConf.to_container(cfg, resolve=True)),  # type: ignore[arg-type]
            dir=out_dir,
        )

    # ---- data --------------------------------------------------------------
    # A relative data_dir is resolved against the repository root.
    data_dir = cfg.data.data_dir
    if not os.path.isabs(data_dir):
        data_dir = os.path.normpath(os.path.join(_PROJECT_ROOT, data_dir))
    print(f"[{model_name}] cwd={os.getcwd()}  data_dir={data_dir}")
    train_ds = TrajectoryDataset(os.path.join(data_dir, "train_data.pkl"))
    val_ds = TrajectoryDataset(os.path.join(data_dir, "val_data.pkl"))

    # Optionally cap the number of training geometries (validation and test are
    # unchanged). The first N geometry ids are kept, so smaller sets are nested.
    max_train_geoms = cfg.data.get("max_train_geometries", None)
    # Optionally restrict training to the geometry ids listed in a JSON file.
    geom_file = cfg.data.get("train_geometry_ids", None)
    if geom_file:
        import json as _json
        with open(geom_file) as _f:
            want = {int(g) for g in _json.load(_f)}
        before = len(train_ds.samples)
        train_ds.samples = [s for s in train_ds.samples
                            if int(s.get("geometry_id", -1)) in want]
        kept = len({int(s.get("geometry_id", -1)) for s in train_ds.samples})
        print(f"[{model_name}] train restricted to {kept} geometries from "
              f"{os.path.basename(geom_file)}: {before} -> {len(train_ds.samples)} samples")
        if not train_ds.samples:
            raise SystemExit(f"train_geometry_ids left no training samples ({geom_file})")

    if max_train_geoms:
        keep = sorted(set(int(s.get("geometry_id", -1)) for s in train_ds.samples))[:int(max_train_geoms)]
        keep_set = set(keep)
        train_ds.samples = [s for s in train_ds.samples if int(s.get("geometry_id", -1)) in keep_set]
        print(f"[{model_name}] capped training to {len(keep)} geometries: {keep}")

    # Optionally cap the number of training trajectories (first M trajectory
    # ids); combines with the geometry cap.
    max_train_trajs = cfg.data.get("max_train_trajectories", None)
    if max_train_trajs:
        keep_t = sorted(set(int(s.get("traj_id", -1)) for s in train_ds.samples))[:int(max_train_trajs)]
        keep_tset = set(keep_t)
        train_ds.samples = [s for s in train_ds.samples if int(s.get("traj_id", -1)) in keep_tset]
        print(f"[{model_name}] capped training to {len(keep_t)} trajectories")

    # Optionally cap the validation trajectories as well.
    max_val_trajs = cfg.data.get("max_val_trajectories", None)
    if max_val_trajs:
        keep_v = sorted(set(int(s.get("traj_id", -1)) for s in val_ds.samples))[:int(max_val_trajs)]
        keep_vset = set(keep_v)
        val_ds.samples = [s for s in val_ds.samples if int(s.get("traj_id", -1)) in keep_vset]
        print(f"[{model_name}] capped validation to {len(keep_v)} trajectories")

    feat_dim, out_dim, ppt = train_ds.feat_dim, train_ds.out_dim, train_ds.ppt
    print(f"[{model_name}] feat_dim={feat_dim} out_dim={out_dim} ppt={ppt} "
          f"| train={len(train_ds)} val={len(val_ds)}")

    # Region labels (interp / extrap) from splits.json, when available.
    region_by_gid = {}
    _splits_path = os.path.join(data_dir, "splits.json")
    if os.path.exists(_splits_path):
        with open(_splits_path) as f:
            _split = json.load(f)
        region_by_gid = {int(k): v.get("region", "train")
                         for k, v in _split.get("labels", {}).items()}
        print(f"[{model_name}] region labels loaded for {len(region_by_gid)} held-out geometries")

    y_mean = y_std = None
    if cfg.data.normalize_y:
        y_mean, y_std = compute_y_stats(train_ds)
        print(f"[{model_name}] y_mean={y_mean.numpy()}  y_std={y_std.numpy()}")

    online = arch.lower() == "online_ranp"
    ctx = cfg.data.context
    excl_ctx = ctx.get("exclude_from_target", True)
    # Per-sample context sizes need the padding and target masks, which only the
    # split models (cnp / anp / pfn) handle.
    MASKED_ARCHS = ("cnp", "anp", "pfn")
    per_sample = bool(ctx.get("per_sample", False))
    if per_sample and arch not in MASKED_ARCHS:
        raise SystemExit(
            f"data.context.per_sample=true is only supported for the split models "
            f"(cnp / anp / pfn and their spatial_* variants), not '{model_name}' (arch={arch}). "
            f"Set data.context.per_sample=false for this model.")
    # Calibration regime: context from another trajectory of the same deployment.
    # Split models take it as a padded context set; indexed models take one
    # sequence (calibration points, then the target transit) and need one context
    # size per batch. Not available for the online model.
    INDEXED_ARCHS = ("ranp", "rcnp")
    calibration = bool(ctx.get("calibration", False))
    indexed = arch in INDEXED_ARCHS
    if calibration and arch not in MASKED_ARCHS + INDEXED_ARCHS:
        raise SystemExit(
            f"data.context.calibration=true is only supported for the split models "
            f"(cnp / anp / pfn and their spatial_* variants) and the indexed ones "
            f"(ranp / rcnp), not '{model_name}' (arch={arch}). "
            f"Set data.context.calibration=false for this model.")
    # Translation augmentation (training only); only meaningful for models that
    # receive the sensor positions.
    aug_m = float(cfg.data.get("trans_aug_m", 0.0) or 0.0)
    if aug_m > 0:
        _sc0 = cfg.model.get("spatial", None)
        if not (_sc0 is not None and bool(_sc0.get("enabled", False))):
            raise SystemExit(
                "data.trans_aug_m > 0 requires a spatial model (model=spatial_cnp"
                " / spatial_anp): translating (sensor_pos, y) is only learnable "
                "when the model consumes sensor_pos.")
        print(f"[{model_name}] translation augmentation ON: per-sample rigid "
              f"shift U(-{aug_m:.0f}, +{aug_m:.0f}) m in x/y (train only).")
    rot_aug = bool(cfg.data.get("rot_aug", False))
    if rot_aug:
        _sc0 = cfg.model.get("spatial", None)
        if not (_sc0 is not None and bool(_sc0.get("enabled", False))):
            raise SystemExit("data.rot_aug=true requires a spatial model (it rotates "
                             "sensor_pos with y).")
        print(f"[{model_name}] rotation augmentation ON: per-sample rotation and "
              f"reflection about the array centroid (train only).")
    _sd = cfg.data.get("sensor_drop", None) or {}
    sensor_drop = None
    if str(_sd.get("mode", "none")).lower() != "none":
        _sc0 = cfg.model.get("spatial", None)
        if not (_sc0 is not None and bool(_sc0.get("enabled", False))) or online:
            raise SystemExit("data.sensor_drop requires a spatial model (the spatial "
                             "encoder is what drops the masked sensors).")
        if _sc0.get("encoder", "legacy") == "legacy" and not bool(_sc0.get("tokenize", True)):
            raise SystemExit("data.sensor_drop needs model.spatial.tokenize=true")
        sensor_drop = (str(_sd.mode).lower(), float(_sd.get("p", 0.3)),
                       int(_sd.get("min_keep", 3)))
        print(f"[{model_name}] sensor dropout ON: mode={sensor_drop[0]} p={sensor_drop[1]:g} "
              f"min_keep={sensor_drop[2]}, one mask per sample for context and "
              f"targets (train only; validation keeps every sensor).")
    pos_err = float(cfg.data.get("sensor_pos_err_m", 0.0) or 0.0)
    if pos_err > 0:
        _sc0 = cfg.model.get("spatial", None)
        if not (_sc0 is not None and bool(_sc0.get("enabled", False))) or online:
            raise SystemExit("data.sensor_pos_err_m requires a spatial model (it perturbs "
                             "the sensor_pos the model is given).")
        print(f"[{model_name}] sensor-position error ON: sd ~ U(0, {pos_err:g}) m per sample, "
              f"N(0, sd^2) per sensor and axis, shared by context and targets (train only).")
    if online:
        chunk_size = cfg.data.get("chunk_size", 8)
        train_collate = make_online_collate(
            ctx.min, ctx.max, chunk_size, seed=seed,
            ctx_sample_mode=ctx.sample_mode)
        val_collate = make_online_collate(
            ctx.min, ctx.max, chunk_size,
            fixed_ctx=ctx.eval, seed=seed + 1,
            ctx_sample_mode=ctx.sample_mode)
    else:
        if per_sample:
            print(f"[{model_name}] per-sample context batching on: "
                  f"per-sample context sizes + padding mask (train only; "
                  f"validation keeps the fixed ctx={ctx.eval}).")
        if calibration:
            print(f"[{model_name}] CALIBRATION context ON: context points come "
                  f"from a DIFFERENT trajectory of the same (geometry, theta); "
                  f"targets = all {ppt} points of the sample's own trajectory. "
                  f"Validation uses the same regime (fixed ctx={ctx.eval}), so "
                  f"best.pt is selected on the calibration metric.")
        train_collate = make_collate(
            ctx.min, ctx.max, seed=seed,
            ctx_sample_mode=ctx.sample_mode,
            exclude_ctx_from_target=excl_ctx, per_sample=per_sample,
            aug_m=aug_m, calibration=calibration, dataset=train_ds, indexed=indexed,
            rot_aug=rot_aug, sensor_drop=sensor_drop, pos_err=pos_err)
        # Validation uses a fixed context size; in the calibration regime its
        # context also comes from another trajectory (of the validation pool).
        val_collate = make_collate(
            ctx.min, ctx.max,
            fixed_ctx=ctx.eval, seed=seed + 1,
            ctx_sample_mode=ctx.sample_mode,
            exclude_ctx_from_target=excl_ctx,
            calibration=calibration, dataset=val_ds, indexed=indexed)

    # pin_memory speeds up host-to-device copies on CUDA; persistent_workers keeps
    # the worker processes (and their RNG state) across epochs.
    nw = int(cfg.training.num_workers)
    pin = (device.type == "cuda")
    persist = nw > 0
    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True,
        num_workers=nw, drop_last=True, collate_fn=train_collate,
        pin_memory=pin, persistent_workers=persist)
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=nw, collate_fn=val_collate,
        pin_memory=pin, persistent_workers=persist)

    # ---- spatial front end ---------------------------------------------------
    # Enabled with model.spatial.enabled; n_sensors is read from the data.
    spatial_cfg = None
    _sc = cfg.model.get("spatial", None)
    if _sc is not None and bool(_sc.get("enabled", False)):
        if not train_ds.has_sensor_pos:
            raise SystemExit(
                "model.spatial.enabled=true but the dataset has no 'sensor_pos' "
                "(use a layout split: data=layout_ood or data=layout_seen).")
        spatial_cfg = dict(OmegaConf.to_container(_sc, resolve=True))  # type: ignore[arg-type]
        spatial_cfg["n_sensors"] = train_ds.n_sensors
        print(f"[{model_name}] spatial encoder ON: n_sensors={train_ds.n_sensors} "
              f"tokenize={spatial_cfg.get('tokenize', True)} "
              f"pos={spatial_cfg.get('use_position', True)} "
              f"attn={spatial_cfg.get('use_attention', True)} "
              f"pool={spatial_cfg.get('pooling', 'attention')}")

    # ---- model -------------------------------------------------------------
    model, conv = build_model(
        arch, cfg.model.num_hidden, feat_dim, out_dim,
        rnn_type=cfg.model.get("rnn_type", "lstm"),
        rnn_layers=cfg.model.get("rnn_layers", 1),
        rnn_dropout=cfg.model.get("rnn_dropout", 0.0),
        dropout=cfg.model.get("dropout", 0.1),
        max_context=cfg.model.get("max_context", 128),
        spatial_cfg=spatial_cfg,
        n_layers=int(cfg.model.get("n_layers", 6)),
        n_heads=int(cfg.model.get("n_heads", 8)),
        ffn_mult=int(cfg.model.get("ffn_mult", 4)),
        readout=str(cfg.model.get("readout", "joint")),
        n_cross_layers=int(cfg.model.get("n_cross_layers", 2)),
        full_cov=bool(cfg.model.get("full_cov", False)),
        attn_ffn=bool(cfg.model.get("attn_ffn", False)),
        latent_pooling=cfg.model.get("latent_pooling", None),
        frame=str(cfg.model.get("frame", "absolute")))
    model = model.to(device)
    if str(cfg.model.get("frame", "absolute")).lower() == "centroid":
        model.set_y_stats(y_mean, y_std)          # None (unnormalized) keeps 0 / 1
        print(f"[{model_name}] array-centered frame ON: labels relative to the "
              f"sensor centroid inside the model")
    # Optional variance floor in physical units^2 (model.var_floor_phys); the
    # default is 1e-3 in model units.
    var_floor_phys = cfg.model.get("var_floor_phys", None)
    if var_floor_phys is not None:
        floor = physical_var_floor(var_floor_phys, y_std if cfg.data.normalize_y else None)
        set_var_floor(model, floor)
        print(f"[{model_name}] variance floor {var_floor_phys:g} physical units^2 "
              f"-> {floor.numpy()} in model units")
    # Global acoustic statistics for the spatial front end's 'standardize' mode.
    _se = getattr(model, "spatial_encoder", None)
    if _se is not None and getattr(_se, "norm_mode", "") == "standardize":
        xm, xs_ = compute_x_stats(train_ds)
        _se.set_acoustic_stats(xm, xs_)
        print(f"[{model_name}] acoustic standardize: mean={xm:.4g} std={xs_:.4g}")
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
        f.write("epoch,train_loss,train_nll,train_kl,train_mae,"
                "val_loss,val_nll,val_kl,val_mae,lr,sec\n")

    # KL weight for the latent models, warmed up linearly from 0 to base_beta
    # over kl_warmup_epochs (0 disables the warmup).
    base_beta = cfg.training.beta if is_latent(arch) else 0.0
    kl_warmup = int(cfg.training.get("kl_warmup_epochs", 0))

    def beta_at(ep):
        if base_beta == 0.0 or kl_warmup <= 0:
            return base_beta
        return base_beta * min(1.0, ep / float(kl_warmup))

    ckpt_cfg = flat_ckpt_config(cfg, str(device))
    ckpt_cfg["spatial"] = spatial_cfg  # None if disabled; lets eval rebuild the model

    # ---- periodic visualizations (optional) --------------------------------
    viz_cfg = wb.get("viz", None)
    use_viz = use_wandb and viz_cfg is not None and bool(viz_cfg.get("enabled", False))
    fixed_idx = []
    deg_pools = deg_labels = None
    if use_viz:
        fixed_idx = viz.select_fixed_trajectories(
            val_ds, int(viz_cfg.get("n_trajectories", 6)), seed=seed)
        if viz_cfg.get("plots", {}).get("degradation_scatter", False):
            # The degradation scatter needs the data pools and region labels.
            splits_path = os.path.join(data_dir, "splits.json")
            if os.path.exists(splits_path):
                with open(splits_path) as f:
                    split = json.load(f)
                deg_labels = {int(k): v for k, v in split.get("labels", {}).items()}
                deg_pools = {"train": train_ds, "val": val_ds}
                test_path = os.path.join(data_dir, "test_data.pkl")
                if os.path.exists(test_path):
                    deg_pools["test"] = TrajectoryDataset(test_path)
            else:
                print(f"[{model_name}] no splits.json; degradation_scatter disabled")
        if viz_cfg.get("watch_gradients", False) and wandb.run is not None:
            wandb.watch(model, log="all", log_freq=max(1, int(viz_cfg.get("every_n_epochs", 50))))
        _ms = list(viz_cfg.get("milestones", []) or [])
        print(f"[{model_name}] viz on: {len(fixed_idx)} fixed trajectories, "
              f"at epochs {_ms} then every {viz_cfg.get('every_n_epochs', 50)}")

    best_val_mae = float("inf")  # best.pt is selected by validation MAE (physical units)
    best_epoch = 0
    # Early stopping after `early_stop_patience` epochs without a validation MAE
    # improvement larger than early_stop_min_delta (0 or null disables it).
    # best.pt is updated on every improvement.
    es_patience = int(cfg.training.get("early_stop_patience", 0) or 0)
    es_min_delta = float(cfg.training.get("early_stop_min_delta", 0.0))
    es_ref = float("inf")        # val MAE at the last patience reset
    es_counter = 0
    epoch_bar = tqdm(range(1, cfg.training.epochs + 1), desc=f"{model_name} epochs", dynamic_ncols=True)
    for ep in epoch_bar:
        t0 = time.time()
        beta = beta_at(ep)   # KL warmup (0 -> base_beta over kl_warmup epochs)
        tr = run_epoch(model, conv, train_loader, device, beta, opt,
                       desc=f"ep{ep} train", y_mean=y_mean, y_std=y_std)
        va = run_epoch(model, conv, val_loader, device, beta, None,
                       desc=f"ep{ep} val", y_mean=y_mean, y_std=y_std)
        sched.step()
        dt = time.time() - t0
        lr_now = opt.param_groups[0]["lr"]
        with open(log_path, "a") as f:
            f.write(f"{ep},{tr[0]:.6f},{tr[1]:.6f},{tr[4]:.6e},{tr[2]:.6f},"
                    f"{va[0]:.6f},{va[1]:.6f},{va[4]:.6e},{va[2]:.6f},{lr_now:.2e},{dt:.1f}\n")
        # Validation MAE by held-out region (interp / extrap), from the
        # per-geometry MAE in va[3].
        region_maes = {}
        if region_by_gid:
            buckets = defaultdict(list)
            for gid, m in va[3].items():
                buckets[region_by_gid.get(gid, "train")].append(m)
            for reg in ("interp", "extrap"):
                if buckets.get(reg):
                    region_maes[f"val/mae_{reg}"] = float(np.mean(buckets[reg]))
            if buckets.get("interp") and buckets.get("extrap"):
                region_maes["val/mae_gap_extrap_interp"] = (
                    region_maes["val/mae_extrap"] - region_maes["val/mae_interp"])

        # ---- best-checkpoint selection + early stopping (by val MAE) ----------
        if va[2] < best_val_mae:
            best_val_mae = va[2]; best_epoch = ep
                # "model_name" stores the architecture, used to rebuild the model.
            t.save({"model": model.state_dict(), "config": ckpt_cfg,
                    "model_name": arch, "convention": conv,
                    "feat_dim": feat_dim, "out_dim": out_dim, "epoch": ep,
                    "y_mean": y_mean, "y_std": y_std},
                   os.path.join(out_dir, "best.pt"))
            if use_wandb and wandb.run is not None:
                wandb.run.summary["best_val_mae"] = best_val_mae
                wandb.run.summary["best_val_loss"] = va[0]
                wandb.run.summary["best_epoch"] = ep
        if va[2] < es_ref - es_min_delta:
            es_ref = va[2]; es_counter = 0
        else:
            es_counter += 1

        epoch_bar.set_postfix(tr_mae=f"{tr[2]:.2f}", va_mae=f"{va[2]:.2f}",
                              va_loss=f"{va[0]:.3f}",
                              es=(f"{es_counter}/{es_patience}" if es_patience else "off"))

        if use_wandb:
            wandb.log({
                "epoch": ep,
                "train/loss": tr[0], "train/nll": tr[1], "train/kl": tr[4], "train/mae": tr[2],
                "val/loss": va[0], "val/nll": va[1], "val/kl": va[4], "val/mae": va[2],
                "lr": lr_now, "beta": beta, "epoch_time_sec": dt,
                "es_counter": es_counter,
                **region_maes,
            }, step=ep)

        # periodic figures (also on the final epoch so the last state is logged)
        if use_viz and should_log_viz(ep, cfg.training.epochs, viz_cfg):
            try:
                viz.log_visualizations(
                    model, conv, ep=ep, viz_cfg=viz_cfg, val_ds=val_ds,
                    fixed_idx=fixed_idx, device=device,
                    val_ctx=ctx.eval,
                    ctx_sample_mode=ctx.sample_mode,
                    exclude_ctx_from_target=excl_ctx,
                    chunk_size=cfg.data.get("chunk_size", 8),
                    y_mean=y_mean, y_std=y_std,
                    deg_pools=deg_pools, deg_labels=deg_labels)
            except Exception as e:  # viz must never crash training
                print(f"[{model_name}] viz failed at epoch {ep}: {e}")

        # early stop: no val-MAE improvement for `es_patience` epochs
        if es_patience and es_counter >= es_patience:
            print(f"[{model_name}] early stop at epoch {ep} "
                  f"(no val-MAE improvement for {es_patience} epochs; "
                  f"best={best_val_mae:.4f} @ epoch {best_epoch})")
            break

    t.save({"model": model.state_dict(), "config": ckpt_cfg,
            "model_name": arch, "convention": conv,
            "feat_dim": feat_dim, "out_dim": out_dim, "epoch": ep,
            "y_mean": y_mean, "y_std": y_std},
           os.path.join(out_dir, "last.pt"))
    print(f"[{model_name}] done. best val mae={best_val_mae:.4f} @ epoch {best_epoch} "
          f"(stopped at {ep}/{cfg.training.epochs}) -> {out_dir}/best.pt")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
