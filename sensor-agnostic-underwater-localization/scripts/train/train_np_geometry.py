"""
train_np_geometry.py
====================

Unified trainer for the neural-process baselines, selected with the Hydra
``model`` group. Each model config carries two identity fields: ``arch`` (which
class to build, dispatched below) and ``name`` (display name for runs/dirs).

    arch=cnp   -> anp.DeterministicModel   (attentive CNP, no latent, no RNN)
    arch=anp   -> anp.LatentModel          (attentive latent NP)
    arch=ranp  -> r_anp.LatentModel        (recurrent attentive latent NP; LSTM)
    arch=rcnp  -> r_anp.DeterministicModel (bonus: recurrent CNP, indexed conv.)
    arch=online_ranp -> online_r_anp.OnlineLatentModel (causal streaming)

model=cnp/anp are FLAT baselines (no spatial encoding: raw acoustics only,
never sensor_pos) measuring the out-of-position degradation per architecture;
model=spatial_cnp/spatial_anp are the same archs with the sensor-position-aware
front end enabled (the models meant to close that gap).

WHY TWO CALLING CONVENTIONS (the one thing that matters here)
-------------------------------------------------------------
* cnp / anp  (anp.py): forward(context_x, context_y, target_x, target_y, beta)
                       , the caller pre-splits context and target into
                        separate tensors.

* ranp (r_anp.py): forward(x_seq, context_indices, context_y, target_indices,
                        target_y, beta), the caller passes the FULL sequence
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
    python train_np_geometry.py model=spatial_cnp
    python train_np_geometry.py model=anp data.normalize_y=true \
        data.context.sample_mode=first training.epochs=500
    python train_np_geometry.py model=ranp wandb.enabled=false

Outputs (to the Hydra run dir, or ``out_dir`` if set):
    best.pt, last.pt, train_log.csv, config.yaml
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

# --- model imports: support both ../src/models and a local copy -------------
# Add the repo root (parent of src/) so that `from src.models.anp import ...`
# works regardless of the (Hydra-changed) current working directory.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

# Anchor Hydra's output dir (and any config path) to the repo root regardless of
# the launch directory: `${repo_root:}` in config/train.yaml resolves to here.
# Registered at import time so it is available when Hydra composes the config.
OmegaConf.register_resolver("repo_root", lambda: _PROJECT_ROOT, replace=True)
import src.models.anp as anp_mod  # type: ignore  # noqa: E402
import src.models.r_anp as ranp_mod  # type: ignore  # noqa: E402
import src.models.online_r_anp as online_mod  # type: ignore  # noqa: E402
import src.models.pfn as pfn_mod  # type: ignore  # noqa: E402

import viz  # type: ignore  # noqa: E402  # periodic W&B training visualizations


# --------------------------------------------------------------------------- #
# Model factory
# --------------------------------------------------------------------------- #
def build_model(name, num_hidden, input_dim, output_dim,
                rnn_type="lstm", rnn_layers=1, rnn_dropout=0.0, dropout=0.1,
                max_context=128, spatial_cfg=None,
                n_layers=6, n_heads=8, ffn_mult=4,
                readout="joint", n_cross_layers=2, full_cov=False,
                attn_ffn=False):
    name = name.lower()
    if spatial_cfg and name == "online_ranp":
        raise ValueError("spatial encoder is not wired for online_ranp")
    if name == "pfn":  # masked single-stack transformer (spatial_cfg -> spatial_pfn)
        return pfn_mod.PFNModel(num_hidden, input_dim, output_dim, dropout=dropout,
                                spatial_cfg=spatial_cfg, n_layers=n_layers,
                                n_heads=n_heads, ffn_mult=ffn_mult,
                                readout=readout, n_cross_layers=n_cross_layers), "split"
    if name == "cnp":
        return anp_mod.DeterministicModel(num_hidden, input_dim, output_dim,
                                          dropout=dropout, spatial_cfg=spatial_cfg,
                                          full_cov=full_cov, attn_ffn=attn_ffn), "split"
    if name == "anp":
        return anp_mod.LatentModel(num_hidden, input_dim, output_dim,
                                   dropout=dropout, spatial_cfg=spatial_cfg,
                                   full_cov=full_cov, attn_ffn=attn_ffn), "split"
    if name == "ranp":  # spatial_cfg enables the spatial_ranp front end
        return (ranp_mod.LatentModel(num_hidden, input_dim, output_dim,
                                     rnn_type=rnn_type, rnn_layers=rnn_layers,
                                     rnn_dropout=rnn_dropout, dropout=dropout,
                                     spatial_cfg=spatial_cfg, full_cov=full_cov,
                                     attn_ffn=attn_ffn),
                "indexed")
    if name == "rcnp":  # recurrent CNP (spatial_cfg -> spatial_rcnp)
        return (ranp_mod.DeterministicModel(num_hidden, input_dim, output_dim,
                                            rnn_type=rnn_type, rnn_layers=rnn_layers,
                                            rnn_dropout=rnn_dropout, dropout=dropout,
                                            spatial_cfg=spatial_cfg, full_cov=full_cov,
                                     attn_ffn=attn_ffn),
                "indexed")
    if name == "online_ranp":  # streaming / online-deployable latent RANP
        return (online_mod.OnlineLatentModel(num_hidden, input_dim, output_dim,
                                             rnn_type=rnn_type, rnn_layers=rnn_layers,
                                             rnn_dropout=rnn_dropout, dropout=dropout,
                                             max_context=max_context),
                "online")
    raise ValueError(f"unknown model '{name}' (use cnp|anp|pfn|ranp|rcnp|online_ranp)")


def is_latent(name):
    return name.lower() in ("anp", "ranp", "online_ranp")  # posterior + KL


def should_log_viz(ep, total_epochs, viz_cfg):
    """Whether to build+log the training figures at this epoch.

    Schedule: the explicit `milestones` epochs (dense early on), then every
    `every_n_epochs` once past the last milestone, plus always the final epoch.
    With the default config (milestones=[5,10,25,50,100], every_n_epochs=100)
    this fires at 5, 10, 25, 50, 100, then 200, 300, 400, ... If `milestones`
    is empty it degrades to a plain every-N-epochs cadence.
    """
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
    """Single global mean/std of the acoustic features (for the spatial encoder's
    'standardize' mode). A subsample is enough -- the global scale is stable."""
    n = min(len(dataset.samples), max_samples)
    xs = np.concatenate(
        [np.asarray(dataset.samples[i]["X"], dtype=np.float32).ravel() for i in range(n)])
    return float(xs.mean()), float(xs.std() + 1e-8)


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
    """NumPy Generator seeded uniquely per DataLoader worker.

    A ``default_rng(seed)`` captured in a collate closure is forked to every
    worker with the SAME seed, so with num_workers>1 all workers draw the
    identical context-size sequence -> context diversity collapses by ~num_workers
    (the whole point of per-batch context sampling is lost). Seeding by
    (base_seed, worker_id) via a SeedSequence gives each worker a decorrelated
    stream while staying deterministic for a given run seed. Single-process
    loading (worker_info is None) reduces to worker id 0."""
    wi = t.utils.data.get_worker_info()
    wid = 0 if wi is None else wi.id
    base = 0 if base_seed is None else int(base_seed)
    return np.random.default_rng([base, wid])


def make_collate(ctx_min, ctx_max, fixed_ctx=None, seed=None, ctx_sample_mode="random",
                 exclude_ctx_from_target=True, per_sample=False, aug_m=0.0,
                 calibration=False, dataset=None):
    """Builds a batch carrying BOTH calling conventions:
       - context_x/context_y/target_x/target_y  (for cnp/anp)
       - x_seq + context_indices/target_indices (for ranp)
       Context is a subset of the points determined by ctx_sample_mode.
       ctx_sample_mode='first'  -> take the first n_ctx points (ordered prefix)
       ctx_sample_mode='random' -> random permutation (original behaviour)

       Target-set composition (exclude_ctx_from_target):
         True  -> targets are the COMPLEMENT of the context (non-overlapping;
                  the model is scored only on unseen points). [default]
         False -> targets are ALL points (context is a subset of the targets,
                  the standard NP convention).

       per_sample (Option C): if True, each sample in the batch gets its OWN
       context size (instead of one size shared by the whole batch). Context is
       padded to the batch-max and a `context_mask` marks the real slots; targets
       become ALL points with a per-sample `target_mask` selecting the scored set.
       This decouples context-size variability from batch size, so a large GPU
       batch still carries many context sizes. SPLIT convention only (cnp/anp/
       spatial_*); it emits no context_indices, so ranp/rcnp/online can't use it.
       Ignored when fixed_ctx is set (validation keeps one fixed size for all).

       calibration (the CALIBRATION regime): if True the context points come
       from a DIFFERENT trajectory of the SAME (geometry, theta) as the
       targets -- a separate 'calibration transit' -- and the targets become ALL
       points of the sample's own trajectory. This is the operationally faithful
       setting (the calibration track is never the track you want to localize)
       and it removes the trajectory-continuation shortcut that same-trajectory
       context makes available: with a prefix of the target's OWN path revealed,
       a model can score well by extrapolating that path and ignoring the
       acoustics entirely. Needs `dataset` (donor lookup) and the SPLIT
       convention. `exclude_ctx_from_target` is moot here -- context and targets
       are disjoint by construction."""
    _rng = {"g": None}   # created lazily, once per DataLoader worker

    # Donor pools for the CALIBRATION regime: dataset indices grouped by
    # (geometry_id, theta), so a sample's context is drawn from another
    # trajectory of the SAME deployment (same array layout, same channel).
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

        # ---- CALIBRATION regime: context from a SEPARATE trajectory ----------
        # Built BEFORE augmentation so the donor's labels can take the same
        # rigid shift below (one physical scene => one coordinate frame).
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
                    # Deterministic fallback. With few trajectories per (geometry,
                    # theta) -- e.g. 10 in the layout-heavy prior -- the random
                    # draw hits the target's own path with probability 1/k each
                    # time, so pure rejection sampling eventually exhausts its
                    # retries and would kill a multi-hour run. Scan the pool
                    # instead; this succeeds whenever the group holds >= 2
                    # distinct trajectories, which the collate factory asserts.
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

        # ---- translation augmentation (train-only; aug_m == 0 disables) -------
        # Per-sample rigid 2-D shift of the WHOLE scene (sensor_pos AND y
        # together). The simulated channel depends only on RELATIVE source-
        # sensor geometry (acoustic_data_generator.py: d0 = ||r_pos - traj||),
        # so the acoustic features X stay EXACTLY valid while the scene moves ->
        # free sensor-displacement diversity along precisely the layout-OOD axis
        # (teaches the absolute-frame Fourier position encoding offsets beyond
        # the training hull of array centres, where the extrap penalty lives).
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
                # The calibration transit belongs to the SAME scene, so it takes
                # the SAME rigid shift -- otherwise the context labels and the
                # target labels would sit in different frames.
                _cal["y"] = _cal["y"].clone()
                _cal["y"][..., :2] = _cal["y"][..., :2] + delta

        # ---- CALIBRATION return: context is a different trajectory ------------
        # Targets are ALL points of this sample's own trajectory; context and
        # targets are disjoint by construction, so nothing needs excluding.
        # No x_seq/context_indices are emitted: cross-trajectory context cannot
        # be expressed as indices into one trajectory, so an indexed (ranp/rcnp/
        # online) model fails loudly here rather than silently using the wrong
        # points. main() also blocks those archs up front.
        if _cal is not None:
            return {
                "context_x": _cal["x"], "context_y": _cal["y"],
                "context_mask": _cal["mask"],
                "target_x": Xs, "target_y": ys,
                "target_mask": t.ones(B, ppt, dtype=t.bool),
                "gids": gids, "thetas": ths, "sensor_pos": sensor_pos,
            }

        # ---- Option C: per-sample context sizes (padded + masked) -------------
        if per_sample and fixed_ctx is None:
            ncs = rng.integers(ctx_min, ctx_max + 1, size=B).clip(1, ppt)
            max_ctx = int(ncs.max())
            cx = t.zeros(B, max_ctx, Xs.size(-1), dtype=Xs.dtype)
            cy = t.zeros(B, max_ctx, ys.size(-1), dtype=ys.dtype)
            cmask = t.zeros(B, max_ctx, dtype=t.bool)
            tmask = t.ones(B, ppt, dtype=t.bool)     # which points to SCORE
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
    return collate


def make_online_collate(ctx_min, ctx_max, chunk_size, fixed_ctx=None, seed=None,
                        ctx_sample_mode="random"):
    """Collate for the 'online' (streaming) convention. Emits the FULL ordered
    trajectory plus the timesteps that become context ('fixes') and the chunk
    size. Temporal order is preserved (never shuffled). ctx_idx are the fix
    timesteps, shared across the batch; the model reveals them causally.
       ctx_sample_mode='random' -> fixes scattered across the trajectory
                                    (realistic: periodic position fixes)
       ctx_sample_mode='first'  -> fixes are an initial prefix (known start,
                                    then dead-reckon)."""
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
        # Per-sample context-batching (Option C) supplies these; None otherwise.
        # Pass them ONLY when present: the latent split model (anp/spatial_anp)
        # does not accept context_mask/target_mask, so passing them (even as None)
        # would TypeError. per_sample is guarded to cnp/spatial_cnp anyway.
        kw = {}
        cmask = batch.get("context_mask")
        if cmask is not None:
            kw["context_mask"] = cmask.to(device)
        tmask = batch.get("target_mask")
        if tmask is not None:
            kw["target_mask"] = tmask.to(device)
        return model(cx, cy, tx, ty, beta, predict_with_prior=predict_with_prior,
                     sensor_pos=sp, **kw)
    else:  # indexed (ranp / rcnp; sensor_pos used only by spatial_r* front ends)
        x_seq = batch["x_seq"].to(device)
        ci = batch["context_indices"].to(device)
        ti = batch["target_indices"].to(device)
        return model(x_seq, ci, cy, ti, ty, beta, predict_with_prior=predict_with_prior,
                     sensor_pos=sp)


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
            # Deployment-faithful validation/inference: latent models predict from
            # the PRIOR latent (context only, no peeking at target labels) whenever
            # we are NOT training. Training uses the posterior (teacher forcing).
            # The posterior + KL/NLL are still computed for logging; only the z
            # that drives the prediction differs. Deterministic models ignore it.
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
            # Per-sample context-batching scores only each sample's target subset;
            # mask the MAE the same way (else padded/context points would count).
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
    # kl appended LAST so existing tr[0..3]/va[0..3] index uses stay valid.
    return tot_loss / n, tot_nll / n, tot_mae / n, per_geo_mae, tot_kl / n


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
    ctx = cfg.data.context
    return {
        # "model" holds the ARCH (dispatch key for eval's build_model); "name"
        # is the display name (e.g. spatial_cnp). Checkpoint keys keep the
        # historical flat names so eval_np_geometry.py needs no changes.
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
    # Resolve a relative data_dir against the repo root (not the launch cwd) so
    # training works regardless of where it is launched from.
    data_dir = cfg.data.data_dir
    if not os.path.isabs(data_dir):
        data_dir = os.path.normpath(os.path.join(_PROJECT_ROOT, data_dir))
    print(f"[{model_name}] cwd={os.getcwd()}  data_dir={data_dir}")
    train_ds = TrajectoryDataset(os.path.join(data_dir, "train_data.pkl"))
    val_ds = TrajectoryDataset(os.path.join(data_dir, "val_data.pkl"))

    # Optionally cap the number of TRAINING geometries (val/test untouched) so a
    # "how many sensor layouts do we need?" sweep holds the held-out set fixed and
    # only varies training coverage. Keeps the FIRST N distinct geometry_ids
    # (sorted) -> nested subsets (10 subset of 20 subset of 40) for a clean curve.
    max_train_geoms = cfg.data.get("max_train_geometries", None)
    if max_train_geoms:
        keep = sorted(set(int(s.get("geometry_id", -1)) for s in train_ds.samples))[:int(max_train_geoms)]
        keep_set = set(keep)
        train_ds.samples = [s for s in train_ds.samples if int(s.get("geometry_id", -1)) in keep_set]
        print(f"[{model_name}] capped training to {len(keep)} geometries: {keep}")

    # Optionally cap the number of TRAINING trajectories (source paths). Composes
    # with the geometry cap -> training = (first N layouts) x (first M paths).
    # Keeps the first M distinct traj_ids (sorted) -> nested subsets. Needs data
    # processed with traj_id (geometry / topology modes); no-op on older data.
    max_train_trajs = cfg.data.get("max_train_trajectories", None)
    if max_train_trajs:
        keep_t = sorted(set(int(s.get("traj_id", -1)) for s in train_ds.samples))[:int(max_train_trajs)]
        keep_tset = set(keep_t)
        train_ds.samples = [s for s in train_ds.samples if int(s.get("traj_id", -1)) in keep_tset]
        print(f"[{model_name}] capped training to {len(keep_t)} trajectories")

    # Optionally cap the VALIDATION trajectories too. Set this equal to
    # max_train_trajectories to validate on the SAME source paths as training
    # (pure layout-generalization, matching the original test) -- this removes the
    # train/val path mismatch that inflates the late-epoch overfitting overshoot.
    # Only the val pool is touched; the geometries (held-out layouts) are untouched.
    max_val_trajs = cfg.data.get("max_val_trajectories", None)
    if max_val_trajs:
        keep_v = sorted(set(int(s.get("traj_id", -1)) for s in val_ds.samples))[:int(max_val_trajs)]
        keep_vset = set(keep_v)
        val_ds.samples = [s for s in val_ds.samples if int(s.get("traj_id", -1)) in keep_vset]
        print(f"[{model_name}] capped validation to {len(keep_v)} trajectories")

    feat_dim, out_dim, ppt = train_ds.feat_dim, train_ds.out_dim, train_ds.ppt
    print(f"[{model_name}] feat_dim={feat_dim} out_dim={out_dim} ppt={ppt} "
          f"| train={len(train_ds)} val={len(val_ds)}")

    # Region labels (interp / extrap) for held-out-geometry reporting. Present
    # only for the geometry split (splits.json); absent for topology / within-
    # geometry data, in which case the per-region breakdown is simply skipped.
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
    # Per-sample context batching (Option C): decouples context-size variability
    # from batch size so a large GPU batch stays diverse. Only the deterministic
    # split model handles the padding/target masks; block it elsewhere loudly.
    per_sample = bool(ctx.get("per_sample", False))
    if per_sample and arch not in ("cnp", "pfn"):
        raise SystemExit(
            f"data.context.per_sample=true is only supported for the deterministic "
            f"split model (model=cnp / spatial_cnp / pfn), not '{model_name}' (arch={arch}). "
            f"Set data.context.per_sample=false for this model.")
    # CALIBRATION regime: context drawn from a DIFFERENT trajectory of the same
    # (geometry, theta). Needs the padded context mask, so same restriction as
    # per_sample; indexed/online conventions cannot express cross-trajectory
    # context at all. Block anything else loudly.
    calibration = bool(ctx.get("calibration", False))
    if calibration and arch not in ("cnp", "pfn"):
        raise SystemExit(
            f"data.context.calibration=true is only supported for the deterministic "
            f"split model (model=cnp / spatial_cnp), not '{model_name}' (arch={arch}). "
            f"Set data.context.calibration=false for this model.")
    # Translation augmentation (train only): rigid per-sample 2-D shift of the
    # scene. Only meaningful when the model actually SEES sensor_pos; for a flat
    # model shifting the labels alone would just inject label noise.
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
            print(f"[{model_name}] per-sample context batching ON (Option C): "
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
            aug_m=aug_m, calibration=calibration, dataset=train_ds)
        # Validation stays standard fixed-ctx (comparable across configs); the
        # fixed_ctx path ignores per_sample by construction. Under CALIBRATION it
        # also draws its context cross-trajectory (donors from the VAL pool), so
        # the selection metric matches the regime we actually care about.
        val_collate = make_collate(
            ctx.min, ctx.max,
            fixed_ctx=ctx.eval, seed=seed + 1,
            ctx_sample_mode=ctx.sample_mode,
            exclude_ctx_from_target=excl_ctx,
            calibration=calibration, dataset=val_ds)

    # pin_memory speeds host->device copies on CUDA (harmless/off on CPU).
    # persistent_workers avoids re-forking workers every epoch when num_workers>0
    # (and lets each worker's rng keep advancing across epochs -> more variety).
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

    # ---- spatial encoder config (sensor-position-aware front end) -----------
    # Enabled via model.spatial.enabled; n_sensors is inferred from the data
    # (present only for the geometry split, which carries sensor_pos).
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
        attn_ffn=bool(cfg.model.get("attn_ffn", False)))
    model = model.to(device)
    # For the spatial encoder's 'standardize' acoustic mode, set the global
    # acoustic mean/std from the train data (preserves cross-sensor amplitude
    # ratios, unlike per-sensor LayerNorm).
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

    # latent models use beta; deterministic ignore it (no KL). The effective KL
    # weight is linearly warmed up from 0 to base_beta over kl_warmup_epochs (a
    # constant beta from epoch 1 invites posterior collapse); kl_warmup_epochs=0
    # disables the warmup (constant base_beta).
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
            # The degradation scatter needs the train/val/test pools + region
            # labels (from splits.json). Loaded once here, reused every log step.
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
    # Early stopping: stop after `early_stop_patience` epochs with no val-MAE
    # improvement (> min_delta). 0/null disables (train the full `epochs`).
    es_patience = int(cfg.training.get("early_stop_patience", 0) or 0)
    es_min_delta = float(cfg.training.get("early_stop_min_delta", 0.0))
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
        # Break the (blended) val MAE down by held-out region so interp vs extrap
        # are tracked separately, the blended number is dominated by the few
        # extrapolation geometries. va[3] is {geometry_id: mae} from the val pass.
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
        if va[2] < best_val_mae - es_min_delta:
            best_val_mae = va[2]; best_epoch = ep; es_counter = 0
            # "model_name" stores the ARCH: eval_np_geometry.py dispatches on it.
            t.save({"model": model.state_dict(), "config": ckpt_cfg,
                    "model_name": arch, "convention": conv,
                    "feat_dim": feat_dim, "out_dim": out_dim, "epoch": ep,
                    "y_mean": y_mean, "y_std": y_std},
                   os.path.join(out_dir, "best.pt"))
            if use_wandb and wandb.run is not None:
                wandb.run.summary["best_val_mae"] = best_val_mae
                wandb.run.summary["best_val_loss"] = va[0]
                wandb.run.summary["best_epoch"] = ep
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
