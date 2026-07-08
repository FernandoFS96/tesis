"""
viz.py
======

Training-time visualizations for the neural-process geometry baselines, designed
to be logged to Weights & Biases *periodically* (e.g. every N epochs) so you can
watch the model improve as it trains.

Everything in W&B is just ``wandb.log({key: media}, step=ep)``. Logging a
``wandb.Image`` under the SAME key across epochs gives you a step slider in the
UI -- scrub it to animate how the prediction tightens onto the ground truth.

Plots provided (each independently toggleable from config -- see
``log_visualizations``):

  pred_trajectory     predicted-vs-true trajectory overlay for a FIXED set of
                      validation trajectories, with context ("fix") points and
                      the model's predicted uncertainty drawn as ellipses.
  degradation_scatter per-geometry MAE vs sensor-centroid distance, coloured by
                      region (train / interp / extrap) -- the out-of-position
                      generalization metric, watched over training.
  calibration         predicted std vs realized error (binned) -- is the latent
                      model's uncertainty trustworthy?
  error_histogram     distribution of per-point localization errors (heavy tails
                      hide behind a decent mean MAE).
  streaming_drift     ONLINE model only: error as a function of
                      timesteps-since-last-fix -- dead-reckoning drift between
                      position fixes.

The plot builders are deliberately self-contained (their own small forward
helpers) so this module can also be imported by the eval script.

POSITION CONVENTION: y is (ppt, 3). The top-down plots use dims (0, 1) = (x, y).

NORMALIZATION: when y_mean/y_std are given the model lives in normalized space;
predictions are converted back to physical units here (mean*std+mean,
std_phys = sqrt(var)*y_std) so every figure is in metres.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch as t
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import wandb


# --------------------------------------------------------------------------- #
# Fixed-trajectory selection (called once at startup)
# --------------------------------------------------------------------------- #
def select_fixed_trajectories(val_ds, n, seed=0):
    """Pick n trajectory indices from val_ds, spread across geometries so the
    panel is representative. Returns a sorted list of dataset indices.

    These indices are chosen ONCE and reused every logging epoch, so the only
    thing changing across the step slider is the model -- not which trajectory
    you happen to be looking at."""
    n = min(n, len(val_ds))
    by_geo = defaultdict(list)
    for i in range(len(val_ds)):
        gid = int(val_ds.samples[i].get("geometry_id", -1))
        by_geo[gid].append(i)
    rng = np.random.default_rng(seed)
    # round-robin one per geometry until we have n
    picked, geos = [], sorted(by_geo)
    pos = {g: 0 for g in geos}
    for g in geos:
        rng.shuffle(by_geo[g])
    while len(picked) < n:
        progressed = False
        for g in geos:
            if pos[g] < len(by_geo[g]):
                picked.append(by_geo[g][pos[g]]); pos[g] += 1
                progressed = True
                if len(picked) >= n:
                    break
        if not progressed:
            break
    return sorted(picked)


# --------------------------------------------------------------------------- #
# Context-index drawing (mirrors the trainer / eval conventions)
# --------------------------------------------------------------------------- #
def _draw_ctx_idx(ppt, n_ctx, ctx_sample_mode, rng, device):
    n_ctx = max(1, min(n_ctx, ppt))
    if ctx_sample_mode == "first":
        return t.arange(n_ctx, device=device, dtype=t.long)
    return t.as_tensor(np.sort(rng.permutation(ppt)[:n_ctx]),
                       device=device, dtype=t.long)


# --------------------------------------------------------------------------- #
# Per-trajectory prediction over ALL points (so the whole path can be drawn)
# --------------------------------------------------------------------------- #
@t.no_grad()
def _predict_offline_all(model, conv, X, y_norm, ctx_idx, device, sensor_pos=None):
    """X,y_norm: (1, ppt, .). Returns mean, var over ALL ppt points (normalized
    space). conv in {'split', 'indexed'}. sensor_pos (1, n_sensors, 3) is passed
    for spatial-encoder models (split only); ignored by flat / recurrent ones."""
    ppt = X.size(1)
    tgt_idx = t.arange(ppt, device=device, dtype=t.long)
    cy = y_norm[:, ctx_idx, :]
    if conv == "split":
        cx = X[:, ctx_idx, :]; tx = X[:, tgt_idx, :]
        mean, var, *_ = model(cx, cy, tx, None, sensor_pos=sensor_pos)
    else:  # indexed (ranp)
        mean, var, *_ = model(X, ctx_idx, cy, tgt_idx, None)
    return mean, var


@t.no_grad()
def _predict_online_all(model, X, y_norm, ctx_idx, chunk_size, device):
    """Drive the real streaming deployment API over one trajectory, returning
    the live prior predictions (mean, var) for ALL points in normalized space.
    Mirrors eval_np_geometry.per_geometry_mae_online."""
    ppt = X.size(1)
    ctx_set = set(int(j) for j in ctx_idx.tolist())
    state = model.init_state(1, device)
    mean = t.zeros_like(y_norm); var = t.ones_like(y_norm)
    for s in range(0, ppt, chunk_size):
        e = min(s + chunk_size, ppt)
        m, v, h = model.step(X[:, s:e], state)
        mean[:, s:e], var[:, s:e] = m, v
        local = [j for j in range(s, e) if j in ctx_set]
        if local:
            loc = t.tensor([j - s for j in local], device=device, dtype=t.long)
            sel = t.tensor(local, device=device, dtype=t.long)
            model.register_context(h[:, loc, :], y_norm[:, sel, :], state)
    return mean, var


def _gather_predictions(model, conv, ds, fixed_idx, device, *, val_ctx,
                        ctx_sample_mode, exclude_ctx_from_target, chunk_size,
                        y_mean, y_std, seed=0):
    """Run the model on the fixed trajectories and return a list of per-trajectory
    dicts with everything the figures need, all in PHYSICAL units:

        {gid, true(ppt,Dy), pred(ppt,Dy), std(ppt,Dy), ctx_mask(ppt,),
         tgt_mask(ppt,)}
    """
    model.eval()
    ym = y_mean.to(device) if y_mean is not None else None
    ys = y_std.to(device) if y_std is not None else None
    rng = np.random.default_rng(seed)
    out = []
    for i in fixed_idx:
        X, y, gid, _, sensor_pos = ds[i]
        X = X.unsqueeze(0).to(device); y = y.unsqueeze(0).to(device)
        sp = sensor_pos.unsqueeze(0).to(device) if sensor_pos.numel() > 0 else None
        ppt = X.size(1)
        y_norm = (y - ym) / ys if ym is not None else y
        ctx_idx = _draw_ctx_idx(ppt, val_ctx, ctx_sample_mode, rng, device)
        if conv == "online":
            mean, var = _predict_online_all(model, X, y_norm, ctx_idx, chunk_size, device)
        else:
            mean, var = _predict_offline_all(model, conv, X, y_norm, ctx_idx, device, sensor_pos=sp)
        # back to physical units
        if ym is not None:
            mean = mean * ys + ym
            std = t.sqrt(var.clamp_min(1e-12)) * ys
        else:
            std = t.sqrt(var.clamp_min(1e-12))
        ctx_mask = t.zeros(ppt, dtype=t.bool); ctx_mask[ctx_idx.cpu()] = True
        tgt_mask = ~ctx_mask if exclude_ctx_from_target else t.ones(ppt, dtype=t.bool)
        if tgt_mask.sum() == 0:
            tgt_mask = t.ones(ppt, dtype=t.bool)
        out.append({
            "gid": int(gid),
            "true": y.squeeze(0).cpu().numpy(),
            "pred": mean.squeeze(0).cpu().numpy(),
            "std": std.squeeze(0).cpu().numpy(),
            "ctx_mask": ctx_mask.numpy(),
            "tgt_mask": tgt_mask.numpy(),
            "ctx_idx": ctx_idx.cpu().numpy(),
        })
    return out


# --------------------------------------------------------------------------- #
# Figure builders
# --------------------------------------------------------------------------- #
def build_trajectory_figure(preds, ep, plot_uncertainty=True, max_panels=12):
    """Grid of top-down (x,y) panels: true path, predicted target points, context
    fixes, and (optionally) per-point uncertainty ellipses (1 std)."""
    preds = preds[:max_panels]
    n = len(preds)
    ncols = min(3, n); nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.2 * nrows),
                            squeeze=False)
    for k, p in enumerate(preds):
        ax = axs[k // ncols][k % ncols]
        true, pred, std = p["true"], p["pred"], p["std"]
        cm, tm = p["ctx_mask"], p["tgt_mask"]
        # ground-truth path
        ax.plot(true[:, 0], true[:, 1], "-", color="0.4", lw=1.5, label="true", zorder=1)
        # uncertainty ellipses on target points (1 std in x,y)
        if plot_uncertainty:
            for j in np.nonzero(tm)[0]:
                e = Ellipse((pred[j, 0], pred[j, 1]),
                            width=2 * std[j, 0], height=2 * std[j, 1],
                            facecolor="#c0392b", edgecolor="none", alpha=0.10, zorder=2)
                ax.add_patch(e)
        # predicted target points
        ax.scatter(pred[tm, 0], pred[tm, 1], s=18, c="#c0392b",
                   edgecolor="k", linewidth=0.2, label="pred", zorder=3)
        # context fixes (on the true path)
        ax.scatter(true[cm, 0], true[cm, 1], s=42, marker="^", c="#2980b9",
                   edgecolor="k", linewidth=0.3, label="context fix", zorder=4)
        # per-trajectory target MAE
        err = np.sqrt(((pred[tm] - true[tm]) ** 2).sum(-1))
        ax.set_title(f"geo {p['gid']}  MAE={err.mean():.2f} m", fontsize=9)
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(fontsize=7, loc="best")
    # hide unused axes
    for k in range(n, nrows * ncols):
        axs[k // ncols][k % ncols].axis("off")
    fig.suptitle(f"Predicted vs true trajectories — epoch {ep}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def build_calibration_figure(preds, ep, n_bins=12):
    """Binned reliability: mean predicted std (x) vs RMS realized error (y) per
    coordinate, pooled over target points. A well-calibrated model sits on y=x."""
    std = np.concatenate([p["std"][p["tgt_mask"]] for p in preds], axis=0).reshape(-1)
    err = np.concatenate(
        [np.abs(p["pred"][p["tgt_mask"]] - p["true"][p["tgt_mask"]]) for p in preds],
        axis=0).reshape(-1)
    if std.size == 0:
        return None
    order = np.argsort(std)
    std, err = std[order], err[order]
    edges = np.linspace(0, len(std), n_bins + 1).astype(int)
    xs, ys = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            xs.append(std[a:b].mean())
            ys.append(np.sqrt((err[a:b] ** 2).mean()))
    fig, ax = plt.subplots(figsize=(5.2, 5))
    lim = max(max(xs), max(ys)) * 1.05 if xs else 1.0
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.6, label="ideal (y=x)")
    ax.plot(xs, ys, "o-", color="#8e44ad", label="model")
    ax.set_xlabel("predicted std [m]"); ax.set_ylabel("RMS realized error [m]")
    ax.set_title(f"Uncertainty calibration — epoch {ep}")
    ax.legend(); ax.grid(alpha=0.3); ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    fig.tight_layout()
    return fig


def build_error_histogram_figure(preds, ep):
    """Distribution of per-point Euclidean localization errors over target points."""
    errs = np.concatenate(
        [np.sqrt(((p["pred"][p["tgt_mask"]] - p["true"][p["tgt_mask"]]) ** 2).sum(-1))
         for p in preds], axis=0)
    if errs.size == 0:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(errs, bins=40, color="#16a085", alpha=0.85, edgecolor="k", linewidth=0.2)
    ax.axvline(errs.mean(), color="k", ls="--", lw=1, label=f"mean={errs.mean():.2f}")
    ax.axvline(np.median(errs), color="#c0392b", ls="--", lw=1,
               label=f"median={np.median(errs):.2f}")
    ax.set_xlabel("per-point error [m]"); ax.set_ylabel("count")
    ax.set_title(f"Localization error distribution — epoch {ep}")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def build_streaming_drift_figure(preds, ep, max_gap=None):
    """ONLINE only. Mean error as a function of timesteps-since-last-fix, showing
    dead-reckoning drift between position fixes."""
    gap_err = defaultdict(list)
    for p in preds:
        ctx = np.sort(p["ctx_idx"])
        err = np.sqrt(((p["pred"] - p["true"]) ** 2).sum(-1))
        for ti in range(len(err)):
            prev = ctx[ctx <= ti]
            gap = (ti - prev[-1]) if prev.size else ti  # steps since last fix
            if not p["ctx_mask"][ti]:                    # score non-fix points
                gap_err[int(gap)].append(err[ti])
    if not gap_err:
        return None
    gaps = sorted(gap_err)
    if max_gap is not None:
        gaps = [g for g in gaps if g <= max_gap]
    means = [np.mean(gap_err[g]) for g in gaps]
    counts = [len(gap_err[g]) for g in gaps]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(gaps, means, "o-", color="#d35400")
    ax.set_xlabel("timesteps since last fix"); ax.set_ylabel("mean error [m]")
    ax.set_title(f"Streaming dead-reckoning drift — epoch {ep}")
    ax.grid(alpha=0.3)
    # annotate support
    for g, m, c in zip(gaps, means, counts):
        if c < 5:
            ax.plot(g, m, "o", color="0.7", ms=4)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Degradation scatter (needs the train/val/test pools + splits labels)
# --------------------------------------------------------------------------- #
@t.no_grad()
def _per_geometry_mae(model, conv, ds, device, *, val_ctx, ctx_sample_mode,
                      exclude_ctx_from_target, chunk_size, y_mean, y_std, seed=0):
    """Single-draw per-geometry MAE on a pool (lightweight version of the eval
    routine, scoring target points only)."""
    model.eval()
    ym = y_mean.to(device) if y_mean is not None else None
    ys = y_std.to(device) if y_std is not None else None
    rng = np.random.default_rng(seed)
    err_sum = defaultdict(float); err_cnt = defaultdict(int)
    for i in range(len(ds)):
        X, y, gid, _, sensor_pos = ds[i]
        X = X.unsqueeze(0).to(device); y = y.unsqueeze(0).to(device)
        sp = sensor_pos.unsqueeze(0).to(device) if sensor_pos.numel() > 0 else None
        ppt = X.size(1)
        y_norm = (y - ym) / ys if ym is not None else y
        ctx_idx = _draw_ctx_idx(ppt, val_ctx, ctx_sample_mode, rng, device)
        if conv == "online":
            mean, _ = _predict_online_all(model, X, y_norm, ctx_idx, chunk_size, device)
        else:
            mean, _ = _predict_offline_all(model, conv, X, y_norm, ctx_idx, device, sensor_pos=sp)
        mean = mean * ys + ym if ym is not None else mean
        ctx_mask = t.zeros(ppt, dtype=t.bool); ctx_mask[ctx_idx.cpu()] = True
        tgt = (~ctx_mask if exclude_ctx_from_target else t.ones(ppt, dtype=t.bool))
        if tgt.sum() == 0:
            tgt = t.ones(ppt, dtype=t.bool)
        tgt = tgt.to(device)
        dist = t.sqrt(((mean[:, tgt] - y[:, tgt]) ** 2).sum(-1) + 1e-12)
        err_sum[int(gid)] += float(dist.sum()); err_cnt[int(gid)] += dist.numel()
    return {g: err_sum[g] / err_cnt[g] for g in err_sum}


def build_degradation_figure(model, conv, pools, labels, device, ep, *,
                             val_ctx, ctx_sample_mode, exclude_ctx_from_target,
                             chunk_size, y_mean, y_std):
    """MAE vs sensor-centroid distance, coloured by region. Returns (fig, train_ref)."""
    rows = []
    for nm, ds in pools.items():
        gmae = _per_geometry_mae(model, conv, ds, device, val_ctx=val_ctx,
                                 ctx_sample_mode=ctx_sample_mode,
                                 exclude_ctx_from_target=exclude_ctx_from_target,
                                 chunk_size=chunk_size, y_mean=y_mean, y_std=y_std)
        for gid, m in gmae.items():
            region = labels.get(gid, {}).get("region", "train")
            dist = labels.get(gid, {}).get("dist_from_center", 0.0)
            rows.append({"pool": nm, "gid": gid, "mae": m, "region": region, "dist": dist})
    if not rows:
        return None
    train_ref = float(np.mean([r["mae"] for r in rows if r["region"] == "train"])) \
        if any(r["region"] == "train" for r in rows) else float("nan")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    colors = {"train": "#2980b9", "interp": "#27ae60", "extrap": "#c0392b"}
    for reg in ["train", "interp", "extrap"]:
        pts = [(r["dist"], r["mae"]) for r in rows if r["region"] == reg]
        if pts:
            xs, ys_ = zip(*pts)
            ax.scatter(xs, ys_, c=colors[reg], label=reg, s=60,
                       edgecolor="k", linewidth=0.3)
    if np.isfinite(train_ref):
        ax.axhline(train_ref, color="#2980b9", ls="--", lw=1, alpha=0.6)
    ax.set_xlabel("sensor-centroid distance from training centroids [m]")
    ax.set_ylabel("localization MAE [m]")
    ax.set_title(f"Out-of-position degradation — epoch {ep}")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, train_ref


# --------------------------------------------------------------------------- #
# Top-level dispatcher
# --------------------------------------------------------------------------- #
def log_visualizations(model, conv, *, ep, viz_cfg, val_ds, fixed_idx, device,
                       val_ctx, ctx_sample_mode, exclude_ctx_from_target,
                       chunk_size, y_mean, y_std, deg_pools=None, deg_labels=None):
    """Build the enabled figures and log them to W&B under stable keys (so the
    step slider animates them across epochs). Respects viz_cfg.plots toggles.

    deg_pools / deg_labels are only needed for the degradation scatter."""
    plots = viz_cfg.get("plots", {})
    logs = {}

    # The trajectory / calibration / histogram / drift plots all share one
    # forward pass over the fixed trajectories.
    needs_fixed = any(plots.get(k, False) for k in
                      ("pred_trajectory", "calibration", "error_histogram",
                       "streaming_drift"))
    if needs_fixed and fixed_idx:
        preds = _gather_predictions(
            model, conv, val_ds, fixed_idx, device, val_ctx=val_ctx,
            ctx_sample_mode=ctx_sample_mode,
            exclude_ctx_from_target=exclude_ctx_from_target,
            chunk_size=chunk_size, y_mean=y_mean, y_std=y_std)

        if plots.get("pred_trajectory", False):
            fig = build_trajectory_figure(
                preds, ep, plot_uncertainty=viz_cfg.get("plot_uncertainty", True))
            logs["viz/pred_trajectory"] = wandb.Image(fig); plt.close(fig)

        if plots.get("calibration", False):
            fig = build_calibration_figure(preds, ep)
            if fig is not None:
                logs["viz/calibration"] = wandb.Image(fig); plt.close(fig)

        if plots.get("error_histogram", False):
            fig = build_error_histogram_figure(preds, ep)
            if fig is not None:
                logs["viz/error_histogram"] = wandb.Image(fig); plt.close(fig)

        if plots.get("streaming_drift", False) and conv == "online":
            fig = build_streaming_drift_figure(preds, ep)
            if fig is not None:
                logs["viz/streaming_drift"] = wandb.Image(fig); plt.close(fig)

    if plots.get("degradation_scatter", False) and deg_pools and deg_labels:
        res = build_degradation_figure(
            model, conv, deg_pools, deg_labels, device, ep, val_ctx=val_ctx,
            ctx_sample_mode=ctx_sample_mode,
            exclude_ctx_from_target=exclude_ctx_from_target,
            chunk_size=chunk_size, y_mean=y_mean, y_std=y_std)
        if res is not None:
            fig, train_ref = res
            logs["viz/degradation_scatter"] = wandb.Image(fig); plt.close(fig)

    if logs:
        wandb.log(logs, step=ep)
