"""
eval_suite.py
=============

Paper-grade evaluation suite for the sensor-position-agnostic localization
models. A registry of independent *evaluators* (each emits tidy rows + optional
figure) driven over one or more checkpoints, so every table/figure regenerates
from a single command.

    python scripts/eval/eval_suite.py \
        --models spatial_cnp=<run>/best.pt rcnp=<run>/best.pt cnp=<run>/best.pt \
        --tiers 1,2,3,4 --out-dir output/eval_suite

Multiple checkpoints may share a label (repeat `label=path`) -> treated as seeds
and aggregated (mean +/- std). Convention: split (cnp/anp/spatial_*) and indexed
(ranp/rcnp). online_ranp is out of scope here (streaming eval lives in
compare_offline_online.py). All context sampling is deployment-CAUSAL: context is
a first-prefix, targets are the future complement (see Project_Overview.md).
"""
import argparse, os, json
from collections import defaultdict
import numpy as np
import torch as t
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_np_geometry import build_model, TrajectoryDataset

# Current EVALUATION seed (not a model seed). The driver sets this per slot and
# also calls torch.manual_seed(EVAL_SEED); every evaluator derives its numpy RNG
# from it, so running with 3 eval seeds varies the test-trajectory subsample and
# all perturbations (corruption/shuffle/dropout/permutation) -> metric error bars
# on the SAME trained checkpoint, no retraining.
EVAL_SEED = 0


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_model(path, device):
    ck = t.load(path, map_location=device, weights_only=False)
    cfg = ck.get("config", {})
    spatial_cfg = cfg.get("spatial", None)
    # attn_ffn was added to the checkpoint config only after some runs were
    # trained, so INFER it from the weights when absent: the retrofit is the only
    # thing that creates parameters named '*.ffn.*' in an anp/r_anp model.
    # Rebuilding without it would silently drop the sublayer and then fail the
    # strict load, so infer rather than default.
    sd = ck["model"]
    attn_ffn = cfg.get("attn_ffn", None)
    if attn_ffn is None:
        attn_ffn = any((".ffn." in k or k.endswith(".ffn_norm.weight"))
                       for k in sd.keys()) and ck["model_name"] != "pfn"
    model, conv = build_model(
        ck["model_name"], cfg.get("num_hidden", 128), ck["feat_dim"], ck.get("out_dim", 3),
        rnn_type=cfg.get("rnn_type", "lstm"), rnn_layers=cfg.get("rnn_layers", 1),
        rnn_dropout=cfg.get("rnn_dropout", 0.0), dropout=cfg.get("dropout", 0.1),
        max_context=cfg.get("max_context", 128), spatial_cfg=spatial_cfg,
        n_layers=cfg.get("n_layers", 6), n_heads=cfg.get("n_heads", 8),
        ffn_mult=cfg.get("ffn_mult", 4), readout=cfg.get("readout", "joint"),
        n_cross_layers=cfg.get("n_cross_layers", 2),
        full_cov=cfg.get("full_cov", False), attn_ffn=bool(attn_ffn))
    model.to(device).load_state_dict(sd)
    model.eval()
    if conv == "online":
        raise SystemExit(f"{path}: online_ranp is not covered by eval_suite "
                         "(use scripts/eval/compare_offline_online.py).")
    return {"model": model, "conv": conv,
            "ym": ck["y_mean"].to(device) if ck.get("y_mean") is not None else None,
            "ys": ck["y_std"].to(device) if ck.get("y_std") is not None else None,
            "spatial": spatial_cfg is not None and bool(spatial_cfg.get("enabled", False)),
            "data_dir": cfg.get("data_dir"), "feat_dim": ck["feat_dim"]}


def load_data(data_dir, device, load_cap=20):
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if not os.path.isabs(data_dir):
        data_dir = os.path.normpath(os.path.join(repo, data_dir))
    labels = {}
    sp = os.path.join(data_dir, "splits.json")
    if os.path.exists(sp):
        labels = {int(k): v.get("region", "?") for k, v in json.load(open(sp))["labels"].items()}

    # Load the held-out (interp/extrap) pool grouped by geometry, up to load_cap
    # trajectories per geometry. Per eval seed the driver subsamples cap_traj of
    # these, so different seeds evaluate different test trajectories -> test-set
    # sampling variance shows up as metric error bars.
    by_gid = defaultdict(list)
    for pool in ("val", "test"):
        p = os.path.join(data_dir, f"{pool}_data.pkl")
        if not os.path.exists(p):
            continue
        ds = TrajectoryDataset(p)
        for i in range(len(ds)):
            X, y, gid, th, sensor_pos = ds[i]
            if len(by_gid[int(gid)]) >= load_cap:
                continue
            by_gid[int(gid)].append(
                (X.unsqueeze(0).to(device), y.unsqueeze(0).to(device), int(gid),
                 sensor_pos.unsqueeze(0).to(device) if sensor_pos.numel() > 0 else None))
    by_gid = dict(by_gid)

    # Training-layout centroids for the displacement axis, from the raw sensor-
    # position files (avoids loading the ~51 GB train pkl). A training geometry is
    # any gid NOT labelled interp/extrap.
    import glob, re
    ds_root = os.path.dirname(os.path.dirname(data_dir))
    train_centroids = {}
    for f in glob.glob(os.path.join(ds_root,
            "position_set_*", "channel_option_*", "random", "channel_info",
            "sensor_positions_*.npy")):
        gid = int(re.search(r"position_set_(\d+)", f).group(1))
        if labels.get(gid) in ("interp", "extrap") or gid in train_centroids:
            continue
        P = np.load(f); P = P.T if P.shape[0] == 3 else P
        train_centroids[gid] = t.tensor(P[:, :2].mean(0), dtype=t.float32, device=device)
    tc = t.stack(list(train_centroids.values())) if train_centroids else None
    any_items = next((v[0] for v in by_gid.values() if v), None)
    S = any_items[3].shape[1] if (any_items is not None and any_items[3] is not None) else None
    return {"by_gid": by_gid, "test": [], "train": [], "labels": labels,
            "train_centroids": tc, "n_sensors": S}


def subsample_test(by_gid, cap_traj, eseed):
    """Pick cap_traj trajectories per geometry, seeded by the eval seed."""
    rng = np.random.default_rng([eseed, 9999])
    out = []
    for gid, items in by_gid.items():
        k = min(cap_traj, len(items))
        idx = rng.permutation(len(items))[:k]
        out += [items[i] for i in idx]
    return out


# --------------------------------------------------------------------------- #
# Sensor-layout helpers (feat is tau-major / sensor-minor: feat = tau * S)
# --------------------------------------------------------------------------- #
def _zero_sensors(X, S, drop_idx):
    tau = X.shape[-1] // S
    Xr = X.view(*X.shape[:-1], tau, S).clone()
    Xr[..., drop_idx] = 0.0
    return Xr.reshape(*X.shape)


def _permute_sensors(X, S, perm):
    tau = X.shape[-1] // S
    Xr = X.view(*X.shape[:-1], tau, S)[..., perm]
    return Xr.reshape(*X.shape)


def make_ctx_tgt(ppt, nctx, mode, rng, device):
    nctx = max(1, min(nctx, ppt))
    if mode == "first":
        ctx = t.arange(nctx, device=device)
    elif mode == "single_late":
        ctx = t.arange(nctx, device=device) + (ppt - nctx)
    elif mode in ("random", "scattered"):
        ctx = t.as_tensor(np.sort(rng.permutation(ppt)[:nctx]), device=device)
    else:
        raise ValueError(mode)
    m = t.ones(ppt, dtype=t.bool, device=device); m[ctx] = False
    tgt = t.nonzero(m, as_tuple=False).squeeze(-1)
    if tgt.numel() == 0:
        tgt = t.arange(ppt, device=device)
    return ctx.long(), tgt.long()


# --------------------------------------------------------------------------- #
# Prediction primitive (physical-unit mean at the target indices)
# --------------------------------------------------------------------------- #
@t.no_grad()
def predict(b, X, y, sp, ctx, tgt, *, corrupt_ac=None, ctx_noise=False,
            sensor_mask=None, zero_sensors=None, shuffle_perm=None):
    ym, ys, conv = b["ym"], b["ys"], b["conv"]
    Xin = X
    if zero_sensors is not None and len(zero_sensors):
        Xin = _zero_sensors(Xin, sensor_mask_S(b, sp), zero_sensors)
    ppt = X.size(1)
    cy = y[:, ctx]
    if ctx_noise:
        cy = t.randn_like(cy) * cy.std()
    cy = (cy - ym) / ys if ym is not None else cy
    kw = {}
    if b["spatial"] and sensor_mask is not None:
        kw["sensor_mask"] = sensor_mask
    if shuffle_perm is not None:
        pos = t.empty(ppt, dtype=t.long, device=X.device); pos[shuffle_perm] = t.arange(ppt, device=X.device)
        Xs = Xin[:, shuffle_perm]
        ci, ti = pos[ctx], pos[tgt]
        if conv == "split":
            tx = Xs[:, ti]
            if corrupt_ac is not None:
                tx = _corrupt(tx, corrupt_ac)
            mean = b["model"](Xs[:, ci], cy, tx, None, sensor_pos=sp, **kw)[0]
        else:
            mean = b["model"](Xs, ci, cy, ti, None,
                              sensor_pos=sp if b["spatial"] else None, **kw)[0]
        return mean * ys + ym if ym is not None else mean
    if conv == "split":
        tx = Xin[:, tgt]
        if corrupt_ac is not None:
            tx = _corrupt(tx, corrupt_ac)
        mean = b["model"](Xin[:, ctx], cy, tx, None, sensor_pos=sp, **kw)[0]
    else:
        Xseq = Xin
        if corrupt_ac is not None:
            Xseq = Xin.clone(); Xseq[:, tgt] = _corrupt(Xin[:, tgt], corrupt_ac)
        mean = b["model"](Xseq, ctx, cy, tgt, None,
                          sensor_pos=sp if b["spatial"] else None, **kw)[0]
    return mean * ys + ym if ym is not None else mean


def _corrupt(x, mode):
    if mode == "noise":
        return t.randn_like(x) * x.std()
    if mode == "zero":
        return t.zeros_like(x)
    raise ValueError(mode)


def sensor_mask_S(b, sp):
    return sp.shape[1]


def mae_of(mean, y, tgt):
    return t.sqrt(((mean - y[:, tgt]) ** 2).sum(-1) + 1e-12)[0]  # (n_tgt,)


# --------------------------------------------------------------------------- #
# Evaluators — each returns (rows, figspec_or_None). rows: list of dicts with
# keys model,test,metric,x,value.
# --------------------------------------------------------------------------- #
def _mae_over(items, b, sp_of, nctx=10, mode="first", **pred_kw):
    """Mean per-trajectory MAE over items; returns (overall, per_gid_list)."""
    rng = np.random.default_rng([EVAL_SEED, 0]); errs = []; per = []
    for X, y, gid, sp in items:
        ctx, tgt = make_ctx_tgt(X.size(1), nctx, mode, rng, X.device)
        mean = predict(b, X, y, sp, ctx, tgt, **pred_kw)
        e = mae_of(mean, y, tgt).mean().item()
        errs.append(e); per.append((gid, e))
    return float(np.mean(errs)) if errs else float("nan"), per


def ev_region_mae(models, data, args):
    rows = []
    for lbl, b in models.items():
        # train pool -> 'train'; val+test -> interp/extrap by label
        tr, _ = _mae_over(data["train"], b, None) if data["train"] else (float("nan"), [])
        _, per = _mae_over(data["test"], b, None)
        buck = defaultdict(list)
        for gid, e in per:
            buck[data["labels"].get(gid, "?")].append(e)
        allm = float(np.mean([e for _, e in per])) if per else float("nan")
        rows += [{"model": lbl, "test": "region_mae", "metric": "overall", "x": "", "value": allm},
                 {"model": lbl, "test": "region_mae", "metric": "train", "x": "", "value": tr}]
        for reg in ("interp", "extrap"):
            if buck.get(reg):
                rows.append({"model": lbl, "test": "region_mae", "metric": reg, "x": "",
                             "value": float(np.mean(buck[reg]))})
    return rows, None


def ev_displacement(models, data, args):
    rows = []; series = {}
    tc = data["train_centroids"]
    if tc is None:
        print("[displacement] no training centroids (no sensor_pos); skipping."); return rows, None
    for lbl, b in models.items():
        _, per = _mae_over(data["test"], b, None)
        gid_sp = {gid: sp for X, y, gid, sp in data["test"]}
        xs, ys_ = [], []
        for gid, e in per:
            sp = gid_sp[gid]
            if sp is None: continue
            c = sp[0, :, :2].mean(0)
            d = t.linalg.norm(tc - c[None, :], dim=1).min().item()  # nearest train centroid
            xs.append(d); ys_.append(e)
            rows.append({"model": lbl, "test": "displacement", "metric": "mae_vs_dist",
                         "x": round(d, 1), "value": e})
        series[lbl] = (np.array(xs), np.array(ys_))
    return rows, ("displacement", series)


def ev_acoustic_reliance(models, data, args):
    rows = []
    for lbl, b in models.items():
        base, _ = _mae_over(data["test"], b, None)
        noise, _ = _mae_over(data["test"], b, None, corrupt_ac="noise")
        zero, _ = _mae_over(data["test"], b, None, corrupt_ac="zero")
        rows += [{"model": lbl, "test": "acoustic_reliance", "metric": "base", "x": "", "value": base},
                 {"model": lbl, "test": "acoustic_reliance", "metric": "noise_ratio", "x": "", "value": noise / base},
                 {"model": lbl, "test": "acoustic_reliance", "metric": "zero_ratio", "x": "", "value": zero / base}]
    return rows, None


def ev_shuffle(models, data, args):
    rows = []
    for lbl, b in models.items():
        base, _ = _mae_over(data["test"], b, None)
        rng = np.random.default_rng([EVAL_SEED, 1]); errs = []
        for X, y, gid, sp in data["test"]:
            ctx, tgt = make_ctx_tgt(X.size(1), 10, "first", rng, X.device)
            perm = t.randperm(X.size(1), device=X.device)
            mean = predict(b, X, y, sp, ctx, tgt, shuffle_perm=perm)
            errs.append(mae_of(mean, y, tgt).mean().item())
        shf = float(np.mean(errs))
        rows += [{"model": lbl, "test": "temporal_shuffle", "metric": "base", "x": "", "value": base},
                 {"model": lbl, "test": "temporal_shuffle", "metric": "shuffle_ratio", "x": "", "value": shf / base}]
    return rows, None


def ev_context_label(models, data, args):
    rows = []
    for lbl, b in models.items():
        base, _ = _mae_over(data["test"], b, None)
        corr, _ = _mae_over(data["test"], b, None, ctx_noise=True)
        rows += [{"model": lbl, "test": "context_label", "metric": "base", "x": "", "value": base},
                 {"model": lbl, "test": "context_label", "metric": "corrupt_ratio", "x": "", "value": corr / base}]
    return rows, None


def ev_fix_sparsity(models, data, args):
    rows = []; series = {}
    ks = [10, 5, 2, 1]  # 0 excluded: cross-attention over an empty context set is undefined
    for lbl, b in models.items():
        vals = []
        for k in ks:
            m, _ = _mae_over(data["test"], b, None, nctx=k)
            vals.append(m); rows.append({"model": lbl, "test": "fix_sparsity", "metric": "mae",
                                         "x": k, "value": m})
        series[lbl] = (np.array(ks), np.array(vals))
    return rows, ("fix_sparsity", series)


def ev_drift(models, data, args):
    rows = []; series = {}
    nbins = 4; ppt_ctx = 10
    for lbl, b in models.items():
        binvals = defaultdict(list)
        for X, y, gid, sp in data["test"]:
            ppt = X.size(1)
            ctx = t.arange(ppt_ctx, device=X.device); tgt = t.arange(ppt_ctx, ppt, device=X.device)
            mean = predict(b, X, y, sp, ctx, tgt)
            d = mae_of(mean, y, tgt)
            for j, idx in enumerate(tgt.tolist()):
                binvals[min(nbins - 1, (idx - ppt_ctx) // 10)].append(d[j].item())
        xs = sorted(binvals); vals = [float(np.mean(binvals[bn])) for bn in xs]
        for bn, v in zip(xs, vals):
            rows.append({"model": lbl, "test": "drift", "metric": "mae", "x": bn, "value": v})
        series[lbl] = (np.array(xs), np.array(vals))
    return rows, ("drift", series)


def ev_fix_placement(models, data, args):
    rows = []; N = 5
    modes = {"first": "first", "scattered": "random", "single_early": "first_1", "single_late": "single_late"}
    for lbl, b in models.items():
        for name, spec in modes.items():
            rng = np.random.default_rng([EVAL_SEED, 2])
            if spec == "first_1":
                m, _ = _mae_over(data["test"], b, None, nctx=1, mode="first")
            else:
                m, _ = _mae_over(data["test"], b, None, nctx=N, mode=spec)
            rows.append({"model": lbl, "test": "fix_placement", "metric": name, "x": "", "value": m})
    return rows, None


def ev_permutation(models, data, args):
    rows = []
    for lbl, b in models.items():
        rng = np.random.default_rng([EVAL_SEED, 3]); devs = []
        for X, y, gid, sp in data["test"]:
            if sp is None: continue
            S = sp.shape[1]
            ctx, tgt = make_ctx_tgt(X.size(1), 10, "first", rng, X.device)
            base = predict(b, X, y, sp, ctx, tgt)
            perm = t.randperm(S, device=X.device)
            Xp = _permute_sensors(X, S, perm); spp = sp[:, perm, :]
            permd = predict(b, Xp, y, spp, ctx, tgt)
            devs.append((permd - base).abs().max().item())
        rows.append({"model": lbl, "test": "permutation", "metric": "max_abs_dev", "x": "",
                     "value": float(np.mean(devs)) if devs else float("nan")})
    return rows, None


def ev_sensor_dropout(models, data, args):
    rows = []; series = {}
    for lbl, b in models.items():
        if data["n_sensors"] is None: continue
        S = data["n_sensors"]; ks = [0, 1, 2, 4, min(6, S - 1)]
        rng = np.random.default_rng([EVAL_SEED, 4]); vals = []
        for k in ks:
            errs = []
            for X, y, gid, sp in data["test"]:
                ctx, tgt = make_ctx_tgt(X.size(1), 10, "first", rng, X.device)
                if k == 0:
                    mean = predict(b, X, y, sp, ctx, tgt)
                elif b["spatial"]:
                    drop = t.as_tensor(rng.permutation(S)[:k], device=X.device)
                    m = t.ones(1, S, dtype=t.bool, device=X.device); m[0, drop] = False
                    mean = predict(b, X, y, sp, ctx, tgt, sensor_mask=m)
                else:  # flat / recurrent: no principled masking -> zero-fill the dropped slices
                    drop = list(rng.permutation(S)[:k])
                    mean = predict(b, X, y, sp, ctx, tgt, zero_sensors=drop)
                errs.append(mae_of(mean, y, tgt).mean().item())
            vals.append(float(np.mean(errs)))
            rows.append({"model": lbl, "test": "sensor_dropout", "metric": "mae", "x": k,
                         "value": vals[-1]})
        series[lbl] = (np.array(ks), np.array(vals))
    return rows, ("sensor_dropout", series)


TIERS = {
    1: [ev_region_mae, ev_displacement],
    2: [ev_acoustic_reliance, ev_shuffle, ev_context_label],
    3: [ev_fix_sparsity, ev_drift, ev_fix_placement],
    4: [ev_permutation, ev_sensor_dropout],
}


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def save_fig(kind, series, out_dir):
    # series: {label: (xs, ys, es)} where es is per-point std across seeds (0 if 1 seed).
    plt.figure(figsize=(6, 4))
    if kind == "displacement":
        for lbl, (xs, ys_, _es) in series.items():
            if len(xs) == 0: continue
            order = np.argsort(xs); xs, ys_ = xs[order], ys_[order]
            nb = min(8, max(2, len(xs) // 20))                 # bin for a readable curve
            edges = np.quantile(xs, np.linspace(0, 1, nb + 1))
            bx, by = [], []
            for i in range(nb):
                m = (xs >= edges[i]) & (xs <= edges[i + 1])
                if m.any(): bx.append(xs[m].mean()); by.append(ys_[m].mean())
            plt.plot(bx, by, marker="o", label=lbl)
        plt.xlabel("array displacement from training hull [m]"); plt.ylabel("MAE [m]")
        plt.title("Displacement robustness")
    elif kind in ("fix_sparsity", "sensor_dropout", "drift"):
        for lbl, (xs, ys_, es) in series.items():
            if es is not None and np.any(es > 0):
                plt.errorbar(xs, ys_, yerr=es, marker="o", capsize=3, label=lbl)
            else:
                plt.plot(xs, ys_, marker="o", label=lbl)
        xl = {"fix_sparsity": "context fixes", "sensor_dropout": "sensors dropped",
              "drift": "horizon bin (steps from last fix)"}[kind]
        plt.xlabel(xl); plt.ylabel("MAE [m]"); plt.title(kind.replace("_", " ") + " (mean ± std over seeds)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    p = os.path.join(out_dir, f"{kind}.png"); plt.savefig(p, dpi=120); plt.close()
    return p


def _series_from_agg(agg_rows, test):
    """Rebuild {model: (xs, ys_mean, ys_std)} for a curve test from aggregated rows."""
    per = defaultdict(list)
    for r in agg_rows:
        if r["test"] != test:
            continue
        try:
            xv = float(r["x"])
        except (ValueError, TypeError):
            continue
        per[r["model"]].append((xv, r["mean"], r["std"]))
    out = {}
    for m, items in per.items():
        items.sort()
        out[m] = (np.array([i[0] for i in items]),
                  np.array([i[1] for i in items]),
                  np.array([i[2] for i in items]))
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="label=path.pt entries (one trained checkpoint per model)")
    ap.add_argument("--tiers", default="1,2,3,4")
    ap.add_argument("--data-dir", default=None, help="override; else from first checkpoint")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--eval-seeds", default="0,1,2",
                    help="EVAL seeds: each re-samples the test trajectories and the "
                         "perturbations on the SAME checkpoints -> metric error bars.")
    ap.add_argument("--cap-traj", type=int, default=6,
                    help="trajectories sampled per geometry per eval seed")
    ap.add_argument("--device", default="cuda" if t.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = t.device(args.device)

    # {label: [all trained-seed checkpoints]}. Error bars fold BOTH training-seed
    # spread (however many exist per model) and eval-seed spread together.
    models = {}
    for spec in args.models:
        lbl, path = spec.split("=", 1)
        models.setdefault(lbl, []).append(load_model(path, device))
    max_ms = max(len(bs) for bs in models.values())
    print(f"[eval_suite] model seeds: {{lbl: n}} = { {k: len(v) for k, v in models.items()} }")

    data_dir = args.data_dir or models[next(iter(models))][0]["data_dir"]
    data = load_data(data_dir, device, load_cap=max(15, args.cap_traj * 3))
    print(f"[eval_suite] data_dir={data_dir}  held-out geoms={len(data['by_gid'])}  "
          f"n_sensors={data['n_sensors']}")

    eval_seeds = [int(x) for x in args.eval_seeds.split(",") if x.strip()]
    tiers = [int(x) for x in args.tiers.split(",") if x.strip()]
    # For each eval seed (fresh test subsample + perturbation seed) evaluate every
    # available model-seed checkpoint on that SAME subsample, then aggregate over
    # all (eval_seed x model_seed) samples -> mean +/- std with whatever seeds exist.
    global EVAL_SEED
    per_seed_rows = []; fig_tests = []
    for es in eval_seeds:
        EVAL_SEED = es
        t.manual_seed(es)
        data["test"] = subsample_test(data["by_gid"], args.cap_traj, es)
        for mi in range(max_ms):
            flat_mi = {lbl: bs[mi] for lbl, bs in models.items() if len(bs) > mi}
            if not flat_mi:
                continue
            print(f"[eval_suite] --- eval seed {es}, model-seed slot {mi}: "
                  f"{len(data['test'])} items, {list(flat_mi)} ---")
            for tier in tiers:
                for ev in TIERS[tier]:
                    rows, fig = ev(flat_mi, data, args)
                    for r in rows:
                        r["seed"] = f"{es}.{mi}"
                    per_seed_rows += rows
                    if fig is not None and fig[0] not in fig_tests:
                        fig_tests.append(fig[0])

    # aggregate mean/std/n over seeds, keyed by (model, test, metric, x)
    buckets = defaultdict(list)
    for r in per_seed_rows:
        buckets[(r["model"], r["test"], r["metric"], r["x"])].append(r["value"])
    agg_rows = [{"model": m, "test": tst, "metric": met, "x": x,
                 "mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
                for (m, tst, met, x), v in buckets.items()]

    # raw (per-seed) + aggregated CSVs
    raw_csv = os.path.join(args.out_dir, "results_per_seed.csv")
    with open(raw_csv, "w") as f:
        f.write("model,test,metric,x,seed,value\n")
        for r in per_seed_rows:
            f.write(f"{r['model']},{r['test']},{r['metric']},{r['x']},{r['seed']},{r['value']:.4f}\n")
    csv_path = os.path.join(args.out_dir, "results.csv")
    with open(csv_path, "w") as f:
        f.write("model,test,metric,x,mean,std,n\n")
        for r in agg_rows:
            f.write(f"{r['model']},{r['test']},{r['metric']},{r['x']},"
                    f"{r['mean']:.4f},{r['std']:.4f},{r['n']}\n")
    print(f"[eval_suite] wrote {len(agg_rows)} aggregated rows -> {csv_path} "
          f"(+ per-seed raw -> {raw_csv})")

    # figures from aggregated means (± std band)
    for kind in fig_tests:
        p = save_fig(kind, _series_from_agg(agg_rows, kind), args.out_dir)
        print(f"  figure -> {p}")

    # compact console summary (mean ± std), grouped by test
    by_test = defaultdict(list)
    for r in agg_rows:
        by_test[r["test"]].append(r)
    for test, rs in by_test.items():
        print(f"\n=== {test} ===")
        by_model = defaultdict(dict)
        for r in rs:
            key = f"{r['metric']}" + (f"@{r['x']}" if r["x"] != "" else "")
            by_model[r["model"]][key] = (r["mean"], r["std"])
        for mdl, kv in by_model.items():
            print(f"  {mdl:20s} " + "  ".join(
                f"{k}={m:.2f}±{s:.1f}" for k, (m, s) in kv.items()))


if __name__ == "__main__":
    main()
