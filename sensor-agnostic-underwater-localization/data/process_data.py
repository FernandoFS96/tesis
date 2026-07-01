"""
data_process_random_positions.py
================================

Preprocessing for BOTH studies. Defaults come from the ``preprocess`` block of
``config/data_pipeline.yaml``; any CLI flag overrides the config.

MODES (preprocess.mode, or --mode)
----------------------------------
* ``topology`` -- THREE-TOPOLOGY study (acoustic_data_generator.py). Reads the
  ``<data_root>/<method>/<topology>/channel_option_<theta>/`` tree and produces
  ONE training-ready dataset PER topology (ellipsoidal / random / aligned). All
  channel options (thetas) of a topology are POOLED, and the train/val/test
  split is by TRAJECTORY INDEX (default ratio 70/20/10, rescaled to whatever
  n_traj the data actually contains) -- the same trajectory indices for every
  theta, so no trajectory leaks across splits.
  Output (dict schema, matching geometry mode so the trainer reads it as-is):
      <save-dir>/topology_<name>/{train,val,test}_data.pkl
        -> list of {"X":(ppt,feat_dim), "y":(ppt,3), "theta":float, "topology":str}
      <save-dir>/topology_<name>/metadata.pkl

  The modes below operate on the POSITION-SET study (random_position_generator.py),
  reading the ``position_set_XX/channel_option_<theta>/random/`` tree:

* ``legacy``  -- the ORIGINAL within-geometry split, preserved from
  ``data_process_topology.py``: for each (theta, geometry) it splits the 100
  trajectories 70/20/10 into train/val/test. Every geometry appears in all three
  splits (val/test are unseen TRAJECTORIES of SEEN layouts), so a no-spatial-
  encoder baseline can generalize -- this reproduces the "normal model like
  before". Output uses the dict schema (matching geometry/topology modes) so the
  trainer reads it unchanged:
      <save-dir>/within_geometry_split/{train,val,test}_data.pkl
        -> list of {"X":(ppt,feat_dim), "y":(ppt,3), "theta":float, "topology":str}
      <save-dir>/within_geometry_split/metadata.pkl  (*_thetas / *_topologies)

* ``geometry`` -- the NEW geometry-level split for measuring robustness to
  UNSEEN sensor layouts. The 20 position-sets are partitioned into disjoint
  train / val / test GEOMETRY pools (default 12/4/4). A geometry never crosses
  pools, so the val/test sets genuinely measure generalization to layouts the
  model never saw. Held-out geometries are further labelled INTERPOLATION
  (inside the training-centroid convex hull) or EXTRAPOLATION (outside), giving
  a built-in degradation-vs-displacement axis. Each emitted sample is enriched
  with the sensor positions and a geometry id so the downstream spatial encoder
  can consume them.

WHY THE SPLIT AXIS CHANGED
--------------------------
The legacy split asks "given THIS geometry, generalize to new trajectories /
channel conditions". The new question is "generalize to geometries never seen in
training". That can only be answered by holding out whole geometries, which is
what ``geometry`` mode does. This mirrors the ID/OOD task-family design of the
magnetic-localization paper, with sensor layout as the held-out factor.

FEATURE FORMAT (geometry mode)
------------------------------
Per your choice, the acoustic part keeps the legacy sequence shape
``(ppt, feat_dim)`` = ``(50, 1010)`` for df=100, IDENTICAL to legacy, so the same
acoustic backbone can consume it unchanged. The geometry is supplied ALONGSIDE,
not folded into X:

    sample = {
        "X":          float32 (ppt, feat_dim),     # acoustic, as in legacy
        "y":          float32 (ppt, 3),            # target source coords
        "sensor_pos": float32 (n_sensors, 3),      # this geometry's layout (x,y,z)
        "geometry_id": int,                        # position-set index
        "theta":       float,
    }

For the spatial-encoder model we can reshape X to ``(ppt, tau, n_sensors)`` on
the fly (tau = feat_dim // n_sensors = 101) and attend over the n_sensors axis
using ``sensor_pos``; the processor records ``tau`` and ``n_sensors`` in metadata
so this reshape is unambiguous.

OUTPUTS
-------
legacy mode (within-geometry split):
    <save-dir>/within_geometry_split/{train,val,test}_data.pkl, metadata.pkl

geometry mode:
    <save-dir>/geometry_split/
        train_data.pkl   list of sample dicts (see above)
        val_data.pkl
        test_data.pkl
        metadata.pkl     thetas, geometry_ids, interp/extrap labels, tau, n_sensors
        splits.json      FROZEN record: which position-sets are train/val/test,
                         each held-out set's interp/extrap label + centroid/dist.
                         Version-control this -- it is the experiment contract.

USAGE
-----
Topology task (default; data_root from the preprocess block = ./data/topology_task):
    python data_process_random_positions.py --mode topology

Within-geometry (RANDOM task; point --data-root at the <method>_<mode> folder):
    python data_process_random_positions.py \
        --data-root ./data/random_task/spiral_shared --mode legacy

Held-out geometry split (RANDOM task, auto interp/extrap, 12/4/4):
    python data_process_random_positions.py \
        --data-root ./data/random_task/spiral_shared --mode geometry \
        --train-geoms 12 --val-geoms 4 --test-geoms 4

Reproduce a previously frozen split:
    python data_process_random_positions.py \
        --data-root ./data/random_task/spiral_shared --mode geometry \
        --splits-file ./data/random_task/spiral_shared/processed/geometry_split/splits.json
"""

import os
import re
import json
import glob
import pickle
import argparse

import numpy as np
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# Default data-pipeline config (config/data_pipeline.yaml at repo root).
DEFAULT_TRAJ_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "data_pipeline.yaml")

TOPOLOGIES = ("ellipsoidal", "random", "aligned")


def load_traj_config(path=None):
    """Load the trajectory/preprocess config from YAML via OmegaConf."""
    return OmegaConf.load(path or DEFAULT_TRAJ_CONFIG_PATH)


# --------------------------------------------------------------------------- #
# Reshapes (identical semantics to data_process_topology.py)
# --------------------------------------------------------------------------- #
def reshape_input_data(data):
    """(tau, ppt, n_traj, n_sensors) -> (n_traj, ppt, tau*n_sensors)."""
    num_time_points, num_points_per_traj, num_trajs, num_sensors = data.shape
    return data.transpose(2, 1, 0, 3).reshape(
        num_trajs, num_points_per_traj, num_time_points * num_sensors)


def reshape_output_data(trajectories):
    """(3, n_traj, ppt) -> (n_traj, ppt, 3)."""
    return trajectories.transpose(1, 2, 0)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def find_position_sets(data_root):
    return sorted(d for d in glob.glob(os.path.join(data_root, "position_set_*"))
                  if os.path.isdir(d))


def set_index(set_dir):
    m = re.search(r"position_set_(\d+)$", set_dir)
    return int(m.group(1)) #type: ignore[no-any-return]


def find_thetas(set_dir):
    opts = []
    for d in glob.glob(os.path.join(set_dir, "channel_option_*")):
        m = re.search(r"channel_option_([0-9.]+)$", d)
        if m and os.path.isdir(os.path.join(d, "random")):
            opts.append(m.group(1))
    return sorted(opts, key=float)


def paths_for(set_dir, theta):
    base = os.path.join(set_dir, f"channel_option_{theta}", "random")
    return {
        "filtered": os.path.join(base, "filtered_data", "filtered_data.npy"),
        "traj":     os.path.join(base, "trajectory", "trajectories.npy"),
        "pos":      os.path.join(base, "channel_info", f"sensor_positions_{theta}.npy"),
    }


# =========================================================================== #
# LEGACY MODE  (within-geometry 70/20/10, preserved from data_process_topology)
# =========================================================================== #
def process_and_save_legacy(input_paths, output_paths, save_dir,
                            topology_labels, seed=18):
    loaded_train, loaded_val, loaded_test = [], [], []
    train_thetas, val_thetas, test_thetas = [], [], []
    train_topo, val_topo, test_topo = [], [], []

    print("Within-geometry split (70/20/10 within each geometry)...")
    for i, (inp, out) in tqdm(enumerate(zip(input_paths, output_paths)),
                              total=len(input_paths), leave=False):
        X = reshape_input_data(np.load(inp)).astype(np.float32)
        y = reshape_output_data(np.load(out)).astype(np.float32)

        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            X, y, train_size=0.7, random_state=18, shuffle=True)
        X_val, X_te, y_val, y_te = train_test_split(
            X_tmp, y_tmp, test_size=1/3, random_state=19, shuffle=True)

        # Emit dict samples (same schema as geometry / topology modes) so the
        # dict-based trainer (TrajectoryDataset) consumes them unchanged.
        theta, topology = topology_labels[i]
        for j in range(X_tr.shape[0]):
            loaded_train.append({"X": X_tr[j], "y": y_tr[j], "theta": float(theta), "topology": topology})
            train_thetas.append(theta); train_topo.append(topology)
        for j in range(X_val.shape[0]):
            loaded_val.append({"X": X_val[j], "y": y_val[j], "theta": float(theta), "topology": topology})
            val_thetas.append(theta); val_topo.append(topology)
        for j in range(X_te.shape[0]):
            loaded_test.append({"X": X_te[j], "y": y_te[j], "theta": float(theta), "topology": topology})
            test_thetas.append(theta); test_topo.append(topology)

    idx = np.random.permutation(len(loaded_train))
    loaded_train = [loaded_train[i] for i in idx]
    train_thetas = [train_thetas[i] for i in idx]
    train_topo = [train_topo[i] for i in idx]

    os.makedirs(save_dir, exist_ok=True)
    for fn, obj in [("train_data.pkl", loaded_train), ("val_data.pkl", loaded_val),
                    ("test_data.pkl", loaded_test)]:
        with open(os.path.join(save_dir, fn), "wb") as f:
            pickle.dump(obj, f)
    with open(os.path.join(save_dir, "metadata.pkl"), "wb") as f:
        pickle.dump({
            "train_thetas": train_thetas, "val_thetas": val_thetas, "test_thetas": test_thetas,
            "train_topologies": train_topo, "val_topologies": val_topo, "test_topologies": test_topo,
        }, f)
    print(f"  within-geometry saved -> {save_dir}  "
          f"(train={len(loaded_train)} val={len(loaded_val)} test={len(loaded_test)})")


def run_legacy(data_root, set_dirs, thetas, save_base):
    """Within-geometry split pooled over ALL geometries (each geometry
    contributes to train/val/test internally), matching the original
    data_process_topology.py behaviour."""
    input_paths, output_paths, topo_labels = [], [], []
    for sd in set_dirs:
        for th in thetas:
            p = paths_for(sd, th)
            if os.path.exists(p["filtered"]) and os.path.exists(p["traj"]):
                input_paths.append(p["filtered"]); output_paths.append(p["traj"])
                topo_labels.append((float(th), "random"))
    save_dir = os.path.join(save_base, "within_geometry_split")
    process_and_save_legacy(input_paths, output_paths, save_dir, topo_labels)


# =========================================================================== #
# GEOMETRY MODE  (held-out layouts, interp/extrap)
# =========================================================================== #
def geometry_descriptor(set_dir, thetas):
    """Centroid (x, y) and aperture from this geometry's sensor positions.
    Positions are theta-independent; use the first available theta."""
    for th in thetas:
        pos_path = paths_for(set_dir, th)["pos"]
        if os.path.exists(pos_path):
            pos = np.load(pos_path)            # (3, n_sensors)
            xy = pos[:2].T                     # (n_sensors, 2)
            centroid = xy.mean(0)
            from itertools import combinations
            ap = max(np.linalg.norm(xy[a] - xy[b])
                     for a, b in combinations(range(xy.shape[0]), 2))
            return centroid, float(ap), pos
    raise FileNotFoundError(f"no sensor_positions in {set_dir}")


def assign_geometry_split(set_dirs, thetas, n_train, n_val, n_test, seed=0):
    """Deterministically choose train/val/test GEOMETRY pools and label held-out
    sets interp/extrap. Strategy: central geometries (near the centroid-of-
    centroids) become interpolation hold-outs; peripheral ones become
    extrapolation hold-outs; the rest train. Verified against the training hull."""
    from scipy.spatial import Delaunay

    ids, centroids = [], []
    for sd in set_dirs:
        c, ap, _ = geometry_descriptor(sd, thetas)
        ids.append(set_index(sd)); centroids.append(c)
    ids = np.array(ids); C = np.array(centroids)
    center = C.mean(0)
    dist = np.linalg.norm(C - center, axis=1)
    order = np.argsort(dist)                    # central first

    n_hold = n_val + n_test
    n_interp = n_hold // 2
    n_extrap = n_hold - n_interp
    central = list(order[:n_interp])            # indices into ids
    peripheral = list(order[::-1][:n_extrap])

    # Interleave interp/extrap into val then test so both get a balanced mix.
    interp_ids = [int(ids[i]) for i in central]
    extrap_ids = [int(ids[i]) for i in peripheral]
    half_i = n_interp // 2
    half_e = n_extrap // 2
    val_ids = interp_ids[:half_i] + extrap_ids[:half_e]
    test_ids = interp_ids[half_i:] + extrap_ids[half_e:]
    # Trim/pad to requested sizes if odd.
    held = set(val_ids + test_ids)
    val_ids = val_ids[:n_val]; test_ids = test_ids[:n_test]
    held = set(val_ids + test_ids)
    train_ids = [int(s) for s in ids if int(s) not in held]
    train_ids = train_ids[:n_train] if n_train else train_ids

    # Label held-out sets vs the TRAIN convex hull.
    train_pts = C[[list(ids).index(s) for s in train_ids]]
    hull = Delaunay(train_pts)
    def region(sid):
        c = C[list(ids).index(sid)]
        return "interp" if hull.find_simplex(c) >= 0 else "extrap"

    labels = {}
    for sid in val_ids + test_ids:
        c = C[list(ids).index(sid)]
        labels[sid] = {"region": region(sid),
                       "centroid": [float(c[0]), float(c[1])],
                       "dist_from_center": float(np.linalg.norm(c - center))}

    return {
        "train": sorted(train_ids), "val": sorted(val_ids), "test": sorted(test_ids),
        "center_of_centroids": [float(center[0]), float(center[1])],
        "labels": labels,
    }


def build_geometry_samples(set_dirs_by_id, ids, thetas):
    """Assemble enriched sample dicts for a list of geometry ids."""
    samples = []
    geom_ids, theta_list, regions = [], [], []
    tau_seen, ns_seen = set(), set()
    for sid in ids:
        sd = set_dirs_by_id[sid]
        for th in thetas:
            p = paths_for(sd, th)
            if not (os.path.exists(p["filtered"]) and os.path.exists(p["traj"])
                    and os.path.exists(p["pos"])):
                continue
            filtered = np.load(p["filtered"])      # (tau, ppt, n_traj, n_sensors)
            tau, ppt, n_traj, n_sensors = filtered.shape
            tau_seen.add(tau); ns_seen.add(n_sensors)
            X = reshape_input_data(filtered).astype(np.float32)   # (n_traj, ppt, tau*ns)
            y = reshape_output_data(np.load(p["traj"])).astype(np.float32)  # (n_traj, ppt, 3)
            pos = np.load(p["pos"]).T.astype(np.float32)          # (n_sensors, 3)
            for j in range(X.shape[0]):
                samples.append({
                    "X": X[j], "y": y[j], "sensor_pos": pos,
                    "geometry_id": int(sid), "theta": float(th),
                })
                geom_ids.append(int(sid)); theta_list.append(float(th))
    assert len(tau_seen) == 1 and len(ns_seen) == 1, \
        f"inconsistent tau/n_sensors: {tau_seen} {ns_seen}"
    return samples, geom_ids, theta_list, tau_seen.pop(), ns_seen.pop()


def run_geometry(data_root, set_dirs, thetas, save_base,
                 n_train, n_val, n_test, splits_file=None, shuffle_seed=42):
    set_dirs_by_id = {set_index(sd): sd for sd in set_dirs}

    if splits_file and os.path.exists(splits_file):
        with open(splits_file) as f:
            split = json.load(f)
        print(f"Loaded frozen split from {splits_file}")
    else:
        split = assign_geometry_split(set_dirs, thetas, n_train, n_val, n_test)

    print(f"  TRAIN geoms ({len(split['train'])}): {split['train']}")
    print(f"  VAL   geoms ({len(split['val'])}): {split['val']}")
    print(f"  TEST  geoms ({len(split['test'])}): {split['test']}")
    for sid, info in sorted(split["labels"].items(), key=lambda kv: int(kv[0])):
        pool = "val" if int(sid) in split["val"] else "test"
        print(f"    set {int(sid):2d} [{pool:4s}] {info['region']:6s} "
              f"dist={info['dist_from_center']:.2f}")

    save_dir = os.path.join(save_base, "geometry_split")
    os.makedirs(save_dir, exist_ok=True)

    meta = {"thetas": thetas, "split": split}
    for pool in ("train", "val", "test"):
        samples, gids, ths, tau, ns = build_geometry_samples(
            set_dirs_by_id, split[pool], thetas)
        if pool == "train":
            rng = np.random.default_rng(shuffle_seed)
            perm = rng.permutation(len(samples))
            samples = [samples[i] for i in perm]
            gids = [gids[i] for i in perm]; ths = [ths[i] for i in perm]
        with open(os.path.join(save_dir, f"{pool}_data.pkl"), "wb") as f:
            pickle.dump(samples, f)
        meta[f"{pool}_geometry_ids"] = gids
        meta[f"{pool}_thetas"] = ths
        meta["tau"] = int(tau); meta["n_sensors"] = int(ns)
        meta["feat_dim"] = int(tau * ns)
        print(f"  {pool:5s}: {len(samples)} samples "
              f"(X=(ppt,{tau*ns}), sensor_pos=({ns},3))")

    with open(os.path.join(save_dir, "metadata.pkl"), "wb") as f:
        pickle.dump(meta, f)
    with open(os.path.join(save_dir, "splits.json"), "w") as f:
        json.dump(split, f, indent=2)
    print(f"  geometry split saved -> {save_dir}")
    print(f"  frozen split contract -> {os.path.join(save_dir, 'splits.json')}")


# =========================================================================== #
# TOPOLOGY MODE  (three-topology study: per-topology, split by trajectory index)
# =========================================================================== #
def find_topologies(data_root, method):
    """Ordered list of (name, topology_dir) for the three-topology layout.
    Layout groups by METHOD first: <data_root>/<method>/<topology>/."""
    mroot = os.path.join(data_root, method)
    return [(t, os.path.join(mroot, t)) for t in TOPOLOGIES
            if os.path.isdir(os.path.join(mroot, t))]


def topo_paths_for(topo_dir, theta):
    # topo_dir already includes the method level (<data_root>/<method>/<topology>).
    base = os.path.join(topo_dir, f"channel_option_{theta}")
    return {
        "filtered": os.path.join(base, "filtered_data", "filtered_data.npy"),
        "traj":     os.path.join(base, "trajectory", "trajectories.npy"),
    }


def find_topo_thetas(topo_dir):
    opts = []
    for d in glob.glob(os.path.join(topo_dir, "channel_option_*")):
        m = re.search(r"channel_option_([0-9.]+)$", d)
        if m and os.path.isdir(os.path.join(d, "trajectory")):
            opts.append(m.group(1))
    return sorted(opts, key=float)


def trajectory_index_split(n_traj, n_train, n_val, n_test, seed=0):
    """A FIXED, reproducible partition of the trajectory indices [0, n_traj).

    (n_train, n_val, n_test) are treated as RATIOS and rescaled to the actual
    n_traj found in the data, so the same config auto-scales to any trajectory
    count: 70/20/10 -> 70/20/10 at n_traj=100, 140/40/20 at n_traj=200, or
    105/30/15 at n_traj=150. ALL trajectories are used (val/test are rounded and
    any remainder goes to train, so the three splits always sum to n_traj).

    The same indices are applied to every channel option (theta), so all thetas
    of a given physical trajectory land in the SAME split -- this is what keeps
    val/test from leaking trajectories the model trained on. Returns three
    sorted index lists (train, val, test)."""
    total = n_train + n_val + n_test
    if total <= 0:
        raise ValueError(
            f"split weights must be positive, got {n_train}/{n_val}/{n_test}")
    n_va = int(round(n_traj * n_val / total))
    n_te = int(round(n_traj * n_test / total))
    n_tr = n_traj - n_va - n_te
    if n_tr < 0:
        raise ValueError(
            f"rescaled split {n_tr}/{n_va}/{n_te} invalid for n_traj={n_traj}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_traj)
    tr = sorted(int(i) for i in perm[:n_tr])
    va = sorted(int(i) for i in perm[n_tr:n_tr + n_va])
    te = sorted(int(i) for i in perm[n_tr + n_va:n_tr + n_va + n_te])
    return tr, va, te


def run_topology(data_root, method, save_base, thetas_filter,
                 n_train, n_val, n_test, split_seed=0, shuffle_seed=42):
    """Build one training-ready dataset PER topology. Within each topology all
    channel options are pooled and the 70/20/10 split is by trajectory index.

    Output (per topology), dict schema matching geometry mode so the trainer's
    TrajectoryDataset consumes it unchanged:
        <save-base>/topology_<name>/
            train_data.pkl / val_data.pkl / test_data.pkl
                -> list of {"X":(ppt,feat_dim), "y":(ppt,3), "theta":float,
                            "topology":str}
            metadata.pkl   -> thetas, per-split thetas/topologies, the frozen
                              trajectory-index split, tau / n_sensors / feat_dim
    """
    topos = find_topologies(data_root, method)
    if not topos:
        raise SystemExit(
            f"No topology folders {TOPOLOGIES} under {data_root}/{method} "
            f"(is this the three-topology layout? expected "
            f"<data_root>/{method}/<topology>/channel_option_*/).")

    for name, tdir in topos:
        thetas = find_topo_thetas(tdir)
        if thetas_filter:
            thetas = [t for t in thetas if t in thetas_filter]
        if not thetas:
            print(f"  [skip] topology '{name}': no thetas found")
            continue

        # n_traj / shapes from the first theta (shared across thetas).
        first = topo_paths_for(tdir, thetas[0])
        probe = np.load(first["filtered"])              # (tau, ppt, n_traj, n_sensors)
        tau, ppt, n_traj, n_sensors = probe.shape
        tr_idx, va_idx, te_idx = trajectory_index_split(
            n_traj, n_train, n_val, n_test, seed=split_seed)
        split_idx = {"train": tr_idx, "val": va_idx, "test": te_idx}

        pools = {p: [] for p in ("train", "val", "test")}        # sample dicts
        pool_thetas = {p: [] for p in ("train", "val", "test")}

        for th in thetas:
            p = topo_paths_for(tdir, th)
            if not (os.path.exists(p["filtered"]) and os.path.exists(p["traj"])):
                print(f"  [warn] topology '{name}' theta {th}: missing files, skipped")
                continue
            X = reshape_input_data(np.load(p["filtered"])).astype(np.float32)   # (n_traj, ppt, feat)
            y = reshape_output_data(np.load(p["traj"])).astype(np.float32)      # (n_traj, ppt, 3)
            for pool, idxs in split_idx.items():
                for j in idxs:
                    pools[pool].append({
                        "X": X[j], "y": y[j],
                        "theta": float(th), "topology": name,
                    })
                    pool_thetas[pool].append(float(th))

        # Shuffle the train pool (keep each sample paired with its theta record).
        rng = np.random.default_rng(shuffle_seed)
        perm = rng.permutation(len(pools["train"]))
        pools["train"] = [pools["train"][i] for i in perm]
        pool_thetas["train"] = [pool_thetas["train"][i] for i in perm]

        save_dir = os.path.join(save_base, f"topology_{name}")
        os.makedirs(save_dir, exist_ok=True)
        for pool in ("train", "val", "test"):
            with open(os.path.join(save_dir, f"{pool}_data.pkl"), "wb") as f:
                pickle.dump(pools[pool], f)
        meta = {
            "study": "topology",
            "topology": name,
            "method": method,
            "thetas": thetas,
            "n_traj": int(n_traj),
            "traj_split": {"train": tr_idx, "val": va_idx, "test": te_idx},
            "split_seed": int(split_seed),
            "tau": int(tau), "n_sensors": int(n_sensors),
            "feat_dim": int(tau * n_sensors),
            "train_thetas": pool_thetas["train"],
            "val_thetas": pool_thetas["val"],
            "test_thetas": pool_thetas["test"],
            "train_topologies": [name] * len(pools["train"]),
            "val_topologies": [name] * len(pools["val"]),
            "test_topologies": [name] * len(pools["test"]),
        }
        with open(os.path.join(save_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(meta, f)
        print(f"  topology '{name}': split {len(tr_idx)}/{len(va_idx)}/{len(te_idx)} "
              f"trajectories x {len(thetas)} thetas -> "
              f"train={len(pools['train'])} val={len(pools['val'])} "
              f"test={len(pools['test'])} samples "
              f"(X=(ppt={ppt},{tau*n_sensors})) -> {save_dir}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Preprocess underwater-localization data. Defaults are read "
                    "from the `preprocess` block of config/data_pipeline.yaml; "
                    "any CLI flag overrides the config. Modes: topology (three-"
                    "topology study, split by trajectory index) | geometry | "
                    "legacy | all (position-set study).")
    ap.add_argument("--config", default=None,
                    help="Path to the YAML config (default: config/data_pipeline.yaml).")
    ap.add_argument("--mode", choices=["topology", "legacy", "geometry", "all"],
                    default=None, help="Override preprocess.mode.")
    ap.add_argument("--data-root", default=None, help="Override preprocess.data_root.")
    ap.add_argument("--method", default=None,
                    help="Trajectory-method subfolder for topology mode "
                         "(<method>/<topology>/...). Override preprocess.method.")
    ap.add_argument("--save-dir", default=None,
                    help="Output base (default: <data-root>/processed).")
    ap.add_argument("--thetas", default=None,
                    help="Comma list to restrict thetas (default: all found).")
    # topology-mode trajectory-index split.
    ap.add_argument("--traj-train", type=int, default=None)
    ap.add_argument("--traj-val", type=int, default=None)
    ap.add_argument("--traj-test", type=int, default=None)
    ap.add_argument("--traj-seed", type=int, default=None)
    # geometry-mode held-out-layout split.
    ap.add_argument("--train-geoms", type=int, default=None)
    ap.add_argument("--val-geoms", type=int, default=None)
    ap.add_argument("--test-geoms", type=int, default=None)
    ap.add_argument("--splits-file", default=None,
                    help="Reuse a frozen splits.json instead of recomputing.")
    args = ap.parse_args()

    # ---- Resolve every knob: CLI flag (if given) > config > hard default. ----
    cfg = load_traj_config(args.config)
    pp = cfg.get("preprocess", {}) or {} #type: ignore[assignment]
    topo_sp = pp.get("topology_split", {}) or {}
    geom_sp = pp.get("geometry_split", {}) or {}

    mode = args.mode or str(pp.get("mode", "topology"))
    method = args.method or str(pp.get("method", cfg.get("method", "hermite"))) #type: ignore[assignment]
    data_root = args.data_root or str(pp.get("data_root", "./data/topology_task"))
    cfg_save = pp.get("save_dir", None)
    explicit_save = args.save_dir or (str(cfg_save) if cfg_save else None)
    # Topology raw lives under <data_root>/<method>/, so its processed output
    # defaults under that same method subfolder -> spiral and hermite never
    # overwrite each other. Position-set modes already encode <method>_<mode> in
    # data_root, so <data_root>/processed is already method-separated for them.
    if mode == "topology":
        save_base = explicit_save or os.path.join(data_root, method, "processed")
    else:
        save_base = explicit_save or os.path.join(data_root, "processed")

    # thetas filter (CLI comma list > config list > all-found).
    if args.thetas:
        thetas_filter = [t.strip() for t in args.thetas.split(",")]
    elif pp.get("thetas", None):
        thetas_filter = [str(t) for t in pp.get("thetas")] #type: ignore[assignment]
    else:
        thetas_filter = None

    n_train = args.traj_train if args.traj_train is not None else int(topo_sp.get("n_train", 70))
    n_val = args.traj_val if args.traj_val is not None else int(topo_sp.get("n_val", 20))
    n_test = args.traj_test if args.traj_test is not None else int(topo_sp.get("n_test", 10))
    split_seed = args.traj_seed if args.traj_seed is not None else int(topo_sp.get("seed", 0))

    n_tr_g = args.train_geoms if args.train_geoms is not None else int(geom_sp.get("n_train", 12))
    n_va_g = args.val_geoms if args.val_geoms is not None else int(geom_sp.get("n_val", 4))
    n_te_g = args.test_geoms if args.test_geoms is not None else int(geom_sp.get("n_test", 4))

    print("=" * 64)
    print(f"Preprocessing  mode={mode}")
    print(f"  data-root : {data_root}")
    print(f"  save-base : {save_base}")
    print("=" * 64)

    if mode == "topology":
        print("\n--- TOPOLOGY (per-topology, split by trajectory index) ---")
        print(f"  method={method}  split(traj ratio)={n_train}/{n_val}/{n_test} "
              f"(rescaled to n_traj)  seed={split_seed}  "
              f"thetas={thetas_filter or 'all'}")
        run_topology(data_root, method, save_base, thetas_filter,
                     n_train, n_val, n_test, split_seed=split_seed)
        print("\nDone.")
        return

    # ---- position-set study (legacy / geometry) ----
    set_dirs = find_position_sets(data_root)
    if not set_dirs:
        raise SystemExit(f"No position_set_* under {data_root}")
    thetas = find_thetas(set_dirs[0])
    if thetas_filter:
        thetas = [t for t in thetas if t in thetas_filter]
    print(f"  geometries: {len(set_dirs)} | thetas: {thetas}")

    if mode in ("legacy", "all"):
        print("\n--- LEGACY (within-geometry 70/20/10) ---")
        run_legacy(data_root, set_dirs, thetas, save_base)

    if mode in ("geometry", "all"):
        print("\n--- GEOMETRY (held-out layouts, interp/extrap) ---")
        run_geometry(data_root, set_dirs, thetas, save_base,
                     n_tr_g, n_va_g, n_te_g, splits_file=args.splits_file)

    print("\nDone.")


if __name__ == "__main__":
    main()
