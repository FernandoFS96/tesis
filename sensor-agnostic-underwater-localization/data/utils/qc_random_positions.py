"""
qc_random_positions.py
======================

Quality-control and characterization suite for the 20 random-position datasets produced by ``data_generator_random_positions.py``.

It serves two purposes at once (both are reported):

  A. EXPERIMENT-VALIDITY CHECKS  -- assertions that the generated data actually implements the intended design, i.e. that *sensor geometry is the only factor that varies between position-sets*:
       * trajectories are byte-identical across all 20 position-sets (per theta);
       * the 20 sensor layouts are genuinely different from one another;
       * shapes / feature dimension match the spec (df -> Lf*n_sensors);
       * no NaN/Inf, no degenerate (all-zero) sensor channels;
       * the receiver depth (z-row) is constant within a layout.

  B. PAPER CHARACTERIZATION  -- figures and a summary table suitable for the "data generation / preparation" section:
       * sensor-layout scatter across all sets (geometry coverage);
       * pairwise layout-distance matrix (how different the 20 sets are);
       * per-set / per-theta feature statistics (level, dynamic range);
       * feature-distribution overlap across sets (is the displacement actually changing the acoustic features?);
       * trajectory ensemble plot (the shared paths);
       * coverage of the operational area by sensors vs. by the source.

Everything reads the REAL output layout written by the generator:

    <data-root>/
      _manifest.pkl
      position_set_00/
        channel_option_<theta>/random/
          filtered_data/filtered_data.npy        (tau, ppt, n_traj, n_sensors)
          trajectory/trajectories.npy            (3, n_traj, ppt)
          channel_info/sensor_positions_<theta>.npy   (3, n_sensors)
      ...

USAGE
-----
    python qc_random_positions.py --data-root ./data_random_positions

    # restrict to a subset while iterating
    python qc_random_positions.py --data-root ./data_random_positions \
        --thetas 0.0,0.3 --max-sets 5

    # control where figures/tables go
    python qc_random_positions.py --data-root ./data_random_positions \
        --out-dir ./qc_report

OUTPUTS (under --out-dir, default <data-root>/qc_report)
    figures/*.png   publication-ready figures (300 dpi)
    qc_summary.csv  per-(set,theta) numeric summary
    qc_report.txt   human-readable pass/fail log of the validity checks

The script never modifies the dataset. It exits with code 1 if any HARD validity check fails, so it can be used in CI / as a pre-training gate.
"""

import os
import re
import sys
import glob
import pickle
import argparse
from itertools import combinations

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# --------------------------------------------------------------------------- #
# Discovery helpers
# --------------------------------------------------------------------------- #
def find_position_sets(data_root):
    sets = sorted(glob.glob(os.path.join(data_root, "position_set_*")))
    sets = [s for s in sets if os.path.isdir(s)]
    return sets


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


# --------------------------------------------------------------------------- #
# Report object
# --------------------------------------------------------------------------- #
class Report:
    def __init__(self):
        self.lines = []
        self.hard_failures = 0
        self.soft_warnings = 0

    def ok(self, msg):
        self.lines.append(f"[ PASS ] {msg}")

    def fail(self, msg):
        self.lines.append(f"[ FAIL ] {msg}")
        self.hard_failures += 1

    def warn(self, msg):
        self.lines.append(f"[ WARN ] {msg}")
        self.soft_warnings += 1

    def info(self, msg):
        self.lines.append(f"         {msg}")

    def dump(self, path):
        with open(path, "w") as f:
            f.write("\n".join(self.lines) + "\n")
        print("\n".join(self.lines))
        print(f"\nHard failures: {self.hard_failures} | Warnings: {self.soft_warnings}")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_arrays(set_dir, theta):
    p = paths_for(set_dir, theta)
    for k, v in p.items():
        if not os.path.exists(v):
            raise FileNotFoundError(f"missing {k}: {v}")
    filtered = np.load(p["filtered"])   # (tau, ppt, n_traj, n_sensors)
    traj = np.load(p["traj"])           # (3, n_traj, ppt)
    pos = np.load(p["pos"])             # (3, n_sensors)
    return filtered, traj, pos


def reshape_features(filtered):
    """Match data_process_topology.reshape_input_data:
       (tau, ppt, n_traj, n_sensors) -> (n_traj, ppt, tau*n_sensors)."""
    tau, ppt, n_traj, n_sensors = filtered.shape
    return filtered.transpose(2, 1, 0, 3).reshape(n_traj, ppt, tau * n_sensors)


# --------------------------------------------------------------------------- #
# A. VALIDITY CHECKS
# --------------------------------------------------------------------------- #
def check_validity(data_root, set_dirs, thetas, rep, n_sensors_expected,
                   df, fmin, fmax, report):
    report.info("=" * 64)
    report.info("A. EXPERIMENT-VALIDITY CHECKS")
    report.info("=" * 64)

    # Expected feature length Lf from df.
    Lf_expected = len(np.arange(fmin, fmax, df))
    # range_m includes the endpoint when it lands within one step.
    if (np.arange(fmin, fmax, df)[-1] + df) <= fmax:
        Lf_expected += 1

    # ---- per-(set,theta) basic integrity, collect for cross-checks ----
    traj_hash = {}   # theta -> {set_idx: hash}
    layouts = {}     # theta -> {set_idx: pos (3,n_sensors)}
    feat_shapes = set()

    for si, sd in enumerate(set_dirs):
        for th in thetas:
            try:
                filtered, traj, pos = load_arrays(sd, th)
            except FileNotFoundError as e:
                report.fail(f"set {si} theta {th}: {e}")
                continue

            # finite values
            if not np.all(np.isfinite(filtered)):
                report.fail(f"set {si} theta {th}: filtered_data contains NaN/Inf")
            if not np.all(np.isfinite(traj)):
                report.fail(f"set {si} theta {th}: trajectories contain NaN/Inf")
            if not np.all(np.isfinite(pos)):
                report.fail(f"set {si} theta {th}: sensor_positions contain NaN/Inf")

            tau, ppt, n_traj, n_sensors = filtered.shape

            # sensor count
            if n_sensors != n_sensors_expected:
                report.fail(f"set {si} theta {th}: n_sensors={n_sensors} "
                            f"!= expected {n_sensors_expected}")
            if pos.shape != (3, n_sensors):
                report.fail(f"set {si} theta {th}: sensor_positions shape "
                            f"{pos.shape} != (3,{n_sensors})")

            # feature length (tau) matches df
            if tau != Lf_expected:
                report.fail(f"set {si} theta {th}: tau={tau} != Lf({df}Hz)="
                            f"{Lf_expected}")

            feat_shapes.add((ppt, tau * n_sensors))

            # no all-zero (dead) sensor channels
            per_sensor_energy = np.sum(np.abs(filtered) ** 2, axis=(0, 1, 2))
            dead = np.where(per_sensor_energy == 0)[0]
            if dead.size:
                report.fail(f"set {si} theta {th}: dead sensor(s) {dead.tolist()}")

            # receiver depth constant within a layout
            if np.unique(np.round(pos[2], 6)).size != 1:
                report.warn(f"set {si} theta {th}: z-row of sensors not constant")

            traj_hash.setdefault(th, {})[si] = hash(traj.tobytes())
            layouts.setdefault(th, {})[si] = pos

    # ---- (1) trajectories identical across sets, per theta ----
    for th in thetas:
        hashes = traj_hash.get(th, {})
        uniq = set(hashes.values())
        if len(hashes) <= 1:
            report.warn(f"theta {th}: <=1 set available, cannot cross-check "
                        f"trajectory sharing")
        elif len(uniq) == 1:
            report.ok(f"theta {th}: trajectories byte-identical across "
                      f"{len(hashes)} position-sets (shared-trajectory design OK)")
        else:
            report.fail(f"theta {th}: trajectories DIFFER across sets "
                        f"({len(uniq)} distinct) -- geometry is not the only "
                        f"varying factor!")

    # ---- (2) layouts genuinely different across sets ----
    for th in thetas:
        lay = layouts.get(th, {})
        if len(lay) <= 1:
            continue
        identical_pairs = []
        for (a, pa), (b, pb) in combinations(sorted(lay.items()), 2):
            if np.allclose(pa, pb):
                identical_pairs.append((a, b))
        if identical_pairs:
            report.fail(f"theta {th}: identical sensor layouts for set-pairs "
                        f"{identical_pairs} -- displacement not applied!")
        else:
            report.ok(f"theta {th}: all {len(lay)} sensor layouts are distinct")

    # ---- (3) single feature shape across the whole corpus ----
    if len(feat_shapes) == 1:
        ppt, feat = next(iter(feat_shapes))
        report.ok(f"consistent feature tensor across corpus: "
                  f"(n_traj, ppt={ppt}, feat_dim={feat})")
    else:
        report.fail(f"inconsistent feature shapes across corpus: {feat_shapes}")

    # ---- (4) layouts identical across thetas within a set (positions should
    #          not depend on theta, only on the layout seed) ----
    if len(thetas) > 1:
        for si, sd in enumerate(set_dirs):
            per_theta = []
            for th in thetas:
                if th in layouts and si in layouts[th]:
                    per_theta.append(layouts[th][si])
            if len(per_theta) > 1:
                allsame = all(np.allclose(per_theta[0], p) for p in per_theta[1:])
                if not allsame:
                    report.warn(f"set {si}: sensor layout varies across theta "
                                f"(expected identical within a set)")
        report.ok("checked layout consistency across thetas within each set")

    return layouts, Lf_expected


# --------------------------------------------------------------------------- #
# B. CHARACTERIZATION + FIGURES
# --------------------------------------------------------------------------- #
def fig_layouts_scatter(layouts, theta_ref, traj_ref, fig_dir):
    """All sensor layouts overlaid, plus the shared source trajectory cloud."""
    lay = layouts[theta_ref]
    fig, ax = plt.subplots(figsize=(7, 6))

    # source trajectory cloud (shared) in light grey
    xs = traj_ref[0].ravel()
    ys = traj_ref[1].ravel()
    ax.scatter(xs, ys, s=2, c="0.8", alpha=0.4, label="source positions (shared)")

    cmap = matplotlib.colormaps['viridis']
    n = len(lay)
    for i, (si, pos) in enumerate(sorted(lay.items())):
        ax.scatter(pos[0], pos[1], s=55, color=cmap(i / max(n - 1, 1)),
                   edgecolor="k", linewidth=0.3,
                   label=f"set {si}" if n <= 8 else None)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Sensor layouts across {n} position-sets (theta={theta_ref})")
    ax.set_aspect("equal", adjustable="datalim")
    if n <= 8:
        ax.legend(loc="best", fontsize=8)
    else:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=n - 1))
        cb = fig.colorbar(sm, ax=ax)
        cb.set_label("position-set index")
    fig.tight_layout()
    p = os.path.join(fig_dir, "layouts_scatter.png")
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return p


def fig_layout_distance_matrix(layouts, theta_ref, fig_dir):
    """Pairwise layout distance: mean over sensors of per-sensor Euclidean
    displacement after optimal index matching is NOT done -- sensors are not
    identity-matched across sets, so we use a permutation-invariant geometry
    descriptor: sorted pairwise inter-sensor distance signature."""
    lay = layouts[theta_ref]
    items = sorted(lay.items())
    idxs = [si for si, _ in items]

    def signature(pos):
        xy = pos[:2].T  # (n_sensors, 2)
        d = []
        for a, b in combinations(range(xy.shape[0]), 2):
            d.append(np.linalg.norm(xy[a] - xy[b]))
        return np.sort(np.array(d))

    sigs = [signature(p) for _, p in items]
    n = len(sigs)
    D = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            D[a, b] = np.linalg.norm(sigs[a] - sigs[b])

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(D, cmap="magma")
    ax.set_xticks(range(n)); ax.set_xticklabels(idxs, fontsize=7, rotation=90)
    ax.set_yticks(range(n)); ax.set_yticklabels(idxs, fontsize=7)
    ax.set_xlabel("position-set"); ax.set_ylabel("position-set")
    ax.set_title("Layout dissimilarity\n(||sorted inter-sensor distance||)")
    fig.colorbar(im, ax=ax, label="geometry distance")
    fig.tight_layout()
    p = os.path.join(fig_dir, "layout_distance_matrix.png")
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return p, D


def fig_feature_stats(summary_rows, fig_dir):
    """Per-set feature level + dynamic range, grouped by theta."""
    thetas = sorted(set(r["theta"] for r in summary_rows), key=float)
    sets = sorted(set(r["set"] for r in summary_rows))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    cmap = matplotlib.colormaps['coolwarm'] #plt.cm.coolwarm
    for ti, th in enumerate(thetas):
        means = [next(r["feat_mean_abs"] for r in summary_rows
                      if r["set"] == s and r["theta"] == th) for s in sets]
        rng = [next(r["feat_dyn_range_db"] for r in summary_rows
                    if r["set"] == s and r["theta"] == th) for s in sets]
        c = cmap(ti / max(len(thetas) - 1, 1))
        axes[0].plot(sets, means, marker="o", color=c, label=f"theta={th}")
        axes[1].plot(sets, rng, marker="s", color=c, label=f"theta={th}")

    axes[0].set_xlabel("position-set"); axes[0].set_ylabel("mean |feature|")
    axes[0].set_title("Feature level across sets")
    axes[1].set_xlabel("position-set"); axes[1].set_ylabel("dynamic range [dB]")
    axes[1].set_title("Feature dynamic range across sets")
    for a in axes:
        a.legend(fontsize=7, ncol=2)
        a.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(fig_dir, "feature_stats_across_sets.png")
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return p


def fig_feature_distributions(data_root, set_dirs, theta_ref, fig_dir, max_sets=6):
    """Overlaid histograms of (log) feature magnitude for several sets at a fixed
    theta -- visual evidence that sensor displacement shifts the acoustic
    features (i.e. the task really changes)."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    cmap = matplotlib.colormaps['viridis'] #plt.cm.viridis
    sel = set_dirs[:max_sets]
    for i, sd in enumerate(sel):
        filtered, _, _ = load_arrays(sd, theta_ref)
        mag = np.abs(filtered).ravel()
        mag = mag[mag > 0]
        ax.hist(np.log10(mag), bins=120, histtype="step", density=True,
                color=cmap(i / max(len(sel) - 1, 1)),
                label=os.path.basename(sd).replace("position_set_", "set "))
    ax.set_xlabel("log10 |feature|")
    ax.set_ylabel("density")
    ax.set_title(f"Feature-magnitude distribution by position-set "
                 f"(theta={theta_ref})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(fig_dir, "feature_distributions.png")
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return p


def fig_trajectories(traj_ref, theta_ref, fig_dir, max_traj=40):
    """The shared trajectory ensemble (top-down)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    n = min(max_traj, traj_ref.shape[1])
    for k in range(n):
        ax.plot(traj_ref[0, k], traj_ref[1, k], lw=0.8, alpha=0.6)
    ax.scatter(traj_ref[0, :n, 0], traj_ref[1, :n, 0], s=12, c="k",
               zorder=3, label="start")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"Shared source trajectories (theta={theta_ref}, "
                 f"{n} of {traj_ref.shape[1]} shown)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(fig_dir, "shared_trajectories.png")
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return p


def fig_between_vs_within(data_root, set_dirs, theta_ref, fig_dir, max_sets=10):
    """Quantify how much displacement changes features: compare the spread of
    per-set MEAN feature vectors (between-set) against the within-set spread.
    A between/within ratio > 1 means geometry meaningfully changes the data."""
    means = []
    within = []
    sel = set_dirs[:max_sets]
    for sd in sel:
        feats = reshape_features(load_arrays(sd, theta_ref)[0])  # (n_traj,ppt,F)
        flat = np.abs(feats).reshape(-1, feats.shape[-1])        # (samples, F)
        means.append(flat.mean(axis=0))
        within.append(flat.std(axis=0).mean())
    means = np.stack(means)                       # (sets, F)
    between = means.std(axis=0).mean()
    within = float(np.mean(within))
    ratio = between / (within + 1e-12)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.bar(["between-set\n(geometry)", "within-set\n(trajectory)"],
           [between, within], color=["#c0392b", "#2980b9"])
    ax.set_ylabel("mean feature-magnitude spread")
    ax.set_title(f"Geometry vs. trajectory variation\n"
                 f"between/within ratio = {ratio:.2f}  (theta={theta_ref})")
    fig.tight_layout()
    p = os.path.join(fig_dir, "between_vs_within.png")
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return p, ratio


# --------------------------------------------------------------------------- #
# Summary table
# --------------------------------------------------------------------------- #
def build_summary(set_dirs, thetas):
    rows = []
    for si, sd in enumerate(set_dirs):
        for th in thetas:
            filtered, traj, pos = load_arrays(sd, th)
            mag = np.abs(filtered)
            nz = mag[mag > 0]
            dyn_db = 20 * np.log10(nz.max() / nz.min()) if nz.size else float("nan")
            xy = pos[:2].T
            # array aperture = max pairwise sensor distance
            aperture = max(np.linalg.norm(xy[a] - xy[b])
                           for a, b in combinations(range(xy.shape[0]), 2))
            rows.append({
                "set": si,
                "theta": th,
                "tau": filtered.shape[0],
                "ppt": filtered.shape[1],
                "n_traj": filtered.shape[2],
                "n_sensors": filtered.shape[3],
                "feat_dim": filtered.shape[0] * filtered.shape[3],
                "feat_mean_abs": float(mag.mean()),
                "feat_dyn_range_db": float(dyn_db),
                "array_aperture_m": float(aperture),
                "sensor_centroid_x": float(pos[0].mean()),
                "sensor_centroid_y": float(pos[1].mean()),
            })
    return rows


def write_csv(rows, path):
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="QC / characterization for the "
                                             "random-position datasets.")
    ap.add_argument("--data-root", required=True,
                    help="Root dir written by data_generator_random_positions.py")
    ap.add_argument("--out-dir", default=None,
                    help="Where to write the report (default: <data-root>/qc_report)")
    ap.add_argument("--thetas", default=None,
                    help="Comma list to restrict thetas (default: all found)")
    ap.add_argument("--max-sets", type=int, default=None,
                    help="Only inspect the first K position-sets")
    ap.add_argument("--rep", type=int, default=1, help="rep used at generation")
    ap.add_argument("--n-sensors", type=int, default=10)
    ap.add_argument("--df", type=float, default=100.0)
    ap.add_argument("--fmin", type=float, default=10000.0)
    ap.add_argument("--fmax", type=float, default=20000.0)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.data_root, "qc_report")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    set_dirs = find_position_sets(args.data_root)
    if not set_dirs:
        print(f"ERROR: no position_set_* under {args.data_root}", file=sys.stderr)
        sys.exit(1)
    if args.max_sets:
        set_dirs = set_dirs[:args.max_sets]

    thetas = find_thetas(set_dirs[0])
    if args.thetas:
        want = [t.strip() for t in args.thetas.split(",")]
        thetas = [t for t in thetas if t in want]
    if not thetas:
        print("ERROR: no thetas found / matched", file=sys.stderr)
        sys.exit(1)

    report = Report()
    report.info(f"data-root : {args.data_root}")
    report.info(f"position-sets inspected : {len(set_dirs)}")
    report.info(f"thetas : {thetas}")

    # Optional manifest cross-check
    man_path = os.path.join(args.data_root, "_manifest.pkl")
    if os.path.exists(man_path):
        with open(man_path, "rb") as f:
            man = pickle.load(f)
        report.ok(f"manifest found (master_seed={man.get('master_seed')}, "
                  f"declared sets={man.get('n_position_sets')})")
    else:
        report.warn("no _manifest.pkl found -- skipping seed cross-check")

    # ---- A. validity ----
    layouts, Lf = check_validity(
        args.data_root, set_dirs, thetas, args.rep, args.n_sensors,
        args.df, args.fmin, args.fmax, report)

    # ---- B. characterization ----
    report.info("=" * 64)
    report.info("B. CHARACTERIZATION (figures + table)")
    report.info("=" * 64)

    theta_ref = thetas[0]
    _, traj_ref, _ = load_arrays(set_dirs[0], theta_ref)

    rows = build_summary(set_dirs, thetas)
    write_csv(rows, os.path.join(out_dir, "qc_summary.csv"))
    report.ok(f"wrote qc_summary.csv ({len(rows)} rows)")

    figs = []
    figs.append(fig_layouts_scatter(layouts, theta_ref, traj_ref, fig_dir))
    dm_path, _ = fig_layout_distance_matrix(layouts, theta_ref, fig_dir)
    figs.append(dm_path)
    figs.append(fig_feature_stats(rows, fig_dir))
    figs.append(fig_feature_distributions(args.data_root, set_dirs, theta_ref, fig_dir))
    figs.append(fig_trajectories(traj_ref, theta_ref, fig_dir))
    bw_path, ratio = fig_between_vs_within(args.data_root, set_dirs, theta_ref, fig_dir)
    figs.append(bw_path)
    for fpath in figs:
        report.ok(f"figure: {os.path.relpath(fpath, out_dir)}")

    report.info(f"between/within feature-variation ratio (theta={theta_ref}): "
                f"{ratio:.2f}")
    if ratio < 1.0:
        report.warn("between/within ratio < 1: sensor displacement changes the "
                    "features LESS than trajectory variation -- the robustness "
                    "task may be weak. Inspect feature_distributions.png.")
    else:
        report.ok("sensor displacement induces a clear between-set feature shift "
                  "(ratio >= 1): the robustness task is well-posed.")

    report.dump(os.path.join(out_dir, "qc_report.txt"))
    print(f"\nReport written to: {out_dir}")
    if report.hard_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
