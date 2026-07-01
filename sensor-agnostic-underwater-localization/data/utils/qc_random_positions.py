"""
qc_random_positions.py
======================

Quality-control and characterization suite for the 20 random-position datasets produced by ``data_generator_random_positions.py``.

It serves two purposes at once (both are reported):

  A. EXPERIMENT-VALIDITY CHECKS  -- assertions that the generated data actually implements the intended design. The trajectory invariant depends on the generation mode (read from _manifest.pkl, overridable with --shared-/--distinct-trajectories):
       * SHARED mode (sensor-displacement study): trajectories are byte-identical across all position-sets (per theta) -- geometry is the only varying factor;
       * DISTINCT mode: trajectories DIFFER across all position-sets (each set has its own per-set-seeded ensemble);
       * the sensor layouts are genuinely different from one another (both modes);
       * shapes / feature dimension match the spec (df -> Lf*n_sensors);
       * no NaN/Inf, no degenerate (all-zero) sensor channels;
       * the receiver depth (z-row) is constant within a layout.

  B. PAPER CHARACTERIZATION  -- figures and a summary table suitable for the "data generation / preparation" section:
       * sensor-layout scatter across all sets (geometry coverage);
       * pairwise layout-distance matrix (how different the 20 sets are);
       * per-set / per-theta feature statistics (level, dynamic range);
       * feature-distribution overlap across sets (is the displacement actually changing the acoustic features?);
       * trajectory ensemble plot -- the single shared ensemble (shared mode) or several sets' own ensembles overlaid (distinct mode);
       * coverage of the operational area by sensors vs. by the source.

Note: in DISTINCT mode trajectories ALSO differ per set, so the between-set
feature variation is confounded (geometry + trajectory) and is reported
descriptively rather than as a geometry-only well-posedness gate.

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
    python qc_random_positions.py --data-root ./data_random_positions/hermite_shared

    # restrict to a subset while iterating
    python qc_random_positions.py --data-root ./data_random_positions/hermite_shared \
        --thetas 0.0,0.3 --max-sets 5

    # control where figures/tables go
    python qc_random_positions.py --data-root ./data_random_positions/hermite_shared \
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
#
# This QC tool understands TWO dataset layouts and auto-detects which one a
# given --data-root holds. In both the comparison axis is a set of "groups":
#
#   * POSITION-SET study (random_position_generator.py): groups are the
#     position_set_XX directories (varying sensor layouts of the 'random'
#     topology); per group ->
#         <root>/position_set_XX/channel_option_<theta>/random/...
#
#   * THREE-TOPOLOGY study (acoustic_data_generator.py): groups are the three
#     sensor topologies (ellipsoidal / random / aligned); per group ->
#         <root>/<topology>/<method>/channel_option_<theta>/...
#
# The same validity checks apply to both: trajectories must be SHARED across
# the groups (geometry is the only varying factor) and the sensor layouts must
# DIFFER across groups. Only the directory walk and the labels change.
# --------------------------------------------------------------------------- #
TOPOLOGIES = ("ellipsoidal", "random", "aligned")


def detect_mode(data_root):
    """Return 'position_sets' or 'topologies' by inspecting the directory tree."""
    if sorted(glob.glob(os.path.join(data_root, "position_set_*"))):
        return "position_sets"
    if any(os.path.isdir(os.path.join(data_root, t)) for t in TOPOLOGIES):
        return "topologies"
    return "position_sets"  # default; main() errors out if nothing is found


def find_groups(data_root, mode):
    """Ordered list of (label, group_dir) for the detected mode."""
    if mode == "topologies":
        groups = []
        for t in TOPOLOGIES:
            d = os.path.join(data_root, t)
            if os.path.isdir(d):
                groups.append((t, d))
        return groups
    sets = sorted(glob.glob(os.path.join(data_root, "position_set_*")))
    out = []
    for s in sets:
        if os.path.isdir(s):
            out.append((os.path.basename(s).replace("position_set_", "set "), s))
    return out


def option_base(group_dir, theta, mode, method):
    """Directory holding trajectory/ filtered_data/ channel_info/ for one
    (group, theta), for the detected layout."""
    if mode == "topologies":
        return os.path.join(group_dir, method, f"channel_option_{theta}")
    return os.path.join(group_dir, f"channel_option_{theta}", "random")


def find_thetas(group_dir, mode, method):
    opts = []
    search_root = os.path.join(group_dir, method) if mode == "topologies" else group_dir
    for d in glob.glob(os.path.join(search_root, "channel_option_*")):
        m = re.search(r"channel_option_([0-9.]+)$", d)
        if m and os.path.isdir(os.path.join(option_base(group_dir, m.group(1),
                                                         mode, method), "trajectory")):
            opts.append(m.group(1))
    return sorted(opts, key=float)


def paths_for(group_dir, theta, mode, method):
    base = option_base(group_dir, theta, mode, method)
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
def load_arrays(group_dir, theta, mode, method):
    p = paths_for(group_dir, theta, mode, method)
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


def traj_shape_metrics(traj):
    """Permutation-free descriptors of a trajectory ENSEMBLE's spatial
    configuration, averaged over its trajectories. Used to quantify how the
    per-set trajectory configuration differs between position-sets (in distinct
    mode these vary across sets; in shared mode they are identical by design).

    traj: (3, n_traj, ppt). Returns a dict of scalar metrics.
    """
    xy = traj[:2]                                  # (2, n_traj, ppt)
    steps = np.diff(xy, axis=2)                     # (2, n_traj, ppt-1)
    step_len = np.linalg.norm(steps, axis=0)        # (n_traj, ppt-1)
    path_len = step_len.sum(axis=1)                 # (n_traj,) total path length
    net_disp = np.linalg.norm(xy[:, :, -1] - xy[:, :, 0], axis=0)  # (n_traj,)
    radius = np.linalg.norm(xy, axis=0)             # (n_traj, ppt) dist from origin
    radial_max = radius.max(axis=1)                 # (n_traj,) farthest reach

    # Mean absolute turning angle between consecutive step vectors [deg].
    a = steps[:, :, :-1]
    b = steps[:, :, 1:]
    dot = (a * b).sum(axis=0)
    cross = a[0] * b[1] - a[1] * b[0]
    turn = np.abs(np.arctan2(cross, dot))           # (n_traj, ppt-2) radians
    mean_turn_deg = np.degrees(turn).mean() if turn.size else float("nan")

    # Start-point spread across trajectories (how dispersed the launch points are).
    start = xy[:, :, 0]                             # (2, n_traj)
    start_spread = float(np.mean(start.std(axis=1)))

    return {
        "traj_path_len_mean": float(path_len.mean()),
        "traj_net_disp_mean": float(net_disp.mean()),
        "traj_radial_max_mean": float(radial_max.mean()),
        "traj_mean_turn_deg": float(mean_turn_deg),
        "traj_start_spread": start_spread,
    }


# --------------------------------------------------------------------------- #
# A. VALIDITY CHECKS
# --------------------------------------------------------------------------- #
def check_validity(data_root, set_dirs, labels, thetas, rep, n_sensors_expected,
                   df, fmin, fmax, report, mode, method,
                   distinct_trajectories=False, unit="position-set"):
    report.info("=" * 64)
    report.info("A. EXPERIMENT-VALIDITY CHECKS")
    report.info("=" * 64)

    # Expected feature length Lf from df.
    Lf_expected = len(np.arange(fmin, fmax, df))
    # range_m includes the endpoint when it lands within one step.
    if (np.arange(fmin, fmax, df)[-1] + df) <= fmax:
        Lf_expected += 1

    # ---- per-(group,theta) basic integrity, collect for cross-checks ----
    traj_hash = {}   # theta -> {group_idx: hash}
    layouts = {}     # theta -> {group_idx: pos (3,n_sensors)}
    feat_shapes = set()

    for si, sd in enumerate(set_dirs):
        gl = labels[si]
        for th in thetas:
            try:
                filtered, traj, pos = load_arrays(sd, th, mode, method)
            except FileNotFoundError as e:
                report.fail(f"{unit} {gl} theta {th}: {e}")
                continue

            # finite values
            if not np.all(np.isfinite(filtered)):
                report.fail(f"{unit} {gl} theta {th}: filtered_data contains NaN/Inf")
            if not np.all(np.isfinite(traj)):
                report.fail(f"{unit} {gl} theta {th}: trajectories contain NaN/Inf")
            if not np.all(np.isfinite(pos)):
                report.fail(f"{unit} {gl} theta {th}: sensor_positions contain NaN/Inf")

            tau, ppt, n_traj, n_sensors = filtered.shape

            # sensor count
            if n_sensors != n_sensors_expected:
                report.fail(f"{unit} {gl} theta {th}: n_sensors={n_sensors} "
                            f"!= expected {n_sensors_expected}")
            if pos.shape != (3, n_sensors):
                report.fail(f"{unit} {gl} theta {th}: sensor_positions shape "
                            f"{pos.shape} != (3,{n_sensors})")

            # feature length (tau) matches df
            if tau != Lf_expected:
                report.fail(f"{unit} {gl} theta {th}: tau={tau} != Lf({df}Hz)="
                            f"{Lf_expected}")

            feat_shapes.add((ppt, tau * n_sensors))

            # no all-zero (dead) sensor channels
            per_sensor_energy = np.sum(np.abs(filtered) ** 2, axis=(0, 1, 2))
            dead = np.where(per_sensor_energy == 0)[0]
            if dead.size:
                report.fail(f"{unit} {gl} theta {th}: dead sensor(s) {dead.tolist()}")

            # receiver depth constant within a layout
            if np.unique(np.round(pos[2], 6)).size != 1:
                report.warn(f"{unit} {gl} theta {th}: z-row of sensors not constant")

            traj_hash.setdefault(th, {})[si] = hash(traj.tobytes())
            layouts.setdefault(th, {})[si] = pos

    # ---- (1) cross-group trajectory invariant, per theta ----
    # The expected invariant depends on the generation mode:
    #   shared   -> trajectories must be byte-IDENTICAL across all groups
    #               (sensor geometry is the only varying factor). This is the
    #               ALWAYS-expected invariant for the three-topology study.
    #   distinct -> trajectories must DIFFER across all groups (position-set
    #               study with per-set trajectories); any collision means the
    #               per-set seeding silently did not take effect.
    for th in thetas:
        hashes = traj_hash.get(th, {})
        uniq = set(hashes.values())
        if len(hashes) <= 1:
            report.warn(f"theta {th}: <=1 {unit} available, cannot cross-check "
                        f"trajectory {'distinctness' if distinct_trajectories else 'sharing'}")
        elif distinct_trajectories:
            if len(uniq) == len(hashes):
                report.ok(f"theta {th}: trajectories distinct across "
                          f"{len(hashes)} {unit}s (per-set design OK)")
            else:
                report.fail(f"theta {th}: only {len(uniq)} distinct trajectory "
                            f"ensembles across {len(hashes)} {unit}s -- per-set "
                            f"seeding did not take; {unit}s share trajectories!")
        elif len(uniq) == 1:
            report.ok(f"theta {th}: trajectories byte-identical across "
                      f"{len(hashes)} {unit}s (shared-trajectory design OK)")
        else:
            report.fail(f"theta {th}: trajectories DIFFER across {unit}s "
                        f"({len(uniq)} distinct) -- geometry is not the only "
                        f"varying factor!")

    # ---- (2) layouts genuinely different across groups ----
    for th in thetas:
        lay = layouts.get(th, {})
        if len(lay) <= 1:
            continue
        identical_pairs = []
        for (a, pa), (b, pb) in combinations(sorted(lay.items()), 2):
            if np.allclose(pa, pb):
                identical_pairs.append((labels[a], labels[b]))
        if identical_pairs:
            report.fail(f"theta {th}: identical sensor layouts for {unit}-pairs "
                        f"{identical_pairs} -- displacement/topology not applied!")
        else:
            report.ok(f"theta {th}: all {len(lay)} sensor layouts are distinct")

    # ---- (3) single feature shape across the whole corpus ----
    if len(feat_shapes) == 1:
        ppt, feat = next(iter(feat_shapes))
        report.ok(f"consistent feature tensor across corpus: "
                  f"(n_traj, ppt={ppt}, feat_dim={feat})")
    else:
        report.fail(f"inconsistent feature shapes across corpus: {feat_shapes}")

    # ---- (4) layouts identical across thetas within a group (positions should
    #          not depend on theta, only on the layout / topology) ----
    if len(thetas) > 1:
        for si, sd in enumerate(set_dirs):
            per_theta = []
            for th in thetas:
                if th in layouts and si in layouts[th]:
                    per_theta.append(layouts[th][si])
            if len(per_theta) > 1:
                allsame = all(np.allclose(per_theta[0], p) for p in per_theta[1:])
                if not allsame:
                    report.warn(f"{unit} {labels[si]}: sensor layout varies across "
                                f"theta (expected identical within a {unit})")
        report.ok(f"checked layout consistency across thetas within each {unit}")

    return layouts, Lf_expected


# --------------------------------------------------------------------------- #
# B. CHARACTERIZATION + FIGURES
# --------------------------------------------------------------------------- #
def fig_layouts_scatter(layouts, theta_ref, source_clouds, fig_dir, labels,
                        distinct=False, unit="position-set"):
    """All sensor layouts overlaid, plus the source-position cloud(s).

    ``source_clouds`` is a list of (3, n_traj, ppt+1) trajectory arrays: a single
    shared ensemble (shared / topology mode), or one ensemble per group in
    distinct mode (so the grey cloud honestly reflects differing source paths)."""
    lay = layouts[theta_ref]
    fig, ax = plt.subplots(figsize=(7, 6))

    # Source trajectory cloud(s) in light grey. Pool every provided ensemble so
    # the cloud reflects the true source coverage (1 ensemble shared, or N).
    xs = np.concatenate([c[0].ravel() for c in source_clouds])
    ys = np.concatenate([c[1].ravel() for c in source_clouds])
    cloud_lbl = (f"source positions (varies per {unit})" if distinct
                 else "source positions (shared)")
    ax.scatter(xs, ys, s=2, c="0.8", alpha=0.4, label=cloud_lbl)

    cmap = matplotlib.colormaps['viridis']
    n = len(lay)
    for i, (si, pos) in enumerate(sorted(lay.items())):
        ax.scatter(pos[0], pos[1], s=55, color=cmap(i / max(n - 1, 1)),
                   edgecolor="k", linewidth=0.3,
                   label=str(labels[si]) if n <= 8 else None)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Sensor layouts across {n} {unit}s (theta={theta_ref})")
    ax.set_aspect("equal", adjustable="datalim")
    if n <= 8:
        ax.legend(loc="best", fontsize=8)
    else:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=n - 1))
        cb = fig.colorbar(sm, ax=ax)
        cb.set_label(f"{unit} index")
    fig.tight_layout()
    p = os.path.join(fig_dir, "layouts_scatter.png")
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return p


def fig_layout_distance_matrix(layouts, theta_ref, fig_dir, labels,
                               unit="position-set"):
    """Pairwise layout distance: mean over sensors of per-sensor Euclidean
    displacement after optimal index matching is NOT done -- sensors are not
    identity-matched across groups, so we use a permutation-invariant geometry
    descriptor: sorted pairwise inter-sensor distance signature."""
    lay = layouts[theta_ref]
    items = sorted(lay.items())
    idxs = [labels[si] for si, _ in items]

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
    ax.set_xlabel(unit); ax.set_ylabel(unit)
    ax.set_title("Layout dissimilarity\n(||sorted inter-sensor distance||)")
    fig.colorbar(im, ax=ax, label="geometry distance")
    fig.tight_layout()
    p = os.path.join(fig_dir, "layout_distance_matrix.png")
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return p, D


def fig_feature_stats(summary_rows, fig_dir, labels, unit="position-set"):
    """Per-group feature level + dynamic range, grouped by theta."""
    thetas = sorted(set(r["theta"] for r in summary_rows), key=float)
    sets = sorted(set(r["set"] for r in summary_rows))
    xpos = list(range(len(sets)))
    xticklabels = [str(labels[s]) for s in sets]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    cmap = matplotlib.colormaps['coolwarm'] #plt.cm.coolwarm
    for ti, th in enumerate(thetas):
        means = [next(r["feat_mean_abs"] for r in summary_rows
                      if r["set"] == s and r["theta"] == th) for s in sets]
        rng = [next(r["feat_dyn_range_db"] for r in summary_rows
                    if r["set"] == s and r["theta"] == th) for s in sets]
        c = cmap(ti / max(len(thetas) - 1, 1))
        axes[0].plot(xpos, means, marker="o", color=c, label=f"theta={th}")
        axes[1].plot(xpos, rng, marker="s", color=c, label=f"theta={th}")

    axes[0].set_xlabel(unit); axes[0].set_ylabel("mean |feature|")
    axes[0].set_title(f"Feature level across {unit}s")
    axes[1].set_xlabel(unit); axes[1].set_ylabel("dynamic range [dB]")
    axes[1].set_title(f"Feature dynamic range across {unit}s")
    for a in axes:
        a.set_xticks(xpos); a.set_xticklabels(xticklabels, fontsize=7, rotation=90)
        a.legend(fontsize=7, ncol=2)
        a.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(fig_dir, "feature_stats_across_sets.png")
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return p


def fig_feature_distributions(data_root, set_dirs, theta_ref, fig_dir, labels,
                              mode, method, max_sets=6, unit="position-set"):
    """Overlaid histograms of (log) feature magnitude for several groups at a
    fixed theta -- visual evidence that the sensor geometry shifts the acoustic
    features (i.e. the task really changes across groups)."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    cmap = matplotlib.colormaps['viridis'] #plt.cm.viridis
    sel = set_dirs[:max_sets]
    for i, sd in enumerate(sel):
        filtered, _, _ = load_arrays(sd, theta_ref, mode, method)
        mag = np.abs(filtered).ravel()
        mag = mag[mag > 0]
        ax.hist(np.log10(mag), bins=120, histtype="step", density=True,
                color=cmap(i / max(len(sel) - 1, 1)),
                label=str(labels[i]))
    ax.set_xlabel("log10 |feature|")
    ax.set_ylabel("density")
    ax.set_title(f"Feature-magnitude distribution by {unit} "
                 f"(theta={theta_ref})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(fig_dir, "feature_distributions.png")
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return p


def fig_trajectories(traj_ref, theta_ref, fig_dir, max_traj=40):
    """The single shared trajectory ensemble (top-down), for SHARED mode."""
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


def fig_trajectories_by_set(per_set_trajs, theta_ref, fig_dir,
                            max_sets=6, max_traj=20):
    """DISTINCT mode: each position-set has its OWN trajectory ensemble, so plot
    several sets in different colours to make the per-set difference visible
    (this is the figure that 'shared_trajectories.png' cannot represent)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    sel = per_set_trajs[:max_sets]
    cmap = matplotlib.colormaps['viridis']
    for i, (si, traj) in enumerate(sel):
        c = cmap(i / max(len(sel) - 1, 1))
        n = min(max_traj, traj.shape[1])
        for k in range(n):
            ax.plot(traj[0, k], traj[1, k], lw=0.7, alpha=0.5, color=c)
        # one labelled proxy line per set
        ax.plot([], [], color=c, lw=1.5, label=f"set {si}")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"Per-set source trajectories (theta={theta_ref}, "
                 f"{len(sel)} sets, {max_traj} traj/set shown)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    p = os.path.join(fig_dir, "trajectories_by_set.png")
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return p


def fig_trajectories_grid(per_set_trajs, theta_ref, fig_dir, max_traj=30):
    """DISTINCT mode: one small panel PER position-set, each showing that set's
    own trajectory ensemble side by side -- so the per-set configuration
    differences are directly comparable across all sets. Shared x/y limits so
    panels are visually comparable."""
    n = len(per_set_trajs)
    if n == 0:
        return None
    ncols = min(5, n)
    nrows = int(np.ceil(n / ncols))
    # Common limits across all panels.
    allx = np.concatenate([t[0].ravel() for _, t in per_set_trajs])
    ally = np.concatenate([t[1].ravel() for _, t in per_set_trajs])
    xlim = (allx.min(), allx.max())
    ylim = (ally.min(), ally.max())

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.6 * nrows),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, (si, traj) in enumerate(per_set_trajs):
        ax = axes[i // ncols][i % ncols]
        ax.axis("on")
        m = min(max_traj, traj.shape[1])
        for k in range(m):
            ax.plot(traj[0, k], traj[1, k], lw=0.5, alpha=0.6)
        ax.set_title(f"set {si}", fontsize=8)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Per-set trajectory configurations (theta={theta_ref}, "
                 f"{n} sets)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = os.path.join(fig_dir, "trajectories_grid_by_set.png")
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def fig_traj_metrics_by_set(summary_rows, theta_ref, fig_dir):
    """DISTINCT mode: per-set trajectory-configuration metrics across sets at a
    fixed theta -- shows numerically how each set's trajectory ensemble differs
    (path length, reach, turning)."""
    rs = sorted((r for r in summary_rows if r["theta"] == theta_ref),
                key=lambda r: r["set"])
    sets = [r["set"] for r in rs]
    metrics = [
        ("traj_path_len_mean", "mean path length [m]"),
        ("traj_radial_max_mean", "mean farthest reach [m]"),
        ("traj_mean_turn_deg", "mean turn / step [deg]"),
        ("traj_start_spread", "start-point spread [m]"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (key, label) in zip(axes.ravel(), metrics):
        vals = [r[key] for r in rs]
        ax.plot(sets, vals, marker="o")
        ax.set_xlabel("position-set"); ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        rng = (max(vals) - min(vals)) if vals else 0.0
        ax.set_title(f"{label}  (spread={rng:.3g})", fontsize=9)
    fig.suptitle(f"Per-set trajectory-configuration metrics (theta={theta_ref})",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = os.path.join(fig_dir, "traj_metrics_by_set.png")
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return p


def fig_between_vs_within(data_root, set_dirs, theta_ref, fig_dir, mode, method,
                          max_sets=10, distinct=False):
    """Compare the spread of per-group MEAN feature vectors (between-group)
    against the within-group spread.

    Interpretation depends on the mode:
      * SHARED / TOPOLOGY -- trajectories are identical across groups, so the
        between-group variation isolates the effect of SENSOR GEOMETRY.
        ratio > 1 => the geometry-robustness task is well-posed.
      * DISTINCT -- both geometry AND trajectories differ across groups, so the
        between-group term is CONFOUNDED (geometry + trajectory) and cannot be
        attributed to geometry alone. The ratio is still reported but labelled
        accordingly."""
    means = []
    within = []
    sel = set_dirs[:max_sets]
    for sd in sel:
        feats = reshape_features(load_arrays(sd, theta_ref, mode, method)[0])  # (n_traj,ppt,F)
        flat = np.abs(feats).reshape(-1, feats.shape[-1])        # (samples, F)
        means.append(flat.mean(axis=0))
        within.append(flat.std(axis=0).mean())
    means = np.stack(means)                       # (sets, F)
    between = means.std(axis=0).mean()
    within = float(np.mean(within))
    ratio = between / (within + 1e-12)

    if distinct:
        between_lbl = "between-set\n(geometry + trajectory)"
        title = (f"Between-set vs within-set feature variation\n"
                 f"ratio = {ratio:.2f}  (theta={theta_ref}; "
                 f"confounded: geometry AND trajectory differ)")
    else:
        between_lbl = "between-set\n(geometry)"
        title = (f"Geometry vs. trajectory variation\n"
                 f"between/within ratio = {ratio:.2f}  (theta={theta_ref})")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.bar([between_lbl, "within-set\n(trajectory)"],
           [between, within], color=["#c0392b", "#2980b9"])
    ax.set_ylabel("mean feature-magnitude spread")
    ax.set_title(title)
    fig.tight_layout()
    p = os.path.join(fig_dir, "between_vs_within.png")
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return p, ratio


# --------------------------------------------------------------------------- #
# Summary table
# --------------------------------------------------------------------------- #
def build_summary(set_dirs, labels, thetas, mode, method):
    rows = []
    for si, sd in enumerate(set_dirs):
        for th in thetas:
            filtered, traj, pos = load_arrays(sd, th, mode, method)
            mag = np.abs(filtered)
            nz = mag[mag > 0]
            dyn_db = 20 * np.log10(nz.max() / nz.min()) if nz.size else float("nan")
            xy = pos[:2].T
            # array aperture = max pairwise sensor distance
            aperture = max(np.linalg.norm(xy[a] - xy[b])
                           for a, b in combinations(range(xy.shape[0]), 2))
            row = {
                "set": si,
                "group": labels[si],
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
            }
            # Per-set trajectory-configuration metrics (vary across sets in
            # distinct mode; identical across sets in shared mode).
            row.update(traj_shape_metrics(traj))
            rows.append(row)
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
    ap = argparse.ArgumentParser(
        description="QC / characterization for the random-position datasets "
                    "(position_set_* layout) OR the three-topology datasets "
                    "(<topology>/<method>/... layout). The layout is auto-detected.")
    ap.add_argument("--data-root", required=True,
                    help="For the position-set study: the dir containing "
                         "position_set_*. For the three-topology study: the dir "
                         "containing the <topology> folders (e.g. ./data).")
    ap.add_argument("--out-dir", default=None,
                    help="Where to write the report (default: <data-root>/qc_report)")
    ap.add_argument("--method", default="hermite",
                    help="Trajectory method subfolder for the three-topology "
                         "layout (<topology>/<method>/...). Ignored for the "
                         "position-set layout. Default: hermite.")
    ap.add_argument("--thetas", default=None,
                    help="Comma list to restrict thetas (default: all found)")
    ap.add_argument("--max-sets", type=int, default=None,
                    help="Only inspect the first K groups (position-sets or topologies)")
    ap.add_argument("--rep", type=int, default=1, help="rep used at generation")
    ap.add_argument("--n-sensors", type=int, default=10)
    ap.add_argument("--df", type=float, default=None,
                    help="Frequency resolution used at generation. Default is "
                         "auto: 50 for the three-topology layout, 100 for the "
                         "random-position layout.")
    ap.add_argument("--fmin", type=float, default=10000.0)
    ap.add_argument("--fmax", type=float, default=20000.0)
    ap.add_argument("--distinct-trajectories", dest="distinct_trajectories",
                    action="store_true", default=None,
                    help="Force per-set (distinct) trajectory invariant "
                         "(default: read from _manifest.pkl).")
    ap.add_argument("--shared-trajectories", dest="distinct_trajectories",
                    action="store_false", default=None,
                    help="Force shared (identical) trajectory invariant "
                         "(default: read from _manifest.pkl).")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.data_root, "qc_report")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Auto-detect the dataset layout and the comparison axis ("group").
    mode = detect_mode(args.data_root)
    method = args.method
    unit = "topology" if mode == "topologies" else "position-set"
    df = args.df if args.df is not None else (50.0 if mode == "topologies" else 100.0)

    groups = find_groups(args.data_root, mode)
    if not groups:
        where = ("<topology>/ folders" if mode == "topologies"
                 else "position_set_*")
        print(f"ERROR: no {where} under {args.data_root}", file=sys.stderr)
        sys.exit(1)
    if args.max_sets:
        groups = groups[:args.max_sets]
    labels = [g[0] for g in groups]
    set_dirs = [g[1] for g in groups]

    thetas = find_thetas(set_dirs[0], mode, method)
    if args.thetas:
        want = [t.strip() for t in args.thetas.split(",")]
        thetas = [t for t in thetas if t in want]
    if not thetas:
        print("ERROR: no thetas found / matched", file=sys.stderr)
        sys.exit(1)

    report = Report()
    report.info(f"data-root : {args.data_root}")
    report.info(f"layout    : {mode}"
                f"{f' (method={method})' if mode == 'topologies' else ''}")
    report.info(f"{unit}s inspected : {len(set_dirs)}  ->  {labels}")
    report.info(f"thetas : {thetas}  | df : {df}")

    # In the three-topology study, trajectories are ALWAYS shared across the
    # three topologies (geometry is the only varying factor), so the shared
    # invariant is forced. The position-set study reads the mode from the
    # manifest (overridable on the CLI).
    if mode == "topologies":
        distinct_trajectories = False
        report.info("trajectory mode : SHARED across topologies (by design)")
    else:
        # Optional manifest cross-check + source of truth for the trajectory mode.
        man_distinct = None
        man_path = os.path.join(args.data_root, "_manifest.pkl")
        if os.path.exists(man_path):
            with open(man_path, "rb") as f:
                man = pickle.load(f)
            man_distinct = man.get('distinct_trajectories')
            report.ok(f"manifest found (master_seed={man.get('master_seed')}, "
                      f"declared sets={man.get('n_position_sets')}, "
                      f"distinct_trajectories={man_distinct})")
        else:
            report.warn("no _manifest.pkl found -- skipping seed cross-check")

        if args.distinct_trajectories is not None:
            distinct_trajectories = bool(args.distinct_trajectories)
        elif man_distinct is not None:
            distinct_trajectories = bool(man_distinct)
        else:
            distinct_trajectories = False
            report.warn("trajectory mode unknown (no manifest flag, no CLI "
                        "override) -- assuming SHARED. Pass "
                        "--distinct-trajectories if wrong.")
        report.info(f"trajectory mode : "
                    f"{'DISTINCT (per-set)' if distinct_trajectories else 'SHARED'}")

    # ---- A. validity ----
    layouts, Lf = check_validity(
        args.data_root, set_dirs, labels, thetas, args.rep, args.n_sensors,
        df, args.fmin, args.fmax, report, mode, method,
        distinct_trajectories=distinct_trajectories, unit=unit)

    # ---- B. characterization ----
    report.info("=" * 64)
    report.info("B. CHARACTERIZATION (figures + table)")
    report.info("=" * 64)

    theta_ref = thetas[0]
    _, traj_ref, _ = load_arrays(set_dirs[0], theta_ref, mode, method)

    # Source-position cloud(s) for the layout scatter: one shared ensemble, or
    # every group's own ensemble in distinct mode. In distinct mode we also keep
    # a per-group list of (idx, traj) for the per-group trajectory figure.
    per_set_trajs = []
    if distinct_trajectories:
        for si, sd in enumerate(set_dirs):
            try:
                per_set_trajs.append((si, load_arrays(sd, theta_ref, mode, method)[1]))
            except FileNotFoundError:
                continue
        source_clouds = [t for _, t in per_set_trajs] or [traj_ref]
    else:
        source_clouds = [traj_ref]

    rows = build_summary(set_dirs, labels, thetas, mode, method)
    write_csv(rows, os.path.join(out_dir, "qc_summary.csv"))
    report.ok(f"wrote qc_summary.csv ({len(rows)} rows)")

    figs = []
    figs.append(fig_layouts_scatter(layouts, theta_ref, source_clouds, fig_dir,
                                    labels, distinct=distinct_trajectories, unit=unit))
    dm_path, _ = fig_layout_distance_matrix(layouts, theta_ref, fig_dir, labels,
                                            unit=unit)
    figs.append(dm_path)
    figs.append(fig_feature_stats(rows, fig_dir, labels, unit=unit))
    figs.append(fig_feature_distributions(args.data_root, set_dirs, theta_ref,
                                          fig_dir, labels, mode, method, unit=unit))
    if distinct_trajectories:
        figs.append(fig_trajectories_by_set(per_set_trajs, theta_ref, fig_dir))
        grid = fig_trajectories_grid(per_set_trajs, theta_ref, fig_dir)
        if grid:
            figs.append(grid)
        figs.append(fig_traj_metrics_by_set(rows, theta_ref, fig_dir))
    else:
        figs.append(fig_trajectories(traj_ref, theta_ref, fig_dir))
    bw_path, ratio = fig_between_vs_within(args.data_root, set_dirs, theta_ref,
                                           fig_dir, mode, method,
                                           distinct=distinct_trajectories)
    figs.append(bw_path)
    for fpath in figs:
        report.ok(f"figure: {os.path.relpath(fpath, out_dir)}")

    # Between/within interpretation. In distinct mode the between-set term mixes
    # geometry AND trajectory differences, so it cannot certify a geometry-only
    # task -- report it as descriptive rather than a pass/fail gate.
    if distinct_trajectories:
        # Summarise how much the trajectory CONFIGURATION varies across sets.
        rs = [r for r in rows if r["theta"] == theta_ref]
        if rs:
            def spread(key):
                v = [r[key] for r in rs]
                return min(v), max(v), float(np.std(v))
            pl = spread("traj_path_len_mean")
            rr = spread("traj_radial_max_mean")
            report.info(f"per-set trajectory config (theta={theta_ref}, "
                        f"{len(rs)} sets): path_len mean-range "
                        f"[{pl[0]:.1f},{pl[1]:.1f}] std={pl[2]:.1f} m; "
                        f"reach mean-range [{rr[0]:.1f},{rr[1]:.1f}] std={rr[2]:.1f} m "
                        f"-- see traj_metrics_by_set.png / trajectories_grid_by_set.png")
        report.info(f"between/within feature-variation ratio (theta={theta_ref}): "
                    f"{ratio:.2f} -- CONFOUNDED (both geometry and trajectories "
                    f"differ across sets); not attributable to geometry alone.")
    else:
        report.info(f"between/within feature-variation ratio (theta={theta_ref}): "
                    f"{ratio:.2f}")
        if ratio < 1.0:
            report.warn(f"between/within ratio < 1: the sensor geometry changes the "
                        f"features LESS than trajectory variation -- the {unit} "
                        f"robustness task may be weak. Inspect feature_distributions.png.")
        else:
            report.ok(f"the sensor geometry induces a clear between-{unit} feature "
                      f"shift (ratio >= 1): the robustness task is well-posed.")

    report.dump(os.path.join(out_dir, "qc_report.txt"))
    print(f"\nReport written to: {out_dir}")
    if report.hard_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
