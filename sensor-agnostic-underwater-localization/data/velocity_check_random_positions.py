"""
velocity_check_random_positions.py
==================================

Post-hoc velocity & trajectory characterization for the 20 random-position
datasets produced by ``data_generator_random_positions.py``.

WHY THIS CAN BE DONE AFTERWARDS
-------------------------------
Instantaneous velocity is a deterministic function of the trajectory positions,
which are already saved to disk for every theta in every position-set
(``channel_info/trajs_<theta>.npy``, shape ``(3, n_traj, ppt+1)``). Nothing about
the channel simulation alters or destroys this information, so velocity never has
to be computed at generation time -- it is fully recoverable here, exactly as the
QC script recovers feature statistics.

WHAT IT REPRODUCES FROM THE ORIGINAL GENERATOR
----------------------------------------------
The original ``data_generator_topology.save_velocity_histogram`` computes, per
theta:

    diffs      = np.diff(traj, axis=2)            # (3, n_traj, ppt)
    step_dists = np.linalg.norm(diffs, axis=0)    # (n_traj, ppt)
    speeds     = step_dists / T_tot               # (n_traj, ppt)  [per-jump m/s]
    stats      = mean / std / min / max  over all jumps

This tool reproduces that EXACT formula and the per-theta histogram (same axis
labels and stats box as the original), and adds the cross-dataset structure that
is specific to the new experiment.

THE KEY STRUCTURAL POINT
------------------------
Because the corrected generator shares ONE trajectory ensemble across all 20
position-sets (sensor geometry is the only varying factor), the velocity
distribution for a given theta is IDENTICAL across all 20 datasets. Therefore:

  * we report ONE velocity profile per theta (not 20 redundant ones);
  * we VERIFY that the per-set trajectories are byte-identical across the 20
    sets -- a velocity-side re-confirmation of the shared-trajectory invariant
    (the same thing the QC script checks via hashing). If they are NOT identical,
    that is a hard error and is reported per theta.

It still *iterates over all 20 datasets for each theta* (as requested): it loads
every (set, theta) trajectory file, checks them against the reference, and only
then emits the per-theta velocity profile.

OUTPUTS  (under --out-dir, default <data-root>/velocity_report)
    figures/velocity_hist_theta_<theta>.png   per-theta histogram (original style)
    figures/velocity_by_theta_summary.png     mean +/- std vs theta across all thetas
    figures/speed_along_trajectory.png         mean speed vs trajectory point index
    velocity_summary.csv                       per-theta stats + cross-set agreement
    velocity_report.txt                        human-readable pass/fail log

USAGE
-----
    python velocity_check_random_positions.py --data-root ./data_random_positions

    # If your T_tot differs from the default 6.0 s, pass it explicitly so the
    # per-jump speed units match your channel exactly:
    python velocity_check_random_positions.py --data-root ./data_random_positions \
        --t-tot 6.0

    # restrict thetas / sets while iterating
    python velocity_check_random_positions.py --data-root ./data_random_positions \
        --thetas 0.0,0.3 --max-sets 5

NOTE ON UNITS
-------------
The original code divides the per-jump displacement by ``T_tot`` (the total
simulated signal duration, 6 s) rather than by a per-step dt. We preserve that
exact convention so the numbers are directly comparable to the original
generator's histograms. The label is therefore "per-jump speed (Δs / T_tot)",
matching the source. If you later want true per-step physical speed, divide by
the real inter-point time instead; this tool exposes ``--t-tot`` for that.
"""

import os
import re
import sys
import glob
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Default total simulated duration used as the per-jump time in the original
# generator (data_generator_topology: params['ci']['T_tot'] = 6.0).
DEFAULT_T_TOT = 6.0


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def find_position_sets(data_root):
    sets = sorted(glob.glob(os.path.join(data_root, "position_set_*")))
    return [s for s in sets if os.path.isdir(s)]


def find_thetas(set_dir):
    opts = []
    for d in glob.glob(os.path.join(set_dir, "channel_option_*")):
        m = re.search(r"channel_option_([0-9.]+)$", d)
        if m and os.path.isdir(os.path.join(d, "random")):
            opts.append(m.group(1))
    return sorted(opts, key=float)


def traj_path(set_dir, theta):
    """The full (3, n_traj, ppt+1) trajectory saved by the generator.
    Prefer the channel_info copy (has the +1 endpoint needed for diff);
    fall back to the target trajectory if missing."""
    info = os.path.join(set_dir, f"channel_option_{theta}", "random",
                        "channel_info", f"trajs_{theta}.npy")
    if os.path.exists(info):
        return info, "channel_info"
    # Fallback: target trajectory (3, n_traj, ppt) -- one fewer point.
    tgt = os.path.join(set_dir, f"channel_option_{theta}", "random",
                       "trajectory", "trajectories.npy")
    return tgt, "trajectory"


# --------------------------------------------------------------------------- #
# Velocity computation (EXACT reproduction of the original semantics)
# --------------------------------------------------------------------------- #
def compute_speeds(traj, t_tot):
    """traj: (3, n_traj, ppt+1) -> speeds: (n_traj, n_jumps)  [per-jump m/s].

    Mirrors data_generator_topology.save_velocity_histogram:
        diffs = np.diff(traj, axis=2); step = ||diffs||; speed = step / T_tot.
    """
    assert traj.ndim == 3 and traj.shape[0] == 3, \
        "traj must be (3, n_traj, ppt+1)"
    diffs = np.diff(traj, axis=2)                  # (3, n_traj, n_jumps)
    step_dists = np.linalg.norm(diffs, axis=0)     # (n_traj, n_jumps)
    speeds = step_dists / float(t_tot)             # (n_traj, n_jumps)
    return speeds


def speed_stats(speeds):
    flat = speeds.ravel()
    return {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "median": float(np.median(flat)),
        "p95": float(np.percentile(flat, 95)),
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_hist_for_theta(speeds, theta, stats, fig_dir, bins=40):
    """Per-theta histogram in the original generator's visual style."""
    flat = speeds.ravel()
    plt.figure(figsize=(8, 5))
    plt.hist(flat, bins=bins, edgecolor="k")
    plt.xlabel("Per-jump speed (Δs / T_tot) [m/s]")
    plt.ylabel("Frequency")
    plt.title(f"Velocity histogram — theta={theta}")
    txt = (f"mean={stats['mean']:.2f} m/s\nstd={stats['std']:.2f} m/s\n"
           f"min={stats['min']:.2f} m/s\nmax={stats['max']:.2f} m/s")
    plt.gca().text(0.98, 0.95, txt, ha="right", va="top",
                   transform=plt.gca().transAxes,
                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, lw=0.5))
    plt.tight_layout()
    p = os.path.join(fig_dir, f"velocity_hist_theta_{theta}.png")
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    return p


def fig_summary_by_theta(per_theta_stats, fig_dir):
    """Mean +/- std of per-jump speed as a function of theta."""
    thetas = sorted(per_theta_stats.keys(), key=float)
    x = [float(t) for t in thetas]
    means = [per_theta_stats[t]["mean"] for t in thetas]
    stds = [per_theta_stats[t]["std"] for t in thetas]
    mins = [per_theta_stats[t]["min"] for t in thetas]
    maxs = [per_theta_stats[t]["max"] for t in thetas]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(x, means, yerr=stds, fmt="o-", capsize=4, label="mean ± std")
    ax.fill_between(x, mins, maxs, alpha=0.15, label="min–max range")
    ax.set_xlabel("theta (channel variability)")
    ax.set_ylabel("per-jump speed [m/s]")
    ax.set_title("Source-velocity profile across theta groups")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(fig_dir, "velocity_by_theta_summary.png")
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return p


def fig_speed_along_trajectory(ref_traj_by_theta, t_tot, fig_dir):
    """Mean speed vs trajectory-point index, one line per theta. Reveals whether
    the source accelerates/decelerates along the spiral (a sanity check on the
    trajectory generator)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    thetas = sorted(ref_traj_by_theta.keys(), key=float)
    cmap = matplotlib.colormaps['viridis']
    for i, th in enumerate(thetas):
        speeds = compute_speeds(ref_traj_by_theta[th], t_tot)  # (n_traj, n_jumps)
        mean_per_step = speeds.mean(axis=0)                    # (n_jumps,)
        ax.plot(np.arange(mean_per_step.size), mean_per_step,
                color=cmap(i / max(len(thetas) - 1, 1)), label=f"theta={th}")
    ax.set_xlabel("trajectory point index (jump)")
    ax.set_ylabel("mean per-jump speed [m/s]")
    ax.set_title("Mean source speed along the trajectory")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(fig_dir, "speed_along_trajectory.png")
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #
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
        description="Post-hoc velocity / trajectory characterization for the "
                    "random-position datasets.")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--thetas", default=None,
                    help="Comma list to restrict thetas (default: all found)")
    ap.add_argument("--max-sets", type=int, default=None,
                    help="Only inspect the first K position-sets")
    ap.add_argument("--t-tot", type=float, default=DEFAULT_T_TOT,
                    help=f"Per-jump time used as the velocity denominator "
                         f"(default {DEFAULT_T_TOT}, matching the original "
                         f"generator's params['ci']['T_tot']).")
    ap.add_argument("--bins", type=int, default=40)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.data_root, "velocity_report")
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

    log = []
    def line(s): log.append(s); print(s)

    line(f"data-root : {args.data_root}")
    line(f"position-sets : {len(set_dirs)} | thetas : {thetas}")
    line(f"T_tot (per-jump denominator) : {args.t_tot} s")
    line("=" * 64)

    per_theta_stats = {}
    ref_traj_by_theta = {}
    rows = []
    hard_failures = 0

    # For each theta group, iterate over ALL position-sets, verify the
    # trajectories agree, then compute the (shared) velocity profile once.
    for th in thetas:
        ref_traj = None
        ref_src = None
        agree = True
        max_disagreement = 0.0
        n_checked = 0

        for si, sd in enumerate(set_dirs):
            tp, src = traj_path(sd, th)
            if not os.path.exists(tp):
                line(f"[ WARN ] theta {th} set {si}: missing trajectory file")
                continue
            traj = np.load(tp)
            n_checked += 1
            if ref_traj is None:
                ref_traj = traj
                ref_src = src
            else:
                if traj.shape != ref_traj.shape:
                    agree = False
                    line(f"[ FAIL ] theta {th} set {si}: trajectory shape "
                         f"{traj.shape} != reference {ref_traj.shape}")
                else:
                    d = float(np.abs(traj - ref_traj).max())
                    max_disagreement = max(max_disagreement, d)
                    if d != 0.0:
                        agree = False

        if ref_traj is None:
            line(f"[ FAIL ] theta {th}: no trajectories found in any set")
            hard_failures += 1
            continue

        # Need the (3, n_traj, ppt+1) form for a clean diff. If we only had the
        # target (3, n_traj, ppt), diff still works but covers ppt-1 jumps.
        if ref_src == "trajectory":
            line(f"[ WARN ] theta {th}: using target trajectory "
                 f"(3,n_traj,ppt); channel_info copy not found. Velocity covers "
                 f"ppt-1 jumps instead of ppt.")

        # Cross-set agreement verdict (velocity-side invariant check).
        if agree and max_disagreement == 0.0:
            line(f"[ PASS ] theta {th}: trajectories identical across "
                 f"{n_checked} sets (max|diff|=0) -> single shared velocity "
                 f"profile is valid")
        else:
            line(f"[ FAIL ] theta {th}: trajectories DIFFER across sets "
                 f"(max|diff|={max_disagreement:.4g}) -> velocity is NOT shared; "
                 f"shared-trajectory invariant violated")
            hard_failures += 1

        # Velocity profile (computed on the reference; identical across sets if
        # the check above passed).
        speeds = compute_speeds(ref_traj, args.t_tot)
        st = speed_stats(speeds)
        per_theta_stats[th] = st
        ref_traj_by_theta[th] = ref_traj

        line(f"         theta {th}: speed mean={st['mean']:.3f} "
             f"std={st['std']:.3f} min={st['min']:.3f} max={st['max']:.3f} "
             f"median={st['median']:.3f} p95={st['p95']:.3f} [m/s]  "
             f"(n_traj={ref_traj.shape[1]}, jumps={speeds.shape[1]})")

        # Per-theta histogram (original style).
        fig_hist_for_theta(speeds, th, st, fig_dir, bins=args.bins)

        rows.append({
            "theta": th,
            "n_sets_checked": n_checked,
            "cross_set_max_diff": max_disagreement,
            "shared_ok": int(agree and max_disagreement == 0.0),
            "n_traj": ref_traj.shape[1],
            "n_jumps": speeds.shape[1],
            "v_mean": st["mean"], "v_std": st["std"],
            "v_min": st["min"], "v_max": st["max"],
            "v_median": st["median"], "v_p95": st["p95"],
        })

    # Cross-theta figures.
    if per_theta_stats:
        s1 = fig_summary_by_theta(per_theta_stats, fig_dir)
        s2 = fig_speed_along_trajectory(ref_traj_by_theta, args.t_tot, fig_dir)
        line(f"         figure: {os.path.relpath(s1, out_dir)}")
        line(f"         figure: {os.path.relpath(s2, out_dir)}")

    write_csv(rows, os.path.join(out_dir, "velocity_summary.csv"))
    line(f"         wrote velocity_summary.csv ({len(rows)} theta rows)")

    line("=" * 64)
    line(f"Hard failures: {hard_failures}")
    with open(os.path.join(out_dir, "velocity_report.txt"), "w") as f:
        f.write("\n".join(log) + "\n")
    print(f"\nReport written to: {out_dir}")
    if hard_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
