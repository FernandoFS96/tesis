"""Rebuild the three descriptive figures (task, trajectories, environment prior).

These three were originally produced inline for a talk, in Spanish, and the
generating script was never kept -- so the labels could not be changed without
redrawing them. This script replaces them from the data, in English, and lives in
the repo so it does not happen twice.

Data sources differ from the originals on purpose:
  * task / trajectories come from `pfn10kd`, the dataset benchmark B1 actually
    reports, where every deployment has its OWN trajectories. The original was
    drawn from `random_task`, whose trajectories are SHARED across deployments --
    the very property Section 4.2 shows to be a shortcut. Keeping that figure
    would have illustrated the paper with the flaw it argues against.
  * the environment prior comes from `pathD`, which is what B2 uses.

Palette: slots 1-2 of the validated catalogue for the two-series panels. The
environment scatter uses a SINGLE-HUE light->dark ramp built from slot 1 rather
than viridis: a sequential quantity must not be encoded on a multi-hue rainbow.
"""
import glob, os, pickle, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG  = os.path.join(os.environ.get("PAPER_OUT", os.path.join(REPO, "paper")), "figures")
os.chdir(REPO)
os.makedirs(FIG, exist_ok=True)

C1, C2 = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
# Rampa de un solo tono (regla: secuencial = un tono, claro -> oscuro; nunca
# arcoiris). Arranca en un azul medio-claro y no en casi-blanco: con 4000
# puntos sobre fondo blanco, el extremo claro desaparecia.
BLUES = LinearSegmentedColormap.from_list("slot1", ["#9ec3ef", "#2a78d6", "#10325d"])

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 150,
})

# Per-figure base font size. The figures are drawn wider than they are printed,
# so text shrinks by (printed width / drawn width): fig_task 0.50, fig_trajectories
# 0.43, fig_regions 0.79 in the AOR layout. @font(n) sets the base size n for one
# figure; titles, legends and labels inside it are relative ("medium", "large",
# ...) and follow it, so this number is the only knob. ~7 pt printed = 7 / scale.
import functools
def font(size):
    def deco(fn):
        @functools.wraps(fn)
        def wrapped(*a, **k):
            with plt.rc_context({"font.size": size}):
                return fn(*a, **k)
        return wrapped
    return deco

ROOT = "data/pfn10kd/hermite_distinct"
# Deployment directories are zero-padded to a VARIABLE width (position_set_00 ...
# position_set_9999), so the name cannot be formatted from the index. Resolve it
# once by listing and indexing numerically.
_DIRS = {int(re.search(r"position_set_(\d+)$", d).group(1)): d
         for d in glob.glob(f"{ROOT}/position_set_*")}

def deployment(i):
    """(sensors, trajectories) for deployment i, channel option 0.1.

    BOTH arrays are (coordinate, item, ...) -- sensor_positions is (3, S) and
    trajectories is (3, T, P), i.e. axis 0 is x/y/z, NOT the item index. Reading
    axis 0 as the trajectory index plots x against x and produces a plausible-
    looking but meaningless figure; the giveaway is that the third slice is
    identically zero, which is the z coordinate of a 2-D task, not a dead
    trajectory.
    """
    base = f"{_DIRS[i]}/channel_option_0.1/random"
    sp = np.load(f"{base}/channel_info/sensor_positions_0.1.npy")
    tr = np.load(f"{base}/trajectory/trajectories.npy")
    return sp, tr

# --------------------------------------------------------------- 1. the task
@font(13)
def fig_task(sets=(0, 5, 11, 23)):
    fig, axes = plt.subplots(1, len(sets), figsize=(10.2, 3.9), sharex=True, sharey=True)
    for ax, i in zip(axes, sets):
        sp, tr = deployment(i)
        for j in range(tr.shape[1]):
            ax.plot(tr[0, j], tr[1, j], color=C1, linewidth=1.1, alpha=0.85, zorder=2)
        ax.scatter(sp[0], sp[1], marker="^", s=34, color=C2, edgecolor="white",
                   linewidth=0.7, zorder=3,
                   label="hydrophones" if i == sets[0] else None)
        ax.set_title(f"deployment {i}", fontsize="medium")
        ax.set_xlabel("x [m]")
        # Three ticks: at this font size the end labels of neighboring panels
        # ("500" and "-500") ran into each other.
        ax.set_xticks([-400, 0, 400])
    axes[0].set_ylabel("y [m]")
    axes[0].plot([], [], color=C1, linewidth=1.6, label="source trajectories")
    # Legend below the panels (inside the first one it covered the hydrophones),
    # as a figure legend so tight_layout can reserve room for it.
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize="medium",
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("The array moves between deployments and each has "
                 "its own trajectories", fontsize="large", y=0.99)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(f"{FIG}/fig_task.pdf", bbox_inches="tight"); plt.close(fig)
    print("  fig_task.pdf")

# ------------------------------------------------------- 2. Hermite trajectories
@font(14)
def fig_trajectories(n_sets=6):
    sp, _ = deployment(0)
    # Taller than wide: the axes keep an equal aspect, so the extra height goes
    # to a larger plot instead of to white space.
    fig, ax = plt.subplots(figsize=(5.0, 5.8))
    paths = []
    for i in range(n_sets):
        _, tr = deployment(i)
        paths.extend([(tr[0, j], tr[1, j]) for j in range(tr.shape[1])])
    for x, y in paths:
        ax.plot(x, y, color=C1, linewidth=1.0, alpha=0.55, zorder=2)
    hx, hy = paths[0]
    ax.plot(hx, hy, color=C2, linewidth=1.8, zorder=4,
            label=f"one trajectory ({hx.size} points)")
    ax.scatter(hx[::5], hy[::5], s=34, color=C2, edgecolor="white",
               linewidth=1.1, zorder=5)
    ax.scatter(sp[0], sp[1], marker="^", s=38, color=INK, edgecolor="white",
               linewidth=0.7, zorder=3, label="hydrophones (deployment 0)")
    ax.plot([], [], color=C1, linewidth=1.4, alpha=0.55,
            label=f"trajectories from {n_sets} deployments")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    # The title stays at the base size: wider than the plot, it would widen the
    # whole figure and shrink everything again when it is printed. Same reason
    # for the one-column legend.
    ax.set_title("Hermite trajectories: erratic by design", fontsize="medium", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1, fontsize="medium")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_trajectories.pdf", bbox_inches="tight"); plt.close(fig)
    print("  fig_trajectories.pdf")

# ------------------------------------------------------- 3. environment prior
def fig_environment_prior():
    envs = {}
    for f in sorted(glob.glob("data/pathD/hermite_distinct/_manifest*.pkl")):
        envs.update(pickle.load(open(f, "rb")).get("environments", {}))
    if not envs:
        print("  (environment prior: no manifests)"); return
    h  = np.array([e["h0"] for e in envs.values()])
    c  = np.array([e["c"]  for e in envs.values()])
    r  = np.array([e["c2"] / e["c"] for e in envs.values()])
    k  = np.array([e["k"]  for e in envs.values()])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.4, 3.6))
    s = axL.scatter(h, c, c=r, cmap=BLUES, s=5, linewidth=0, alpha=0.85)
    cb = fig.colorbar(s, ax=axL, pad=0.02)
    cb.set_label("bottom / water sound speed", fontsize=8.2, color=INK2)
    cb.outline.set_edgecolor(GRID)
    axL.set_xlabel("water depth [m]"); axL.set_ylabel("sound speed [m/s]")
    axL.set_title(f"Environment prior: {len(envs):,} sampled deployments",
                  fontsize=10.5, loc="left")

    n, _, _ = axR.hist(k, bins=40, color=C1, linewidth=0)
    axR.axvline(1.7, color=C2, linewidth=2.0, linestyle=(0, (5, 2)))
    # Cabecera libre sobre las barras para la llamada: dentro del area de datos
    # el texto caia encima del histograma y no se leia.
    axR.set_ylim(0, n.max() * 1.28)
    axR.annotate("fixed value in the earlier project (1.7)", xy=(1.7, n.max()),
                 xytext=(9, 14), textcoords="offset points", fontsize=8,
                 color=C2, ha="left", va="bottom")
    axR.set_xlabel("spreading factor $k$"); axR.set_ylabel("deployments")
    axR.set_title("Before: a constant. Now: an axis", fontsize=10.5, loc="left")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_environment_prior.pdf", bbox_inches="tight"); plt.close(fig)
    print("  fig_environment_prior.pdf")

# --------------------------------------------------------- 4. interp / extrap
@font(10)
def fig_regions():
    """Where the B1 held-out arrays sit, from splits.json and the raw positions.

    The split ranks deployments by centroid distance from the centre of all
    centroids: the most central 1000 and the most peripheral 1000 are held out,
    the ring in between trains. Labels are then taken against the training hull,
    which is what the outline shows.
    """
    import json
    from scipy.spatial import ConvexHull
    sp = json.load(open(f"{ROOT}/processed/geometry_split/splits.json"))
    lab = {int(k): v["region"] for k, v in sp["labels"].items()}
    pos = {i: deployment(i)[0][:2].T for i in _DIRS}          # (S, 2) per deployment
    cen = {i: p.mean(0) for i, p in pos.items()}
    train = np.array([cen[i] for i in sp["train"]])
    hull = train[np.r_[ConvexHull(train).vertices, ConvexHull(train).vertices[:1]]]
    c0 = np.array(sp["center_of_centroids"])
    TRAIN = "#a9a8a2"
    groups = [("train", TRAIN, f"training ({len(train)})"),
              ("interp", C1, f"interp ({sum(v == 'interp' for v in lab.values())})"),
              ("extrap", C2, f"extrap ({sum(v == 'extrap' for v in lab.values())})")]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.8, 3.25))
    for key, col, txt in groups:
        P = train if key == "train" else np.array([cen[i] for i in lab if lab[i] == key])
        axL.scatter(P[:, 0], P[:, 1], s=1.6, color=col, linewidths=0, label=txt,
                    rasterized=True)
    axL.plot(hull[:, 0], hull[:, 1], color=INK2, lw=1.1, label="training hull")
    axL.set_title("Array centroids, all 10,000 deployments", fontsize="medium", loc="left")

    # One array of each kind, at the median distance of its group and pointing
    # in different directions so the three do not overlap.
    def typical(ids, angle=None):
        d = {i: np.linalg.norm(cen[i] - c0) for i in ids}
        if angle is not None:
            a = {i: np.degrees(np.arctan2(*(cen[i] - c0)[::-1])) for i in ids}
            d = {i: v for i, v in d.items() if abs((a[i] - angle + 180) % 360 - 180) < 20}
        med = np.median(list(d.values()))
        return min(d, key=lambda i: abs(d[i] - med))
    picks = [(typical(sp["train"], angle=200), "#8f8e88", "training"),
             (typical([i for i in lab if lab[i] == "interp"]), C1, "interp"),
             (typical([i for i in lab if lab[i] == "extrap"], angle=45), C2, "extrap")]
    axR.plot(hull[:, 0], hull[:, 1], color=INK2, lw=1.1)
    for i, col, txt in picks:
        P = pos[i]
        for q in P:                                   # spokes tie sensors to their array
            axR.plot([cen[i][0], q[0]], [cen[i][1], q[1]], color=col, lw=0.6,
                     alpha=0.55, zorder=2)
        axR.scatter(P[:, 0], P[:, 1], s=14, color=col, edgecolors="white",
                    linewidths=0.6, zorder=3, label=f"{txt} array")
        axR.scatter(*cen[i], s=46, marker="+", color=col, linewidths=1.4, zorder=4)
    axR.set_title("One array of each kind (+ centroid)", fontsize="medium", loc="left")

    lim = np.abs(np.concatenate([np.array(list(cen.values())), *pos.values()])).max() * 1.03
    for ax in (axL, axR):
        ax.set_aspect("equal"); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("x [m]")
    axL.set_ylabel("y [m]")
    axL.legend(loc="lower left", fontsize="small", markerscale=4, handlelength=1.2,
               borderaxespad=0.2)
    axR.legend(loc="lower left", fontsize="small", handlelength=1.0, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_regions.pdf", bbox_inches="tight", dpi=300); plt.close(fig)
    print("  fig_regions.pdf")

if __name__ == "__main__":
    print("Rebuilding descriptive figures:")
    fig_task(); fig_trajectories(); fig_environment_prior(); fig_regions()
