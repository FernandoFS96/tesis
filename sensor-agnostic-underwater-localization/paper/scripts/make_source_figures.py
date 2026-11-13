"""Descriptive figures drawn from the B1 dataset (Figs. 3 and 5).

    python paper/scripts/make_source_figures.py
    PAPER_OUT=path/to/manuscript python paper/scripts/make_source_figures.py

Needs the raw B1 deployments in data/pfn10kd/hermite_distinct.
"""
import functools, glob, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG  = os.path.join(os.environ.get("PAPER_OUT", os.path.join(REPO, "paper")), "figures")
os.chdir(REPO)
os.makedirs(FIG, exist_ok=True)

C1, C2 = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 150,
})


def font(size):
    """Base font size for one figure. The figures are scaled down when placed in
    the paper (Fig. 3a by about 0.50, 3b by 0.43, Fig. 5 by 0.79), so the printed
    size is roughly size times that factor. Titles and legends are relative."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapped(*a, **k):
            with plt.rc_context({"font.size": size}):
                return fn(*a, **k)
        return wrapped
    return deco


ROOT = "data/pfn10kd/hermite_distinct"
_DIRS = {int(re.search(r"position_set_(\d+)$", d).group(1)): d
         for d in glob.glob(f"{ROOT}/position_set_*")}


def deployment(i):
    """Sensor positions (3, S) and trajectories (3, T, P) of deployment i.
    Axis 0 is the coordinate in both arrays."""
    base = f"{_DIRS[i]}/channel_option_0.1/random"
    sp = np.load(f"{base}/channel_info/sensor_positions_0.1.npy")
    tr = np.load(f"{base}/trajectory/trajectories.npy")
    return sp, tr


@font(13)
def fig_task(sets=(0, 5, 11, 23)):
    """Fig. 3a: four deployments with their arrays and trajectories."""
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
        ax.set_xticks([-400, 0, 400])
    axes[0].set_ylabel("y [m]")
    axes[0].plot([], [], color=C1, linewidth=1.6, label="source trajectories")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize="medium",
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("The array moves between deployments and each has "
                 "its own trajectories", fontsize="large", y=0.99)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(f"{FIG}/fig_task.pdf", bbox_inches="tight"); plt.close(fig)
    print("  fig_task.pdf")


@font(14)
def fig_trajectories(n_sets=6):
    """Fig. 3b: trajectories from several deployments, one highlighted."""
    sp, _ = deployment(0)
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
    ax.set_title("Hermite trajectories: erratic by design", fontsize="medium", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1, fontsize="medium")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_trajectories.pdf", bbox_inches="tight"); plt.close(fig)
    print("  fig_trajectories.pdf")


@font(10)
def fig_regions():
    """Fig. 5: where the B1 held-out arrays sit relative to the training ones."""
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

    # One array of each kind at the median distance of its group, in different
    # directions so they do not overlap.
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
        for q in P:
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
    print("Generating figures:")
    fig_task(); fig_trajectories(); fig_regions()
