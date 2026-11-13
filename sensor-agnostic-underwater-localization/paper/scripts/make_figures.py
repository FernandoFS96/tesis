"""Paper figures built from the evaluation results (Figs. 4, 6 and 7).

    python paper/scripts/make_figures.py
    PAPER_OUT=path/to/manuscript python paper/scripts/make_figures.py
"""
import csv, math, os, statistics as st
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EV   = os.path.join(REPO, os.environ.get("EVAL_ROOT", os.path.join("output", "eval_paper")))
FIG  = os.path.join(os.environ.get("PAPER_OUT", os.path.join(REPO, "paper")), "figures")
os.makedirs(FIG, exist_ok=True)

C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

# The base font size; titles, legends and labels are set relative to it.
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 150,
})
fmt = FuncFormatter(lambda v, _: f"{v:g}")


def load(g):
    f = os.path.join(EV, g, "results.csv")
    return list(csv.DictReader(open(f))) if os.path.exists(f) else None


def load_seeds(g):
    """Results re-aggregated over training seeds, as in make_tables.py."""
    f = os.path.join(EV, g, "results_per_seed.csv")
    if not os.path.exists(f): return None
    acc = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(f)):
        acc[(r["model"], r["test"], r["metric"], r["x"])][r["seed"].split(".")[1]].append(float(r["value"]))
    rows = []
    for (m, te, me, x), per in acc.items():
        v = [st.mean(z) for z in per.values()]
        sd = float("nan") if any(z != z for z in v) else (st.stdev(v) if len(v) > 1 else 0.0)
        rows.append({"model": m, "test": te, "metric": me, "x": x, "mean": st.mean(v), "std": sd})
    return rows


def get(rows, model, test, metric):
    if rows is None: return None
    r = next((x for x in rows if x["model"]==model and x["test"]==test
              and x["metric"]==metric), None)
    if r is None: return None
    m, s = float(r["mean"]), float(r["std"])
    return None if m != m else (m, s)


STAGE1 = {"spatial_cnp":"S-AttnCNP","spatial_anp":"S-AttnLNP","spatial_ranp":"S-RecLNP",
          "spatial_rcnp":"S-RecCNP", "rcnp":"RecCNP","ranp":"RecLNP","flat_cnp":"AttnCNP","flat_anp":"AttnLNP"}
BENCH  = {"pfn":"PFN","rcnp":"S-RecCNP","ranp":"S-RecLNP","cnp_ffn":"S-AttnCNP"}


def fig_benchmarks():
    """Fig. 6: B1 and B2 MAE for prefix- and calibration-trained models."""
    specs = [("B1 · unseen deployments", "b1_prefix", "b1_calib"),
             ("B2 · unseen environments",    "b2_prefix", "b2_calib")]
    order = ["pfn","rcnp","cnp_ffn"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
    ymax = 0
    for ax, (title, gp, gc) in zip(axes, specs):
        P, C = load(gp), load(gc)
        if P is None:
            ax.text(.5,.5,"pending",ha="center",va="center",color=INK2,
                    transform=ax.transAxes); ax.set_title(title,loc="left"); continue
        ms = [m for m in order if get(P,m,"region_mae","overall")]
        xs = range(len(ms)); w = 0.38
        for k,(rows,lab,col) in enumerate(((P,"prefix",C1),(C,"calibration",C2))):
            v = [get(rows,m,"region_mae","overall") for m in ms]
            pos = [x + (k-0.5)*w for x in xs]
            ax.bar(pos, [z[0] if z else 0 for z in v], w*0.86, color=col,
                   label=lab, linewidth=0)
            for p, z in zip(pos, v):
                if z: ax.annotate(f"{z[0]:.1f}", (p, z[0]),
                                  ha="center", va="bottom", fontsize="small", color=INK2,
                                  xytext=(0,2), textcoords="offset points")
            ymax = max(ymax, max((z[0] for z in v if z), default=0))
        ax.set_xticks(list(xs)); ax.set_xticklabels([BENCH.get(m,m) for m in ms], fontsize="medium")
        ax.set_title(title, loc="left", fontsize="medium")
        ax.yaxis.set_major_formatter(fmt)
    axes[0].set_ylabel("region MAE [m]")
    axes[0].set_ylim(0, ymax*1.22)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize="medium",
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(f"{FIG}/fig_benchmarks.pdf"); plt.close(fig)
    print("  fig_benchmarks.pdf")


def fig_shortcut():
    """Fig. 4: error against the noise ratio on the shared development set."""
    rows, extra = load("s4_family"), load("s4_family_srec")
    if rows is not None and extra is not None:
        have = {r["model"] for r in rows}
        rows = rows + [r for r in extra if r["model"] not in have]
    if rows is None: print("  (shortcut figure pending)"); return
    # The flat attentive models (80 and 138 m) never learned the task, so their
    # ratio says nothing about how they localize; they are left out.
    SKIP = {"flat_cnp", "flat_anp"}
    pts = []
    for m in sorted({x["model"] for x in rows} - SKIP):
        o = get(rows,m,"region_mae","overall"); nz = get(rows,m,"acoustic_reliance","noise_ratio")
        if o and nz: pts.append((nz[0], o[0], m))
    fig, ax = plt.subplots(figsize=(4.9, 3.1))
    ax.axvspan(0.9, 2.0, color=C2, alpha=0.10, linewidth=0)
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=42, color=C1,
               edgecolor="white", linewidth=1.0, zorder=3)
    # Labels go to the left of points in the right quarter of the log axis.
    xmax = max(p[0] for p in pts)
    for x,y,m in pts:
        right = math.log(x) > 0.75 * math.log(xmax)
        ax.annotate(STAGE1.get(m, m.replace("_"," ")), (x,y),
                    xytext=(-6 if right else 6, 4), textcoords="offset points",
                    ha="right" if right else "left", fontsize="medium", color=INK)
    ax.set_xscale("log"); ax.set_xlabel("noise ratio  (MAE with noise / MAE)")
    ax.set_ylabel("region MAE [m]")
    ax.set_xticks([1,2,5,10,20]); ax.set_xticklabels(["1","2","5","10","20"])
    ax.set_xlim(0.8, xmax*1.5)
    ax.annotate("does not use\nthe acoustics", xy=(1.35, 0.55), xycoords=("data","axes fraction"),
                fontsize="small", color=INK2, ha="center", va="center", linespacing=1.3)
    ax.yaxis.set_major_formatter(fmt)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_shortcut.pdf"); plt.close(fig)
    print("  fig_shortcut.pdf")


def fig_curves_b1():
    """Fig. 7: the four B1 diagnostics (sensor dropout, fix sparsity,
    displacement, drift). S-RecLNP comes from its own group, b1_ranp."""
    P, R = load_seeds("b1_prefix"), load_seeds("b1_ranp")
    if P is None: print("  (B1 curves pending)"); return
    series = [("pfn", P, C1, "o"), ("rcnp", P, C2, "s"),
              ("cnp_ffn", P, C3, "^"), ("ranp", R or P, C4, "D")]
    panels = [("sensor_dropout", "sensors dropped", "(a) sensor dropout", [0, 1, 2, 4, 6]),
              ("fix_sparsity", "context fixes", "(b) fix sparsity", [1, 2, 5, 10]),
              ("displacement", "distance to nearest training centroid [m]", "(c) displacement", None),
              ("drift", "target points after the context", "(d) drift", [0, 1, 2, 3])]
    DRIFT_TICKS = ["1–10", "11–20", "21–30", "31–40"]   # blocks of ten targets
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.6))
    for ax, (test, xl, title, ticks) in zip(axes.flat, panels):
        for m, rows, col, mk in series:
            pts = [(float(x["x"]), float(x["mean"]), float(x["std"]))
                   for x in rows if x["model"] == m and x["test"] == test]
            if not pts: continue
            pts.sort()
            xs, ys, es = (np.array(v) for v in zip(*pts))
            if test == "displacement":           # one point per deployment: 8 quantile bins
                edges = np.quantile(xs, np.linspace(0, 1, 9))
                b = [(xs >= lo) & (xs <= hi) for lo, hi in zip(edges[:-1], edges[1:])]
                xs = np.array([xs[k].mean() for k in b if k.any()])
                ys = np.array([ys[k].mean() for k in b if k.any()])
                es = None
            ax.errorbar(xs, ys, yerr=es, color=col, marker=mk, markersize=4,
                        linewidth=1.4, capsize=2, elinewidth=0.8,
                        label=BENCH[m])
        ax.set_xlabel(xl, fontsize="medium"); ax.set_title(title, loc="left", fontsize="medium")
        if ticks: ax.set_xticks(ticks)
        ax.yaxis.set_major_formatter(fmt); ax.xaxis.set_major_formatter(fmt)
        if test == "drift": ax.set_xticklabels(DRIFT_TICKS)
    for ax in axes[:, 0]: ax.set_ylabel("MAE [m]")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize="medium", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{FIG}/fig_curves_b1.pdf"); plt.close(fig)
    print("  fig_curves_b1.pdf")


if __name__ == "__main__":
    print("Generating figures:")
    fig_benchmarks(); fig_shortcut(); fig_curves_b1()
