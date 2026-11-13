"""Paper figures. Vector PDF so they scale cleanly in the document.

Palette: slots 1-3 of the validated catalogue (blue #2a78d6, orange #eb6834,
aqua #1baf7a). Only the subsets the palette certifies are used: 1-2 for bars
and lines (adjacent pairs), 1-3 for scatter (all pairs). Every figure has an
equivalent table in the text, which is the table view the relief rule requires
for the low-contrast slots.
"""
import csv, glob, math, os, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# EVAL_ROOT: output/eval_paper (val+test pooled, default) or output/eval_paper_test.
EV   = os.path.join(REPO, os.environ.get("EVAL_ROOT", os.path.join("output", "eval_paper")))
FIG  = os.path.join(os.environ.get("PAPER_OUT", os.path.join(REPO, "paper")), "figures")
os.makedirs(FIG, exist_ok=True)

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

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
    """Rows re-aggregated over TRAINING seeds (sample sd), as in make_tables.py:
    in B1 the three eval seeds draw the same transits, so n is three, not nine."""
    from collections import defaultdict
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
# One attentive model in the benchmarks (cnp_ffn); see make_tables.py.
BENCH  = {"pfn":"PFN","rcnp":"S-RecCNP","ranp":"S-RecLNP","cnp_ffn":"S-AttnCNP"}
NICE = {"pfn":"PFN","rcnp":"RCNP","ranp":"RANP","cnp":"CNP","cnp_ffn":"CNP+FFN",
        "spatial_cnp":"Spatial CNP","spatial_anp":"Spatial ANP",
        "spatial_ranp":"Spatial RANP","flat_cnp":"Flat CNP","flat_anp":"Flat ANP"}

# ---------------------------------------------------------------- 1. B1 y B2
def fig_benchmarks():
    specs = [("B1 \u00b7 unseen deployments", "b1_prefix", "b1_calib"),
             ("B2 \u00b7 unseen environments",    "b2_prefix", "b2_calib")]
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
            # thin bars, 2 pt of surface gap between adjacent fills
            b = ax.bar(pos, [z[0] if z else 0 for z in v], w*0.86, color=col,
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
    # Legend above the panels: inside the B1 panel it crowded the bar values.
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize="medium",
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(f"{FIG}/fig_benchmarks.pdf"); plt.close(fig)
    print("  fig_benchmarks.pdf")

# ---------------------------------------------------------------- 3. atajos
def fig_shortcut():
    """MAE against the noise ratio. One series with direct labels: for
    scatter the palette only certifies three hues under all-pairs, and there are
    seven points here."""
    # s4_family plus the models only its _srec re-run has (the S-RecCNP); rows of
    # models already in s4_family keep their published values (see make_tables).
    rows, extra = load("s4_family"), load("s4_family_srec")
    if rows is not None and extra is not None:
        have = {r["model"] for r in rows}
        rows = rows + [r for r in extra if r["model"] not in have]
    if rows is None:
        leg = os.path.join(REPO,"output","eval_suite_3seed","hermite_main","results.csv")
        rows = list(csv.DictReader(open(leg))) if os.path.exists(leg) else None
    if rows is None: print("  (shortcut figure pending)"); return
    # Los baselines planos quedan FUERA. Con 79,8 y 138,4 m de MAE no convergen a
    # nada utilizable, y un cociente de ruido cercano a 1 en un modelo que no ha
    # resuelto la tarea no significa "no usa la acustica": significa que no usa
    # nada. Atribuirles un mecanismo seria leer una senal en el ruido.
    SKIP = {"flat_cnp", "flat_anp"}
    pts = []
    for m in sorted({x["model"] for x in rows} - SKIP):
        o = get(rows,m,"region_mae","overall"); nz = get(rows,m,"acoustic_reliance","noise_ratio")
        if o and nz: pts.append((nz[0], o[0], m))
    fig, ax = plt.subplots(figsize=(4.9, 3.1))
    ax.axvspan(0.9, 2.0, color=C2, alpha=0.10, linewidth=0)
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=42, color=C1,
               edgecolor="white", linewidth=1.0, zorder=3)
    # Label to the left when the point sits in the right quarter of the (log)
    # axis, so the name does not run off the canvas ("Spatial CNP" did). On the
    # log scale a linear "x > xmax/3" put the S-AttnLNP label beside the
    # S-RecLNP point.
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
    # in axes coordinates, now that the limits are set
    # A media altura de la banda: arriba chocaba con la etiqueta de RANP, que es
    # justo el punto que ocupa el extremo superior de esa franja.
    ax.annotate("does not use\nthe acoustics", xy=(1.35, 0.55), xycoords=("data","axes fraction"),
                fontsize="small", color=INK2, ha="center", va="center", linespacing=1.3)
    ax.yaxis.set_major_formatter(fmt)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_shortcut.pdf"); plt.close(fig)
    print("  fig_shortcut.pdf")

# ---------------------------------------------------------------- 4. B3
def fig_b3_protocol():
    """How much each front end loses when the context protocol changes.

    The parity between position-free and position-aware holds only in the
    CALIBRATION regime. Under the suite's contiguous-prefix protocol the
    position-aware model is barely affected (1.1x) while the position-free one
    loses a factor of ~2.2. That asymmetry is the evidence that the position-free
    front end is actually recovering geometry from the calibration transit:
    remove the well-conditioned context and the recovery fails.
    """
    import os
    def native(exp):
        v=[]
        for f in sorted(glob.glob(os.path.join(REPO,"output","hydra","training","*",exp,"seed_*","train_log.csv"))):
            r=[float(x["val_mae"]) for x in csv.DictReader(open(f)) if x.get("val_mae")]
            if r: v.append(min(r))
        return st.mean(v) if v else None
    # Etiquetas cortas: "8000 deploy." es comun a las cuatro barras y repetido
    # bajo cada una se solapaba con la vecina; va al titulo.
    cells = [("g8000_t10","10 traj./deploy."), ("g8000_t50","50 traj./deploy.")]
    arms  = [("PC","position-free","sin_posiciones"), ("PA","position-aware","con_posiciones")]
    rows=[]
    for cell, clab in cells:
        for pref, alab, mlab in arms:
            n = native(f"{pref}_{cell}")
            r = load(f"b3_{cell}")
            sv = get(r, mlab, "region_mae", "overall") if r else None
            if n is None or sv is None: continue
            rows.append((f"{alab}\n{clab}", n, sv[0]))
    if not rows: print("  (B3 protocol panel pending)"); return
    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    xs = range(len(rows)); w = 0.38
    for k,(vals,lab,col) in enumerate(((  [r[1] for r in rows], "calibration context", C1),
                                       (  [r[2] for r in rows], "contiguous prefix",   C2))):
        pos=[x+(k-0.5)*w for x in xs]
        ax.bar(pos, vals, w*0.86, color=col, label=lab, linewidth=0)
        for px,v in zip(pos,vals):
            ax.annotate(f"{v:.0f}", (px,v), ha="center", va="bottom", fontsize=7.6,
                        color=INK2, xytext=(0,2), textcoords="offset points")
    ax.set_xticks(list(xs)); ax.set_xticklabels([r[0] for r in rows], fontsize=8)
    ax.set_ylabel("MAE [m]"); ax.yaxis.set_major_formatter(fmt)
    ax.set_ylim(0, max(max(r[1],r[2]) for r in rows)*1.25)
    ax.set_title("B3 at 8000 distinct deployments", fontsize=9.5, loc="left")
    ax.legend(loc="upper left", fontsize=8.2, bbox_to_anchor=(0.0, 0.94))
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_b3_protocol.pdf"); plt.close(fig)
    print("  fig_b3_protocol.pdf")

def fig_b3():
    """Scaling curve in B3's NATIVE regime (val MAE), not the suite's."""
    def vals(exp):
        out=[]
        for f in sorted(glob.glob(os.path.join(REPO,"output","hydra","training","*",exp,"seed_*","train_log.csv"))):
            r=[float(x["val_mae"]) for x in csv.DictReader(open(f)) if x.get("val_mae")]
            if r: out.append(min(r))
        return out
    G = [100,400,1600,4000,8000]
    series = [("position-free", "PC", C1), ("position-aware", "PA", C2)]
    fig, ax = plt.subplots(figsize=(4.9, 3.0))
    for lab, pref, col in series:
        xs, ys, es = [], [], []
        for g in G:
            v = vals(f"{pref}_g{g}_t10")
            if v: xs.append(g); ys.append(st.mean(v)); es.append(st.stdev(v) if len(v)>1 else 0)
        if not xs: continue
        ax.errorbar(xs, ys, yerr=es, color=col, linewidth=2.0, marker="o",
                    markersize=5.5, markeredgecolor="white", markeredgewidth=1.0,
                    capsize=2.5, elinewidth=0.8, label=lab, zorder=3)
        # No direct labels at the right edge: the two series CONVERGE there
        # (~103 m both), which is precisely the finding, so any label on the last
        # point collides with the other. The legend carries identity and a single
        # callout tells the finding.
    # 50 trajectories per deployment, as a loose reference
    xs2, ys2 = [], []
    for g in (1600, 8000):
        v = vals(f"PC_g{g}_t50")
        if v: xs2.append(g); ys2.append(st.mean(v))
    if xs2:
        ax.plot(xs2, ys2, color=C1, linewidth=2.0, linestyle=(0,(4,2)),
                marker="s", markersize=5, markeredgecolor="white",
                markeredgewidth=1.0, label="position-free, 50 traj./deploy.", zorder=3)
    ax.set_xscale("log"); ax.set_xticks(G); ax.set_xticklabels([str(g) for g in G])
    ax.set_xlabel("distinct deployments seen in training")
    ax.set_ylabel("val MAE, calibration regime [m]")
    ax.set_ylim(0, 540); ax.yaxis.set_major_formatter(fmt)
    # The callout goes in the right margin, outside the data: placed inside, its
    # leader line crossed the orange series.
    ax.set_xlim(85, 26000)
    ax.annotate("at 8000 deployments\nthe two series\nmeet (~103 m)",
                xy=(8400, 103), xytext=(10500, 180), textcoords="data",
                ha="left", va="center", fontsize=7.4, color=INK2, linespacing=1.4,
                arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8,
                                shrinkA=2, shrinkB=2))
    ax.legend(loc="lower left", fontsize=7.8)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_b3.pdf"); plt.close(fig)
    print("  fig_b3.pdf")

# ---------------------------------------------------------------- B1 curves
def fig_curves_b1():
    """Appendix F: the four B1 diagnostic curves in one 2x2 figure, paper names.

    Replaces the eval_suite PNGs, which carried the raw run labels. S-RecLNP is
    taken from the three-seed b1_ranp group, as in Table 5.
    """
    import numpy as np
    P, R = load_seeds("b1_prefix"), load_seeds("b1_ranp")
    if P is None: print("  (B1 curves pending)"); return
    C4 = "#eda100"                       # categorical slot 4, after C1-C3
    series = [("pfn", P, C1, "o"), ("rcnp", P, C2, "s"),
              ("cnp_ffn", P, C3, "^"), ("ranp", R or P, C4, "D")]
    panels = [("sensor_dropout", "sensors dropped", "(a) sensor dropout", [0, 1, 2, 4, 6]),
              ("fix_sparsity", "context fixes", "(b) fix sparsity", [1, 2, 5, 10]),
              ("displacement", "distance to nearest training centroid [m]", "(c) displacement", None),
              ("drift", "target points after the context", "(d) drift", [0, 1, 2, 3])]
    # eval_suite bins drift targets in blocks of ten points after the 10-point context
    DRIFT_TICKS = ["1\u201310", "11\u201320", "21\u201330", "31\u201340"]
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.6))
    for ax, (test, xl, title, ticks) in zip(axes.flat, panels):
        for m, rows, col, mk in series:
            pts = [(float(x["x"]), float(x["mean"]), float(x["std"]))
                   for x in rows if x["model"] == m and x["test"] == test]
            if not pts: continue
            pts.sort()
            xs, ys, es = (np.array(v) for v in zip(*pts))
            if test == "displacement":           # per-deployment points: 8 quantile bins
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
        if test == "drift": ax.set_xticklabels(DRIFT_TICKS)   # after the formatter, or it wins
    for ax in axes[:, 0]: ax.set_ylabel("MAE [m]")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize="medium", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{FIG}/fig_curves_b1.pdf"); plt.close(fig)
    print("  fig_curves_b1.pdf")

if __name__ == "__main__":
    print("Generating figures:")
    fig_benchmarks(); fig_shortcut(); fig_b3(); fig_b3_protocol(); fig_curves_b1()
