"""Write the paper's LaTeX table fragments from the evaluation results.

Reads output/eval_paper/<group>/results.csv (as written by
scripts/eval/eval_suite.py) and writes tables/*.tex. A table whose results are
missing is replaced by a visible placeholder box.

    python paper/scripts/make_tables.py
    PAPER_OUT=path/to/manuscript python paper/scripts/make_tables.py
"""
import csv, os
import statistics as st
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EV   = os.path.join(REPO, os.environ.get("EVAL_ROOT", os.path.join("output", "eval_paper")))
TAB  = os.path.join(os.environ.get("PAPER_OUT", os.path.join(REPO, "paper")), "tables")
os.makedirs(TAB, exist_ok=True)


def load(group):
    f = os.path.join(EV, group, "results.csv")
    if not os.path.exists(f):
        return None
    return list(csv.DictReader(open(f)))


def load_dev(group):
    """Development-set group plus any model that was only evaluated in the
    second pass <group>_srec. Models present in both keep the values of the
    main group."""
    base, extra = load(group), load(group + "_srec")
    if base is None or extra is None:
        return base or extra
    have = {r["model"] for r in base}
    return base + [r for r in extra if r["model"] not in have]


def load_seeds(group):
    """Results re-aggregated over training seeds.

    In B1 and B2 every evaluation seed draws the same transits, so the
    independent replicates are the training seeds: average over evaluation
    seeds first, then take the sample sd across training seeds.
    """
    f = os.path.join(EV, group, "results_per_seed.csv")
    if not os.path.exists(f):
        return None
    acc = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(f)):
        key = (r["model"], r["test"], r["metric"], r["x"])
        acc[key][r["seed"].split(".")[1]].append(float(r["value"]))
    rows = []
    for (m, te, me, x), per in acc.items():
        v = [st.mean(z) for z in per.values()]
        sd = float("nan") if any(z != z for z in v) else (st.stdev(v) if len(v) > 1 else 0.0)
        rows.append({"model": m, "test": te, "metric": me, "x": x, "mean": st.mean(v),
                     "std": sd, "n": str(len(v))})
    return rows


def cell(rows, model, test, metric, dec=2, with_std=False):
    if rows is None: return "--"
    r = next((x for x in rows if x["model"]==model and x["test"]==test
              and x["metric"]==metric), None)
    if r is None: return "--"
    m, s = float(r["mean"]), float(r["std"])
    if m != m: return "--"
    txt = f"{m:.{dec}f}"
    if with_std:
        txt += " $\\pm$ " + f"{s:.{dec}f}"
    return txt


def nval(rows, model):
    if rows is None: return "--"
    r = next((x for x in rows if x["model"]==model and x["test"]=="region_mae"
              and x["metric"]=="overall"), None)
    return r["n"] if r else "--"


REGION_ORDER = ["interp", "extrap", "shallow", "deep"]
def regions(rows):
    """Region labels present in the results, in reading order."""
    if rows is None: return []
    have = {x["metric"] for x in rows if x["test"]=="region_mae"} - {"overall", "train"}
    known = [r for r in REGION_ORDER if r in have]
    return known + sorted(have - set(known))


def write(name, body):
    with open(os.path.join(TAB, name), "w") as f:
        f.write(body.rstrip() + "\n")
    print(f"  wrote tables/{name}")


def pending(name, what):
    write(name, r"\begin{center}\fbox{\parbox{0.86\linewidth}{\centering\small"
                r"\textbf{PENDING}\\[2pt]" + what + r"}}\end{center}")


# Result labels -> names used in the paper. The label "rcnp" is the flat
# recurrent model on the development set but the spatial one on B1 and B2,
# hence two maps.
STAGE1 = {"spatial_cnp":"S-AttnCNP", "spatial_anp":"S-AttnLNP", "spatial_ranp":"S-RecLNP",
          "spatial_rcnp":"S-RecCNP", "rcnp":"RecCNP", "ranp":"RecLNP", "flat_cnp":"AttnCNP", "flat_anp":"AttnLNP"}
BENCH  = {"pfn":"PFN", "rcnp":"S-RecCNP", "ranp":"S-RecLNP", "cnp_ffn":"S-AttnCNP"}
NICE = {"full":"full", "no_tokenize":"no tokenize", "no_position":"no position",
        "no_attention":"no attention", "mean_pool":"mean pool"}
def nice(m, group=None):
    if group is not None and m in group: return group[m]
    return NICE.get(m, m.replace("_", r"\_"))


def t_diagnostics():
    """Input-side controls on the shared development set."""
    rows = load_dev("s4_family")
    if rows is None:
        return pending("tab_diagnostics.tex", "Shortcut diagnostics on \\texttt{random\\_task}.")
    order = ["spatial_cnp","rcnp","spatial_rcnp","spatial_anp","spatial_ranp","ranp","flat_cnp","flat_anp"]
    present = [m for m in order if any(x["model"]==m for x in rows)]
    L = [r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Model & \multicolumn{1}{c}{MAE [m]} & \multicolumn{1}{c}{noise} "
         r"& \multicolumn{1}{c}{zero} & \multicolumn{1}{c}{order} \\",
         r"\midrule"]
    for m in present:
        L.append(f"{nice(m, STAGE1)} & {cell(rows,m,'region_mae','overall',with_std=True)} & "
                 f"{cell(rows,m,'acoustic_reliance','noise_ratio')} & "
                 f"{cell(rows,m,'acoustic_reliance','zero_ratio')} & "
                 f"{cell(rows,m,'temporal_shuffle','shuffle_ratio')} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_diagnostics.tex", "\n".join(L))
    # A macro rather than plain text, so the caption can \input it safely.
    write("tab_diagnostics_src.tex", r"\newcommand{\diagsource}{Source: \texttt{s4\_family}.}")


def t_ab():
    """Data-side control: shared against own trajectories."""
    A, B = load_dev("s4_family"), load_dev("ab_distinct")
    if A is None or B is None:
        return pending("tab_ab.tex", "Trajectory-mode A/B.")
    L = [r"\begin{tabular}{llrrrrr}", r"\toprule",
         r"Model & Trajectories & \multicolumn{1}{c}{MAE [m]} & \multicolumn{1}{c}{interp.} "
         r"& \multicolumn{1}{c}{extrap.} & \multicolumn{1}{c}{noise} & \multicolumn{1}{c}{order} \\",
         r"\midrule"]
    has = lambda m: any(x["model"] == m for x in A + B)
    ms = [m for m in ("spatial_cnp","rcnp","spatial_rcnp") if has(m)]
    for i, m in enumerate(ms):
        for rows, lab in ((A, "shared"), (B, "own")):
            L.append(f"{nice(m, STAGE1) if lab=='shared' else ''} & {lab} & "
                     f"{cell(rows,m,'region_mae','overall',with_std=True)} & "
                     f"{cell(rows,m,'region_mae','interp')} & "
                     f"{cell(rows,m,'region_mae','extrap')} & "
                     f"{cell(rows,m,'acoustic_reliance','noise_ratio')} & "
                     f"{cell(rows,m,'temporal_shuffle','shuffle_ratio')} \\\\")
        if i < len(ms) - 1: L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_ab.tex", "\n".join(L))


def t_bench(tag, gp, gc, extra=None, fname=None):
    """B1 or B2: prefix- and calibration-trained models, with region MAE.

    `extra` is a separate group for models evaluated on their own; when a model
    appears there, its row is taken from it.
    """
    P, C = load_seeds(gp), load_seeds(gc)
    if P is None:
        return pending(fname, f"Benchmark {tag}: no results yet.")
    X = load_seeds(extra) if extra else None
    regs = regions(P) or regions(C)
    order = ["pfn","rcnp","cnp_ffn","ranp"]
    present = [m for m in order
               if any(x["model"]==m for x in P)
               or (C and any(x["model"]==m for x in C))
               or (X and any(x["model"]==m for x in X))]
    ncol = "l" + "r"*(2 + 2*len(regs) + 2)
    head = (r"Model & \multicolumn{1}{c}{prefix} "
            + "".join(fr"& \multicolumn{{1}}{{c}}{{{r}}} " for r in regs)
            + r"& \multicolumn{1}{c}{calib.} "
            + "".join(fr"& \multicolumn{{1}}{{c}}{{{r}}} " for r in regs)
            + r"& \multicolumn{1}{c}{noise} & \multicolumn{1}{c}{order} \\")
    L = [fr"\begin{{tabular}}{{{ncol}}}", r"\toprule", head, r"\midrule"]
    for m in present:
        src = X if X and any(x["model"]==m for x in X) else P
        row = [nice(m, BENCH), cell(src,m,'region_mae','overall',with_std=True)]
        row += [cell(src,m,'region_mae',r) for r in regs]
        row += [cell(C,m,'region_mae','overall',with_std=True) if C else "--"]
        row += [cell(C,m,'region_mae',r) if C else "--" for r in regs]
        row += [cell(src,m,'acoustic_reliance','noise_ratio'),
                cell(src,m,'temporal_shuffle','shuffle_ratio')]
        L.append(" & ".join(row) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write(fname, "\n".join(L))


def t_ablation():
    """Ablation of the spatial front end (Appendix A)."""
    rows = load("s4_ablation")
    if rows is None:
        return pending("tab_ablation.tex", "Spatial-encoder ablation.")
    order = ["full","no_tokenize","no_position","no_attention","mean_pool"]
    present = [m for m in order if any(x["model"]==m for x in rows)]
    L = [r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Variant & \multicolumn{1}{c}{MAE [m]} & \multicolumn{1}{c}{interp.} "
         r"& \multicolumn{1}{c}{extrap.} & \multicolumn{1}{c}{$n$} \\", r"\midrule"]
    for m in present:
        L.append(f"{nice(m)} & {cell(rows,m,'region_mae','overall',with_std=True)} & "
                 f"{cell(rows,m,'region_mae','interp')} & "
                 f"{cell(rows,m,'region_mae','extrap')} & {nval(rows,m)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_ablation.tex", "\n".join(L))


def t_latent():
    """Latent models decoded from the prior and from the posterior (Appendix B),
    from scripts/eval/latent_gap.py."""
    f = os.path.join(EV, "latent_gap_v2", "results.csv")
    if not os.path.exists(f):
        return pending("tab_latent.tex", "Prior-versus-posterior latent gap.")
    rows = list(csv.DictReader(open(f)))
    L = [r"\begin{tabular}{lrrr}", r"\toprule",
         r"Model & \multicolumn{1}{c}{$z$ from the prior} "
         r"& \multicolumn{1}{c}{$z$ from the posterior} & \multicolumn{1}{c}{ratio} \\",
         r"\midrule"]
    for key in ("spatial_anp", "flat_anp"):
        r = next((x for x in rows if x["model"] == key), None)
        if not r: continue
        pm, ps = float(r["prior_mean"]), float(r["prior_std"])
        qm, qs = float(r["post_mean"]), float(r["post_std"])
        L.append(f"{STAGE1[key]} & {pm:.2f} $\\pm$ {ps:.2f} & {qm:.2f} $\\pm$ {qs:.2f} "
                 f"& {pm/qm:.2f}$\\times$ \\\\")
    L.append(r"\addlinespace")
    L.append(r"\multicolumn{4}{l}{\footnotesize\itshape no latent: the two columns must coincide} \\")
    for key in ("spatial_cnp", "flat_cnp"):
        r = next((x for x in rows if x["model"] == key), None)
        if not r: continue
        pm, ps = float(r["prior_mean"]), float(r["prior_std"])
        L.append(f"{STAGE1[key]} & {pm:.2f} $\\pm$ {ps:.2f} & {pm:.2f} $\\pm$ {ps:.2f} "
                 f"& 1.00$\\times$ \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_latent.tex", "\n".join(L))


def t_depth():
    """B2 by training-set depth band (Appendix C): the full model and the
    shallow-only and deep-only specialists, all scored on the same held-out set."""
    F, S, D = load("b2_prefix"), load("ds_shallow"), load("ds_deep")
    if None in (F, S, D):
        return pending("tab_depth.tex", "B2 depth-stratified study.")
    def g(rows, m, k):
        r = next((x for x in rows if x["model"] == m and x["test"] == "region_mae"
                  and x["metric"] == k), None)
        return float(r["mean"]) if r else None
    order = [(k, BENCH[k]) for k in ("pfn","rcnp","cnp_ffn")]
    L = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"& \multicolumn{2}{c}{full (3200)} & \multicolumn{1}{c}{shallow-only}"
         r" & \multicolumn{1}{c}{deep-only} & \multicolumn{2}{c}{deep/shallow ratio} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-4}\cmidrule(lr){5-5}\cmidrule(lr){6-7}",
         r"Model & \multicolumn{1}{c}{shallow} & \multicolumn{1}{c}{deep}"
         r" & \multicolumn{1}{c}{on shallow} & \multicolumn{1}{c}{on deep}"
         r" & \multicolumn{1}{c}{specialist} & \multicolumn{1}{c}{full} \\",
         r"\midrule"]
    for key, nm in order:
        fs, fd = g(F,key,"shallow"), g(F,key,"deep")
        ss, dd = g(S,key,"shallow"), g(D,key,"deep")
        if None in (fs,fd,ss,dd): continue
        L.append(f"{nm} & {fs:.2f} & {fd:.2f} & {ss:.2f} & {dd:.2f} & "
                 f"{dd/ss:.2f}$\\times$ & {fd/fs:.2f}$\\times$ \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_depth.tex", "\n".join(L))


if __name__ == "__main__":
    print("Writing tables:")
    t_diagnostics(); t_ab()
    t_bench("B1", "b1_prefix", "b1_calib", extra="b1_ranp", fname="tab_b1.tex")
    t_bench("B2", "b2_prefix", "b2_calib", fname="tab_b2.tex")
    t_ablation(); t_latent(); t_depth()
