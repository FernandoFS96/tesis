"""Emit the paper's .tex table fragments from output/eval_paper/<group>/results.csv.

Idempotent and re-runnable: any table whose evaluation group does not exist yet
is emitted as a loud placeholder in the PDF, so once the evaluation chain
finishes you re-run this script and recompile. It never invents numbers.

siunitx is not used: it is not installed on this machine (kpsewhich cannot find
it), so numbers are formatted by hand -- with a decimal POINT, English style.
"""
import csv, os, glob, re
import statistics as st
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# EVAL_ROOT selects the evaluation tree: output/eval_paper (validation and test
# halves pooled, the default) or output/eval_paper_test (test half only).
EV   = os.path.join(REPO, os.environ.get("EVAL_ROOT", os.path.join("output", "eval_paper")))
# PAPER_OUT redirects output to another manuscript folder (e.g. paper_elsevier)
# without touching the original paper/ tree.
TAB  = os.path.join(os.environ.get("PAPER_OUT", os.path.join(REPO, "paper")), "tables")
os.makedirs(TAB, exist_ok=True)

def load(group):
    f = os.path.join(EV, group, "results.csv")
    if not os.path.exists(f):
        return None
    return list(csv.DictReader(open(f)))

def load_dev(group):
    """Development-set group plus the models that only its <group>_srec re-run
    has (the S-RecCNP, 2026-11-12). Rows of models already in <group> keep their
    published values: the re-run reproduces their MAE exactly but redraws the
    noise of the input controls, which would move ratios in the second decimal."""
    base, extra = load(group), load(group + "_srec")
    if base is None or extra is None:
        return base or extra
    have = {r["model"] for r in base}
    return base + [r for r in extra if r["model"] not in have]

def load_seeds(group):
    """results.csv rows re-aggregated over TRAINING seeds.

    The per-seed label is "<eval seed>.<model index>". In B1 and B2 every transit
    of a held-out deployment is drawn, so the three evaluation seeds see the same
    data and only the ablation noise differs between them: the independent
    replicates are the three training seeds, not nine evaluations. Mean over the
    eval seeds of each training seed, then the sample sd across training seeds.
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
        # stdev raises on nan (the empty "train" region); cell() prints "--" for it
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
    """Region labels actually present, in reading order.

    `sorted()` put extrap before interp, which reads backwards from how the story
    is told (inside the training hull first, outside after). Unknown labels go
    last, alphabetically.
    """
    if rows is None: return []
    have = {x["metric"] for x in rows if x["test"]=="region_mae"} - {"overall", "train"}
    known = [r for r in REGION_ORDER if r in have]
    return known + sorted(have - set(known))

def write(name, body):
    with open(os.path.join(TAB, name), "w") as f:
        f.write(body.rstrip() + "\n")
    print(f"  escrito tables/{name}")

def pending(name, what):
    # Sin f-string: escapar llaves dentro de un f-string ya me dejo una vez un
    # \fbox sin cerrar y el PDF no compilaba.
    write(name, r"\begin{center}\fbox{\parbox{0.86\linewidth}{\centering\small"
                r"\textbf{PENDING}\\[2pt]" + what + r"}}\end{center}")

# Model names follow the Neural Process Family convention: encoder (Attn / Rec)
# + C (deterministic) or L (latent), with S- for the spatial front end. The SAME
# result-file label means different models in different table groups -- "rcnp"
# is the flat recurrent model in the stage-1 runs but the spatial one in B1/B2 --
# so the mapping is per group, never global. Derived from the run configs.
STAGE1 = {"spatial_cnp":"S-AttnCNP", "spatial_anp":"S-AttnLNP", "spatial_ranp":"S-RecLNP",
          "spatial_rcnp":"S-RecCNP", "rcnp":"RecCNP", "ranp":"RecLNP", "flat_cnp":"AttnCNP", "flat_anp":"AttnLNP"}
# The benchmarks report one attentive model: cnp_ffn, whose attention layers are
# complete transformer blocks like the PFN's. The plain "cnp" run (no FFN) is
# quoted once in the text, not tabulated.
BENCH  = {"pfn":"PFN", "rcnp":"S-RecCNP", "ranp":"S-RecLNP", "cnp_ffn":"S-AttnCNP"}
NICE = {
        "full":"full", "no_tokenize":"no tokenize", "no_position":"no position",
        "no_attention":"no attention", "mean_pool":"mean pool",
        "sin_posiciones":"position-free", "con_posiciones":"position-aware",
        "con_pos_notoken":"position-aware, untokenized"}
def nice(m, group=None):
    if group is not None and m in group: return group[m]
    return NICE.get(m, m.replace("_", r"\_"))

# ---------------------------------------------------------------- 4.1 controles
def t_diagnostics():
    rows = load_dev("s4_family")
    src  = "s4_family"
    if rows is None:   # la suite vieja, 2 checkpoints por modelo
        leg = os.path.join(REPO, "output", "eval_suite_3seed", "hermite_main", "results.csv")
        if os.path.exists(leg):
            rows, src = list(csv.DictReader(open(leg))), "eval\\_suite\\_3seed (2 seeds)"
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
    # Como macro, no como texto suelto: \input dentro de \caption{} inserta un
    # \par al final del fichero y LaTeX aborta con "extra }".
    # Escape the underscore: the group name goes straight into \texttt and
    # "s4_family" unescaped is a subscript, which aborts the build.
    write("tab_diagnostics_src.tex",
          r"\newcommand{\diagsource}{Source: \texttt{"
          + src.replace("_", r"\_") + r"}.}")

# ---------------------------------------------------------------- 4.2 A/B
def t_ab():
    # The shared arm is the same checkpoints as Table 3. ab_shared re-ran them and
    # redrew the ablation noise (23.65 against 23.60), so the row comes from
    # s4_family and the two tables quote one number.
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

# ---------------------------------------------------------------- B1 y B2
def t_bench(tag, gp, gc, extra=None, fname=None):
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
        # `extra` WINS over the main group. b1_prefix still carries the original
        # one-seed RANP row, so preferring P silently kept the old 63.23 +- 0.00
        # and ignored the three-seed b1_ranp group entirely.
        src = (X if X and any(x["model"]==m for x in X)
               else (P if any(x["model"]==m for x in P) else P))
        row = [nice(m, BENCH), cell(src,m,'region_mae','overall',with_std=True)]
        row += [cell(src,m,'region_mae',r) for r in regs]
        row += [cell(C,m,'region_mae','overall',with_std=True) if C else "--"]
        row += [cell(C,m,'region_mae',r) if C else "--" for r in regs]
        row += [cell(src,m,'acoustic_reliance','noise_ratio'),
                cell(src,m,'temporal_shuffle','shuffle_ratio')]
        L.append(" & ".join(row) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write(fname, "\n".join(L))

# ---------------------------------------------------------------- B3
def t_b3():
    cells = [("g100_t10",100,10),("g400_t10",400,10),("g1600_t10",1600,10),
             ("g4000_t10",4000,10),("g8000_t10",8000,10),
             ("g1600_t50",1600,50),("g8000_t50",8000,50)]
    have = [(c,g,tj) for c,g,tj in cells if load(f"b3_{c}") is not None]
    if not have:
        return pending("tab_b3.tex", "Benchmark B3: position-free scaling curve.")
    L = [r"\begin{tabular}{rrlrr}", r"\toprule",
         r"\multicolumn{1}{c}{Deploy.} & \multicolumn{1}{c}{Traj.} & Front end "
         r"& \multicolumn{1}{c}{MAE [m]} & \multicolumn{1}{c}{noise} \\",
         r"\midrule"]
    for c,g,tj in have:
        rows = load(f"b3_{c}")
        ms = [m for m in ("sin_posiciones","con_posiciones","con_pos_notoken")
              if any(x["model"]==m for x in rows)]
        for i,m in enumerate(ms):
            L.append(f"{g if i==0 else ''} & {tj if i==0 else ''} & {nice(m)} & "
                     f"{cell(rows,m,'region_mae','overall',with_std=True)} & "
                     f"{cell(rows,m,'acoustic_reliance','noise_ratio')} \\\\")
        L.append(r"\addlinespace")
    missing = [c for c,_,_ in cells if load(f"b3_{c}") is None]
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_b3.tex", "\n".join(L))
    # Los nombres de celda llevan guion bajo: sin escapar, LaTeX los lee como
    # subindice y aborta con "Missing $ inserted".
    esc = lambda z: z.replace("_", r"\_")
    write("tab_b3_note.tex",
          ("Cells still pending: \\texttt{" + "}, \\texttt{".join(esc(m) for m in missing) + "}."
           if missing else "All seven cells measured."))

# ---------------------------------------------------------------- anexo A
def t_ablation():
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


# ------------------------------------------------- B3 en su regimen nativo
def t_b3_native():
    """B3 measured the way it was trained and selected: val MAE, CALIBRATION regime.

    The suite samples context as a causal contiguous prefix, which is right for
    B1 and B2 (the context is the opening of the transit itself). But B3 was
    trained with `data=calib_bb`: 40 context points SCATTERED AT RANDOM over a
    DIFFERENT trajectory. scripts/eval/b3_classical.py measures what that buys a
    least-squares geometry fit: 40 scattered points 9.9 m (median), 40 contiguous
    11.4 m, the suite's 10 contiguous 606 m, so the point count matters far more
    than the spread. Evaluating B3 with the prefix protocol measures it off-regime
    and penalises it systematically. This table uses its native metric; the
    suite's version goes in the appendix so the gap is visible.
    """
    import csv as _csv
    cells = [("g100_t10",100,10),("g400_t10",400,10),("g1600_t10",1600,10),
             ("g4000_t10",4000,10),("g8000_t10",8000,10),
             ("g1600_t50",1600,50),("g8000_t50",8000,50)]
    def vals(exp):
        """Best val MAE per seed, and whether the runs ever left chance level.

        A run whose best val MAE is within 10% of its FIRST epoch never learned:
        it sits at the chance level of the task rather than reporting a weak
        result. Quoting such a number as if it measured an architecture is the
        same error as reading an acoustic-reliance ratio off an unconverged
        model, so the table marks it instead of printing it bare.
        """
        out, chance = [], []
        for f in sorted(glob.glob(os.path.join(REPO,"output","hydra","training","*",exp,"seed_*","train_log.csv"))):
            r=[float(x["val_mae"]) for x in _csv.DictReader(open(f)) if x.get("val_mae")]
            if r:
                out.append(min(r)); chance.append(min(r) > 0.90 * r[0])
        return out, (bool(chance) and all(chance))
    L = [r"\begin{tabular}{rrlr}", r"\toprule",
         r"\multicolumn{1}{c}{Deploy.} & \multicolumn{1}{c}{Traj.} & Front end "
         r"& \multicolumn{1}{c}{val MAE [m]} \\",
         r"\midrule"]
    any_row = False
    for c,g,tj in cells:
        first = True
        for pref,lbl in (("PC","sin_posiciones"),("PA","con_posiciones"),("PAC","con_pos_notoken")):
            v, at_chance = vals(f"{pref}_{c}")
            if not v: continue
            any_row = True
            m = st.mean(v); sd = st.stdev(v) if len(v)>1 else 0.0
            cell = f"{m:.1f}" + " $\\pm$ " + f"{sd:.1f}"
            if at_chance: cell = r"\textit{" + cell + r"}$^{\dagger}$"
            L.append(f"{g if first else ''} & {tj if first else ''} & {nice(lbl)} & "
                     + cell + " \\\\")
            first = False
        if not first: L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]
    if any_row:
        if any("dagger" in x for x in L):
            L.insert(-1, r"\multicolumn{4}{l}{\footnotesize $^{\dagger}$ never left chance "
                         r"level ($\approx 481$ m): the run did not converge, so the number is "
                         r"not a measurement of that variant.} \\")
        write("tab_b3_native.tex", "\n".join(L))
    else: pending("tab_b3_native.tex", "Benchmark B3 in its native regime.")


# ------------------------------------------------- the latent inference gap
def t_latent():
    import csv as _csv
    # latent_gap_v2 (scripts/eval/latent_gap.py) runs Table 3's protocol and
    # checkpoints; the old latent_gap group had one draw and a duplicate seed.
    f = os.path.join(EV, "latent_gap_v2", "results.csv")
    if not os.path.exists(f):
        return pending("tab_latent.tex", "Prior-versus-posterior latent gap.")
    rows = list(_csv.DictReader(open(f)))
    NAME = STAGE1
    L = [r"\begin{tabular}{lrrr}", r"\toprule",
         r"Model & \multicolumn{1}{c}{$z$ from the prior} "
         r"& \multicolumn{1}{c}{$z$ from the posterior} & \multicolumn{1}{c}{ratio} \\",
         r"\midrule"]
    for key in ("spatial_anp", "flat_anp"):
        r = next((x for x in rows if x["model"] == key), None)
        if not r: continue
        pm, ps = float(r["prior_mean"]), float(r["prior_std"])
        qm, qs = float(r["post_mean"]), float(r["post_std"])
        L.append(f"{NAME[key]} & {pm:.2f} $\\pm$ {ps:.2f} & {qm:.2f} $\\pm$ {qs:.2f} "
                 f"& {pm/qm:.2f}$\\times$ \\\\")
    L.append(r"\addlinespace")
    L.append(r"\multicolumn{4}{l}{\footnotesize\itshape no latent: the two columns must coincide} \\")
    for key in ("spatial_cnp", "flat_cnp"):
        r = next((x for x in rows if x["model"] == key), None)
        if not r: continue
        pm, ps = float(r["prior_mean"]), float(r["prior_std"])
        L.append(f"{NAME[key]} & {pm:.2f} $\\pm$ {ps:.2f} & {pm:.2f} $\\pm$ {ps:.2f} "
                 f"& 1.00$\\times$ \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_latent.tex", "\n".join(L))


# ------------------------------------------------- B2 depth-stratified study
def t_depth():
    """Three training arms against the same held-out pool.

    Two comparisons with different control properties, and the table keeps them
    apart on purpose. Specialist-vs-generalist holds the band's own deployments
    fixed and adds the other band's, so it measures transfer between bands, not
    coverage. Specialist-vs-specialist is matched to 1.6% in size and total in
    coverage, so the specialist ratio is clean.
    """
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


# ------------------------------------------------- MFP reference (appendix)
# Rows run from the most to the least knowledge of the environment.
MFP_ROWS = {"B1": [("exact", r"exact (upper bound)"),
                   ("c15", r"sound speed $+15$ m/s"),
                   ("h25", r"water depth $+2.5$ m"),
                   ("compound", r"both, positions off by $2$ m")],
            "B2": [("true", r"exact (upper bound)"),
                   ("focal", r"fitted to $10$ fixes"),
                   ("nominal", r"center of the prior")]}
def t_mfp():
    """Matched-field processing beside the calibration-trained PFN.

    MFP rows come from output/eval_paper/mfp/results.csv (points 11-50 of every
    held-out transit, as the protocol); the PFN rows are the calibration column of
    tab_b1 / tab_b2, so the two tables cannot disagree.
    """
    M = load("mfp")
    if M is None:
        return pending("tab_mfp.tex", "MFP reference on B1 and B2.")
    L = [r"\begin{tabular}{llrrr}", r"\toprule"]
    for bench, group, regs, title in (("B1", "b1_calib", ("interp", "extrap"), "B1: unseen deployments"),
                                      ("B2", "b2_calib", ("shallow", "deep"), "B2: unseen environments")):
        if bench == "B2":
            L.append(r"\midrule")
        L += [fr"\multicolumn{{2}}{{l}}{{\emph{{{title}}}}} & \multicolumn{{1}}{{c}}{{overall}}"
              + "".join(fr" & \multicolumn{{1}}{{c}}{{{r}}}" for r in regs) + r" \\", r"\midrule"]
        for key, label in MFP_ROWS[bench]:
            r = next((x for x in M if x["bench"] == bench and x["variant"] == key), None)
            vals = [f"{float(r[k]):.2f}" if r and r.get(k) else "--" for k in ("overall",) + regs]
            L.append(f"MFP & {label} & " + " & ".join(vals) + r" \\")
        C = load_seeds(group)
        L.append(r"PFN, calibration & none (learned) & "
                 + " & ".join(cell(C, "pfn", "region_mae", k) for k in ("overall",) + regs) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_mfp.tex", "\n".join(L))

if __name__ == "__main__":
    print("Generando tablas:")
    t_diagnostics(); t_ab()
    t_bench("B1", "b1_prefix", "b1_calib", extra="b1_ranp", fname="tab_b1.tex")
    t_bench("B2", "b2_prefix", "b2_calib", fname="tab_b2.tex")
    t_b3(); t_b3_native(); t_ablation(); t_latent(); t_depth(); t_mfp()
