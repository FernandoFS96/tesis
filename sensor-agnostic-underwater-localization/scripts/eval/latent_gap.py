"""
latent_gap.py
=============

Prior-versus-posterior latent gap (Appendix D), on the protocol of Table 3: the
held-out pool of the SHARED development set, six transits per deployment and
three evaluation seeds on each of the three training seeds, the latest run of
each seed (the rule the s4_family evaluation used).

    python scripts/eval/latent_gap.py --out-dir output/eval_paper/latent_gap_v2

The prior column is computed with eval_suite.predict, so it must reproduce the
MAE column of Table 3. The deterministic models carry no latent and are the
control: their two columns must coincide.
"""
import argparse, glob, os, re, sys, csv, warnings
import statistics as st
from collections import defaultdict
import numpy as np
import torch as t

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_suite as E

SPEC = [("spatial_anp", "paper_spatial_anp", True), ("flat_anp", "paper_flat_anp", True),
        ("spatial_cnp", "paper_spatial_cnp_full", False), ("flat_cnp", "paper_flat_cnp", False)]


def ckpts(exp):
    """Latest run per training seed, as for s4_family."""
    best = {}
    for p in glob.glob(f"output/hydra/training/*/{exp}/seed_*/best.pt"):
        s = int(p.split("/seed_")[1].split("/")[0])
        d = re.search(r"/training/([\d-]+)/", p).group(1)
        if s not in best or d > best[s][0]:
            best[s] = (d, p)
    return [p for _, p in sorted(best.values())]


@t.no_grad()
def posterior_mae(b, items, nctx=10):
    """Decoder handed z ~ q(z | C u T), the latent it was trained with."""
    ym, ys, errs = b["ym"], b["ys"], []
    for X, y, gid, sp in items:
        ctx = t.arange(nctx, device=X.device); tgt = t.arange(nctx, X.size(1), device=X.device)
        cy, ty = y[:, ctx], y[:, tgt]
        if ym is not None:
            cy, tyn = (cy - ym) / ys, (ty - ym) / ys
        else:
            tyn = ty
        m = b["model"](X[:, ctx], cy, X[:, tgt], tyn, 1.0,
                       predict_with_prior=False, sensor_pos=sp)[0]
        m = m * ys + ym if ym is not None else m
        errs.append(E.mae_of(m, y, tgt).mean().item())
    return float(np.mean(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/random_task/hermite_shared/processed/geometry_split")
    ap.add_argument("--out-dir", default="output/eval_paper/latent_gap_v2")
    ap.add_argument("--eval-seeds", default="0,1,2")
    ap.add_argument("--cap-traj", type=int, default=6)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    dev = t.device("cuda" if t.cuda.is_available() else "cpu")
    data = E.load_data(args.data_dir, dev, load_cap=max(15, args.cap_traj * 3))

    per = defaultdict(list)   # label -> [(eval seed, training seed, prior, posterior)]
    used = {}
    for label, exp, latent in SPEC:
        paths = ckpts(exp); used[label] = paths
        for mi, p in enumerate(paths):
            b = E.load_model(p, dev)
            for es in (int(x) for x in args.eval_seeds.split(",")):
                E.EVAL_SEED = es; t.manual_seed(es)
                items = E.subsample_test(data["by_gid"], args.cap_traj, es)
                prior, _ = E._mae_over(items, b, None)
                post = posterior_mae(b, items) if latent else prior
                per[label].append((es, mi, prior, post))
                print(f"  {label:<12} seed {mi} eval {es}: prior {prior:7.2f}  posterior {post:7.2f}"
                      f"  ({len(items)} transits)", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "results_per_seed.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "eval_seed", "train_seed", "prior", "post"])
        for label, rs in per.items():
            for es, mi, pr, po in rs:
                w.writerow([label, es, mi, f"{pr:.4f}", f"{po:.4f}"])
    with open(os.path.join(args.out_dir, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        # *_std: population sd over the n evaluations, the convention of Table 3;
        # *_sd_train: sample sd over training seeds after averaging the eval seeds.
        w.writerow(["model", "prior_mean", "prior_std", "post_mean", "post_std",
                    "prior_sd_train", "post_sd_train", "n", "n_train"])
        for label, rs in per.items():
            P = [r[2] for r in rs]; Q = [r[3] for r in rs]
            byt = defaultdict(list)
            for es, mi, pr, po in rs:
                byt[mi].append((pr, po))
            Pt = [st.mean(v[0] for v in z) for z in byt.values()]
            Qt = [st.mean(v[1] for v in z) for z in byt.values()]
            w.writerow([label] + [f"{v:.4f}" for v in (st.mean(P), st.pstdev(P), st.mean(Q),
                        st.pstdev(Q), st.stdev(Pt), st.stdev(Qt))] + [len(P), len(Pt)])
            print(f"{label:<12} prior {st.mean(P):7.2f} +- {st.pstdev(P):5.2f}   posterior "
                  f"{st.mean(Q):7.2f} +- {st.pstdev(Q):5.2f}   ratio {st.mean(P)/st.mean(Q):.2f}"
                  f"   n={len(P)} ({len(Pt)} training seeds)")
    with open(os.path.join(args.out_dir, "checkpoints.txt"), "w") as f:
        for label, paths in used.items():
            f.writelines(f"{label} {p}\n" for p in paths)
    print(f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
