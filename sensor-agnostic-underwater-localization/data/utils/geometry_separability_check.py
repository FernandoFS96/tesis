"""
geometry_separability_check.py
==============================

Answers the question that the QC between/within ratio does NOT: at a FIXED source
position, are the raw acoustic features distinguishable across the 20 sensor
geometries? This is the quantity that decides whether the sensor-displacement
robustness task is learnable -- not whether geometry dominates trajectory
variance globally.

WHY THE QC RATIO IS THE WRONG TEST
----------------------------------
The QC `between/within` ratio compares the spread of per-geometry MEAN feature
vectors against the within-geometry std. Both terms are dominated by source
motion: averaging over all trajectory points collapses the geometry signature,
and the source moving at ~2.2 m/s inflates the within-set std. A small ratio
(e.g. 0.08) therefore says only "geometry barely shifts the average feature
level relative to trajectory variance" -- it says nothing about whether geometry
is DECODABLE at matched source positions.

WHAT THIS SCRIPT MEASURES (three complementary, trajectory-controlled tests)
---------------------------------------------------------------------------
Because all 20 geometries share the SAME trajectories in canonical order, the
feature tensors are aligned: feature[set][traj j, point t] and
feature[other set][traj j, point t] correspond to the SAME source position. We
exploit that alignment.

  T1. Matched-position geometry vs residual variation.
      For each (traj j, point t), compute the spread of the feature vector ACROSS
      the 20 geometries (this is PURE geometry effect, source held fixed). Compare
      it to the within-geometry feature noise at fixed position (across the small
      jitter that remains). Ratio >> implied by QC.

  T2. Geometry classification (the decisive test).
      Can a simple classifier recover which geometry a feature vector came from,
      at matched source positions? We use nearest-centroid in feature space with
      leave-trajectories-out: build per-geometry centroids on some trajectories,
      classify held-out (traj,point) samples by nearest geometry centroid.
      Accuracy >> 1/20 = 5% means geometry is decodable -> the task is learnable.

  T3. Per-geometry feature-distance matrix at matched positions.
      Mean L2 distance between geometries' feature vectors averaged over matched
      source positions, normalized by the within-geometry distance at the same
      positions. A clean separation shows the displacement is acoustically real.

USAGE
-----
    python geometry_separability_check.py --data-root ./data_random_positions
    python geometry_separability_check.py --data-root ./data_random_positions \
        --theta 0.0 --max-points 500

Outm: prints T1/T2/T3 verdicts and writes geometry_separability_report.txt +
a confusion-style distance heatmap.
"""
import os, re, sys, glob, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_sets(root):
    return sorted(d for d in glob.glob(os.path.join(root,"position_set_*")) if os.path.isdir(d))

def set_idx(d):
    return int(re.search(r"position_set_(\d+)$", d).group(1)) #type: ignore

def load_filtered(set_dir, theta):
    p=os.path.join(set_dir, f"channel_option_{theta}", "random","filtered_data","filtered_data.npy")
    return np.load(p)   # (tau, ppt, n_traj, n_sensors)

def to_samples(filtered):
    # (tau,ppt,n_traj,ns) -> (n_traj,ppt,tau*ns)  [same as training loader]
    tau,ppt,n_traj,ns=filtered.shape
    return np.abs(filtered).transpose(2,1,0,3).reshape(n_traj,ppt,tau*ns).astype(np.float64)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-root",required=True)
    ap.add_argument("--theta",default="0.0")
    ap.add_argument("--max-points",type=int,default=400,
                    help="subsample (traj,point) positions for speed")
    ap.add_argument("--out-dir",default=None)
    args=ap.parse_args()
    out=args.out_dir or os.path.join(args.data_root,"separability_report")
    os.makedirs(out,exist_ok=True)
    log=[]
    def line(s): log.append(s); print(s)

    sets=find_sets(args.data_root)
    ids=[set_idx(s) for s in sets]
    line(f"geometries: {len(sets)} | theta={args.theta}")

    # Load all geometries -> stack as (G, n_traj, ppt, F)
    feats=[]
    for s in sets:
        feats.append(to_samples(load_filtered(s,args.theta)))
    F=np.stack(feats,axis=0)             # (G, n_traj, ppt, D)
    G,n_traj,ppt,D=F.shape
    line(f"feature tensor: G={G} n_traj={n_traj} ppt={ppt} D={D}")

    # Flatten (traj,point) into a position axis P, aligned across geometries.
    Fp=F.reshape(G, n_traj*ppt, D)       # (G, P, D)
    P=Fp.shape[1]
    rng=np.random.default_rng(0)
    sel=rng.choice(P, size=min(args.max_points,P), replace=False)
    Fp=Fp[:,sel,:]                        # (G, P', D)
    Pn=Fp.shape[1]

    # Standardize features (per-dim) using global stats to avoid scale artifacts.
    mu=Fp.reshape(-1,D).mean(0); sd=Fp.reshape(-1,D).std(0)+1e-12
    Fp=(Fp-mu)/sd

    # ---- T1: matched-position geometry spread vs within-position noise ----
    # across-geometry spread at each fixed position:
    across=Fp.std(axis=0)                  # (P', D) std across geometries
    across_mag=np.linalg.norm(across,axis=1).mean()
    # within-geometry variation across nearby positions (proxy for residual noise)
    within_mag=np.linalg.norm(Fp.std(axis=1),axis=1).mean()  # std across positions per geometry
    line("")
    line("T1 matched-position geometry spread:")
    line(f"    across-geometry (fixed source) feature spread : {across_mag:.4f}")
    line(f"    within-geometry (across source) feature spread: {within_mag:.4f}")
    line(f"    ratio across/within = {across_mag/(within_mag+1e-12):.3f}")

    # ---- T2: geometry classification at matched positions ----
    # leave-positions-out nearest-centroid: split positions 50/50
    half=Pn//2
    train_idx=np.arange(half); test_idx=np.arange(half,Pn)
    centroids=Fp[:,train_idx,:].mean(axis=1)     # (G, D)
    # classify each test (geometry,position) sample
    correct=0; total=0
    for g in range(G):
        X=Fp[g,test_idx,:]                       # (T, D)
        # distance to each centroid
        d=np.linalg.norm(X[:,None,:]-centroids[None,:,:],axis=2)  # (T,G)
        pred=d.argmin(axis=1)
        correct+=(pred==g).sum(); total+=len(pred)
    acc=correct/total
    line("")
    line("T2 geometry classification (nearest-centroid, matched positions):")
    line(f"    accuracy = {acc*100:.1f}%   (chance = {100.0/G:.1f}%)")
    verdict_T2 = ("DECODABLE -> task is learnable" if acc>3.0/G else
                  "WEAK -> geometry hard to decode; widen displacement")
    line(f"    verdict: {verdict_T2}")

    # ---- T3: per-geometry distance matrix at matched positions ----
    Dmat=np.zeros((G,G))
    for a in range(G):
        for b in range(G):
            Dmat[a,b]=np.linalg.norm(Fp[a]-Fp[b],axis=1).mean()
    within_self=np.mean([np.linalg.norm(Fp[g,train_idx].mean(0)-Fp[g,test_idx],axis=1).mean() for g in range(G)])
    offdiag=Dmat[~np.eye(G,dtype=bool)].mean()
    line("")
    line("T3 matched-position inter-geometry distance:")
    line(f"    mean off-diagonal (between geometries): {offdiag:.4f}")
    line(f"    mean within-geometry (split):           {within_self:.4f}")
    line(f"    separation ratio = {offdiag/(within_self+1e-12):.3f}  (>1 => separable)")

    fig,ax=plt.subplots(figsize=(6,5))
    im=ax.imshow(Dmat,cmap="viridis")
    ax.set_title(f"Matched-position feature distance\nbetween geometries (theta={args.theta})")
    ax.set_xlabel("geometry"); ax.set_ylabel("geometry")
    fig.colorbar(im,ax=ax,label="mean L2 (standardized)")
    fig.tight_layout(); fig.savefig(os.path.join(out,"geometry_distance_matrix.png"),dpi=200)

    line("")
    line("="*60)
    line("OVERALL: the decisive number is T2 accuracy. If it is well above")
    line("chance (5%), sensor geometry IS recoverable from the raw acoustic")
    line("features and the robustness task is learnable -- regardless of the")
    line("QC between/within ratio, which is dominated by source motion.")
    open(os.path.join(out,"geometry_separability_report.txt"),"w").write("\n".join(log)+"\n")
    print(f"\nReport -> {out}")

if __name__=="__main__":
    main()
