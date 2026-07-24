# Project Overview — Sensor-Agnostic Underwater Localization

## 1. From a single geometry to a sensor-agnostic meta-model

The starting point was a working but **narrow** system: localize an underwater
source from the acoustic channel it induces on a **fixed** array of moored
sensors. On a single, known sensor topology this is a well-posed regression —
`acoustics → position` — and every neural-process baseline (CNP, ANP, and their
recurrent variants) solved it to comparable accuracy.

The thesis goal is to lift that system into a **sensor-agnostic meta-model**: one
model that localizes correctly *regardless of where the sensors are*, and in
particular that stays accurate when the array is **displaced between deployments**
— sensors are moored, drift, and are re-laid; you cannot assume the training
layout at test time. The system must treat a new layout as a new instance of the
same physics, not as a distribution shift it has never seen.

## 2. The data this demanded

A fixed-topology dataset cannot test layout generalization, so the project
introduced the **random task**: many randomly-placed sensor layouts, each a
compact array whose centre is translated by a per-set random offset — so layouts
differ by genuine **displacement**, the axis of interest. This yields the
controlled study (`shared` mode: one trajectory ensemble reused across all
layouts, so geometry is the *only* varying factor) and a realistic variant
(`distinct` mode: own trajectories per layout).

Two evaluation splits express the meta-learning question:

- **`layout_seen`** — every layout appears in training (val/test are new
  trajectories of seen layouts). Isolates the *aliasing* problem.
- **`layout_ood`** — whole layouts are **held out**, and each is labelled
  **interp** (its array centre falls inside the training hull) or **extrap**
  (outside it). This is the headline sensor-displacement test.

## 3. Why the existing models were insufficient

- **Flat CNP / ANP** consume a raw, flattened acoustic vector with no notion of
  *where* each sensor is. When the layout changes, the `acoustics → position` map
  changes, and a model that memorised it for training layouts **collapses** on
  held-out ones. Empirically, moving the flat CNP from the fixed-topology task to
  `layout_ood` degraded it severely; a first sensor-aware model already recovered
  roughly **70%** of that lost accuracy on the spiral OOD task.

- **Recurrent models (RCNP / RANP)** appeared to "solve" `layout_ood` with the
  lowest error — but diagnostics showed this was an artefact, not localization.
  Corrupting every target point's acoustics changed their error by only ~**10%**
  (a genuine localizer degrades by an order of magnitude), while shuffling the
  trajectory order collapsed them almost completely. They were **dead-reckoning**:
  extrapolating a smooth path from the early position fixes while *ignoring the
  layout-dependent acoustics entirely* — which is exactly why they looked
  "displacement-robust" (they never used the layout). That strategy fails the
  operating regime that motivates the whole project: a no-GPS deployment with
  **sparse** position fixes and **long** horizons, where dead-reckoning drifts
  without bound.

The task therefore needed a model that performs *genuine, displacement-robust
acoustic localization*, and an evaluation that rewards it rather than the
extrapolation shortcut.

## 4. The Spatial Neural Process

The core contribution is a **sensor-position-aware front end** placed before the
neural-process encoders. Instead of a scrambled flat vector, each trajectory
point becomes a *set of per-sensor tokens*: each token pairs that sensor's
acoustic slice with a **Fourier encoding of its physical position**, tokens
attend across sensors, and the set is pooled to a per-point embedding. The result
is **permutation-equivariant over sensors** and explicitly geometry-aware — a
displaced layout is simply a new set of `(position, measurement)` pairs, not an
unrecognizable input. Ablation flags expose the ladder (tokenization → position →
cross-sensor attention → pooling) so each ingredient's contribution is attributable.

Reaching a fair, trustworthy comparison also required fixing the **foundations**
shared by all models:

- **ELBO correction** — the latent models' KL was mis-scaled relative to the
  reconstruction term, over-weighting it by ~two orders of magnitude and
  collapsing the posterior; renormalizing to per-target-point units revived the
  latent.
- **Decoder parity & deterministic evaluation** — equalized regularization
  across decoders and switched latent models to predict from the distribution
  mean at test time, so comparisons differ by *architecture*, not incidental noise.
- **Deployment-causal evaluation** — the evaluator now enforces that context
  fixes precede the targets (no peeking at a *future* fix), so recurrent numbers
  are measured under the real no-GPS-then-localize protocol rather than an
  optimistic one.

## 5. Tuning the Spatial NP to its optimum

Optimization proceeded diagnostically rather than by blind search:

1. **Underfitting, not over-fitting.** On `layout_ood` the spatial CNP's train
   and validation error tracked each other almost exactly — the model
   *generalized to unseen layouts perfectly* but under-fit even the training
   ones (its error sat far above the single-fixed-layout floor). The bottleneck
   was **representational capacity**, not data or regularization.

2. **Capacity: depth ≫ width.** Adding a second cross-sensor attention layer cut
   error by ~**19%**; a third layer and doubling width added only marginal gains
   and began to over-fit. Acoustic-normalization choice made no meaningful
   difference. Depth — *rounds of cross-sensor geometric reasoning* — was the
   strong axis, and it **saturated**: two very different capacity allocations
   reached the same validation ceiling.

3. **The residual was displacement-structured.** With capacity exhausted, the
   remaining error was almost entirely on **extrap** layouts: in-hull (interp)
   error nearly matched training, while out-of-hull layouts carried a ~**30%**
   relative penalty. The absolute-frame position encoding could not extrapolate
   to array centres it never saw — a *coverage* problem, not a capacity one.

4. **Translation augmentation closed it.** Because the simulated channel depends
   only on *relative* source–sensor geometry, rigidly shifting `(sensor_pos, y)`
   per sample during training is **physically exact** and synthesizes unlimited
   displaced layouts along precisely the OOD axis, at zero data-generation cost.
   This eliminated the extrap penalty (from ~+30% down to ~**0%**, extrap now on
   par with interp) and dropped overall validation error by a further ~**27%**.

Cumulatively, tuning reduced the spatial CNP's `layout_ood` error by ~**45%**
from the first working version to the augmented model.

Supporting efficiency work made this iteration practical: a data-pipeline fix
(parallel workers + pinned memory) cut epoch time by ~**47%**; a per-sample
context-batching scheme decoupled context diversity from batch size (in a
controlled test it improved equal-wall-clock accuracy by ~**44%**); and
per-worker RNG seeding removed a hidden diversity loss when scaling workers.

## 6. Results we are satisfied with

On the erratic-trajectory `layout_ood` test set, the augmented Spatial CNP is now
the **best model on every axis**, including the headline metric that previously
favored the recurrent baseline:

- **Headline accuracy:** ~**21%** lower error than the RCNP — the model that had
  appeared to render the contribution unnecessary is now beaten on its own metric.
- **Displacement robustness:** the interp-vs-extrap gap is **eliminated** (~0%),
  versus a ~30% penalty before augmentation.
- **Genuine localization:** corrupting target acoustics inflates its error by
  more than an order of magnitude (it *uses* the acoustics), and it is completely
  invariant to trajectory-order shuffling — the opposite profile to the
  dead-reckoning recurrent baseline.
- **No-GPS / fix-scarce regime:** as position fixes drop toward one, its error
  barely moves, while the recurrent baseline's error explodes — at a single fix
  the Spatial CNP's error is ~**88%** lower.
- **Bounded error:** its accuracy is flat across the prediction horizon, whereas
  the recurrent baseline drifts as it moves away from the fixes.

The one regime where the recurrent model still leads is the first few points
*immediately after dense fixes* — short-horizon, fix-rich, i.e. the least
deployment-relevant corner — which points to a natural future direction: a
**spatial-recurrent fusion** combining the Spatial NP's flat, displacement-robust
localization with the recurrent model's near-fix precision.

## 7. Status

The Spatial NP is validated as a genuine sensor-agnostic localizer that is robust
to sensor displacement and to sparse position fixes, and that outperforms all
prior baselines on the headline task. Remaining consolidation (multi-seed
confirmation, an augmentation-strength saturation sweep, cross-trajectory-type
replication, and the spatial-recurrent fusion) is planned but does not change the
central result.
