# Usage guide — generate, process, train

A practical, command-by-command walkthrough of the config-driven pipeline. For
the conceptual overview (what each piece *is*), see [OVERVIEW.md](OVERVIEW.md).

> **Run every command from the repository root**
> (`sensor-agnostic-underwater-localization/`). Generation writes to `./data/…`
> relative to the launch dir; training resolves paths against the repo root
> automatically.

```
 generate  →  process  →  train  (→ eval)
 (data_pipeline.yaml)  (preprocess block)  (Hydra: dataset+data+model+experiment)
```

Two choices define an experiment end-to-end:
- **dataset identity** = `task` + `method` + `mode` (e.g. `random_spiral_shared`).
  Used by BOTH generation and training so paths always line up.
- **split** = how the data is carved into train/val/test
  (`topology` / `within_geometry` / `geometry`).

---

## Stage 1 — Generate data

`python data/generate.py` — reads `config/data_pipeline.yaml` (shared `channel:`
knobs + `spiral`/`hermite` params + the two task blocks) and layers a **dataset
identity** from `config/dataset/<name>.yaml`.

### 1a. The two tasks

| Task | What it produces | Output root |
|---|---|---|
| **topology** | one dataset per sensor **topology** (ellipsoidal / random / aligned), all sharing the same trajectories + channels per θ | `data/topology_task/<method>/<topology>/channel_option_<θ>/` |
| **random** | `n_position_sets` datasets of the **random** topology, each a **translated compact sensor array** (per-set random centre offset → real sensor-displacement diversity) | `data/random_task/<method>_<mode>/position_set_XX/channel_option_<θ>/random/` |

Each output dir contains `trajectory/`, `filtered_data/`, `channel_info/`
(sensor positions + full trajectory). The random task also writes `_manifest.pkl`
(seeds, positions, `layout_params`).

### 1b. Dataset identities (`config/dataset/`)

Each file sets just `task` / `method` / `mode`:

| `dataset=` | task | method | mode | raw output |
|---|---|---|---|---|
| `topology_spiral` / `topology_hermite` | topology | spiral / hermite | – | `data/topology_task/<method>/…` |
| `random_spiral_shared` / `random_hermite_shared` | random | spiral / hermite | shared | `data/random_task/<method>_shared/` |
| `random_spiral_distinct` / `random_hermite_distinct` | random | spiral / hermite | distinct | `data/random_task/<method>_distinct/` |

- **`shared`** = one trajectory ensemble reused by every position-set (sensor
  geometry is the ONLY varying factor — the controlled displacement study).
- **`distinct`** = each set gets its own trajectories (realistic deployment).

```bash
python data/generate.py dataset=topology_spiral
python data/generate.py dataset=random_hermite_shared
```

To generate a **single topology** (e.g. a matched single-geometry reference):
```bash
python data/generate.py dataset=topology_spiral topology_task.topologies=[random]
```

### 1c. Shared channel knobs (`channel:`; override on the CLI, dotlist style)

| Knob | Current | Meaning |
|---|---|---|
| `channel_options` | `[0.1, 0.2]` | θ values (channel variability; low band 0.1–0.3). |
| `n_traj` | 400 | trajectories per (θ[, topology / position-set]). |
| `ppt` | 50 | points per trajectory. |
| `df` | 50.0 | freq. resolution → **feat_dim = (10000/df + 1) × n_sensors** (df=50→2010). |
| `snr` / `rep` | 10.0 / 1 | filtering SNR [dB] / repetitions. |
| `master_seed` | 11 | global seed (layout seed = `master_seed + 1000 + set_idx`). |
| `nop` | -1 | joblib processes (-1 = all cores). |

### 1d. Sensor-layout distribution (`random_task.layout`) — the displacement axis

Each position-set places a **compact array whose centre is translated** by a
per-set random offset, so layouts differ by genuine *displacement* (not a
re-jitter of one fixed box). All fractions are of the trajectory field extent, so
they work for any trajectory method.

| Knob | Current | Meaning |
|---|---|---|
| `offset_frac` | 0.3 | max array-centre translation, ± fraction of field extent. **This is the OOD axis.** |
| `aperture_frac` | 0.5 | array span (compactness); keeps localization conditioning comparable to the single-geometry task. |
| `scale_jitter` | 0.0 | optional ± per-set aperture jitter. |
| `min_span` | 20.0 | floor on array span [m]. |

Other task knobs: `topology_task.topologies`, `random_task.n_position_sets`
(current **200**), `random_task.distinct_trajectories`.

```bash
# More layouts + a wider displacement axis (harder OOD):
python data/generate.py dataset=random_spiral_shared \
    random_task.n_position_sets=120 random_task.layout.offset_frac=0.4
```

> ⚠️ Generation runs the channel physics and is **compute-heavy** — cost scales
> ≈ `n_position_sets × n_thetas × n_traj`. A 200×400×2θ pool is ~125 GB raw and
> many hours. Smoke-test first:
> `channel.n_traj=4 channel.ppt=6 channel.channel_options=[0.1] random_task.n_position_sets=4`

### 1e. QC / preview
```bash
python data/utils/visualize_trajectories.py            # trajectory shape preview (no physics)
python data/utils/qc_random_positions.py --data-root data/random_task/<method>_<mode>
python data/utils/geometry_separability_check.py
```

---

## Stage 2 — Process into train/val/test

`python data/process_data.py` — reads the `preprocess:` block of
`config/data_pipeline.yaml`; CLI flags override. Every mode emits the same
sample schema:
`{"X": (ppt, feat_dim), "y": (ppt, 3), "theta", "traj_id", [+ "sensor_pos", "geometry_id"]}`.

`traj_id` and `sensor_pos` are what enable the trajectory subsampling and the
spatial encoder (Stage 3).

| `--mode` | For task | Split axis | Output dir |
|---|---|---|---|
| `topology` | topology | **trajectory index** 70/20/10, pooling all θ; one dataset per topology | `data/topology_task/<method>/processed/topology_<name>/` |
| `legacy` | random | **within-geometry** 70/20/10 (all layouts seen in train) | `data/random_task/<m>_<mode>/processed/within_geometry_split/` |
| `geometry` | random | **held-out layouts** (disjoint geometry pools + interp/extrap labels + `splits.json`) | `…/processed/geometry_split/` |
| `all` | random | `legacy` + `geometry` | both dirs |

`data_root` differs by mode: `topology` defaults to `./data/topology_task`; the
random modes need `--data-root data/random_task/<method>_<mode>`.

**Geometry mode (the displacement study):** whole layouts are held out into
disjoint train/val/test pools; **all trajectories of every geometry are used** in
each pool (novelty at val/test = the *sensor layout*, not the source path).
Held-out layouts are labelled `interp` (inside the training-centroid hull) or
`extrap` (outside), giving a built-in degradation-vs-displacement axis.
`splits.json` freezes the contract — version-control it.

```bash
# Topology, one run PER method (outputs auto-separate by method):
python data/process_data.py --mode topology --method spiral

# Random, held-out geometries (the displacement study):
python data/process_data.py --data-root data/random_task/hermite_shared --mode geometry \
    --train-geoms 160 --val-geoms 20 --test-geoms 20

# Random, both splits at once:
python data/process_data.py --data-root data/random_task/spiral_shared --mode all
```

Useful flags: `--thetas 0.1,0.2`, `--traj-train/--traj-val/--traj-test`
(topology split counts), `--train-geoms/--val-geoms/--test-geoms` (geometry
pools; must sum to `n_position_sets`), `--splits-file <splits.json>` (reuse a
frozen split), `--save-dir`, `--method`.

> Processing a 200×400 pool holds the whole train pool in RAM (~50 GB) and writes
> ~60 GB of pickles. Run the methods **sequentially**, not in parallel.

---

## Stage 3 — Train

`python scripts/train/train_np_geometry.py` (Hydra). Composes five groups:
**`dataset`** (identity) + **`data`** (split view) + **`model`** + **`wandb`** +
optional **`experiment`**. `data_dir` is derived from the composed `dataset`, so
training points at exactly what `generate.py dataset=<same>` built.

### 3a. Models (`model=`)
`cnp` · `anp` (latent) · `ranp` (recurrent latent) · `rcnp` (recurrent det.) ·
`online_ranp` (streaming). All read `feat_dim` from the data.

### 3b. Splits (`data=`)
- `data=topology` (+ `data.topology=ellipsoidal|random|aligned`) — single fixed layout.
- `data=within_geometry` — random task, all layouts seen in training.
- `data=geometry` — random task, **held-out layouts** (the displacement/OOD task; carries `sensor_pos`).

### 3c. Spatial encoder (`model.spatial.*`) — sensor-position-aware front end

Turns each point's flat `feat_dim` vector into per-sensor tokens, tags each with
its Fourier-encoded physical position, attends across sensors and pools →
permutation-equivariant over sensors. **Requires `data=geometry`** (needs
`sensor_pos`); `n_sensors` is inferred from the data. Off ⇒ the flat baseline.

| Knob | Default | Meaning |
|---|---|---|
| `enabled` | false* | master switch (`true` in `model/cnp.yaml` currently). |
| `tokenize` | true | per-sensor tokens (permutation-equivariant). `false` = flat+position control. |
| `use_position` | true | add Fourier position features per sensor. |
| `use_attention` | true | cross-sensor self-attention (else Deep-Sets pooling only). |
| `pooling` | attention | `attention` (learned query) \| `mean`. |
| `norm_acoustic` | layernorm | balance acoustic vs position scale: `layernorm` \| `standardize` (global scale) \| `none`. **Required — `none` collapses to constant prediction.** |
| `n_attn_layers` | 1 | cross-sensor attention depth. |
| `n_fourier_bands` / `min_wavelength` / `max_wavelength` | 8 / 10.0 / 1000.0 | position encoding; wavelengths in **metres**, must span the displacement scale. |
| `pos_dim` | 2 | use (x, y); sensors share a constant depth. |

```bash
# full spatial model
python scripts/train/train_np_geometry.py experiment=spatial_geometry model=cnp
# ablations (attribute the gain):
python scripts/train/train_np_geometry.py experiment=spatial_geometry model.spatial.enabled=false        # flat baseline
python scripts/train/train_np_geometry.py experiment=spatial_geometry model.spatial.tokenize=false       # flat + position (not invariant)
python scripts/train/train_np_geometry.py experiment=spatial_geometry model.spatial.use_position=false   # invariance alone
python scripts/train/train_np_geometry.py experiment=spatial_geometry model.spatial.use_attention=false  # Deep-Sets (no self-attn)
```

### 3d. Data-budget subsampling (`data.max_*`) — "how much data do we need?" sweeps

Cap the **training** pool without regenerating; **val/test geometries stay
fixed**, so every cell of a sweep is comparable. Keeps the first N ids (sorted) →
nested subsets.

| Knob | Meaning |
|---|---|
| `max_train_geometries` | # training sensor layouts (≤ the split's train pool). |
| `max_train_trajectories` | # training source paths; composes → train = (N layouts × M paths). |
| `max_val_trajectories` | # **validation** paths. **Set equal to `max_train_trajectories`** to train and validate on the *same* paths (pure layout-generalization; avoids a train/val path mismatch that inflates late-epoch overfitting). |

```bash
# layout-budget sweep (val/test fixed)
for G in 40 80 120 160; do
  python scripts/train/train_np_geometry.py experiment=spatial_geometry model=cnp \
      data.max_train_geometries=$G exp_name=spatial_cnp_g$G
done
```

### 3e. Early stopping (`training.early_stop_*`)
Stops after `early_stop_patience` epochs with no val-MAE improvement
(> `early_stop_min_delta`). `0`/`null` disables. The counter shows in the tqdm
bar as `es=<counter>/<patience>`. `best.pt` is always the val-MAE minimum.

### 3f. Experiments (`experiment=`)
One file pins dataset + data + model + `exp_name` + training tweaks:

| `experiment=` | What |
|---|---|
| `topology` | single fixed layout, per topology |
| `within_geometry` | random task, all layouts seen |
| `geometry_ood` | random task, held-out layouts, **flat** baseline |
| `spatial_geometry` | random task, held-out layouts, **spatial encoder on** |

### 3g. Common overrides
- Model / seed: `model=cnp`, `seed=1`
- Optim: `training.epochs=500 training.lr=5e-5 training.batch_size=8 training.early_stop_patience=50`
- Context: `data.ctx_min=2 data.ctx_max=40 data.val_ctx=10 data.ctx_sample_mode=first|random data.exclude_ctx_from_target=true`
- Logging: `wandb.enabled=false`, `exp_name=my_run`, `device=cpu`

> **Batch size:** the context size is drawn **once per batch** and shared by all
> samples in it, so batch size controls context-size diversity per epoch — small
> batches (≈8) are strongly preferred, and their gradient noise also regularizes.

Outputs: `output/hydra/training/<date>/<exp_name>/<model>_seed_<seed>/`
(`best.pt`, `last.pt`, `train_log.csv`, `config.yaml`). W&B logs train/val
loss/nll/mae, `es_counter`, and — on the geometry split — `val/mae_interp`,
`val/mae_extrap`, `val/mae_gap_extrap_interp`, plus periodic figures.

> **Validation is deployment-honest:** latent models (`anp`/`ranp`) predict from
> the **prior** at val/eval (no peeking at target labels); training uses the
> posterior (standard ELBO teacher forcing). The CSV `train_mae` for latent
> models is therefore a teacher-forced number — compare models on `val`.

### The one consistency rule
Use the **same `dataset=<id>`** at generation and training — `data_dir` is
derived from the identity (`${dataset.method}` for topology,
`${dataset.method}_${dataset.mode}` for random).

---

## Stage 4 — Evaluate
```bash
python scripts/eval/eval_np_geometry.py --ckpt <run>/best.pt --out-dir <run>/eval \
    --eval-ctx 10 --n-context-draws 3 --ctx-sample-mode first
```
Reports per-pool MAE (train/val/test), degradation vs train, **interp vs extrap
held-out MAE + gap**, a per-geometry CSV and `degradation_curve.png`.
`--data-dir` defaults to the checkpoint's; the spatial encoder is rebuilt from
the checkpoint automatically. `splits.json` is used if present, skipped otherwise.

Other flags: `--eval-ctx-sweep "1,2,5,10,20"` (context-reliance diagnostic),
`--shuffle-temporal` (order-invariance diagnostic), `--chunk-size` (online only).

Cross-model comparisons: `scripts/eval/compare_baselines.py`,
`scripts/eval/compare_offline_online.py`.

---

## Worked end-to-end examples

**A. Single-geometry reference (trajectory-budget study)**
```bash
python data/generate.py dataset=topology_hermite topology_task.topologies=[random]
python data/process_data.py --mode topology --method hermite
for T in 100 150 200 280; do
  python scripts/train/train_np_geometry.py dataset=topology_hermite data=topology \
      data.topology=random model=cnp data.max_train_trajectories=$T exp_name=baseline_cnp_t$T
done
```

**B. Sensor-displacement, held-out layouts — flat baseline vs spatial encoder**
```bash
python data/generate.py dataset=random_hermite_shared
python data/process_data.py --data-root data/random_task/hermite_shared --mode geometry \
    --train-geoms 160 --val-geoms 20 --test-geoms 20
# flat baseline (the gap)
python scripts/train/train_np_geometry.py experiment=geometry_ood model=cnp
# spatial encoder (closes it)
python scripts/train/train_np_geometry.py experiment=spatial_geometry model=cnp
python scripts/eval/eval_np_geometry.py --ckpt <run>/best.pt --out-dir <run>/eval \
    --eval-ctx 10 --n-context-draws 3 --ctx-sample-mode first
```

**C. Within-geometry (aliasing baseline — all layouts seen)**
```bash
python data/process_data.py --data-root data/random_task/spiral_shared --mode legacy
python scripts/train/train_np_geometry.py experiment=within_geometry
```

---

## Quick reference — what can vary

| Axis | Values | Set via |
|---|---|---|
| Task | topology, random | dataset id / `task=` |
| Trajectory method | spiral, hermite | dataset id / `method=` |
| Cross-set mode (random) | shared, distinct | dataset id / `mode=` |
| θ (channel variability) | any list | `channel.channel_options=[…]` |
| Feature dim | via `df` (50→2010, 100→1010) | `channel.df=` |
| # trajectories / points | any | `channel.n_traj=`, `channel.ppt=` |
| # sensor layouts | any | `random_task.n_position_sets=` |
| Displacement axis / array size | fractions of field extent | `random_task.layout.offset_frac=`, `.aperture_frac=` |
| Split | topology / within_geometry / geometry | `--mode` (process) + `data=` (train) |
| Model | cnp / anp / ranp / rcnp / online_ranp | `model=` |
| Spatial encoder + ablations | on/off, tokenize, position, attention, norm | `model.spatial.*` |
| Training data budget | # layouts × # paths (val/test fixed) | `data.max_train_geometries=`, `data.max_train_trajectories=`, `data.max_val_trajectories=` |
| Early stopping | patience / min-delta | `training.early_stop_patience=`, `.early_stop_min_delta=` |
