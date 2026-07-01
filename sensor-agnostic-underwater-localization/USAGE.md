# Usage guide — generate, process, train

A practical, command-by-command walkthrough of the config-driven pipeline. For
the conceptual overview (what each piece *is*), see [OVERVIEW.md](OVERVIEW.md).

> **Run every command from the repository root**
> (`sensor-agnostic-underwater-localization/`). Generation writes to `./data/…`
> relative to the launch dir; training resolves paths against the repo root
> automatically.

The pipeline has three stages, each driven by config:

```
 generate  →  process  →  train  (→ eval)
 (data_pipeline.yaml)  (preprocess block)  (Hydra: dataset+data+model+experiment)
```

Two choices define an experiment end-to-end:
- **dataset identity** = `task` + `method` + `mode` (e.g. `random_spiral_shared`).
  Used by BOTH generation and training so paths always line up.
- **split** = how the generated data is carved into train/val/test
  (`topology` / `within_geometry` / `geometry`).

---

## Stage 1 — Generate data

Entry point: `python data/generate.py` — reads `config/data_pipeline.yaml`
(shared `channel:` knobs + `spiral`/`hermite` params + the two task blocks) and
layers a **dataset identity** from `config/dataset/<name>.yaml`.

### 1a. The two tasks

| Task | What it produces | Output root |
|---|---|---|
| **topology** | one dataset per sensor **topology** (ellipsoidal / random / aligned), all sharing the same trajectories + channels per θ | `data/topology_task/<method>/<topology>/channel_option_<θ>/` |
| **random** | `n_position_sets` datasets of the **random** topology, each a distinct seeded layout | `data/random_task/<method>_<mode>/position_set_XX/channel_option_<θ>/random/` |

Each output dir contains `trajectory/`, `filtered_data/`, and `channel_info/`
(sensor positions + full trajectory).

### 1b. Trajectory shape (`method`) and cross-set mode

- `method`: **`spiral`** (original outward spiral) or **`hermite`** (piecewise-cubic).
- `mode` (random task only): **`shared`** = one trajectory ensemble reused by all
  position-sets (geometry is the only varying factor) — or **`distinct`** = each
  set gets its own trajectories.

### 1c. The dataset identity files (`config/dataset/`)

Each file sets just `task` / `method` / `mode`. Selecting one is the clean way
to generate:

| `dataset=` | task | method | mode | raw output |
|---|---|---|---|---|
| `topology_spiral`         | topology | spiral  | –        | `data/topology_task/…/spiral/…` |
| `topology_hermite`        | topology | hermite | –        | `data/topology_task/…/hermite/…` |
| `random_spiral_shared`    | random   | spiral  | shared   | `data/random_task/spiral_shared/` |
| `random_hermite_shared`   | random   | hermite | shared   | `data/random_task/hermite_shared/` |
| `random_spiral_distinct`  | random   | spiral  | distinct | `data/random_task/spiral_distinct/` |

```bash
python data/generate.py dataset=topology_spiral
python data/generate.py dataset=random_spiral_shared
```
Missing combination (e.g. hermite+distinct)? Override `mode` on the CLI or add a
one-line dataset file:
```bash
python data/generate.py dataset=random_hermite_shared mode=distinct   # -> hermite_distinct
```

### 1d. Shared channel knobs (override any on the CLI, dotlist style)

Defined in `channel:` of `config/data_pipeline.yaml`:

| Knob | Default | Meaning |
|---|---|---|
| `channel_options` | `[0.1,0.2,0.3]` | θ values (channel variability; 0.0≈deterministic). |
| `n_traj` | 100 | trajectories per (θ[, topology/position-set]). |
| `ppt` | 50 | points per trajectory. |
| `df` | 50.0 | freq. resolution → **feature dim = (10000/df + 1) × 10** (df=50→2010, df=100→1010). |
| `snr` | 10.0 | SNR [dB] of the filtered features. |
| `rep` | 1 | filtering repetitions. |
| `master_seed` | 11 | global seed. |
| `nop` | -1 | joblib processes (-1 = all cores). |

Task-specific: `topology_task.topologies`, `random_task.n_position_sets`,
`random_task.distinct_trajectories`.

```bash
# Full theta sweep, higher-res features, more sensor sets:
python data/generate.py dataset=random_spiral_shared \
    channel.channel_options=[0.0,0.1,0.2,0.3,0.4,0.5] channel.df=100 \
    random_task.n_position_sets=20
```

> ⚠️ Generation runs the acoustic channel physics and is **compute-heavy**
> (cost ≈ n_position_sets × n_thetas, or 3 × n_thetas for topology). Start small
> (`channel.n_traj=4 channel.ppt=6 channel.channel_options=[0.1]`) to smoke-test.

### 1e. Preview trajectories (no physics)
```bash
python data/utils/visualize_trajectories.py           # uses config method + channel n_traj/ppt
```

---

## Stage 2 — Process into train/val/test

Entry point: `python data/data_process_random_positions.py` — reads the
`preprocess:` block of `config/data_pipeline.yaml`; CLI flags override. Every
mode emits the **same sample schema**:
`list of {"X": (ppt, feat_dim), "y": (ppt, 3), "theta", "topology"/…}`.

| `--mode` | For task | Split axis | Output dir |
|---|---|---|---|
| `topology` | topology | **trajectory index** 70/20/10, pooling all θ; one dataset per topology | `data/topology_task/processed/topology_<name>/` |
| `legacy` | random | **within-geometry** 70/20/10 (all layouts seen in train) | `data/random_task/<m>_<mode>/processed/within_geometry_split/` |
| `geometry` | random | **held-out layouts** (disjoint geometry pools, interp/extrap labels + `splits.json`) | `…/processed/geometry_split/` |
| `all` | random | `legacy` + `geometry` | both dirs |

`data_root` differs by mode: `topology` defaults to `./data/topology_task`; the
random modes need `--data-root data/random_task/<method>_<mode>`.

```bash
# Topology (default data_root):
python data/data_process_random_positions.py --mode topology

# Random, within-geometry (reproduces the old converging task):
python data/data_process_random_positions.py \
    --data-root data/random_task/spiral_shared --mode legacy

# Random, held-out geometries (OOD robustness study):
python data/data_process_random_positions.py \
    --data-root data/random_task/spiral_shared --mode geometry
```

Useful flags: `--thetas 0.1,0.2` (subset), `--traj-train/--traj-val/--traj-test`
(topology split counts), `--train-geoms/--val-geoms/--test-geoms` (geometry
pools), `--splits-file <splits.json>` (reuse a frozen geometry split),
`--save-dir`, `--method`.

---

## Stage 3 — Train

Entry point: `python scripts/train/train_np_geometry.py` (Hydra). It composes
five groups: **`dataset`** (identity) + **`data`** (split view) + **`model`** +
**`wandb`** + optional **`experiment`**. `data_dir` is derived from the composed
`dataset`, so training points at exactly what `generate.py dataset=<same>` built.

### 3a. Models (`model=`)
`cnp` · `anp` (latent) · `ranp` (recurrent latent) · `rcnp` (recurrent det.) ·
`online_ranp` (streaming). All read `feat_dim` from the data, so df=50/df=100
data just work.

### 3b. Splits (`data=`)
- `data=topology` (+ `data.topology=ellipsoidal|random|aligned`)
- `data=within_geometry`  (random task, within-geometry)
- `data=geometry`         (random task, held-out layouts / OOD)

### 3c. Recommended: run an experiment (`experiment=`)
One file (`config/experiment/*.yaml`) pins dataset + data + model + `exp_name` +
training tweaks:

```bash
# topology model (per topology):
python scripts/train/train_np_geometry.py experiment=topology data.topology=random

# within-geometry (converging baseline):
python scripts/train/train_np_geometry.py experiment=within_geometry

# held-out geometry (OOD; baselines expected to degrade — no spatial encoder):
python scripts/train/train_np_geometry.py experiment=geometry_ood
```

### 3d. Or compose manually
```bash
python scripts/train/train_np_geometry.py \
    dataset=random_spiral_shared data=within_geometry model=ranp
```

### 3e. Common overrides
- Model / seed: `model=cnp`, `seed=1`
- Optim: `training.epochs=4000 training.lr=5e-4 training.kl_warmup_epochs=500`
- Context sampling: `data.ctx_sample_mode=random data.val_ctx=10 data.exclude_ctx_from_target=true`
- Logging: `wandb.enabled=false`, `exp_name=my_run`, `device=cpu`
- Sweep a topology: run the topology experiment 3× with `data.topology=ellipsoidal|random|aligned`.

Outputs land in `output/hydra/training/<date>/<exp_name>/<model>_seed_<seed>/`
(`best.pt`, `last.pt`, `train_log.csv`, `config.yaml`) regardless of launch dir.

### The one consistency rule
Use the **same `dataset=<id>`** at generation and training. For the random task,
training's `data_dir` interpolates `…/${dataset.method}_${dataset.mode}/…`, so a
mismatch points at a folder you never generated. (Topology's processed path is
method-independent, so it reflects whichever method you last generated.)

---

## Stage 4 — Evaluate
```bash
python scripts/eval/eval_np_geometry.py --ckpt <run>/best.pt --out-dir <run>/eval
```
`--data-dir` defaults to the `data_dir` stored in the checkpoint; `splits.json`
(region interp/extrap breakdown) is used if present (geometry split) and skipped
otherwise.

---

## Worked end-to-end examples

**A. Three-topology study, spiral, ANP per topology**
```bash
python data/generate.py dataset=topology_spiral
python data/data_process_random_positions.py --mode topology
for topo in ellipsoidal random aligned; do
  python scripts/train/train_np_geometry.py experiment=topology data.topology=$topo
done
```

**B. Sensor-displacement, within-geometry (reproduce old converging ANP)**
```bash
python data/generate.py dataset=random_spiral_shared
python data/data_process_random_positions.py \
    --data-root data/random_task/spiral_shared --mode legacy
python scripts/train/train_np_geometry.py experiment=within_geometry
```

**C. Sensor-displacement, held-out geometries (OOD robustness)**
```bash
python data/generate.py dataset=random_spiral_shared
python data/data_process_random_positions.py \
    --data-root data/random_task/spiral_shared --mode geometry
python scripts/train/train_np_geometry.py experiment=geometry_ood model=anp
python scripts/eval/eval_np_geometry.py \
    --ckpt output/hydra/training/*/geometry_ood_spiral/anp_seed_0/best.pt \
    --out-dir eval_geometry_ood
```

**D. Hermite variant of any of the above** — swap the dataset id:
`dataset=topology_hermite` or `dataset=random_hermite_shared`, and set
`data.traj_method=hermite` isn't needed (it comes from the dataset identity).

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
| # sensor layouts (random) | any | `random_task.n_position_sets=` |
| Split | topology / within_geometry / geometry | `--mode` (process) + `data=` (train) |
| Model | cnp / anp / ranp / rcnp / online_ranp | `model=` |
