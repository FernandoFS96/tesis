# Project Overview — Sensor-Agnostic Underwater Localization

This document is the single reference for how the repository fits together, from acoustic data generation through preprocessing, model definitions, and training.
It reflects the current state of the code and configs.

---

## 0. Repository map

```
config/                     # Hydra (training/eval) + plain-YAML (generation)
  train.yaml                # composition root: defaults(dataset+data+model+wandb+experiment) + seed/device/training
  data_pipeline.yaml        # data generation + preprocessing (plain OmegaConf, no Hydra)
  dataset/                  # SHARED identity (task/method/mode) read by BOTH generate.py and training
  data/                     # split view + context sampling: _base + geometry | topology | within_geometry
  experiment/              # one file == one runnable experiment (bundles dataset+data+model+exp_name)
  model/                    # cnp | anp | ranp | rcnp | online_ranp
  wandb/default.yaml        # W&B + periodic-visualization config
data/
  generate.py                               # UNIFIED generation entry: generate.py dataset=<name>
  acoustic_data_generator.py                # library: channel physics + topology task runner
  random_position_generator.py              # library: random (position-set) task runner
  process_data.py          # preprocessing -> training-ready pickles (all modes)
  utils/                                    # QC + velocity + trajectory-preview tools
  topology_task/<method>/<topology>/...     # Problem 1 raw + processed data (see §7)
  random_task/<method>_<mode>/...           # Problem 2 raw + processed data (see §7)
src/models/
  anp.py            # CNP (DeterministicModel) + ANP (LatentModel), attentive, offline
  r_anp.py          # RCNP + RANP: recurrent (LSTM/GRU) attentive NPs, offline
  online_r_anp.py   # OnlineLatentModel: streaming/causal RANP
scripts/
  train/train_np_geometry.py  # unified trainer for all 5 model variants
  train/viz.py                # periodic W&B training visualizations
  eval/                       # eval_np_geometry.py, compare_baselines.py, compare_offline_online.py
```

## 1. The two "studies" (mental model)

The whole pipeline serves **two parallel experiments**, both localizing an acoustic source from filtered channel features:

| | **Three-topology study** | **Position-set study (sensor-displacement)** |
|---|---|---|
| Generate | `python data/generate.py dataset=topology_spiral` | `python data/generate.py dataset=random_spiral_shared` |
| Dataset identity | `config/dataset/topology_*.yaml` | `config/dataset/random_*.yaml` |
| Varying factor | sensor **topology** (ellip / rand / alig) | sensor **layout** (N rand sets of the `random` topology) |
| Shared across variants | one traj. ensemble per theta (all 3 topologies) | one ensemble per theta (shared mode) or per-set (distinct mode) |
| Default `df` / feature dim | `df=50` → **2010** | `df=50` → **2010** |
| Processing modes | `topology` | `legacy` (within-geometry), `geometry` (held-out layouts) |

> **Datasets are not cross-compatible**: different `df` ⇒ different feature > dimension. The model's `input_dim` is inferred from the data at train time, so each model is tied to the feature dim it was trained on.

Flow: **generate** (channel physics) → **process** (split into train/val/test pickles) → **train** (Hydra) → **eval**.

---

## 2. Data generation

### 2.1 Acoustic channel & core parameters (`generate_params`)

Both generators reuse the same channel physics. Key knobs:

| Param | Meaning | Notes |
|---|---|---|
| `channel_option` (θ) | channel variability | θ scales the surface/bottom/intra-path variance terms. θ=0.0 ≈ deterministic channel; larger θ ⇒ more randomness. "Low-variance band" = 0.1–0.3. |
| `df` | frequency resolution [Hz] | feature length `Lf = len(range(fmin, fmax, df))` over `fmin=10000, fmax=20000` (B=10000). `df=100→Lf=101`, `df=50→Lf=201`, `df=25→Lf=401`. |
| `n_sensors` | hydrophones | default **10** |
| `n_traj` | trajectories | default 100 |
| `ppt` | points per trajectory | default 50 |
| `snr`, `rep` | filtering SNR [dB] / repetitions | default 10 / 1 |

**Per-point feature dimension = `Lf × n_sensors`** (e.g. 101×10=1010, 201×10=2010).

### 2.2 Sensor topologies (`channel.generate_sensor_positions`)

Layouts are fit to the spatial extent of the trajectory ensemble:
- **ellipsoidal** — sensors on an ellipse (semi-axes `a`, `b=a/2`) around the trajectory centroid.
- **random** — uniform scatter inside a centered box (seeded for reproducibility).
- **aligned** — sensors on a horizontal line through the centroid.

The three-topology generator emits all three sharing one trajectory ensemble, so tpology is the only varying factor. The position-set generator emits only `random`, but with a **different seeded layout per position-set**.

### 2.3 Trajectory shapes (`config/data_pipeline.yaml: method`)

Selected by `method`; each has its own param block. The dispatch is `channel.generate_trajectories → _generate_<method>_trajectories`.

- **`spiral`** — the original outward spiral (`radio∈[250,350]`, `omega∈[0.8,2.5]`, `aux_1=[0,1.4·ppt]`, `aux_2=[0.2·ppt,1.4·ppt]`).
- **`hermite`** — piecewise-cubic Hermite, `n_segments` pieces with C1 continuity (smooth, slowly-turning headings). A newer, more varied trajectory family.

Add a new shape by adding a param block here **and** a matching `_generate_<method>_trajectories` method in `acoustic_data_generator.py`.

### 2.4 Generation — one entry point, two tasks (`data/generate.py`)

A single dispatcher reads `data_pipeline.yaml` (base: `channel:` + `topology_task:`
+ `random_task:` + spiral/hermite params) and layers a `dataset=<name>` identity
(`config/dataset/*.yaml`, the same file training composes). **Run from the repo
root** so `./data/…` resolves to `<repo>/data/…`.

```bash
python data/generate.py dataset=topology_spiral        # Problem 1
python data/generate.py dataset=random_spiral_shared   # Problem 2
# low-level / ad-hoc (no dataset identity), dotlist overrides:
python data/generate.py task=random method=hermite channel.df=100 \
    random_task.n_position_sets=5 channel.channel_options=[0.1,0.2]
```
Internally it calls `acoustic_data_generator.run_topology_task(cfg)` and
`random_position_generator.run_random_task(cfg)` — those two files are now
**libraries** (the physics/trajectory code + task runners), not standalone CLIs.
Both suppress the channel's legacy `./data/channel_option_*` side-effect write.

**Problem 1 — topology task** → `topology_task.out_dir` (default `./data/topology_task`):
```
<out_dir>/<method>/<topology>/channel_option_<theta>/
    trajectory/trajectories.npy                 # (3, n_traj, ppt)  target coords
    filtered_data/filtered_data.npy             # (tau, ppt, n_traj, n_sensors)
    channel_info/sensor_positions_<theta>.npy   # (3, n_sensors)
    channel_info/trajs_<theta>.npy              # (3, n_traj, ppt+1) full trajectory
```
One shared trajectory ensemble per θ; a canonical ordering (`specific=`) keeps
trajectory *i* **row-aligned across all three topologies** (identical paths).

**Problem 2 — random task** → `random_task.out_dir` (default `./data/random_task`):
```
<out_dir>/<method>_<mode>/                  # e.g. spiral_shared, hermite_distinct
    position_set_XX/
      channel_option_<theta>/random/
        trajectory/trajectories.npy
        filtered_data/filtered_data.npy
        channel_info/{sensor_positions_<theta>,trajs_<theta>,channel_h_<theta>}.npy
    _manifest.pkl                           # seeds, positions, method, mode
```
`n_position_sets` datasets of the `random` topology, each a distinct seeded
layout. `<mode>` = `shared` (`distinct_trajectories: false`, one ensemble reused
across sets) or `distinct` (per-set trajectories).

### 2.5 QC / preview tools (`data/utils/`)
- `qc_random_positions.py` — validity + characterization; **auto-detects** the position-set vs three-topology layout. Checks shared-trajectory invariant, distinct layouts, feature-dim consistency, NaNs, dead channels.
- `velocity_check_random_positions.py` — per-θ source-velocity profiles (same dual-layout auto-detection).
- `visualize_trajectories.py` — quick preview of the configured trajectory shape (regenerates from config; defaults `n_traj`/`ppt` from the `channel:` block).

---

## 3. Data processing (`data/process_data.py`)

Config: the `preprocess:` block of `data_pipeline.yaml` (CLI flags override).
`mode` selects the study/split. **All modes emit the same dict-sample schema** so the trainer reads them uniformly:
`{"X": (ppt, feat_dim), "y": (ppt, 3), "theta": float, ...}`.

| Mode | Study | Split axis | Output dir | Notes |
|---|---|---|---|---|
| `topology` | three-topology | **trajectory index** 70/20/10 within each topology, pooling all θ | `<root>/<method>/processed/topology_<name>/` | method-separated; same trajectory indices across θ ⇒ no leakage; one dataset per topology |
| `legacy` | position-set | **within-geometry** 70/20/10 (all layouts seen in train) | `<root>/processed/within_geometry_split/` | reproduces the old repo; val/test = new *trajectories* of *seen* layouts |
| `geometry` | position-set | **held-out layouts** (disjoint train/val/test geometry pools; interp/extrap labels) | `<root>/processed/geometry_split/` | OOD to unseen sensor layouts; adds `sensor_pos`, `geometry_id`, `splits.json` |
| `all` | position-set | `legacy` + `geometry` | both dirs | |

**Split semantics matter a lot for these baselines** (which have *no* spatial
encoder — they never see `sensor_pos`):
- `within_geometry` / `topology`: every layout appears in training ⇒ the model can generalize (this is the task that converges).
- `geometry`: whole layouts held out ⇒ a no-spatial-encoder baseline **cannot** generalize by construction (this is the gap a future spatial encoder closes).

Sample counts examples: within-geometry over 20 geoms × 6 θ × 100 traj ⇒ 8400/2400/1200; topology over 3 θ × 100 traj ⇒ 210/60/30 per topology.

---

## 4. Model definitions (`src/models/`)

All models predict a Gaussian per target point: `mean` and `var = 1e-3 + softplus(...)`. Latent models add a KL term. The decoder body is the **legacy plain `relu(Linear(...))` stack** (no LayerNorm/dropout in the MLP body); LayerNorm lives only inside the attention blocks.

| Model | Class | Latent? | Recurrent? | Convention |
|---|---|---|---|---|
| `cnp` | `anp.DeterministicModel` | no | no | `split` |
| `anp` | `anp.LatentModel` | yes (posterior+KL) | no | `split` |
| `rcnp` | `r_anp.DeterministicModel` | no | LSTM/GRU | `indexed` |
| `ranp` | `r_anp.LatentModel` | yes | LSTM/GRU | `indexed` |
| `online_ranp` | `online_r_anp.OnlineLatentModel` | yes | LSTM/GRU (streaming) | `online` |

**Calling conventions** (handled by `model_forward` in the trainer):
- `split` — caller pre-splits context/target: `forward(cx, cy, tx, ty, beta)`.
- `indexed` — caller passes full sequence + integer index tensors; the model runs its RNN over the whole sequence and splits by index.
- `online` — `forward_streaming(x_seq, y_seq, ctx_idx, chunk_size, beta)`: reveals context ("fixes") causally in chunks; validation uses the **prior** latent (no peeking) to reflect deployment.

**Architecture (ANP / `anp.LatentModel`)** — the reference offline model:
- `LatentEncoder`: `[x,y]` → input proj → 2× self-attention → mean-pool → `relu(penultimate)` → `mu`, `log_var` with `log_var = 3·tanh(log_var)` stabilizer; reparameterized `z`.
- `DeterministicEncoder`: 2× self-attention on context + 2× cross-att. (query=target, keys=context) → per-target representation `r`.
- `Decoder`: concat `[r, z, target_x]` → 3× `relu(Linear)` → `mean`, `var`.
- Prior latent from context; **posterior** latent from all points during training (so the KL term is active). `num_hidden=128`, 4 attention heads.

`r_anp.*` adds a `TemporalEncoder` (LSTM/GRU over the trajectory) before the attentive NP; `online_r_anp` replaces it with a `StreamingTemporalEncoder` + an `OnlineState` ring buffer (`max_context` fixes) for causal, bounded-cost inference.

Model configs (`config/model/*.yaml`): `num_hidden: 128`, `dropout` (0.0 for cnp/anp, 0.2 for recurrent), and `rnn_type/rnn_layers/rnn_dropout` + `max_context` (online) for the recurrent variants.

---

## 5. Training protocol (`scripts/train/train_np_geometry.py`)

Single Hydra entry point for all five models. `train.yaml` composes five groups:
`dataset` (identity: task/method/mode), `data` (split view + context sampling),
`model`, `wandb`, and an opt-in `experiment`. The cleanest way to launch is an
**experiment** (one file pinning dataset+data+model+exp_name):
```bash
python scripts/train/train_np_geometry.py experiment=within_geometry
python scripts/train/train_np_geometry.py experiment=topology data.topology=aligned model=ranp
```
Or compose manually: `dataset=<id> data=<split> model=<name>`. Because the `data`
config derives `data_dir` from `${dataset.method}_${dataset.mode}`, training always
points at the dataset that `generate.py dataset=<same id>` produced — no manual
path/sync bookkeeping.

**Loss.** Gaussian NLL (computed inside the model) plus `beta · KL` for latent models. KL formula is the standard Gaussian KL between prior and posterior, summed over the latent dim and averaged over the batch.

**KL warmup.** `beta` is **linearly warmed 0 → `training.beta` over `training.kl_warmup_epochs`** (per-epoch, latent models only). A constant full KL from epoch 1 causes posterior collapse; set `kl_warmup_epochs: 0` to disable.

**Optimization** (`config/train.yaml`):

| Setting | Value |
|---|---|
| optimizer | AdamW |
| lr | `5.0e-4` |
| weight_decay | `1.0e-4` |
| scheduler | CosineAnnealingLR (`T_max = epochs`) |
| grad clip | 5.0 |
| epochs | 4000 |
| batch_size | 16 |
| beta / kl_warmup_epochs | 1.0 / 500 |
| seed | 0 |
| device | auto (cuda if available) |

**Context / target sampling** (`config/data/*.yaml`, applied in `make_collate`):
- `ctx_min`/`ctx_max`: random context size per batch (train); `val_ctx`: fixed for validation.
- `ctx_sample_mode`: `first` (ordered prefix) or `random` (scattered).
- `exclude_ctx_from_target`: `true` ⇒ targets are the **complement** of context (scored only on unseen points); `false` ⇒ targets are **all** points (standard NP convention, matches the old converging setup).

**Targets & metric.** `y` (source coords) is z-normalized using train-set stats (`normalize_y: true`); predictions are **denormalized** so reported MAE is in physical units (meters). `X` (acoustic features) is used **raw** (no standardization).

**Outputs.** Per Hydra run dir: `best.pt` (selected by val MAE), `last.pt`, `train_log.csv`, `config.yaml`. W&B logs train/val loss/nll/mae, lr, beta, plus periodic figures (`viz.py`) every `every_n_epochs`. The degradation-scatter figure needs `splits.json` and auto-disables for `topology`/`within_geometry`.

### Which `data=` config to use

| data group | Task | Baseline expectation |
|---|---|---|
| `within_geometry` | position-set, within-geometry (target = all points) | **converges** (reproduces old repo) |
| `topology` | three-topology, per-topology | converges (per-topology model) |
| `geometry` | position-set, held-out layouts (OOD) | degrades — no spatial encoder (by design) |

---

## 6. Evaluation (`scripts/eval/`)

- `eval_np_geometry.py` — load a checkpoint, evaluate MAE/uncertainty on val/test.
- `compare_baselines.py`, `compare_offline_online.py` — cross-model comparisons. Checkpoints store a flat config (`num_hidden`, `rnn_*`, `ctx_*`, `beta`,
`kl_warmup_epochs`, feat/out dims, `y_mean/y_std`) so eval can reconstruct the model.

---

## 7. On-disk data layout

Two parallel, self-describing roots under `data/` (created by `generate.py` when
run from the repo root):
```
data/
  topology_task/<method>/<topology>/channel_option_<θ>/…            # Problem 1 (raw)
  topology_task/<method>/processed/topology_<name>/{train,val,test}_data.pkl
  topology_task/<method>/validation/…                               # QC/velocity figures
  random_task/<method>_<mode>/position_set_XX/…/random/…            # Problem 2 (raw)
  random_task/<method>_<mode>/processed/{within_geometry_split,geometry_split}/…
```
- **Run generation from the repo ROOT** so the config's `./data/topology_task`
  and `./data/random_task` land at `<repo>/data/…`. (Running from inside `data/`
  would produce a `data/data/…` nesting.)
- Training resolves a relative `data_dir` against the **repo root**, so training
  can be launched from anywhere (e.g. the repo root).

---

## 8. End-to-end recipes

All three steps key off the same `dataset` identity, so nothing can drift.

**Problem 1 — topology:**
```bash
python data/generate.py dataset=topology_spiral
python data/process_data.py --mode topology    # data_root=./data/topology_task
python scripts/train/train_np_geometry.py experiment=topology data.topology=random
```

**Problem 2 — random, within-geometry** (reproduces the old converging setup):
```bash
python data/generate.py dataset=random_spiral_shared
python data/process_data.py \
    --data-root data/random_task/spiral_shared --mode legacy
python scripts/train/train_np_geometry.py experiment=within_geometry
```

**Problem 2 — random, held-out geometries (OOD):**
```bash
python data/generate.py dataset=random_spiral_shared
python data/process_data.py \
    --data-root data/random_task/spiral_shared --mode geometry
python scripts/train/train_np_geometry.py experiment=geometry_ood
```

**Evaluate** any run (data_dir defaults to the checkpoint's):
```bash
python scripts/eval/eval_np_geometry.py --ckpt <run>/best.pt --out-dir <run>/eval
```

---

## 9. Gotchas / caveats

- **`df` is shared now** (`channel.df`, default 50 → feature dim 2010 for both
  tasks). A model's `input_dim` is tied to the feature dim it trained on, so keep
  `df` consistent between the data you train and evaluate on.
- **No spatial encoder** in any current baseline: they never see `sensor_pos`, so the held-out-`geometry` split is expected to fail — use `within_geometry`/`topology` for tasks the baselines can actually solve.
- **`exclude_ctx_from_target`**: `within_geometry.yaml` sets it `false` (all-points targets, matches old); `geometry.yaml`/`topology.yaml` set it `true`.
- **Use the same `dataset=<id>` for generation and training** — that's the single
  source of truth. Training derives `data_dir` from `${dataset.method}_${dataset.mode}`,
  so a matching `generate.py dataset=<id>` guarantees the paths line up.
- **Both trajectory methods coexist**: topology raw/processed/validation are all
  under `topology_task/<method>/…`, and the random task under
  `random_task/<method>_<mode>/…`, so spiral and hermite never overwrite each other.
- **Preprocess `data_root` differs by mode**: `topology` mode defaults to `./data/topology_task`; the `legacy`/`geometry` modes need `--data-root data/random_task/<method>_<mode>`.
- **Preprocessing is still a manual step** (`process_data.py`) — it
  is not driven by the `dataset` group yet, so pass the matching `--data-root`.
