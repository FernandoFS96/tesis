# Usage

Practical guide to the four-stage pipeline: **generate → process → train → evaluate**.
Run every command from the repository root. Data generation writes to `./data/…`;
training/eval resolve paths against the repo root automatically.

---

## Repository structure

```
config/                     # Hydra config (composed at run time)
  data_pipeline.yaml        # data generation + preprocessing (plain OmegaConf, no Hydra)
  dataset/                  # dataset IDENTITY (task/method/mode) shared by generation AND training
  data/                    # split view + context sampling: _base + layout_ood | layout_seen | topology
  model/                   # cnp | anp | spatial_cnp | spatial_anp | ranp | rcnp | online_ranp (+ _spatial/_rnn_base fragments)
  experiment/              # dataset+split bundles: layout_ood | layout_seen | topology
  train.yaml               # top-level training config (composes the groups above)
  wandb/                   # logging + periodic figures
data/                       # data-generation & preprocessing code
  generate.py              # single entry point for generation (dataset=<identity>)
  process_data.py          # raw -> train/val/test splits
  acoustic_data_generator.py, random_position_generator.py   # channel physics + layout generation
  utils/                   # QC / visualization helpers
src/models/                 # anp.py (CNP/ANP + SpatialEncoder), r_anp.py (RANP/RCNP), online_r_anp.py
scripts/
  train/train_np_geometry.py   # unified trainer for all model families
  eval/eval_np_geometry.py     # robust evaluation battery
  eval/compare_baselines.py, compare_offline_online.py
```

Two choices define an experiment end-to-end: the **dataset identity**
(`task` + `method` + `mode`, e.g. `random_hermite_shared`) and the **split**
(`topology` / `layout_seen` / `layout_ood`). The same `dataset=<id>` is used for
generation and training, so paths always line up.

---

## Stage 1 — Generate data

`python data/generate.py dataset=<identity>` reads `config/data_pipeline.yaml`
(shared `channel:` knobs + trajectory params + the two task blocks) and the
chosen `config/dataset/<id>.yaml`.

- **Tasks**: `topology` (one dataset per fixed sensor arrangement — the original
  single-geometry problem) and `random` (N randomly-placed sensor layouts, each a
  compact array translated by a per-set offset — the sensor-displacement study).
- **Methods** (trajectory shape): `spiral` (smooth) | `hermite` (erratic).
- **Modes** (random task only): `shared` (one trajectory ensemble reused by every
  layout — geometry is the only varying factor) | `distinct` (own trajectories per layout).

```bash
python data/generate.py dataset=topology_hermite
python data/generate.py dataset=random_hermite_shared
# override any channel knob dotlist-style:
python data/generate.py dataset=random_spiral_shared \
    random_task.n_position_sets=200 random_task.layout.offset_frac=0.4
```

Key knobs (`config/data_pipeline.yaml`): `channel.df` (frequency resolution →
feature dim), `channel.snr`, `channel.n_traj`, `random_task.n_position_sets`
(layout diversity — the axis that matters for OOD), `random_task.layout.offset_frac`
(displacement magnitude). Generation is compute-heavy; smoke-test with small
`n_traj`/`n_position_sets` first.

---

## Stage 2 — Process into train/val/test

`python data/process_data.py` reads the `preprocess:` block (CLI flags override).
`--mode` selects the split:

| `--mode` | Task | Split axis | Output |
|---|---|---|---|
| `topology` | topology | trajectory index 70/20/10 | `…/topology_<name>/` |
| `legacy` | random | **seen-layout** (all layouts in train) → feeds `data=layout_seen` | `…/within_geometry_split/` |
| `geometry` | random | **held-out displaced layouts** (interp/extrap labels + `splits.json`) → feeds `data=layout_ood` | `…/geometry_split/` |
| `all` | random | `legacy` + `geometry` | both |

```bash
python data/process_data.py --mode topology --method hermite
python data/process_data.py --data-root data/random_task/hermite_shared --mode geometry \
    --train-geoms 160 --val-geoms 20 --test-geoms 20
```

The geometry mode holds whole layouts out into disjoint pools and labels held-out
layouts `interp` (inside the training-centroid hull) or `extrap` (outside), giving
a built-in degradation-vs-displacement axis. Run modes sequentially (large in-RAM pool).

---

## Stage 3 — Train

`python scripts/train/train_np_geometry.py` — Hydra composes `dataset` + `data` +
`model` + `wandb` (+ optional `experiment`). `exp_name` is auto-derived from the
composition, so run/output names are always truthful.

**Models** (`model=`): `cnp` · `anp` · `spatial_cnp` · `spatial_anp` (spatial
front end on) · `ranp` · `rcnp` · `online_ranp`. Each config carries `arch` (what
is built) and `name` (display id).

**Splits** (`data=`): `layout_ood` (held-out displaced layouts — the headline OOD
task; carries `sensor_pos`) · `layout_seen` (all layouts seen) · `topology` (single
fixed layout).

```bash
python scripts/train/train_np_geometry.py                                   # defaults (cnp)
python scripts/train/train_np_geometry.py model=spatial_cnp                 # spatial model
python scripts/train/train_np_geometry.py experiment=topology model=rcnp    # dataset+split bundle
```

Key knobs:

| Knob | Meaning |
|---|---|
| `model.num_hidden`, `model.spatial.n_attn_layers` | encoder width / cross-sensor attention depth |
| `model.spatial.{tokenize,use_position,use_attention,pooling,norm_acoustic}` | spatial ablation ladder |
| `data.context.{min,max,eval,sample_mode,exclude_from_target}` | context-set sampling (`sample_mode=first`+`exclude_from_target=true` = deployment-causal) |
| `data.context.per_sample` | per-sample context sizes (padded+masked) — decouples context diversity from batch size (cnp/spatial_cnp only) |
| `data.trans_aug_m` | translation augmentation: metres of per-sample rigid scene shift (train only; spatial models only) |
| `data.max_train_geometries`, `data.max_train_trajectories` | data-budget caps for sweeps |
| `training.{epochs,batch_size,lr,num_workers,early_stop_patience}` | optimisation; `num_workers>0` enables `pin_memory`+`persistent_workers` |
| `training.beta`, `training.kl_warmup_epochs` | KL weight (latent models; normalized ELBO units) |

Outputs per run dir: `best.pt` (min val MAE), `last.pt`, `train_log.csv`
(train/val loss·nll·kl·mae), resolved `config.yaml`.

---

## Stage 4 — Evaluate

`python scripts/eval/eval_np_geometry.py --ckpt <run>/best.pt --out-dir <run>/eval`
rebuilds the model from the checkpoint and runs the robust battery. `data_dir`
defaults to the checkpoint's.

Key flags:

| Flag | Purpose |
|---|---|
| `--eval-ctx N` / `--n-context-draws K` | context size and number of random draws to average |
| `--eval-ctx-sweep 1,2,5,10,20` | MAE-vs-context sweep (fix-sparsity / temporal-reliance diagnostic) |
| `--shuffle-temporal` | permute point order (recurrent models collapse; per-point localizers are invariant) |
| `--require-causal` | hard-fail unless deployment-causal (`ctx=first` prefix + complement targets); use for no-GPS numbers |
| `--ctx-sample-mode`, `--exclude-ctx-from-target` | override the scored task (defaults read from checkpoint) |

The report includes overall MAE, the per-region **interp vs extrap** breakdown
(displacement robustness), a per-geometry CSV, and a degradation-vs-displacement
curve. `compare_baselines.py` and `compare_offline_online.py` aggregate multiple
runs / the streaming model.

**Diagnostic axes that matter beyond headline MAE** (see `Project_Overview.md`):
acoustic reliance (corrupt target acoustics), temporal reliance (`--shuffle-temporal`),
fix-sparsity (`--eval-ctx-sweep`), and drift-vs-horizon — these separate genuine
acoustic localization from motion extrapolation.
