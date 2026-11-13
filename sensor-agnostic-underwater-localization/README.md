# Sensor-agnostic underwater acoustic localization

Code for *A Competitive Mean Absolute Error Does Not Demonstrate Localization:
Control Protocol and Benchmarks for Sensor-Agnostic Underwater Acoustic
Localization With Prior-Data Fitted Networks* (Applied Ocean Research).

It contains the acoustic simulator used to generate the datasets, the neural
process models with the spatial front end (S-AttnCNP, S-RecCNP, PFN, ...), the
trainer, the evaluation suite with the input-side controls, and the scripts that
build the paper's tables and figures.

```
config/              Hydra configuration (models, splits, datasets, training)
  data_pipeline.yaml   simulator and preprocessing settings
data/                simulator (generate.py) and preprocessing (process_data.py)
splits/              the train / validation / test splits used in the paper
src/models/          models: anp.py (CNP, ANP, spatial front end), r_anp.py
                     (recurrent models), pfn.py, online_r_anp.py
scripts/train/       train_np_geometry.py
scripts/eval/        eval_suite.py and the reference baselines
paper/scripts/       tables and figures of the paper
```

Install the dependencies with `pip install -r requirements.txt` and run every
command from the repository root.

## Datasets

Datasets are not distributed; they are regenerated from the configuration and
the master seed (`channel.master_seed: 11`). Each deployment's layout,
trajectories and environment depend only on the seed and the deployment index.

| Dataset | Deployments | Trajectories each | Generation |
|---|---|---|---|
| Development set, shared | 200 | 400 (shared) | `dataset=random_hermite_shared` |
| Development set, distinct | 200 | 400 (own) | `dataset=random_hermite_distinct` |
| B1, unseen deployments | 10 000 | 3 | see below |
| B2, unseen environments | 4 000 | 6 | see below |

```bash
# Development set (both variants), split with splits/dev.json
python data/generate.py dataset=random_hermite_shared
python data/generate.py dataset=random_hermite_distinct
python data/process_data.py --data-root data/random_task/hermite_shared --mode geometry --splits-file splits/dev.json
python data/process_data.py --data-root data/random_task/hermite_distinct --mode geometry --splits-file splits/dev.json

# B1
python data/generate.py dataset=random_hermite_distinct random_task.n_position_sets=10000 \
    channel.n_traj=3 random_task.out_dir=./data/pfn10kd
python data/process_data.py --data-root data/pfn10kd/hermite_distinct --mode geometry --splits-file splits/b1.json

# B2: one pinned array, a sampled environment per deployment
python data/generate.py dataset=random_hermite_distinct random_task.n_position_sets=4000 \
    channel.n_traj=6 "channel.channel_options=[0.1]" random_task.out_dir=./data/pathD \
    random_task.layout.fixed_seed=777 random_task.environment.enabled=true
python data/process_data.py --data-root data/pathD/hermite_distinct --mode geometry --splits-file splits/b2.json
```

Generation is parallelized over all cores (`channel.nop`), and a run can be
split into shards with `random_task.start_set` / `random_task.end_set`.

## Training

`scripts/train/train_np_geometry.py` composes `dataset`, `data` (split), `model`
and `wandb`. The models of the paper:

| Paper | `model=` | Notes |
|---|---|---|
| S-AttnCNP | `spatial_cnp` | on B1/B2 with `model.attn_ffn=true` |
| S-AttnLNP | `spatial_anp` | |
| S-RecCNP | `spatial_rcnp` | |
| S-RecLNP | `spatial_ranp` | |
| PFN | `spatial_pfn` | |
| AttnCNP, AttnLNP, RecCNP, RecLNP | `cnp`, `anp`, `rcnp`, `ranp` | with `data.trans_aug_m=0` |

```bash
# Development set (shared variant)
python scripts/train/train_np_geometry.py model=spatial_cnp dataset=random_hermite_shared \
    data.context.per_sample=false seed=0

# B1 (use data/pathD/... for B2); prefix regime
python scripts/train/train_np_geometry.py model=spatial_pfn \
    data.data_dir=data/pfn10kd/hermite_distinct/processed/geometry_split \
    data.max_train_geometries=null data.max_train_trajectories=null data.max_val_trajectories=null \
    data.context.per_sample=false seed=0

# Calibration regime: replace data.context.per_sample=false by data.context.calibration=true
```

The paper uses seeds 0, 1 and 2. Appendix C trains on depth bands of B2 with
`data.train_geometry_ids=splits/b2_train_shallow.json` (or `b2_train_deep.json`).
Each run writes `best.pt`, `last.pt`, `train_log.csv` and `config.yaml` to
`output/hydra/training/<date>/<exp_name>/seed_<seed>/`.

Logging to Weights & Biases is off by default; enable it with
`wandb.enabled=true` after `wandb login`.

## Evaluation

`scripts/eval/eval_suite.py` evaluates one or more checkpoints; checkpoints that
share a label are treated as training seeds.

```bash
python scripts/eval/eval_suite.py --tiers 1,2,3,4 --out-dir output/eval_paper/b1_prefix \
    --models pfn=<run_seed0>/best.pt pfn=<run_seed1>/best.pt pfn=<run_seed2>/best.pt \
             rcnp=<run_seed0>/best.pt ...
```

- tier 1: MAE overall and by region, and against displacement from the training arrays
- tier 2: input-side controls (noise, zero, order)
- tier 3: number and placement of the context fixes, error along the transit
- tier 4: sensor dropout and sensor permutation

`scripts/eval/latent_gap.py` (Appendix B), `trivial_baselines.py` and
`trajectory_baseline.py` are scored on the same transits.

## Tables and figures

With the results in `output/eval_paper/<group>/`:

```bash
python paper/scripts/make_tables.py
python paper/scripts/make_figures.py
python paper/scripts/make_source_figures.py   # needs the raw B1 data
```

The evaluation groups read by the scripts are `s4_family` and `ab_distinct`
(development set), `s4_ablation`, `latent_gap_v2`, `b1_prefix`, `b1_calib`,
`b1_ranp`, `b2_prefix`, `b2_calib`, `ds_shallow` and `ds_deep`.
