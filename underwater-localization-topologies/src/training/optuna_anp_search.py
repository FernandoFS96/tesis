#!/usr/bin/env python3
"""
Optuna wrapper for train_anp_topologies_masked.py

- Minimizes validation MAE (default).
- Uses SQLite storage so for resume.
- Supports pruning (requires a small patch in the training function: trial.report + trial.should_prune).

Run (single process):
  python optuna_anp_search.py \
    --data-dir /path/to/data_processed_topologies \
    --topologies aligned,ellipsoidal,random \
    --objective-topology aligned \
    --n-trials 50 \
    --storage sqlite:///optuna_anp.db \
    --study-name anp_masked_v1

Resume:
  python optuna_anp_search.py ... same --storage and --study-name

Parallel: start multiple processes pointing to the same storage+study:
  # terminal 1
  python optuna_anp_search.py ... --n-trials 100
  # terminal 2
  python optuna_anp_search.py ... --n-trials 100
"""

import argparse
import json
import os
from pathlib import Path

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# IMPORTANT: run this script from the project root (so src/ is importable), or PYTHONPATH must include the project root.
import src.training.train_anp_topologies_masked as train_mod


def make_objective(args):
    topologies = [t.strip() for t in args.topologies.split(",") if t.strip()]
    if args.aggregate_topologies:
        obj_topos = topologies
    else:
        obj_topos = [args.objective_topology]

    results_root = Path(args.results_dir) / args.study_name
    results_root.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.trial.Trial) -> float:
        # ---------------------------
        # 1) Sample hyperparameters
        # ---------------------------
        hp = {
            # model / optimizer
            "num_hidden": trial.suggest_int("num_hidden", 64, 256, step=32),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),

            # ANP training dynamics
            "kl_warmup_epochs": trial.suggest_int("kl_warmup_epochs", 100, 800, step=100),

            # context sampling
            "ctx_sample_mode": trial.suggest_categorical("ctx_sample_mode", ["first", "random"]),

            # masking
            "sensor_drop_mode": trial.suggest_categorical("sensor_drop_mode", ["bernoulli", "k_uniform"]),
            "sensor_drop_p": trial.suggest_float("sensor_drop_p", 0.0, 0.5),
            "mask_fill": trial.suggest_categorical("mask_fill", ["train_mean", "zero"]),
        }

        # batch-size often interacts with lr; keep a small menu
        hp["batch_size"] = trial.suggest_categorical("batch_size", [4, 8, 12, 16])

        # ---------------------------
        # 2) Per-trial output folder
        # ---------------------------
        trial_dir = results_root / f"trial_{trial.number:05d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        with open(trial_dir / "hparams.json", "w") as f:
            json.dump(hp, f, indent=2)

        # ---------------------------
        # 3) Train/eval on chosen topology/topologies
        # ---------------------------
        maes = {}
        for topo in obj_topos:
            train_data, val_data, _ = train_mod.load_topology_data(args.data_dir, topo)
            if train_data is None:
                raise RuntimeError(f"Missing data for topology={topo} in {args.data_dir}")

            save_dir = trial_dir / f"topology_{topo}"
            save_dir.mkdir(parents=True, exist_ok=True)

            # NOTE: requires the small patch below so train_anp_topology_masked accepts:
            #   num_hidden, lr, weight_decay, trial (optional)
            best_mae = train_mod.train_anp_topology_masked(
                train_data=train_data,
                val_data=val_data,
                save_dir=str(save_dir),
                topology_name=topo,
                batch_size=hp["batch_size"],
                epochs=args.epochs,
                patience=args.patience,
                device=args.device,
                ctx_sample_mode=hp["ctx_sample_mode"],
                num_sensors=args.num_sensors,
                num_time_points=args.num_time_points,
                sensor_drop_mode=hp["sensor_drop_mode"],
                sensor_drop_p=hp["sensor_drop_p"],
                mask_fill=hp["mask_fill"],
                mask_in_val=args.mask_in_val,
                kl_warmup_epochs=hp["kl_warmup_epochs"],
                # new (patch)
                num_hidden=hp["num_hidden"],
                lr=hp["lr"],
                weight_decay=hp["weight_decay"],
                trial=trial,
                report_every=args.report_every,
            )
            maes[topo] = float(best_mae)

        # aggregate objective
        value = sum(maes.values()) / len(maes)
        trial.set_user_attr("mae_per_topology", maes)
        trial.set_user_attr("objective_topologies", obj_topos)
        trial.set_user_attr("trial_dir", str(trial_dir))
        return value

    return objective


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--topologies", type=str, default="aligned,ellipsoidal,random")
    p.add_argument("--objective-topology", type=str, default="aligned", help="Topology used for objective when --aggregate-topologies is false.")
    p.add_argument("--aggregate-topologies", action="store_true", help="If set, objective = mean MAE over ALL topologies (slower but more robust).")

    # training settings (keep these fixed during HPO; tune them only if you really need)
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--patience", type=int, default=200)
    p.add_argument("--device", type=str, default="cuda")

    # data/model sizes needed by masking layout
    p.add_argument("--num-sensors", type=int, default=10)
    p.add_argument("--num-time-points", type=int, default=201)
    p.add_argument("--mask-in-val", action="store_true", help="If set, apply masking also during validation (harder).")

    # optuna
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--timeout", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--storage", type=str, default="sqlite:///optuna_anp.db")
    p.add_argument("--study-name", type=str, default="anp_masked_hpo")
    p.add_argument("--results-dir", type=str, default="results/optuna")
    p.add_argument("--report-every", type=int, default=25, help="Epoch interval for trial.report() in training (pruning granularity).")

    # For GPU training, keep n_jobs=1 and parallelize with multiple processes (same --storage).
    p.add_argument("--n-jobs", type=int, default=1)

    args = p.parse_args()

    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=max(1, args.report_every), interval_steps=args.report_every)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        load_if_exists=True,
    )

    objective = make_objective(args)
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, n_jobs=args.n_jobs)

    print("\nBest trial:")
    bt = study.best_trial
    print("  value (mean val MAE):", bt.value)
    print("  params:", bt.params)
    print("  user attrs:", bt.user_attrs)


if __name__ == "__main__":
    main()
