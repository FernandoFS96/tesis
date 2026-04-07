#!/usr/bin/env python3
"""
Optuna wrapper for ANP and RANP masked training scripts.

- Selects model with --model anp|ranp.
- Minimizes weighted validation score.
- Uses SQLite storage for resume.
- Supports pruning (requires trial.report + trial.should_prune hooks in the training function).

Run ANP (single process):
    cd underwater-localization-topologies/src/training
    python optuna_anp_search.py \
    --model anp \
    --data-dir /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_low_variance \
    --topologies aligned \
    --objective-topology aligned \
    --mask-in-val \
    --n-trials 50 \
    --storage sqlite:////home/fernando/tesis/underwater-localization-topologies/results/optuna_anp.db \
    --study-name anp_masked_lowvar_aligned_v2 \
    --constant-liar \
    --cleanup-trial-checkpoints \
    --disable-pruning

    - High Variance data:
    python optuna_anp_search.py \
    --model anp \
    --data-dir /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_high_variance \
    --topologies aligned \
    --objective-topology aligned \
    --mask-in-val \
    --n-trials 50 \
    --storage sqlite:////home/fernando/tesis/underwater-localization-topologies/results/optuna_anp.db \
    --study-name anp_masked_highvar_aligned_v2 \
    --constant-liar \
    --cleanup-trial-checkpoints \
    --disable-pruning

    With nohup and redirect to log:
    nohup python optuna_anp_search.py \
        --model anp \
        --data-dir /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_high_variance \
        --topologies aligned \
        --objective-topology aligned \
        --mask-in-val \
        --n-trials 100 \
        --storage sqlite:////home/fernando/tesis/underwater-localization-topologies/results/optuna_anp.db \
        --study-name anp_masked_highvar_aligned_v2 \
        --constant-liar \
        --cleanup-trial-checkpoints \
        --disable-pruning \
        > nohup/optuna_anp_masked_highvar_aligned_v2_$(date +%F_%H%M%S)_$$.log 2>&1 & 

Run RANP (single process):
    cd underwater-localization-topologies/src/training
    python optuna_anp_search.py \
    --model ranp \
    --data-dir /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_low_variance \
    --topologies aligned \
    --objective-topology aligned \
    --mask-in-val \
    --n-trials 50 \
    --storage sqlite:////home/fernando/tesis/underwater-localization-topologies/results/optuna_ranp.db \
    --study-name ranp_masked_lowvar_aligned_v2 \
    --constant-liar \
    --cleanup-trial-checkpoints \
    --disable-pruning

    - High Variance data:
    python optuna_anp_search.py \
    --model ranp \
    --data-dir /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_high_variance \
    --topologies aligned \
    --objective-topology aligned \
    --mask-in-val \
    --n-trials 20 \
    --storage sqlite:////home/fernando/tesis/underwater-localization-topologies/results/optuna_ranp.db \
    --study-name ranp_masked_highvar_aligned_v2 \
    --constant-liar \
    --cleanup-trial-checkpoints \
    --disable-pruning

    With nohup and redirect to log:
    nohup python optuna_anp_search.py \
        --model ranp \
        --data-dir /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_low_variance \
        --topologies ellipsoidal \
        --objective-topology ellipsoidal \
        --mask-in-val \
        --n-trials 100 \
        --storage sqlite:////home/fernando/tesis/underwater-localization-topologies/results/optuna_ranp.db \
        --study-name ranp_masked_lowvar_ellipsoidal_v2 \
        --constant-liar \
        --cleanup-trial-checkpoints \
        --disable-pruning \
        > nohup/optuna_ranp_masked_lowvar_ellipsoidal_v2_$(date +%F_%H%M%S)_$$.log 2>&1 &

    monitor with:
    tail -f nohup/optuna_ranp_masked_highvar_random_v2_$(date +%F_%H%M%S)_$$.log 2>&1 &

Parallel: start multiple processes pointing to the same storage+study:
  # terminal 1
  python optuna_anp_search.py ... --n-trials 100
  # terminal 2
  python optuna_anp_search.py ... --n-trials 100
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path
import time
from typing import Callable

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# IMPORTANT: run this script from the project root (so src/ is importable), or PYTHONPATH must include the project root.
import importlib.util as _importlib_util


def _make_best_model_callback(results_root: Path) -> Callable:
    """Returns an Optuna callback that copies the best trial checkpoints to best_model/."""

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        if study.best_trial.number != trial.number:
            return

        trial_dir_str = trial.user_attrs.get("trial_dir")
        if trial_dir_str is None:
            return
        trial_dir = Path(trial_dir_str)
        best_dir = results_root / "best_model"

        if best_dir.exists():
            shutil.rmtree(best_dir)
        best_dir.mkdir(parents=True, exist_ok=True)

        # Copy hparams.json
        hparams_src = trial_dir / "hparams.json"
        if hparams_src.exists():
            shutil.copy2(str(hparams_src), str(best_dir / "hparams.json"))

        # Copy all best checkpoint variants from each topology subdirectory.
        for topo_dir in sorted(trial_dir.iterdir()):
            if not topo_dir.is_dir():
                continue
            dest_dir = best_dir / topo_dir.name
            copied_any = False
            for ckpt_src in topo_dir.glob("best_checkpoint*.pth.tar"):
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(ckpt_src), str(dest_dir / ckpt_src.name))
                copied_any = True
            if not copied_any:
                legacy_ckpt = topo_dir / "best_checkpoint.pth.tar"
                if legacy_ckpt.exists():
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(legacy_ckpt), str(dest_dir / "best_checkpoint.pth.tar"))

        # Save trial metadata alongside the checkpoint
        info = {
            "trial_number": trial.number,
            "value": trial.value,
            "params": trial.params,
            "user_attrs": {k: v for k, v in trial.user_attrs.items() if k != "trial_dir"},
        }
        with open(str(best_dir / "trial_info.json"), "w") as f:
            json.dump(info, f, indent=2)

        print(
            f"\n[best_model] Trial {trial.number} is new best "
            f"(MAE={trial.value:.6f}). Checkpoints saved to {best_dir}\n"
        )

    return callback


def _infer_study_version(study_name: str) -> str:
    """Infer version tag (vN) from study name; fallback to 'vunknown'."""
    m = re.search(r"(v\d+)$", study_name.strip().lower())
    return m.group(1) if m else "vunknown"


def _resolve_results_root(results_dir: str, model: str, study_name: str) -> Path:
    """Build results path as <results_dir>/<model>/<version>/<study_name>."""
    version = _infer_study_version(study_name)
    return Path(results_dir) / model / version / study_name


def make_objective(args, results_root: Path):
    topologies = [t.strip() for t in args.topologies.split(",") if t.strip()]
    if args.aggregate_topologies:
        obj_topos = topologies
    else:
        obj_topos = [args.objective_topology]

    results_root.mkdir(parents=True, exist_ok=True)

    # Load the appropriate training module once per study (importlib needed for hyphenated filename)
    if args.model == "anp":
        import src.training.train_anp_topologies_masked as _train_mod
        _train_fn = _train_mod.train_anp_topology_masked
    else:
        _spec = _importlib_util.spec_from_file_location(
            "train_ranp",
            Path(__file__).parent / "train_r-anp_topologies_masked.py",
        )
        _train_mod = _importlib_util.module_from_spec(_spec)
        _spec.loader.exec_module(_train_mod)
        _train_fn = _train_mod.train_ranp_topology_masked
    train_mod = _train_mod

    def objective(trial: optuna.trial.Trial) -> float:
        t0 = time.perf_counter()
        minutes_per_topology = {}
        weighted_scores = {}
        inverse_holdout_maes = {}
        fixed_holdout_maes = {}
        legacy_maes = {}
        # ---------------------------
        # 1) Sample hyperparameters
        # ---------------------------
        hp = {
            # model / optimizer
            "num_hidden": trial.suggest_int("num_hidden", 128, 320, step=64), 
            "weight_decay": trial.suggest_categorical("weight_decay", [1e-4, 5e-5, 1e-5, 5e-6]),
            "lr": None,  # set later based on model type

            # training dynamics
            "kl_warmup_epochs": trial.suggest_int("kl_warmup_epochs", 500, 2000, step=500),

            # context sampling
            "ctx_sample_mode": trial.suggest_categorical("ctx_sample_mode", ["first"]),#trial.suggest_categorical("ctx_sample_mode", ["first", "random"]),

            # masking
            "sensor_drop_mode": trial.suggest_categorical("sensor_drop_mode", ["bernoulli", "k_uniform"]),
            "sensor_drop_p": trial.suggest_categorical("sensor_drop_p", [0.3]),
            "mask_fill": trial.suggest_categorical("mask_fill", ["train_mean"]),#("mask_fill", ["train_mean", "zero"]),
        }

        # batch-size often interacts with lr; keep a small menu
        hp["batch_size"] = trial.suggest_categorical("batch_size", [8, 16])

        # RANP-specific: RNN encoder hyperparameters
        if args.model == "ranp":
            hp["lr"]             = trial.suggest_categorical("lr", [9e-4, 7e-4, 5e-4])
            hp["rnn_type"]       = trial.suggest_categorical("rnn_type", ["lstm", "gru"])
            hp["rnn_hidden_dim"] = trial.suggest_categorical("rnn_hidden_dim", [32, 64, 128])
            hp["rnn_layers"]     = trial.suggest_int("rnn_layers", 1, 2, step=1)
            hp["rnn_dropout"]    = trial.suggest_categorical("rnn_dropout", [0.1, 0.2])
        else:
            hp["lr"] = trial.suggest_categorical("lr", [5e-4, 3e-4, 1e-4])

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
        for topo in obj_topos:
            train_data, val_data, _ = train_mod.load_topology_data(args.data_dir, topo)
            if train_data is None:
                raise RuntimeError(f"Missing data for topology={topo} in {args.data_dir}")

            save_dir = trial_dir / f"topology_{topo}"
            save_dir.mkdir(parents=True, exist_ok=True)

            topo_t0 = time.perf_counter()

            # training function already has Optuna pruning hooks (trial.report + trial.should_prune), so it will report intermediate MAE values and prune unpromising trials.
            train_kwargs = dict(
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
                num_hidden=hp["num_hidden"],
                lr=hp["lr"],
                weight_decay=hp["weight_decay"],
                trial=trial,
                report_every=args.report_every,
                save_checkpoints=True,
                holdout_frac=args.holdout_frac,
                es_context_frac=args.es_context_frac,
                es_weight_fixed=args.es_weight_fixed,
                es_weight_inverse=args.es_weight_inverse,
                include_fixed_holdout_in_es=args.include_fixed_holdout_in_es,
            )
            if args.model == "ranp":
                train_kwargs.update(
                    rnn_type=hp["rnn_type"],
                    rnn_hidden_dim=hp["rnn_hidden_dim"],
                    rnn_layers=hp["rnn_layers"],
                    rnn_dropout=hp["rnn_dropout"],
                )
            best_metrics = _train_fn(**train_kwargs)
            minutes_per_topology[topo] = (time.perf_counter() - topo_t0) / 60.0
            weighted_scores[topo] = float(best_metrics["weighted_score"])
            inverse_holdout_maes[topo] = float(best_metrics["inverse_holdout_mae"])
            fixed_holdout_maes[topo] = float(best_metrics["fixed_holdout_mae"])
            legacy_maes[topo] = float(best_metrics["legacy_mae_ctx040"])

        value = sum(weighted_scores.values()) / len(weighted_scores)

        trial.set_user_attr("weighted_score_per_topology", weighted_scores)
        trial.set_user_attr("inverse_holdout_mae_per_topology", inverse_holdout_maes)
        trial.set_user_attr("fixed_holdout_mae_per_topology", fixed_holdout_maes)
        trial.set_user_attr("legacy_mae_ctx040_per_topology", legacy_maes)
        trial.set_user_attr("minutes_per_topology", minutes_per_topology)
        trial.set_user_attr("objective_topologies", obj_topos)
        trial.set_user_attr("trial_dir", str(trial_dir))
        trial.set_user_attr("objective_value_name", "weighted_validation_score")
        return value

    return objective


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--model", type=str, default="anp", choices=["anp", "ranp"], help="Model to tune: 'anp' = train_anp_topologies_masked, 'ranp' = train_r-anp_topologies_masked.")
    p.add_argument("--topologies", type=str, default="aligned,ellipsoidal,random")
    p.add_argument("--objective-topology", type=str, default="aligned", help="Topology used for objective when --aggregate-topologies is false.")
    p.add_argument("--aggregate-topologies", action="store_true", help="If set, objective = mean MAE over ALL topologies (slower but more robust).")

    # training settings (keep these fixed during HPO; tune them only if you really need)
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--patience", type=int, default=250)
    p.add_argument("--device", type=str, default="cuda")

    # data/model sizes needed by masking layout
    p.add_argument("--num-sensors", type=int, default=10)
    p.add_argument("--num-time-points", type=int, default=201)
    p.add_argument("--mask-in-val", action="store_true", help="If set, apply masking also during validation (harder).")
    p.add_argument("--holdout-frac", type=float, default=0.2, help="Validation holdout tail fraction for fixed/inverse metrics.")
    p.add_argument("--es-context-frac", type=float, default=0.4, help="Context fraction of pre-holdout window for holdout validation metrics.")
    p.add_argument("--es-weight-fixed", type=float, default=0.2, help="Weight for fixed-holdout MAE in weighted validation score (used only with --include-fixed-holdout-in-es).")
    p.add_argument("--es-weight-inverse", type=float, default=0.8, help="Weight for inverse-holdout MAE in weighted validation score (used only with --include-fixed-holdout-in-es).")
    p.add_argument("--include-fixed-holdout-in-es", action="store_true", help="Include fixed-holdout metric in training early stopping/objective. Default optimizes inverse holdout only.")

    # optuna
    p.add_argument("--n-trials", type=int, default=200)
    p.add_argument("--timeout", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--storage", type=str, default="sqlite:///optuna_anp.db")
    p.add_argument("--study-name", type=str, default="anp_masked_hpo")
    p.add_argument("--results-dir", type=str, default="results/optuna")
    p.add_argument("--report-every", type=int, default=50, help="Epoch interval for trial.report() in training (pruning granularity).")
    p.add_argument("--disable-pruning", action="store_true", help="Disable Optuna pruning so all trials run to completion.")
    p.add_argument("--constant-liar", action="store_true", help="Enable constant_liar in TPESampler to reduce duplicate hyperparameter suggestions when running multiple parallel workers on the same study.")

    # For GPU training, keep n_jobs=1 and parallelize with multiple processes (same --storage).
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--cleanup-trial-checkpoints", action="store_true", help="After each trial, delete its checkpoint files to save space (only the best model is kept in best_model/).")

    args = p.parse_args()

    sampler = TPESampler(seed=args.seed, constant_liar=args.constant_liar)
    if args.disable_pruning:
        pruner = optuna.pruners.NopPruner()
    else:
        pruner = MedianPruner(
            n_startup_trials=10,   # wait for more completed trials before pruning starts
            n_warmup_steps=1000,   # allow at least 1000 updates before pruning
            interval_steps=250,    # check every 250 updates
            n_min_trials=5,        # require at least 5 trials reporting at this step
        )

    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        load_if_exists=True,
    )

    results_root = _resolve_results_root(args.results_dir, args.model, args.study_name)
    results_root.mkdir(parents=True, exist_ok=True)

    objective = make_objective(args, results_root)

    callbacks = [_make_best_model_callback(results_root)]

    if args.cleanup_trial_checkpoints:
        def _cleanup_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            """Delete checkpoint files from trial dirs that are NOT the current best."""
            if trial.state != optuna.trial.TrialState.COMPLETE:
                return
            if study.best_trial.number == trial.number:
                return  # keep the best (already copied by the other callback)
            trial_dir_str = trial.user_attrs.get("trial_dir")
            if trial_dir_str is None:
                return
            for ckpt in Path(trial_dir_str).rglob("*.pth.tar"):
                try:
                    ckpt.unlink()
                except OSError:
                    pass
        callbacks.append(_cleanup_callback)

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout,
                   n_jobs=args.n_jobs, callbacks=callbacks)

    print("\nBest trial:")
    bt = study.best_trial
    print("  value (mean weighted validation score):", bt.value)
    print("  params:", bt.params)
    print("  user attrs:", bt.user_attrs)

    best_dir = results_root / "best_model"
    if best_dir.exists():
        print(f"\nBest model checkpoints saved to: {best_dir}")


if __name__ == "__main__":
    main()
