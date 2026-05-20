"""
optuna_anp.py
==============================================================================
Hyperparameter optimisation for all ANP model variants using Optuna.

Controllable via flags — one Optuna run per model variant:

    # ANP dual-target (default — SoC% + Cycle)
    python optuna_anp.py --target_col all

    # ANP SoC-only (full features)
    python optuna_anp.py --target_col "SoC (%)"

    # ANP Cycle-only (full features)
    python optuna_anp.py --target_col Cycle

    # ANP SoC-only (RF-reduced features: 11 features)
    python optuna_anp.py --target_col "SoC (%)" --reduced_features

    # ANP Cycle-only (RF-reduced features: 12 features)
    python optuna_anp.py --target_col Cycle --reduced_features

    # ANP Cycle-only (RF-reduced + cycle-level aggregation)
    python optuna_anp.py --target_col Cycle --reduced_features --aggregate_by_cycle

Output structure:
    --out_dir/<model_variant>/        # e.g. optuna_results/anp_soc_reduced/
        study.db                      # SQLite — resumable across runs
        best_params.json              # Best hyperparameters
        all_trials.csv                # All trial results
        optimization.png              # Optimization history plot
        importance.png                # Hyperparameter importance (fANOVA)
        parallel_coordinate.png       # Parallel coordinate plot
        trial_000/
            best.pt                   # Best model weights
            config.json               # Full config (loadable by evaluate*.py)
            metrics.csv               # Per-epoch metrics
            plots/                    # Training curve PNGs
        trial_001/
        ...

The config.json saved per trial contains all fields that evaluate.py, evaluate_anp.py, and test_physical_coherence.py need to auto-detect 
the model configuration (target_col, num_hidden, attn_dropout, use_reduced_features, aggregate_by_cycle, ctx_cycles, tgt_cycles, measurements_per_cycle).

Fixed parameters (not optimised):
    ctx_cycles     = 60  (1 800 measurement-level rows, or 60 cycle-level rows)
    tgt_cycles     = 60
    meas_per_cycle = 30
    epochs         = 1000  (per trial)
    early_stopping = 100   (patience in epochs)
    episodes       = 100   (training steps per epoch)

Search space (categorical):
    num_hidden   : [64, 128, 192, 256]
    batch_size   : [2, 4, 6, 8]
    lr           : [1e-4, 5e-4, 1e-3, 5e-3]
    beta         : [0.01, 0.1, 0.5, 1.0, 2.0]
    attn_dropout : [0.0, 0.1, 0.2]

No pruning — every trial runs to completion or early stopping.
==============================================================================
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import optuna
from optuna.samplers import TPESampler
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_CSIC_ROOT  = _SCRIPT_DIR.parent
for _p in [str(_CSIC_ROOT), str(_SCRIPT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from train_utils import (
    load_prepared_data,
    validate_targets,
    get_task_splits,
    sort_task_by_cycle,
    make_batch,
    evaluate,
    generate_all_plots,
    REDUCED_FEATURE_SETS,
    get_feature_indices,
    aggregate_by_cycle,
)
from models.anp import LatentModel


# ==============================================================================
# FIXED PARAMETERS
# ==============================================================================

CTX_CYCLES     = 60
TGT_CYCLES     = 60
MEAS_PER_CYCLE = 30
EPOCHS         = 1000
EARLY_STOP     = 100
EPISODES       = 100
LR_MIN         = 1e-5
SEED           = 18

TRAIN_IDS = list(range(17))
VAL_IDS   = list(range(17, 22))
TEST_IDS  = list(range(22, 25))

LR_CHOICES:      List[float] = [1e-4, 5e-4, 1e-3, 5e-3]
BETA_CHOICES:    List[float] = [0.01, 0.1, 0.5, 1.0, 2.0]
DROPOUT_CHOICES: List[float] = [0.0, 0.1, 0.2]


# ==============================================================================
# HELPERS
# ==============================================================================

def model_variant_tag(target_col: str, reduced: bool, aggregate: bool) -> str:
    """
    Build a short filesystem-safe tag identifying the model variant.
    Used as the output subdirectory name and Optuna study name.

    Examples:
        "all",  reduced=False, aggregate=False → "anp_dual"
        "SoC (%)", reduced=False              → "anp_soc"
        "SoC (%)", reduced=True               → "anp_soc_reduced"
        "Cycle",   reduced=True, aggregate=True → "anp_cycle_agg_reduced"
    """
    if target_col == "all":
        tag = "anp_dual"
    elif "SoC" in target_col:
        tag = "anp_soc_reduced" if reduced else "anp_soc"
    else:  # Cycle
        if aggregate and reduced:
            tag = "anp_cycle_agg_reduced"
        elif reduced:
            tag = "anp_cycle_reduced"
        else:
            tag = "anp_cycle"
    return tag


def early_stop_key(target_col: str) -> str:
    """Return the val metric key used for early stopping and best-model tracking."""
    return "val/mae_Cycle" if target_col == "Cycle" else "val/mae_SoC_pct"


def apply_target_filter(data: dict, target_col: str) -> dict:
    """
    Filter y columns in all splits to the selected target.
    Returns a shallow copy — original data is not mutated.
    If target_col is 'all', returns data unchanged.
    """
    if target_col == "all":
        return data

    filtered_synth = [
        (X, y[[target_col]]) for X, y in data["normalized_synth_datasets"]
    ]
    rd_X, rd_y = data["normalized_real_dataset"]
    filtered_real = (rd_X, rd_y[[target_col]])

    for k in ["y_mean", "y_std"]:
        if k in data["denorm_values"]:
            data["denorm_values"][k] = {
                target_col: data["denorm_values"][k][target_col]
            }

    return {
        **data,
        "normalized_synth_datasets": filtered_synth,
        "normalized_real_dataset":   filtered_real,
    }


def apply_feature_reduction(data: dict, target_col: str,
                             x_col_names: List[str]) -> dict:
    """
    Filter X to the RF-selected compact feature set for target_col.
    Returns a shallow copy. Raises ValueError if features not found.
    """
    feature_cols = REDUCED_FEATURE_SETS.get(target_col)
    if feature_cols is None:
        return data  # 'all' target or unknown — no reduction

    missing = [f for f in feature_cols if f not in x_col_names]
    if missing:
        raise ValueError(
            f"Feature reduction failed — columns not found: {missing}\n"
            f"Available: {x_col_names[:20]}"
        )

    print(f"     feature reduction: {len(x_col_names)} → {len(feature_cols)} features")

    filtered_synth = [
        (X[feature_cols], y) for X, y in data["normalized_synth_datasets"]
    ]
    rd_X, rd_y = data["normalized_real_dataset"]
    filtered_real = (rd_X[feature_cols], rd_y)

    return {
        **data,
        "normalized_synth_datasets": filtered_synth,
        "normalized_real_dataset":   filtered_real,
    }


def apply_cycle_aggregation(data: dict) -> dict:
    """
    Aggregate EIS measurements to one representative row per cycle.
    Returns a shallow copy.
    """
    print(f"     aggregating by cycle ...")
    filtered_synth = [
        aggregate_by_cycle(X, y) for X, y in data["normalized_synth_datasets"]
    ]
    rd_X, rd_y = data["normalized_real_dataset"]
    filtered_real = aggregate_by_cycle(rd_X, rd_y)
    n_rows = filtered_synth[0][0].shape[0]
    print(f"     → {n_rows} rows/task (was ~{data['normalized_synth_datasets'][0][0].shape[0]})")
    return {
        **data,
        "normalized_synth_datasets": filtered_synth,
        "normalized_real_dataset":   filtered_real,
    }


# ==============================================================================
# OBJECTIVE
# ==============================================================================

def objective(
    trial:              optuna.Trial,
    data:               dict,
    device:             torch.device,
    out_dir:            Path,
    target_col:         str,
    use_reduced:        bool,
    use_aggregate:      bool,
    x_col_names_orig:   List[str],
    ctx_rows:           int,
    tgt_rows:           int,
) -> float:
    """
    Optuna objective — trains one ANP configuration and returns best val metric.

    Applies target filtering, feature reduction, and cycle aggregation according to the flags passed in. 
    All these transformations are applied on a copy of the data dict inside each trial.

    The config.json saved per trial includes all fields needed by the evaluation scripts (target_col, num_hidden, attn_dropout, use_reduced_features, aggregate_by_cycle, ctx_cycles, tgt_cycles, measurements_per_cycle).

    Args:
        trial:            Optuna trial — handles hyperparameter sampling.
        data:             Pre-loaded, pre-filtered data dict (already has target and feature filtering applied at study level).
        device:           Torch device.
        out_dir:          Root output directory for this study variant.
        target_col:       Target column(s): 'all', 'SoC (%)', or 'Cycle'.
        use_reduced:      Whether X was filtered to reduced feature set.
        use_aggregate:    Whether data is cycle-aggregated (1 row per cycle).
        x_col_names_orig: Original X column names (before feature reduction).
        ctx_rows:         Context window in rows (accounts for aggregation).
        tgt_rows:         Target window in rows.

    Returns:
        Best val metric achieved (lower is better).
    """
    # ── Sample hyperparameters ────────────────────────────────────────────────
    num_hidden   = trial.suggest_categorical("num_hidden",   [64, 128, 192, 256])
    batch_size   = trial.suggest_categorical("batch_size",   [2, 4, 6, 8])
    lr           = trial.suggest_categorical("lr",           LR_CHOICES)
    beta         = trial.suggest_categorical("beta",         BETA_CHOICES)
    attn_dropout = trial.suggest_categorical("attn_dropout", DROPOUT_CHOICES)

    trial_id  = trial.number
    trial_dir = out_dir / f"trial_{trial_id:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  ── Trial {trial_id:03d} ──────────────────────────────────")
    print(f"     num_hidden={num_hidden}  batch={batch_size}  "
          f"lr={lr:.0e}  beta={beta}  dropout={attn_dropout}")

    # ── Data dims ─────────────────────────────────────────────────────────────
    target_cols = list(data["normalized_synth_datasets"][0][1].columns)
    input_dim   = data["normalized_synth_datasets"][0][0].shape[1]
    output_dim  = len(target_cols)
    denorm_values = {
        "y_mean": data["denorm_values"]["y_mean"],
        "y_std":  data["denorm_values"]["y_std"],
    }

    # Feature cols for config.json (used by evaluate scripts)
    feature_cols_for_config = (
        REDUCED_FEATURE_SETS.get(target_col) if use_reduced else None
    )

    # ── Task splits ───────────────────────────────────────────────────────────
    train_tasks, val_tasks, _ = get_task_splits(data, TRAIN_IDS, VAL_IDS, TEST_IDS)

    def presort(tasks):
        return [sort_task_by_cycle(X, y) for X, y in tasks]

    train_sorted  = presort(train_tasks)
    val_sorted    = presort(val_tasks)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = LatentModel(
        num_hidden=num_hidden,
        input_dim=input_dim,
        output_dim=output_dim,
        attn_dropout=attn_dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"     input_dim={input_dim}  output_dim={output_dim}  params={n_params:,}")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR_MIN)

    # ── Early stopping metric key ─────────────────────────────────────────────
    es_key = early_stop_key(target_col)

    # ── Training loop ─────────────────────────────────────────────────────────
    steps_per_epoch  = max(1, EPISODES // batch_size)
    best_metric      = float("inf")
    best_model_state = None
    no_improve       = 0
    metrics_rows: list = []

    pbar = tqdm(range(1, EPOCHS + 1), desc=f"Trial {trial_id:03d}",
                unit="ep", dynamic_ncols=True, leave=True)

    for epoch in pbar:
        model.train()
        ep_losses, ep_nlls, ep_kls = [], [], []

        for _ in range(steps_per_epoch):
            ctx_x, ctx_y, tgt_x, tgt_y = make_batch(
                train_sorted, batch_size, ctx_rows, tgt_rows, device
            )
            optimizer.zero_grad()
            _, _, loss, kl, nll = model(ctx_x, ctx_y, tgt_x, tgt_y, beta=beta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_losses.append(loss.item())
            ep_nlls.append(nll.item())
            ep_kls.append(kl.item())

        scheduler.step()

        train_loss = float(np.mean(ep_losses))
        row = {
            "epoch":      epoch,
            "train/loss": train_loss,
            "train/nll":  float(np.mean(ep_nlls)),
            "train/kl":   float(np.mean(ep_kls)),
            "lr":         scheduler.get_last_lr()[0],
        }

        val_metrics = evaluate(
            model, val_sorted, ctx_rows, tgt_rows, device,
            denorm_values, target_cols, beta=beta, split_name="val"
        )
        row.update(val_metrics)

        current_metric = val_metrics.get(es_key, float("inf"))
        pbar.set_postfix({
            "tr_loss": f"{train_loss:.3f}",
            "val_loss": f"{val_metrics.get('val/loss', float('nan')):.3f}",
            "es_val": f"{current_metric:.3f}",
            "best": f"{min(best_metric, current_metric):.3f}",
            "es": f"{no_improve}/{EARLY_STOP}",
        })

        if current_metric < best_metric:
            best_metric      = current_metric
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve       = 0
        else:
            no_improve += 1

        metrics_rows.append(row)
        if no_improve >= EARLY_STOP:
            pbar.set_description(f"Trial {trial_id:03d} ⏹ ep={epoch}")
            break

    pbar.close()

    # ── Save trial outputs ────────────────────────────────────────────────────
    # best.pt — loadable by all evaluate scripts
    torch.save(
        {
            "trial":       trial_id,
            "epoch":       epoch,
            "model":       best_model_state,
            "val_MAE":     best_metric,
            "params":      trial.params,
            "n_params":    n_params,
            "target_cols": target_cols,
        },
        trial_dir / "best.pt",
    )

    # config.json — all fields that evaluate scripts read via load_anp_model()
    config_data = {
        # Architecture (at top level for evaluate scripts)
        "num_hidden":          num_hidden,
        "attn_dropout":        attn_dropout,
        # Training target
        "target_col":          target_col,
        # Feature selection — used by load_anp_model() to detect input_dim
        "use_reduced_features": use_reduced,
        # Cycle aggregation — used by load_anp_model() to set rows_per_cycle
        "aggregate_by_cycle":  use_aggregate,
        # Window sizes — used by load_anp_model() for ctx/tgt row counts
        "ctx_cycles":           CTX_CYCLES,
        "tgt_cycles":           TGT_CYCLES,
        "measurements_per_cycle": MEAS_PER_CYCLE,
        # Optuna params (full, including duplicates for convenience)
        "params":          trial.params,
        # Trial metadata
        "trial_id":        trial_id,
        "val_MAE":         best_metric,
        "es_metric":       es_key,
        "n_params":        n_params,
        "epochs_run":      epoch,
        "target_cols":     target_cols,
        "input_dim":       input_dim,
        "output_dim":      output_dim,
    }
    with open(trial_dir / "config.json", "w") as f:
        json.dump(config_data, f, indent=2)

    # metrics.csv
    pd.DataFrame(metrics_rows).to_csv(trial_dir / "metrics.csv", index=False)

    # training plots
    generate_all_plots(trial_dir, target_cols)

    print(f"     best {es_key} = {best_metric:.4f}  "
          f"(epoch {epoch}  →  trial_{trial_id:03d}/)")

    return best_metric


# ==============================================================================
# STUDY-LEVEL PLOTS
# ==============================================================================

def plot_study_results(study: optuna.Study, out_dir: Path) -> None:
    """Save optimization history, hyperparameter importance, and parallel coord plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from optuna.visualization.matplotlib import (
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate,
        )

        fig, ax = plt.subplots(figsize=(11, 4))
        plot_optimization_history(study, ax=ax)
        ax.set_title("Optimization history — best val metric per trial")
        fig.tight_layout()
        fig.savefig(out_dir / "optimization.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  ✓  optimization.png")

        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed) >= 4:
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_param_importances(study, ax=ax)
            ax.set_title("Hyperparameter importance (fANOVA)")
            fig.tight_layout()
            fig.savefig(out_dir / "importance.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print("  ✓  importance.png")

            fig, ax = plt.subplots(figsize=(13, 5))
            plot_parallel_coordinate(study, ax=ax)
            ax.set_title("Parallel coordinate — hyperparameter combinations")
            fig.tight_layout()
            fig.savefig(out_dir / "parallel_coordinate.png", dpi=150,
                        bbox_inches="tight")
            plt.close(fig)
            print("  ✓  parallel_coordinate.png")

    except Exception as exc:
        print(f"  ⚠  Could not generate study plots: {exc}")


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Optuna hyperparameter search for ANP model variants. "
            "Use flags to select which model to optimise. "
            "Results go to --out_dir/<variant>/ and are auto-detected by evaluate scripts."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Model variant flags ───────────────────────────────────────────────────
    p.add_argument("--target_col", type=str, default="all", choices=["all", "SoC (%)", "Cycle"],
        help=(
            "Target to predict: 'all' = dual-target (SoC%%+Cycle), "
            "'SoC (%%)' = SoC-only, 'Cycle' = Cycle-only."
        ),
    )
    p.add_argument("--reduced_features", action="store_true",
        help=(
            "Use RF-identified compact feature set. "
            "11 features for SoC (%%), 12 for Cycle. "
            "No effect when --target_col all."
        ),
    )
    p.add_argument("--aggregate_by_cycle", action="store_true",
        help=(
            "Aggregate EIS measurements to 1 representative row per cycle "
            "before training. Only meaningful for Cycle-target models. "
            "Context/target windows become 60 rows instead of 1800."
        ),
    )
    # ── Search settings ───────────────────────────────────────────────────────
    p.add_argument("--n_trials",   type=int, default=200)
    p.add_argument("--data_dir",   type=str, default="../csic_real_synth_load/prepared_data")
    p.add_argument("--out_dir",    type=str, default="./optuna_results",
                   help=(
                       "Root output directory. Results go to "
                       "<out_dir>/<variant>/ where <variant> is auto-named "
                       "from the model flags (e.g. anp_soc_reduced)."
                   ))
    p.add_argument("--study_name", type=str, default="",
                   help=(
                       "Optuna study name (default: auto-named from variant). "
                       "Used for the study.db — set explicitly to resume a "
                       "specific interrupted search."
                   ))
    args = p.parse_args()

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Derive variant tag ────────────────────────────────────────────────────
    variant   = model_variant_tag(args.target_col, args.reduced_features,
                                  args.aggregate_by_cycle)
    study_name = args.study_name or f"anp_{variant}"
    out_dir    = Path(args.out_dir) / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🔧  Device    : {device}")
    print(f"📁  Out dir   : {out_dir}")
    print(f"🏷  Variant   : {variant}")
    print(f"🎯  Target    : {args.target_col}")
    print(f"🔬  Reduced   : {args.reduced_features}")
    print(f"🔄  Aggregate : {args.aggregate_by_cycle}")
    print(f"🔍  Trials    : {args.n_trials}")
    print(f"📊  ES metric : {early_stop_key(args.target_col)}")

    total_combos = 4 * 4 * 4 * 5 * 3   # 4×4×4×5×3 = 960
    print(f"\n  Search space:")
    print(f"    num_hidden   : [64, 128, 192, 256]")
    print(f"    batch_size   : [2, 4, 6, 8]")
    print(f"    lr           : {LR_CHOICES}")
    print(f"    beta         : {BETA_CHOICES}")
    print(f"    attn_dropout : {DROPOUT_CHOICES}")
    print(f"  Total combinations : {total_combos}")
    print(f"  Coverage           : {args.n_trials/total_combos*100:.1f}%")

    # ── Load data once ────────────────────────────────────────────────────────
    print(f"\n📂  Loading data from: {args.data_dir}")
    data = load_prepared_data(args.data_dir)
    validate_targets(data)
    x_col_names_orig = list(data["normalized_synth_datasets"][0][0].columns)
    print(f"   original input_dim  : {len(x_col_names_orig)}")

    # ── Apply transformations at study level (once, shared across trials) ─────
    # Transformations are applied in order: target → features → aggregation.
    # Each returns a shallow copy, so the original data is never mutated.
    if args.target_col != "all":
        print(f"   filtering targets → [{args.target_col}]")
        data = apply_target_filter(data, args.target_col)

    if args.reduced_features and args.target_col != "all":
        data = apply_feature_reduction(data, args.target_col, x_col_names_orig)
    elif args.reduced_features:
        print("   ⚠  --reduced_features has no effect with --target_col all "
              "(no single-target RF set defined for dual-target)")

    if args.aggregate_by_cycle:
        data = apply_cycle_aggregation(data)

    # Final data dims after all transformations
    final_input_dim = data["normalized_synth_datasets"][0][0].shape[1]
    final_output_dim = len(data["normalized_synth_datasets"][0][1].columns)
    print(f"   final  input_dim   : {final_input_dim}")
    print(f"   final  output_dim  : {final_output_dim}")

    # ctx/tgt rows differ for aggregated models
    if args.aggregate_by_cycle:
        ctx_rows = CTX_CYCLES   # 1 row per cycle
        tgt_rows = TGT_CYCLES
        print(f"   ctx_rows={ctx_rows}  tgt_rows={tgt_rows}  (cycle-level)")
    else:
        ctx_rows = CTX_CYCLES * MEAS_PER_CYCLE   # 1800
        tgt_rows = TGT_CYCLES * MEAS_PER_CYCLE
        print(f"   ctx_rows={ctx_rows}  tgt_rows={tgt_rows}  (measurement-level)")

    # ── Create / resume Optuna study ──────────────────────────────────────────
    storage = f"sqlite:///{out_dir / 'study.db'}"
    study   = optuna.create_study(
        study_name     = study_name,
        storage        = storage,
        direction      = "minimize",
        sampler        = TPESampler(seed=SEED),
        pruner         = optuna.pruners.NopPruner(),
        load_if_exists = True,
    )

    n_done = sum(1 for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE)
    if n_done > 0:
        print(f"\n  ↩  Resuming study — {n_done} trials already complete")

    print(f"\n🚀  Starting optimisation...\n")

    study.optimize(
        lambda trial: objective(
            trial, data, device, out_dir,
            args.target_col, args.reduced_features, args.aggregate_by_cycle,
            x_col_names_orig, ctx_rows, tgt_rows,
        ),
        n_trials          = args.n_trials,
        n_jobs            = 1,
        show_progress_bar = False,
    )

    # ── Results ───────────────────────────────────────────────────────────────
    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"  Best trial   : #{best.number:03d}")
    print(f"  Best {early_stop_key(args.target_col)}: {best.value:.4f}")
    print(f"\n  Best hyperparameters:")
    for k, v in best.params.items():
        print(f"    {k:<20} = {v}")
    print(f"{'='*60}\n")

    best_info = {
        "trial":      best.number,
        "val_loss":   best.value,
        "params":     best.params,
        "trial_dir":  f"trial_{best.number:03d}",
        "variant":    variant,
        "target_col": args.target_col,
        "reduced":    args.reduced_features,
        "aggregate":  args.aggregate_by_cycle,
    }
    with open(out_dir / "best_params.json", "w") as f:
        json.dump(best_info, f, indent=2)
    print("  ✓  best_params.json")

    study.trials_dataframe().to_csv(out_dir / "all_trials.csv", index=False)
    print("  ✓  all_trials.csv")

    print(f"\n📈  Saving study plots...")
    plot_study_results(study, out_dir)

    # Ready-to-use train_anp.py command
    params = best.params
    reduced_flag  = " \\\n        --reduced_features"    if args.reduced_features    else ""
    aggregate_flag= " \\\n        --aggregate_by_cycle"  if args.aggregate_by_cycle  else ""
    print(f"\n✅  Optimisation complete.")
    print(f"\n  Best checkpoint: {out_dir / f'trial_{best.number:03d}' / 'best.pt'}")
    print(f"\n  To run full training with best hyperparameters:")
    print(f"\n    python train_anp.py \\")
    print(f"        --target_col   \"{args.target_col}\" \\")
    print(f"        --num_hidden   {params['num_hidden']} \\")
    print(f"        --batch_size   {params['batch_size']} \\")
    print(f"        --lr           {params['lr']:.0e} \\")
    print(f"        --beta         {params['beta']} \\")
    print(f"        --attn_dropout {params['attn_dropout']}{reduced_flag}{aggregate_flag} \\")
    print(f"        --run_dir      ./runs/{variant}\n")


if __name__ == "__main__":
    main()
