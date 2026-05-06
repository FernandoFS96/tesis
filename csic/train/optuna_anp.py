"""
optuna_anp.py
==============================================================================
Hyperparameter optimisation for the ANP model using Optuna.

Fixed parameters (not optimised):
    ctx_cycles     = 60   (1 800 context rows)
    tgt_cycles     = 60   (1 800 target rows)
    meas_per_cycle = 30
    epochs         = 200  (per trial)
    early_stopping = 40   (patience per trial)

Search space (all categorical — controlled sweep):
    num_hidden   : [64, 128, 192, 256]
    batch_size   : [2, 4, 6, 8]
    lr           : [1e-4, 5e-4, 1e-3, 5e-3]
    beta         : [0.1, 0.5, 1.0, 2.0]
    attn_dropout : [0.0, 0.1, 0.2]

No pruning — every trial runs to completion (full EPOCHS or early stopping).

Per-trial outputs (in --out_dir/trial_NNN/):
    best.pt        Best model weights for this trial
    metrics.csv    Per-epoch loss, NLL, KL, MAE
    plots/         Training curve PNGs (same as train_anp.py)

Global outputs (in --out_dir/):
    study.db           SQLite database — resumable across runs
    best_params.json   Best hyperparameters found
    all_trials.csv     All trial results
    optimization.png   Optimization history
    importance.png     Hyperparameter importance (fANOVA)

Usage:
    python optuna_anp.py --n_trials 200 --data_dir ../csic_real_synth_load/prepared_data

Resume an interrupted search (picks up from study.db automatically):
    python optuna_anp.py --n_trials 200 --data_dir ../csic_real_synth_load/prepared_data

Run the best config after search:
    python train_anp.py --num_hidden 128 --batch_size 6 --lr 3e-4 --beta 0.1 ...
    (exact command printed at the end of the search)
==============================================================================
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import optuna
from optuna.samplers import TPESampler
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

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
)

from models.anp import LatentModel
import models.anp as anp_module

# ==============================================================================
# FIXED EXPERIMENT PARAMETERS
# ==============================================================================

CTX_CYCLES     = 60
TGT_CYCLES     = 60
MEAS_PER_CYCLE = 30
EPOCHS         = 1000
EARLY_STOP     = 100    # patience in epochs (not in val-check intervals)
EPISODES       = 100
LR_MIN         = 1e-5  # cosine annealing floor — always fixed
SEED           = 18

TRAIN_IDS = list(range(17))
VAL_IDS   = list(range(17, 22))
TEST_IDS  = list(range(22, 25))

# ── Categorical search space ──────────────────────────────────────────────────
# lr: four values spanning two orders of magnitude
LR_CHOICES: List[float] = [1e-4, 5e-4, 1e-3, 5e-3]

# beta: KL weight in ELBO = NLL + beta * KL
#   0.01 → almost free latent space, model behaves like deterministic
#   0.10 → light regularisation
#   0.50 → moderate regularisation
#   1.00 → standard beta-VAE / ANP default
#   2.00 → strong KL regularisation, encourages diverse latent
BETA_CHOICES: List[float] = [0.1, 0.5, 1.0, 2.0]

# attn_dropout: dropout probability inside MultiheadAttention
DROPOUT_CHOICES: List[float] = [0.0, 0.1, 0.2]


# ==============================================================================
# OBJECTIVE
# ==============================================================================

def objective(
    trial:      optuna.Trial,
    data:       dict,
    device:     torch.device,
    out_dir:    Path,
) -> float:
    """
    Optuna objective — trains one ANP configuration and returns best val loss.

    No pruning: every trial runs to its natural end (EPOCHS epochs or early stopping). 
    Per-trial outputs (weights, metrics, plots) are saved to out_dir/trial_NNN/.

    Args:
        trial:   Optuna trial (handles parameter sampling).
        data:    Pre-loaded data dictionary.
        device:  Torch device.
        out_dir: Root output directory; trial subdir created inside.

    Returns:
        Best validation loss achieved during this trial (lower is better).
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
    print(f"     num_hidden={num_hidden}  batch_size={batch_size}  "
          f"lr={lr:.0e}  beta={beta}  dropout={attn_dropout}")

    # ── Data ──────────────────────────────────────────────────────────────────
    target_cols = list(data["normalized_synth_datasets"][0][1].columns)
    input_dim   = data["normalized_synth_datasets"][0][0].shape[1]
    output_dim  = len(target_cols)
    ctx_rows    = CTX_CYCLES * MEAS_PER_CYCLE
    tgt_rows    = TGT_CYCLES * MEAS_PER_CYCLE

    train_tasks, val_tasks, _ = get_task_splits(data, TRAIN_IDS, VAL_IDS, TEST_IDS)

    def presort(tasks):
        return [sort_task_by_cycle(X, y) for X, y in tasks]

    train_sorted = presort(train_tasks)
    val_sorted   = presort(val_tasks)
    denorm_values = {
        "y_mean": data["denorm_values"]["y_mean"],
        "y_std":  data["denorm_values"]["y_std"],
    }

    # ── Model — inject attn_dropout for this trial ────────────────────────────
    # Temporarily patch MultiheadAttention dropout, then restore immediately.
    # This is isolated per-trial and does not affect other trials or anp.py.
    _orig_init = anp_module.MultiheadAttention.__init__

    def _patched_init(self, num_hidden_k):
        _orig_init(self, num_hidden_k)
        self.attn_dropout = nn.Dropout(p=attn_dropout)

    anp_module.MultiheadAttention.__init__ = _patched_init
    model = LatentModel(
        num_hidden=num_hidden,
        input_dim=input_dim,
        output_dim=output_dim,
    ).to(device)
    anp_module.MultiheadAttention.__init__ = _orig_init  # restore immediately

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"     params: {n_params:,}")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR_MIN)

    # ── Training loop ─────────────────────────────────────────────────────────
    steps_per_epoch  = max(1, EPISODES // batch_size)
    best_mae_soc    = float("inf")
    best_model_state = None
    no_improve       = 0
    metrics_rows: list = []

    pbar = tqdm(
        range(1, EPOCHS + 1),
        desc=f"Trial {trial_id:03d} "
             f"[h={num_hidden} b={batch_size} lr={lr:.0e} β={beta} d={attn_dropout}]",
        unit="ep",
        leave=True,   # keep the bar visible after the trial ends
    )

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
        train_nll  = float(np.mean(ep_nlls))
        train_kl   = float(np.mean(ep_kls))
        lr_now     = scheduler.get_last_lr()[0]

        row = {
            "epoch":      epoch,
            "train/loss": train_loss,
            "train/nll":  train_nll,
            "train/kl":   train_kl,
            "lr":         lr_now,
        }

        # Validate every epoch (same as train_anp.py default)
        val_metrics = evaluate(
            model, val_sorted, ctx_rows, tgt_rows, device,
            denorm_values, target_cols, beta=beta, split_name="val"
        )
        row.update(val_metrics)
        val_loss = val_metrics["val/loss"]

        # Update progress bar with current metrics
        soc_key = "val/mae_SoC_pct"
        pbar.set_postfix({
            "loss": f"{train_loss:.3f}",
            "val": f"{val_loss:.3f}",
            "mae_soc": f"{val_metrics.get(soc_key, float('nan')):.2f}",
            "best_mae_soc": f"{min(best_mae_soc, val_metrics.get(soc_key, float('nan'))):.2f}",
            "E_S": f"{no_improve}/{EARLY_STOP}",
        })

        # Track best and apply early stopping
        current_mae_soc = val_metrics.get("val/mae_SoC_pct", float("inf"))
        if current_mae_soc  < best_mae_soc:
            best_mae_soc = current_mae_soc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        metrics_rows.append(row)

        if no_improve >= EARLY_STOP:
            pbar.set_description(
                f"Trial {trial_id:03d} ⏹ early stop ep={epoch}"
            )
            break
    pbar.close()

    # ── Save trial outputs ────────────────────────────────────────────────────
    # 1. Best model weights
    torch.save(
        {
            "trial":      trial_id,
            "epoch":      epoch,
            "model":      best_model_state,
            "val_MAE":   best_mae_soc,
            "params":     trial.params,
            "n_params":   n_params,
            "target_cols": target_cols,
        },
        trial_dir / "best.pt",
    )

    # 2. Metrics CSV
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(trial_dir / "metrics.csv", index=False)

    # 3. Training plots (reuses generate_all_plots from train_utils)
    generate_all_plots(trial_dir, target_cols)

    # 4. Trial config JSON (quick reference without opening study.db)
    with open(trial_dir / "config.json", "w") as f:
        json.dump(
            {
                "trial_id":    trial_id,
                "val_MAE":    best_mae_soc,
                "params":      trial.params,
                "n_params":    n_params,
                "ctx_cycles":  CTX_CYCLES,
                "tgt_cycles":  TGT_CYCLES,
                "ctx_rows":    ctx_rows,
                "tgt_rows":    tgt_rows,
                "epochs_run":  epoch,
                "target_cols": target_cols,
            },
            f, indent=2,
        )

    print(f"     best val_MAE = {best_mae_soc:.2f}  "
          f"(epoch {epoch}  |  saved to trial_{trial_id:03d}/)")

    return best_mae_soc


# ==============================================================================
# PLOTTING
# ==============================================================================

def plot_study_results(study: optuna.Study, out_dir: Path) -> None:
    """
    Save optimization history and hyperparameter importance plots for the full study.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from optuna.visualization.matplotlib import (
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate,
        )

        # Optimization history
        fig, ax = plt.subplots(figsize=(11, 4))
        plot_optimization_history(study, ax=ax)
        ax.set_title("Optimization history — best val loss per trial")
        fig.tight_layout()
        fig.savefig(out_dir / "optimization.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  ✓  optimization.png")

        # Hyperparameter importance
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

        # Parallel coordinate (useful to spot interaction patterns)
        if len(completed) >= 4:
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
        description="Optuna hyperparameter search for ANP — no pruning, per-trial outputs saved",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir",   type=str, default="../csic_real_synth_load/prepared_data")
    p.add_argument("--n_trials",   type=int, default=200, help="Total number of trials to run")
    p.add_argument("--out_dir",    type=str, default="./optuna_results", help="Root output directory")
    p.add_argument("--study_name", type=str, default="anp_battery", help="Optuna study name (used for study.db)")
    args = p.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🔧  Device  : {device}")
    print(f"📁  Out dir : {out_dir}")
    print(f"🔍  Trials  : {args.n_trials}")
    print(f"\n  Search space:")
    print(f"    num_hidden   : {[64, 128, 192, 256]}")
    print(f"    batch_size   : {[2, 4, 6, 8]}")
    print(f"    lr           : {LR_CHOICES}")
    print(f"    beta         : {BETA_CHOICES}")
    print(f"    attn_dropout : {DROPOUT_CHOICES}")
    total_combinations = 4 * 4 * 4 * 4 * 3
    print(f"\n  Total possible combinations : {total_combinations}")
    print(f"  Trials requested            : {args.n_trials}")
    print(f"  Coverage                    : {args.n_trials/total_combinations*100:.1f}%")

    # Load data once — shared across all trials
    print(f"\n📂  Loading data...")
    data = load_prepared_data(args.data_dir)
    validate_targets(data)

    # Create or resume study — NopPruner disables all pruning
    storage = f"sqlite:///{out_dir / 'study.db'}"
    study   = optuna.create_study(
        study_name     = args.study_name,
        storage        = storage,
        direction      = "minimize",
        sampler        = TPESampler(seed=SEED),
        pruner         = optuna.pruners.NopPruner(),  # no pruning
        load_if_exists = True,
    )

    n_existing = len([t for t in study.trials
                      if t.state == optuna.trial.TrialState.COMPLETE])
    if n_existing > 0:
        print(f"\n  ↩  Resuming study — {n_existing} trials already completed")

    print(f"\n🚀  Starting optimisation...\n")

    study.optimize(
        lambda trial: objective(trial, data, device, out_dir),
        n_trials          = args.n_trials,
        n_jobs            = 1,   # single GPU — always 1
        show_progress_bar = False,  # we print per-trial ourselves
    )

    # ── Results ───────────────────────────────────────────────────────────────
    best = study.best_trial
    print(f"\n{'='*58}")
    print(f"  Best trial   : #{best.number:03d}")
    print(f"  Best val loss: {best.value:.2f}")
    print(f"\n  Best hyperparameters:")
    for k, v in best.params.items():
        print(f"    {k:<20} = {v}")
    print(f"{'='*58}\n")

    # Save best params
    best_info = {
        "trial":     best.number,
        "val_loss":  best.value,
        "params":    best.params,
        "trial_dir": f"trial_{best.number:03d}",
    }
    with open(out_dir / "best_params.json", "w") as f:
        json.dump(best_info, f, indent=2)
    print(f"  ✓  best_params.json")

    # Save all trials CSV
    study.trials_dataframe().to_csv(out_dir / "all_trials.csv", index=False)
    print(f"  ✓  all_trials.csv")

    # Study-level plots
    print(f"\n📈  Saving study plots...")
    plot_study_results(study, out_dir)

    # Print ready-to-use training command
    params = best.params
    print(f"\n✅  Optimisation complete.")
    print(f"\n  Best checkpoint: {out_dir / f'trial_{best.number:03d}' / 'best.pt'}")
    print(f"\n  To run full training with best hyperparameters:")
    print(f"\n    python train_anp.py \\")
    print(f"        --num_hidden   {params['num_hidden']} \\")
    print(f"        --batch_size   {params['batch_size']} \\")
    print(f"        --lr           {params['lr']:.0e} \\")
    print(f"        --beta         {params['beta']} \\")
    print(f"        --ctx_cycles   {CTX_CYCLES} \\")
    print(f"        --tgt_cycles   {TGT_CYCLES} \\")
    print(f"        --epochs       1000\n")


if __name__ == "__main__":
    main()
