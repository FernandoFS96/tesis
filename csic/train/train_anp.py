"""
train_anp.py
==============================================================================
ANP training script for battery SoC and Cycle prediction.

All data utilities, episode construction, evaluation, and plotting live in
train_utils.py.  This script contains only:
    - Config dataclass      (all tunable hyperparameters)
    - train()               (main training loop)
    - eval_only()           (load checkpoint and evaluate)
    - parse_args() / main() (CLI entry point)

Professor's approach (cycle-based context/target windows):
    Context = first ctx_cycles × measurements_per_cycle rows
              e.g. 50 cycles × 30 meas/cycle = 1 500 rows
    Target  = next tgt_cycles × measurements_per_cycle rows
              e.g. cycles 51-100 → 1 500 rows
    Evaluation covers the remaining cycles (101-1000) — never seen in training.

Usage:
    # Full training run (defaults)
        python train_anp.py

    # ANP dual-target (SoC% + Cycle)
        python train_anp.py \
            --data_dir ../csic_real_synth_load/prepared_data \
            --target_col all
    
    # ANP solo SoC
        python train_anp.py \
            --data_dir ../csic_real_synth_load/prepared_data \
            --target_col "SoC (%)" \
            --reduced_features
        
    # ANP solo Cycle
        python train_anp.py \
            --data_dir ../csic_real_synth_load/prepared_data \
            --target_col Cycle \
            --reduced_features

    # Evaluate a saved checkpoint
        python train_anp.py --eval_only --ckpt ./runs/20260501_120000/best.pt

Outputs (inside run_dir, auto-generated as ./runs/<timestamp>/):
    best.pt            Checkpoint with lowest validation loss
    last.pt            Checkpoint at the final epoch
    metrics.csv        Per-epoch metrics (loss, NLL, KL, MAE, LR)
    config.json        Full configuration snapshot
    test_metrics.json  Final test-set metrics
    plots/             Training curve PNGs (generated automatically)
==============================================================================
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Enable memory-efficient attention kernels (Flash Attention 2 if supported, memory-efficient SDPA otherwise). 
# Applied automatically to all F.scaled_dot_product_attention calls throughout the model.
from torch.nn.attention import sdpa_kernel, SDPBackend
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)


from train_utils import (
    load_prepared_data,
    validate_targets,
    get_task_splits,
    sort_task_by_cycle,
    make_batch,
    evaluate,
    generate_all_plots,
)

try:
    from models.anp import LatentModel
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from models.anp import LatentModel


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class Config:
    """
    All hyperparameters and paths for a training run.

    Modify defaults here or override them via CLI arguments (see parse_args).
    """

    # ── Data ──────────────────────────────────────────────────────────────────
    data_dir: str = "../csic_real_synth_load/prepared_data"

    # ── Task split (0-based indices over the 25 synthetic datasets)
    # Default follows the OOD strategy requested by the professor:
    #   - Train : datasets 1-17  (diverse conditions, seen during training)
    #   - Val   : datasets 18-22 (intermediate OOD — never seen in training)
    #   - Test  : datasets 23-25 (extreme-parameter OOD — hardest evaluation)

    train_task_ids: List[int] = field(default_factory=lambda: list(range(17)))
    val_task_ids:   List[int] = field(default_factory=lambda: list(range(17, 22)))
    test_task_ids:  List[int] = field(default_factory=lambda: list(range(22, 25)))

    target_col: str = "all"   # "all" | "SoC (%)" | "Cycle"

    # ── Feature selection ─────────────────────────────────────────────────────────
    # When True, filters X to the compact RF-identified feature set for target_col.
    # Only affects when target_col is 'SoC (%)' or 'Cycle'. Ignored when target_col='all'.
    use_reduced_features: bool = False

    # ── Model ─────────────────────────────────────────────────────────────────
    num_hidden: int = 128   # Hidden dimension for all encoders and decoder
    input_dim:  int = 201   # Number of X features (auto-detected from pkl)
    output_dim: int = 2     # Number of targets: SoC (%) + Cycle

    # ── Episode construction (professor's cycle-based approach) ────────────────
    # Context  = first ctx_cycles complete cycles of the trajectory
    # Target   = next  tgt_cycles complete cycles (immediately after context)
    ctx_cycles:              int = 60   # cycles used as context
    tgt_cycles:              int = 60   # cycles used as target during training
    measurements_per_cycle:  int = 30   # measurements per cycle in the dataset

    # ── Training ──────────────────────────────────────────────────────────────
    epochs:             int   = 1000
    early_stopping:     int   = 200    # patience: epochs without val improvement
    episodes_per_epoch: int   = 100    # total episodes drawn per epoch
    batch_size:         int   = 4      # episodes per GPU forward pass
    lr:                 float = 5e-4
    lr_min:             float = 5e-5   # cosine annealing minimum LR
    attn_dropout:       float = 0.1    # dropout in attention layers
    beta:               float = 1.0    # KL weight in ELBO: loss = NLL + beta*KL
    grad_clip:          float = 1.0    # max gradient norm before clipping
    seed:               int   = 18

    # ── Logging / checkpointing ───────────────────────────────────────────────
    run_dir:   str  = ""    # auto-generated as ./runs/<timestamp>/ if empty
    log_every: int  = 10    # print to console every N epochs
    val_every: int  = 1     # run validation every N epochs
    eval_only: bool = False
    ckpt:      str  = ""    # checkpoint path for --eval_only mode

    def __post_init__(self) -> None:
        if not self.run_dir:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            target = self.target_col.replace(" ", "").replace("%", "")
            if self.use_reduced_features and self.target_col != "all":
                target += "_reduced"
            self.run_dir = f"./runs/anp_{target}/{ts}"

    @property
    def ctx_rows(self) -> int:
        """Number of context rows = ctx_cycles × measurements_per_cycle."""
        return self.ctx_cycles * self.measurements_per_cycle

    @property
    def tgt_rows(self) -> int:
        """Number of target rows = tgt_cycles × measurements_per_cycle."""
        return self.tgt_cycles * self.measurements_per_cycle


# ==============================================================================
# TRAINING LOOP
# ==============================================================================

def train(cfg: Config) -> tuple:
    """
    Run the full training pipeline.

    Steps:
        1. Set random seeds for reproducibility.
        2. Load and validate data; split into train/val/test tasks.
        3. Instantiate the ANP model, Adam optimizer, and cosine LR scheduler.
        4. For each epoch, draw steps_per_epoch batches, compute ELBO loss, and back-propagate.
        5. Every val_every epochs, evaluate on the val set and track the best checkpoint. Early stopping if patience is exceeded.
        6. After training, evaluate the best checkpoint on the test set.
        7. Generate and save training plots.

    Args:
        cfg: Config dataclass with all hyperparameters.

    Returns:
        (model, test_metrics)
    """

    # ── Reproducibility ───────────────────────────────────────────────────────
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧  Device: {device}")

    # ── Run directory ─────────────────────────────────────────────────────────
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁  Run dir: {run_dir}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"\n📂  Loading data from: {cfg.data_dir}")
    data = load_prepared_data(cfg.data_dir)
    validate_targets(data)

    # ── Feature reduction (optional) ──────────────────────────────────────────────
    if cfg.use_reduced_features:
        from train_utils import apply_feature_reduction
        data = apply_feature_reduction(data, cfg.target_col)
    # ── Target filtering ──────────────────────────────────────────────────────────
    if cfg.target_col != "all":
        for split_key in ["normalized_synth_datasets"]:
            data[split_key] = [
                (X, y[[cfg.target_col]]) for X, y in data[split_key]
            ]
        rd = data["normalized_real_dataset"]
        data["normalized_real_dataset"] = (rd[0], rd[1][[cfg.target_col]])
        # Actualizar también denorm_values para que solo tenga el target seleccionado
        for k in ["y_mean", "y_std"]:
            data["denorm_values"][k] = {
                cfg.target_col: data["denorm_values"][k][cfg.target_col]
            }

    target_cols    = list(data["normalized_synth_datasets"][0][1].columns)
    cfg.output_dim = len(target_cols)
    cfg.input_dim  = data["normalized_synth_datasets"][0][0].shape[1]
    print(f" input_dim  = {cfg.input_dim}")
    print(f" output_dim = {cfg.output_dim}  {target_cols}")
    print(f" ctx_rows   = {cfg.ctx_rows} ({cfg.ctx_cycles} cycles × {cfg.measurements_per_cycle} meas/cycle)")
    print(f" tgt_rows   = {cfg.tgt_rows} ({cfg.tgt_cycles} cycles × {cfg.measurements_per_cycle} meas/cycle)")

    train_tasks, val_tasks, test_tasks = get_task_splits(
        data, cfg.train_task_ids, cfg.val_task_ids, cfg.test_task_ids
    )
    denorm_values = {
        "y_mean": data["denorm_values"]["y_mean"],
        "y_std":  data["denorm_values"]["y_std"],
    }

    # Pre-sort all tasks by cycle once at startup
    def presort(tasks: list) -> list:
        return [sort_task_by_cycle(X, y) for X, y in tasks]

    train_sorted = presort(train_tasks)
    val_sorted   = presort(val_tasks)
    test_sorted  = presort(test_tasks)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = LatentModel(
        num_hidden=cfg.num_hidden,
        input_dim=cfg.input_dim,
        output_dim=cfg.output_dim,
        attn_dropout=cfg.attn_dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠  ANP model: {n_params:,} trainable parameters")

    # ── Optimizer and scheduler ───────────────────────────────────────────────
    optimizer = Adam(model.parameters(), lr=cfg.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr_min)

    # ── Save config snapshot ──────────────────────────────────────────────────
    cfg_dict = asdict(cfg)
    cfg_dict["target_cols"] = target_cols
    cfg_dict["n_params"]    = n_params
    cfg_dict["ctx_rows"]    = cfg.ctx_rows
    cfg_dict["tgt_rows"]    = cfg.tgt_rows
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)

    # ── Training state ────────────────────────────────────────────────────────
    best_val_MAE          = float("inf")
    epochs_without_improve = 0
    metrics_rows           = []
    steps_per_epoch        = max(1, cfg.episodes_per_epoch // cfg.batch_size)

    print(f"\n🚀  Starting training — {cfg.epochs} epochs  |  "
          f"{steps_per_epoch} steps/epoch  |  batch_size={cfg.batch_size}\n")

    pbar = tqdm(range(1, cfg.epochs + 1), desc="Train", unit="epoch", dynamic_ncols=True)

    for epoch in pbar:
        model.train()
        ep_losses, ep_nlls, ep_kls = [], [], []

        # ── Gradient steps ────────────────────────────────────────────────────
        for _ in range(steps_per_epoch):
            ctx_x, ctx_y, tgt_x, tgt_y = make_batch(
                train_sorted,
                batch_size=cfg.batch_size,
                ctx_rows=cfg.ctx_rows,
                tgt_rows=cfg.tgt_rows,
                device=device,
            )

            optimizer.zero_grad()
            _, _, loss, kl, nll = model(
                ctx_x, ctx_y, tgt_x, tgt_y, beta=cfg.beta
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
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

        # ── Validation ────────────────────────────────────────────────────────
        if epoch % cfg.val_every == 0 or epoch == cfg.epochs:
            val_metrics = evaluate(
                model, val_sorted,
                ctx_rows=cfg.ctx_rows,
                tgt_rows=cfg.tgt_rows,
                device=device,
                denorm_values=denorm_values,
                target_cols=target_cols,
                beta=cfg.beta,
                split_name="val",
            )
            row.update(val_metrics)

            # early stopping over SoC MAE desnormalized back to percentage points
            if cfg.target_col == "Cycle":
                es_metric_key = "val/mae_Cycle"
            elif cfg.target_col == "SoC (%)":
                es_metric_key = "val/mae_SoC_pct"
            else:  # "all" — mantener SoC como criterio principal
                es_metric_key = "val/mae_SoC_pct"

            current_val = val_metrics.get(es_metric_key, float("inf"))

            if current_val < best_val_MAE:
                best_val_MAE = current_val
                epochs_without_improve = 0
                torch.save(
                    {
                        "epoch":     epoch,
                        "model":     model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "val_MAE":   best_val_MAE,
                        "cfg":       cfg_dict,
                    },
                    run_dir / "best.pt",
                )
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= cfg.early_stopping:
                    print(f"\n⏹  Early stopping at epoch {epoch} "
                          f"(no improvement for {cfg.early_stopping} epochs)")
                    metrics_rows.append(row)
                    break

        # ── Progress bar ──────────────────────────────────────────────────────
        postfix: dict = {
            "loss": f"{train_loss:.2f}",
            #"nll":  f"{train_nll:.2f}",
            #"kl":   f"{train_kl:.3f}",
            #"lr":   f"{lr_now:.1e}",
        }
        if "val/loss" in row:
            postfix["val_loss"] = f"{row['val/loss']:.2f}"
            if "val/mae_SoC_pct" in row:
                postfix["mae_soc"] = f"{row['val/mae_SoC_pct']:.2f}"
                postfix["best_soc"] = f"{best_val_MAE:.2f}"
            if "val/mae_Cycle" in row:
                postfix["mae_cyc"] = f"{row['val/mae_Cycle']:.2f}"
                postfix["best_cyc"] = f"{best_val_MAE:.2f}"
            postfix["E_S"] = f"{epochs_without_improve}/{cfg.early_stopping}"
        pbar.set_postfix(postfix)

        metrics_rows.append(row)

    # ── Save last checkpoint and metrics CSV ──────────────────────────────────
    torch.save(
        {
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cfg":       cfg_dict,
        },
        run_dir / "last.pt",
    )

    pd.DataFrame(metrics_rows).to_csv(run_dir / "metrics.csv", index=False)
    print(f"\n📊  Metrics saved to: {run_dir / 'metrics.csv'}")

    # ── Final test evaluation ─────────────────────────────────────────────────
    print(f"\n🏁  Final evaluation on TEST set (best checkpoint)...")
    best_ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model"])

    test_metrics = evaluate(
        model, test_sorted,
        ctx_rows=cfg.ctx_rows,
        tgt_rows=cfg.tgt_rows,
        device=device,
        denorm_values=denorm_values,
        target_cols=target_cols,
        beta=cfg.beta,
        split_name="test",
    )

    print(f"\n  {'─'*52}")
    print(f"  {'Metric':<38} {'Value':>10}")
    print(f"  {'─'*52}")
    for k, v in sorted(test_metrics.items()):
        print(f"  {k:<38} {v:>10.4f}")
    print(f"  {'─'*52}")

    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    # ── Generate plots ────────────────────────────────────────────────────────
    generate_all_plots(run_dir, target_cols)

    print(f"\n✅  Training complete. All outputs in: {run_dir}\n")
    return model, test_metrics


# ==============================================================================
# EVAL-ONLY MODE
# ==============================================================================

def eval_only(cfg: Config) -> None:
    """
    Load a saved checkpoint and evaluate it on val and test sets.

    Args:
        cfg: Config with cfg.ckpt pointing to a valid .pt file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_prepared_data(cfg.data_dir)
    validate_targets(data)

    # ── Feature reduction (optional) ──────────────────────────────────────────────
    if cfg.use_reduced_features:
        from train_utils import apply_feature_reduction
        data = apply_feature_reduction(data, cfg.target_col)

    if cfg.target_col != "all":
        data["normalized_synth_datasets"] = [
            (X, y[[cfg.target_col]]) for X, y in data["normalized_synth_datasets"]
        ]
        rd = data["normalized_real_dataset"]
        data["normalized_real_dataset"] = (rd[0], rd[1][[cfg.target_col]])
        for k in ["y_mean", "y_std"]:
            data["denorm_values"][k] = {
                cfg.target_col: data["denorm_values"][k][cfg.target_col]
            }

    target_cols    = list(data["normalized_synth_datasets"][0][1].columns)
    cfg.output_dim = len(target_cols)
    cfg.input_dim  = data["normalized_synth_datasets"][0][0].shape[1]

    _, val_tasks, test_tasks = get_task_splits(
        data, cfg.train_task_ids, cfg.val_task_ids, cfg.test_task_ids
    )
    denorm_values = {
        "y_mean": data["denorm_values"]["y_mean"],
        "y_std":  data["denorm_values"]["y_std"],
    }

    def presort(tasks: list) -> list:
        return [sort_task_by_cycle(X, y) for X, y in tasks]

    model = LatentModel(cfg.num_hidden, cfg.input_dim, cfg.output_dim, attn_dropout=cfg.attn_dropout).to(device)
    ckpt  = torch.load(cfg.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"✓ Checkpoint loaded: {cfg.ckpt}  (epoch {ckpt.get('epoch', '?')})")

    for split_name, tasks in [
        ("val",  presort(val_tasks)),
        ("test", presort(test_tasks)),
    ]:
        metrics = evaluate(
            model, tasks,
            ctx_rows=cfg.ctx_rows,
            tgt_rows=cfg.tgt_rows,
            device=device,
            denorm_values=denorm_values,
            target_cols=target_cols,
            beta=cfg.beta,
            split_name=split_name,
        )
        print(f"\n── {split_name.upper()} ─────────────────────────────")
        for k, v in sorted(metrics.items()):
            print(f"   {k:<42} {v:.4f}")


# ==============================================================================
# CLI
# ==============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ANP training — battery SoC and Cycle prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir",       type=str,   default=None)
    p.add_argument("--run_dir",        type=str,   default="")
    p.add_argument("--target_col",     type=str, default="all", choices=["all", "SoC (%)", "Cycle"], help="Target a predecir: 'all' entrena ambos, o uno solo")
    p.add_argument( "--reduced_features", action="store_true", dest="use_reduced_features", help=("Filter X to the compact RF-identified feature set for the selected target_col. No effect when target_col='all'."),)
    p.add_argument("--num_hidden",     type=int,   default=128)
    p.add_argument("--ctx_cycles",     type=int,   default=60)
    p.add_argument("--tgt_cycles",     type=int,   default=60)
    p.add_argument("--meas_per_cycle", type=int,   default=30, dest="measurements_per_cycle")
    p.add_argument("--epochs",         type=int,   default=1000)
    p.add_argument("--early_stop",     type=int,   default=200, dest="early_stopping")
    p.add_argument("--episodes",       type=int,   default=100, dest="episodes_per_epoch")
    p.add_argument("--batch_size",     type=int,   default=4)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--lr_min",         type=float, default=5e-5)
    p.add_argument("--attn_dropout",   type=float, default=0.2)
    p.add_argument("--beta",           type=float, default=0.5)
    p.add_argument("--grad_clip",      type=float, default=1.0)
    p.add_argument("--seed",           type=int,   default=18)
    p.add_argument("--log_every",      type=int,   default=10)
    p.add_argument("--val_every",      type=int,   default=1)
    p.add_argument("--eval_only",      action="store_true")
    p.add_argument("--ckpt",           type=str,   default="")
    p.add_argument("--train_ids",      type=int,   nargs="+", default=None)
    p.add_argument("--val_ids",        type=int,   nargs="+", default=None)
    p.add_argument("--test_ids",       type=int,   nargs="+", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Config(
        data_dir               = args.data_dir or Config.data_dir,
        run_dir                = args.run_dir,
        target_col             = args.target_col,
        use_reduced_features   = args.use_reduced_features,
        num_hidden             = args.num_hidden,
        ctx_cycles             = args.ctx_cycles,
        tgt_cycles             = args.tgt_cycles,
        measurements_per_cycle = args.measurements_per_cycle,
        epochs                 = args.epochs,
        early_stopping         = args.early_stopping,
        episodes_per_epoch     = args.episodes_per_epoch,
        batch_size             = args.batch_size,
        lr                     = args.lr,
        lr_min                 = args.lr_min,
        attn_dropout           = args.attn_dropout,
        beta                   = args.beta,
        grad_clip              = args.grad_clip,
        seed                   = args.seed,
        log_every              = args.log_every,
        val_every              = args.val_every,
        eval_only              = args.eval_only,
        ckpt                   = args.ckpt,
    )

    if args.train_ids is not None:
        cfg.train_task_ids = args.train_ids
    if args.val_ids is not None:
        cfg.val_task_ids = args.val_ids
    if args.test_ids is not None:
        cfg.test_task_ids = args.test_ids

    if cfg.eval_only:
        if not cfg.ckpt:
            raise ValueError("--eval_only requires --ckpt <checkpoint_path>")
        eval_only(cfg)
    else:
        train(cfg)


if __name__ == "__main__":
    main()
