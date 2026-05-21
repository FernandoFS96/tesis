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
              e.g. 60 cycles × 30 meas/cycle = 1 800 rows
    Target  = next  tgt_cycles × measurements_per_cycle rows
              e.g. cycles 61-120 → 1 800 rows
    Evaluation covers the remaining cycles (121-1000) — never seen in training.

Supported model variants (controlled by flags):
    # ANP dual-target (SoC% + Cycle)
    python train_anp.py --target_col all

    # ANP SoC-only (full features)
    python train_anp.py --target_col "SoC (%)"

    # ANP SoC-only (RF-reduced: 11 features)
    python train_anp.py --target_col "SoC (%)" --reduced_features

    # ANP Cycle-only (full features)
    python train_anp.py --target_col Cycle

    # ANP Cycle-only (RF-reduced: 12 features)
    python train_anp.py --target_col Cycle --reduced_features

    # ANP Cycle-only (RF-reduced + cycle-level aggregation)
    python train_anp.py --target_col Cycle --reduced_features --aggregate_by_cycle

    # ANP Cycle-only (RF-reduced + aggregation + SoC-enriched features)
    python train_anp.py \
        --target_col Cycle --reduced_features --aggregate_by_cycle \
        --enrich_soc_predictions \
        --anp_soc_run_dir ./optuna_results/anp_soc_reduced/trial_031

    # Evaluate a saved checkpoint
    python train_anp.py --eval_only --ckpt ./runs/anp_Cycle/20260519/best.pt

Outputs (inside run_dir, auto-generated as ./runs/<variant>/<timestamp>/):
    best.pt            Checkpoint with lowest validation metric
    last.pt            Checkpoint at the final epoch
    metrics.csv        Per-epoch metrics (loss, NLL, KL, MAE, LR)
    config.json        Full configuration snapshot (read by evaluate scripts)
    test_metrics.json  Final test-set metrics
    plots/             Training curve PNGs
==============================================================================
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

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
    REDUCED_FEATURE_SETS,
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

    The run_dir is auto-generated from the model variant flags if not specified,
    so different variants land in clearly named subdirectories:
        ./runs/anp_all/<timestamp>/
        ./runs/anp_SoC_reduced/<timestamp>/
        ./runs/anp_Cycle_agg_reduced/<timestamp>/
        ./runs/anp_Cycle_soc_enriched/<timestamp>/
    """

    # ── Data ──────────────────────────────────────────────────────────────────
    data_dir: str = "../csic_real_synth_load/prepared_data"

    # Task split (0-based indices over the 25 synthetic datasets)
    train_task_ids: List[int] = field(default_factory=lambda: list(range(17)))
    val_task_ids:   List[int] = field(default_factory=lambda: list(range(17, 22)))
    test_task_ids:  List[int] = field(default_factory=lambda: list(range(22, 25)))

    # ── Target ────────────────────────────────────────────────────────────────
    target_col: str = "all"   # "all" | "SoC (%)" | "Cycle"

    # ── Feature pipeline ──────────────────────────────────────────────────────
    use_reduced_features: bool = False
    # When True, filters X to the compact RF-identified feature set for
    # target_col. No effect when target_col='all'.

    aggregate_by_cycle: bool = False
    # When True, averages EIS measurements per cycle before training.
    # ctx/tgt windows operate at cycle-level (1 row/cycle) instead of
    # measurement-level (30 rows/cycle). Recommended for Cycle models.

    enrich_soc_predictions: bool = False
    # When True, adds 4 per-cycle SoC statistics from a pre-trained
    # ANP-SoC model as additional features to the cycle-aggregated X:
    #   soc_pred_mean, soc_pred_min, soc_pred_max, soc_pred_range
    # Requires aggregate_by_cycle=True and anp_soc_run_dir to be set.

    anp_soc_run_dir: str = ""
    # Path to a pre-trained ANP-SoC run directory (must contain best.pt and
    # config.json). Required when enrich_soc_predictions=True.
    # Example: ./optuna_results/anp_soc_reduced/trial_031

    # ── Model ─────────────────────────────────────────────────────────────────
    num_hidden: int = 128
    input_dim:  int = 201   # auto-detected from pkl after all transformations
    output_dim: int = 2     # auto-detected from target_col

    # ── Episode construction ───────────────────────────────────────────────────
    ctx_cycles:             int = 60
    tgt_cycles:             int = 60
    measurements_per_cycle: int = 30

    # ── Training ──────────────────────────────────────────────────────────────
    epochs:             int   = 1000
    early_stopping:     int   = 200
    episodes_per_epoch: int   = 100
    batch_size:         int   = 4
    lr:                 float = 5e-4
    lr_min:             float = 5e-5
    attn_dropout:       float = 0.1
    beta:               float = 1.0
    grad_clip:          float = 1.0
    seed:               int   = 18

    # ── Logging / checkpointing ───────────────────────────────────────────────
    run_dir:   str  = ""
    log_every: int  = 10
    val_every: int  = 1
    eval_only: bool = False
    ckpt:      str  = ""

    def __post_init__(self) -> None:
        if not self.run_dir:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Build a descriptive tag from the active variant flags
            if self.target_col == "all":
                tag = "anp_all"
            else:
                base = self.target_col.replace(" ", "").replace("(%)", "pct").replace("(", "").replace(")", "")
                parts = [f"anp_{base}"]
                if self.use_reduced_features:
                    parts.append("reduced")
                if self.aggregate_by_cycle:
                    parts.append("agg")
                if self.enrich_soc_predictions:
                    parts.append("enriched")
                tag = "_".join(parts)
            self.run_dir = f"./runs/{tag}/{ts}"

    @property
    def ctx_rows(self) -> int:
        """Context window in rows. 1 row/cycle if aggregate_by_cycle, else 30."""
        return self.ctx_cycles if self.aggregate_by_cycle \
            else self.ctx_cycles * self.measurements_per_cycle

    @property
    def tgt_rows(self) -> int:
        """Target window in rows."""
        return self.tgt_cycles if self.aggregate_by_cycle \
            else self.tgt_cycles * self.measurements_per_cycle


# ==============================================================================
# DATA PIPELINE HELPERS
# ==============================================================================

def _load_anp_soc_model(
    soc_run_dir: Path,
    device:      torch.device,
) -> tuple:
    """
    Load a pre-trained ANP-SoC model from a run directory.

    Reads config.json to detect num_hidden, attn_dropout, and whether the
    model used reduced features (which determines soc_feat_cols).

    Returns:
        (soc_model, soc_feat_cols)
        soc_feat_cols: list of X column names to filter to, or None for full.
    """
    soc_ckpt = soc_run_dir / "best.pt"
    soc_cfg  = soc_run_dir / "config.json"

    if not soc_ckpt.exists():
        raise FileNotFoundError(f"ANP-SoC checkpoint not found: {soc_ckpt}")

    cfg_data     = {}
    num_hidden   = 128
    attn_dropout = 0.1
    target_col   = "SoC (%)"
    use_reduced  = False

    if soc_cfg.exists():
        with soc_cfg.open() as f:
            cfg_data = json.load(f)
        num_hidden   = (cfg_data.get("num_hidden")
                        or cfg_data.get("params", {}).get("num_hidden", 128))
        attn_dropout = cfg_data.get("attn_dropout", 0.1)
        target_col   = cfg_data.get("target_col", "SoC (%)")
        use_reduced  = cfg_data.get("use_reduced_features", False)

    soc_feat_cols = REDUCED_FEATURE_SETS.get(target_col) if use_reduced else None
    soc_input_dim = len(soc_feat_cols) if soc_feat_cols else None  # resolved later

    raw = torch.load(soc_ckpt, map_location="cpu")

    # Infer actual input_dim from the checkpoint weight shape
    lat_key = next(
        (k for k in raw["model"]
         if "latent_encoder.input_projection.linear_layer.weight" in k), None
    )
    if lat_key:
        # weight shape: [num_hidden, input_dim + output_dim(=1)]
        soc_input_dim = raw["model"][lat_key].shape[1] - 1

    model = LatentModel(
        num_hidden=num_hidden,
        input_dim=soc_input_dim,
        output_dim=1,
        attn_dropout=attn_dropout,
    )
    model.load_state_dict(raw["model"])
    model.eval().to(device)

    print(f"   SoC model loaded: {soc_run_dir.name}  "
          f"(num_hidden={num_hidden}  input_dim={soc_input_dim}  "
          f"features={'reduced' if soc_feat_cols else 'all'})")
    return model, soc_feat_cols


def _build_pipeline(
    data:        dict,
    cfg:         Config,
    device:      torch.device,
    data_raw:    dict,
) -> dict:
    """
    Apply the full feature/aggregation/enrichment pipeline to data.

    Order of transformations:
        1. Feature reduction (X columns filtered for single-target models)
        2. Target filtering  (y columns filtered to cfg.target_col)
        3. Cycle aggregation (30 rows/cycle → 1 row/cycle)
        4. SoC enrichment    (add 4 per-cycle SoC statistics from ANP-SoC)

    data_raw is the untransformed data used as source for SoC enrichment
    (the ANP-SoC needs the original 202-feature X, not the filtered version).

    Returns the transformed data dict (shallow copy — originals untouched).
    """
    from train_utils import apply_feature_reduction, aggregate_by_cycle

    # 1. Feature reduction
    if cfg.use_reduced_features and cfg.target_col != "all":
        data = apply_feature_reduction(data, cfg.target_col)
        print(f"   Feature reduction → {data['normalized_synth_datasets'][0][0].shape[1]} features")

    # 2. Target filtering
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

    # 3. Cycle aggregation
    if cfg.aggregate_by_cycle:
        print(f"\n🔄  Aggregating measurements by cycle...")
        data["normalized_synth_datasets"] = [
            aggregate_by_cycle(X, y) for X, y in data["normalized_synth_datasets"]
        ]
        real_X, real_y = data["normalized_real_dataset"]
        data["normalized_real_dataset"] = aggregate_by_cycle(real_X, real_y)
        n_cyc = len(data["normalized_synth_datasets"][0][0])
        print(f"   Dataset size: {n_cyc} rows/task  "
              f"(was ~{n_cyc * cfg.measurements_per_cycle})")
        print(f"   ctx_rows = {cfg.ctx_rows}  tgt_rows = {cfg.tgt_rows}  (cycle-level)")

    # 4. SoC enrichment (requires aggregate_by_cycle + anp_soc_run_dir)
    if cfg.enrich_soc_predictions:
        if not cfg.aggregate_by_cycle:
            raise ValueError(
                "--enrich_soc_predictions requires --aggregate_by_cycle. "
                "SoC enrichment only makes sense for cycle-level data."
            )
        if not cfg.anp_soc_run_dir:
            raise ValueError(
                "--enrich_soc_predictions requires --anp_soc_run_dir pointing "
                "to a trained ANP-SoC checkpoint."
            )

        from train_utils import enrich_with_soc_predictions

        soc_run_dir  = Path(cfg.anp_soc_run_dir)
        soc_model, soc_feat_cols = _load_anp_soc_model(soc_run_dir, device)

        print(f"\n🔬  Enriching with ANP-SoC predictions...")

        # Build raw measurement-level tasks from data_raw (full 202 features)
        n_tasks  = len(data_raw["normalized_synth_datasets"])
        tasks_raw_all = [
            sort_task_by_cycle(*data_raw["normalized_synth_datasets"][i])
            for i in range(n_tasks)
        ]
        tasks_agg_all = [
            sort_task_by_cycle(*data["normalized_synth_datasets"][i])
            for i in range(n_tasks)
        ]

        enriched_synth = enrich_with_soc_predictions(
            tasks_raw     = tasks_raw_all,
            tasks_agg     = tasks_agg_all,
            anp_soc_model = soc_model,
            soc_feat_cols = soc_feat_cols,
            device        = device,
            ctx_cycles    = cfg.ctx_cycles,
            meas_per_cycle= cfg.measurements_per_cycle,
        )
        data["normalized_synth_datasets"] = enriched_synth

        # Enrich real dataset
        real_X_raw, real_y_raw = sort_task_by_cycle(*data_raw["normalized_real_dataset"])
        real_X_agg, real_y_agg = sort_task_by_cycle(*data["normalized_real_dataset"])
        enriched_real = enrich_with_soc_predictions(
            tasks_raw     = [(real_X_raw, real_y_raw)],
            tasks_agg     = [(real_X_agg, real_y_agg)],
            anp_soc_model = soc_model,
            soc_feat_cols = soc_feat_cols,
            device        = device,
            ctx_cycles    = cfg.ctx_cycles,
            meas_per_cycle= cfg.measurements_per_cycle,
        )
        data["normalized_real_dataset"] = enriched_real[0]

        del soc_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"   ✓  Enrichment complete")

    return data


# ==============================================================================
# TRAINING LOOP
# ==============================================================================

def train(cfg: Config) -> tuple:
    """
    Run the full training pipeline for any ANP model variant.

    Steps:
        1. Set random seeds for reproducibility.
        2. Load data; save raw copy before transformations.
        3. Apply pipeline: feature reduction → target filtering →
           cycle aggregation → SoC enrichment.
        4. Detect final input_dim / output_dim from transformed data.
        5. Instantiate ANP model, Adam optimizer, cosine LR scheduler.
        6. Training loop with validation and early stopping.
        7. Final test evaluation and plot generation.

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

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🔧  Device: {device}")
    print(f"📁  Run dir: {run_dir}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\n📂  Loading data from: {cfg.data_dir}")
    data = load_prepared_data(cfg.data_dir)
    validate_targets(data)

    # Save raw reference BEFORE any transformation.
    # data_raw is used by enrich_with_soc_predictions() to access the full
    # 202-feature measurement-level X needed for ANP-SoC inference.
    data_raw = copy.copy(data)

    # ── Apply full data pipeline ──────────────────────────────────────────────
    data = _build_pipeline(data, cfg, device, data_raw)

    # ── Detect final dims from transformed data ───────────────────────────────
    # This must happen AFTER all pipeline steps so input_dim reflects any
    # feature reduction (+aggregation +enrichment) that was applied.
    target_cols    = list(data["normalized_synth_datasets"][0][1].columns)
    cfg.output_dim = len(target_cols)
    cfg.input_dim  = data["normalized_synth_datasets"][0][0].shape[1]

    print(f"\n input_dim  = {cfg.input_dim}")
    print(f" output_dim = {cfg.output_dim}  {target_cols}")
    print(f" ctx_rows   = {cfg.ctx_rows} ({cfg.ctx_cycles} cycles × "
          f"{cfg.measurements_per_cycle} meas/cycle)")
    print(f" tgt_rows   = {cfg.tgt_rows} ({cfg.tgt_cycles} cycles × "
          f"{cfg.measurements_per_cycle} meas/cycle)")

    # ── Task splits ───────────────────────────────────────────────────────────
    train_tasks, val_tasks, test_tasks = get_task_splits(
        data, cfg.train_task_ids, cfg.val_task_ids, cfg.test_task_ids
    )
    denorm_values = {
        "y_mean": data["denorm_values"]["y_mean"],
        "y_std":  data["denorm_values"]["y_std"],
    }

    def presort(tasks: list) -> list:
        return [sort_task_by_cycle(X, y) for X, y in tasks]

    train_sorted = presort(train_tasks)
    val_sorted   = presort(val_tasks)
    test_sorted  = presort(test_tasks)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = LatentModel(
        num_hidden   = cfg.num_hidden,
        input_dim    = cfg.input_dim,    # correct dim after all transformations
        output_dim   = cfg.output_dim,
        attn_dropout = cfg.attn_dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠  ANP model: {n_params:,} trainable parameters")

    # ── Optimizer and scheduler ───────────────────────────────────────────────
    optimizer = Adam(model.parameters(), lr=cfg.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr_min)

    # ── Save config snapshot ──────────────────────────────────────────────────
    # All fields including enrich_soc_predictions and anp_soc_run_dir are saved
    # so evaluate scripts can auto-detect the model configuration via config.json.
    cfg_dict = asdict(cfg)
    cfg_dict.update({
        "target_cols": target_cols,
        "n_params":    n_params,
        "ctx_rows":    cfg.ctx_rows,
        "tgt_rows":    cfg.tgt_rows,
    })
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)

    # ── Early stopping metric ─────────────────────────────────────────────────
    es_key = ("val/mae_Cycle" if cfg.target_col == "Cycle"
               else "val/mae_SoC_pct")

    # ── Training state ────────────────────────────────────────────────────────
    best_val_MAE          = float("inf")
    epochs_without_improve = 0
    metrics_rows           = []
    steps_per_epoch        = max(1, cfg.episodes_per_epoch // cfg.batch_size)

    print(f"\n🚀  Starting training — {cfg.epochs} epochs  |  "
          f"{steps_per_epoch} steps/epoch  |  batch_size={cfg.batch_size}  |  "
          f"ES metric: {es_key}\n")

    pbar = tqdm(range(1, cfg.epochs + 1), desc="Train", unit="epoch",
                dynamic_ncols=True)

    for epoch in pbar:
        model.train()
        ep_losses, ep_nlls, ep_kls = [], [], []

        for _ in range(steps_per_epoch):
            ctx_x, ctx_y, tgt_x, tgt_y = make_batch(
                train_sorted,
                batch_size = cfg.batch_size,
                ctx_rows   = cfg.ctx_rows,
                tgt_rows   = cfg.tgt_rows,
                device     = device,
            )
            optimizer.zero_grad()
            _, _, loss, kl, nll = model(ctx_x, ctx_y, tgt_x, tgt_y, beta=cfg.beta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
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

        # ── Validation ────────────────────────────────────────────────────────
        if epoch % cfg.val_every == 0 or epoch == cfg.epochs:
            val_metrics = evaluate(
                model, val_sorted,
                ctx_rows      = cfg.ctx_rows,
                tgt_rows      = cfg.tgt_rows,
                device        = device,
                denorm_values = denorm_values,
                target_cols   = target_cols,
                beta          = cfg.beta,
                split_name    = "val",
            )
            row.update(val_metrics)
            current_val = val_metrics.get(es_key, float("inf"))

            if current_val < best_val_MAE:
                best_val_MAE          = current_val
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
        postfix: dict = {"loss": f"{train_loss:.3f}"}
        if "val/loss" in row:
            postfix["val_loss"] = f"{row['val/loss']:.3f}"
            if "val/mae_SoC_pct" in row:
                postfix["mae_soc"] = f"{row['val/mae_SoC_pct']:.2f}"
            if "val/mae_Cycle" in row:
                postfix["mae_cyc"] = f"{row['val/mae_Cycle']:.2f}"
            postfix["best"]  = f"{best_val_MAE:.3f}"
            postfix["E_S"]   = f"{epochs_without_improve}/{cfg.early_stopping}"
        pbar.set_postfix(postfix)
        metrics_rows.append(row)

    # ── Save last checkpoint and metrics ──────────────────────────────────────
    torch.save(
        {"epoch": epoch, "model": model.state_dict(),
         "optimizer": optimizer.state_dict(), "cfg": cfg_dict},
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
        ctx_rows      = cfg.ctx_rows,
        tgt_rows      = cfg.tgt_rows,
        device        = device,
        denorm_values = denorm_values,
        target_cols   = target_cols,
        beta          = cfg.beta,
        split_name    = "test",
    )

    print(f"\n  {'─'*52}")
    print(f"  {'Metric':<38} {'Value':>10}")
    print(f"  {'─'*52}")
    for k, v in sorted(test_metrics.items()):
        print(f"  {k:<38} {v:>10.4f}")
    print(f"  {'─'*52}")

    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    generate_all_plots(run_dir, target_cols)
    print(f"\n✅  Training complete. All outputs in: {run_dir}\n")
    return model, test_metrics


# ==============================================================================
# EVAL-ONLY MODE
# ==============================================================================

def eval_only(cfg: Config) -> None:
    """
    Load a saved checkpoint and evaluate it on val and test sets.
    Applies the same data pipeline as train() (feature reduction, aggregation,
    enrichment) so results are directly comparable.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_prepared_data(cfg.data_dir)
    validate_targets(data)

    # Save raw reference before any transformation (needed for enrichment)
    data_raw = copy.copy(data)

    # Apply full pipeline (same as train())
    data = _build_pipeline(data, cfg, device, data_raw)

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

    model = LatentModel(
        num_hidden   = cfg.num_hidden,
        input_dim    = cfg.input_dim,
        output_dim   = cfg.output_dim,
        attn_dropout = cfg.attn_dropout,
    ).to(device)

    ckpt = torch.load(cfg.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"✓ Checkpoint loaded: {cfg.ckpt}  (epoch {ckpt.get('epoch', '?')})")
    print(f"  input_dim={cfg.input_dim}  output_dim={cfg.output_dim}  "
          f"targets={target_cols}")

    for split_name, tasks in [
        ("val",  presort(val_tasks)),
        ("test", presort(test_tasks)),
    ]:
        metrics = evaluate(
            model, tasks,
            ctx_rows      = cfg.ctx_rows,
            tgt_rows      = cfg.tgt_rows,
            device        = device,
            denorm_values = denorm_values,
            target_cols   = target_cols,
            beta          = cfg.beta,
            split_name    = split_name,
        )
        print(f"\n── {split_name.upper()} ─────────────────────────────")
        for k, v in sorted(metrics.items()):
            print(f"   {k:<42} {v:.4f}")


# ==============================================================================
# CLI
# ==============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ANP training — battery SoC and Cycle prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--data_dir",    type=str, default=None)
    p.add_argument("--run_dir",     type=str, default="")
    # Target
    p.add_argument("--target_col",  type=str, default="all",
                   choices=["all", "SoC (%)", "Cycle"])
    # Feature pipeline
    p.add_argument(
        "--reduced_features", action="store_true", dest="use_reduced_features",
        help="Filter X to the RF-identified compact feature set for target_col.",
    )
    p.add_argument(
        "--aggregate_by_cycle", action="store_true",
        help="Average EIS measurements per cycle (1 row/cycle). "
             "Recommended for Cycle models.",
    )
    p.add_argument(
        "--enrich_soc_predictions", action="store_true",
        help="Add per-cycle SoC statistics from a pre-trained ANP-SoC model "
             "as additional features. Requires --aggregate_by_cycle and "
             "--anp_soc_run_dir.",
    )
    p.add_argument(
        "--anp_soc_run_dir", type=str, default="",
        help="Path to a pre-trained ANP-SoC run directory (best.pt + config.json). "
             "Required when --enrich_soc_predictions is set.",
    )
    # Architecture
    p.add_argument("--num_hidden",     type=int,   default=128)
    # Episode construction
    p.add_argument("--ctx_cycles",     type=int,   default=60)
    p.add_argument("--tgt_cycles",     type=int,   default=60)
    p.add_argument("--meas_per_cycle", type=int,   default=30,
                   dest="measurements_per_cycle")
    # Training
    p.add_argument("--epochs",     type=int,   default=1000)
    p.add_argument("--early_stop", type=int,   default=200,
                   dest="early_stopping")
    p.add_argument("--episodes",   type=int,   default=100,
                   dest="episodes_per_epoch")
    p.add_argument("--batch_size", type=int,   default=4)
    p.add_argument("--lr",         type=float, default=5e-4)
    p.add_argument("--lr_min",     type=float, default=8e-5)
    p.add_argument("--attn_dropout", type=float, default=0.2)
    p.add_argument("--beta",       type=float, default=0.5)
    p.add_argument("--grad_clip",  type=float, default=1.0)
    p.add_argument("--seed",       type=int,   default=18)
    p.add_argument("--log_every",  type=int,   default=10)
    p.add_argument("--val_every",  type=int,   default=1)
    # Eval mode
    p.add_argument("--eval_only",  action="store_true")
    p.add_argument("--ckpt",       type=str,   default="")
    # Task split overrides
    p.add_argument("--train_ids",  type=int, nargs="+", default=None)
    p.add_argument("--val_ids",    type=int, nargs="+", default=None)
    p.add_argument("--test_ids",   type=int, nargs="+", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Config(
        data_dir                = args.data_dir or Config.data_dir,
        run_dir                 = args.run_dir,
        target_col              = args.target_col,
        use_reduced_features    = args.use_reduced_features,
        aggregate_by_cycle      = args.aggregate_by_cycle,
        enrich_soc_predictions  = args.enrich_soc_predictions,
        anp_soc_run_dir         = args.anp_soc_run_dir,
        num_hidden              = args.num_hidden,
        ctx_cycles              = args.ctx_cycles,
        tgt_cycles              = args.tgt_cycles,
        measurements_per_cycle  = args.measurements_per_cycle,
        epochs                  = args.epochs,
        early_stopping          = args.early_stopping,
        episodes_per_epoch      = args.episodes_per_epoch,
        batch_size              = args.batch_size,
        lr                      = args.lr,
        lr_min                  = args.lr_min,
        attn_dropout            = args.attn_dropout,
        beta                    = args.beta,
        grad_clip               = args.grad_clip,
        seed                    = args.seed,
        log_every               = args.log_every,
        val_every               = args.val_every,
        eval_only               = args.eval_only,
        ckpt                    = args.ckpt,
    )

    if args.train_ids is not None: cfg.train_task_ids = args.train_ids
    if args.val_ids   is not None: cfg.val_task_ids   = args.val_ids
    if args.test_ids  is not None: cfg.test_task_ids  = args.test_ids

    if cfg.eval_only:
        if not cfg.ckpt:
            raise ValueError("--eval_only requires --ckpt <checkpoint_path>")
        eval_only(cfg)
    else:
        train(cfg)


if __name__ == "__main__":
    main()
