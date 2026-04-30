"""
train_anp.py
─────────────────────────────────────────────────────────────────────────────
Entrenamiento del modelo ANP para predicción de SoC(%) y Cycle en baterías.

Uso:
    python train_anp.py --data_dir <ruta_a_prepared_data> [opciones]

Ejemplos:
    # Entrenamiento completo con defaults
    python train_anp.py --data_dir ./prepared_data

    # Solo validar un checkpoint existente
    python train_anp.py --data_dir ./prepared_data --eval_only --ckpt ./runs/exp1/best.pt

Salidas (en --run_dir, por defecto ./runs/<timestamp>/):
    best.pt          ← checkpoint con menor val_loss
    last.pt          ← checkpoint del último epoch
    metrics.csv      ← loss y NLL/KL por epoch
    config.json      ← configuración del experimento
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import math
import os
import pickle
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR


# ─── Importar modelo ANP ─────────────────────────────────────────────────────
# Ajusta el import según la ubicación de tu anp.py
try:
    from models.anp import LatentModel
except ImportError:
    import sys
    # Add csic/ to PYTHONPATH when this script is run as: python train_anp.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from models.anp import LatentModel


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # ── Datos ─────────────────────────────────────────────────────────────────
    #data_dir: str = "./prepared_data"
    data_dir: str = "../csic_real_synth_load/prepared_data"

    # ── Split de tareas (índices 0-based sobre los 25 datasets sintéticos)
    # El split por defecto sigue la estrategia OOD del profesor:
    #   - Train:   17 datasets centrales (índices 0-16)
    #   - Val OOD: 5 datasets intermedios no vistos (índices 17-21)
    #   - Test OOD: 3 datasets de parámetros extremos (índices 22-24)
    #
    # AJUSTA estos índices cuando tengas el metadata de temperatura/CD/CC
    # para poner los datasets de parámetros extremos en test.
    train_task_ids: List[int] = field(
        default_factory=lambda: list(range(17))           # datasets 1-17
    )
    val_task_ids: List[int] = field(
        default_factory=lambda: list(range(17, 22))       # datasets 18-22
    )
    test_task_ids: List[int] = field(
        default_factory=lambda: list(range(22, 25))       # datasets 23-25
    )

    # ── Modelo ────────────────────────────────────────────────────────────────
    num_hidden: int = 128       # dimensión oculta de todos los encoders
    input_dim: int  = 201       # nº de features X (ajustar si cambia el pkl)
    output_dim: int = 2         # SoC(%) + Cycle

    # ── Episodios ─────────────────────────────────────────────────────────────
    # Fracción de la trayectoria usada como contexto
    ctx_min_frac: float = 0.10  # mínimo 10% de la trayectoria
    ctx_max_frac: float = 0.60  # máximo 60% de la trayectoria

    # Tamaños de contexto fijos para validación (fracción de T)
    val_ctx_fracs: List[float] = field(
        default_factory=lambda: [0.10, 0.30, 0.50]
    )

    # Número máximo de puntos de contexto y target por episodio
    # (submuestreo para limitar memoria si T es muy grande)
    max_context_pts: int  = 512
    max_target_pts:  int  = 1024

    # ── Entrenamiento ─────────────────────────────────────────────────────────
    epochs:          int   = 200
    episodes_per_epoch: int = 100   # episodios por época (batches de 1 tarea)
    lr:              float = 3e-4
    lr_min:          float = 1e-5   # lr mínimo para cosine annealing
    beta:            float = 1.0    # peso del término KL en ELBO
    grad_clip:       float = 1.0
    seed:            int   = 42

    # ── Logging / Guardado ────────────────────────────────────────────────────
    run_dir:         str   = ""     # se auto-genera si está vacío
    log_every:       int   = 10     # imprimir cada N epochs
    val_every:       int   = 10     # validar cada N epochs
    eval_only:       bool  = False
    ckpt:            str   = ""     # checkpoint a cargar (para eval_only)

    def __post_init__(self):
        if not self.run_dir:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = f"./runs/{ts}"


# ══════════════════════════════════════════════════════════════════════════════
# CARGA Y SPLIT DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

def load_prepared_data(data_dir: str):
    data_dir_path = Path(data_dir).expanduser()
    script_dir = Path(__file__).resolve().parent
    csic_root = script_dir.parent

    if data_dir_path.is_absolute():
        candidate_dirs = [data_dir_path]
    else:
        candidate_dirs = [
            (Path.cwd() / data_dir_path),
            (script_dir / data_dir_path),
            (csic_root / data_dir_path),
            (csic_root / "csic_real_synth_load" / "prepared_data"),
        ]

    # Elimina duplicados preservando orden
    seen = set()
    unique_candidate_dirs = []
    for cand in candidate_dirs:
        resolved = cand.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_candidate_dirs.append(resolved)

    pkl_path = None
    for cand in unique_candidate_dirs:
        trial = cand / "prepared_data.pkl"
        if trial.exists():
            pkl_path = trial
            break

    if pkl_path is None:
        tried = "\n".join(f"  - {p / 'prepared_data.pkl'}" for p in unique_candidate_dirs)
        raise FileNotFoundError(
            "No se encontró prepared_data.pkl en ninguna ruta candidata:\n"
            f"{tried}"
        )

    print(f"✓ Cargando PKL: {pkl_path}")
    with pkl_path.open("rb") as f:
        try:
            data = pickle.load(f)
        except AttributeError as e:
            raise RuntimeError(
                "No se pudo deserializar prepared_data.pkl por incompatibilidad de versiones "
                "(normalmente pandas). Recomendación: activa el mismo entorno con el que "
                "se creó el PKL o vuelve a generar el archivo ejecutando load.py."
            ) from e
    return data


def validate_targets(data, expected_targets=("SoC (%)", "Cycle")):
    """Comprueba que el pkl contiene ambos targets."""
    synth = data["normalized_synth_datasets"]
    actual = list(synth[0][1].columns)
    missing = [t for t in expected_targets if t not in actual]
    if missing:
        raise ValueError(
            f"\n{'='*60}\n"
            f"  TARGETS FALTANTES EN EL PKL: {missing}\n"
            f"  Targets actuales: {actual}\n\n"
            f"  SOLUCIÓN: edita load.py → 'targets': ['SoC (%)', 'Cycle']\n"
            f"  y re-ejecuta:  python load.py\n"
            f"{'='*60}\n"
        )
    print(f"✓ Targets verificados: {actual}")


def get_task_splits(data, cfg: Config):
    """Devuelve listas de (X_df, y_df) para cada split."""
    synth = data["normalized_synth_datasets"]
    n = len(synth)

    def get_tasks(ids):
        out = []
        for i in ids:
            if i >= n:
                raise IndexError(
                    f"task_id={i} fuera de rango (hay {n} datasets sintéticos)"
                )
            X, y = synth[i]
            out.append((X, y))
        return out

    train_tasks = get_tasks(cfg.train_task_ids)
    val_tasks   = get_tasks(cfg.val_task_ids)
    test_tasks  = get_tasks(cfg.test_task_ids)

    print(f"✓ Split de tareas:")
    print(f"   Train: {len(train_tasks)} tareas  (ids {cfg.train_task_ids})")
    print(f"   Val:   {len(val_tasks)} tareas  (ids {cfg.val_task_ids})")
    print(f"   Test:  {len(test_tasks)} tareas  (ids {cfg.test_task_ids})")
    return train_tasks, val_tasks, test_tasks


# ══════════════════════════════════════════════════════════════════════════════
# CONSTRUCCIÓN DE EPISODIOS
# ══════════════════════════════════════════════════════════════════════════════

def sort_task_by_cycle(X: pd.DataFrame, y: pd.DataFrame):
    """Ordena las filas por Cycle (y dentro del ciclo por índice original)."""
    if "Cycle" in X.columns:
        order = X["Cycle"].argsort(kind="stable")
        return X.iloc[order].reset_index(drop=True), y.iloc[order].reset_index(drop=True)
    return X.reset_index(drop=True), y.reset_index(drop=True)


def make_episode(
    X: pd.DataFrame,
    y: pd.DataFrame,
    ctx_frac: float,
    max_ctx: int,
    max_tgt: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Construye un episodio ANP a partir de una tarea ordenada temporalmente.

    Contexto  = primeras ctx_frac*T filas (prefijo causal)
    Target    = todas las filas (incluye el contexto — práctica estándar en NPs)

    Aplica submuestreo si ctx/tgt superan max_ctx/max_tgt.

    Returns:
        context_x  (1, Nc, D)
        context_y  (1, Nc, O)
        target_x   (1, Nt, D)
        target_y   (1, Nt, O)
    """
    T = len(X)
    N_c = max(1, int(ctx_frac * T))

    # Submuestreo del contexto si es necesario
    ctx_idx = np.arange(N_c)
    if len(ctx_idx) > max_ctx:
        ctx_idx = np.sort(np.random.choice(ctx_idx, max_ctx, replace=False))

    # Target: todas las filas (submuestreadas si es necesario)
    tgt_idx = np.arange(T)
    if len(tgt_idx) > max_tgt:
        tgt_idx = np.sort(np.random.choice(tgt_idx, max_tgt, replace=False))

    X_vals = torch.tensor(X.values, dtype=torch.float32)
    y_vals = torch.tensor(y.values, dtype=torch.float32)

    ctx_x = X_vals[ctx_idx].unsqueeze(0).to(device)   # (1, Nc, D)
    ctx_y = y_vals[ctx_idx].unsqueeze(0).to(device)   # (1, Nc, O)
    tgt_x = X_vals[tgt_idx].unsqueeze(0).to(device)   # (1, Nt, D)
    tgt_y = y_vals[tgt_idx].unsqueeze(0).to(device)   # (1, Nt, O)

    return ctx_x, ctx_y, tgt_x, tgt_y


# ══════════════════════════════════════════════════════════════════════════════
# MÉTRICAS
# ══════════════════════════════════════════════════════════════════════════════

def denorm_mae(pred_mean, target_y, denorm_values, target_cols):
    """
    Calcula MAE desnormalizado para cada target.
    pred_mean, target_y: tensores (1, Nt, O)
    Devuelve dict {col: mae_value}
    """
    result = {}
    for i, col in enumerate(target_cols):
        mean_val = denorm_values["y_mean"].get(col, 0.0)
        std_val  = denorm_values["y_std"].get(col, 1.0)
        pred_dn = pred_mean[0, :, i].detach().cpu() * std_val + mean_val
        true_dn = target_y[0, :, i].detach().cpu() * std_val + mean_val
        result[col] = (pred_dn - true_dn).abs().mean().item()
    return result


# ══════════════════════════════════════════════════════════════════════════════
# BUCLE DE VALIDACIÓN
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(
    model: nn.Module,
    tasks: list,
    cfg: Config,
    device: torch.device,
    denorm_values: dict,
    target_cols: list,
    split_name: str = "val",
) -> dict:
    """
    Evalúa el modelo sobre un conjunto de tareas con contextos fijos.
    Devuelve métricas promedio por fracción de contexto.
    """
    model.eval()
    results = {}

    for ctx_frac in cfg.val_ctx_fracs:
        losses, nlls, kls = [], [], []
        mae_per_col = {col: [] for col in target_cols}

        for X, y in tasks:
            X_s, y_s = sort_task_by_cycle(X, y)
            ctx_x, ctx_y, tgt_x, tgt_y = make_episode(
                X_s, y_s, ctx_frac,
                cfg.max_context_pts, cfg.max_target_pts, device
            )

            pred_mean, pred_var, loss, kl, nll = model(
                ctx_x, ctx_y, tgt_x, tgt_y, beta=cfg.beta
            )

            losses.append(loss.item())
            nlls.append(nll.item())
            kls.append(kl.item())

            mae = denorm_mae(pred_mean, tgt_y, denorm_values, target_cols)
            for col, val in mae.items():
                mae_per_col[col].append(val)

        key = f"ctx{int(ctx_frac*100):02d}"
        results[f"{split_name}/{key}/loss"] = float(np.mean(losses))
        results[f"{split_name}/{key}/nll"]  = float(np.mean(nlls))
        results[f"{split_name}/{key}/kl"]   = float(np.mean(kls))
        for col in target_cols:
            safe = col.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct")
            results[f"{split_name}/{key}/mae_{safe}"] = float(np.mean(mae_per_col[col]))

    model.train()
    return results


# ══════════════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

def train(cfg: Config):
    # ── Reproducibilidad ──────────────────────────────────────────────────────
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧  Device: {device}")

    # ── Directorio de run ─────────────────────────────────────────────────────
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁  Run dir: {run_dir}")

    # ── Datos ─────────────────────────────────────────────────────────────────
    print(f"\n📂  Cargando datos desde: {cfg.data_dir}")
    data = load_prepared_data(cfg.data_dir)
    validate_targets(data)

    # Columnas de targets (en el orden que vienen del pkl)
    target_cols = list(data["normalized_synth_datasets"][0][1].columns)
    cfg.output_dim = len(target_cols)
    cfg.input_dim  = data["normalized_synth_datasets"][0][0].shape[1]
    print(f"   input_dim  = {cfg.input_dim}")
    print(f"   output_dim = {cfg.output_dim}  {target_cols}")

    train_tasks, val_tasks, test_tasks = get_task_splits(data, cfg)
    denorm_values = {
        "y_mean": data["denorm_values"]["y_mean"],
        "y_std":  data["denorm_values"]["y_std"],
    }

    # Pre-ordenar temporalmente todas las tareas (ordenación única al inicio)
    def presort(tasks):
        return [sort_task_by_cycle(X, y) for X, y in tasks]

    train_sorted = presort(train_tasks)
    val_sorted   = presort(val_tasks)
    test_sorted  = presort(test_tasks)

    # ── Modelo ────────────────────────────────────────────────────────────────
    model = LatentModel(
        num_hidden=cfg.num_hidden,
        input_dim=cfg.input_dim,
        output_dim=cfg.output_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠  Modelo ANP: {n_params:,} parámetros entrenables")

    # ── Optimizador ───────────────────────────────────────────────────────────
    optimizer = Adam(model.parameters(), lr=cfg.lr)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.lr_min
    )

    # ── Guardar configuración ─────────────────────────────────────────────────
    cfg_dict = asdict(cfg)
    cfg_dict["target_cols"] = target_cols
    cfg_dict["n_params"] = n_params
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)

    # ── CSV de métricas ───────────────────────────────────────────────────────
    metrics_path = run_dir / "metrics.csv"
    metrics_rows = []

    # ── Entrenamiento ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    t0 = time.time()

    print(f"\n🚀  Iniciando entrenamiento — {cfg.epochs} epochs\n")

    pbar = tqdm(range(1, cfg.epochs + 1), desc="Training", unit="epoch")
    for epoch in pbar:
        model.train()
        ep_losses, ep_nlls, ep_kls = [], [], []

        for _ in range(cfg.episodes_per_epoch):
            # Samplear una tarea aleatoria del train set
            X, y = random.choice(train_sorted)

            # Fracción de contexto aleatoria por episodio
            ctx_frac = random.uniform(cfg.ctx_min_frac, cfg.ctx_max_frac)

            ctx_x, ctx_y, tgt_x, tgt_y = make_episode(
                X, y, ctx_frac,
                cfg.max_context_pts, cfg.max_target_pts, device
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
            "epoch": epoch,
            "train/loss": train_loss,
            "train/nll":  train_nll,
            "train/kl":   train_kl,
            "lr":         lr_now,
        }

        # ── Validación ────────────────────────────────────────────────────────
        if epoch % cfg.val_every == 0 or epoch == cfg.epochs:
            val_metrics = evaluate(
                model, val_sorted, cfg, device,
                denorm_values, target_cols, split_name="val"
            )
            row.update(val_metrics)

            # Métrica principal para early stopping: val loss con ctx=30%
            main_val_key = "val/ctx30/loss"
            main_val = val_metrics.get(main_val_key, float("inf"))

            if main_val < best_val_loss:
                best_val_loss = main_val
                torch.save(
                    {"epoch": epoch, "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "val_loss": best_val_loss, "cfg": cfg_dict},
                    run_dir / "best.pt"
                )

        # ── Actualizar barra de progreso ───────────────────────────────────────
        postfix_dict = {
            "loss": f"{train_loss:.4f}",
            "nll": f"{train_nll:.4f}",
            "kl": f"{train_kl:.5f}",
            "lr": f"{lr_now:.2e}",
        }
        if "val/ctx30/loss" in row:
            postfix_dict["val_loss"] = f"{row['val/ctx30/loss']:.4f}"
            postfix_dict["mae_soc"] = f"{row.get('val/ctx30/mae_SoC_pct', float('nan')):.4f}"
            postfix_dict["mae_cyc"] = f"{row.get('val/ctx30/mae_Cycle', float('nan')):.4f}"
        
        pbar.set_postfix(postfix_dict)
        metrics_rows.append(row)

    # Guardar último checkpoint
    torch.save(
        {"epoch": cfg.epochs, "model": model.state_dict(),
         "optimizer": optimizer.state_dict(), "cfg": cfg_dict},
        run_dir / "last.pt"
    )

    # Guardar métricas
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    print(f"\n📊  Métricas guardadas en: {metrics_path}")

    # ── Evaluación final en test ──────────────────────────────────────────────
    print(f"\n🏁  Evaluación final en TEST (mejor checkpoint)...")
    best_ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model"])

    test_metrics = evaluate(
        model, test_sorted, cfg, device,
        denorm_values, target_cols, split_name="test"
    )

    print(f"\n  {'─'*55}")
    print(f"  {'Métrica':<40} {'Valor':>10}")
    print(f"  {'─'*55}")
    for k, v in sorted(test_metrics.items()):
        print(f"  {k:<40} {v:>10.4f}")
    print(f"  {'─'*55}")

    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\n✅  Entrenamiento completado. Resultados en: {run_dir}\n")
    return model, test_metrics


# ══════════════════════════════════════════════════════════════════════════════
# SOLO EVALUACIÓN (--eval_only)
# ══════════════════════════════════════════════════════════════════════════════

def eval_only(cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data   = load_prepared_data(cfg.data_dir)
    validate_targets(data)

    target_cols   = list(data["normalized_synth_datasets"][0][1].columns)
    cfg.output_dim = len(target_cols)
    cfg.input_dim  = data["normalized_synth_datasets"][0][0].shape[1]

    _, val_tasks, test_tasks = get_task_splits(data, cfg)
    denorm_values = {"y_mean": data["denorm_values"]["y_mean"],
                     "y_std":  data["denorm_values"]["y_std"]}

    def presort(tasks):
        return [sort_task_by_cycle(X, y) for X, y in tasks]

    model = LatentModel(cfg.num_hidden, cfg.input_dim, cfg.output_dim).to(device)
    ckpt  = torch.load(cfg.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"✓ Checkpoint cargado desde: {cfg.ckpt}  (epoch {ckpt.get('epoch','?')})")

    for split_name, tasks in [("val", presort(val_tasks)), ("test", presort(test_tasks))]:
        metrics = evaluate(model, tasks, cfg, device, denorm_values, target_cols, split_name)
        print(f"\n── {split_name.upper()} ──────────────────────")
        for k, v in sorted(metrics.items()):
            print(f"   {k:<45} {v:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Entrenamiento ANP — Baterías")
    p.add_argument("--data_dir",     type=str,   default=None,
                   help="Ruta al directorio que contiene prepared_data.pkl")
    p.add_argument("--run_dir",      type=str,   default="")
    p.add_argument("--num_hidden",   type=int,   default=128)
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--episodes",     type=int,   default=100,
                   dest="episodes_per_epoch")
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--beta",         type=float, default=1.0)
    p.add_argument("--ctx_min",      type=float, default=0.10,
                   dest="ctx_min_frac")
    p.add_argument("--ctx_max",      type=float, default=0.60,
                   dest="ctx_max_frac")
    p.add_argument("--max_ctx_pts",  type=int,   default=512,
                   dest="max_context_pts")
    p.add_argument("--max_tgt_pts",  type=int,   default=1024,
                   dest="max_target_pts")
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--eval_only",    action="store_true")
    p.add_argument("--ckpt",         type=str,   default="")
    p.add_argument("--log_every",    type=int,   default=10)
    p.add_argument("--val_every",    type=int,   default=10)

    # Split manual (opcional)
    p.add_argument("--train_ids", type=int, nargs="+", default=None,
                   help="Índices 0-based de los datasets de train")
    p.add_argument("--val_ids",   type=int, nargs="+", default=None)
    p.add_argument("--test_ids",  type=int, nargs="+", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir if args.data_dir is not None else Config.data_dir
    cfg  = Config(
        data_dir          = data_dir,
        run_dir           = args.run_dir,
        num_hidden        = args.num_hidden,
        epochs            = args.epochs,
        episodes_per_epoch= args.episodes_per_epoch,
        lr                = args.lr,
        beta              = args.beta,
        ctx_min_frac      = args.ctx_min_frac,
        ctx_max_frac      = args.ctx_max_frac,
        max_context_pts   = args.max_context_pts,
        max_target_pts    = args.max_target_pts,
        grad_clip         = args.grad_clip,
        seed              = args.seed,
        eval_only         = args.eval_only,
        ckpt              = args.ckpt,
        log_every         = args.log_every,
        val_every         = args.val_every,
    )

    # Sobreescribir split si se pasa por argumento
    if args.train_ids is not None: cfg.train_task_ids = args.train_ids
    if args.val_ids   is not None: cfg.val_task_ids   = args.val_ids
    if args.test_ids  is not None: cfg.test_task_ids  = args.test_ids

    if cfg.eval_only:
        if not cfg.ckpt:
            raise ValueError("--eval_only requiere --ckpt <ruta_al_checkpoint>")
        eval_only(cfg)
    else:
        train(cfg)


if __name__ == "__main__":
    main()
