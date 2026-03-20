"""
load_optuna_model.py
====================
Utility to load an ANP or RANP model that was saved by the Optuna HPO search (optuna_anp_search.py).

Directory layout produced by the search script
-----------------------------------------------
<results_dir>/<model>/<version>/<study_name>/best_model/
    hparams.json                          ← hyperparameters sampled by Optuna
    trial_info.json                       ← trial number, objective value, user attrs
    topology_<name>/best_checkpoint.pth.tar           ← model + optimizer state dicts

The loader only needs the ``best_model/`` directory (or a checkpoint file directly).
It reads ``hparams.json`` to reconstruct the architecture and then loads the weights from ``best_checkpoint.pth.tar``.

Typical usage
-------------
**Load the best ANP model for a given topology and use it for inference:**

    from src.utils.load_optuna_model import load_optuna_best_model

    model, hparams, meta = load_optuna_best_model(best_model_dir="src/training/results/optuna/anp/v1/anp_masked_lowvar_ellipsoidal_v1/best_model",
        topology="ellipsoidal",
        model_type="anp",          # "anp" | "ranp" | "auto"
        device="cuda",
    )
    model.eval()

    # model is ready for inference — pass your (context_x, context_y, target_x)
    with torch.no_grad():
        y_mean, y_var, *_ = model(context_x, context_y, target_x)

**Load a RANP model:**

    model, hparams, meta = load_optuna_best_model(best_model_dir="src/training/results/optuna/ranp/v1/ranp_masked_lowvar_ellipsoidal_v1/best_model",
        topology="ellipsoidal",
        model_type="ranp",
        device="cuda",
    )
    model.eval()

    # RANP forward signature is different — pass full sequence + index tensors:
    with torch.no_grad():
        y_mean, y_var, *_ = model(x_seq, # (B, T, Dx+S)
            context_indices,  # (Nc,)
            context_y,        # (B, Nc, 3)
            target_indices,   # (Nt,)
        )

**Access hparams and trial metadata:**

    print(hparams)   # dict — same keys as hparams.json
    print(meta)      # dict — trial_number, value (best MAE), user_attrs, etc.
                     #        None if trial_info.json is absent

**Load directly from a checkpoint path (bypassing the directory convention):**

    from src.utils.load_optuna_model import load_model_from_checkpoint

    model, hparams = load_model_from_checkpoint(checkpoint_path="path/to/best_checkpoint.pth.tar",
        hparams_path="path/to/hparams.json",
        model_type="anp",
        num_sensors=10,
        num_time_points=201,
        output_dim=3,
        device="cpu",
    )

**Resolve and load from study metadata (no manual path building):**

    from src.utils.load_optuna_model import load_optuna_best_model_from_study

    model, hparams, meta = load_optuna_best_model_from_study(
        results_dir="src/training/results/optuna",
        study_name="anp_masked_lowvar_ellipsoidal_v2",
        topology="ellipsoidal",
        model_type="anp",
        device="cuda",
    )

Notes
-----
- ``num_sensors`` and ``num_time_points`` are needed to reconstruct ``input_dim = num_time_points * num_sensors + num_sensors`` (the masked feature dimension that both training scripts produce).
- ``output_dim`` is 3 for the (x, y, z) navigation targets used in this project.
- The function always returns the model in **eval mode**.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_hparams(hparams_path: Path) -> Dict[str, Any]:
    with open(hparams_path, "r") as f:
        return json.load(f)


def _load_trial_info(trial_info_path: Path) -> Optional[Dict[str, Any]]:
    if not trial_info_path.exists():
        return None
    with open(trial_info_path, "r") as f:
        return json.load(f)


def _detect_model_type(hparams: Dict[str, Any]) -> str:
    """Return 'ranp' if RNN-specific keys are present, else 'anp'."""
    return "ranp" if "rnn_type" in hparams else "anp"


def _infer_study_version(study_name: str) -> str:
    m = re.search(r"(v\d+)$", study_name.strip().lower())
    return m.group(1) if m else "vunknown"


def resolve_optuna_best_model_dir(
    results_dir: str | Path,
    study_name: str,
    model_type: str,
    version: Optional[str] = None,
) -> Path:
    """Resolve best_model dir from the current Optuna output layout.

    Layout: <results_dir>/<model>/<version>/<study_name>/best_model
    """
    model = model_type.lower().strip()
    if model not in {"anp", "ranp"}:
        raise ValueError(f"model_type must be 'anp' or 'ranp', got: {model_type}")
    version_tag = version or _infer_study_version(study_name)
    return Path(results_dir) / model / version_tag / study_name / "best_model"


def _build_anp_model(hparams: Dict[str, Any], input_dim: int, output_dim: int) -> nn.Module:
    from src.models.anp import LatentModel
    return LatentModel(num_hidden=hparams["num_hidden"],
        input_dim=input_dim,
        output_dim=output_dim,
    )


def _build_ranp_model(hparams: Dict[str, Any], input_dim: int, output_dim: int) -> nn.Module:
    from src.models.r_anp import LatentModel
    return LatentModel(num_hidden=hparams["num_hidden"],
        input_dim=input_dim,
        output_dim=output_dim,
        rnn_type=hparams.get("rnn_type", "lstm"),
        rnn_layers=hparams.get("rnn_layers", 1),
        rnn_dropout=hparams.get("rnn_dropout", 0.0),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_optuna_best_model(best_model_dir: str | Path,
    topology: str,
    model_type: str = "auto",
    num_sensors: int = 10,
    num_time_points: int = 201,
    output_dim: int = 3,
    device: str | torch.device = "cpu",
    load_optimizer: bool = False,
) -> Tuple[nn.Module, Dict[str, Any], Optional[Dict[str, Any]]]:
    """Load the best Optuna model for a given topology from *best_model_dir*.

    Parameters
    ----------
    best_model_dir:
        Path to the ``best_model/`` directory produced by ``optuna_anp_search.py``.
    topology:
        Topology name (e.g. ``"ellipsoidal"``).  Used to locate ``topology_<name>/best_checkpoint.pth.tar`` inside *best_model_dir*.
    model_type:
        ``"anp"``, ``"ranp"``, or ``"auto"`` (default).  When ``"auto"``, the type is inferred from the keys present in ``hparams.json``.
    num_sensors:
        Number of sensors used during training (default: 10).
    num_time_points:
        Number of time points per sensor (default: 201).
    output_dim:
        Output dimensionality (default: 3 for x/y/z).
    device:
        Target device for the model (``"cpu"``, ``"cuda"``, etc.).
    load_optimizer:
        If ``True``, also return the optimizer state dict in *meta* under the key ``"optimizer_state_dict"``.

    Returns
    -------
    model:
        Instantiated model with loaded weights, set to ``eval()`` mode.
    hparams:
        Dictionary of hyperparameters as stored in ``hparams.json``.
    meta:
        Dictionary from ``trial_info.json`` (trial number, best MAE, etc.), or ``None`` if the file is not present.  If *load_optimizer* is ``True``, also contains ``"optimizer_state_dict"``.
    """
    best_model_dir = Path(best_model_dir)
    hparams_path   = best_model_dir / "hparams.json"
    trial_info_path = best_model_dir / "trial_info.json"
    ckpt_path      = best_model_dir / f"topology_{topology}" / "best_checkpoint.pth.tar"

    if not hparams_path.exists():
        raise FileNotFoundError(f"hparams.json not found in {best_model_dir}")
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Available topology dirs: {[d.name for d in best_model_dir.iterdir() if d.is_dir()]}"
        )

    hparams = _load_hparams(hparams_path)
    meta    = _load_trial_info(trial_info_path)

    return load_model_from_checkpoint(checkpoint_path=ckpt_path,
        hparams=hparams,
        model_type=model_type,
        num_sensors=num_sensors,
        num_time_points=num_time_points,
        output_dim=output_dim,
        device=device,
        load_optimizer=load_optimizer,
        meta=meta,
    )


def load_model_from_checkpoint(checkpoint_path: str | Path,
    hparams: Dict[str, Any] | str | Path,
    model_type: str = "auto",
    num_sensors: int = 10,
    num_time_points: int = 201,
    output_dim: int = 3,
    device: str | torch.device = "cpu",
    load_optimizer: bool = False,
    meta: Optional[Dict[str, Any]] = None,
) -> Tuple[nn.Module, Dict[str, Any], Optional[Dict[str, Any]]]:
    """Load a model directly from a checkpoint file and a hparams dict/path.

    Parameters
    ----------
    checkpoint_path:
        Path to ``best_checkpoint.pth.tar``.
    hparams:
        Either a ``dict`` of hyperparameters, or a path to ``hparams.json``.
    model_type:
        ``"anp"``, ``"ranp"``, or ``"auto"``.
    num_sensors, num_time_points, output_dim, device, load_optimizer:
        Same as :func:`load_optuna_best_model`.
    meta:
        Optional metadata dict to propagate to the return value.

    Returns
    -------
    Same three-tuple as :func:`load_optuna_best_model`.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if not isinstance(hparams, dict):
        hparams = _load_hparams(Path(hparams))

    # input_dim: Dx = P*S, then +S for the appended mask features
    input_dim = num_time_points * num_sensors + num_sensors

    # Resolve model type
    resolved_type = _detect_model_type(hparams) if model_type == "auto" else model_type
    if resolved_type == "anp":
        model = _build_anp_model(hparams, input_dim, output_dim)
    elif resolved_type == "ranp":
        model = _build_ranp_model(hparams, input_dim, output_dim)
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. Use 'anp', 'ranp', or 'auto'.")

    # Load weights
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    if load_optimizer and meta is not None and "optimizer" in ckpt:
        meta = dict(meta) if meta else {}
        meta["optimizer_state_dict"] = ckpt["optimizer"]

    return model, hparams, meta


def load_optuna_best_model_from_study(
    results_dir: str | Path,
    study_name: str,
    topology: str,
    model_type: str,
    version: Optional[str] = None,
    num_sensors: int = 10,
    num_time_points: int = 201,
    output_dim: int = 3,
    device: str | torch.device = "cpu",
    load_optimizer: bool = False,
) -> Tuple[nn.Module, Dict[str, Any], Optional[Dict[str, Any]]]:
    """Convenience wrapper to load best model using results root + study name."""
    best_model_dir = resolve_optuna_best_model_dir(
        results_dir=results_dir,
        study_name=study_name,
        model_type=model_type,
        version=version,
    )
    return load_optuna_best_model(
        best_model_dir=best_model_dir,
        topology=topology,
        model_type=model_type,
        num_sensors=num_sensors,
        num_time_points=num_time_points,
        output_dim=output_dim,
        device=device,
        load_optimizer=load_optimizer,
    )
