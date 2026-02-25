#!/usr/bin/env python3
"""
inspect_optuna_v2.py

Post-hoc analysis of an Optuna study stored in an RDB (SQLite).

Adds to your current inspect script:
- Full trials export to CSV
- Summary grouped by a hyperparameter (default: batch_size)
- Top-N trials per group (batch_size buckets)
- Global hyperparameter importances
- Hyperparameter importances conditioned on each group (e.g., per batch size)
- (Optional) interactive plots saved as HTML (requires plotly)

Usage:
  python inspect_optuna_v2.py \
    --storage "sqlite:////home/fernando/tesis/underwater-localization-topologies/results/optuna_anp.db" \
    --study-name anp_masked_v3 \
    --output-dir results/optuna \
    --group-param batch_size \
    --top-n 5 \
    --importance-evaluator fanova \
    --importance-seed 0 \
    --make-plots

Notes:
- Importances are computed using only COMPLETE trials (Optuna requirement).
- FanovaImportanceEvaluator requires scikit-learn. If you don't have it, use PED-ANOVA.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import optuna
import pandas as pd
from optuna.importance import get_param_importances
from optuna.study import StudyDirection


def _study_direction_str(study: optuna.Study) -> str:
    # single-objective study
    if study.direction == StudyDirection.MINIMIZE:
        return "minimize"
    return "maximize"


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _list_studies(storage: str) -> None:
    summaries = optuna.study.get_all_study_summaries(storage=storage)
    for s in summaries:
        best = s.best_trial.value if s.best_trial is not None else None
        print(f"{s.study_name} n_trials={s.n_trials} best={best}")


def _load_complete_df(study: optuna.Study) -> pd.DataFrame:
    df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
    # "state" is string-like in the dataframe
    df = df[df["state"] == "COMPLETE"].copy()
    # Ensure numeric where applicable
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def _group_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    g = df.groupby(group_col)["value"]
    summary = pd.DataFrame({
        "n_trials": g.size(),
        "best": g.min(),
        "median": g.median(),
        "mean": g.mean(),
        "std": g.std(),
        "p10": g.quantile(0.10),
        "p25": g.quantile(0.25),
        "p75": g.quantile(0.75),
        "p90": g.quantile(0.90),
    }).sort_values("best")
    return summary


def _top_trials_per_group(df: pd.DataFrame, group_col: str, top_n: int, cols: List[str]) -> Dict[Any, pd.DataFrame]:
    out: Dict[Any, pd.DataFrame] = {}
    for gv, sub in df.sort_values("value").groupby(group_col):
        # Keep only existing columns to avoid KeyError if some params are absent
        keep_cols = [c for c in cols if c in sub.columns]
        out[gv] = sub[keep_cols].head(top_n).copy()
    return out


def _make_substudy_from_trials(study: optuna.Study, trials: List[optuna.trial.FrozenTrial]) -> optuna.Study:
    s = optuna.create_study(direction=_study_direction_str(study))  # in-memory
    for t in trials:
        s.add_trial(t)
    return s


def _importance_evaluator(name: str, seed: Optional[int]) -> Any:
    name = name.lower().strip()
    if name == "fanova":
        # Requires scikit-learn
        from optuna.importance import FanovaImportanceEvaluator
        return FanovaImportanceEvaluator(seed=seed)
    if name in ("pedanova", "ped-anova", "ped_anova"):
        from optuna.importance import PedAnovaImportanceEvaluator
        return PedAnovaImportanceEvaluator()
    raise ValueError(f"Unknown importance evaluator: {name}. Use 'fanova' or 'pedanova'.")


def _compute_importances(study: optuna.Study, evaluator_name: str, seed: Optional[int], params: Optional[List[str]] = None) -> Dict[str, float]:
    evaluator = _importance_evaluator(evaluator_name, seed)
    imp = get_param_importances(study, evaluator=evaluator, params=params, normalize=True)
    return dict(imp)


def _save_dict_csv(d: Dict[str, float], path: Path, key_name: str = "param", value_name: str = "importance") -> None:
    df = pd.DataFrame([{key_name: k, value_name: v} for k, v in d.items()])
    df.to_csv(path, index=False)


def _try_make_plots(study: optuna.Study, outdir: Path, params: Optional[List[str]] = None) -> None:
    """Save a few Optuna interactive plots as HTML (requires plotly)."""
    try:
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_slice,
            plot_parallel_coordinate,
        )
    except Exception as e:
        print(f"[plots] Skipping plots (optuna.visualization/plotly not available): {e}")
        return

    def _save(fig, name: str):
        try:
            fig.write_html(str(outdir / name))
            print(f"[plots] wrote {name}")
        except Exception as e:
            print(f"[plots] failed to save {name}: {e}")

    _save(plot_optimization_history(study), "opt_history.html")
    _save(plot_param_importances(study), "param_importances.html")

    if params:
        _save(plot_slice(study, params=params), "slice.html")
        _save(plot_parallel_coordinate(study, params=params), "parallel_coordinate.html")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--storage", type=str,
                    default="sqlite:////home/fernando/tesis/underwater-localization-topologies/optuna_anp.db")
    ap.add_argument("--study-name", type=str, default="anp_masked_v2")
    ap.add_argument("--output-dir", type=str, default="results/optuna")

    ap.add_argument("--group-param", type=str, default="batch_size",
                    help="Group trials by this parameter name (without 'params_' prefix)." )
    ap.add_argument("--top-n", type=int, default=5)

    ap.add_argument("--importance-evaluator", type=str, default="fanova",
                    choices=["fanova", "pedanova"],
                    help="Importance evaluator. fanova needs scikit-learn; pedanova is lighter.")
    ap.add_argument("--importance-seed", type=int, default=0,
                    help="Seed for fANOVA (ignored by PED-ANOVA).")
    ap.add_argument("--min-trials-importance", type=int, default=10,
                    help="Minimum COMPLETE trials required to compute per-group importances.")
    ap.add_argument("--make-plots", action="store_true",
                    help="If set, save Optuna interactive plots as HTML (requires plotly).")

    args = ap.parse_args()

    storage = args.storage
    outdir = Path(args.output_dir) / args.study_name
    _safe_mkdir(outdir)

    print("=== Studies in storage ===")
    _list_studies(storage)

    study = optuna.load_study(study_name=args.study_name, storage=storage)
    print("\nSTUDY NAME:", study.study_name)
    print("DIRECTION:", study.direction)
    print("BEST VALUE:", study.best_value)
    print("BEST PARAMS:", json.dumps(study.best_params, indent=2))

    # Export all trials (including PRUNED/FAIL) to CSV
    df_all = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
    all_csv = outdir / f"{args.study_name}_trials_all.csv"
    df_all.to_csv(all_csv, index=False)
    print("\n[export] wrote:", all_csv)

    # Complete-only df for analyses
    df = _load_complete_df(study)
    complete_csv = outdir / f"{args.study_name}_trials_complete.csv"
    df.to_csv(complete_csv, index=False)
    print("[export] wrote:", complete_csv)

    print("\nTop-10 COMPLETE trials:")
    print(df.sort_values("value").head(10)[["number", "value"]].to_string(index=False))

    # -----------------------
    # Grouped analysis
    # -----------------------
    group_col = f"params_{args.group_param}"
    if group_col not in df.columns:
        print(f"\n[warn] group column '{group_col}' not found in trials dataframe.")
        print("       Available param columns:", [c for c in df.columns if c.startswith("params_")])
        return

    summary = _group_summary(df, group_col)
    summary_csv = outdir / f"{args.study_name}_summary_by_{args.group_param}.csv"
    summary.to_csv(summary_csv)
    print(f"\n=== Summary by {group_col} ===")
    print(summary.to_string())
    print("[export] wrote:", summary_csv)

    # Top-N per group
    cols = [
        "number", "value",
        f"params_{args.group_param}",
        "params_lr", "params_num_hidden", "params_weight_decay",
        "params_kl_warmup_epochs", "params_sensor_drop_mode", "params_sensor_drop_p",
        "params_mask_fill",
    ]
    top_per_group = _top_trials_per_group(df, group_col, args.top_n, cols)

    for gv, tdf in top_per_group.items():
        print(f"\n=== {args.group_param}={gv} | top-{args.top_n} ===")
        print(tdf.to_string(index=False))
        tdf.to_csv(outdir / f"{args.study_name}_top{args.top_n}_{args.group_param}_{gv}.csv", index=False)

    # -----------------------
    # Global importances
    # -----------------------
    all_param_names = sorted({k for t in study.trials if t.state.name == "COMPLETE" for k in t.params.keys()})
    try:
        imp_global = _compute_importances(
            study,
            evaluator_name=args.importance_evaluator,
            seed=args.importance_seed,
            params=all_param_names,
        )
        imp_csv = outdir / f"{args.study_name}_param_importances_{args.importance_evaluator}.csv"
        _save_dict_csv(imp_global, imp_csv)
        print("\n=== Global hyperparameter importances ===")
        for k, v in list(imp_global.items())[:10]:
            print(f"{k:20s} {v:.3f}")
        print("[export] wrote:", imp_csv)
    except Exception as e:
        print(f"\n[warn] Could not compute global importances ({args.importance_evaluator}): {e}")

    # -----------------------
    # Conditional importances per group
    # -----------------------
    print(f"\n=== Importances per {args.group_param} bucket ===")
    per_group_rows = []
    for gv, _subdf in df.groupby(group_col):
        trials_g = [
            t for t in study.trials
            if t.state.name == "COMPLETE" and t.params.get(args.group_param, None) == gv
        ]
        if len(trials_g) < args.min_trials_importance:
            print(f"- {args.group_param}={gv}: {len(trials_g)} trials (skip, need >= {args.min_trials_importance})")
            continue

        substudy = _make_substudy_from_trials(study, trials_g)

        try:
            imp_g = _compute_importances(
                substudy,
                evaluator_name=args.importance_evaluator,
                seed=args.importance_seed,
                params=all_param_names,
            )
        except Exception as e:
            print(f"- {args.group_param}={gv}: importance failed: {e}")
            continue

        print(f"\n[{args.group_param}={gv}] top importances:")
        for k, v in list(imp_g.items())[:5]:
            print(f"  {k:20s} {v:.3f}")

        per_csv = outdir / f"{args.study_name}_importances_{args.importance_evaluator}_{args.group_param}_{gv}.csv"
        _save_dict_csv(imp_g, per_csv)

        row = {"group_value": gv, "n_trials": len(trials_g)}
        row.update({f"imp_{k}": v for k, v in imp_g.items()})
        per_group_rows.append(row)

    if per_group_rows:
        wide = pd.DataFrame(per_group_rows).sort_values("group_value")
        wide_csv = outdir / f"{args.study_name}_importances_{args.importance_evaluator}_by_{args.group_param}_WIDE.csv"
        wide.to_csv(wide_csv, index=False)
        print("\n[export] wrote:", wide_csv)

    # -----------------------
    # Optional plots (global)
    # -----------------------
    if args.make_plots:
        candidate_params = ["lr", "num_hidden", "weight_decay", "kl_warmup_epochs", "sensor_drop_p"]
        plot_params = [p for p in candidate_params if p in all_param_names]
        _try_make_plots(study, outdir, params=plot_params)


if __name__ == "__main__":
    main()
