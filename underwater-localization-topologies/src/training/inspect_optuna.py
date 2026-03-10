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
  python inspect_optuna.py \
    --storage "sqlite:////home/fernando/tesis/underwater-localization-topologies/results/optuna_anp.db" \
    --study-name anp_masked_v6 \
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


def _add_time_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Expand user_attrs minutes_per_topology into columns and total minutes."""
    col = "user_attrs_minutes_per_topology"
    if col not in df.columns:
        return df

    def _as_dict(v: Any) -> Dict[str, float]:
        if isinstance(v, dict):
            return {str(k): float(vv) for k, vv in v.items()}
        return {}

    minutes_series = df[col].apply(_as_dict)
    all_keys = sorted({k for d in minutes_series for k in d.keys()})
    for k in all_keys:
        df[f"minutes_{k}"] = minutes_series.apply(lambda d: d.get(k, pd.NA))
    df["minutes_total"] = minutes_series.apply(lambda d: sum(d.values()) if d else pd.NA)
    return df


def _make_mae_by_group_fig(study: optuna.Study, group_param: str) -> Any:
    """Box + strip chart of objective values (MAE) grouped by *group_param*."""
    import plotly.graph_objects as go
    from collections import defaultdict

    groups: Dict[Any, List[float]] = defaultdict(list)
    for t in study.trials:
        if t.state.name == "COMPLETE" and group_param in t.params and t.value is not None:
            groups[t.params[group_param]].append(t.value)

    fig = go.Figure()
    for gv in sorted(groups.keys()):
        vals = groups[gv]
        fig.add_trace(go.Box(
            y=vals,
            name=str(gv),
            boxpoints="all",
            jitter=0.35,
            pointpos=-1.8,
            marker_size=4,
        ))

    fig.update_layout(
        title=f"Objective (MAE) distribution by {group_param}  "
              f"[{len([t for t in study.trials if t.state.name == 'COMPLETE'])} COMPLETE trials]",
        xaxis_title=group_param,
        yaxis_title="MAE (objective value)",
        showlegend=False,
        height=450,
    )
    return fig


def _combine_figs_html(figs_with_titles: List[tuple], output_path: Path) -> None:
    """Write multiple plotly figures to a single self-contained HTML file.

    *figs_with_titles* is a list of (title_str, plotly_figure).
    The first figure bundles plotly.js; the rest reference the same bundle.
    """
    parts: List[str] = []
    for i, (title, fig) in enumerate(figs_with_titles):
        include_js = "cdn" if i == 0 else False
        div = fig.to_html(full_html=False, include_plotlyjs=include_js)
        parts.append(f"<h3 style='font-family:sans-serif;margin-top:28px'>{title}</h3>\n{div}")

    html = (
        "<!DOCTYPE html>\n"
        "<html><head><meta charset='utf-8'></head>\n"
        "<body style='background:#fff'>\n"
        + "\n".join(parts)
        + "\n</body></html>"
    )
    output_path.write_text(html, encoding="utf-8")


def _try_make_plots(
    study: optuna.Study,
    outdir: Path,
    params: Optional[List[str]] = None,
    group_param: Optional[str] = None,
    group_trials: Optional[Dict[Any, List[optuna.trial.FrozenTrial]]] = None,
) -> None:
    """Save a few Optuna interactive plots as HTML (requires plotly).

    When *group_param* and *group_trials* are provided, an additional
    ``slice_<group_param>_<value>.html`` is written for each group so you
    can compare the hyperparameter-response surface across batch sizes.
    """
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

    # Include the group param (e.g. batch_size) in the global slice plot
    slice_params = list(params) if params else []
    if group_param and group_param not in slice_params:
        slice_params.insert(0, group_param)

    if slice_params:
        try:
            slice_fig = plot_slice(study, params=slice_params)
            figs: List[tuple] = [("Slice plot", slice_fig)]
            if group_param:
                try:
                    mae_fig = _make_mae_by_group_fig(study, group_param)
                    figs.append((f"MAE distribution by {group_param}", mae_fig))
                except Exception as e:
                    print(f"[plots] could not build MAE-by-group chart: {e}")
            _combine_figs_html(figs, outdir / "slice.html")
            print("[plots] wrote slice.html")
        except Exception as e:
            print(f"[plots] failed to save slice.html: {e}")
        _save(plot_parallel_coordinate(study, params=slice_params), "parallel_coordinate.html")

    # Per-group slice plots (one per batch_size value)
    if group_param and group_trials:
        per_group_outdir = outdir / f"slice_by_{group_param}"
        per_group_outdir.mkdir(parents=True, exist_ok=True)
        for gv, trials_g in sorted(group_trials.items(), key=lambda x: x[0]):
            if not trials_g:
                continue
            substudy = _make_substudy_from_trials(study, trials_g)
            # params available in this sub-group
            available = {k for t in trials_g for k in t.params.keys()}
            sub_params = [p for p in (params or []) if p in available]
            if not sub_params:
                continue
            fname = f"slice_{group_param}_{gv}.html"
            try:
                fig = plot_slice(substudy, params=sub_params)
                fig.update_layout(
                    title=f"Slice plot — {group_param}={gv} "
                          f"({len(trials_g)} COMPLETE trials)"
                )
                fig.write_html(str(per_group_outdir / fname))
                print(f"[plots] wrote slice_by_{group_param}/{fname}")
            except Exception as e:
                print(f"[plots] failed slice for {group_param}={gv}: {e}")


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
    ap.add_argument("--min-trials-importance", type=int, default=7,
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
    df_all = _add_time_metrics(df_all)
    all_csv = outdir / f"{args.study_name}_trials_all.csv"
    df_all.to_csv(all_csv, index=False)
    print("\n[export] wrote:", all_csv)

    # Complete-only df for analyses
    df = _load_complete_df(study)
    df = _add_time_metrics(df)
    complete_csv = outdir / f"{args.study_name}_trials_complete.csv"
    df.to_csv(complete_csv, index=False)
    print("[export] wrote:", complete_csv)

    print("\nTop-10 COMPLETE trials:")
    top_cols = ["number", "value"]
    if "minutes_total" in df.columns:
        top_cols.append("minutes_total")
    print(df.sort_values("value").head(10)[top_cols].to_string(index=False))

    if "minutes_total" in df.columns:
        print("\nTop-10 slowest COMPLETE trials:")
        print(df.sort_values("minutes_total", ascending=False).head(10)[["number", "value", "minutes_total"]].to_string(index=False))

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

    if "minutes_total" in df.columns:
        time_summary = (
            df.groupby(group_col)["minutes_total"]
            .agg(["count", "min", "median", "mean", "max"])
            .rename(columns={"count": "n_trials"})
            .sort_values("mean")
        )
        time_summary_csv = outdir / f"{args.study_name}_time_summary_by_{args.group_param}.csv"
        time_summary.to_csv(time_summary_csv)
        print(f"\n=== Time summary by {group_col} (minutes_total) ===")
        print(time_summary.to_string())
        print("[export] wrote:", time_summary_csv)

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

        # Build per-group trial lists for per-batch-size slice plots
        group_trials_map: Optional[Dict[Any, List[optuna.trial.FrozenTrial]]] = None
        if group_col in df.columns:
            group_trials_map = {
                gv: [
                    t for t in study.trials
                    if t.state.name == "COMPLETE"
                    and t.params.get(args.group_param, None) == gv
                ]
                for gv in df[group_col].unique()
            }

        _try_make_plots(
            study,
            outdir,
            params=plot_params,
            group_param=args.group_param,
            group_trials=group_trials_map,
        )


if __name__ == "__main__":
    main()
