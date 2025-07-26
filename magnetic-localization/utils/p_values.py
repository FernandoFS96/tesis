#!/usr/bin/env python3
"""p_values.py
==============
Script para comparar diferentes métodos en un CSV y generar p-valores / métricas.

Uso rápido
----------
$ python p_values.py ruta/al/archivo.csv [--alpha 0.05] [--higher-is-better]

* **El fichero CSV** debe tener una columna llamada (por ejemplo) `folder` con el nombre del dataset y el resto de columnas son los métodos.
* Por defecto se asume que **menor es mejor** (p. ej. MAE); indique `--higher-is-better` si es al revés.

Salidas
-------
1. Se genera un fichero `<nombre_csv>_stats.csv` **en la misma carpeta donde está el CSV de entrada** (ya no en la carpeta del script).
2. Se muestra un resumen tabulado por consola.
"""

import argparse
import pathlib
import sys
from typing import List

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.stats.multitest as multitest
from tabulate import tabulate

# -----------------------------------------------------------------------------
# Estadística auxiliar
# -----------------------------------------------------------------------------

def friedman_test(
    data_2d: np.ndarray,
    comp_index: int,
    *,
    alpha: float = 0.05,
    higher_is_better: bool = False,
):
    """Friedman + Davenport + post‑hoc (Demsar 2006)."""
    n_methods, n_reps = data_2d.shape

    ranks = np.empty_like(data_2d, dtype=float)
    for k in range(n_reps):
        col = data_2d[:, k]
        ranks[:, k] = (
            stats.rankdata(-col, method="average")
            if higher_is_better
            else stats.rankdata(col, method="average")
        )

    avg_rank = ranks.mean(axis=1)
    chi2 = (
        12
        * n_reps
        / (n_methods * (n_methods + 1))
        * (np.square(avg_rank).sum() - n_methods * (n_methods + 1) ** 2 / 4)
    )
    friedman_p = stats.chi2.sf(chi2, df=n_methods - 1)

    davenport = chi2 * (n_reps - 1) / (n_reps * (n_methods - 1))
    davenport_p = stats.f.sf(davenport, dfn=n_methods - 1, dfd=(n_methods - 1) * (n_reps - 1))

    z = (avg_rank[comp_index] - avg_rank) / np.sqrt(
        n_methods * (n_methods + 1) / (6 * n_reps)
    )
    posthoc_unc = stats.norm.cdf(z)
    _, posthoc_corr, *_ = multitest.multipletests(posthoc_unc, alpha=alpha, method="holm")

    return dict(
        friedman_p=float(friedman_p),
        davenport_p=float(davenport_p),
        posthoc_unc=posthoc_unc,
        posthoc_corr=posthoc_corr,
    )

# -----------------------------------------------------------------------------
# Cálculo principal
# -----------------------------------------------------------------------------

def compute_stats(
    data_2d: np.ndarray,
    methods: List[str],
    *,
    alpha: float,
    higher_is_better: bool,
):
    means = data_2d.mean(axis=1)
    comp_index = int(np.argmax(means) if higher_is_better else np.argmin(means))
    baseline = data_2d[comp_index]

    # Wilcoxon pareado
    wilcoxon_p = []
    for i, row in enumerate(data_2d):
        if i == comp_index:
            wilcoxon_p.append(1.0)
        else:
            alt = "less" if higher_is_better else "greater"
            wilcoxon_p.append(stats.wilcoxon(row, baseline, alternative=alt).pvalue)
    wilcoxon_p = np.array(wilcoxon_p)
    _, wilcoxon_corr, *_ = multitest.multipletests(wilcoxon_p, alpha=alpha, method="holm")

    # Friedman
    fr_res = friedman_test(data_2d, comp_index, alpha=alpha, higher_is_better=higher_is_better)

    rows = []
    for i, m in enumerate(methods):
        rows.append(
            {
                "method": m,
                "mean_metric": means[i],
                "wilcoxon_p": wilcoxon_p[i],
                "wilcoxon_p_corr": wilcoxon_corr[i],
                "friedman_posthoc_p": fr_res["posthoc_unc"][i],
                "friedman_posthoc_p_corr": fr_res["posthoc_corr"][i],
                "is_baseline": i == comp_index,
            }
        )
    df_out = pd.DataFrame(rows)
    df_out["friedman_p"] = fr_res["friedman_p"]
    df_out["davenport_p"] = fr_res["davenport_p"]
    return df_out, comp_index

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args(argv: List[str]):
    p = argparse.ArgumentParser(
        description="Compute paired Wilcoxon & Friedman stats from a CSV matrix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("csv_path", type=pathlib.Path, help="Input CSV")
    p.add_argument("--alpha", type=float, default=0.05, help="Significance level α")
    p.add_argument("--higher-is-better", dest="hib", action="store_true",
                   help="Set if larger metric = better")
    p.add_argument("--output-name", type=str, default=None,
                   help="Stem for the output CSV with p‑values")
    p.add_argument("--latex", action="store_true", help="Print LaTeX table")
    p.add_argument("--transpose", action="store_true",
                   help="Force a 90° transpose before analysis")
    return p.parse_args(argv)


def auto_orient(df: pd.DataFrame, force_T: bool):
    """
    Returns (methods, data_2d ndarray) ready for compute_stats().
    Accepts:
      1) classic: dataset id col + method columns
      2) transposed: method id row‑index + dataset columns
    """
    if force_T:
        df = df.set_index(df.columns[0]).T.reset_index().rename(columns={'index': 'folder'})

    if df.columns[0].lower() in ("folder", "dataset"):
        # classic orientation
        methods = [c for c in df.columns if c.lower() not in ("folder", "dataset")]
        data = df[methods].to_numpy(float).T          # (n_methods, n_datasets)
        return methods, data

    # try transposed layout: first col is *metric*, rows are methods
    if "folder" not in df.columns and "dataset" not in df.columns:
        df_tr = df.set_index(df.columns[0]).T.reset_index().rename(columns={'index': 'folder'})
        methods = [c for c in df_tr.columns if c.lower() != "folder"]
        data = df_tr[methods].to_numpy(float).T
        return methods, data

    raise ValueError("Could not determine CSV orientation; try --transpose.")


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    csv_path = args.csv_path.expanduser().resolve()
    if not csv_path.exists():
        sys.exit(f"ERROR – file not found: {csv_path}")

    df_in = pd.read_csv(csv_path)
    methods, data = auto_orient(df_in, args.transpose)

    stats_df, comp = compute_stats(data, methods,
                                   alpha=args.alpha,
                                   higher_is_better=args.hib)

    out_dir = csv_path.parent
    stem = args.output_name or f"{csv_path.stem}_stats"
    out_path = out_dir / f"{stem}.csv"
    stats_df.to_csv(out_path, index=False)

    print(tabulate(
        stats_df[["method", "mean_metric", "wilcoxon_p_corr", "friedman_posthoc_p_corr",
                  "is_baseline"]],
        headers="keys", floatfmt=".4f", tablefmt="github"
    ))
    print("\nSaved statistics to", out_path)
    print("Baseline method:", stats_df.loc[stats_df.is_baseline, "method"].iloc[0])


if __name__ == "__main__":
    main()