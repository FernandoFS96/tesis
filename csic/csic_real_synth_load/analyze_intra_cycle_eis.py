"""
analyze_intra_cycle_eis_denorm.py
─────────────────────────────────────────────────────────────────────────────
Analiza cuánto varía el EIS dentro de un mismo ciclo, en unidades físicas
reales (ohmios), desnormalizando los datos del pkl con los mismos valores
de media y std usados en el entrenamiento.

Genera:
    - Tabla resumen por dataset (consola)
    - intra_cycle_eis_variability.csv  ← tabla completa exportada

Uso:
    python analyze_intra_cycle_eis.py --data_dir ./prepared_data
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


# ─── Carga y desnormalización (igual que en visualize_data.py) ────────────────

def load_data(data_dir: str):
    pkl = Path(data_dir) / "prepared_data.pkl"
    if not pkl.exists():
        raise FileNotFoundError(f"No se encontró prepared_data.pkl en {data_dir}")
    with pkl.open("rb") as f:
        return pickle.load(f)


def denormalize_df(df: pd.DataFrame, mean: pd.Series, std: pd.Series) -> pd.DataFrame:
    df = df.copy()
    cols = [c for c in df.columns if c in mean.index]
    df[cols] = df[cols] * std[cols] + mean[cols]
    return df


# ─── Análisis ────────────────────────────────────────────────────────────────

EIS_PREFIXES = ["Zre", "Zim", "Zmag", "Phase"]


def analyze_dataset(X_dn: pd.DataFrame, y_dn: pd.DataFrame, ds_idx: int) -> dict:
    """
    Para un dataset desnormalizado, calcula por cada grupo de features EIS:
      - std intra-ciclo media   (cuánto varía el EIS dentro de un ciclo)
      - std inter-ciclo media   (cuánto varía el EIS entre ciclos, usando la media por ciclo)
      - rango intra-ciclo medio (max - min dentro del ciclo, promediado)
      - rango inter-ciclo medio (max - min de las medias por ciclo)
    """
    if "Cycle" not in X_dn.columns:
        raise ValueError("Columna 'Cycle' no encontrada. Regenera el pkl con Cycle como feature.")

    cycle_col = X_dn["Cycle"].values
    n_cycles  = len(np.unique(cycle_col))
    counts    = pd.Series(cycle_col).value_counts()

    row = {
        "dataset":         ds_idx + 1,
        "n_filas":         len(X_dn),
        "n_ciclos":        n_cycles,
        "mediciones/ciclo_min":  counts.min(),
        "mediciones/ciclo_max":  counts.max(),
        "mediciones/ciclo_media": round(counts.mean(), 1),
    }

    for prefix in EIS_PREFIXES:
        cols = [c for c in X_dn.columns if c.startswith(f"{prefix}_")]
        if not cols:
            continue

        # ── Std intra-ciclo ───────────────────────────────────────────────────
        # Para cada ciclo y cada columna, calculamos la std de las mediciones.
        # Promediamos sobre columnas primero (las 50 frecuencias) y luego sobre ciclos.
        intra_std = (
            X_dn[cols + ["Cycle"]]
            .groupby("Cycle")[cols]
            .std()
            .mean(axis=1)      # media sobre las 50 frecuencias para ese ciclo
            .mean()            # media sobre todos los ciclos
        )

        # ── Rango intra-ciclo (max - min dentro del ciclo) ───────────────────
        intra_range = (
            X_dn[cols + ["Cycle"]]
            .groupby("Cycle")[cols]
            .apply(lambda g: (g.max() - g.min()).mean())  # rango medio sobre frecuencias
            .mean()            # media sobre ciclos
        )

        # ── Std inter-ciclo ───────────────────────────────────────────────────
        # Primero calculamos la media de cada columna por ciclo,
        # luego la std de esas medias a lo largo de los ciclos.
        cycle_means = (
            X_dn[cols + ["Cycle"]]
            .groupby("Cycle")[cols]
            .mean()
        )
        inter_std = cycle_means.std().mean()

        # ── Rango inter-ciclo (max - min de las medias por ciclo) ────────────
        inter_range = (cycle_means.max() - cycle_means.min()).mean()

        # ── Ratio: qué fracción de la variación total es intra-ciclo ─────────
        total_std = X_dn[cols].std().mean()
        ratio = intra_std / total_std if total_std > 0 else float("nan")

        row[f"{prefix}_intra_std_ohm"]   = round(intra_std,   5)
        row[f"{prefix}_intra_range_ohm"] = round(intra_range, 5)
        row[f"{prefix}_inter_std_ohm"]   = round(inter_std,   5)
        row[f"{prefix}_inter_range_ohm"] = round(inter_range, 5)
        row[f"{prefix}_ratio_intra_total"] = round(ratio, 4)

    # ── SoC ──────────────────────────────────────────────────────────────────
    soc_cols = [c for c in y_dn.columns if "SoC" in c]
    if soc_cols:
        sc = soc_cols[0]
        soc_with_cycle = y_dn[[sc]].assign(Cycle=cycle_col)
        intra_soc = soc_with_cycle.groupby("Cycle")[sc].std().mean()
        inter_soc = soc_with_cycle.groupby("Cycle")[sc].mean().std()
        intra_soc_range = soc_with_cycle.groupby("Cycle")[sc].apply(lambda g: g.max() - g.min()).mean()
        row["SoC_intra_std_pct"]   = round(intra_soc, 4)
        row["SoC_intra_range_pct"] = round(intra_soc_range, 4)
        row["SoC_inter_std_pct"]   = round(inter_soc, 4)

    return row


def print_summary(rows: list):
    df = pd.DataFrame(rows)

    print("\n" + "=" * 75)
    print("VARIABILIDAD EIS INTRA-CICLO vs INTER-CICLO — UNIDADES FÍSICAS (Ω / %)")
    print("=" * 75)

    # Una tabla compacta por prefijo
    for prefix in EIS_PREFIXES:
        intra_col  = f"{prefix}_intra_std_ohm"
        inter_col  = f"{prefix}_inter_std_ohm"
        range_col  = f"{prefix}_intra_range_ohm"
        ratio_col  = f"{prefix}_ratio_intra_total"

        if intra_col not in df.columns:
            continue

        print(f"\n── {prefix} ─────────────────────────────────────────────────────")
        print(f"   {'Dataset':<10} {'Std intra-ciclo':>17} {'Rango intra-ciclo':>19} "
              f"{'Std inter-ciclo':>17} {'Ratio intra/total':>18}")
        print(f"   {'-'*73}")
        for _, r in df.iterrows():
            print(f"   D{int(r.dataset):<9} {r[intra_col]:>15.5f} Ω  "
                  f"{r[range_col]:>15.5f} Ω  "
                  f"{r[inter_col]:>15.5f} Ω  "
                  f"{r[ratio_col]:>16.1%}")

        # Media sobre todos los datasets
        print(f"   {'─'*73}")
        print(f"   {'MEDIA':<10} {df[intra_col].mean():>15.5f} Ω  "
              f"{df[range_col].mean():>15.5f} Ω  "
              f"{df[inter_col].mean():>15.5f} Ω  "
              f"{df[ratio_col].mean():>16.1%}")

    # SoC
    if "SoC_intra_std_pct" in df.columns:
        print(f"\n── SoC (%) ──────────────────────────────────────────────────────────")
        print(f"   {'Dataset':<10} {'Std intra-ciclo':>17} {'Rango intra-ciclo':>19} "
              f"{'Std inter-ciclo':>17}")
        print(f"   {'-'*65}")
        for _, r in df.iterrows():
            print(f"   D{int(r.dataset):<9} {r['SoC_intra_std_pct']:>17.4f} %  "
                  f"{r['SoC_intra_range_pct']:>17.4f} %  "
                  f"{r['SoC_inter_std_pct']:>17.4f} %")
        print(f"   {'─'*65}")
        print(f"   {'MEDIA':<10} {df['SoC_intra_std_pct'].mean():>17.4f} %  "
              f"{df['SoC_intra_range_pct'].mean():>17.4f} %  "
              f"{df['SoC_inter_std_pct'].mean():>17.4f} %")

    print("\n" + "=" * 75)
    print("INTERPRETACIÓN")
    print("=" * 75)
    print("""
  Std intra-ciclo  : desviación típica del EIS entre las ~30 mediciones
                     de un mismo ciclo (variación debida al estado de carga)
  Rango intra-ciclo: max - min del EIS dentro del ciclo (promediado sobre ciclos)
  Std inter-ciclo  : desviación típica de la MEDIA por ciclo a lo largo
                     de los 1000 ciclos (variación debida al envejecimiento)
  Ratio intra/total: fracción de la variación total que es intra-ciclo

  Si std_intra >> std_inter  → la variación de SoC domina sobre el envejecimiento.
                               Una sola medición por ciclo (mismo SoC) es suficiente.
  Si std_intra ~  std_inter  → ambas dimensiones son relevantes.
                               Conviene mantener varias mediciones por ciclo.
""")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str,
                   default="../csic_real_synth_load/prepared_data")
    p.add_argument("--out_csv", type=str, default="intra_cycle_eis_variability.csv")
    args = p.parse_args()

    print(f"📂  Cargando datos desde: {args.data_dir}")
    data = load_data(args.data_dir)

    synth    = data["normalized_synth_datasets"]
    real_ds  = data["normalized_real_dataset"]
    denorm   = data["denorm_values"]

    X_mean = pd.Series(denorm["X_mean"])
    X_std  = pd.Series(denorm["X_std"])
    y_mean = pd.Series(denorm["y_mean"])
    y_std  = pd.Series(denorm["y_std"])

    print(f"   Desnormalizando {len(synth)} datasets sintéticos...")
    rows = []
    for i, (X_norm, y_norm) in enumerate(synth):
        X_dn = denormalize_df(X_norm, X_mean, X_std)
        y_dn = denormalize_df(y_norm, y_mean, y_std)
        row  = analyze_dataset(X_dn, y_dn, i)
        rows.append(row)
        print(f"   ✓ Dataset {i+1:2d}/{len(synth)}")

    print_summary(rows)

    # Exportar CSV completo
    out_path = Path(args.out_csv)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\n📊  Tabla completa guardada en: {out_path}\n")


if __name__ == "__main__":
    main()