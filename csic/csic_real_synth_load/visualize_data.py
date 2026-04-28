"""
visualize_data.py
─────────────────────────────────────────────────────────────────────────────
Visualización exploratoria de los datos pre-generados para el experimento
ANP / RANP sobre baterías.

Uso:
    python visualize_data.py --data_dir ./prepared_data


Plots generados (guardados en <data_dir>/plots/):
    1.  soc_trajectories.png        — Trayectorias SoC(%) por ciclo (sint. + real)
    2.  soc_distribution.png        — Distribución de SoC: sintético vs real
    3.  dataset_lengths.png         — Número de ciclos por dataset sintético
    4.  nyquist.png                 — Gráficos de Nyquist (Zre vs -Zim) muestreados
    5.  eis_heatmap_synth.png       — Heatmap Zre a lo largo de ciclos (1 dataset sintético)
    6.  eis_heatmap_real.png        — Heatmap Zre a lo largo de ciclos (dataset real)
    7.  feature_correlation.png     — Correlación features vs SoC (top-20)
    8.  potential_vs_soc.png        — Potential vs SoC coloreado por dataset
    9.  soc_boxplot_per_dataset.png — Boxplot de SoC por cada dataset sintético
    10. pca_features.png            — PCA 2D de las features X (sint. vs real)
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─── Paleta ──────────────────────────────────────────────────────────────────
SYNTH_CMAP   = "tab20"
REAL_COLOR   = "#e63946"
SYNTH_ALPHA  = 0.55
FIG_DPI      = 150

# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_data(data_dir: Path):
    pkl = data_dir / "prepared_data.pkl"
    if not pkl.exists():
        raise FileNotFoundError(f"No se encontró prepared_data.pkl en {data_dir}")
    with pkl.open("rb") as f:
        return pickle.load(f)


def get_cycle_column(X: pd.DataFrame):
    """Devuelve los valores de ciclo si están en X, sino None."""
    for col in ("Cycle", "cycle"):
        if col in X.columns:
            return X[col].values
    return None


def savefig(fig, path: Path, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}")


# ─── Plot 1: Trayectorias SoC ─────────────────────────────────────────────────

def plot_soc_trajectories(synth_datasets, real_dataset, out_dir):
    fig, ax = plt.subplots(figsize=(12, 5))
    cmap = cm.get_cmap(SYNTH_CMAP, len(synth_datasets))

    for i, (X, y) in enumerate(synth_datasets):
        soc = y.iloc[:, 0].values          # primera (y única) columna: SoC(%)
        cycles = get_cycle_column(X)
        x_axis = cycles if cycles is not None else np.arange(len(soc))
        ax.plot(x_axis, soc, color=cmap(i), alpha=SYNTH_ALPHA, linewidth=0.9,
                label=f"Synth {i+1}" if i < 5 else "_nolegend_")

    X_r, y_r = real_dataset
    soc_r = y_r.iloc[:, 0].values
    cycles_r = get_cycle_column(X_r)
    x_r = cycles_r if cycles_r is not None else np.arange(len(soc_r))
    ax.plot(x_r, soc_r, color=REAL_COLOR, linewidth=2.0, zorder=5, label="Real")

    ax.set_xlabel("Ciclo (normalizado)" if get_cycle_column(X) is None else "Ciclo")
    ax.set_ylabel("SoC (%) — normalizado")
    ax.set_title("Trayectorias SoC por ciclo: sintético (gris) vs real (rojo)")
    handles, labels = ax.get_legend_handles_labels()
    # Solo primeras 5 synth + real en leyenda
    ax.legend(handles, labels, fontsize=7, ncol=3)
    savefig(fig, out_dir / "soc_trajectories.png")


# ─── Plot 2: Distribución SoC ────────────────────────────────────────────────

def plot_soc_distribution(synth_datasets, real_dataset, out_dir):
    all_synth_soc = np.concatenate([y.iloc[:, 0].values for _, y in synth_datasets])
    real_soc = real_dataset[1].iloc[:, 0].values

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = 60
    ax.hist(all_synth_soc, bins=bins, density=True, alpha=0.65,
            color="#457b9d", label=f"Sintético (N={len(all_synth_soc):,})")
    ax.hist(real_soc, bins=bins, density=True, alpha=0.75,
            color=REAL_COLOR, label=f"Real (N={len(real_soc):,})")
    ax.set_xlabel("SoC (%) — normalizado")
    ax.set_ylabel("Densidad")
    ax.set_title("Distribución de SoC: sintético vs real")
    ax.legend()
    savefig(fig, out_dir / "soc_distribution.png")


# ─── Plot 3: Longitudes de datasets ──────────────────────────────────────────

def plot_dataset_lengths(synth_datasets, out_dir):
    lengths = [len(X) for X, _ in synth_datasets]
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = cm.get_cmap(SYNTH_CMAP, len(lengths))(np.linspace(0, 1, len(lengths)))
    bars = ax.bar(range(1, len(lengths) + 1), lengths, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(np.mean(lengths), color="black", linestyle="--", linewidth=1.2,
               label=f"Media = {np.mean(lengths):.0f}")
    ax.set_xlabel("Dataset sintético")
    ax.set_ylabel("Número de ciclos")
    ax.set_title("Longitud (ciclos) de cada dataset sintético")
    ax.legend()
    # Anotar min/max
    ax.annotate(f"min={min(lengths)}", xy=(np.argmin(lengths)+1, min(lengths)),
                xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    ax.annotate(f"max={max(lengths)}", xy=(np.argmax(lengths)+1, max(lengths)),
                xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    savefig(fig, out_dir / "dataset_lengths.png")


# ─── Plot 4: Nyquist ─────────────────────────────────────────────────────────

def plot_nyquist(synth_datasets, real_dataset, out_dir, n_samples=5):
    """Dibuja gráficos de Nyquist (Zre vs -Zim) para n_samples ciclos aleatorios."""
    def extract_eis(X_df):
        zre_cols  = [f"Zre_{i}"  for i in range(50) if f"Zre_{i}"  in X_df.columns]
        zim_cols  = [f"Zim_{i}"  for i in range(50) if f"Zim_{i}"  in X_df.columns]
        return X_df[zre_cols].values, X_df[zim_cols].values

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Sintético: tomar el primer dataset, n_samples ciclos
    X_s, _ = synth_datasets[0]
    zre_s, zim_s = extract_eis(X_s)
    if zre_s.shape[1] == 0:
        axes[0].text(0.5, 0.5, "Columnas Zre/Zim no encontradas en X",
                     ha="center", transform=axes[0].transAxes)
    else:
        idx = np.random.choice(len(zre_s), min(n_samples, len(zre_s)), replace=False)
        cmap_nyq = cm.get_cmap("viridis", len(idx))
        for k, i in enumerate(sorted(idx)):
            axes[0].plot(zre_s[i], -zim_s[i], "o-", markersize=3,
                         color=cmap_nyq(k), alpha=0.8, label=f"Ciclo {i}")
        axes[0].set_xlabel("Z$_{re}$ (normalizado)")
        axes[0].set_ylabel("-Z$_{im}$ (normalizado)")
        axes[0].set_title(f"Nyquist — Sintético dataset 1 ({n_samples} ciclos)")
        axes[0].legend(fontsize=7)

    # Real
    X_r, _ = real_dataset
    zre_r, zim_r = extract_eis(X_r)
    if zre_r.shape[1] == 0:
        axes[1].text(0.5, 0.5, "Columnas Zre/Zim no encontradas", ha="center",
                     transform=axes[1].transAxes)
    else:
        idx_r = np.random.choice(len(zre_r), min(n_samples, len(zre_r)), replace=False)
        cmap_r = cm.get_cmap("plasma", len(idx_r))
        for k, i in enumerate(sorted(idx_r)):
            axes[1].plot(zre_r[i], -zim_r[i], "o-", markersize=3,
                         color=cmap_r(k), alpha=0.8, label=f"Ciclo {i}")
        axes[1].set_xlabel("Z$_{re}$ (normalizado)")
        axes[1].set_ylabel("-Z$_{im}$ (normalizado)")
        axes[1].set_title(f"Nyquist — Real ({n_samples} ciclos)")
        axes[1].legend(fontsize=7)

    savefig(fig, out_dir / "nyquist.png")


# ─── Plot 5 & 6: Heatmap EIS (Zre) a lo largo de ciclos ─────────────────────

def plot_eis_heatmap(X_df, title, out_path):
    zre_cols = [f"Zre_{i}" for i in range(50) if f"Zre_{i}" in X_df.columns]
    if not zre_cols:
        print(f"  ⚠  Heatmap omitido (no hay cols Zre): {out_path.name}")
        return
    Z = X_df[zre_cols].values.T   # (50 frecuencias, T ciclos)
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(Z, aspect="auto", cmap="RdYlBu_r", origin="lower")
    plt.colorbar(im, ax=ax, label="Zre (normalizado)")
    ax.set_xlabel("Ciclo (índice)")
    ax.set_ylabel("Frecuencia EIS (índice)")
    ax.set_title(title)
    savefig(fig, out_path)


# ─── Plot 7: Correlación features → SoC ─────────────────────────────────────

def plot_feature_correlation(synth_datasets, out_dir, top_n=20):
    X_all = pd.concat([X for X, _ in synth_datasets], axis=0)
    y_all = pd.concat([y for _, y in synth_datasets], axis=0)
    soc = y_all.iloc[:, 0]

    corr = X_all.corrwith(soc).abs().sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = cm.get_cmap("coolwarm")(np.linspace(0.2, 0.8, len(corr)))
    ax.barh(corr.index[::-1], corr.values[::-1], color=colors[::-1])
    ax.set_xlabel("|Correlación de Pearson| con SoC(%)")
    ax.set_title(f"Top-{top_n} features más correladas con SoC (datos sintéticos)")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, label="|r|=0.5")
    ax.legend(fontsize=8)
    savefig(fig, out_dir / "feature_correlation.png")


# ─── Plot 8: Potential vs SoC ────────────────────────────────────────────────

def plot_potential_vs_soc(synth_datasets, real_dataset, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = cm.get_cmap(SYNTH_CMAP, len(synth_datasets))

    if "Potential" not in synth_datasets[0][0].columns:
        print("  ⚠  Columna 'Potential' no encontrada, omitiendo plot 8.")
        plt.close(fig)
        return

    for i, (X, y) in enumerate(synth_datasets):
        ax.scatter(X["Potential"].values, y.iloc[:, 0].values,
                   s=4, alpha=0.3, color=cmap(i),
                   label=f"Synth {i+1}" if i < 5 else "_nolegend_")

    X_r, y_r = real_dataset
    ax.scatter(X_r["Potential"].values, y_r.iloc[:, 0].values,
               s=10, alpha=0.8, color=REAL_COLOR, zorder=5, label="Real")

    ax.set_xlabel("Potential (normalizado)")
    ax.set_ylabel("SoC (%) — normalizado")
    ax.set_title("Potential vs SoC por dataset")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=7, markerscale=2)
    savefig(fig, out_dir / "potential_vs_soc.png")


# ─── Plot 9: Boxplot SoC por dataset sintético ───────────────────────────────

def plot_soc_boxplot(synth_datasets, real_dataset, out_dir):
    data_to_plot = [y.iloc[:, 0].values for _, y in synth_datasets]
    labels = [str(i + 1) for i in range(len(synth_datasets))]

    fig, ax = plt.subplots(figsize=(14, 5))
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.5),
                    whiskerprops=dict(linewidth=0.8),
                    flierprops=dict(marker=".", markersize=2, alpha=0.4))

    cmap = cm.get_cmap(SYNTH_CMAP, len(synth_datasets))
    for patch, color in zip(bp["boxes"], [cmap(i) for i in range(len(synth_datasets))]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Línea horizontal con media del real
    real_mean = real_dataset[1].iloc[:, 0].mean()
    ax.axhline(real_mean, color=REAL_COLOR, linestyle="--", linewidth=1.5,
               label=f"Media SoC real = {real_mean:.2f}")
    ax.set_xlabel("Dataset sintético")
    ax.set_ylabel("SoC (%) — normalizado")
    ax.set_title("Distribución de SoC por dataset sintético")
    ax.legend(fontsize=9)
    savefig(fig, out_dir / "soc_boxplot_per_dataset.png")


# ─── Plot 10: PCA 2D features ────────────────────────────────────────────────

def plot_pca(synth_datasets, real_dataset, out_dir, max_points=3000):
    X_synth = pd.concat([X for X, _ in synth_datasets], axis=0)
    X_real  = real_dataset[0]

    # Submuestrear si hay demasiados puntos
    if len(X_synth) > max_points:
        X_synth = X_synth.sample(max_points, random_state=42)

    X_combined = pd.concat([X_synth, X_real], axis=0).fillna(0)
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_combined.values)

    n_s = len(X_synth)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(components[:n_s, 0], components[:n_s, 1],
               s=8, alpha=0.35, color="#457b9d", label=f"Sintético (N={n_s:,})")
    ax.scatter(components[n_s:, 0], components[n_s:, 1],
               s=20, alpha=0.8, color=REAL_COLOR, label=f"Real (N={len(X_real):,})", zorder=5)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title("PCA 2D de features X: sintético vs real")
    ax.legend(fontsize=9)
    savefig(fig, out_dir / "pca_features.png")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualización de datos preparados ANP/RANP")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directorio con prepared_data.pkl (salida de load.py)")
    parser.add_argument("--seed", type=int, default=0, help="Semilla aleatoria")
    args = parser.parse_args()

    np.random.seed(args.seed)
    data_dir = Path(args.data_dir)
    out_dir  = data_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📂  Cargando datos desde: {data_dir}")
    data = load_data(data_dir)

    synth_datasets = data["normalized_synth_datasets"]   # lista de (X, y) DataFrames
    real_dataset   = data["normalized_real_dataset"]     # (X, y) DataFrames

    print(f"   Datasets sintéticos : {len(synth_datasets)}")
    print(f"   Muestras reales     : {len(real_dataset[0])}")
    print(f"   Nº features (X)     : {real_dataset[0].shape[1]}")
    print(f"   Targets (y)         : {list(real_dataset[1].columns)}")
    print(f"\n🎨  Generando plots en: {out_dir}\n")

    print("Plot 1/10 — Trayectorias SoC")
    plot_soc_trajectories(synth_datasets, real_dataset, out_dir)

    print("Plot 2/10 — Distribución SoC")
    plot_soc_distribution(synth_datasets, real_dataset, out_dir)

    print("Plot 3/10 — Longitudes de datasets")
    plot_dataset_lengths(synth_datasets, out_dir)

    print("Plot 4/10 — Nyquist")
    plot_nyquist(synth_datasets, real_dataset, out_dir, n_samples=8)

    print("Plot 5/10 — Heatmap EIS sintético")
    X_s0, _ = synth_datasets[0]
    plot_eis_heatmap(X_s0, "Heatmap Zre a lo largo de ciclos — Sintético dataset 1",
                     out_dir / "eis_heatmap_synth.png")

    print("Plot 6/10 — Heatmap EIS real")
    X_r, _ = real_dataset
    plot_eis_heatmap(X_r, "Heatmap Zre a lo largo de ciclos — Dataset Real",
                     out_dir / "eis_heatmap_real.png")

    print("Plot 7/10 — Correlación features → SoC")
    plot_feature_correlation(synth_datasets, out_dir)

    print("Plot 8/10 — Potential vs SoC")
    plot_potential_vs_soc(synth_datasets, real_dataset, out_dir)

    print("Plot 9/10 — Boxplot SoC por dataset")
    plot_soc_boxplot(synth_datasets, real_dataset, out_dir)

    print("Plot 10/10 — PCA 2D features")
    plot_pca(synth_datasets, real_dataset, out_dir)

    print(f"\n✅  Todos los plots guardados en: {out_dir}\n")


if __name__ == "__main__":
    main()