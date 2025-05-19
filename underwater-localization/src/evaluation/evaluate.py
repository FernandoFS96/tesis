import pickle
import random
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch as T
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.models.anp import LatentModel
from src.models.mlp import MLP
from src.utils.nav_dataset import NavigationTrajectoryDataset

# Default context percentages to evaluate ANP
CONTEXTS = [2, 4, 6, 8, 10, 15, 25, 40, 60, 80]


class Evaluator:
    def __init__(
        self,
        data_path: Path,
        theta_path: Path,
        anp_ckpt: Path,
        mlp_dir: Path,
        result_dir: Path,
        batch_size: int = 1
    ):
        self.data, self.theta_groups, self.theta_values = self._load_data(data_path, theta_path)
        self.anp = self._load_anp(anp_ckpt)
        self.mlps = self._load_mlps(mlp_dir)
        self.result_dir = result_dir
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.errors: List[Dict[str, List[float]]] = []

        self.traj_seed = random.randint(0, 100)

    def _load_data(self, data_path: Path, theta_path: Path):
        with data_path.open('rb') as f:
            data = pickle.load(f)
        with theta_path.open('rb') as f:
            theta_dict = pickle.load(f)
        val_thetas: List[float] = theta_dict.get('val_thetas', [])

        groups: Dict[float, List[Any]] = {}
        for sample, theta in zip(data, val_thetas):
            groups.setdefault(theta, []).append(sample)

        theta_values = sorted(groups.keys())
        theta_groups = [groups[t] for t in theta_values]
        return data, theta_groups, theta_values

    def _load_anp(self, ckpt: Path) -> T.nn.Module:
        model = LatentModel(num_hidden=128, input_dim=self.data[0][0].shape[-1], output_dim=self.data[0][1].shape[-1])
        model = model.cuda().eval()
        state = T.load(ckpt)
        model_state = state.get('model', state)
        model.load_state_dict(model_state)
        return model

    def _load_mlps(self, mlp_dir: Path) -> Dict[str, T.nn.Module]:
        mlps: Dict[str, T.nn.Module] = {}
        entries: List[tuple[float, Path]] = []
        for d in mlp_dir.iterdir():
            if d.is_dir() and d.name.startswith('MLP_model_channel_option_'):
                try:
                    val = float(d.name.split('_')[-1])
                    entries.append((val, d))
                except ValueError:
                    continue
        # sort by numeric channel
        for val, d in sorted(entries, key=lambda x: x[0]):
            key = f"MLP({val:.1f})"
            ckpt = d / 'best_model.pth'
            if ckpt.exists():
                model = MLP(
                    input_dim=self.data[0][0].shape[-1],
                    output_dim=self.data[0][1].shape[-1]
                )
                model.load_state_dict(T.load(ckpt))
                mlps[key] = model.cuda().eval()
        # combined
        combined = mlp_dir / 'combined_model.pth'
        if combined.exists():
            model = MLP(
                input_dim=self.data[0][0].shape[-1],
                output_dim=self.data[0][1].shape[-1]
            )
            model.load_state_dict(T.load(combined))
            mlps['DRS'] = model.cuda().eval()
        return mlps

    def _compute_errors(self):
        # prepare error dicts for each channel
        self.errors = []
        model_names = list(self.mlps.keys()) + [f"ANP({c}%)" for c in CONTEXTS]
        for group in tqdm(self.theta_groups, desc="Channels"):
            ds = NavigationTrajectoryDataset(group)
            loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
            errs = {name: [] for name in model_names}
            with T.no_grad():
                for x, y in loader:
                    x, y = x.cuda(), y.cuda()
                    # MLP errors
                    for name, m in self.mlps.items():
                        pred = m(x)
                        mae = T.mean((pred - y).abs(), dim=[1, 2])
                        errs[name].extend(mae.cpu().tolist())
                    # ANP errors for each context
                    total = x.size(1)
                    for c in CONTEXTS:
                        n = int((c / 100) * total)
                        c_ind = T.arange(n)
                        t_ind = T.arange(total)
                        cx, cy = x[:, c_ind, :], y[:, c_ind, :]
                        tx, ty = x[:, t_ind, :], y[:, t_ind, :]
                        pred, *_ = self.anp(cx, cy, tx)
                        mae = T.mean((pred - ty).abs(), dim=[1, 2])
                        errs[f"ANP({c}%)"].extend(mae.cpu().tolist())
            self.errors.append(errs)
        return self.errors

    def eval_anp_mean(self) -> Dict[int, List[float]]:
        errs = self._compute_errors()
        return {c: [np.mean(e[f"ANP({c}%)"]) for e in errs] for c in CONTEXTS}

    def eval_mlp_mean(self) -> Dict[str, List[float]]:
        errs = self._compute_errors()
        return {name: [np.mean(e[name]) for e in errs] for name in self.mlps.keys()}

    def eval_heatmap(self) -> tuple[Dict[str, List[float]], Dict[int, List[float]]]:
        errs = self._compute_errors()
        mlp_maes = {name: [np.mean(e[name]) for e in errs] for name in self.mlps.keys()}
        anp_maes = {c: [np.mean(e[f"ANP({c}%)"]) for e in errs] for c in CONTEXTS}
        return mlp_maes, anp_maes

    def eval_pvals(self) -> List[Dict[str, Any]]:
        errs = self._compute_errors()
        significance: List[Dict[str, Any]] = []
        alpha = 0.05
        for e in errs:
            means = {m: np.mean(vals) for m, vals in e.items()}
            ref = min(means, key=means.get)
            ref_err = e[ref]
            others, pvals = [], []
            for m, vals in e.items():
                if m == ref:
                    continue
                _, p = wilcoxon(ref_err, vals)
                others.append(m)
                pvals.append(p)
            rej, p_adj, *_ = multipletests(pvals, alpha=alpha, method='holm')
            sig = {ref: {'significant': False}}
            for m, p0, p1, r in zip(others, pvals, p_adj, rej):
                sig[m] = {'p_orig': float(p0), 'p_adj': float(p1), 'significant': bool(r)}
            significance.append(sig)
        return significance

    def plot_mean(self, anp_means: Dict[int, List[float]]):
        # compute the mean MAE at each context size
        means = [np.mean(anp_means[c]) for c in CONTEXTS]

        # plot the mean MAE vs context size
        # create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(CONTEXTS, means, marker='o', linestyle='-')
        # set the title and labels
        ax.set_title('Mean ANP MAE vs Context Size', fontsize=20)
        ax.set_xlabel('Context (%)', fontsize=19)
        ax.set_ylabel('Mean MAE', fontsize=19)
        # set the x-ticks to be the context sizes
        ax.set_xticks(CONTEXTS)
        ax.set_xticklabels([str(c) for c in CONTEXTS], fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=18)

        # tighten the x-axis so you’re not cutting off your end‐points
        ax.set_xlim(min(CONTEXTS) - 1, max(CONTEXTS) + 1)

        # draw a light vertical line at each tick if you like
        ax.grid(which='both', linestyle='--', alpha=0.5)
        
        fig.tight_layout()
        fig.savefig(self.result_dir / 'mean_mae.png')
        plt.close(fig)

    def plot_heatmap(self, mlp_maes: Dict[str, List[float]], anp_maes: Dict[int, List[float]]):
        rows = [mlp_maes[k] for k in mlp_maes] + [anp_maes[c] for c in CONTEXTS]
        labels = list(mlp_maes.keys()) + [f"ANP({c}%)" for c in CONTEXTS]
        data = np.array(rows)
        plt.figure(figsize=(14, 10))
        sns.heatmap(data, annot=True, fmt='.1f', cmap = 'viridis', xticklabels=self.theta_values, yticklabels=labels, annot_kws={"size":15})
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15, rotation=0) 
        plt.title('MAE: MLP vs ANP', fontsize=18)
        plt.xlabel('Theta', fontsize=17)
        plt.ylabel('Model / Context', fontsize=17)
        plt.tight_layout()
        plt.savefig(self.result_dir / 'heatmap.png')
        plt.close()


    def plot_heatmap_p(self, mlp_maes, anp_maes, sig_list):
        # --- tu preparación de datos/labels/annot como antes ---
        rows  = [mlp_maes[k] for k in mlp_maes] + [anp_maes[c] for c in CONTEXTS]
        labels = list(mlp_maes.keys()) + [f"ANP({c}%)" for c in CONTEXTS]
        data = np.array(rows)
        annot = [
            [f"{data[i,j]:.1f}{('★' if not sig_list[j].get(labels[i],{}).get('significant',False) else '')}"
             for j in range(data.shape[1])]
            for i in range(data.shape[0])
        ]

        # --- dibujamos el heatmap y guardamos el eje ---
        plt.figure(figsize=(14, 10))
        ax = sns.heatmap(
            data, annot=annot, fmt="", cmap="viridis",
            xticklabels=self.theta_values, yticklabels=labels,
            annot_kws={"size":20}, cbar=False
        )

        # --- resaltamos con un rectángulo rojo la celda óptima de cada columna ---
        col_mins = np.argmin(data, axis=0)   # para cada theta (columna), índice de la fila mínima
        for j, i in enumerate(col_mins):
            # (j, i) es la esquina superior izquierda en coordenadas de celda
            rect = patches.Rectangle(
                (j, i), 1, 1,
                fill=False, edgecolor="red", linewidth=3
            )
            ax.add_patch(rect)

        # estilo final
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20, rotation=0)
        plt.title('MAE con p-ajustadas (★=p>0.05)', fontsize=24)
        plt.xlabel('Theta', fontsize=22)
        plt.ylabel('Modelo / Contexto', fontsize=22)
        plt.tight_layout()
        plt.savefig(self.result_dir / 'heatmap_pvals_highlight.png')
        plt.close()

    def plot_trajs(self, num_samples: int = 2, context_frac: float = 0.4, theta_idx=1):
        """
        Plot a few example trajectories comparing ANP and the combined MLP (DRS).
        - num_samples: cuántas trayectorias mostrar (máximo 4).
        - context_frac: fracción de puntos usados como contexto.
        """
        random.seed(self.traj_seed + theta_idx)
        # Seleccionar muestras del canal theta_idx
        group = self.theta_groups[theta_idx]
        samples = random.sample(group, k=min(2, len(group)))
        fig, axes = plt.subplots(2, 1, figsize=(10, 18))
        axes = axes.flatten()

        with T.no_grad():
            for i, (x_np, y_np) in enumerate(samples):
                x = T.FloatTensor(x_np).unsqueeze(0).cuda()      # [1, T, D]
                y = T.FloatTensor(y_np).unsqueeze(0).cuda()      # [1, T, 3]

                # Prepara los datos de contexto y predicción
                T_pts = x.size(1)
                n_ctx  = int(context_frac * T_pts)
                c_ind = T.arange(n_ctx)
                t_ind = T.arange(T_pts)
                cx, cy = x[:, c_ind, :], y[:, c_ind, :]
                tx, ty = x[:, t_ind, :], y[:, t_ind, :]
                
                # Predicción ANP
                y_anp, *_ = self.anp(cx, cy, tx)
                traj_anp = y_anp.squeeze().cpu().numpy()[:, :2]

                # Predicción DRS
                y_drs = self.mlps['DRS'](x).squeeze().cpu().numpy()[:, :2]
                traj_drs = y_drs

                # Verdadero
                traj_gt = ty.squeeze().cpu().numpy()[:, :2]

                ax = axes[i]
                # dibuja ground truth
                ax.plot(traj_gt[:, 0], traj_gt[:, 1],
                        linestyle='--', label='GT', alpha=0.7)
                # dibuja ANP
                ax.plot(traj_anp[:, 0], traj_anp[:, 1],
                        label='ANP', color="red", linewidth=2)
                # dibuja DRS
                ax.plot(traj_drs[:, 0], traj_drs[:, 1],
                        label='DRS', color="green", linewidth=2)

                ax.set_title(f"Trajectory {i+1}", fontsize=22)
                ax.set_xlabel("X", fontsize=20)
                ax.set_ylabel("Y", fontsize=20)
                # gira label 'Y' 90 grados
                ax.yaxis.label.set_rotation(0)
                ax.tick_params(axis='both', which='major', labelsize=18)
                ax.legend(fontsize=15, loc='best')
                ax.grid(True)

        plt.suptitle("Example Trajectory Predictions (ANP vs DRS)", fontsize=20)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        out_path = self.result_dir / 'trajectories.png'
        plt.savefig(out_path)
        plt.close(fig)
        print(f"Saved trajectory comparison plot to {out_path}")

    def plot_trajs_ci(self, context_frac=0.2, k=1.0, theta_idx=1):
        """
        Genera un grid de 4x2 plots para hasta 4 trayectorias del canal θ indicado,
        mostrando ground truth, ANP mean y sus intervalos de confianza ±k·σ.
        """
        print(f"traj_seed: {self.traj_seed}")
        # Seleccionar muestras del canal theta_idx
        random.seed(self.traj_seed + theta_idx)
        group = self.theta_groups[theta_idx]
        samples = random.sample(group, k=min(2, len(group)))
        fig, axes = plt.subplots(nrows=len(samples), ncols=2, figsize=(16, 6 * len(samples)))

        for i, (x_np, y_np) in enumerate(samples):
            x = T.FloatTensor(x_np).unsqueeze(0).cuda()
            y = T.FloatTensor(y_np).unsqueeze(0).cuda()
            size = x.size(1)

            # Contexto
            n_ctx = max(1, int(context_frac * size))
            cx, cy = x[:, :n_ctx, :], y[:, :n_ctx, :]
            tx = x

            # Predicción ANP: mean y var
            with T.no_grad():
                y_mean, y_var, *_ = self.anp(cx, cy, tx)
            y_mean = y_mean.squeeze(0).cpu().numpy()
            y_std = T.sqrt(y_var).squeeze(0).cpu().numpy()

            # Predicción DRS
            with T.no_grad():
                drs_out = self.mlps["DRS"](x)
            drs_pred = drs_out.squeeze(0).cpu().numpy()

            # Intervalos de confianza
            lower = y_mean - k * y_std
            upper = y_mean + k * y_std
            t = np.arange(size)

            for d in range(2):
                axis_label = 'x' if d == 0 else 'y'
                ax = axes[i, d] if len(samples) > 1 else axes[d]
                # Ground truth
                ax.plot(t, y.cpu().squeeze(0).numpy()[:, d], '--', label='Ground Truth')
                # ANP mean
                ax.plot(t, y_mean[:, d], label='ANP mean', color='red')
                # Intervalo de confianza en colo rojo
                ax.fill_between(t, lower[:, d], upper[:, d], color='red', alpha=0.2, label=f'±{k}σ')
                # DRS
                ax.plot(t, drs_pred[:, d], label='DRS', color='green')
                ax.set_title(f"Traj {i+1} – {axis_label} Axis", fontsize=18)

                # Aumentar tamaño de ticks
                ax.tick_params(axis='both', which='major', labelsize=14)

                ax.legend(fontsize=14)
                ax.grid(True)

        plt.tight_layout()
        out_path = self.result_dir / f'traj_ci_theta{theta_idx}_k{k}.png'
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Guardado gráfico CI para θ={self.theta_values[theta_idx]:.2f}: {out_path}")


    def run(self, modes: List[str]):
        if 'mean' in modes:
            anp_means = self.eval_anp_mean()
            self.plot_mean(anp_means)
        if 'heatmap' in modes or 'pvals' in modes:
            mlp_maes, anp_maes = self.eval_heatmap()
            if 'heatmap' in modes:
                self.plot_heatmap(mlp_maes, anp_maes)
            if 'pvals' in modes:
                sig = self.eval_pvals()
                self.plot_heatmap_p(mlp_maes, anp_maes, sig)
        if 'trajectories' in modes:
            self.plot_trajs()
            if 'ci_trajectories' in modes:
                self.plot_trajs_ci(context_frac=0.2, k=2)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate ANP & MLP models cleanly')
    parser.add_argument('--data-path', type=Path, required=True)
    parser.add_argument('--theta-path', type=Path, required=True)
    parser.add_argument('--anp-path', type=Path, required=True)
    parser.add_argument('--mlp-dir', type=Path, required=True)
    parser.add_argument('--result-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument(
        '--eval-modes',
        nargs='+',
        choices=['mean', 'heatmap', 'pvals', 'trajectories', 'ci_trajectories'],
        default=['mean', 'heatmap', 'pvals', 'trajectories', 'ci_trajectories'],
        help='Which evaluations to run'
    )
    args = parser.parse_args()

    ev = Evaluator(
        data_path=args.data_path,
        theta_path=args.theta_path,
        anp_ckpt=args.anp_path,
        mlp_dir=args.mlp_dir,
        result_dir=args.result_dir,
        batch_size=args.batch_size
    )
    ev.run(args.eval_modes)


if __name__ == '__main__':
    main()
