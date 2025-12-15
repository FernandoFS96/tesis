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
        return {c: [np.mean(e[f"ANP({c}%)"]) for e in errs] for c in CONTEXTS} # type: ignore

    def eval_mlp_mean(self) -> Dict[str, List[float]]:
        errs = self._compute_errors()
        return {name: [np.mean(e[name]) for e in errs] for name in self.mlps.keys()} # type: ignore

    def eval_heatmap(self) -> tuple[Dict[str, List[float]], Dict[int, List[float]]]:
        errs = self._compute_errors()
        mlp_maes = {name: [np.mean(e[name]) for e in errs] for name in self.mlps.keys()}
        anp_maes = {c: [np.mean(e[f"ANP({c}%)"]) for e in errs] for c in CONTEXTS}
        return mlp_maes, anp_maes # type: ignore

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

    def plot_trajs(
        self,
        num_samples: int = 2,
        context_frac: float = 0.4,
        theta_idx: int = 1,
        traj_indices=None,
        invert_indices=None,  # <- NEW
    ):
        """
        Plot a few example trajectories comparing ANP and the combined MLP (DRS).

        Parameters
        ----------
        num_samples : int
            How many trajectories to plot when choosing randomly.
            Ignored if `traj_indices` is given.
        context_frac : float
            Fraction of points used as context.
        theta_idx : int
            Index of theta group in self.theta_groups.
        traj_indices : int | list[int] | None
            Specific trajectory index/indices within that theta group.
            If None, choose `num_samples` at random.
        invert_indices : list[int] | None
            Indices (within that theta group) of trajectories
            that should be *flipped* in both axes (x -> -x, y -> -y).
        """
        group = self.theta_groups[theta_idx]

        # Normalize invert_indices
        if invert_indices is None:
            invert_set = set()
        else:
            invert_set = set(invert_indices)

        # Sensor positions on an ellipse
        # most left = (-1000, 0), most right = (1000, 0)
        n_sensors = 10
        a = 1000.0 # semi-major axis (x)
        b = 500.0 # semi-minor axis (y)
        sensor_ids = np.arange(n_sensors)
        theta = 2 * np.pi * sensor_ids / n_sensors
        sensor_x = a * np.cos(theta)
        sensor_y = b * np.sin(theta)

        # --- Select trajectories ---
        if traj_indices is not None:
            if isinstance(traj_indices, int):
                traj_indices = [traj_indices]

            indices = []
            samples = []
            for idx in traj_indices:
                if idx < 0 or idx >= len(group):
                    raise IndexError(
                        f"traj_idx {idx} out of range for theta_idx={theta_idx} "
                        f"(n_trajs={len(group)})"
                    )
                indices.append(idx)
                samples.append(group[idx])
        else:
            random.seed(self.traj_seed + theta_idx)
            n_to_sample = min(num_samples, 4, len(group))
            indices = random.sample(range(len(group)), k=n_to_sample)
            samples = [group[idx] for idx in indices]

        n_plots = len(samples)
        #fig, axes = plt.subplots(n_plots, 1, figsize=(10, 9 * n_plots))
        fig, axes = plt.subplots(
            n_plots, 1,
            figsize=(11, 10 * n_plots),
            constrained_layout=True
        )
        if n_plots == 1:
            axes = [axes]

        ref_xlim = None
        ref_ylim = None
        with T.no_grad():
            for ax, traj_idx, (x_np, y_np) in zip(axes, indices, samples):
                x = T.FloatTensor(x_np).unsqueeze(0).cuda()      # [1, T, D]
                y = T.FloatTensor(y_np).unsqueeze(0).cuda()      # [1, T, 3]

                T_pts = x.size(1)
                n_ctx  = int(context_frac * T_pts)
                c_ind = T.arange(n_ctx)
                t_ind = T.arange(T_pts)
                cx, cy = x[:, c_ind, :], y[:, c_ind, :]
                tx, ty = x[:, t_ind, :], y[:, t_ind, :]

                # ANP prediction
                y_anp, *_ = self.anp(cx, cy, tx)
                traj_anp = y_anp.squeeze().cpu().numpy()[:, :2]

                # DRS prediction
                y_drs = self.mlps['MLP(0.1)'](x).squeeze().cpu().numpy()[:, :2]
                traj_drs = y_drs

                # Ground truth
                traj_gt = ty.squeeze().cpu().numpy()[:, :2]

                # ----- OPTIONAL: flip both axes for selected trajectories -----
                if traj_idx in invert_set:
                    traj_gt  = -traj_gt
                    traj_anp = -traj_anp
                    traj_drs = -traj_drs
                    # NOTE: sensors stay fixed in global coordinates;
                    # if prefer them to flip with the traj do:
                    # sensor_x_plot, sensor_y_plot = -sensor_x, -sensor_y
                sensor_x_plot, sensor_y_plot = sensor_x, sensor_y

                # --------- Compute / reuse axis limits ---------
                # Combine all x,y used in this plot
                xs = np.concatenate([
                    traj_gt[:, 0], traj_anp[:, 0], traj_drs[:, 0]#, sensor_x_plot
                ])
                ys = np.concatenate([
                    traj_gt[:, 1], traj_anp[:, 1], traj_drs[:, 1]#, sensor_y_plot
                ])

                if ref_xlim is None:
                    # First trajectory: define the reference limits (with small margin)
                    x_min, x_max = xs.min(), xs.max()
                    y_min, y_max = ys.min(), ys.max()
                    x_margin = 0.05 * (x_max - x_min) if x_max > x_min else 1.0
                    y_margin = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
                    ref_xlim = (x_min - x_margin, x_max + x_margin)
                    ref_ylim = (y_min - y_margin, y_max + y_margin)
                
                # For subsequent trajectories, compute their own limits
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()
                
                x_margin = 0.05 * (x_max - x_min) if x_max > x_min else 1.0
                y_margin = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
                
                # Use the same limits for *all* subplots
                ax.set_xlim(x_min - x_margin, x_max + x_margin)#ax.set_xlim(ref_xlim)
                ax.set_ylim(y_min - y_margin, y_max + y_margin)#ax.set_ylim(ref_ylim)

                # Force equal scale on both axes
                ax.set_aspect('equal', adjustable='box')
                # ----- Plot -----
                ax.plot(traj_gt[:, 0],  traj_gt[:, 1],
                        linestyle='--', label='GT', alpha=0.7)
                ax.plot(traj_anp[:, 0], traj_anp[:, 1],
                        label='ANP', color="red", linewidth=2)
                ax.plot(traj_drs[:, 0], traj_drs[:, 1],
                        label='DR-MLP', color="green", linewidth=2)
                
                # ----- Plot sensors on the ellipse
                ax.scatter(sensor_x_plot, sensor_y_plot, s=60, color="red", label="Sensors", zorder=5)
                for sid, (sx, sy) in enumerate(zip(sensor_x_plot, sensor_y_plot)):
                    ax.text(sx, sy + 30, str(sid), ha='center', va='bottom', fontsize=16)

                ax.set_title( f"Theta group {theta_idx} – Trajectory {traj_idx}", fontsize=22)

                ax.set_xlabel("X[m]", fontsize=22)
                ax.set_ylabel("Y[m]", fontsize=22)
                ax.yaxis.label.set_rotation(90)
                ax.tick_params(axis='both', which='major', labelsize=20)
                ax.legend(fontsize=18, loc='upper right')
                ax.grid(True)

        plt.suptitle("Example Trajectory Predictions (ANP vs DRS)", fontsize=22)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        out_path = self.result_dir / 'trajectories.png'
        plt.savefig(out_path)
        plt.close(fig)
        print(f"Saved trajectory comparison plot to {out_path}")



    def plot_trajs_overlay(self, traj_indices, context_frac: float = 0.4, theta_idx: int = 0, invert_indices=None, mlp_key: str = 'MLP(0.1)',):
        """
        Plotea varias trayectorias (GT + ANP + MLP) en un único eje.
        """
        group = self.theta_groups[theta_idx]

        # Normalizar argumentos
        if isinstance(traj_indices, int):
            traj_indices = [traj_indices]

        if invert_indices is None:
            invert_set = set()
        else:
            invert_set = set(invert_indices)

        # --- Sensores en la elipse (como antes) ---------------------------
        n_sensors = 10
        a = 1000.0
        b = 500.0
        sensor_ids = np.arange(n_sensors)
        theta = 2 * np.pi * sensor_ids / n_sensors
        sensor_x = a * np.cos(theta)
        sensor_y = b * np.sin(theta)
        # ------------------------------------------------------------------

        fig, ax = plt.subplots(figsize=(8, 8))

        all_xs, all_ys = [], []

        with T.no_grad():
            for i, traj_idx in enumerate(traj_indices):
                x_np, y_np = group[traj_idx]

                x = T.FloatTensor(x_np).unsqueeze(0).cuda()  # [1, T, D]
                y = T.FloatTensor(y_np).unsqueeze(0).cuda()  # [1, T, 3]

                T_pts = x.size(1)
                n_ctx = int(context_frac * T_pts)
                c_ind = T.arange(n_ctx)
                t_ind = T.arange(T_pts)
                cx, cy = x[:, c_ind, :], y[:, c_ind, :]
                tx, ty = x[:, t_ind, :], y[:, t_ind, :]

                # ANP
                y_anp, *_ = self.anp(cx, cy, tx)
                traj_anp = y_anp.squeeze().cpu().numpy()[:, :2]

                # MLP (DRS u otro)
                mlp_model = self.mlps[mlp_key]
                y_mlp = mlp_model(x).squeeze().cpu().numpy()[:, :2]

                # GT
                traj_gt = ty.squeeze().cpu().numpy()[:, :2]

                # ¿Invertir?
                if traj_idx in invert_set:
                    traj_gt  = -traj_gt
                    traj_anp = -traj_anp
                    y_mlp    = -y_mlp

                # Guardar para límites globales
                all_xs.extend([traj_gt[:, 0], traj_anp[:, 0], y_mlp[:, 0]])
                all_ys.extend([traj_gt[:, 1], traj_anp[:, 1], y_mlp[:, 1]])

                # Para que la leyenda no tenga entradas duplicadas,
                # sólo ponemos label en la primera trayectoria
                first = (i == 0)
                label_gt  = 'GT' if first else None
                label_anp = 'ANP' if first else None
                label_mlp = 'DR-MLP' if first else None

                ax.plot(
                    traj_gt[:, 0], traj_gt[:, 1],
                    linestyle='--',
                    color='tab:blue',
                    alpha=0.7,
                    label=label_gt,
                )
                ax.plot(traj_anp[:, 0], traj_anp[:, 1],
                        color="red", linewidth=2, label=label_anp)
                ax.plot(y_mlp[:, 0], y_mlp[:, 1],
                        color="green", linewidth=2, label=label_mlp)

        # Límites basados solo en las trayectorias
        xs = np.concatenate(all_xs)
        ys = np.concatenate(all_ys)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        x_margin = 0.05 * (x_max - x_min) if x_max > x_min else 1.0
        y_margin = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.set_aspect('equal', adjustable='box')

        # Sensores (una sola vez)
        ax.scatter(sensor_x,sensor_y,s=60,facecolors="red",edgecolors="black",linewidths=1.0,marker="o",label="Sensors",zorder=5,)
        for sid, (sx, sy) in enumerate(zip(sensor_x, sensor_y)):
            ax.text(sx, sy + 30, str(sid), ha='center', va='bottom', fontsize=14)

        ax.set_xlabel("X [m]", fontsize=16)
        ax.set_ylabel("Y [m]", fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.grid(True)
        ax.legend(loc='upper right', fontsize=12)

        ax.set_title(
            #f"Theta group {theta_idx} – Trajectories {traj_indices}",
            "Trajectories 1 & 2",
            fontsize=20
        )

        plt.tight_layout()
        out_path = self.result_dir / 'trajectories_overlay.png'
        plt.savefig(out_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Saved overlay plot to {out_path}")



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

    
    def plot_all_trajs(self, context_frac: float = 0.4):
        """
        For every trajectory in the validation set, plot GT vs ANP vs DRS (XY plane)
        in an individual figure and save it under:
            <self.result_dir>/all_trajectories/
        If you construct Evaluator with
            result_dir = Path('/home/fernando/tesis/underwater-localization/results/evaluation/low_variance')
        then all plots will end up inside:
            /home/fernando/tesis/underwater-localization/results/evaluation/low_variance/all_trajectories
        """
        # Subfolder inside the requested directory
        out_dir = self.result_dir / 'all_trajectories'
        out_dir.mkdir(parents=True, exist_ok=True)
        if 'DRS' not in self.mlps:
            raise RuntimeError("DRS model not loaded in self.mlps (missing 'DRS' key in self.mlps).")
        num_saved = 0
        with T.no_grad():
            # Iterate over theta groups so we keep track of θ in filenames
            for theta_idx, (theta, group) in enumerate(zip(self.theta_values, self.theta_groups)):
                for traj_idx, (x_np, y_np) in enumerate(group):
                    # x_np: [T, D], y_np: [T, 3]
                    x = T.FloatTensor(x_np).unsqueeze(0).cuda()  # [1, T, D]
                    y = T.FloatTensor(y_np).unsqueeze(0).cuda()  # [1, T, 3]
                    T_pts = x.size(1)
                    n_ctx = max(1, int(context_frac * T_pts))
                    # Context and target indices
                    c_ind = T.arange(n_ctx, device=x.device)
                    t_ind = T.arange(T_pts, device=x.device)
                    cx, cy = x[:, c_ind, :], y[:, c_ind, :]
                    tx, ty = x[:, t_ind, :], y[:, t_ind, :]
                    # --- ANP prediction ---
                    y_anp, *_ = self.anp(cx, cy, tx)
                    traj_anp = y_anp.squeeze(0).cpu().numpy()[:, :2]  # use X,Y
                    # --- DRS (combined MLP) prediction ---
                    y_drs = self.mlps['DRS'](x)
                    traj_drs = y_drs.squeeze(0).cpu().numpy()[:, :2]
                    # --- MLP prediction ---
                    y_mlp = self.mlps['MLP(0.0)'](x)
                    traj_mlp = y_mlp.squeeze(0).cpu().numpy()[:, :2]
                    # --- Ground truth ---
                    traj_gt = ty.squeeze(0).cpu().numpy()[:, :2]
                    # --- Plot: one trajectory per figure ---
                    fig, ax = plt.subplots(figsize=(8, 8))
                    # Ground truth
                    ax.plot(
                        traj_gt[:, 0], traj_gt[:, 1],
                        linestyle='--', label='GT', alpha=0.7
                    )
                    # ANP prediction
                    ax.plot(
                        traj_anp[:, 0], traj_anp[:, 1],
                        label='ANP', color='red', linewidth=2
                    )
                    # DRS prediction
                    ax.plot(
                        traj_drs[:, 0], traj_drs[:, 1],
                        label='DRS', color='green', linewidth=2
                    )
                    # MLP prediction
                    ax.plot(
                        traj_mlp[:, 0], traj_drs[:, 1],
                        label='MLP(θ=0.1)', color='orange', linewidth=2
                    )
                    ax.set_title(f"θ = {theta:.2f} – Trajectory {traj_idx}", fontsize=18)
                    ax.set_xlabel("X[m]", fontsize=18)
                    ax.set_ylabel("Y[m]", fontsize=18)
                    ax.yaxis.label.set_rotation(90)
                    ax.tick_params(axis='both', which='major', labelsize=18)
                    ax.legend(fontsize=16)
                    ax.grid(True)
                    ax.set_aspect('equal', adjustable='box')
                    fig.tight_layout()
                    # Filename encodes theta and trajectory index
                    filename = f"theta_{theta:.2f}_traj_{traj_idx:03d}.png"
                    fig.savefig(out_dir / filename, dpi=300)
                    plt.close(fig)
                    num_saved += 1
        print(f"Saved {num_saved} individual trajectory plots to {out_dir}")

    def plot_single_traj(self, theta_value: float, traj_idx: int, context_frac: float = 0.4):
        """
        Plot GT vs ANP vs DRS for one specific trajectory:
            - theta_value: numeric value of theta (e.g. 0.0, 0.1, ...)
            - traj_idx: index of the trajectory inside that theta group (0-based)
        The figure is saved to: <result_dir>/single_trajectories/theta_<theta>_traj_<idx>.png
        """
        # Locate the theta group (robust to tiny float differences)
        theta_arr = np.array(self.theta_values, dtype=float)
        theta_idx = int(np.argmin(np.abs(theta_arr - theta_value)))
        if not np.isclose(theta_arr[theta_idx], theta_value, atol=1e-6):
            raise ValueError(f"Theta {theta_value} not found. Available thetas: {self.theta_values}")
        group = self.theta_groups[theta_idx]

        if traj_idx < 0 or traj_idx >= len(group):
            raise IndexError(f"traj_idx {traj_idx} out of range for theta={theta_arr[theta_idx]} (n={len(group)})")

        x_np, y_np = group[traj_idx]

        with T.no_grad():
            x = T.FloatTensor(x_np).unsqueeze(0).cuda()      # [1, T, D]
            y = T.FloatTensor(y_np).unsqueeze(0).cuda()      # [1, T, 3]

            # Prepare context and target sets (same logic as plot_trajs)
            T_pts = x.size(1)
            n_ctx = max(1, int(context_frac * T_pts))
            c_ind = T.arange(n_ctx, device=x.device)
            t_ind = T.arange(T_pts, device=x.device)
            cx, cy = x[:, c_ind, :], y[:, c_ind, :]
            tx, ty = x[:, t_ind, :], y[:, t_ind, :]

            # ANP prediction
            y_anp, *_ = self.anp(cx, cy, tx)
            traj_anp = y_anp.squeeze(0).cpu().numpy()[:, :2]

            # DRS prediction
            y_drs = self.mlps['DRS'](x)
            traj_drs = y_drs.squeeze(0).cpu().numpy()[:, :2]

            # Ground truth
            traj_gt = ty.squeeze(0).cpu().numpy()[:, :2]

        # Plot in XY plane
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(traj_gt[:, 0], traj_gt[:, 1], linestyle='--', label='GT', alpha=0.7)
        ax.plot(traj_anp[:, 0], traj_anp[:, 1], label='ANP', color="red", linewidth=2)
        ax.plot(traj_drs[:, 0], traj_drs[:, 1], label='DRS', color="green", linewidth=2)

        ax.set_title(f"theta={theta_arr[theta_idx]:.2f}, traj_idx={traj_idx}", fontsize=18)
        ax.set_xlabel("X", fontsize=16)
        ax.set_ylabel("Y", fontsize=16)
        ax.yaxis.label.set_rotation(0)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.legend(fontsize=12, loc='best')
        ax.grid(True)

        out_dir = self.result_dir / 'single_trajectories'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"theta_{theta_arr[theta_idx]:.2f}_traj_{traj_idx:03d}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved single trajectory plot to {out_path}")



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
            self.plot_trajs(theta_idx=0, traj_indices=[24, 26], invert_indices=[26])#.plot_trajs()
            if 'ci_trajectories' in modes:
                self.plot_trajs_ci(context_frac=0.2, k=2)
        if 'all_trajectories' in modes:
            self.plot_all_trajs(context_frac=0.4)
        self.plot_single_traj(theta_value=self.theta_values[0], traj_idx=24, context_frac=0.3)
        self.plot_trajs_overlay(traj_indices=[24, 26], theta_idx=0,invert_indices=[26])


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
        choices=['mean', 'heatmap', 'pvals', 'trajectories', 'ci_trajectories', 'all_trajectories'],
        default=['mean', 'heatmap', 'pvals', 'trajectories', 'ci_trajectories', 'all_trajectories'],
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
