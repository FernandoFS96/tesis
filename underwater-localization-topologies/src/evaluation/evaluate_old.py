import os
import torch as t
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import random
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from src.models.anp import LatentModel
from src.models.mlp import MLP
from src.utils.nav_dataset import NavigationTrajectoryDataset

def load_data_and_group_by_theta(data_path, theta_values_path):
    """Load data and group it by theta values."""
    # Load trajectory data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    # Load corresponding theta values for each sample
    with open(theta_values_path, 'rb') as f:
        theta_dict = pickle.load(f)
    val_thetas = theta_dict.get('val_thetas', [])

    # Group samples by their theta value
    groups = {}
    for sample, theta in zip(data, val_thetas):
        groups.setdefault(theta, []).append(sample)

    # Sort by theta and return groups
    theta_values = sorted(groups.keys())
    theta_groups = [groups[t] for t in theta_values]
    return data, theta_groups, theta_values


def load_anp_model(model_path, input_dim):
    """Load a pre-trained ANP model."""
    model = LatentModel(num_hidden=128, input_dim=input_dim).cuda()
    state = t.load(model_path)
    # Support checkpoints with or without nested 'model' key
    model_state = state['model'] if 'model' in state else state
    model.load_state_dict(model_state)
    model.eval()
    return model


def load_mlp_models(model_dir, input_dim, output_dim):
    """Load all MLP models for each channel option and the combined model, in ascending order with custom labels."""
    mlp_models = {}

    # 1) Recolecta todas las carpetas de canal y extrae su valor numérico
    channel_entries = []
    for entry in os.listdir(model_dir):
        if entry.startswith('MLP_model_channel_option_'):
            try:
                theta_val = float(entry.split('MLP_model_channel_option_')[1])
            except ValueError:
                continue
            channel_entries.append((theta_val, entry))

    # 2) Ordénalos por el valor de theta
    channel_entries.sort(key=lambda x: x[0])

    # 3) Carga los modelos en ese orden y renombra las claves a 'MLP_x.x'
    for theta_val, entry in channel_entries:
        key = f"MLP_{theta_val:.1f}"  # e.g. 'MLP_0.6'
        model_path = os.path.join(model_dir, entry, 'best_model.pth')
        if os.path.exists(model_path):
            mlp_model = MLP(input_dim=input_dim, output_dim=output_dim)
            mlp_model.load_state_dict(t.load(model_path))
            mlp_model.cuda().eval()
            mlp_models[key] = mlp_model

    # 4) Añade el modelo combinado (DRS) al final
    combined_path = os.path.join(model_dir, 'combined_model.pth')
    if os.path.exists(combined_path):
        drs_model = MLP(input_dim=input_dim, output_dim=output_dim)
        drs_model.load_state_dict(t.load(combined_path))
        drs_model.cuda().eval()
        mlp_models['DRS'] = drs_model
    return mlp_models

def calculate_mae_for_theta_groups(anp_model, mlp_models, theta_groups, context_percent=10):
    """Calculate MAE for ANP and MLP models grouped by theta."""
    mae_values_anp = []
    mae_values_mlp = {key: [] for key in mlp_models.keys()}

    for group in theta_groups:
        dataset = NavigationTrajectoryDataset(group)
        if len(dataset) == 0:
            print(f"[WARN] No samples for a theta group, skipping.")
            continue

        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        total_mae_anp = 0.0
        total_mae_mlp = {key: 0.0 for key in mlp_models.keys()}
        count = 0

        with t.no_grad():
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

                # Split context and target points
                total_points = x_batch.size(1)

                num_context_points = int((context_percent / 100) * total_points)
                context_indices = t.arange(num_context_points)
                target_indices = t.arange(total_points)
#
                context_x = x_batch[:, context_indices, :]
                context_y = y_batch[:, context_indices, :]
                target_x = x_batch[:, target_indices, :]
                target_y = y_batch[:, target_indices, :]

                #num_context = int((context_percent / 100) * total_points)
#
                #context_x = x_batch[:, :num_context, :]
                #context_y = y_batch[:, :num_context, :]
                #target_x = x_batch[:, num_context:, :]
                #target_y = y_batch[:, num_context:, :]

                # ANP prediction and MAE
                y_pred_mean_anp, *_ = anp_model(context_x, context_y, target_x)
                mae_anp = t.mean(t.abs(y_pred_mean_anp - target_y)).item()
                total_mae_anp += mae_anp

                # MLP predictions and MAE
                for key, mlp_model in mlp_models.items():
                    y_pred_mlp = mlp_model(x_batch)
                    mae_mlp = t.mean(t.abs(y_pred_mlp - y_batch)).item()
                    total_mae_mlp[key] += mae_mlp

                count += 1

        # Compute and store average
        avg_mae_anp = total_mae_anp / count
        print(f"ANP MAE ({context_percent}% context): {avg_mae_anp:.4f}")
        mae_values_anp.append(avg_mae_anp)

        for key in mlp_models.keys():
            mae_values_mlp[key].append(total_mae_mlp[key] / count)

    return mae_values_anp, mae_values_mlp


def collect_mae_by_sample(anp_model, mlp_models, theta_groups, context_percent):
    """
    Devuelve, para cada canal theta, un dict:
      maes[channel][model_name] = [mae_muestra1, mae_muestra2, ...]
    donde 'model_name' es 'MLP_0.0', ..., 'DRS' o 'ANP_10%' etc.
    """
    maes = []
    model_names = list(mlp_models.keys()) + [f"ANP_{context_percent}%"]

    for group in theta_groups:
        ds = NavigationTrajectoryDataset(group)
        loader = DataLoader(ds, batch_size=1, shuffle=False)
        # inicializo listas
        channel_maes = {name: [] for name in model_names}

        with t.no_grad():
            for x, y in loader:
                x, y = x.cuda(), y.cuda()

                # Split context and target points
                total_points = x.size(1)

                num_context_points = int((context_percent / 100) * total_points)
                context_indices = t.arange(num_context_points)
                target_indices = t.arange(total_points)
#
                cx = x[:, context_indices, :]
                cy = y[:, context_indices, :]
                tx = x[:, target_indices, :]
                ty = y[:, target_indices, :]

                #num_context = int((context_percent / 100) * total_points)
#
                #context_x = x_batch[:, :num_context, :]
                #context_y = y_batch[:, :num_context, :]
                #target_x = x_batch[:, num_context:, :]
                #target_y = y_batch[:, num_context:, :]

                y_pred, *_ = anp_model(cx, cy, tx)
                mae_anp = t.mean((y_pred - ty).abs()).item()
                channel_maes[f"ANP_{context_percent}%"].append(mae_anp)

                # pred MLPs
                for name, m in mlp_models.items():
                    mae_m = t.mean((m(x) - y).abs()).item()
                    channel_maes[name].append(mae_m)

        maes.append(channel_maes)
    return maes  # lista de dicts, uno por canal

def plot_heatmap(mae_values_mlp, mae_values_anp_context, theta_values, context_percentages, output_path):
    """Plot a heatmap of MAE values for MLP and ANP models across theta groups."""
    # Combine MLP and ANP rows
    combined_rows = [mae for mae in mae_values_mlp.values()]
    combined_rows += [mae_values_anp_context[p] for p in context_percentages]

    # Pad rows to equal length
    max_len = len(theta_values)
    padded = [row + [np.nan] * (max_len - len(row)) for row in combined_rows]

    # Labels
    y_labels = list(mae_values_mlp.keys()) + [f"ANP ({p}%)" for p in context_percentages]

    plt.figure(figsize=(14, 10))
    sns.heatmap(np.array(padded), annot=True, fmt=".2f",
                cmap="viridis", xticklabels=theta_values,
                yticklabels=y_labels, annot_kws={"size":14})
    # Tick label sizes
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14, rotation=0)  # rotation=0 for horizontal y-labels

    plt.xlabel("Theta Parameter", fontsize=16)
    plt.ylabel("Model / Context (%)", fontsize=16)
    plt.title("MAE: MLP vs. ANP across context sizes and theta", fontsize=18)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def plot_predictions_for_random_trajectories(anp_model, drs_model, data, result_dir, num_samples=4):
    """Plot a few example trajectories comparing ANP and combined MLP (DRS)."""
    os.makedirs(result_dir, exist_ok=True)
    samples = data[:num_samples]
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    with t.no_grad():
        for i, (x, y_true) in enumerate(samples):
            x = t.FloatTensor(x).unsqueeze(0).cuda()
            y_true = t.FloatTensor(y_true).unsqueeze(0).cuda()

            total_points = x.size(1)
            num_context_points = int(0.4 * total_points)
            context_indices = t.arange(num_context_points)
            target_indices = t.arange(total_points)

            cx, cy = x[:, context_indices, :], y_true[:, context_indices, :]
            tx, ty = x[:, target_indices, :], y_true[:, target_indices, :]

            #num_context = int(0.4 * total_points)
            #cx, cy = x[:, :num_context, :], y_true[:, :num_context, :]
            #tx, ty = x[:, num_context:, :], y_true[:, num_context:, :]

            # ANP predictions
            y_anp, *_ = anp_model(cx, cy, tx)
            y_anp = y_anp.squeeze().cpu().numpy()[:, :2]

            # DRS predictions
            y_drs = drs_model(x).squeeze().cpu().numpy()[:, :2]

            # Ground truth
            y_gt = ty.squeeze().cpu().numpy()[:, :2]
            
            # Plotting trajectories
            ax = axes[i]
            ax.plot(y_gt[:,0], y_gt[:,1], '--', label="Ground Truth")
            ax.plot(y_anp[:,0], y_anp[:,1], label="ANP", color="red")
            ax.plot(y_drs[:,0], y_drs[:,1], label="DRS", color="green")
            ax.set_title(f"Trajectory {i+1}", fontsize=18)
            ax.set_xlabel("X Coordinate", fontsize=16)
            ax.set_ylabel("Y Coordinate", fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.legend(fontsize=14, loc='best')

    plt.suptitle("Example Trajectory Predictions", fontsize=20)
    plt.tight_layout(rect=[0,0.03,1,0.95])
    path = os.path.join(result_dir, 'trajectory_comparison.png')
    plt.savefig(path)
    plt.show()
    print(f"Saved example trajectories to {path}")


def plot_mean_mae_anp_context(mae_values_anp_context, context_percentages, output_path):
    """Plot mean MAE of ANP across different context sizes."""
    means = [np.mean(mae_values_anp_context[p]) for p in context_percentages]
    plt.figure(figsize=(10,6))
    plt.plot(context_percentages, means, marker='o')
    plt.title('Mean ANP MAE vs Context Size', fontsize=16)
    plt.xlabel('Context (%)', fontsize=14)
    plt.ylabel('Mean MAE', fontsize=14)
    # Tick label sizes
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14, rotation=0)  
    
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

def plot_heatmap_with_p_values(
    mae_mlp: dict[str, list[float]],
    mae_anp_ctx: dict[int, list[float]],
    significance: list[dict],
    theta_values: list[float],
    context_percentages: list[int],
    output_path: str
):
    """
    Muestra un heatmap de MAEs para MLP vs ANP (varios contextos),
    anotando con un '*' las celdas donde la comparación contra el
    modelo de referencia es estadísticamente significativa.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 1) Construir la matriz de MAEs
    rows = [mae_mlp[k] for k in mae_mlp.keys()]
    rows += [mae_anp_ctx[p] for p in context_percentages]
    max_len = len(theta_values)
    data = [r + [np.nan] * (max_len - len(r)) for r in rows]

    # 2) Etiquetas de filas
    row_labels = list(mae_mlp.keys()) + [f"ANP ({p}%)" for p in context_percentages]

    # 3) Construir matriz de anotaciones con estrellas
    annot = []
    for i, model_name in enumerate(row_labels):
        row_annot = []
        for j in range(len(theta_values)):
            val = data[i][j]
            # localizar si existe significance[j][model_name]['significant']
            sig = False
            if model_name in significance[j]:
                sig = significance[j][model_name].get('significant', False)
            star = "*" if sig else ""
            row_annot.append(f"{val:.2f}{star}")
        annot.append(row_annot)

    # 4) Plot
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        np.array(data),
        annot=np.array(annot),
        fmt="",
        cmap="viridis",
        xticklabels=theta_values,
        yticklabels=row_labels,
        annot_kws={"size": 12}
    )
    plt.xlabel("Theta", fontsize=14)
    plt.ylabel("Modelo / Contexto", fontsize=14)
    plt.title("MAE con p-values ajustados (★ = p<0.05)", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate ANP & MLP models.")
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--theta-values-path', required=True)
    parser.add_argument('--anp-model-path', required=True)
    parser.add_argument('--mlp-model-dir', required=True)
    parser.add_argument('--result-dir', required=True)
    parser.add_argument('--eval-modes', nargs='+', choices=['mean','heatmap','trajectories'],
                        default=['mean','heatmap','trajectories'],
                        help="Which evaluations to run.")
    args=parser.parse_args()

    data,theta_groups,theta_vals = load_data_and_group_by_theta(
        args.data_path, args.theta_values_path)
    print(f"Loaded {len(data)} samples, groups:{theta_vals}")
    in_dim=data[0][0].shape[-1]; out_dim=data[0][1].shape[-1]
    anp=load_anp_model(args.anp_model_path, in_dim)
    mlps=load_mlp_models(args.mlp_model_dir, in_dim, out_dim)
    os.makedirs(args.result_dir,exist_ok=True)
    ctxs=[2,4,6,10,15,20,30,50,80]
    mae_ctx={}
    if 'mean' in args.eval_modes or 'heatmap' in args.eval_modes:
        for cp in ctxs:
            mae, _ = calculate_mae_for_theta_groups(anp,{},theta_groups,context_percent=cp)
            mae_ctx[cp]=mae
    if 'mean' in args.eval_modes:
        plot_mean_mae_anp_context(mae_ctx,ctxs,os.path.join(args.result_dir,'mean_anp.png'))

    mae_mlp={}    
    if 'heatmap' in args.eval_modes:
        _,mae_mlp = calculate_mae_for_theta_groups(anp,mlps,theta_groups,context_percent=10)
        plot_heatmap(mae_mlp,mae_ctx,theta_vals,ctxs,os.path.join(args.result_dir,'heatmap.png'))
    if 'trajectories' in args.eval_modes and 'DRS' in mlps:
        plot_predictions_for_random_trajectories(anp,mlps['DRS'],data,args.result_dir)

    maes = collect_mae_by_sample(anp, mlps, theta_groups, context_percent=10)
    significance = []  # lista de dicts por canal: { model_name: (p_orig,p_adj,reject) }
    alpha = 0.05

    for channel_idx, channel_maes in enumerate(maes):
        # 1) media por modelo
        means = {m: np.mean(errs) for m, errs in channel_maes.items()}
        # 2) referencia = modelo con media mínima
        ref_model = min(means, key=means.get)
        ref_errs  = channel_maes[ref_model]

        # 3) calcula p-vals vs cada otro modelo
        pvals = []
        others = []
        for m, errs in channel_maes.items():
            if m == ref_model: continue
            # test pareado: Wilcoxon (no normalidad asumida)
            stat, p = wilcoxon(ref_errs, errs)
            pvals.append(p)
            others.append(m)

        # 4) ajustar múltiples comparaciones
        rej, p_adj, _, _ = multipletests(pvals, alpha=alpha, method='holm')

        # 5) guardar resultados
        channel_sig = { ref_model: {'significant': False} }
        for m, p0, p1, r in zip(others, pvals, p_adj, rej):
            channel_sig[m] = {
                'p_orig': float(p0),
                'p_adj':  float(p1),
                'significant': bool(r)
            }
        significance.append(channel_sig)
    plot_heatmap_with_p_values(mae_mlp,mae_ctx,significance,theta_vals,ctxs,os.path.join(args.result_dir, 'heatmap_pvals.png'))

if __name__ == "__main__":
    main()
