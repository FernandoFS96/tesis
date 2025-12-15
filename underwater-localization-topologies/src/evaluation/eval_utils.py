# src/evaluation/eval_utils.py
import os
import torch as t
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.utils.nav_dataset import NavigationTrajectoryDataset
from src.models.anp import LatentModel
from src.models.mlp import MLP

def load_models(anp_model_path, mlp_model_paths, input_dim, output_dim, num_hidden):
    """
    Load ANP and MLP models from their respective paths.

    :param anp_model_path: Path to the saved ANP model.
    :param mlp_model_paths: Dictionary with channel options as keys and model paths as values.
    :param input_dim: Input dimension for models.
    :param output_dim: Output dimension for models.
    :param num_hidden: Number of hidden units for both ANP and MLP models.
    :return: Loaded ANP model and a dictionary of MLP models.
    """
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    # Load ANP model
    anp_model = LatentModel(num_hidden=num_hidden, input_dim=input_dim).to(device)
    anp_model.load_state_dict(t.load(anp_model_path)["model"])
    anp_model.eval()

    # Load MLP models
    mlp_models = {}
    for channel_option, model_path in mlp_model_paths.items():
        if os.path.exists(model_path):
            mlp_model = MLP(input_dim, output_dim, num_hidden).to(device)
            mlp_model.load_state_dict(t.load(model_path))
            mlp_model.eval()
            mlp_models[channel_option] = mlp_model

    return anp_model, mlp_models


def calculate_mae_for_theta_groups(anp_model, mlp_models, theta_groups, context_percent=10):
    """
    Calculate MAE for ANP and MLP models across theta groups.

    :param anp_model: Trained ANP model.
    :param mlp_models: Dictionary of trained MLP models.
    :param theta_groups: List of datasets grouped by theta.
    :param context_percent: Percentage of context points to use.
    :return: MAE values for ANP and MLP models.
    """
    mae_values_anp = []
    mae_values_mlp = {key: [] for key in mlp_models.keys()}

    for group in theta_groups:
        dataset = NavigationTrajectoryDataset(group)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        total_mae_anp = 0
        total_mae_mlp = {key: 0 for key in mlp_models.keys()}
        count = 0

        with t.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()

                # Split context and target points
                total_points = x_batch.size(1)
                num_context_points = int((context_percent / 100) * total_points)

                context_indices = t.arange(num_context_points)
                target_indices = t.arange(total_points)

                context_x = x_batch[:, context_indices, :]
                context_y = y_batch[:, context_indices, :]
                target_x = x_batch[:, target_indices, :]
                target_y = y_batch[:, target_indices, :]

                # ANP predictions
                y_pred_mean_anp, _, _, _, _ = anp_model(context_x, context_y, target_x)
                total_mae_anp += t.mean(t.abs(y_pred_mean_anp - target_y)).item()

                # MLP predictions
                for key, mlp_model in mlp_models.items():
                    y_pred_mlp = mlp_model(x_batch)
                    total_mae_mlp[key] += t.mean(t.abs(y_pred_mlp - y_batch)).item()

                count += 1

        # Average MAE for this theta group
        mae_values_anp.append(total_mae_anp / count)
        for key in mlp_models.keys():
            mae_values_mlp[key].append(total_mae_mlp[key] / count)

    return mae_values_anp, mae_values_mlp


def plot_mae_heatmap(mae_values, theta_values, ytick_labels, save_path):
    """
    Plot a heatmap of MAE values.

    :param mae_values: List of MAE values.
    :param theta_values: List of theta values (x-axis labels).
    :param ytick_labels: Labels for the y-axis.
    :param save_path: Path to save the heatmap.
    """
    plt.figure(figsize=(14, 10))
    sns.heatmap(np.array(mae_values), annot=True, fmt=".0f", cmap="viridis",
                xticklabels=theta_values, yticklabels=ytick_labels)
    plt.xlabel("Theta Parameter")
    plt.title("MAE for ANP and MLP Models")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Heatmap saved at: {save_path}")


def plot_trajectories(anp_model, mlp_model, data, num_samples=4, save_path=None):
    """
    Plot predictions for ANP and MLP models on random trajectories.

    :param anp_model: Trained ANP model.
    :param mlp_model: Trained MLP model.
    :param data: Dataset of trajectories.
    :param num_samples: Number of trajectories to plot.
    :param save_path: Path to save the plot.
    """
    import random
    random_samples = random.sample(data, num_samples)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    with t.no_grad():
        for i, (x, y_true) in enumerate(random_samples):
            x = t.FloatTensor(x).unsqueeze(0).cuda()
            y_true = t.FloatTensor(y_true).unsqueeze(0).cuda()

            # ANP predictions
            y_pred_mean_anp, _, _, _, _ = anp_model(x, y_true, x)
            y_pred_anp = y_pred_mean_anp.squeeze().cpu().numpy()

            # MLP predictions
            y_pred_mlp = mlp_model(x)
            y_pred_mlp = y_pred_mlp.squeeze().cpu().numpy()

            # Ground truth
            y_true = y_true.squeeze().cpu().numpy()

            ax = axes[i]
            ax.plot(y_true[:, 0], y_true[:, 1], label="Ground Truth", color="blue")
            ax.plot(y_pred_anp[:, 0], y_pred_anp[:, 1], label="ANP Prediction", color="red")
            ax.plot(y_pred_mlp[:, 0], y_pred_mlp[:, 1], label="MLP Prediction", color="green")

            ax.set_title(f"Trajectory {i+1}")
            ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Trajectory plot saved at: {save_path}")
    plt.show()
