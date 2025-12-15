import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch as t
import pickle
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.models.anp import LatentModel
from src.models.mlp import MLP
from src.utils.nav_dataset import NavigationTrajectoryDataset

# Paths
VAL_DATA_PATH = "/home/fernando/tesis/underwater-localization/data_save/data_processed/val_data_anp.pkl"
ANP_MODEL_PATH = "/home/fernando/tesis/underwater-localization/results/ANP/low_variance/experiment_20250117-153349/best_checkpoint.pth.tar"
MLP_MODEL_PATH = "/home/fernando/tesis/underwater-localization/results/MLP/low_variance/combined_model.pth"
#SENSOR_PATH = "/home/fernando/tesis/underwater-localization/data/channel_option_0/channel_info/sensor_positions_0.npy"

def load_models(input_dim, output_dim):
    # Load ANP
    anp_model = LatentModel(num_hidden=128, input_dim=input_dim).cuda()
    anp_model.load_state_dict(t.load(ANP_MODEL_PATH)['model'])
    anp_model.eval()

    # Load MLP
    mlp_model = MLP(input_dim=input_dim, output_dim=output_dim).cuda()
    mlp_model.load_state_dict(t.load(MLP_MODEL_PATH))
    mlp_model.eval()

    return anp_model, mlp_model

def plot_trajectories(anp_model, mlp_model, dataset, save_dir, sensor_pos = None, num_samples=10):
    os.makedirs(save_dir, exist_ok=True)
    indices = np.random.choice(len(dataset), size=num_samples, replace=False)

    for i, idx in enumerate(indices):
        x, y_true = dataset[idx]
        x = t.FloatTensor(x).unsqueeze(0).cuda()
        y_true = t.FloatTensor(y_true).unsqueeze(0).cuda()

        with t.no_grad():
            # ANP prediction
            context_size = int(0.3 * x.size(1))
            context_idx = t.arange(context_size)
            target_idx = t.arange(x.size(1))

            context_x = x[:, context_idx, :]
            context_y = y_true[:, context_idx, :]
            target_x = x[:, target_idx, :]
            target_y = y_true[:, target_idx, :]

            y_pred_anp, _, _, _, _ = anp_model(context_x, context_y, target_x)
            y_pred_anp = y_pred_anp.squeeze().cpu().numpy()[:, :2]

            # MLP prediction
            y_pred_mlp = mlp_model(x).squeeze().cpu().numpy()[:, :2]

        y_true_np = y_true.squeeze().cpu().numpy()[:, :2]

        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(y_true_np[:, 0], y_true_np[:, 1], '--', label="Ground Truth", color="blue")
        plt.plot(y_pred_anp[:, 0], y_pred_anp[:, 1], label="ANP Prediction", color="red")
        plt.plot(y_pred_mlp[:, 0], y_pred_mlp[:, 1], label="MLP Prediction", color="green")
        #plt.scatter(sensor_pos[:, 0], sensor_pos[:, 1], marker="x", color="black", label="Sensors")
        plt.title(f"Trajectory {i+1}")
        plt.xlabel("X Axis")
        plt.ylabel("Y Axis")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"trajectory_{i+1}.png"))
        plt.close()

def main():
    # Load validation data
    with open(VAL_DATA_PATH, 'rb') as f:
        val_data = pickle.load(f)
    dataset = NavigationTrajectoryDataset(val_data)

    input_dim = val_data[0][0].shape[-1]
    output_dim = val_data[0][1].shape[-1]

    # Load sensor positions
    #sensor_pos = np.load(SENSOR_PATH)

    # Load models
    anp_model, mlp_model = load_models(input_dim, output_dim)

    # Plot
    save_dir = "/home/fernando/tesis/underwater-localization/results/evaluation/trajectory_plots"
    #plot_trajectories(anp_model, mlp_model, dataset, sensor_pos, save_dir)
    plot_trajectories(anp_model, mlp_model, dataset, save_dir)

if __name__ == "__main__":
    main()
