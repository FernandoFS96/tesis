import numpy as np
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle
import re

# Function to reshape input data
# data: (num_time_points, num_points_per_traj, num_trajs, num_sensors)
def reshape_input_data(data):
    num_time_points, num_points_per_traj, num_trajs, num_sensors = data.shape
    # shape -> (num_trajs, num_points_per_traj, num_time_points * num_sensors)
    X = data.transpose(2, 1, 0, 3).reshape(
        num_trajs,
        num_points_per_traj,
        num_time_points * num_sensors
    )
    return X

# Function to reshape output data (trajectories)
# trajectories: (3, num_trajs, num_points_per_traj)
def reshape_output_data(trajectories):
    # correct transpose for shape (3, n_traj, ppt) -> (n_traj, ppt, 3)
    y = trajectories.transpose(1, 2, 0)
    return y

# Load, reshape, split, and save MLP and ANP datasets
def process_and_save_data(input_paths, output_paths, save_dir, theta_values, split=0.2):
    loaded_train_data_mlp = []
    loaded_val_data_mlp   = []
    train_thetas_mlp      = []
    val_thetas_mlp        = []

    loaded_train_data_anp = []
    loaded_val_data_anp   = []
    train_thetas_anp      = []
    val_thetas_anp        = []

    print(f"Loading, reshaping, and splitting {1-split}/{split} the data...")
    for i, (input_path, output_path) in tqdm(
        enumerate(zip(input_paths, output_paths)),
        total=len(input_paths), leave=False
    ):
        # Load raw npy files
        input_data  = np.load(input_path)
        output_data = np.load(output_path)

        # Reshape
        X = reshape_input_data(input_data)
        y = reshape_output_data(output_data)

        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=split, random_state=18, shuffle=True
        )

        # ----- MLP preparation -----
        X_train_mlp = X_train.reshape(-1, X_train.shape[2])
        y_train_mlp = y_train.reshape(-1, y_train.shape[2])
        X_val_mlp   = X_val.reshape(-1, X_val.shape[2])
        y_val_mlp   = y_val.reshape(-1, y_val.shape[2])

        id_mlp_train = np.random.permutation(len(X_train_mlp))
        id_mlp_val   = np.random.permutation(len(X_val_mlp))

        loaded_train_data_mlp.append([X_train_mlp[id_mlp_train], y_train_mlp[id_mlp_train]])
        train_thetas_mlp.append(theta_values[i])
        loaded_val_data_mlp.append([X_val_mlp[id_mlp_val], y_val_mlp[id_mlp_val]])
        val_thetas_mlp.append(theta_values[i])

        # ----- ANP preparation -----
        for j in range(X_train.shape[0]):
            loaded_train_data_anp.append([X_train[j], y_train[j]])
            train_thetas_anp.append(theta_values[i])
        for j in range(X_val.shape[0]):
            loaded_val_data_anp.append([X_val[j], y_val[j]])
            val_thetas_anp.append(theta_values[i])

    # Shuffle ANP training
    idx = np.random.permutation(len(loaded_train_data_anp))
    loaded_train_data_anp = [loaded_train_data_anp[i] for i in idx]
    train_thetas_anp      = [train_thetas_anp[i]      for i in idx]

    tqdm.write(" Done.")

    os.makedirs(save_dir, exist_ok=True)

    # Save MLP datasets
    with open(os.path.join(save_dir, 'train_data_mlp.pkl'), 'wb') as f:
        pickle.dump(loaded_train_data_mlp, f)
    print(f"Training data saved to {os.path.join(save_dir, 'train_data_mlp.pkl')}")

    with open(os.path.join(save_dir, 'val_data_mlp.pkl'), 'wb') as f:
        pickle.dump(loaded_val_data_mlp, f)
    print(f"Validation data saved to {os.path.join(save_dir, 'val_data_mlp.pkl')}")

    with open(os.path.join(save_dir, 'theta_values_mlp.pkl'), 'wb') as f:
        pickle.dump({'train_thetas': train_thetas_mlp,
                     'val_thetas':   val_thetas_mlp}, f)
    print(f"Theta values saved to {os.path.join(save_dir, 'theta_values_mlp.pkl')}")

    # Save ANP datasets
    with open(os.path.join(save_dir, 'train_data_anp.pkl'), 'wb') as f:
        pickle.dump(loaded_train_data_anp, f)
    print(f"Training data saved to {os.path.join(save_dir, 'train_data_anp.pkl')}")

    with open(os.path.join(save_dir, 'val_data_anp.pkl'), 'wb') as f:
        pickle.dump(loaded_val_data_anp, f)
    print(f"Validation data saved to {os.path.join(save_dir, 'val_data_anp.pkl')}")

    with open(os.path.join(save_dir, 'theta_values_anp.pkl'), 'wb') as f:
        pickle.dump({'train_thetas': train_thetas_anp,
                     'val_thetas':   val_thetas_anp}, f)
    print(f"Theta values saved to {os.path.join(save_dir, 'theta_values_anp.pkl')}")

if __name__ == '__main__':
    base_dir = '/home/fernando/tesis/testeo_data_ANP/data'

    channel_dirs = sorted(
        [os.path.join(base_dir, d) for d in os.listdir(base_dir)
         if os.path.isdir(os.path.join(base_dir, d))
            and re.match(r'channel_option_\d+(\.\d+)?', d)],
        key=lambda x: float(os.path.basename(x).split('channel_option_')[1])
    )

    theta_values = [
        float(os.path.basename(d).split('channel_option_')[1])
        for d in channel_dirs
    ]

    input_paths  = [os.path.join(d, 'filtered_data',   'filtered_data.npy') for d in channel_dirs]
    output_paths = [os.path.join(d, 'trajectory',      'trajectories.npy')     for d in channel_dirs]

    processed_data_dir = os.path.join(base_dir, 'data_processed')

    process_and_save_data(
        input_paths,
        output_paths,
        processed_data_dir,
        theta_values,
        split=0.2
    )
