import numpy as np
import os
import re
import argparse
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Function to reshape input data
def reshape_input_data(data):
    num_time_points, num_points_per_traj, num_trajs, num_sensors = data.shape
    # shape -> (num_trajs, num_points_per_traj, num_time_points * num_sensors)
    return data.transpose(2, 1, 0, 3).reshape(
        num_trajs,
        num_points_per_traj,
        num_time_points * num_sensors
    )

# Function to reshape output trajectories
def reshape_output_data(trajectories):
    # from (3, num_trajs, num_points_per_traj) -> (num_trajs, num_points_per_traj, 3)
    return trajectories.transpose(1, 2, 0)

# Core processing: load, reshape, split and save for MLP and ANP
def process_and_save_data(input_paths, output_paths, save_dir, theta_values, split=0.2):
    loaded_train_data_mlp, loaded_val_data_mlp = [], []
    train_thetas_mlp, val_thetas_mlp = [], []
    loaded_train_data_anp, loaded_val_data_anp = [], []
    train_thetas_anp, val_thetas_anp = [], []

    print(f"Loading, reshaping and splitting data with split={split}...")
    for i, (inp, out) in tqdm(enumerate(zip(input_paths, output_paths)), total=len(input_paths), leave=False):
        X = reshape_input_data(np.load(inp))
        y = reshape_output_data(np.load(out))
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=split, random_state=18, shuffle=True
        )

        # MLP: flatten sequences
        X_tr_mlp = X_train.reshape(-1, X_train.shape[2])
        y_tr_mlp = y_train.reshape(-1, y_train.shape[2])
        X_va_mlp = X_val.reshape(-1, X_val.shape[2])
        y_va_mlp = y_val.reshape(-1, y_val.shape[2])
        perm_tr = np.random.permutation(len(X_tr_mlp))
        perm_va = np.random.permutation(len(X_va_mlp))
        loaded_train_data_mlp.append([X_tr_mlp[perm_tr], y_tr_mlp[perm_tr]])
        loaded_val_data_mlp.append([X_va_mlp[perm_va], y_va_mlp[perm_va]])
        train_thetas_mlp.append(theta_values[i])
        val_thetas_mlp.append(theta_values[i])

        # ANP: keep sequences intact
        for j in range(X_train.shape[0]):
            loaded_train_data_anp.append([X_train[j], y_train[j]])
            train_thetas_anp.append(theta_values[i])
        for j in range(X_val.shape[0]):
            loaded_val_data_anp.append([X_val[j], y_val[j]])
            val_thetas_anp.append(theta_values[i])

    # Shuffle ANP training set
    idx = np.random.permutation(len(loaded_train_data_anp))
    loaded_train_data_anp = [loaded_train_data_anp[i] for i in idx]
    train_thetas_anp = [train_thetas_anp[i] for i in idx]

    os.makedirs(save_dir, exist_ok=True)

    # Save to pickle
    with open(os.path.join(save_dir, 'train_data_mlp.pkl'), 'wb') as f:
        pickle.dump(loaded_train_data_mlp, f)
    with open(os.path.join(save_dir, 'val_data_mlp.pkl'), 'wb') as f:
        pickle.dump(loaded_val_data_mlp, f)
    with open(os.path.join(save_dir, 'theta_values_mlp.pkl'), 'wb') as f:
        pickle.dump({'train_thetas': train_thetas_mlp, 'val_thetas': val_thetas_mlp}, f)

    with open(os.path.join(save_dir, 'train_data_anp.pkl'), 'wb') as f:
        pickle.dump(loaded_train_data_anp, f)
    with open(os.path.join(save_dir, 'val_data_anp.pkl'), 'wb') as f:
        pickle.dump(loaded_val_data_anp, f)
    with open(os.path.join(save_dir, 'theta_values_anp.pkl'), 'wb') as f:
        pickle.dump({'train_thetas': train_thetas_anp, 'val_thetas': val_thetas_anp}, f)

    print(f"Datos procesados guardados en: {save_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Procesa .npy de data/high_variance o data/low_variance'
    )
    parser.add_argument(
        'subset', choices=['high_variance', 'low_variance'],
        help='Subcarpeta dentro de data a procesar'
    )
    parser.add_argument(
        '--split', type=float, default=0.2,
        help='Fracción para validación (por defecto 0.2)'
    )
    parser.add_argument(
        '--save-dir', default=None,
        help='Directorio de salida (por defecto data/<subset>/data_processed)'
    )
    args = parser.parse_args()

    base = './data'
    subset_dir = os.path.join(base, args.subset)
    if not os.path.isdir(subset_dir):
        raise FileNotFoundError(f"No existe {subset_dir}")

    # Detectar canales dentro de la subcarpeta seleccionada
    channel_dirs = sorted(
        [os.path.join(subset_dir, d) for d in os.listdir(subset_dir)
         if os.path.isdir(os.path.join(subset_dir, d)) and re.match(r'channel_option_[0-9.]+', d)],
        key=lambda x: float(os.path.basename(x).split('channel_option_')[1])
    )
    theta_values = [
        float(os.path.basename(d).split('channel_option_')[1])
        for d in channel_dirs
    ]
    input_paths = [os.path.join(d, 'filtered_data', 'filtered_data.npy') for d in channel_dirs]
    output_paths = [os.path.join(d, 'trajectory', 'trajectories.npy') for d in channel_dirs]

    processed = args.save_dir or os.path.join(subset_dir, 'data_processed')
    process_and_save_data(
        input_paths,
        output_paths,
        processed,
        theta_values,
        split=args.split
    )
