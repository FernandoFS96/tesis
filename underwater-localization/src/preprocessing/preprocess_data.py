import os
import numpy as np

def load_and_preprocess_data(base_dir, channel_option, data_type):
    """
    Load and preprocess data from the specified base directory and channel option.

    :param base_dir: Base directory containing data
    :param channel_option: Subfolder denoting channel option (e.g., channel_option_0.1)
    :param data_type: Type of data to load (filtered, power, covariance)
    :return: Processed X (inputs) and y (targets)
    """
    data_path = os.path.join(base_dir, channel_option, f'{data_type}_data', f'{data_type}_data.npy')
    trajectory_data_path = os.path.join(base_dir, channel_option, 'trajectory', 'trajectories.npy')

    if not os.path.exists(data_path) or not os.path.exists(trajectory_data_path):
        print(f"Data not found for {channel_option}, skipping...")
        return None, None

    data = np.load(data_path)
    trajectory_data = np.load(trajectory_data_path)

    if data_type == 'filtered':
        num_time_points, num_points_per_traj, num_trajs, num_sensors = data.shape
        X = data.transpose(2, 1, 0, 3).reshape(num_trajs * num_points_per_traj, num_time_points * num_sensors)
    elif data_type == 'power':
        num_points_per_traj, num_trajs, num_sensors = data.shape
        X = data.transpose(1, 0, 2).reshape(num_trajs * num_points_per_traj, num_sensors)
    elif data_type == 'covariance':
        num_points_per_traj, num_trajs, cov_features = data.shape
        X = data.transpose(1, 0, 2).reshape(num_trajs * num_points_per_traj, cov_features)

    num_coords, num_trajs, num_points_per_traj = trajectory_data.shape
    y = trajectory_data.transpose(1, 2, 0).reshape(num_trajs * num_points_per_traj, num_coords)

    return X, y