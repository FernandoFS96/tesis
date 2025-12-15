import numpy as np
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle
from sklearn.utils import shuffle
import re

# Function to reshape input data
def reshape_input_data(data):
    num_time_points, num_points_per_traj, num_trajs, num_sensors = data.shape
    X = data.transpose(2, 1, 0, 3).reshape(num_trajs, num_points_per_traj, num_time_points * num_sensors)
    #X = data.transpose(2, 1, 0, 3).reshape(num_trajs * num_points_per_traj, num_time_points * num_sensors)
    return X

# Function to reshape output data
def reshape_output_data(trajectories):
    #num_coords, num_trajs, num_points_per_traj = trajectories.shape
    y = trajectories.transpose(1, 2, 0)
    return y

# Function to load, reshape, split, and save the data
def process_and_save_data(input_paths, output_paths, save_dir, split = 0.2):
    loaded_train_data_mlp = []
    loaded_val_data_mlp = []
    train_thetas_mlp = []
    val_thetas_mlp = []

    loaded_train_data_anp = []
    loaded_val_data_anp = []
    train_thetas_anp = []
    val_thetas_anp = []

    # Load, reshape, and split data
    print(f"Loading, reshaping, and splitting {1-split}/{split} the data...")
    for i, (input_path, output_path) in tqdm(enumerate(zip(input_paths, output_paths)), total=len(input_paths), leave=False):
        input_data = np.load(input_path)
        output_data = np.load(output_path)

        # Reshape the input and output data for MLP and ANP, note that we start by loading the ANP case
        X = reshape_input_data(input_data)
        y = reshape_output_data(output_data)
        
        # Split the reshaped data into 80% training and 20% validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=split, random_state=18, shuffle=True
        )

        # Append training and validation data separately for MLP
        X_train_mlp = X_train.reshape(X_train.shape[0] * X_train.shape[1], X_train.shape[2])
        y_train_mlp = y_train.reshape(y_train.shape[0] * y_train.shape[1], y_train.shape[2])
        X_val_mlp = X_val.reshape(X_val.shape[0] * X_val.shape[1], X_val.shape[2])
        y_val_mlp = y_val.reshape(y_val.shape[0] * y_val.shape[1], y_val.shape[2])
        # Sort before appending
        id_mlp_train = np.random.permutation(len(X_train_mlp))
        id_mlp_val = np.random.permutation(len(X_val_mlp))
        loaded_train_data_mlp.append([X_train_mlp[id_mlp_train], y_train_mlp[id_mlp_train]])
        train_thetas_mlp.append(theta_values[i])  # Corresponding theta for training

        loaded_val_data_mlp.append([X_val_mlp[id_mlp_val], y_val_mlp[id_mlp_val]])
        val_thetas_mlp.append(theta_values[i])  # Corresponding theta for validation

        # Append training and validation data separately for ANP
        for j in range(len(X_train)):
            loaded_train_data_anp.append([X_train[j], y_train[j]])
            train_thetas_anp.append(theta_values[i])  # Corresponding theta for training

        for j in range(len(X_val)):
            loaded_val_data_anp.append([X_val[j], y_val[j]])
            val_thetas_anp.append(theta_values[i])  # Corresponding theta for validation

    # Sort train data randomly for ANP, keeping the theta values in sync
    idx = np.random.permutation(len(loaded_train_data_anp))
    loaded_train_data_anp = [loaded_train_data_anp[i] for i in idx]
    train_thetas_anp = [train_thetas_anp[i] for i in idx]

    tqdm.write(" Done.")  # Use tqdm.write to avoid breaking the progress bar

    # Save the training data
    train_save_path = os.path.join(save_dir, 'train_data_mlp.pkl')
    with open(train_save_path, 'wb') as f:
        pickle.dump(loaded_train_data_mlp, f)
    print(f"Training data saved to {train_save_path}")

    # Save the validation data
    val_save_path = os.path.join(save_dir, 'val_data_mlp.pkl')
    with open(val_save_path, 'wb') as f:
        pickle.dump(loaded_val_data_mlp, f)
    print(f"Validation data saved to {val_save_path}")

    # Save the theta values separately
    theta_save_path = os.path.join(save_dir, 'theta_values_mlp.pkl')
    theta_data = {'train_thetas': train_thetas_mlp, 'val_thetas': val_thetas_mlp}
    with open(theta_save_path, 'wb') as f:
        pickle.dump(theta_data, f)
    print(f"Theta values saved to {theta_save_path}")

    # Save the training data
    train_save_path = os.path.join(save_dir, 'train_data_anp.pkl')
    with open(train_save_path, 'wb') as f:
        pickle.dump(loaded_train_data_anp, f)
    print(f"Training data saved to {train_save_path}")

    # Save the validation data
    val_save_path = os.path.join(save_dir, 'val_data_anp.pkl')
    with open(val_save_path, 'wb') as f:
        pickle.dump(loaded_val_data_anp, f)
    print(f"Validation data saved to {val_save_path}")

    # Save the theta values separately
    theta_save_path = os.path.join(save_dir, 'theta_values_anp.pkl')
    theta_data = {'train_thetas': train_thetas_anp, 'val_thetas': val_thetas_anp}
    with open(theta_save_path, 'wb') as f:
        pickle.dump(theta_data, f)
    print(f"Theta values saved to {theta_save_path}")

# Set the base directory where the data is stored
base_dir = os.path.join('/home/fernando/tesis/NP_juan/data2')

# List all channel_option directories inside the base directory
channel_dirs = sorted(
    [os.path.join(base_dir, d) for d in os.listdir(base_dir)
     if os.path.isdir(os.path.join(base_dir, d)) and re.match(r'channel_option_\d+(\.\d+)?', d)],
    key=lambda x: float(os.path.basename(x).split("channel_option_")[1])
)

# Extract theta values from the folder names in sorted order
theta_values = []
for d in channel_dirs:
    folder_name = os.path.basename(d)
    if folder_name.startswith("channel_option_"):
        value_str = folder_name.split("channel_option_")[1]
        theta_values.append(float(value_str))

# Create lists to store the file paths for input and output data
input_paths = [os.path.join(d, 'filtered_data', 'filtered_data.npy') for d in channel_dirs]
output_paths = [os.path.join(d, 'trajectory', 'trajectories.npy') for d in channel_dirs]

# Define the directory to save the processed data
processed_data_dir = os.path.join('/home/fernando/tesis/NP_juan/data2', 'data_processed')
if not os.path.exists(processed_data_dir):
    os.makedirs(processed_data_dir)

# Process and save the data with the 80/20 split
process_and_save_data(input_paths, output_paths, processed_data_dir, split = 0.2)
