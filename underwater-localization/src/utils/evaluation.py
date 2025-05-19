# src/utils/evaluation.py
import torch
import numpy as np
from src.preprocessing.preprocess_data import load_and_preprocess_data

def evaluate_model_on_data(model, X, y):
    """
    Evaluate the model on the given dataset and compute Mean Absolute Error (MAE).

    :param model: Trained PyTorch model
    :param X: Input features (torch.Tensor)
    :param y: Ground truth targets (torch.Tensor)
    :return: MAE value
    """
    model.eval()
    with torch.no_grad():
        predictions = model(X)
        mae = torch.mean(torch.abs(predictions - y)).item()
    return mae

def save_mae_mlp(save_path, mae):
    """
    Save the Mean Absolute Error (MAE) value to a text file.

    :param save_path: File path to save the MAE value
    :param mae: MAE value to save
    """
    with open(save_path, 'w') as f:
        f.write(f"Mean Absolute Error (MAE): {mae:.6f}\n")
    print(f"MAE saved to {save_path}")

def load_model(model_class, model_path, input_dim, output_dim, neurons):
    """
    Load a pre-trained model from the specified path.

    :param model_class: The class of the model to be loaded (e.g., MLP)
    :param model_path: Path to the saved model checkpoint
    :param input_dim: Input dimension for the model
    :param output_dim: Output dimension for the model
    :param neurons: Number of neurons in hidden layers
    :return: Loaded PyTorch model
    """
    model = model_class(input_dim, output_dim, neurons)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Model loaded from {model_path}")
    return model

def compute_mae_matrix(model, channel_options, data_dir, data_type, device):
    """
    Compute the MAE matrix for the model evaluated across all channel options.

    :param model: Trained PyTorch model
    :param channel_options: List of channel options to evaluate
    :param data_dir: Directory containing the data
    :param data_type: Type of data to load (e.g., 'filtered')
    :param device: PyTorch device (e.g., 'cuda' or 'cpu')
    :return: Numpy matrix of MAE values
    """
    mae_matrix = []
    for channel_option in channel_options:
        X, y = load_and_preprocess_data(data_dir, channel_option, data_type)
        if X is None or y is None:
            mae_matrix.append([np.nan])
            continue

        X = torch.tensor(X, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.float32).to(device)

        mae = evaluate_model_on_data(model, X, y)
        mae_matrix.append([mae])

    return np.array(mae_matrix)
