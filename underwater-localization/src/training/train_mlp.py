# src/training/train_mlp.py
import os
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
from src.models.mlp import MLP
from src.preprocessing.preprocess_data import load_and_preprocess_data
from src.utils.evaluation import evaluate_model_on_data, save_mae_mlp
from src.utils.plot_curves import plot_loss_curve, plot_mae_curve
from src.utils.filter_channels import filter_channel_options

def train_mlp(data_dir, channel_options, base_result_dir, mlp_data_type='filtered', neurons=128, num_epochs=10000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(base_result_dir, exist_ok=True)
    all_X, all_y = [], []

    for channel_option in channel_options:
        print(f"Training MLP for {channel_option}...")
        mlp_model_dir = os.path.join(base_result_dir, f'MLP_model_{channel_option}')
        os.makedirs(mlp_model_dir, exist_ok=True)

        X, y = load_and_preprocess_data(data_dir, channel_option, mlp_data_type)
        if X is None or y is None:
            continue

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=18)
        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

        input_dim = X_train.shape[1]
        output_dim = y_train.shape[1]

        model = MLP(input_dim, output_dim, neurons).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()

        best_mae = float('inf')
        mse_values, mae_values = [], []

        # creamos el tqdm y lo guardamos en pbar
        pbar = tqdm(range(num_epochs),
                    desc=f"[{channel_option}] Épocas",
                    unit="época",
                    ncols=150)
        
        for epoch in pbar:
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            mse_values.append(loss.item())
            mae = evaluate_model_on_data(model, X_test, y_test)
            mae_values.append(mae)

            if mae < best_mae:
                best_mae = mae
                torch.save(model.state_dict(), os.path.join(mlp_model_dir, 'best_model.pth'))

            # actualizamos la barra con Loss y MAE formateados
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "MAE":  f"{mae:.4f}"
            })

        plot_loss_curve(mse_values, os.path.join(mlp_model_dir, 'loss_curve.png'))
        plot_mae_curve(mae_values, os.path.join(mlp_model_dir, 'mae_curve.png'))
        save_mae_mlp(os.path.join(mlp_model_dir, 'mlp_mae_results.txt'), best_mae)

    print("Training on combined data...")
    for channel_option in channel_options:
        X, y = load_and_preprocess_data(data_dir, channel_option, mlp_data_type)
        if X is not None and y is not None:
            all_X.append(X)
            all_y.append(y)

    all_X = np.vstack(all_X)
    all_y = np.vstack(all_y)
    X_train, X_test, y_train, y_test = train_test_split(all_X, all_y, test_size=0.3, random_state=18)
    num_epochs = 15000
    train_combined_model(X_train, y_train, X_test, y_test, base_result_dir, neurons, num_epochs)

def train_combined_model(X_train, y_train, X_test, y_test, base_result_dir, neurons, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    model = MLP(input_dim, output_dim, neurons).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    best_mae = float('inf')
    mse_values, mae_values = [], []

    pbar = tqdm(range(num_epochs),
                desc="Combined Model",
                unit="época",
                ncols=150)
    
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        mse_values.append(loss.item())
        mae = evaluate_model_on_data(model, X_test, y_test)
        mae_values.append(mae)

        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), os.path.join(base_result_dir, 'combined_model.pth'))

        # actualizar barra
        pbar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "MAE":  f"{mae:.4f}"
        })
    plot_loss_curve(mse_values, os.path.join(base_result_dir, 'combined_loss_curve.png'))
    plot_mae_curve(mae_values, os.path.join(base_result_dir, 'combined_mae_curve.png'))
    save_mae_mlp(os.path.join(base_result_dir, 'combined_mae_results.txt'), best_mae)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP on specified data directory.")
    parser.add_argument('--data-dir', type=str, required=True, help="Path to the data directory.")
    parser.add_argument('--result-dir', type=str, default='./results/mlp', help="Path to save results.")
    parser.add_argument('--theta-range', type=float, nargs=2, required=True, help="Theta range for training (e.g., 0.0 0.5).")
    parser.add_argument('--input-dim', type=int, required=True, help="Input dimension for the MLP model.")
    parser.add_argument('--output-dim', type=int, required=True, help="Output dimension for the MLP model.")
    args = parser.parse_args()

    all_channel_options = [f'channel_option_{i/10:.1f}' for i in range(11)]

    # Filter channel options based on the theta range
    channel_options = filter_channel_options(all_channel_options, tuple(args.theta_range))

    train_mlp(args.data_dir, channel_options, args.result_dir)
