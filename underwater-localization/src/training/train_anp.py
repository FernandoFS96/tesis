import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import pickle
import argparse
import torch as t
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.models.anp import LatentModel
from src.utils.nav_dataset import NavigationTrajectoryDataset
from src.utils.plots import plot_training_metrics

def save_all_metrics(train_loss, val_loss, train_mae, val_mae, experiment_dir):
    """Save all training and validation metrics for later analysis."""
    metrics = {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_mae': train_mae,
        'val_mae': val_mae,
    }
    with open(os.path.join(experiment_dir, 'metrics.pkl'), 'wb') as f:
        pickle.dump(metrics, f)
    print(f"Metrics saved to {os.path.join(experiment_dir, 'metrics.pkl')}")


def create_experiment_directory(base_dir='./results/anp'):
    """Create a directory to save experiment results."""
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    experiment_dir = os.path.join(base_dir, f"experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def train_anp(train_data_path, val_data_path, result_dir, batch_size=10, epochs=5000, patience=250):
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create experiment directory
    experiment_dir = create_experiment_directory(result_dir)

    # Load training and validation data
    with open(train_data_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(val_data_path, 'rb') as f:
        val_data = pickle.load(f)

    print(f'X train shape: {train_data[0][0].shape}, Y train shape: {train_data[0][1].shape}')
    print(f'X val shape: {val_data[0][0].shape}, Y val shape: {val_data[0][1].shape}')

    ############################ NUEVO #############################
    x0, y0 = train_data[0]
    input_dim = x0.shape[-1]     # p.ej. 4 sensores
    output_dim = y0.shape[-1]    # p.ej. 2 coordenadas (x,y)

    # Create datasets
    train_dataset = NavigationTrajectoryDataset(train_data)
    val_dataset = NavigationTrajectoryDataset(val_data)

    input_dim = train_data[0][0].shape[-1]
    model = LatentModel(num_hidden=128, input_dim=input_dim,output_dim=output_dim).to(device)
    optimizer = t.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    global_step = 0
    best_val_mae = float('inf')
    early_stop_counter = 0

    train_loss_list, val_loss_list = [], []
    train_mae_list, val_mae_list = [], []

    t_init = time.time()

    for epoch in range(epochs):
        t_epoch = time.time()
        pbar = tqdm(total=len(train_loader) + len(val_loader), desc=f"Epoch {epoch + 1}/{epochs}")

        # Training phase
        model.train()
        train_loss, train_mae = 0.0, 0.0
        for x_batch, y_batch in train_loader:
            global_step += 1

            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # Dynamic context size
            total_points = x_batch.size(1)
            context_size = t.randint(5, 50, (1,)).item()  # 5% to 50%
            context_indices = t.arange(context_size)
            target_indices = t.arange(total_points)

            context_x = x_batch[:, context_indices, :]
            context_y = y_batch[:, context_indices, :]
            target_x = x_batch[:, target_indices, :]
            target_y = y_batch[:, target_indices, :]

            # Forward pass
            y_pred_mean, _, loss, kl, nll = model(context_x, context_y, target_x, target_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mae = F.l1_loss(y_pred_mean, target_y, reduction='mean').item()
            train_loss += loss.item()
            train_mae += mae
            pbar.update(1)

        train_loss /= len(train_loader)
        train_mae /= len(train_loader)
        train_loss_list.append(train_loss)
        train_mae_list.append(train_mae)

        # Validation phase
        model.eval()
        val_loss, val_mae = 0.0, 0.0
        with t.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                context_size = t.randint(5, 45, (1,)).item()
                context_indices = t.arange(context_size)
                target_indices = t.arange(x_batch.size(1))

                context_x = x_batch[:, context_indices, :]
                context_y = y_batch[:, context_indices, :]
                target_x = x_batch[:, target_indices, :]
                target_y = y_batch[:, target_indices, :]

                y_pred_mean, _, loss, _, _ = model(context_x, context_y, target_x, target_y)
                mae = F.l1_loss(y_pred_mean, target_y, reduction='mean').item()
                val_loss += loss.item()
                val_mae += mae
                pbar.update(1)

        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        val_loss_list.append(val_loss)
        val_mae_list.append(val_mae)

        # Early stopping
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            early_stop_counter = 0
            t.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
                   os.path.join(experiment_dir, 'best_checkpoint.pth.tar'))
            print(f"Best model saved with val MAE: {best_val_mae:.4f}")
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break
        
        pbar.set_postfix({
            'loss': train_loss,
            'mae': train_mae,
            'val_loss': val_loss,
            'val_mae': val_mae,
            'e_s': early_stop_counter,
            'epoch_t': time.time() - t_epoch,
            'total_t': time.time() - t_init
        })

        pbar.close()

    # Save final model and metrics
    t.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
           os.path.join(experiment_dir, 'last_checkpoint.pth.tar'))
    save_all_metrics(train_loss_list, val_loss_list, train_mae_list, val_mae_list, experiment_dir)

    # Plot training metrics
    metrics_file = os.path.join(experiment_dir, 'metrics.pkl')
    output_plot  = os.path.join(experiment_dir, 'training_curves.png')
    plot_training_metrics(metrics_file, output_plot)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ANP on specified data.")
    parser.add_argument('--train-data', type=str, required=True, help="Path to preprocessed training data (pickle).")
    parser.add_argument('--val-data', type=str, required=True, help="Path to preprocessed validation data (pickle).")
    parser.add_argument('--result-dir', type=str, required=True, help="Directory to save experiment results.")
    parser.add_argument('--batch-size', type=int, default=10, help="Batch size for training.")
    parser.add_argument('--epochs', type=int, default=5000, help="Number of training epochs.")
    parser.add_argument('--patience', type=int, default=250, help="Early stopping patience.")
    args = parser.parse_args()

    train_anp(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        result_dir=args.result_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience
    )
