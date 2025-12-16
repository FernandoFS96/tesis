import sys
import os
# Ensure we import the local src package instead of similarly named siblings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.models.anp import LatentModel, DistributedLatentModel
from src.utils.nav_dataset import NavigationTrajectoryDataset
from src.utils.plots import plot_training_metrics

'''
Use:
# Train one ANP per topology using all theta values
python train_anp_topologies_distributed.py \
    --data-dir /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_low_variance \
    --batch-size 8 \
    --epochs 10000 \
    --patience 1000
'''

def load_topology_data(data_dir, topology):
    """Load all processed data for a specific topology"""
    topology_dir = os.path.join(data_dir, f'topology_{topology}')
    
    # Load train and validation data
    train_path = os.path.join(topology_dir, 'train_data.pkl')
    val_path = os.path.join(topology_dir, 'val_data.pkl')
    metadata_path = os.path.join(topology_dir, 'metadata.pkl')
    
    if not all(os.path.exists(p) for p in [train_path, val_path, metadata_path]):
        print(f"Warning: Missing data files for topology {topology}")
        return None, None, None
    
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    
    with open(val_path, 'rb') as f:
        val_data = pickle.load(f)
    
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    # Basic consistency checks
    assert len(train_data) > 0, f"Empty train_data for topology {topology}"
    assert len(val_data) > 0, f"Empty val_data for topology {topology}"

    # Check one example to infer shapes
    x0, y0 = train_data[0]
    assert x0.ndim == 2 and y0.ndim == 2, \
        f"Expected X and Y to be 2D (T, D), got {x0.ndim}D and {y0.ndim}D"
    assert x0.shape[0] == y0.shape[0], \
        f"Time dimension mismatch between X and Y: {x0.shape[0]} vs {y0.shape[0]}"
    
    return train_data, val_data, metadata


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


def train_anp_topology(train_data, val_data, save_dir, topology_name, 
                       batch_size=8, epochs=5000, patience=200, device='cuda'):
    """Train a single ANP model for a topology using all theta values"""
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nTraining ANP for topology: {topology_name}")
    print(f"  Training set size: {len(train_data)} trajectories")
    print(f"  Validation set size: {len(val_data)} trajectories")
    print(f'  X shape: {train_data[0][0].shape}, Y shape: {train_data[0][1].shape}')

    # Get dimensions from data
    x0, y0 = train_data[0]
    input_dim = x0.shape[-1]     # e.g., 10 sensors * features
    output_dim = y0.shape[-1]    # e.g., 3 coordinates (x,y,z)

    # Sanity checks for distributed setup
    expected_input_dim = args.n_sensors * args.sensor_feature_dim
    assert input_dim == expected_input_dim, (
        f"Expected input_dim={expected_input_dim} "
        f"(n_sensors={args.n_sensors} * sensor_feature_dim={args.sensor_feature_dim}), "
        f"but got {input_dim}. Check your n_sensors / sensor_feature_dim."
    )
    assert output_dim == 3, f"Expected output_dim=3 (x,y,z), got {output_dim}"

    assert args.sensor_emb_dim > 0, "sensor_emb_dim must be > 0"

    # Create datasets
    train_dataset = NavigationTrajectoryDataset(train_data)
    val_dataset = NavigationTrajectoryDataset(val_data)

    # Initialize model
    base_anp  = LatentModel(num_hidden=128, input_dim=args.sensor_emb_dim, output_dim=output_dim).to(device)
    model = DistributedLatentModel(base_anp=base_anp,
                                   n_sensors=args.n_sensors,
                                   in_dim_per_sensor=args.sensor_feature_dim,
                                   emb_dim=args.sensor_emb_dim,
                                   fusion="mean",   # or "max"
                                   ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    best_val_mae = float('inf')
    early_stop_counter = 0

    train_loss_list, val_loss_list = [], []
    train_mae_list, val_mae_list = [], []

    t_init = time.time()

    # Training loop with progress bar
    pbar = tqdm(range(epochs), desc=f"[ANP-{topology_name}]", unit="epoch", ncols=150)
    
    for epoch in pbar:
        # Training phase
        model.train()
        train_loss, train_mae = 0.0, 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # Shape checks
            assert x_batch.ndim == 3, f"x_batch should be (B, T, D), got shape {x_batch.shape}"
            assert y_batch.ndim == 3, f"y_batch should be (B, T, D_y), got shape {y_batch.shape}"
            assert x_batch.size(1) == y_batch.size(1), (
                f"Time dimension mismatch in batch: {x_batch.size(1)} vs {y_batch.size(1)}"
            )
            assert x_batch.size(2) == input_dim, (
                f"Input feature dim mismatch: expected {input_dim}, got {x_batch.size(2)}"
            )
            assert y_batch.size(2) == output_dim, (
                f"Output feature dim mismatch: expected {output_dim}, got {y_batch.size(2)}"
            )

            # Dynamic context size (5% to 95% of points)
            total_points = x_batch.size(1)
            min_context = max(1, int(0.05 * total_points))
            max_context = min(int(0.95 * total_points), total_points - 1)
            
            if max_context > min_context:
                context_size = torch.randint(min_context, max_context, (1,)).item()
            else:
                context_size = min_context
            
            context_indices = torch.arange(context_size)
            target_indices = torch.arange(total_points)

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

        train_loss /= len(train_loader)
        train_mae /= len(train_loader)
        train_loss_list.append(train_loss)
        train_mae_list.append(train_mae)

        # Validation phase
        model.eval()
        val_loss, val_mae = 0.0, 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                total_points = x_batch.size(1)
                min_context = max(1, int(0.05 * total_points))
                max_context = min(int(0.90 * total_points), total_points - 1)
                
                if max_context > min_context:
                    context_size = torch.randint(min_context, max_context, (1,)).item()
                else:
                    context_size = min_context
                
                context_indices = torch.arange(context_size)
                target_indices = torch.arange(x_batch.size(1))

                context_x = x_batch[:, context_indices, :]
                context_y = y_batch[:, context_indices, :]
                target_x = x_batch[:, target_indices, :]
                target_y = y_batch[:, target_indices, :]

                y_pred_mean, _, loss, _, _ = model(context_x, context_y, target_x, target_y)
                mae = F.l1_loss(y_pred_mean, target_y, reduction='mean').item()
                val_loss += loss.item()
                val_mae += mae

        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        val_loss_list.append(val_loss)
        val_mae_list.append(val_mae)

        # Early stopping
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            early_stop_counter = 0
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
                   os.path.join(save_dir, 'best_checkpoint.pth.tar'))
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f"{train_loss:.4f}",
            'MAE': f"{train_mae:.4f}",
            'Val MAE': f"{val_mae:.4f}",
            'Best': f"{best_val_mae:.4f}",
            'ES': f"{early_stop_counter}"
        })

    # Save final model and metrics
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
           os.path.join(save_dir, 'last_checkpoint.pth.tar'))
    save_all_metrics(train_loss_list, val_loss_list, train_mae_list, val_mae_list, save_dir)

    # Plot training metrics
    metrics_file = os.path.join(save_dir, 'metrics.pkl')
    output_plot = os.path.join(save_dir, 'training_curves.png')
    plot_training_metrics(metrics_file, output_plot)
    
    # Save summary
    with open(os.path.join(save_dir, 'training_summary.txt'), 'w') as f:
        f.write(f"ANP Training Summary - Topology: {topology_name}\n")
        f.write("="*50 + "\n")
        f.write(f"Training samples: {len(train_data)} trajectories\n")
        f.write(f"Validation samples: {len(val_data)} trajectories\n")
        f.write(f"Best validation MAE: {best_val_mae:.6f}\n")
        f.write(f"Final epoch: {min(epoch+1, epochs)}\n")
        f.write(f"Early stopping counter: {early_stop_counter}/{patience}\n")
        f.write(f"Training time: {(time.time() - t_init)/60:.2f} minutes\n")
    
    print(f"  Best validation MAE: {best_val_mae:.6f}")
    
    return best_val_mae


def create_comparison_plot(results, save_dir):
    """Create comparison plot for the three topologies"""
    comparison_dir = os.path.join(save_dir, 'topology_comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    topologies = list(results.keys())
    maes = [results[t] for t in topologies]
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    bars = ax.bar(topologies, maes, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, mae in zip(bars, maes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mae:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Sensor Topology', fontsize=14)
    ax.set_ylabel('Validation MAE', fontsize=14)
    ax.set_title('ANP Performance Across Sensor Topologies', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, max(maes) * 1.15])
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'anp_topology_comparison.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    # Save comparison summary
    summary_path = os.path.join(comparison_dir, 'comparison_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("ANP Performance Comparison Across Topologies\n")
        f.write("="*60 + "\n\n")
        f.write("Validation MAE Results:\n")
        f.write("-"*30 + "\n")
        
        # Sort by performance
        sorted_results = sorted(results.items(), key=lambda x: x[1])
        for i, (topology, mae) in enumerate(sorted_results, 1):
            f.write(f"{i}. {topology:<15}: {mae:.6f}\n")
        
        f.write("\n" + "-"*30 + "\n")
        f.write(f"Best topology: {sorted_results[0][0]} (MAE: {sorted_results[0][1]:.6f})\n")
        f.write(f"Performance difference: {sorted_results[-1][1] - sorted_results[0][1]:.6f}\n")
        f.write(f"Relative improvement: {((sorted_results[-1][1] - sorted_results[0][1]) / sorted_results[-1][1] * 100):.2f}%\n")
    
    print(f"\nComparison plot saved to: {comparison_dir}")


def main(args):
    """Main training function"""
    
    print(f"\n{'='*60}")
    print("ANP Training - One Model Per Topology")
    print(f"{'='*60}")
    print(f"Data directory: {args.data_dir}")
    print(f"Results directory: {args.result_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Patience: {args.patience}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Define topologies to train
    topologies = ['ellipsoidal','random','aligned']
    
    # Store results for comparison
    results = {}
    
    # Train one model per topology
    for topology in topologies:
        print(f"\n{'='*40}")
        print(f"Processing topology: {topology}")
        print(f"{'='*40}")
        
        # Load data for this topology
        train_data, val_data, metadata = load_topology_data(args.data_dir, topology)
        
        if train_data is None:
            print(f"Skipping topology {topology} - no data found")
            continue
        
        # Get theta distribution info
        unique_thetas = sorted(list(set(metadata['train_thetas'])))
        print(f"Theta values included: {unique_thetas}")
        print(f"Theta distribution in training set:")
        for theta in unique_thetas:
            count = metadata['train_thetas'].count(theta)
            print(f"  θ={theta:.1f}: {count} trajectories")
        
        # Create save directory for this topology
        topology_dir = os.path.join(args.result_dir, f'ANP_{topology}')
        
        # Train the model
        best_mae = train_anp_topology(
            train_data, val_data, 
            topology_dir, topology,
            args.batch_size, args.epochs, args.patience,
            device
        )
        
        results[topology] = best_mae
    
    # Create comparison plot if we have results for all topologies
    if len(results) == len(topologies):
        create_comparison_plot(results, args.result_dir)
        
        print(f"\n{'='*60}")
        print("Training Complete - Summary")
        print(f"{'='*60}")
        for topology, mae in results.items():
            print(f"  {topology:<15}: MAE = {mae:.6f}")
        print(f"\nAll results saved to: {args.result_dir}")
    else:
        print(f"\nWarning: Only {len(results)}/{len(topologies)} topologies were trained")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train one ANP model per sensor topology")
    parser.add_argument('--data-dir',type=str, required=True,
                        help="Path to processed data directory (e.g., data/data_processed_topologies_low_variance)")
    parser.add_argument('--result-dir',type=str, 
                        default='./results/ANP_topologies_distributed/low_variance',
                        help="Path to save results")
    parser.add_argument('--batch-size',type=int,
                        default=8,
                        help="Batch size for training")
    parser.add_argument('--epochs',type=int,
                        default=5000,
                        help="Number of training epochs")
    parser.add_argument('--patience',type=int,
                        default=200,
                        help="Early stopping patience")
    parser.add_argument("--sensor_emb_dim", type=int, 
                        default=64,
                        help="Dimensionality of the per-time-step fused sensor embedding sent to the ANP.")
    parser.add_argument("--n_sensors", type=int, 
                        default=10,
                        help="Number of sensors (used to reshape 4010 → 10x401).")
    parser.add_argument("--sensor_feature_dim", type=int, 
                        default=401,
                        help="Feature dimension per sensor.")

    args = parser.parse_args()
    main(args)