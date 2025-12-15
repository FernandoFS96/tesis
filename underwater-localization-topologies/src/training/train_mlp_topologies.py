import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.utils.nav_dataset import NavigationTrajectoryDataset
from src.utils.plots import plot_training_metrics

'''
Use:
# Train MLP specialists per theta and topology
python train_mlp_topologies.py \
    --data-dir /home/fernando/tesis/underwater-localization/data/data/data_processed_topologies_low_variance \
    --result-dir /home/fernando/tesis/underwater-localization/results/MLP_topologies/low_variance \
    --batch-size 128 \
    --epochs 10000 \
    --patience 250
'''


# ============================================================================
# MLP Architecture
# ============================================================================
class MLPSpecialist(nn.Module):
    """Simple MLP for baseline comparison - processes full sequences"""
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 256, 128]):
        super(MLPSpecialist, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            predictions: (batch_size, seq_len, output_dim)
        """
        batch_size, seq_len, input_dim = x.size()
        
        # Process each point in the sequence independently
        x_flat = x.view(-1, input_dim)  # (batch_size * seq_len, input_dim)
        out_flat = self.network(x_flat)  # (batch_size * seq_len, output_dim)
        out = out_flat.view(batch_size, seq_len, -1)  # (batch_size, seq_len, output_dim)
        
        return out


# ============================================================================
# Data Loading
# ============================================================================
def load_topology_data_by_theta(data_dir, topology, target_theta):
    """Load data for a specific topology and filter by theta value"""
    topology_dir = os.path.join(data_dir, f'topology_{topology}')
    
    train_path = os.path.join(topology_dir, 'train_data.pkl')
    val_path = os.path.join(topology_dir, 'val_data.pkl')
    metadata_path = os.path.join(topology_dir, 'metadata.pkl')
    
    if not all(os.path.exists(p) for p in [train_path, val_path, metadata_path]):
        print(f"Warning: Missing data files for topology {topology}")
        return None, None, None
    
    with open(train_path, 'rb') as f:
        train_data_all = pickle.load(f)
    with open(val_path, 'rb') as f:
        val_data_all = pickle.load(f)
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    # Filter data by theta
    train_data = []
    for i, theta in enumerate(metadata['train_thetas']):
        if abs(theta - target_theta) < 1e-6:  # Float comparison
            train_data.append(train_data_all[i])
    
    val_data = []
    for i, theta in enumerate(metadata['val_thetas']):
        if abs(theta - target_theta) < 1e-6:
            val_data.append(val_data_all[i])
    
    return train_data, val_data, metadata

def load_topology_data_all_thetas(data_dir, topology):
    """Load ALL data for a specific topology (all theta values combined)"""
    topology_dir = os.path.join(data_dir, f'topology_{topology}')
    
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
    
    return train_data, val_data, metadata

# ============================================================================
# Training Function
# ============================================================================
def train_mlp_specialist(train_data, val_data, save_dir, topology_name, theta_value,
                        batch_size=32, epochs=5000, patience=200, device='cuda'):
    """Train a single MLP specialist for one theta-topology combination"""
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nTraining MLP Specialist")
    print(f"  Topology: {topology_name}")
    print(f"  Theta: {theta_value}")
    print(f"  Training samples: {len(train_data)}")
    print(f"  Validation samples: {len(val_data)}")
    
    if len(train_data) == 0 or len(val_data) == 0:
        print(f"  ERROR: No data found for theta={theta_value}")
        return None
    
    print(f'  X shape: {train_data[0][0].shape}, Y shape: {train_data[0][1].shape}')
    
    # Get dimensions from data
    x0, y0 = train_data[0]
    input_dim = x0.shape[-1]
    output_dim = y0.shape[-1]
    
    # Create datasets
    train_dataset = NavigationTrajectoryDataset(train_data)
    val_dataset = NavigationTrajectoryDataset(val_data)
    
    # Initialize model
    model = MLPSpecialist(input_dim=input_dim, output_dim=output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    best_val_mae = float('inf')
    early_stop_counter = 0
    
    train_loss_list, val_loss_list = [], []
    train_mae_list, val_mae_list = [], []
    
    t_init = time.time()
    
    # Training loop
    pbar = tqdm(range(epochs), desc=f"[MLP-{topology_name}-θ{theta_value}]", unit="epoch", ncols=150)
    
    for epoch in pbar:
        # Training phase
        model.train()
        train_loss, train_mae = 0.0, 0.0
        
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            # Forward pass
            y_pred = model(x_batch)
            loss = F.mse_loss(y_pred, y_batch, reduction='mean')
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            mae = F.l1_loss(y_pred, y_batch, reduction='mean').item()
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
                
                y_pred = model(x_batch)
                loss = F.mse_loss(y_pred, y_batch, reduction='mean')
                mae = F.l1_loss(y_pred, y_batch, reduction='mean').item()
                
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
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_mae': best_val_mae
            }, os.path.join(save_dir, 'best_checkpoint.pth.tar'))
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
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'final_val_mae': val_mae
    }, os.path.join(save_dir, 'last_checkpoint.pth.tar'))
    
    # Save metrics
    metrics = {
        'train_loss': train_loss_list,
        'val_loss': val_loss_list,
        'train_mae': train_mae_list,
        'val_mae': val_mae_list,
    }
    with open(os.path.join(save_dir, 'metrics.pkl'), 'wb') as f:
        pickle.dump(metrics, f)
    
    # Plot training curves
    metrics_file = os.path.join(save_dir, 'metrics.pkl')
    output_plot = os.path.join(save_dir, 'training_curves.png')
    plot_training_metrics(metrics_file, output_plot)
    
    # Save summary
    with open(os.path.join(save_dir, 'training_summary.txt'), 'w') as f:
        f.write(f"MLP Specialist Training Summary\n")
        f.write("="*50 + "\n")
        f.write(f"Topology: {topology_name}\n")
        f.write(f"Theta: {theta_value}\n")
        f.write(f"Training samples: {len(train_data)} trajectories\n")
        f.write(f"Validation samples: {len(val_data)} trajectories\n")
        f.write(f"Best validation MAE: {best_val_mae:.6f}\n")
        f.write(f"Final epoch: {min(epoch+1, epochs)}\n")
        f.write(f"Early stopping counter: {early_stop_counter}/{patience}\n")
        f.write(f"Training time: {(time.time() - t_init)/60:.2f} minutes\n")
        f.write(f"\nModel architecture:\n")
        f.write(f"  Input dim: {input_dim}\n")
        f.write(f"  Output dim: {output_dim}\n")
        f.write(f"  Hidden layers: [256, 512, 256, 128]\n")
    
    print(f"  Best validation MAE: {best_val_mae:.6f}")
    
    return best_val_mae


# ============================================================================
# Comparison Plots
# ============================================================================

def create_general_comparison_plot(general_results, save_dir):
    """Create comparison plot for general models across topologies"""
    comparison_dir = os.path.join(save_dir, 'general_comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    topologies = sorted(general_results.keys())
    maes = [general_results[t] for t in topologies]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    bars = ax.bar(topologies, maes, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, mae in zip(bars, maes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mae:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Topology', fontsize=14)
    ax.set_ylabel('Validation MAE', fontsize=14)
    ax.set_title('General MLP Performance Across Topologies\n(Trained on All Theta Values)', 
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, max(maes) * 1.15])
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'general_mlp_comparison.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save summary
    with open(os.path.join(comparison_dir, 'general_summary.txt'), 'w') as f:
        f.write("General MLP Performance Summary\n")
        f.write("="*60 + "\n\n")
        f.write("Models trained on ALL theta values per topology\n\n")
        f.write("Validation MAE Results:\n")
        f.write("-"*30 + "\n")
        
        sorted_results = sorted(general_results.items(), key=lambda x: x[1])
        for i, (topology, mae) in enumerate(sorted_results, 1):
            f.write(f"{i}. {topology.capitalize():<15}: {mae:.6f}\n")
        
        f.write("\n" + "-"*30 + "\n")
        f.write(f"Best topology: {sorted_results[0][0]} (MAE: {sorted_results[0][1]:.6f})\n")
        f.write(f"Performance difference: {sorted_results[-1][1] - sorted_results[0][1]:.6f}\n")

def create_theta_comparison_plot(results, save_dir, topology):
    """Create comparison plot for different theta values within a topology"""
    comparison_dir = os.path.join(save_dir, f'topology_{topology}', 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    thetas = sorted(results.keys())
    maes = [results[t] for t in thetas]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(thetas)))
    bars = ax.bar(range(len(thetas)), maes, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=1.5)
    
    ax.set_xticks(range(len(thetas)))
    ax.set_xticklabels([f'θ={t:.1f}' for t in thetas])
    
    # Add value labels
    for bar, mae in zip(bars, maes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mae:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Theta Value', fontsize=14)
    ax.set_ylabel('Validation MAE', fontsize=14)
    ax.set_title(f'MLP Specialist Performance Across Theta Values\nTopology: {topology}', 
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, max(maes) * 1.15])
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, f'theta_comparison_{topology}.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save summary
    with open(os.path.join(comparison_dir, 'comparison_summary.txt'), 'w') as f:
        f.write(f"MLP Specialist Performance - Topology: {topology}\n")
        f.write("="*60 + "\n\n")
        f.write("Validation MAE Results:\n")
        f.write("-"*30 + "\n")
        for theta, mae in zip(thetas, maes):
            f.write(f"θ={theta:.1f}: {mae:.6f}\n")
        f.write("\n" + "-"*30 + "\n")
        f.write(f"Best theta: {thetas[np.argmin(maes)]:.1f} (MAE: {min(maes):.6f})\n")
        f.write(f"Worst theta: {thetas[np.argmax(maes)]:.1f} (MAE: {max(maes):.6f})\n")
        f.write(f"MAE range: {max(maes) - min(maes):.6f}\n")


def create_full_comparison_plot(all_results, save_dir):
    """Create comprehensive comparison plot across all topologies and thetas"""
    comparison_dir = os.path.join(save_dir, 'full_comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    topologies = sorted([t for t in all_results.keys() if all_results[t]])
    theta_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Plot 1: Specialists comparison
    ax1 = axes[0]
    x = np.arange(len(theta_values))
    width = 0.25
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for i, (topology, color) in enumerate(zip(topologies, colors)):
        maes = [all_results[topology].get(theta, np.nan) for theta in theta_values]
        offset = width * (i - 1)
        bars = ax1.bar(x + offset, maes, width, label=topology.capitalize(), 
                      color=color, alpha=0.8, edgecolor='black', linewidth=1)
        
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax1.set_xlabel('Theta Value', fontsize=12)
    ax1.set_ylabel('Validation MAE', fontsize=12)
    ax1.set_title('Specialist MLPs: Per-Theta Performance', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'θ={t:.1f}' for t in theta_values])
    ax1.legend(title='Topology', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: General vs Best Specialist per topology
    ax2 = axes[1]
    x2 = np.arange(len(topologies))
    width2 = 0.35
    
    general_maes = [all_results[t].get('ALL', np.nan) for t in topologies]
    best_specialist_maes = [min([all_results[t].get(theta, float('inf')) 
                                  for theta in theta_values]) for t in topologies]
    
    bars1 = ax2.bar(x2 - width2/2, best_specialist_maes, width2, 
                    label='Best Specialist', color='#2E86AB', alpha=0.8, edgecolor='black')
    bars2 = ax2.bar(x2 + width2/2, general_maes, width2, 
                    label='General (All θ)', color='#F18F01', alpha=0.8, edgecolor='black')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax2.set_xlabel('Topology', fontsize=12)
    ax2.set_ylabel('Validation MAE', fontsize=12)
    ax2.set_title('General vs Best Specialist', fontsize=14, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels([t.capitalize() for t in topologies])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'full_comparison.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # Extended summary
    with open(os.path.join(comparison_dir, 'full_summary.txt'), 'w') as f:
        f.write("MLP Complete Performance Summary\n")
        f.write("="*70 + "\n\n")
        
        for topology in topologies:
            f.write(f"\nTopology: {topology.upper()}\n")
            f.write("-"*50 + "\n")
            f.write("Specialists:\n")
            for theta in theta_values:
                if theta in all_results[topology]:
                    mae = all_results[topology][theta]
                    f.write(f"  θ={theta:.1f}: MAE = {mae:.6f}\n")
            if 'ALL' in all_results[topology]:
                f.write(f"General (all θ): MAE = {all_results[topology]['ALL']:.6f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("Comparison: General vs Best Specialist per Topology\n")
        f.write("-"*50 + "\n")
        
        for topology in topologies:
            specialist_maes = {theta: all_results[topology][theta] 
                             for theta in theta_values if theta in all_results[topology]}
            if specialist_maes and 'ALL' in all_results[topology]:
                best_specialist = min(specialist_maes.items(), key=lambda x: x[1])
                general_mae = all_results[topology]['ALL']
                improvement = ((best_specialist[1] - general_mae) / best_specialist[1]) * 100
                
                f.write(f"\n{topology.capitalize()}:\n")
                f.write(f"  Best Specialist: θ={best_specialist[0]:.1f}, MAE={best_specialist[1]:.6f}\n")
                f.write(f"  General Model:   MAE={general_mae:.6f}\n")
                f.write(f"  Difference:      {improvement:+.2f}%\n")


# ============================================================================
# Main Function
# ============================================================================
def main(args):
    print(f"\n{'='*70}")
    print("MLP Specialist Training - Per Theta & Topology")
    print(f"{'='*70}")
    print(f"Data directory: {args.data_dir}")
    print(f"Results directory: {args.result_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Patience: {args.patience}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Define topologies and theta values
    topologies = ['ellipsoidal', 'random', 'aligned']
    theta_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Store all results
    all_results = {topology: {} for topology in topologies}
    general_results = {}  # Para los modelos generales por topología
    
    # Train models
    total_specialists = len(topologies) * len(theta_values)
    total_generals = len(topologies)
    total_models = total_specialists + total_generals
    model_counter = 0
    
    for topology in topologies:
        print(f"\n{'='*70}")
        print(f"Processing Topology: {topology.upper()}")
        print(f"{'='*70}")
        
        topology_results = {}
        
        # Train specialist models for each theta
        for theta in theta_values:
            model_counter += 1
            print(f"\n[Specialist {model_counter}/{total_models}] Topology: {topology}, Theta: {theta}")
            print("-"*70)
            
            # Load data filtered by theta
            train_data, val_data, metadata = load_topology_data_by_theta(
                args.data_dir, topology, theta
            )
            
            if train_data is None or len(train_data) == 0:
                print(f"Skipping {topology} - theta {theta}: no data found")
                continue
            
            # Create save directory
            save_dir = os.path.join(
                args.result_dir, 
                f'topology_{topology}', 
                f'MLP_theta_{theta}'
            )
            
            # Train model
            best_mae = train_mlp_specialist(
                train_data, val_data, save_dir, topology, theta,
                args.batch_size, args.epochs, args.patience, device
            )
            
            if best_mae is not None:
                topology_results[theta] = best_mae
                all_results[topology][theta] = best_mae
        
        # Create comparison plot for this topology's specialists
        if topology_results:
            create_theta_comparison_plot(topology_results, args.result_dir, topology)
        
        # ====================================================================
        # TRAIN GENERAL MLP FOR THIS TOPOLOGY (ALL THETAS COMBINED)
        # ====================================================================
        print(f"\n{'='*70}")
        print(f"Training GENERAL MLP for Topology: {topology.upper()}")
        print(f"{'='*70}")
        model_counter += 1
        print(f"\n[General Model {model_counter}/{total_models}] Topology: {topology}, All Thetas Combined")
        print("-"*70)
        
        # Load ALL data for this topology
        train_data_all, val_data_all, metadata_all = load_topology_data_all_thetas(
            args.data_dir, topology
        )
        
        if train_data_all is not None and len(train_data_all) > 0:
            # Create save directory for general model
            save_dir_general = os.path.join(
                args.result_dir,
                f'topology_{topology}',
                'MLP_all_thetas'
            )
            
            # Get theta distribution info
            unique_thetas = sorted(list(set(metadata_all['train_thetas'])))
            print(f"  Training with data from thetas: {unique_thetas}")
            print(f"  Theta distribution in training set:")
            for theta in unique_thetas:
                count = metadata_all['train_thetas'].count(theta)
                print(f"    θ={theta:.1f}: {count} trajectories")
            
            # Train general model
            best_mae_general = train_mlp_specialist(
                train_data_all, val_data_all, save_dir_general, 
                topology, "ALL",
                args.batch_size, args.epochs, args.patience, device
            )
            
            if best_mae_general is not None:
                general_results[topology] = best_mae_general
                all_results[topology]['ALL'] = best_mae_general
        else:
            print(f"  ERROR: Could not load data for general model")
    
    # Create full comparison plot (including general models)
    if all([all_results[t] for t in topologies]):
        create_full_comparison_plot(all_results, args.result_dir)
        create_general_comparison_plot(general_results, args.result_dir)
        
        print(f"\n{'='*70}")
        print("Training Complete - Summary")
        print(f"{'='*70}")
        
        for topology in topologies:
            print(f"\n{topology.upper()}:")
            print("  Specialists:")
            for theta in theta_values:
                if theta in all_results[topology]:
                    mae = all_results[topology][theta]
                    print(f"    θ={theta:.1f}: MAE = {mae:.6f}")
            if 'ALL' in all_results[topology]:
                print(f"  General (all thetas): MAE = {all_results[topology]['ALL']:.6f}")
        
        print(f"\nAll results saved to: {args.result_dir}")
    else:
        print("\nWarning: Some topologies missing results")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train MLP specialists per theta and topology"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help="Path to processed data directory"
    )
    parser.add_argument(
        '--result-dir',
        type=str,
        default='/home/fernando/tesis/underwater-localization/results/MLP_topologies/low_variance',
        help="Path to save results"
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=5000,
        help="Number of training epochs"
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=200,
        help="Early stopping patience"
    )
    
    args = parser.parse_args()
    main(args)