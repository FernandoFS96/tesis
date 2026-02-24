import csv
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.models.anp import LatentModel
from src.utils.nav_dataset import NavigationTrajectoryDataset
from src.utils.plots import plot_training_metrics

'''
Use:
# Train one ANP per topology using all theta values
python train_anp_topologies.py \
    --data-dir /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_low_variance \
    --batch-size 32 \
    --epochs 5000 \
    --ctx-sample-mode first \
    --patience 250
'''

def compute_y_stats(train_data):
    # train_data: list of (X: (T,D), Y: (T,3))
    Y = np.concatenate([y for _, y in train_data], axis=0) # (N*T, 3)
    y_mean = torch.tensor(Y.mean(axis=0), dtype=torch.float32)
    y_std  = torch.tensor(Y.std(axis=0) + 1e-6, dtype=torch.float32)
    return y_mean, y_std

def kl_beta(epoch, warmup_epochs=500):
    # goes linearly from 0 to 1 in warmup_epochs
    return min(1.0, float(epoch) / float(max(1, warmup_epochs)))


def sample_context_indices(total_points, context_size, mode="first", device="cpu", generator=None):
    if mode == "first":
        return torch.arange(context_size, device=device)
    if mode == "random":
        perm = torch.randperm(total_points, device=device, generator=generator)
        return perm[:context_size].sort().values
    raise ValueError(f"Unknown context sampling mode: {mode}")


def load_topology_data(data_dir, topology):
    """Load all processed data for a specific topology"""
    topology_dir = os.path.join(data_dir, f'topology_{topology}')
    
    # Load train and validation data
    train_path = os.path.join(topology_dir, 'train_data.pkl')
    val_path = os.path.join(topology_dir, 'test_data.pkl')
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


def save_all_metrics(train_loss, val_loss, train_mae, val_mae,
                     experiment_dir,
                     train_nll=None, val_nll=None,
                     train_kl=None,  val_kl=None,
                     train_beta=None,
                     train_var_min=None, train_var_mean=None, train_var_max=None,
                     val_var_min=None,   val_var_mean=None,   val_var_max=None,
                     train_nll_nonctx=None, val_nll_nonctx=None):
    """Save all training and validation metrics for later analysis."""
    metrics = {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_mae': train_mae,
        'val_mae': val_mae,

        # diagnostics (optional)
        'train_nll': train_nll,
        'val_nll': val_nll,
        'train_kl': train_kl,
        'val_kl': val_kl,
        'train_beta': train_beta,

        'train_var_min': train_var_min,
        'train_var_mean': train_var_mean,
        'train_var_max': train_var_max,
        'val_var_min': val_var_min,
        'val_var_mean': val_var_mean,
        'val_var_max': val_var_max,

        'train_nll_nonctx': train_nll_nonctx,
        'val_nll_nonctx': val_nll_nonctx,
    }
    with open(os.path.join(experiment_dir, 'metrics.pkl'), 'wb') as f:
        pickle.dump(metrics, f)

def plot_anp_diagnostics(save_dir,
                         train_nll, val_nll,
                         train_kl, val_kl,
                         betas,
                         train_var_mean, val_var_mean,
                         train_var_min=None, train_var_max=None,
                         val_var_min=None,   val_var_max=None,
                         train_nll_nonctx=None, val_nll_nonctx=None):

    epochs = np.arange(1, len(train_nll) + 1)

    # 1) NLL / KL / beta
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_nll, label="train NLL")
    plt.plot(epochs, val_nll,   label="val NLL")
    if train_nll_nonctx is not None and val_nll_nonctx is not None:
        plt.plot(epochs, train_nll_nonctx, label="train NLL (non-ctx)", linestyle="--")
        plt.plot(epochs, val_nll_nonctx,   label="val NLL (non-ctx)",   linestyle="--")
    plt.plot(epochs, train_kl,  label="train KL")
    plt.plot(epochs, val_kl,    label="val KL")
    plt.plot(epochs, betas,     label="beta", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("ANP diagnostics: NLL / KL / beta")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_diagnostics_nll_kl_beta.png"), dpi=150)
    plt.close()

    # 2) Var stats
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_var_mean, label="train var mean")
    plt.plot(epochs, val_var_mean,   label="val var mean")
    if train_var_min is not None and train_var_max is not None:
        plt.plot(epochs, train_var_min, label="train var min", linestyle="--")
        plt.plot(epochs, train_var_max, label="train var max", linestyle="--")
    if val_var_min is not None and val_var_max is not None:
        plt.plot(epochs, val_var_min, label="val var min", linestyle=":")
        plt.plot(epochs, val_var_max, label="val var max", linestyle=":")
    plt.xlabel("Epoch")
    plt.ylabel("Variance (normalized space)")
    plt.title("ANP diagnostics: predicted variance stats")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_diagnostics_variance.png"), dpi=150)
    plt.close()

def train_anp_topology(train_data, val_data, save_dir, topology_name, 
                       batch_size=8, epochs=5000, patience=200, device='cuda',
                       ctx_sample_mode="first"):
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

    # Create datasets
    train_dataset = NavigationTrajectoryDataset(train_data)
    val_dataset = NavigationTrajectoryDataset(val_data)

    # Initialize model
    model = LatentModel(num_hidden=128, input_dim=input_dim, output_dim=output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Training variables
    best_val_mae = float('inf')
    early_stop_counter = 0
    # Lists to store metrics
    train_loss_list, val_loss_list = [], []
    train_mae_list, val_mae_list = [], []
    
    # Optional diagnostic metrics
    train_nll_list, val_nll_list = [], []
    train_kl_list,  val_kl_list  = [], []
    train_beta_list = []
    train_var_min_list, train_var_mean_list, train_var_max_list = [], [], []
    val_var_min_list,   val_var_mean_list,   val_var_max_list   = [], [], []
    # Optional: NLL only outside the context (to see "real prediction")
    train_nll_nonctx_list, val_nll_nonctx_list = [], []

    # Time tracking
    t_init = time.time()

    # Compute Y statistics for normalization
    y_mean, y_std = compute_y_stats(train_data)
    y_mean = y_mean.to(device)
    y_std  = y_std.to(device)

    # fixed context sizes for validation
    val_fracs = [0.1, 0.3, 0.5]

    # Training loop with progress bar
    pbar = tqdm(range(epochs), desc=f"[ANP-{topology_name}]", unit="epoch", ncols=200)
    
    for epoch in pbar:
        # Training phase
        model.train()
        train_loss, train_mae = 0.0, 0.0
        train_nll, train_kl = 0.0, 0.0
        train_var_min, train_var_mean, train_var_max = 0.0, 0.0, 0.0
        train_nll_nonctx = 0.0
        # Iterate over training batches
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            # KL beta for this epoch
            beta = kl_beta(epoch, warmup_epochs=500)
            # Dynamic context size selection
            total_points = x_batch.size(1)
            min_context = max(1, int(0.05 * total_points))
            max_context = min(int(0.95 * total_points), total_points - 1)
            
            context_size = torch.randint(min_context, max_context + 1, (1,), device=device).item() \
                if max_context > min_context else min_context

            # context indices (same subset for whole batch)
            context_indices = sample_context_indices(
                total_points,
                context_size,
                mode=ctx_sample_mode,
                device=device,
            )
            target_indices  = torch.arange(total_points, device=device)
            
            # Normalize Y, keep raw for loss calculation
            y_batch_raw = y_batch
            y_batch_norm = (y_batch - y_mean) / y_std
            
            # Prepare context and target sets
            context_x = x_batch[:, context_indices, :]
            context_y = y_batch_norm[:, context_indices, :]
            target_x  = x_batch[:, target_indices, :]
            target_y  = y_batch_norm[:, target_indices, :]

            # Forward pass
            y_pred_mean_norm, y_pred_var_norm, loss, kl, nll = model(context_x, context_y, target_x, target_y, beta=beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # diagnostics
            with torch.no_grad():
                # var stats (normalized)
                train_var_min  += y_pred_var_norm.min().item()
                train_var_mean += y_pred_var_norm.mean().item()
                train_var_max  += y_pred_var_norm.max().item()

                # nll and kl (scalars already averaged in the model)
                train_nll += nll.item()
                train_kl  += kl.item()

                # NLL outside the context only (optional)
                non_ctx_mask = torch.ones(total_points, dtype=torch.bool, device=device)
                non_ctx_mask[context_indices] = False

                nll_pointwise = 0.5 * torch.log(2 * torch.pi * y_pred_var_norm) \
                                + 0.5 * ((target_y - y_pred_mean_norm) ** 2) / y_pred_var_norm
                train_nll_nonctx += nll_pointwise[:, non_ctx_mask, :].mean().item()

            #mae = F.l1_loss(y_pred_mean, target_y, reduction='mean').item()
            y_pred_mean = y_pred_mean_norm * y_std + y_mean
            mae = F.l1_loss(y_pred_mean[:, non_ctx_mask, :], y_batch_raw[:, non_ctx_mask, :], reduction='mean').item()

            train_loss += loss.item()
            train_mae += mae
        # Average over batches
        train_loss /= len(train_loader)
        train_mae /= len(train_loader)
        train_loss_list.append(train_loss)
        train_mae_list.append(train_mae)

        # Average diagnostics
        train_nll /= len(train_loader)
        train_kl  /= len(train_loader)
        train_var_min  /= len(train_loader)
        train_var_mean /= len(train_loader)
        train_var_max  /= len(train_loader)
        train_nll_nonctx /= len(train_loader)
        train_nll_list.append(train_nll)
        train_kl_list.append(train_kl)
        train_var_min_list.append(train_var_min)
        train_var_mean_list.append(train_var_mean)
        train_var_max_list.append(train_var_max)
        train_nll_nonctx_list.append(train_nll_nonctx)
        train_beta_list.append(beta)

        # Validation phase
        g = torch.Generator(device=device) # deterministic context selection (same aleatory numbers each epoch -> stable val)
        g.manual_seed(1)
        model.eval()
        val_loss, val_mae = 0.0, 0.0
        val_nll, val_kl = 0.0, 0.0
        val_var_min, val_var_mean, val_var_max = 0.0, 0.0, 0.0
        val_nll_nonctx = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                total_points = x_batch.size(1)
                y_batch_raw = y_batch
                y_batch_norm = (y_batch - y_mean) / y_std

                batch_loss = 0.0
                batch_mae  = 0.0

                for frac in val_fracs:
                    context_size = max(1, min(total_points - 1, int(round(frac * total_points))))
                    # context indices (same subset for whole batch)
                    ctx_idx = sample_context_indices(
                        total_points,
                        context_size,
                        mode=ctx_sample_mode,
                        device=device,
                        generator=g,
                    )
                    tar_idx = torch.arange(total_points, device=device)
                    # Prepare context and target sets
                    context_x = x_batch[:, ctx_idx, :]
                    context_y = y_batch_norm[:, ctx_idx, :]
                    target_x  = x_batch[:, tar_idx, :]
                    target_y  = y_batch_norm[:, tar_idx, :]
                    # Forward pass
                    y_pred_mean_norm, y_pred_var_norm, loss, kl, nll = model(context_x, context_y, target_x, target_y, beta=1.0)
                    # Non-context MAE in meters
                    non_ctx_mask = torch.ones(total_points, dtype=torch.bool, device=device)
                    non_ctx_mask[ctx_idx] = False
                    y_pred_mean = y_pred_mean_norm * y_std + y_mean
                    mae = F.l1_loss(y_pred_mean[:, non_ctx_mask, :], y_batch_raw[:, non_ctx_mask, :], reduction='mean').item()
                    # Accumulate
                    batch_loss += loss.item()
                    batch_mae  += mae
                    # diagnostics
                    val_nll += nll.item()
                    val_kl  += kl.item()
                    val_var_min  += y_pred_var_norm.min().item()
                    val_var_mean += y_pred_var_norm.mean().item()
                    val_var_max  += y_pred_var_norm.max().item()
                    nll_pointwise = 0.5 * torch.log(2 * torch.pi * y_pred_var_norm) \
                                    + 0.5 * ((target_y - y_pred_mean_norm) ** 2) / y_pred_var_norm
                    val_nll_nonctx += nll_pointwise[:, non_ctx_mask, :].mean().item()

                # Average over different context fractions
                val_loss += (batch_loss / len(val_fracs))
                val_mae  += (batch_mae  / len(val_fracs))
        # Average over batches
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        val_loss_list.append(val_loss)
        val_mae_list.append(val_mae)
        # Average diagnostics over batches and context fractions
        den = len(val_loader) * len(val_fracs)
        val_nll /= den
        val_kl  /= den
        val_var_min  /= den
        val_var_mean /= den
        val_var_max  /= den
        val_nll_nonctx /= den
        # Store diagnostics
        val_nll_list.append(val_nll)
        val_kl_list.append(val_kl)
        val_var_min_list.append(val_var_min)
        val_var_mean_list.append(val_var_mean)
        val_var_max_list.append(val_var_max)
        val_nll_nonctx_list.append(val_nll_nonctx)

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
            'Loss': f"{train_loss:.2f}",
            'NLL': f"{train_nll:.2f}",
            'KL': f"{train_kl:.2f}",
            'β': f"{beta:.2f}",
            'varμ': f"{train_var_mean:.2e}",
            'varmin': f"{train_var_min:.2e}",
            'MAE': f"{train_mae:.2f}",
            'Val MAE': f"{val_mae:.2f}",
            'Best': f"{best_val_mae:.2f}",
            'ES': f"{early_stop_counter}"
        })

    # Save final model and metrics
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
           os.path.join(save_dir, 'last_checkpoint.pth.tar'))
    #save_all_metrics(train_loss_list, val_loss_list, train_mae_list, val_mae_list, save_dir)
    save_all_metrics(train_loss_list, val_loss_list, train_mae_list, val_mae_list, save_dir, train_nll=train_nll_list, val_nll=val_nll_list,
                     train_kl=train_kl_list, val_kl=val_kl_list, train_beta=train_beta_list, train_var_min=train_var_min_list, train_var_mean=train_var_mean_list, 
                     train_var_max=train_var_max_list, val_var_min=val_var_min_list, val_var_mean=val_var_mean_list, val_var_max=val_var_max_list, 
                     train_nll_nonctx=train_nll_nonctx_list, val_nll_nonctx=val_nll_nonctx_list)
    plot_anp_diagnostics(save_dir,train_nll_list, val_nll_list,train_kl_list,val_kl_list,train_beta_list,train_var_mean_list, val_var_mean_list,
                         train_var_min=train_var_min_list, train_var_max=train_var_max_list,val_var_min=val_var_min_list,val_var_max=val_var_max_list,
                         train_nll_nonctx=train_nll_nonctx_list, val_nll_nonctx=val_nll_nonctx_list)

    # Save CSV log
    csv_path = os.path.join(save_dir, "training_log.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "epoch",
            "train_loss","train_nll","train_nll_nonctx","train_kl","beta","train_var_min","train_var_mean","train_var_max","train_mae",
            "val_loss","val_nll","val_nll_nonctx","val_kl","val_var_min","val_var_mean","val_var_max","val_mae"
        ])
        for e in range(len(train_loss_list)):
            w.writerow([
                e+1,
                train_loss_list[e], train_nll_list[e], train_nll_nonctx_list[e], train_kl_list[e], train_beta_list[e],
                train_var_min_list[e], train_var_mean_list[e], train_var_max_list[e], train_mae_list[e],
                val_loss_list[e], val_nll_list[e], val_nll_nonctx_list[e], val_kl_list[e],
                val_var_min_list[e], val_var_mean_list[e], val_var_max_list[e], val_mae_list[e],
            ])
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

        f.write("\nDiagnostics (last epoch):\n")
        f.write(f"  Train NLL: {train_nll_list[-1]:.6f}\n")
        f.write(f"  Train NLL (non-ctx): {train_nll_nonctx_list[-1]:.6f}\n")
        f.write(f"  Train KL: {train_kl_list[-1]:.6f}\n")
        f.write(f"  Beta: {train_beta_list[-1]:.3f}\n")
        f.write(f"  Pred var (norm) min/mean/max: {train_var_min_list[-1]:.3e} / {train_var_mean_list[-1]:.3e} / {train_var_max_list[-1]:.3e}\n")

        f.write("\nDiagnostics (best/typical):\n")
        f.write(f"  Min train var_mean: {min(train_var_mean_list):.3e}\n")
        f.write(f"  Max train var_mean: {max(train_var_mean_list):.3e}\n")
        f.write(f"  Min val var_mean:   {min(val_var_mean_list):.3e}\n")
        f.write(f"  Max val var_mean:   {max(val_var_mean_list):.3e}\n")
    
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
    
    # change result dir to include context sampling mode
    args.result_dir = os.path.join(args.result_dir, f'ctx_{args.ctx_sample_mode}')
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
            device,
            ctx_sample_mode=args.ctx_sample_mode,
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
    parser.add_argument('--data-dir',type=str,required=True,help="Path to processed data directory (e.g., data/data_processed_topologies_low_variance)")
    parser.add_argument('--result-dir',type=str,default='./results/ANP_topologies/low_variance',help="Path to save results")
    parser.add_argument('--batch-size',type=int,default=8,help="Batch size for training")
    parser.add_argument('--epochs',type=int,default=10000,help="Number of training epochs")
    parser.add_argument('--patience',type=int,default=500,help="Early stopping patience")
    parser.add_argument('--ctx-sample-mode',type=str,default='first',choices=['random', 'first'],help="Context sampling: random or first (ordered)")
    args = parser.parse_args()
    main(args)