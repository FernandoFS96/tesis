import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.models.anp import LatentModel, DistributedLatentModel
from src.utils.nav_dataset import NavigationTrajectoryDataset

'''
Use example:

python evaluate_topologies.py \
    --data-dir /home/fernando/tesis/underwater-localization-topologies/data/data/data_processed_topologies_low_variance \
    --mlp-dir /home/fernando/tesis/underwater-localization-topologies/results/MLP_topologies/low_variance \
    --anp-dir /home/fernando/tesis/underwater-localization-topologies/results/ANP_topologies/low_variance \
    --output-dir /home/fernando/tesis/underwater-localization-topologies/results/evaluation_topologies \
    --context 30
'''


# ============================================================================
# MLP Architecture (same as training script)
# ============================================================================
class MLPSpecialist(nn.Module):
    """Simple MLP for baseline comparison - same architecture as training"""
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
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            predictions: (batch_size, seq_len, output_dim)
        """
        batch_size, seq_len, input_dim = x.size()
        x_flat = x.view(-1, input_dim)
        out_flat = self.network(x_flat)
        out = out_flat.view(batch_size, seq_len, -1)
        return out


# ============================================================================
# Evaluator Class
# ============================================================================
class TopologyEvaluator:
    def __init__(
        self,
        data_dir: Path,
        mlp_result_dir: Path,
        anp_result_dir: Path,
        output_dir: Path,
        context_percent: int = 40
    ):
        """
        Args:
            data_dir: Path to processed data (e.g., data_processed_topologies_low_variance)
            mlp_result_dir: Path to MLP results (e.g., results/MLP_topologies/low_variance)
            anp_result_dir: Path to ANP results (e.g., results/ANP_topologies/low_variance)
            output_dir: Path to save evaluation results
            context_percent: Percentage of points to use as context for ANP
        """
        self.data_dir = Path(data_dir)
        self.mlp_result_dir = Path(mlp_result_dir)
        self.anp_result_dir = Path(anp_result_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.context_percent = context_percent

        self._y_stats_cache = {} # topology -> (y_mean, y_std) en device
        self.eval_seed = 0 # seed determinista para muestrear contextos en evaluación
        self.mc_samples = 1 # >1 si quieres promediar varias muestras del latente
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Set random seed for reproducibility
        self.seed = 18
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
    
    def load_topology_data(self, topology: str) -> Tuple[Dict[float, List], List[float]]:
        """Load test data organized by theta values"""
        topology_dir = self.data_dir / f'topology_{topology}'
        
        # Load test data and metadata
        test_path = topology_dir / 'test_data.pkl'
        metadata_path = topology_dir / 'metadata.pkl'
        
        if not test_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Missing data files in {topology_dir}")
        
        with open(test_path, 'rb') as f:
            test_data = pickle.load(f)
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Organize data by theta values
        theta_groups = {}
        for sample, theta in zip(test_data, metadata['test_thetas']):
            if theta not in theta_groups:
                theta_groups[theta] = []
            theta_groups[theta].append(sample)
        
        theta_values = sorted(theta_groups.keys())
        
        return theta_groups, theta_values
    
    def get_y_stats(self, topology: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Devuelve (y_mean, y_std) calculados sobre train_data.pkl para la topología,
        y los cachea. Esto garantiza que la normalización en evaluación coincide con training.
        """
        if topology in self._y_stats_cache:
            return self._y_stats_cache[topology]

        topology_dir = self.data_dir / f"topology_{topology}"
        train_path = topology_dir / "train_data.pkl"
        if not train_path.exists():
            raise FileNotFoundError(f"train_data.pkl no encontrado: {train_path}")

        with open(train_path, "rb") as f:
            train_data = pickle.load(f)  # list of (X(T,D), Y(T,3))

        Y = np.concatenate([y for _, y in train_data], axis=0)  # (N*T, 3)
        y_mean = torch.tensor(Y.mean(axis=0), dtype=torch.float32, device=self.device)
        y_std  = torch.tensor(Y.std(axis=0) + 1e-6, dtype=torch.float32, device=self.device)

        self._y_stats_cache[topology] = (y_mean, y_std)
        return y_mean, y_std


    def normalize_y(self, y: torch.Tensor, y_mean: torch.Tensor, y_std: torch.Tensor) -> torch.Tensor:
        return (y - y_mean.view(1, 1, -1)) / y_std.view(1, 1, -1)


    def denormalize_y(self, y_norm: torch.Tensor, y_mean: torch.Tensor, y_std: torch.Tensor) -> torch.Tensor:
        return y_norm * y_std.view(1, 1, -1) + y_mean.view(1, 1, -1)


    def sample_context_indices(self, total_points: int, n_context: int, g: torch.Generator) -> torch.Tensor:
        """
        Subconjunto aleatorio determinista (por generator) y ordenado para indexar.
        """
        perm = torch.randperm(total_points, generator=g, device=self.device)
        return perm[:n_context].sort().values

    
    def load_mlp_models(
        self, 
        topology: str, 
        theta_values: List[float],
        input_dim: int,
        output_dim: int
    ) -> Dict[str, torch.nn.Module]:
        """Load all MLP models for a topology"""
        mlp_dir = self.mlp_result_dir / f'topology_{topology}'
        models = {}
        
        # Load specialist MLPs
        for theta in theta_values:
            checkpoint_path = mlp_dir / f'MLP_theta_{theta:.1f}' / 'best_checkpoint.pth.tar'
            if checkpoint_path.exists():
                try:
                    model = MLPSpecialist(input_dim=input_dim, output_dim=output_dim)
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    model.load_state_dict(checkpoint['model'])
                    model = model.to(self.device).eval()
                    models[f'MLP(θ={theta:.1f})'] = model
                    print(f"Loaded MLP specialist for theta={theta:.1f}")
                except Exception as e:
                    print(f"Warning: Could not load MLP for theta={theta:.1f}: {e}")
            else:
                print(f"Warning: Checkpoint not found for theta={theta:.1f}")
        
        # Load general MLP (trained on all thetas)
        general_path = mlp_dir / 'MLP_all_thetas' / 'best_checkpoint.pth.tar'
        if general_path.exists():
            try:
                model = MLPSpecialist(input_dim=input_dim, output_dim=output_dim)
                checkpoint = torch.load(general_path, map_location=self.device)
                model.load_state_dict(checkpoint['model'])
                model = model.to(self.device).eval()
                models['MLP(all_θ)'] = model
                print(f"Loaded general MLP (all thetas)")
            except Exception as e:
                print(f"Warning: Could not load general MLP: {e}")
        else:
            print(f"Warning: General MLP checkpoint not found")
        
        return models
    
    def load_anp_model(self, topology: str, input_dim: int, output_dim: int, distributed = False) -> torch.nn.Module:
        """Load ANP model for a topology"""
        anp_path = self.anp_result_dir / f'ANP_{topology}' / 'best_checkpoint.pth.tar'
        
        if not anp_path.exists():
            raise FileNotFoundError(f"ANP checkpoint not found: {anp_path}")
        
        sensor_emb_dim = 64 # MUST match the value used during training
        n_sensors = 10
        sensor_feature_dim = 401
        
        if distributed:
            base_anp = LatentModel(num_hidden=128, input_dim=sensor_emb_dim, output_dim=output_dim)
            anp_model = DistributedLatentModel(base_anp=base_anp,
                                               n_sensors=n_sensors,
                                               in_dim_per_sensor=sensor_feature_dim,
                                               emb_dim=sensor_emb_dim,
                                               fusion="mean",)
        else:
            anp_model = LatentModel(num_hidden=128, input_dim=input_dim, output_dim=output_dim)
        checkpoint = torch.load(anp_path, map_location=self.device)
        anp_model.load_state_dict(checkpoint['model'])
        model = anp_model.to(self.device).eval()
        print(f"  Loaded ANP model")
        
        return model
    
    def compute_mae_matrix(
        self,
        theta_groups: Dict[float, List],
        theta_values: List[float],
        mlp_models: Dict[str, torch.nn.Module],
        anp_model: torch.nn.Module,
        y_mean: torch.Tensor,
        y_std: torch.Tensor,
        seed: int = 0,
        mc_samples: int = 1
    ) -> Tuple[np.ndarray, List[str]]:
        """Compute MAE for all models on all theta groups"""
        g = torch.Generator(device=self.device)
        g.manual_seed(torch.seed)
        
        # Model names for rows
        model_names = (
            [f'MLP(θ={theta:.1f})' for theta in theta_values] + 
            ['MLP(all_θ)', f'ANP']
        )
        
        mae_matrix = np.zeros((len(model_names), len(theta_values)))
        
        print("  Computing MAE matrix...")
        for j, theta in enumerate(tqdm(theta_values, desc="    Theta groups", leave=False)):
            group_data = theta_groups[theta]
            ds = NavigationTrajectoryDataset(group_data)
            loader = DataLoader(ds, batch_size=8, shuffle=False)
            
            # Store errors for each model
            model_errors = {name: [] for name in model_names}
            
            with torch.no_grad():
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    total_points = x.size(1)
                    n_context = int((self.context_percent / 100) * total_points)
                    n_context = max(1, min(n_context, total_points - 1))

                    context_idx = self.sample_context_indices(total_points, n_context, g)
                    non_ctx_mask = torch.ones(total_points, dtype=torch.bool, device=self.device)
                    non_ctx_mask[context_idx] = False

                    # ---------- MLPs: MAE en NO-contexto (misma máscara para ser “justos”) ----------
                    for model_name, model in mlp_models.items():
                        pred = model(x)  # (B,T,3) en metros (como tus MLPs)
                        mae = F.l1_loss(pred[:, non_ctx_mask, :], y[:, non_ctx_mask, :], reduction="none").mean(dim=[1, 2])
                        model_errors[model_name].extend(mae.cpu().numpy())

                    # ---------- ANP: normaliza cy/ty, predice, desnormaliza y MAE en NO-contexto ----------
                    target_idx = torch.arange(total_points, device=self.device)
                    cx = x[:, context_idx, :]
                    tx = x[:, target_idx, :]

                    y_norm = self.normalize_y(y, y_mean, y_std)
                    cy = y_norm[:, context_idx, :]

                    # Predicción (sin target_y para no “filtrar” información)
                    if mc_samples <= 1:
                        pred_norm, _, *_ = anp_model(cx, cy, tx)  # (B,T,3) en espacio normalizado
                    else:
                        preds = []
                        for _ in range(mc_samples):
                            p_norm, _, *_ = anp_model(cx, cy, tx)
                            preds.append(p_norm)
                        pred_norm = torch.stack(preds, dim=0).mean(dim=0)

                    pred = self.denormalize_y(pred_norm, y_mean, y_std)
                    mae = F.l1_loss(pred[:, non_ctx_mask, :], y[:, non_ctx_mask, :], reduction="none").mean(dim=[1, 2])
                    model_errors["ANP"].extend(mae.cpu().numpy())
            
            # Compute mean MAE for each model
            for i, model_name in enumerate(model_names):
                if model_name in model_errors and len(model_errors[model_name]) > 0:
                    mae_matrix[i, j] = np.mean(model_errors[model_name])
                else:
                    mae_matrix[i, j] = np.nan  # Model not available
        
        return mae_matrix, model_names
    
    def plot_heatmap(
        self, 
        mae_matrix: np.ndarray,
        model_names: List[str],
        theta_values: List[float],
        topology: str,
        save_dir: Path
    ):
        """Create and save heatmap of MAE values"""
        
        # Find best model for each theta (minimum MAE in each column)
        best_indices = np.nanargmin(mae_matrix, axis=0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Create heatmap
        sns.heatmap(
            mae_matrix,
            annot=True,
            fmt='.3f',
            cmap='viridis',  # Red (bad) to Green (good) reversed
            xticklabels=[f'{t:.1f}' for t in theta_values],
            yticklabels=model_names,
            cbar_kws={'label': 'MAE'},
            ax=ax,
            annot_kws={'size': 9},
            mask=np.isnan(mae_matrix)  # Mask NaN values
        )
        
        # Highlight best model for each theta with blue border
        for j, i in enumerate(best_indices):
            if not np.isnan(mae_matrix[i, j]):
                rect = patches.Rectangle(
                    (j, i), 1, 1,
                    fill=False,
                    edgecolor='blue',
                    linewidth=3
                )
                ax.add_patch(rect)
        
        ax.set_xlabel('Theta Value (Channel Variance)', fontsize=13)
        ax.set_ylabel('Model', fontsize=13)
        ax.set_title(f'MAE Comparison - Topology: {topology.capitalize()}\n(Blue border = best model per theta)', 
                    fontsize=15, fontweight='bold')
        
        plt.tight_layout()
        save_path = save_dir / f'heatmap_{topology}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Saved heatmap to {save_path}")
    
    def plot_trajectories(
        self,
        theta_groups: Dict[float, List],
        mlp_models: Dict[str, torch.nn.Module],
        anp_model: torch.nn.Module,
        topology: str,
        save_dir: Path,
        target_theta: float = 0.1,
        n_trajectories: int = 30
    ):
        """Plot individual trajectory comparisons for a specific theta value"""

        # Get data for target theta
        if target_theta not in theta_groups:
            print(f"    Warning: theta={target_theta} not found, using closest value")
            target_theta = min(theta_groups.keys(), key=lambda x: abs(x - target_theta))

        # Create theta-specific subdirectory
        theta_dir = save_dir / f'trajectories_theta_{target_theta:.1f}'
        theta_dir.mkdir(parents=True, exist_ok=True)

        # Randomly select trajectories
        group_data = theta_groups[target_theta]
        samples = random.sample(group_data, min(n_trajectories, len(group_data)))

        for idx, (x_np, y_np) in enumerate(samples):
            # Create individual figure
            fig, ax = plt.subplots(figsize=(12, 8))

            # Convert to tensors
            x = torch.FloatTensor(x_np).unsqueeze(0).to(self.device)
            y = torch.FloatTensor(y_np).unsqueeze(0).to(self.device)

            with torch.no_grad():
                # Ground truth
                gt_traj = y.squeeze().cpu().numpy()[:, :2]  # Only x, y

                # MLP specialist prediction
                specialist_key = f'MLP(θ={target_theta:.1f})'
                if specialist_key in mlp_models:
                    mlp_spec_pred = mlp_models[specialist_key](x)
                    mlp_spec_traj = mlp_spec_pred.squeeze().cpu().numpy()[:, :2]
                else:
                    mlp_spec_traj = None

                # MLP general prediction
                if 'MLP(all_θ)' in mlp_models:
                    mlp_gen_pred = mlp_models['MLP(all_θ)'](x)
                    mlp_gen_traj = mlp_gen_pred.squeeze().cpu().numpy()[:, :2]
                else:
                    mlp_gen_traj = None

                # ANP prediction
                total_points = x.size(1)
                n_context = int((self.context_percent / 100) * total_points)
                n_context = max(1, min(n_context, total_points - 1))

                cx = x[:, :n_context, :]
                cy = y[:, :n_context, :]
                tx = x

                anp_pred, *_ = anp_model(cx, cy, tx)
                anp_traj = anp_pred.squeeze().cpu().numpy()[:, :2]

            # Plot trajectory
            ax.plot(gt_traj[:, 0], gt_traj[:, 1], color='blue', linestyle='--', 
                   label='GT', linewidth=2.5, alpha=0.8)

            if mlp_spec_traj is not None:
                ax.plot(mlp_spec_traj[:, 0], mlp_spec_traj[:, 1], 
                       color='orange', label=f'MLP(θ={target_theta:.1f})', 
                       linewidth=2, linestyle='-')

            if mlp_gen_traj is not None:
                ax.plot(mlp_gen_traj[:, 0], mlp_gen_traj[:, 1], 
                       color='green', label='DR-MLP', 
                       linewidth=2, linestyle='-')

            ax.plot(anp_traj[:, 0], anp_traj[:, 1], 
                   color='red', label=f'ANP', 
                   linewidth=2, linestyle='-')

            # Mark start and end points
            ax.plot(gt_traj[0, 0], gt_traj[0, 1], 'go', markersize=10, 
                   label='Start', zorder=10)
            ax.plot(gt_traj[-1, 0], gt_traj[-1, 1], 'ro', markersize=10, 
                   label='End', zorder=10)

            # Mark context region for ANP
            #if n_context < total_points:
            #    ax.axvline(x=gt_traj[n_context, 0], color='gray', 
            #              linestyle='--', alpha=0.5, linewidth=1)

            ax.set_xlabel('X [m]', fontsize=16)
            ax.set_ylabel('Y [m]', fontsize=16)
            ax.set_title(f'Trajectory {idx+1} - {topology.capitalize()} (θ={target_theta:.1f})', 
                        fontsize=18, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.grid(True, alpha=0.3, linewidth=2)
            ax.legend(fontsize=16, loc='best')
            #ax.axis('equal')
            ax.set_aspect('equal', adjustable='box')
            plt.tight_layout()

            # Save individual trajectory plot
            save_path = theta_dir / f'trajectory_{idx+1}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

        print(f"Saved {n_trajectories} trajectory plots to {theta_dir}")
    
    def plot_axiswise_ci(
            self,
            theta_groups: Dict[float, List],
            mlp_models: Dict[str, torch.nn.Module],
            anp_model: torch.nn.Module,
            topology: str,
            save_dir: Path,
            target_theta: float = 0.1,
            n_trajectories: int = 2,
            k: float = 1.0,
        ):
            """
            Plot ANP mean ± k·σ and MLP predictions per axis (x, y) side by side.
    
            For a given theta, randomly selects up to `n_trajectories` trajectories
            and generates a grid with:
               rows   = trajectories
               cols   = 2 (x-axis, y-axis)
            Each subplot shows:
               - Ground Truth (dashed)
               - ANP mean (red) with ±k·σ confidence band
               - DR-MLP / general MLP(all_θ) prediction (green), if available
            """
    
            # If exact theta not found, fall back to closest
            if target_theta not in theta_groups:
                print(f"    Warning: theta={target_theta} not found, using closest value")
                target_theta = min(theta_groups.keys(), key=lambda x: abs(x - target_theta))
    
            group_data = theta_groups[target_theta]
            if len(group_data) == 0:
                print(f"    No data available for theta={target_theta}")
                return
    
            # Randomly select trajectories
            samples = random.sample(group_data, k=min(n_trajectories, len(group_data)))
    
            # Use context_percent from evaluator as context fraction
            context_frac = self.context_percent / 100.0
    
            # Prepare figure
            import numpy as np  # already imported at top, but safe here
            fig, axes = plt.subplots(
                nrows=len(samples),
                ncols=2,
                figsize=(16, 6 * len(samples))
            )
    
            # Ensure axes is 2D array for consistent indexing
            axes = np.atleast_2d(axes)
    
            # Try to get the "DR-MLP" / general MLP
            dr_mlp = mlp_models.get('MLP(all_θ)', None)
    
            for i, (x_np, y_np) in enumerate(samples):
                # Convert to tensors and move to device
                x = torch.FloatTensor(x_np).unsqueeze(0).to(self.device)  # (1, T, input_dim)
                y = torch.FloatTensor(y_np).unsqueeze(0).to(self.device)  # (1, T, output_dim)
    
                size = x.size(1)  # sequence length (number of time points)
    
                # Context selection
                n_ctx = max(1, int(context_frac * size))
                n_ctx = min(n_ctx, size)  # just to be safe
    
                cx = x[:, :n_ctx, :]
                cy = y[:, :n_ctx, :]
                tx = x
    
                # ANP prediction: mean and variance (or std)
                with torch.no_grad():
                    # Your ANP forward signature: (mean, var, *rest)
                    y_mean, y_var, *_ = anp_model(cx, cy, tx)
                
                y_mean_np = y_mean.squeeze(0).cpu().numpy()              # (T, output_dim)
                y_std_np  = torch.sqrt(y_var).squeeze(0).cpu().numpy()   # (T, output_dim)
    
                # DR-MLP prediction (general MLP)
                if dr_mlp is not None:
                    with torch.no_grad():
                        dr_pred = dr_mlp(x).squeeze(0).cpu().numpy()  # (T, output_dim)
                else:
                    dr_pred = None
    
                # Ground truth
                y_true_np = y.squeeze(0).cpu().numpy()  # (T, output_dim)
                t = np.arange(size)
    
                # Plot for each axis separately: d = 0 (x), d = 1 (y)
                for d in range(2):
                    axis_label = 'x' if d == 0 else 'y'
                    ax = axes[i, d]
    
                    # Ground truth
                    ax.plot(t, y_true_np[:, d], '--', label='Ground Truth')
    
                    # ANP mean
                    ax.plot(t, y_mean_np[:, d], label='ANP mean', color='red')
    
                    # Confidence interval
                    if y_std_np is not None:
                        lower = y_mean_np[:, d] - k * y_std_np[:, d]
                        upper = y_mean_np[:, d] + k * y_std_np[:, d]
                        ax.fill_between(
                            t, lower, upper,
                            color='red',
                            alpha=0.2,
                            label=f'±{k}σ'
                        )
    
                    # DR-MLP prediction (if available)
                    if dr_pred is not None:
                        ax.plot(t, dr_pred[:, d], label='DR-MLP', color='green')
    
                    ax.set_title(
                        f"Trajectory {i+1} – {axis_label}-axis\n"
                        f"{topology.capitalize()} (θ={target_theta:.1f})",
                        fontsize=16
                    )
                    ax.tick_params(axis='both', which='major', labelsize=12)
                    ax.legend(fontsize=11)
                    ax.grid(True, alpha=0.3)
    
            plt.tight_layout()
    
            # Save in a dedicated filename
            out_path = save_dir / f'axiswise_ci_{topology}_theta{target_theta:.1f}_k{k}.png'
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f" Saved axiswise CI plot to {out_path}")

    def evaluate_topology(self, topology: str):
        """Run complete evaluation for one topology"""
        print(f"\nEvaluating topology: {topology}")
        print("="*60)
        
        # Create topology-specific output directory
        topology_output = self.output_dir / topology
        topology_output.mkdir(exist_ok=True)
        
        # Load data
        print("  Loading test data...")
        theta_groups, theta_values = self.load_topology_data(topology)
        print(f"  Found {len(theta_values)} theta values: {theta_values}")
        
        # Get dimensions from first sample
        first_sample = next(iter(theta_groups.values()))[0]
        input_dim = first_sample[0].shape[-1]
        output_dim = first_sample[1].shape[-1]
        print(f"  Input dim: {input_dim}, Output dim: {output_dim}")
        
        # Load models
        print("  Loading models...")
        mlp_models = self.load_mlp_models(topology, theta_values, input_dim, output_dim)
        anp_model = self.load_anp_model(topology, input_dim, output_dim, distributed = False)

        # Get y normalization stats        
        y_mean, y_std = self.get_y_stats(topology)

        # Compute MAE matrix
        mae_matrix, model_names = self.compute_mae_matrix(theta_groups, theta_values, mlp_models, anp_model, y_mean=y_mean, y_std=y_std, 
                                                          seed=self.eval_seed, mc_samples=self.mc_samples
                                                          )
        
        # Save numerical results
        results_path = topology_output / 'mae_results.txt'
        with open(results_path, 'w') as f:
            f.write(f"MAE Results - Topology: {topology}\n")
            f.write("="*80 + "\n\n")
            f.write("Rows: Models, Columns: Theta values (channel variance)\n\n")
            
            # Header
            f.write(f"{'Model':<25}")
            for theta in theta_values:
                f.write(f"θ={theta:<7.1f}")
            f.write("Mean\n")
            f.write("-"*80 + "\n")
            
            # Results
            for i, model_name in enumerate(model_names):
                f.write(f"{model_name:<25}")
                row_vals = []
                for j in range(len(theta_values)):
                    val = mae_matrix[i, j]
                    if not np.isnan(val):
                        f.write(f"{val:<9.4f}")
                        row_vals.append(val)
                    else:
                        f.write(f"{'N/A':<9}")
                if row_vals:
                    f.write(f"{np.mean(row_vals):<9.4f}\n")
                else:
                    f.write("N/A\n")
            
            f.write("\n" + "-"*80 + "\n")
            f.write("Best model per theta:\n")
            for j, theta in enumerate(theta_values):
                col_vals = mae_matrix[:, j]
                valid_mask = ~np.isnan(col_vals)
                if np.any(valid_mask):
                    best_idx = np.nanargmin(col_vals)
                    f.write(f"  θ={theta:.1f}: {model_names[best_idx]} "
                           f"(MAE={mae_matrix[best_idx, j]:.4f})\n")
            
            # Overall comparison
            f.write("\n" + "-"*80 + "\n")
            f.write("Overall Performance (mean across all theta values):\n")
            for i, model_name in enumerate(model_names):
                row_vals = mae_matrix[i, ~np.isnan(mae_matrix[i, :])]
                if len(row_vals) > 0:
                    f.write(f"  {model_name}: {np.mean(row_vals):.4f}\n")
        
        print(f"    Saved numerical results to {results_path}")
        
        # Create visualizations
        print("  Creating visualizations...")
        self.plot_heatmap(mae_matrix, model_names, theta_values, topology, topology_output)
        
        # Plot trajectories for different theta values
        for theta in [0.1, 0.3, 0.5]:
            if theta in theta_groups:
                self.plot_trajectories(
                    theta_groups, mlp_models, anp_model, 
                    topology, topology_output, target_theta=theta
                )
                # per-axis time-series plots with CIs
                self.plot_axiswise_ci(
                    theta_groups=theta_groups,
                    mlp_models=mlp_models,
                    anp_model=anp_model,
                    topology=topology,
                    save_dir=topology_output,
                    target_theta=theta,
                    n_trajectories=2,
                    k=1.0,   # confidence band multiplier
                )
        
        return mae_matrix, model_names, theta_values
    
    def evaluate_anp_context_sensitivity(self, topology: str):
        """Evaluate ANP performance across different context sizes"""
        print(f"\nEvaluating ANP context sensitivity for topology: {topology}")
        print("="*60)
        
        # Create output directory
        context_output = self.output_dir / topology / 'anp_context_analysis'
        context_output.mkdir(parents=True, exist_ok=True)
        
        # Load data
        print("  Loading test data...")
        theta_groups, theta_values = self.load_topology_data(topology)
        
        # Get dimensions
        first_sample = next(iter(theta_groups.values()))[0]
        input_dim = first_sample[0].shape[-1]
        output_dim = first_sample[1].shape[-1]

        # Get y normalization stats
        y_mean, y_std = self.get_y_stats(topology)
        g = torch.Generator(device=self.device)
        g.manual_seed(self.eval_seed)
        
        # Load ANP model
        print("  Loading ANP model...")
        anp_model = self.load_anp_model(topology, input_dim, output_dim, distributed = False)
        
        # Define context percentages to test
        context_percentages = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 
                              55, 60, 65, 70, 75, 80, 85, 90]
        
        # Store results
        overall_results = {ctx: [] for ctx in context_percentages}
        theta_results = {theta: {ctx: [] for ctx in context_percentages} 
                         for theta in theta_values}
        
        print("  Computing MAE for different context sizes...")
        
        # Evaluate for each context percentage
        for ctx_pct in tqdm(context_percentages, desc="  Context %"):
            
            # Evaluate on each theta group
            for theta in theta_values:
                group_data = theta_groups[theta]
                ds = NavigationTrajectoryDataset(group_data)
                loader = DataLoader(ds, batch_size=8, shuffle=False)
                
                theta_maes = []
                
                with torch.no_grad():
                    for x, y in loader:
                        x, y = x.to(self.device), y.to(self.device)
                        
                        total_points = x.size(1)
                        n_context = int((ctx_pct / 100) * total_points)
                        n_context = max(1, min(n_context, total_points - 1))
                        
                        context_idx = self.sample_context_indices(total_points, n_context, g)
                        non_ctx_mask = torch.ones(total_points, dtype=torch.bool, device=self.device)
                        non_ctx_mask[context_idx] = False

                        cx = x[:, context_idx, :]
                        tx = x  # o x[:, target_idx, :]
                        y_norm = self.normalize_y(y, y_mean, y_std)
                        cy = y_norm[:, context_idx, :]

                        pred_norm, _, *_ = anp_model(cx, cy, tx)
                        pred = self.denormalize_y(pred_norm, y_mean, y_std)

                        mae = F.l1_loss(pred[:, non_ctx_mask, :], y[:, non_ctx_mask, :], reduction="none").mean(dim=[1, 2])
                        theta_maes.extend(mae.cpu().numpy())
                
                # Store results
                mean_mae = np.mean(theta_maes)
                theta_results[theta][ctx_pct] = mean_mae
                overall_results[ctx_pct].extend(theta_maes)
        
        # Compute overall means
        overall_means = {ctx: np.mean(maes) for ctx, maes in overall_results.items()}
        
        # Save numerical results
        results_path = context_output / 'context_sensitivity_results.txt'
        with open(results_path, 'w') as f:
            f.write(f"ANP Context Sensitivity Analysis - Topology: {topology}\n")
            f.write("="*70 + "\n\n")
            
            # Overall results
            f.write("Overall Results (all theta values combined):\n")
            f.write("-"*50 + "\n")
            f.write(f"{'Context %':<12} {'MAE':<10}\n")
            f.write("-"*50 + "\n")
            for ctx in context_percentages:
                f.write(f"{ctx:<12} {overall_means[ctx]:<10.4f}\n")
            
            best_ctx = min(overall_means.items(), key=lambda x: x[1])
            f.write(f"\nBest context: {best_ctx[0]}% (MAE: {best_ctx[1]:.4f})\n")
            
            # Per-theta results
            f.write("\n\nResults by Theta Value:\n")
            f.write("="*70 + "\n")
            for theta in theta_values:
                f.write(f"\nTheta = {theta:.1f}\n")
                f.write("-"*50 + "\n")
                f.write(f"{'Context %':<12} {'MAE':<10}\n")
                f.write("-"*50 + "\n")
                for ctx in context_percentages:
                    f.write(f"{ctx:<12} {theta_results[theta][ctx]:<10.4f}\n")
                
                best_ctx_theta = min(theta_results[theta].items(), key=lambda x: x[1])
                f.write(f"Best context for θ={theta:.1f}: {best_ctx_theta[0]}% "
                       f"(MAE: {best_ctx_theta[1]:.4f})\n")
        
        print(f"    Saved results to {results_path}")
        
        # Create visualizations
        self._plot_context_sensitivity(
            context_percentages, overall_means, theta_results, theta_values,
            topology, context_output
        )
        
        return overall_means, theta_results
    
    def _plot_context_sensitivity(
        self,
        context_percentages: List[int],
        overall_means: Dict[int, float],
        theta_results: Dict[float, Dict[int, float]],
        theta_values: List[float],
        topology: str,
        save_dir: Path
    ):
        """Create plots for context sensitivity analysis"""
        
        # Plot 1: Overall performance vs context
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ctx_list = sorted(context_percentages)
        mae_list = [overall_means[ctx] for ctx in ctx_list]
        
        ax.plot(ctx_list, mae_list, 'b-o', linewidth=2, markersize=6, label='Overall')
        
        # Mark best context
        best_ctx = min(overall_means.items(), key=lambda x: x[1])
        ax.plot(best_ctx[0], best_ctx[1], 'r*', markersize=15, 
               label=f'Best: {best_ctx[0]}%', zorder=10)
        
        ax.set_xlabel('Context Size (%)', fontsize=13)
        ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=13)
        ax.set_title(f'ANP Performance vs Context Size\nTopology: {topology.capitalize()}',
                    fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        plt.savefig(save_dir / f'context_overall_{topology}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Per-theta comparison
        fig, ax = plt.subplots(figsize=(14, 7))
        
        colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(theta_values)))
        
        for theta, color in zip(theta_values, colors):
            mae_list_theta = [theta_results[theta][ctx] for ctx in ctx_list]
            ax.plot(ctx_list, mae_list_theta, '-o', linewidth=2, markersize=5,
                   color=color, label=f'θ={theta:.1f}', alpha=0.8)
        
        ax.set_xlabel('Context Size (%)', fontsize=13)
        ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=13)
        ax.set_title(f'ANP Performance vs Context Size (by Theta)\nTopology: {topology.capitalize()}',
                    fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, ncol=2, loc='best')
        
        plt.tight_layout()
        plt.savefig(save_dir / f'context_per_theta_{topology}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Heatmap of context vs theta
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create matrix: rows=theta, cols=context
        mae_matrix = np.zeros((len(theta_values), len(ctx_list)))
        for i, theta in enumerate(theta_values):
            for j, ctx in enumerate(ctx_list):
                mae_matrix[i, j] = theta_results[theta][ctx]
        
        sns.heatmap(
            mae_matrix,
            annot=True,
            fmt='.3f',
            cmap='viridis',
            xticklabels=[f'{ctx}%' for ctx in ctx_list],
            yticklabels=[f'{theta:.1f}' for theta in theta_values],
            cbar_kws={'label': 'MAE'},
            ax=ax,
            annot_kws={'size': 8}
        )
        
        ax.set_xlabel('Context Size', fontsize=13)
        ax.set_ylabel('Theta Value', fontsize=13)
        ax.set_title(f'ANP Performance Heatmap: Context Size vs Theta\nTopology: {topology.capitalize()}',
                    fontsize=15, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_dir / f'context_heatmap_{topology}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot 4: Best context per theta
        fig, ax = plt.subplots(figsize=(10, 6))
        
        best_contexts = []
        best_maes = []
        for theta in theta_values:
            best = min(theta_results[theta].items(), key=lambda x: x[1])
            best_contexts.append(best[0])
            best_maes.append(best[1])
        
        x = np.arange(len(theta_values))
        bars = ax.bar(x, best_contexts, color=plt.get_cmap('viridis')(np.linspace(0, 1, len(theta_values))),
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add MAE values on top
        for i, (bar, mae) in enumerate(zip(bars, best_maes)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}%\n(MAE:{mae:.3f})', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Theta Value', fontsize=13)
        ax.set_ylabel('Optimal Context Size (%)', fontsize=13)
        ax.set_title(f'Optimal Context Size per Theta\nTopology: {topology.capitalize()}',
                    fontsize=15, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{theta:.1f}' for theta in theta_values])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_dir / f'optimal_context_{topology}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved 4 context sensitivity plots to {save_dir}")
    
    def create_comparison_summary(self, all_results: Dict):
        """Create summary comparing all topologies"""
        summary_path = self.output_dir / 'topology_comparison_summary.txt'
        
        with open(summary_path, 'w') as f:
            f.write("Topology Comparison Summary\n")
            f.write("="*90 + "\n\n")
            
            for topology, (mae_matrix, model_names, theta_values) in all_results.items():
                f.write(f"\nTopology: {topology.upper()}\n")
                f.write("-"*50 + "\n")
                
                # Find overall best model
                overall_mae = np.nanmean(mae_matrix, axis=1)
                valid_models = ~np.isnan(overall_mae)
                
                if np.any(valid_models):
                    best_idx = np.nanargmin(overall_mae)
                    f.write(f"Best overall model: {model_names[best_idx]} "
                           f"(Mean MAE: {overall_mae[best_idx]:.4f})\n\n")
                    
                    # ANP vs MLP general
                    anp_idx = model_names.index(f'ANP')
                    mlp_gen_idx = model_names.index('MLP(all_θ)')
                    
                    anp_mae = overall_mae[anp_idx]
                    mlp_mae = overall_mae[mlp_gen_idx]
                    
                    f.write(f"ANP MAE: {anp_mae:.4f}\n")
                    f.write(f"MLP(all_θ) MAE: {mlp_mae:.4f}\n")
                    
                    if anp_mae < mlp_mae:
                        improvement = ((mlp_mae - anp_mae) / mlp_mae) * 100
                        f.write(f"ANP improvement over general MLP: {improvement:.2f}%\n")
                    else:
                        degradation = ((anp_mae - mlp_mae) / mlp_mae) * 100
                        f.write(f"MLP(all_θ) outperforms ANP by: {degradation:.2f}%\n")
                    
                    # Best specialist vs ANP
                    specialist_indices = [i for i, name in enumerate(model_names) 
                                        if name.startswith('MLP(θ=')]
                    if specialist_indices:
                        best_specialist_mae = np.nanmin([overall_mae[i] for i in specialist_indices])
                        f.write(f"\nBest specialist MAE: {best_specialist_mae:.4f}\n")
                        if anp_mae < best_specialist_mae:
                            improvement = ((best_specialist_mae - anp_mae) / best_specialist_mae) * 100
                            f.write(f"ANP improvement over best specialist: {improvement:.2f}%\n")
        
        print(f"\nSaved comparison summary to {summary_path}")
    
    def _create_cross_topology_context_summary(self, context_results: Dict):
        """Create summary comparing context sensitivity across topologies"""
        
        comparison_dir = self.output_dir / 'cross_topology_context_comparison'
        comparison_dir.mkdir(exist_ok=True)
        
        # Extract context percentages (assuming same for all)
        first_topology = next(iter(context_results.keys()))
        context_percentages = sorted(context_results[first_topology][0].keys())
        
        # Plot: All topologies on same graph
        fig, ax = plt.subplots(figsize=(14, 7))
        
        colors = {'ellipsoidal': '#2E86AB', 'random': '#A23B72', 'aligned': '#F18F01'}
        
        for topology, (overall_means, theta_results) in context_results.items():
            ctx_list = sorted(context_percentages)
            mae_list = [overall_means[ctx] for ctx in ctx_list]
            
            color = colors.get(topology, 'gray')
            ax.plot(ctx_list, mae_list, '-o', linewidth=2.5, markersize=6,
                   color=color, label=topology.capitalize(), alpha=0.8)
            
            # Mark best context
            best_ctx = min(overall_means.items(), key=lambda x: x[1])
            ax.plot(best_ctx[0], best_ctx[1], '*', markersize=12, 
                   color=color, zorder=10)
        
        ax.set_xlabel('Context Size (%)', fontsize=14)
        ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=14)
        ax.set_title('ANP Performance vs Context Size: All Topologies',
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12, loc='best')
        
        plt.tight_layout()
        plt.savefig(comparison_dir / 'all_topologies_context.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save numerical summary
        summary_path = comparison_dir / 'context_comparison_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("ANP Context Sensitivity - Cross-Topology Comparison\n")
            f.write("="*80 + "\n\n")
            
            for topology, (overall_means, theta_results) in context_results.items():
                f.write(f"\n{topology.upper()}\n")
                f.write("-"*50 + "\n")
                
                best_ctx = min(overall_means.items(), key=lambda x: x[1])
                f.write(f"Best overall context: {best_ctx[0]}% (MAE: {best_ctx[1]:.4f})\n")
                
                # MAE range across contexts
                min_mae = min(overall_means.values())
                max_mae = max(overall_means.values())
                f.write(f"MAE range: {min_mae:.4f} - {max_mae:.4f} "
                       f"(Δ={max_mae - min_mae:.4f})\n")
                
                # Sensitivity metric (how much context matters)
                sensitivity = (max_mae - min_mae) / min_mae * 100
                f.write(f"Context sensitivity: {sensitivity:.2f}%\n")
        
        print(f"\nSaved cross-topology context comparison to {comparison_dir}")
    
    def run(self, topologies: List[str] = None, evaluate_context: bool = True):
        """Run evaluation for specified topologies"""
        if topologies is None:
            topologies = ['ellipsoidal', 'random', 'aligned']
        
        all_results = {}
        context_results = {}
        
        for topology in topologies:
            try:
                # Standard evaluation
                results = self.evaluate_topology(topology)
                all_results[topology] = results
                
                # Context sensitivity analysis for ANP
                if evaluate_context:
                    print("\n" + "-"*60)
                    context_res = self.evaluate_anp_context_sensitivity(topology)
                    context_results[topology] = context_res
                
            except Exception as e:
                print(f"Error evaluating topology {topology}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if len(all_results) > 0:
            self.create_comparison_summary(all_results)
        
        # Create cross-topology context comparison
        if evaluate_context and len(context_results) > 0:
            self._create_cross_topology_context_summary(context_results)
        
        print("\n" + "="*60)
        print("Evaluation complete!")
        print(f"Results saved to: {self.output_dir}")
        print("="*60)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate models across topologies')
    parser.add_argument(
        '--data-dir',
        type=Path,
        required=True,
        help='Path to processed data directory'
    )
    parser.add_argument(
        '--mlp-dir',
        type=Path,
        required=True,
        help='Path to MLP results directory'
    )
    parser.add_argument(
        '--anp-dir',
        type=Path,
        required=True,
        help='Path to ANP results directory'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default='./results/evaluation_topologies',
        help='Path to save evaluation results'
    )
    parser.add_argument(
        '--context',
        type=int,
        default=40,
        help='Context percentage for ANP in main evaluation (default: 40)'
    )
    parser.add_argument(
        '--topologies',
        nargs='+',
        choices=['ellipsoidal', 'random', 'aligned'],
        default=['ellipsoidal', 'random', 'aligned'],
        help='Which topologies to evaluate'
    )
    parser.add_argument(
        '--skip-context-analysis',
        action='store_true',
        help='Skip ANP context sensitivity analysis'
    )
    
    args = parser.parse_args()
    
    evaluator = TopologyEvaluator(
        data_dir=args.data_dir,
        mlp_result_dir=args.mlp_dir,
        anp_result_dir=args.anp_dir,
        output_dir=args.output_dir,
        context_percent=args.context
    )
    
    evaluator.run(args.topologies, evaluate_context=not args.skip_context_analysis)


if __name__ == '__main__':
    main()