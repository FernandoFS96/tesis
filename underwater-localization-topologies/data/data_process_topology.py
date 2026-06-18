import numpy as np
import os
import re
import argparse
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm

'''
Use:
    python data_process_topology.py --mode separate

    python data_process_topology.py --data-dir ./data --mode separate --theta-range 0.0 0.6
'''

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
def process_and_save_data(input_paths, output_paths, save_dir, theta_values, topology_labels, split=0.2):
    """
    Process data with topology information
    topology_labels: list of (theta, topology) tuples for each data path
    """
    loaded_train_data, loaded_val_data, loaded_test_data = [], [], []
    train_thetas, val_thetas, test_thetas = [], [], []
    train_topologies, val_topologies, test_topologies = [], [], []

    print(f"Loading, reshaping and splitting data with split={split}...")
    for i, (inp, out) in tqdm(enumerate(zip(input_paths, output_paths)), 
                               total=len(input_paths), leave=False):
        X = reshape_input_data(np.load(inp))
        y = reshape_output_data(np.load(out))

        # 70% train / 30% temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, train_size=0.7, random_state=18, shuffle=True
        )

        # del 30% temporal, 2/3 -> val (20% global), 1/3 -> test (10% global)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=1/3, random_state=19, shuffle=True
        )

        theta, topology = topology_labels[i]

        # ANP: keep sequences intact
        for j in range(X_train.shape[0]):
            loaded_train_data.append([X_train[j], y_train[j]])
            train_thetas.append(theta)
            train_topologies.append(topology)
        for j in range(X_val.shape[0]):
            loaded_val_data.append([X_val[j], y_val[j]])
            val_thetas.append(theta)
            val_topologies.append(topology)
        for j in range(X_test.shape[0]):
            loaded_test_data.append([X_test[j], y_test[j]])
            test_thetas.append(theta)
            test_topologies.append(topology)

    # Shuffle ANP training set
    idx = np.random.permutation(len(loaded_train_data))
    loaded_train_data = [loaded_train_data[i] for i in idx]
    train_thetas = [train_thetas[i] for i in idx]
    train_topologies = [train_topologies[i] for i in idx]

    os.makedirs(save_dir, exist_ok=True)

    # Save to pickle - ANP
    with open(os.path.join(save_dir, 'train_data.pkl'), 'wb') as f:
        pickle.dump(loaded_train_data, f)
    with open(os.path.join(save_dir, 'val_data.pkl'), 'wb') as f:
        pickle.dump(loaded_val_data, f)
    with open(os.path.join(save_dir, 'test_data.pkl'), 'wb') as f:
        pickle.dump(loaded_test_data, f)
    with open(os.path.join(save_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump({
            'train_thetas': train_thetas, 
            'val_thetas': val_thetas,
            'test_thetas': test_thetas,
            'train_topologies': train_topologies,
            'val_topologies': val_topologies,
            'test_topologies': test_topologies
        }, f)

    print(f"Processed data saved to: {save_dir}")
    print(f"  Total training samples: {len(loaded_train_data)}")
    print(f"  Total validation samples: {len(loaded_val_data)}")
    print(f"  Total test samples: {len(loaded_test_data)}")


def detect_variance_group(channel_options):
    """Detect if channel options belong to low or high variance group"""
    theta_values = [float(opt) for opt in channel_options]
    max_theta = max(theta_values)
    
    if max_theta <= 0.3:
        return 'low_variance'
    elif min(theta_values) >= 0.4:
        return 'high_variance'
    else:
        return 'mixed'


def find_data_directory(base_dir, variance_group=None):
    """Find the actual data directory, handling both structures"""
    # Check if data is in variance subdirectory
    if variance_group:
        variance_dir = os.path.join(base_dir, variance_group)
        if os.path.isdir(variance_dir):
            return variance_dir
    
    # Check if data is directly in base_dir
    channel_dirs = [d for d in os.listdir(base_dir)
                   if os.path.isdir(os.path.join(base_dir, d)) 
                   and re.match(r'channel_option_[0-9.]+', d)]
    
    if channel_dirs:
        return base_dir
    
    return None


def process_separate_topologies(channel_options, base_dir, save_base_dir, split=0.2):
    """Process each topology separately"""
    topologies = ['ellipsoidal', 'random', 'aligned']
    
    for topology in topologies:
        print(f"\n=== Processing topology: {topology} ===")
        
        input_paths = []
        output_paths = []
        theta_values = []
        topology_labels = []
        
        # Collect paths for this topology
        for option in channel_options:
            topology_dir = os.path.join(base_dir, f'channel_option_{option}', topology)
            filtered_path = os.path.join(topology_dir, 'filtered_data', 'filtered_data.npy')
            traj_path = os.path.join(topology_dir, 'trajectory', 'trajectories.npy')
            
            if os.path.exists(filtered_path) and os.path.exists(traj_path):
                input_paths.append(filtered_path)
                output_paths.append(traj_path)
                theta_values.append(float(option))
                topology_labels.append((float(option), topology))
            else:
                print(f"  Warning: Missing data for option {option}, topology {topology}")
        
        if input_paths:
            # Process this topology
            topology_save_dir = os.path.join(save_base_dir, f'topology_{topology}')
            process_and_save_data(input_paths, output_paths, topology_save_dir, 
                                theta_values, topology_labels, split=split)
            print(f"  Saved to: {topology_save_dir}")


def process_combined_topologies(channel_options, base_dir, save_base_dir, split=0.2):
    """Process all topologies combined into one dataset"""
    topologies = ['ellipsoidal', 'random', 'aligned']
    
    print(f"\n=== Processing combined topologies ===")
    
    input_paths = []
    output_paths = []
    theta_values = []
    topology_labels = []
    
    # Collect paths for all topologies
    for option in channel_options:
        for topology in topologies:
            topology_dir = os.path.join(base_dir, f'channel_option_{option}', topology)
            filtered_path = os.path.join(topology_dir, 'filtered_data', 'filtered_data.npy')
            traj_path = os.path.join(topology_dir, 'trajectory', 'trajectories.npy')
            
            if os.path.exists(filtered_path) and os.path.exists(traj_path):
                input_paths.append(filtered_path)
                output_paths.append(traj_path)
                theta_values.append(float(option))
                topology_labels.append((float(option), topology))
            else:
                print(f"  Warning: Missing data for option {option}, topology {topology}")
    
    if input_paths:
        # Process combined dataset
        combined_save_dir = os.path.join(save_base_dir, 'combined_topologies')
        process_and_save_data(input_paths, output_paths, combined_save_dir, 
                            theta_values, topology_labels, split=split)
        print(f"  Saved to: {combined_save_dir}")


def process_comparison_dataset(channel_options, base_dir, save_base_dir, split=0.2):
    """Create a special dataset for comparing topologies"""
    topologies = ['ellipsoidal', 'random', 'aligned']
    
    print(f"\n=== Creating comparison dataset ===")
    
    comparison_data = []
    
    for option in channel_options:
        option_data = {'theta': float(option), 'topologies': {}}
        
        # Check if all topologies exist for this option
        all_exist = True
        for topology in topologies:
            topology_dir = os.path.join(base_dir, f'channel_option_{option}', topology)
            filtered_path = os.path.join(topology_dir, 'filtered_data', 'filtered_data.npy')
            traj_path = os.path.join(topology_dir, 'trajectory', 'trajectories.npy')
            
            if not (os.path.exists(filtered_path) and os.path.exists(traj_path)):
                all_exist = False
                break
        
        if all_exist:
            # Load data for all topologies
            for topology in topologies:
                topology_dir = os.path.join(base_dir, f'channel_option_{option}', topology)
                filtered_path = os.path.join(topology_dir, 'filtered_data', 'filtered_data.npy')
                traj_path = os.path.join(topology_dir, 'trajectory', 'trajectories.npy')
                
                X = reshape_input_data(np.load(filtered_path))
                y = reshape_output_data(np.load(traj_path))
                
                option_data['topologies'][topology] = {'X': X, 'y': y}
            
            comparison_data.append(option_data)
    
    # Save comparison dataset
    comparison_save_dir = os.path.join(save_base_dir, 'comparison_dataset')
    os.makedirs(comparison_save_dir, exist_ok=True)
    
    with open(os.path.join(comparison_save_dir, 'comparison_data.pkl'), 'wb') as f:
        pickle.dump(comparison_data, f)
    
    print(f"  Comparison dataset saved to: {comparison_save_dir}")
    print(f"  Total theta values with all topologies: {len(comparison_data)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process .npy data with multiple sensor topologies'
    )
    parser.add_argument(
        '--data-dir', default='./data',
        help='Base data directory (default: ./data)'
    )
    parser.add_argument(
        '--variance', choices=['low_variance', 'high_variance', 'auto'], 
        default='auto',
        help='Variance group (auto will detect based on theta values)'
    )
    parser.add_argument(
        '--mode', choices=['separate', 'combined', 'comparison', 'all'], 
        default='all',
        help='Processing mode: separate topologies, combined, comparison dataset, or all'
    )
    parser.add_argument(
        '--split', type=float, default=0.2,
        help='Validation split fraction (default 0.2)'
    )
    parser.add_argument(
        '--save-dir', default=None,
        help='Output directory (default: auto-generated based on data location)'
    )
    parser.add_argument(
        '--theta-range', nargs=2, type=float, default=None,
        help='Specify theta range to process (e.g., 0.0 0.5 for low variance)'
    )
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"Data Processing with Topologies")
    print(f"{'='*60}")
    
    # Find data directory
    data_dir = find_data_directory(args.data_dir, args.variance)
    
    if data_dir is None:
        print(f"Error: Could not find data in {args.data_dir}")
        if args.variance != 'auto':
            print(f"Also checked {os.path.join(args.data_dir, args.variance)}")
        exit(1)
    
    print(f"Found data in: {data_dir}")
    
    # Detect channel options
    channel_dirs = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d)) 
                   and re.match(r'channel_option_[0-9.]+', d)]
    
    channel_options = sorted([
        d.split('channel_option_')[1] for d in channel_dirs
    ], key=float)
    
    # Filter by theta range if specified
    if args.theta_range:
        min_theta, max_theta = args.theta_range
        channel_options = [opt for opt in channel_options 
                          if min_theta <= float(opt) <= max_theta]
        print(f"Filtering to theta range [{min_theta}, {max_theta}]")
    
    print(f"Found {len(channel_options)} channel options: {channel_options}")
    
    # Auto-detect variance group if needed
    if args.variance == 'auto':
        variance_group = detect_variance_group(channel_options)
        print(f"Auto-detected variance group: {variance_group}")
    else:
        variance_group = args.variance
    
    # Set output directory
    if args.save_dir:
        save_base_dir = args.save_dir
    else:
        save_base_dir = os.path.join(data_dir, f'data_processed_topologies_{variance_group}')
    
    print(f"Output directory: {save_base_dir}")
    
    # Process based on mode
    if args.mode in ['separate', 'all']:
        process_separate_topologies(channel_options, data_dir, save_base_dir, args.split)
    
    if args.mode in ['combined', 'all']:
        process_combined_topologies(channel_options, data_dir, save_base_dir, args.split)
    
    if args.mode in ['comparison', 'all']:
        process_comparison_dataset(channel_options, data_dir, save_base_dir, args.split)
    
    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)