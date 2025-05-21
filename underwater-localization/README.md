# Underwater Localization Project

## Overview
This project explores the use of Machine Learning models, specifically **Attentive Neural Processes (ANP)** and **Multilayer Perceptrons (MLP)**, for underwater localization. The dataset comes from an underwater channel simulation model, generating trajectories (targets) and filtered data (inputs). The project involves training, evaluating, and comparing these models across different conditions determined by a parameter `theta`, representing variance levels.

## Project Structure
The project follows a modular design for maintainability and scalability:

```
underwater-localization/
│
├── data/
│   ├── high_variance/           # Raw and processed data
│   │   ├── channel_option_x.x/
│   │   ├── ...
│   │   └── data_processed/
│   ├── low_variance/
│   └── data_generator_ANP.py    # Geenrates raw data
│
├── src/                   # Source code
│   ├── models/            # Model architectures
│   │   ├── mlp.py
│   │   └── anp.py
│   ├── preprocessing/     # Data preprocessing utilities
│   │   ├── process_data.py
│   │   └── __init__.py
│   ├── training/          # Training scripts
│   │   ├── train_mlp.py
│   │   └── train_anp.py
│   ├── evaluation/        # Evaluation scripts and utilities
│   │   ├── evaluate.py
│   │   └── eval_utils.py
│   ├── utils/             # Helper utilities
│   │   ├── nav_dataset.py
│   │   ├── filter_utils.py
│   │   ├── plot_curves.py
│   │   └── evaluation.py
│
├── results/               # Experiment results
│   ├── MLP/               # MLP experiment results
│   ├── ANP/               # ANP experiment results
│   └── evaluation/        # Evaluation plots and metrics
│
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
```
## Prepend the project root to the PYTHONPATH environment variable so that dhe dependancies is automatically found
- `export PYTHONPATH="$HOME/tesis/underwater-localization:$PYTHONPATH"`

## Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data
The data is organized by `theta` values, with subfolders containing filtered input data and trajectories:
- `data/low_data/channel_option_x.x/filtered_data.npy`
- `data/low_data/channel_option_x.x/trajectory/trajectories.npy`

Processed data is stored in `data_processed/` for training and validation. If not present, use script `process_data.py` in `src/preprocessing/` with the following arguments:
- **subset**: `high_variance` or `low_variance`  
- **--split**: (optional) fraction for validation set (default: `0.2`)  
- **--save-dir**: (optional) custom output directory (default: `data/<subset>/data_processed`)

### Example

```bash
# Process high_variance with default 20% validation split
python src/preprocessing/process_data.py high_variance

# Process low_variance with 30% validation split
python src/preprocessing/process_data.py low_variance --split 0.3
```

## Usage
### Training Models
#### Train MLP
```bash
# Train on low-variance groups (θ 0.0–0.5)
python src/training/train_mlp.py \
--data-dir data/low_variance/ \
--result-dir results/MLP/low_variance \
--theta-range 0.0 0.5 \
--input-dim 4010 \
--output-dim 3

# Train on high-variance groups (θ 0.6–1.0)
python src/training/train_mlp.py \
--data-dir data/high_variance/ \
--result-dir results/MLP/high_variance\
--theta-range 0.6 1.0 \
--input-dim 4010 \
--output-dim 3
```

#### Train ANP
```bash
# Train on low-variance groups (θ 0.0–0.5)
python src/training/train_anp.py \
--train-data data/low_variance/data_processed/train_data_anp.pkl \
--val-data data/low_variance/data_processed/val_data_anp.pkl \
--result-dir results/ANP/low_variance \
--batch-size 8 \
--epochs 5000 \
--patience 200

# Train on high-variance groups (θ 0.6–1.0)
python src/training/train_anp.py \
--train-data data/high_variance/data_processed/train_data_anp.pkl \
--val-data data/high_variance/data_processed/val_data_anp.pkl \
--result-dir results/ANP/high_variance \
--batch-size 8 \
--epochs 5000 \
--patience 200
```

### Evaluate Models
#### Combined Evaluation
```bash
# Evaluación zero-shot sobre low-variance (θ 0.0–0.5)
python src/evaluation/evaluate.py \
--data-path data/low_variance/data_processed/val_data_anp.pkl \
--theta-path data/low_variance/data_processed/theta_values_anp.pkl \
--anp-path results/ANP/low_variance/experiment_20250430-115900/best_checkpoint.pth.tar \
--mlp-dir results/MLP/low_variance \
--result-dir results/evaluation/low_variance \
--batch-size 4  \
--eval-modes mean heatmap pvals trajectories

# Evaluación zero-shot sobre high-variance (θ 0.6–1.0)
python src/evaluation/evaluate.py \
--data-path data/high_variance/data_processed/val_data_anp.pkl \
--theta-path data/high_variance/data_processed/theta_values_anp.pkl \
--anp-path results/ANP/low_variance/experiment_20250430-115900/best_checkpoint.pth.tar \
--mlp-dir results/MLP/high_variance \
--result-dir results/evaluation/high_variance \
--batch-size 4  \
--eval-modes mean heatmap pvals trajectories
```

This evaluates the ANP and MLP models on the requested channel groups:
- **Heatmap**: Compares MAE for ANP (with varying context percentages) and MLP models.
- **Predictions**: Plots predicted trajectories for ANP and MLP (combined).
- Feel free to add any exta functionality or test that you consider

## Features
- **Dynamic Data Selection**: Train and evaluate models on different `theta` ranges.
- **Dynamic Context Percentages**: Evaluate ANP models with varying context sizes.
- **Evaluation Metrics**: Compute Mean Absolute Error (MAE) for ANP and MLP models.
- **Visualization**: Generate heatmaps and trajectory plots for result analysis.