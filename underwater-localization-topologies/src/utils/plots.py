import os
import pickle
import matplotlib.pyplot as plt

def plot_training_metrics(metrics_path, output_path=None):
    """
    Plots training vs validation loss (MSE) and MAE over epochs in a 2x1 grid.

    Args:
        metrics_path (str): Path to metrics.pkl containing keys:
            'train_loss', 'val_loss', 'train_mae', 'val_mae'.
        output_path (str, optional): Where to save the resulting figure (png).
                                     If None, shows the plot interactively.
    """
    # Load metrics
    with open(metrics_path, 'rb') as f:
        metrics = pickle.load(f)

    train_loss = metrics.get('train_loss', [])
    val_loss   = metrics.get('val_loss', [])
    train_mae  = metrics.get('train_mae', [])
    val_mae    = metrics.get('val_mae', [])
    epochs = range(1, len(train_loss) + 1)

    # Create 2x1 subplot grid
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))

    # Plot Loss (MSE) if available
    axes[0].plot(epochs, train_loss, label='Train Loss (MSE)')
    axes[0].plot(epochs, val_loss,   label='Val Loss (MSE)')
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].legend()
    axes[0].grid(True)

    # Plot MAE
    axes[1].plot(epochs, train_mae, label='Train MAE')
    axes[1].plot(epochs, val_mae,   label='Val MAE')
    axes[1].set_title('Training and Validation MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close(fig)
    else:
        plt.show()