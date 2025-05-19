from matplotlib import pyplot as plt

# src/utils/plot_curves.py:

def plot_loss_curve(mse_values, save_path, mse_val=None):
    plt.figure()
    plt.plot(range(1, len(mse_values) + 1), mse_values, label='MSE Loss')
    if mse_val is not None:
        plt.plot(range(1, len(mse_val) + 1), mse_val, label='Validation MSE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('MSE Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"MSE Loss curve saved at {save_path}")


def plot_mae_curve(mae_values, save_path, mae_val=None):
    plt.figure()
    plt.plot(range(1, len(mae_values) + 1), mae_values, label='MAE')
    if mae_val is not None:
        plt.plot(range(1, len(mae_val) + 1), mae_val, label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('MAE Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"MAE curve saved at {save_path}")