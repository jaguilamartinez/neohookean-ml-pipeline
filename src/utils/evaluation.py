import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(y_true, y_pred):
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }


def plot_predictions(y_true, y_pred, title="Predictions", save_path=None):
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    
    n_outputs = y_true.shape[1]
    components = ['xx', 'yy', 'xy'] if n_outputs == 3 else [str(i) for i in range(n_outputs)]
    
    fig, axes = plt.subplots(1, n_outputs, figsize=(5 * n_outputs, 5))
    if n_outputs == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.6, s=20)
        
        # Perfect prediction line
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        ax.set_xlabel(f'True stress_{components[i]}')
        ax.set_ylabel(f'Predicted stress_{components[i]}')
        ax.set_title(f'{title} - {components[i]} (R² = {r2:.3f})')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_residuals(y_true, y_pred, title="Residuals", save_path=None):
    residuals = y_true - y_pred
    
    if y_true.ndim == 1:
        residuals = residuals.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    
    n_outputs = residuals.shape[1]
    components = ['xx', 'yy', 'xy'] if n_outputs == 3 else [str(i) for i in range(n_outputs)]
    
    fig, axes = plt.subplots(1, n_outputs, figsize=(5 * n_outputs, 5))
    if n_outputs == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        ax.scatter(y_pred[:, i], residuals[:, i], alpha=0.6, s=20)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        ax.set_xlabel(f'Predicted stress_{components[i]}')
        ax.set_ylabel('Residuals')
        ax.set_title(f'{title} - {components[i]}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_training_history(history, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Training')
    axes[0].plot(history.history['val_loss'], label='Validation')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE')
    axes[0].legend()
    axes[0].grid(True)
    
    # MAE
    axes[1].plot(history.history['mae'], label='Training')
    axes[1].plot(history.history['val_mae'], label='Validation')
    axes[1].set_title('Mean Absolute Error')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()