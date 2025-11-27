"""Visualization utilities."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


STRAIN_COMPONENTS = ['E_xx', 'E_yy', 'γ_xy']
STRESS_COMPONENTS = ['S_xx', 'S_yy', 'S_xy']


def _save_and_show(fig, save_path):
    """Save figure if path provided, then show."""
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_training_history(history, save_path=None):
    """Plot training and validation loss/MAE curves."""
    if hasattr(history, 'history'):
        history = history.history
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    axes[0].plot(history['loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation', linewidth=2)
    axes[0].set_title('Loss (MSE)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['mae'], label='Train', linewidth=2)
    axes[1].plot(history['val_mae'], label='Validation', linewidth=2)
    axes[1].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    _save_and_show(fig, save_path)


def plot_predictions(y_test, y_pred, save_path=None):
    """Plot true vs predicted stress components."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, (ax, comp) in enumerate(zip(axes, STRESS_COMPONENTS)):
        ax.scatter(y_test[:, i], y_pred[:, i], alpha=0.5, s=20, edgecolors='k', linewidths=0.5)
        
        min_val = min(y_test[:, i].min(), y_pred[:, i].min())
        max_val = max(y_test[:, i].max(), y_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
        
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        ax.set_xlabel(f'True {comp} (Pa)', fontsize=12)
        ax.set_ylabel(f'Predicted {comp} (Pa)', fontsize=12)
        ax.set_title(f'{comp} (R² = {r2:.3f})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    
    plt.tight_layout()
    _save_and_show(fig, save_path)


def plot_residuals(y_test, y_pred, save_path=None):
    """Plot residual analysis for each stress component."""
    residuals = y_test - y_pred
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, (ax, comp) in enumerate(zip(axes, STRESS_COMPONENTS)):
        ax.scatter(y_pred[:, i], residuals[:, i], alpha=0.5, s=20, edgecolors='k', linewidths=0.5)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero residual')
        ax.set_xlabel(f'Predicted {comp} (Pa)', fontsize=12)
        ax.set_ylabel('Residual (Pa)', fontsize=12)
        ax.set_title(f'{comp} Residuals', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    
    plt.tight_layout()
    _save_and_show(fig, save_path)


def plot_data_distribution(strains, stresses, save_path=None):
    """Plot distribution of strain and stress components."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    for i, comp in enumerate(STRAIN_COMPONENTS):
        axes[0, i].hist(strains[:, i], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0, i].set_title(f'Strain {comp}', fontsize=12, fontweight='bold')
        axes[0, i].set_xlabel('Value', fontsize=11)
        axes[0, i].set_ylabel('Frequency', fontsize=11)
        axes[0, i].grid(True, alpha=0.3)
    
    for i, comp in enumerate(STRESS_COMPONENTS):
        axes[1, i].hist(stresses[:, i], bins=50, edgecolor='black', alpha=0.7, color='coral')
        axes[1, i].set_title(f'Stress {comp}', fontsize=12, fontweight='bold')
        axes[1, i].set_xlabel('Value (Pa)', fontsize=11)
        axes[1, i].set_ylabel('Frequency', fontsize=11)
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    _save_and_show(fig, save_path)


def plot_component_metrics(metrics, save_path=None):
    """Plot per-component metrics comparison."""
    if 'components' not in metrics:
        print("No per-component metrics found")
        return
    
    components = list(metrics['components'].keys())
    metric_names = ['r2', 'rmse', 'mae']
    metric_labels = ['R²', 'RMSE (Pa)', 'MAE (Pa)']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, metric, label in zip(axes, metric_names, metric_labels):
        values = [metrics['components'][comp][metric] for comp in components]
        bars = ax.bar(components, values, color=['steelblue', 'coral', 'seagreen'], 
                      edgecolor='black', linewidth=1.5, alpha=0.8)
        ax.set_title(label, fontsize=14, fontweight='bold')
        ax.set_ylabel(label, fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}' if metric == 'r2' else f'{height:.1e}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    _save_and_show(fig, save_path)
