"""Model comparison utilities."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pandas as pd


STRESS_COMPONENTS = ['S_xx', 'S_yy', 'S_xy']
COLOR_BASELINE = 'steelblue'
COLOR_PANN = 'coral'


def _save_and_show(fig, save_path):
    """Save figure if path provided, then show."""
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def _get_limits(y_test, y_pred, index):
    """Get min/max values for axis limits."""
    min_val = min(y_test[:, index].min(), y_pred[:, index].min())
    max_val = max(y_test[:, index].max(), y_pred[:, index].max())
    return min_val, max_val


def compare_predictions(y_test, y_pred_base, y_pred_pann, save_path=None):
    """Compare baseline and PANN predictions side-by-side."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    for i, comp in enumerate(STRESS_COMPONENTS):
        axes[i, 0].scatter(y_test[:, i], y_pred_base[:, i], alpha=0.4, s=15,
                          c=COLOR_BASELINE, edgecolors='none', label='Baseline')
        min_val, max_val = _get_limits(y_test, y_pred_base, i)
        axes[i, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        r2_base = r2_score(y_test[:, i], y_pred_base[:, i])
        axes[i, 0].set_title(f'Baseline - {comp} (R² = {r2_base:.3f})', fontsize=13, fontweight='bold')
        axes[i, 0].set_xlabel(f'True {comp} (Pa)', fontsize=11)
        axes[i, 0].set_ylabel(f'Predicted {comp} (Pa)', fontsize=11)
        axes[i, 0].grid(True, alpha=0.3)
        
        axes[i, 1].scatter(y_test[:, i], y_pred_pann[:, i], alpha=0.4, s=15,
                          c=COLOR_PANN, edgecolors='none', label='PANN')
        min_val, max_val = _get_limits(y_test, y_pred_pann, i)
        axes[i, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        r2_pann = r2_score(y_test[:, i], y_pred_pann[:, i])
        axes[i, 1].set_title(f'PANN - {comp} (R² = {r2_pann:.3f})', fontsize=13, fontweight='bold')
        axes[i, 1].set_xlabel(f'True {comp} (Pa)', fontsize=11)
        axes[i, 1].set_ylabel(f'Predicted {comp} (Pa)', fontsize=11)
        axes[i, 1].grid(True, alpha=0.3)
        
        axes[i, 2].scatter(y_test[:, i], y_pred_base[:, i], alpha=0.4, s=15,
                          c=COLOR_BASELINE, edgecolors='none', label='Baseline')
        axes[i, 2].scatter(y_test[:, i], y_pred_pann[:, i], alpha=0.4, s=15,
                          c=COLOR_PANN, edgecolors='none', label='PANN')
        min_val = min(y_test[:, i].min(), y_pred_base[:, i].min(), y_pred_pann[:, i].min())
        max_val = max(y_test[:, i].max(), y_pred_base[:, i].max(), y_pred_pann[:, i].max())
        axes[i, 2].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        axes[i, 2].set_title(f'Comparison - {comp}', fontsize=13, fontweight='bold')
        axes[i, 2].set_xlabel(f'True {comp} (Pa)', fontsize=11)
        axes[i, 2].set_ylabel(f'Predicted {comp} (Pa)', fontsize=11)
        axes[i, 2].legend(fontsize=10)
        axes[i, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    _save_and_show(fig, save_path)


def compare_residuals(y_test, y_pred_base, y_pred_pann, save_path=None):
    """Compare residual distributions for baseline and PANN."""
    residuals_base = y_test - y_pred_base
    residuals_pann = y_test - y_pred_pann
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (ax, comp) in enumerate(zip(axes, STRESS_COMPONENTS)):
        ax.hist(residuals_base[:, i], bins=50, alpha=0.6, color=COLOR_BASELINE,
               label='Baseline', edgecolor='black', linewidth=0.5)
        ax.hist(residuals_pann[:, i], bins=50, alpha=0.6, color=COLOR_PANN,
               label='PANN', edgecolor='black', linewidth=0.5)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero error')
        
        std_base = np.std(residuals_base[:, i])
        std_pann = np.std(residuals_pann[:, i])
        
        ax.set_title(f'{comp} Residuals\nσ_base={std_base:.2e}, σ_PANN={std_pann:.2e}',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Residual (Pa)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    _save_and_show(fig, save_path)


def compare_metrics(metrics_base, metrics_pann, save_path=None):
    """Compare metrics between baseline and PANN."""
    components = list(metrics_base['components'].keys())
    metric_names = ['r2', 'rmse', 'mae']
    metric_labels = ['R²', 'RMSE (Pa)', 'MAE (Pa)']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, metric, label in zip(axes, metric_names, metric_labels):
        values_base = [metrics_base['components'][comp][metric] for comp in components]
        values_pann = [metrics_pann['components'][comp][metric] for comp in components]
        
        x = np.arange(len(components))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, values_base, width, label='Baseline',
                      color=COLOR_BASELINE, edgecolor='black', linewidth=1.5, alpha=0.8)
        bars2 = ax.bar(x + width/2, values_pann, width, label='PANN',
                      color=COLOR_PANN, edgecolor='black', linewidth=1.5, alpha=0.8)
        
        ax.set_xlabel('Stress Component', fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(f'{label} Comparison', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(components)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}' if metric == 'r2' else f'{height:.1e}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    _save_and_show(fig, save_path)


def generate_comparison_summary(metrics_base, metrics_pann):
    """Generate summary table comparing baseline and PANN."""
    data = {
        'Metric': ['R²', 'RMSE (Pa)', 'MAE (Pa)'],
        'Baseline': [
            f"{metrics_base['r2']:.4f}",
            f"{metrics_base['rmse']:.2e}",
            f"{metrics_base['mae']:.2e}"
        ],
        'PANN': [
            f"{metrics_pann['r2']:.4f}",
            f"{metrics_pann['rmse']:.2e}",
            f"{metrics_pann['mae']:.2e}"
        ]
    }
    
    df = pd.DataFrame(data)
    
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60 + "\n")
    
    return df


def plot_error_distribution_comparison(y_test, y_pred_base, y_pred_pann, save_path=None):
    """Plot error distribution comparison with box plots."""
    errors_base = np.abs(y_test - y_pred_base)
    errors_pann = np.abs(y_test - y_pred_pann)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (ax, comp) in enumerate(zip(axes, STRESS_COMPONENTS)):
        data = [errors_base[:, i], errors_pann[:, i]]
        bp = ax.boxplot(data, labels=['Baseline', 'PANN'],
                        patch_artist=True, widths=0.6)
        
        colors = [COLOR_BASELINE, COLOR_PANN]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title(f'{comp} Absolute Error Distribution', fontsize=13, fontweight='bold')
        ax.set_ylabel('Absolute Error (Pa)', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    _save_and_show(fig, save_path)
