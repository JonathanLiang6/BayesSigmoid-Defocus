"""
基础可视化模块 - 基础绘图函数（剂量反应曲线、后验分布等）
"""

from typing import Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .model import SigmoidModel
from .simulate import SubjectSimulator

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_dose_response_curve(
    model: SigmoidModel,
    simulator: Optional[SubjectSimulator] = None,
    measurements: Optional[List[Tuple[float, float]]] = None,
    dose_range: Tuple[float, float] = (0.5, 6.0),
    n_points: int = 100,
    figsize: Tuple[float, float] = (10, 6),
    title: str = "Dose-Response Curve",
) -> Figure:
    fig, ax = plt.subplots(figsize=figsize)
    doses = np.linspace(dose_range[0], dose_range[1], n_points)
    pred = model.predict(doses)
    ax.plot(doses, pred["mean"], 'b-', linewidth=2, label='Posterior Mean')
    ax.fill_between(doses, pred["lower"], pred["upper"], alpha=0.3, color='blue', label='95% CI')
    
    if simulator is not None:
        true_doses, true_responses = simulator.get_true_curve(dose_range, n_points)
        ax.plot(true_doses, true_responses, 'g--', linewidth=2, label='True Curve')
    
    if measurements is not None and len(measurements) > 0:
        doses_obs = [m[0] for m in measurements]
        responses_obs = [m[1] for m in measurements]
        ax.scatter(doses_obs, responses_obs, c='red', s=100, zorder=5, label='Observed Data')
    
    ax.set_xlabel('Dose (D)', fontsize=12)
    ax.set_ylabel('Response (μm)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_posterior_distribution(
    model: SigmoidModel,
    param_name: str = "threshold",
    figsize: Tuple[float, float] = (8, 5),
    title: Optional[str] = None,
) -> Figure:
    fig, ax = plt.subplots(figsize=figsize)
    posterior = model.trace.posterior[param_name].values.reshape(-1)
    ax.hist(posterior, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    
    mean_val = np.mean(posterior)
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
    
    median_val = np.median(posterior)
    ax.axvline(median_val, color='green', linestyle=':', linewidth=2, label=f'Median: {median_val:.3f}')
    
    param_labels = {
        "baseline": "Baseline",
        "max_response": "Max Response",
        "slope": "Slope",
        "threshold": "Threshold (ED50)",
        "sigma": "Noise Std (sigma)",
    }
    ax.set_xlabel(param_labels.get(param_name, param_name), fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title or f"Posterior Distribution of {param_labels.get(param_name, param_name)}", fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig


def plot_all_posteriors(
    model: SigmoidModel,
    figsize: Tuple[float, float] = (15, 10),
) -> Figure:
    param_names = ["baseline", "max_response", "slope", "threshold", "sigma"]
    param_labels = {
        "baseline": "Baseline",
        "max_response": "Max Response",
        "slope": "Slope",
        "threshold": "Threshold (ED50)",
        "sigma": "Noise Std (sigma)",
    }
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    for i, param in enumerate(param_names):
        ax = axes[i]
        posterior = model.trace.posterior[param].values.reshape(-1)
        ax.hist(posterior, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
        mean_val = np.mean(posterior)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax.set_xlabel(param_labels[param], fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    axes[-1].axis('off')
    plt.suptitle('Parameter Posterior Distributions', fontsize=16, y=1.02)
    plt.tight_layout()
    return fig


def plot_simulation_results(
    results: List[dict],
    figsize: Tuple[float, float] = (14, 10),
) -> Figure:
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    n_measurements = [r["n_measurements"] for r in results]
    errors = [r["threshold_error"] for r in results]
    true_thresholds = [r["true_threshold"] for r in results]
    estimated_thresholds = [r["estimated_threshold"] for r in results]
    
    ax1 = axes[0, 0]
    ax1.hist(n_measurements, bins=range(2, max(n_measurements)+2), alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Number of Measurements', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Measurement Count', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax2 = axes[0, 1]
    ax2.hist(errors, bins=30, alpha=0.7, color='coral', edgecolor='black')
    ax2.set_xlabel('Threshold Estimation Error (D)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Estimation Error', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    ax3 = axes[1, 0]
    ax3.scatter(true_thresholds, estimated_thresholds, alpha=0.6, c='green')
    min_val = min(min(true_thresholds), min(estimated_thresholds))
    max_val = max(max(true_thresholds), max(estimated_thresholds))
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal')
    ax3.set_xlabel('True Threshold (D)', fontsize=12)
    ax3.set_ylabel('Estimated Threshold (D)', fontsize=12)
    ax3.set_title('True vs Estimated Threshold', fontsize=14)
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    ax4.scatter(n_measurements, errors, alpha=0.6, c='purple')
    ax4.set_xlabel('Number of Measurements', fontsize=12)
    ax4.set_ylabel('Threshold Estimation Error (D)', fontsize=12)
    ax4.set_title('Measurements vs Estimation Error', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Simulation Study Results', fontsize=16, y=1.02)
    plt.tight_layout()
    return fig


def save_figure(fig: Figure, filepath: str, dpi: int = 300) -> None:
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')


__all__ = [
    'plot_dose_response_curve',
    'plot_posterior_distribution',
    'plot_all_posteriors',
    'plot_simulation_results',
    'save_figure',
]
