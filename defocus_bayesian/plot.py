"""
Visualization Module - Plot dose-response curves and acquisition functions
"""

from typing import Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from .model import SigmoidModel
from .acquisition import AcquisitionFunction
from .simulate import SubjectSimulator


def plot_dose_response_curve(
    model: SigmoidModel,
    simulator: Optional[SubjectSimulator] = None,
    measurements: Optional[List[Tuple[float, float]]] = None,
    dose_range: Tuple[float, float] = (0.5, 6.0),
    n_points: int = 100,
    figsize: Tuple[float, float] = (10, 6),
    title: str = "Dose-Response Curve",
) -> Figure:
    """
    Plot posterior dose-response curve.
    
    Args:
        model: Fitted Sigmoid model
        simulator: Simulator (for plotting true curve, optional)
        measurements: List of measurement points [(dose, response), ...]
        dose_range: Dose range
        n_points: Number of points for curve
        figsize: Figure size
        title: Chart title
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate dose grid
    doses = np.linspace(dose_range[0], dose_range[1], n_points)
    
    # Predict
    pred = model.predict(doses)
    mean = pred["mean"]
    lower = pred["lower"]
    upper = pred["upper"]
    
    # Plot posterior mean
    ax.plot(doses, mean, 'b-', linewidth=2, label='Posterior Mean')
    
    # Plot confidence interval
    ax.fill_between(doses, lower, upper, alpha=0.3, color='blue', label='95% CI')
    
    # Plot true curve (if provided)
    if simulator is not None:
        true_doses, true_responses = simulator.get_true_curve(dose_range, n_points)
        ax.plot(true_doses, true_responses, 'g--', linewidth=2, label='True Curve')
    
    # Plot measurement points
    if measurements is not None and len(measurements) > 0:
        doses_obs = [m[0] for m in measurements]
        responses_obs = [m[1] for m in measurements]
        ax.scatter(doses_obs, responses_obs, c='red', s=100, zorder=5, label='Observed Data')
    
    # Set labels and title
    ax.set_xlabel('Dose (D)', fontsize=12)
    ax.set_ylabel('Response (μm)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_acquisition_function(
    model: SigmoidModel,
    acquisition: AcquisitionFunction,
    strategy: str = "variance",
    y_best: Optional[float] = None,
    measurements: Optional[List[Tuple[float, float]]] = None,
    figsize: Tuple[float, float] = (10, 4),
    title: str = "Acquisition Function",
) -> Figure:
    """
    Plot acquisition function.
    
    Args:
        model: Fitted Sigmoid model
        acquisition: Acquisition function object
        strategy: Strategy ("variance" or "ei")
        y_best: Current best response value (needed for EI)
        measurements: List of measurement points
        figsize: Figure size
        title: Chart title
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get acquisition function values
    x_grid, acq_values = acquisition.get_acquisition_values(model, strategy, y_best)
    
    # Plot acquisition function
    ax.plot(x_grid, acq_values, 'purple', linewidth=2)
    ax.fill_between(x_grid, 0, acq_values, alpha=0.3, color='purple')
    
    # Mark maximum
    max_idx = np.argmax(acq_values)
    max_dose = x_grid[max_idx]
    max_value = acq_values[max_idx]
    ax.plot(max_dose, max_value, 'r*', markersize=15, label=f'Recommended: {max_dose:.2f} D')
    
    # Mark measured points
    if measurements is not None and len(measurements) > 0:
        doses_obs = [m[0] for m in measurements]
        ax.scatter(doses_obs, [0] * len(doses_obs), c='red', s=100, marker='|', 
                   zorder=5, label='Measured')
    
    # Set labels and title
    strategy_name = "Posterior Variance" if strategy == "variance" else "Expected Improvement"
    ax.set_xlabel('Dose (D)', fontsize=12)
    ax.set_ylabel('Acquisition Value', fontsize=12)
    ax.set_title(f"{title} ({strategy_name})", fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_learning_process(
    model: SigmoidModel,
    acquisition: AcquisitionFunction,
    simulator: Optional[SubjectSimulator] = None,
    measurements: Optional[List[Tuple[float, float]]] = None,
    strategy: str = "variance",
    y_best: Optional[float] = None,
    dose_range: Tuple[float, float] = (0.5, 6.0),
    figsize: Tuple[float, float] = (14, 5),
) -> Figure:
    """
    Plot combined learning process (dose-response curve + acquisition function).
    
    Args:
        model: Fitted Sigmoid model
        acquisition: Acquisition function object
        simulator: Simulator (optional)
        measurements: List of measurement points
        strategy: Acquisition strategy
        y_best: Current best response value
        dose_range: Dose range
        figsize: Figure size
        
    Returns:
        Matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left plot: Dose-response curve
    ax1 = axes[0]
    doses = np.linspace(dose_range[0], dose_range[1], 100)
    pred = model.predict(doses)
    
    ax1.plot(doses, pred["mean"], 'b-', linewidth=2, label='Posterior Mean')
    ax1.fill_between(doses, pred["lower"], pred["upper"], alpha=0.3, color='blue', label='95% CI')
    
    if simulator is not None:
        true_doses, true_responses = simulator.get_true_curve(dose_range, 100)
        ax1.plot(true_doses, true_responses, 'g--', linewidth=2, label='True Curve')
    
    if measurements is not None and len(measurements) > 0:
        doses_obs = [m[0] for m in measurements]
        responses_obs = [m[1] for m in measurements]
        ax1.scatter(doses_obs, responses_obs, c='red', s=100, zorder=5, label='Observed Data')
    
    ax1.set_xlabel('Dose (D)', fontsize=12)
    ax1.set_ylabel('Response (μm)', fontsize=12)
    ax1.set_title('Dose-Response Curve', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Acquisition function
    ax2 = axes[1]
    x_grid, acq_values = acquisition.get_acquisition_values(model, strategy, y_best)
    
    ax2.plot(x_grid, acq_values, 'purple', linewidth=2)
    ax2.fill_between(x_grid, 0, acq_values, alpha=0.3, color='purple')
    
    max_idx = np.argmax(acq_values)
    max_dose = x_grid[max_idx]
    max_value = acq_values[max_idx]
    ax2.plot(max_dose, max_value, 'r*', markersize=15, label=f'Recommended: {max_dose:.2f} D')
    
    if measurements is not None and len(measurements) > 0:
        doses_obs = [m[0] for m in measurements]
        ax2.scatter(doses_obs, [0] * len(doses_obs), c='red', s=100, marker='|', 
                   zorder=5, label='Measured')
    
    strategy_name = "Posterior Variance" if strategy == "variance" else "Expected Improvement"
    ax2.set_xlabel('Dose (D)', fontsize=12)
    ax2.set_ylabel('Acquisition Value', fontsize=12)
    ax2.set_title(f'Acquisition Function ({strategy_name})', fontsize=14)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_posterior_distribution(
    model: SigmoidModel,
    param_name: str = "threshold",
    figsize: Tuple[float, float] = (8, 5),
    title: Optional[str] = None,
) -> Figure:
    """
    Plot posterior distribution of a parameter.
    
    Args:
        model: Fitted Sigmoid model
        param_name: Parameter name
        figsize: Figure size
        title: Chart title
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get posterior samples
    posterior = model.trace.posterior[param_name].values.reshape(-1)
    
    # Plot histogram
    ax.hist(posterior, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Plot mean line
    mean_val = np.mean(posterior)
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
    
    # Plot median line
    median_val = np.median(posterior)
    ax.axvline(median_val, color='green', linestyle=':', linewidth=2, label=f'Median: {median_val:.3f}')
    
    # Set labels
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
    """
    Plot posterior distributions of all parameters.
    
    Args:
        model: Fitted Sigmoid model
        figsize: Figure size
        
    Returns:
        Matplotlib Figure object
    """
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
    
    # Hide extra subplot
    axes[-1].axis('off')
    
    plt.suptitle('Parameter Posterior Distributions', fontsize=16, y=1.02)
    plt.tight_layout()
    return fig


def plot_simulation_results(
    results: List[dict],
    figsize: Tuple[float, float] = (14, 10),
) -> Figure:
    """
    Plot simulation study results.
    
    Args:
        results: List of simulation results
        figsize: Figure size
        
    Returns:
        Matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Extract data
    n_measurements = [r["n_measurements"] for r in results]
    errors = [r["threshold_error"] for r in results]
    true_thresholds = [r["true_threshold"] for r in results]
    estimated_thresholds = [r["estimated_threshold"] for r in results]
    
    # Plot 1: Distribution of number of measurements
    ax1 = axes[0, 0]
    ax1.hist(n_measurements, bins=range(2, max(n_measurements)+2), alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Number of Measurements', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Measurement Count', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Distribution of threshold estimation error
    ax2 = axes[0, 1]
    ax2.hist(errors, bins=30, alpha=0.7, color='coral', edgecolor='black')
    ax2.set_xlabel('Threshold Estimation Error (D)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Estimation Error', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: True vs Estimated threshold
    ax3 = axes[1, 0]
    ax3.scatter(true_thresholds, estimated_thresholds, alpha=0.6, c='green')
    
    # Add diagonal line
    min_val = min(min(true_thresholds), min(estimated_thresholds))
    max_val = max(max(true_thresholds), max(estimated_thresholds))
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal')
    
    ax3.set_xlabel('True Threshold (D)', fontsize=12)
    ax3.set_ylabel('Estimated Threshold (D)', fontsize=12)
    ax3.set_title('True vs Estimated Threshold', fontsize=14)
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Number of measurements vs Error
    ax4 = axes[1, 1]
    ax4.scatter(n_measurements, errors, alpha=0.6, c='purple')
    ax4.set_xlabel('Number of Measurements', fontsize=12)
    ax4.set_ylabel('Threshold Estimation Error (D)', fontsize=12)
    ax4.set_title('Measurements vs Estimation Error', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Simulation Study Results', fontsize=16, y=1.02)
    plt.tight_layout()
    return fig


def save_figure(fig: Figure, filepath: str, dpi: int = 150) -> None:
    """
    Save figure to file.
    
    Args:
        fig: Matplotlib Figure object
        filepath: Save path
        dpi: Resolution
    """
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"Figure saved to: {filepath}")
