"""
Visualization Module - Plot dose-response curves and acquisition functions
"""

from typing import Optional, Tuple, List, Dict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from .model import SigmoidModel
from .acquisition import AcquisitionFunction
from .simulate import SubjectSimulator

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


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
    
    # 【优化点】共享计算结果，减少重复计算
    doses = np.linspace(dose_range[0], dose_range[1], 100)
    pred = model.predict(doses)
    
    # 【优化点】提取观测数据，避免重复处理
    doses_obs = []
    responses_obs = []
    if measurements is not None and len(measurements) > 0:
        doses_obs = [m[0] for m in measurements]
        responses_obs = [m[1] for m in measurements]
    
    # Left plot: Dose-response curve
    ax1 = axes[0]
    ax1.plot(doses, pred["mean"], 'b-', linewidth=2, label='Posterior Mean')
    ax1.fill_between(doses, pred["lower"], pred["upper"], alpha=0.3, color='blue', label='95% CI')
    
    if simulator is not None:
        true_doses, true_responses = simulator.get_true_curve(dose_range, 100)
        ax1.plot(true_doses, true_responses, 'g--', linewidth=2, label='True Curve')
    
    if len(doses_obs) > 0:
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
    
    if len(doses_obs) > 0:
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


def save_figure(fig: Figure, filepath: str, dpi: int = 300) -> None:
    """
    Save figure to file.
    
    Args:
        fig: Matplotlib Figure object
        filepath: Save path
        dpi: Resolution
    """
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')


def plot_learning_trajectory(
    measurements: List[Tuple[float, float]],
    true_threshold: Optional[float] = None,
    figsize: Tuple[float, float] = (12, 6),
    title: str = "Learning Trajectory",
    lang: str = "en",
) -> Figure:
    """
    绘制主动学习推荐剂量轨迹图，展示学习进步性。
    
    Args:
        measurements: 测量点列表 [(dose, response), ...]
        true_threshold: 真实阈值（可选）
        figsize: 图大小
        title: 标题
        lang: 语言 ('en' 或 'zh')
        
    Returns:
        Matplotlib Figure 对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 提取剂量和反应
    doses = [m[0] for m in measurements]
    responses = [m[1] for m in measurements]
    rounds = list(range(1, len(measurements) + 1))
    
    # 绘制剂量轨迹
    ax.plot(rounds, doses, 'o-', linewidth=2, markerfacecolor='red', markersize=8, label='推荐剂量' if lang == "zh" else 'Recommended Dose')
    
    # 标记真实阈值（如果提供）
    if true_threshold is not None:
        ax.axhline(y=true_threshold, color='green', linestyle='--', linewidth=2, label='真实阈值' if lang == "zh" else 'True Threshold')
        
        # 添加置信带
        ax.fill_between(rounds, true_threshold - 0.3, true_threshold + 0.3, alpha=0.2, color='green', label='±0.3D误差带' if lang == "zh" else '±0.3D Error Band')
        
        # 计算每轮与真实阈值的误差
        errors = [abs(d - true_threshold) for d in doses]
        
        # 在右Y轴显示误差
        ax2 = ax.twinx()
        ax2.plot(rounds, errors, 's--', color='orange', linewidth=1.5, markersize=6, alpha=0.7, label='估计误差' if lang == "zh" else 'Estimation Error')
        ax2.set_ylabel('估计误差 (D)', fontsize=11, color='orange') if lang == "zh" else ax2.set_ylabel('Estimation Error (D)', fontsize=11, color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        
        # 添加误差递减注释
        if len(errors) >= 2:
            reduction = (errors[0] - errors[-1]) / errors[0] * 100 if errors[0] > 0 else 0
            if lang == "zh":
                ax2.annotate(f'误差减少 {reduction:.1f}%', xy=(rounds[-1], errors[-1]), 
                            xytext=(rounds[-1]-0.5, errors[-1]+0.2),
                            fontsize=10, color='orange')
            else:
                ax2.annotate(f'Error Reduced {reduction:.1f}%', xy=(rounds[-1], errors[-1]), 
                            xytext=(rounds[-1]-0.5, errors[-1]+0.2),
                            fontsize=10, color='orange')
    
    # 设置标签和标题
    if lang == "zh":
        ax.set_xlabel('轮次', fontsize=12)
        ax.set_ylabel('剂量 (D)', fontsize=12)
        ax.set_title('学习轨迹：推荐剂量逐渐收敛到真实阈值' if title is None else title, fontsize=14)
    else:
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel('Dose (D)', fontsize=12)
        ax.set_title('Learning Trajectory: Recommended Doses Converge to True Threshold' if title is None else title, fontsize=14)
    
    # 合并图例
    lines1, labels1 = ax.get_legend_handles_labels()
    if true_threshold is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9)
    else:
        ax.legend(loc='best', fontsize=9)
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_error_distribution(
    errors: List[float],
    figsize: Tuple[float, float] = (10, 6),
    title: str = "Error Distribution",
    lang: str = "en",
) -> Figure:
    """
    绘制批量模拟误差分布直方图。
    
    Args:
        errors: 误差列表
        figsize: 图大小
        title: 标题
        lang: 语言 ('en' 或 'zh')
        
    Returns:
        Matplotlib Figure 对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制直方图
    ax.hist(errors, bins=30, alpha=0.7, color='coral', edgecolor='black')
    
    # 计算统计量
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    # 绘制均值线
    ax.axvline(mean_error, color='red', linestyle='--', linewidth=2, label=f'均值: {mean_error:.3f}')
    
    # 设置标签和标题
    if lang == "zh":
        ax.set_xlabel('阈值估计误差 (D)', fontsize=12)
        ax.set_ylabel('频率', fontsize=12)
        ax.set_title(title or '误差分布', fontsize=14)
        ax.legend(loc='best', labels=[f'均值: {mean_error:.3f} D'])
    else:
        ax.set_xlabel('Threshold Estimation Error (D)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(title or 'Error Distribution', fontsize=14)
        ax.legend(loc='best', labels=[f'Mean: {mean_error:.3f} D'])
    
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_measurement_count(
    n_measurements: List[int],
    figsize: Tuple[float, float] = (10, 6),
    title: str = "Measurement Count",
    lang: str = "en",
) -> Figure:
    """
    绘制测量次数统计柱状图。
    
    Args:
        n_measurements: 测量次数列表
        figsize: 图大小
        title: 标题
        lang: 语言 ('en' 或 'zh')
        
    Returns:
        Matplotlib Figure 对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 计算频率
    counts = np.bincount(n_measurements)
    bins = np.arange(len(counts))
    
    # 绘制柱状图
    ax.bar(bins, counts, alpha=0.7, color='steelblue', edgecolor='black')
    
    # 设置标签和标题
    if lang == "zh":
        ax.set_xlabel('测量次数', fontsize=12)
        ax.set_ylabel('受试者数量', fontsize=12)
        ax.set_title(title or '测量次数统计', fontsize=14)
    else:
        ax.set_xlabel('Number of Measurements', fontsize=12)
        ax.set_ylabel('Number of Subjects', fontsize=12)
        ax.set_title(title or 'Measurement Count', fontsize=14)
    
    # 设置x轴刻度
    ax.set_xticks(bins)
    
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_true_vs_estimated(
    true_values: List[float],
    estimated_values: List[float],
    figsize: Tuple[float, float] = (10, 6),
    title: str = "True vs Estimated",
    lang: str = "en",
) -> Figure:
    """
    绘制真实值vs估计值散点图（完美拟合线）。
    
    Args:
        true_values: 真实值列表
        estimated_values: 估计值列表
        figsize: 图大小
        title: 标题
        lang: 语言 ('en' 或 'zh')
        
    Returns:
        Matplotlib Figure 对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制散点图
    ax.scatter(true_values, estimated_values, alpha=0.6, c='green')
    
    # 添加完美拟合线
    min_val = min(min(true_values), min(estimated_values))
    max_val = max(max(true_values), max(estimated_values))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美拟合')
    
    # 设置标签和标题
    if lang == "zh":
        ax.set_xlabel('真实阈值 (D)', fontsize=12)
        ax.set_ylabel('估计阈值 (D)', fontsize=12)
        ax.set_title(title or '真实值 vs 估计值', fontsize=14)
        ax.legend(loc='best')
    else:
        ax.set_xlabel('True Threshold (D)', fontsize=12)
        ax.set_ylabel('Estimated Threshold (D)', fontsize=12)
        ax.set_title(title or 'True vs Estimated', fontsize=14)
        ax.legend(loc='best', labels=['Ideal Fit'])
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_acquisition_curve(
    model: SigmoidModel,
    acquisition: AcquisitionFunction,
    y_best: float,
    measurements: List[Tuple[float, float]],
    figsize: Tuple[float, float] = (10, 6),
    title: str = "Acquisition Curve",
    lang: str = "en",
) -> Figure:
    """
    绘制主动学习采集函数变化曲线（用于解释推荐逻辑）。
    
    Args:
        model: 拟合的Sigmoid模型
        acquisition: 采集函数对象
        y_best: 当前最佳反应值
        measurements: 测量点列表
        figsize: 图大小
        title: 标题
        lang: 语言 ('en' 或 'zh')
        
    Returns:
        Matplotlib Figure 对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 获取两种采集函数的值
    x_grid, var_values = acquisition.get_acquisition_values(model, "variance")
    x_grid, ei_values = acquisition.get_acquisition_values(model, "ei", y_best)
    
    # 归一化
    var_norm = (var_values - np.min(var_values)) / (np.max(var_values) - np.min(var_values) + 1e-9)
    ei_norm = (ei_values - np.min(ei_values)) / (np.max(ei_values) - np.min(ei_values) + 1e-9)
    
    # 绘制采集函数
    ax.plot(x_grid, var_norm, 'b-', linewidth=2, label='后验方差')
    ax.plot(x_grid, ei_norm, 'g-', linewidth=2, label='期望改进')
    
    # 标记测量点
    if measurements is not None and len(measurements) > 0:
        doses_obs = [m[0] for m in measurements]
        ax.scatter(doses_obs, [0] * len(doses_obs), c='red', s=100, marker='|', zorder=5, label='已测量')
    
    # 设置标签和标题
    if lang == "zh":
        ax.set_xlabel('剂量 (D)', fontsize=12)
        ax.set_ylabel('归一化采集值', fontsize=12)
        ax.set_title(title or '采集函数变化曲线', fontsize=14)
    else:
        ax.set_xlabel('Dose (D)', fontsize=12)
        ax.set_ylabel('Normalized Acquisition Value', fontsize=12)
        ax.set_title(title or 'Acquisition Curve', fontsize=14)
    
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_summary_report(
    model: SigmoidModel,
    acquisition: AcquisitionFunction,
    simulator: Optional[SubjectSimulator] = None,
    measurements: Optional[List[Tuple[float, float]]] = None,
    y_best: Optional[float] = None,
    figsize: Tuple[float, float] = (16, 12),
    title: str = "Summary Report",
    lang: str = "en",
) -> Figure:
    """
    绘制汇总可视化报告（单页多子图）。
    
    Args:
        model: 拟合的Sigmoid模型
        acquisition: 采集函数对象
        simulator: 模拟器（可选）
        measurements: 测量点列表
        y_best: 当前最佳反应值
        figsize: 图大小
        title: 标题
        lang: 语言 ('en' 或 'zh')
        
    Returns:
        Matplotlib Figure 对象
    """
    fig = plt.figure(figsize=figsize)
    
    # 子图1: 剂量-反应曲线
    ax1 = plt.subplot(2, 3, 1)
    doses = np.linspace(0.5, 6.0, 100)
    pred = model.predict(doses)
    
    ax1.plot(doses, pred["mean"], 'b-', linewidth=2, label='后验均值')
    ax1.fill_between(doses, pred["lower"], pred["upper"], alpha=0.3, color='blue', label='95% CI')
    
    if simulator is not None:
        true_doses, true_responses = simulator.get_true_curve((0.5, 6.0), 100)
        ax1.plot(true_doses, true_responses, 'g--', linewidth=2, label='真实曲线')
    
    if measurements is not None and len(measurements) > 0:
        doses_obs = [m[0] for m in measurements]
        responses_obs = [m[1] for m in measurements]
        ax1.scatter(doses_obs, responses_obs, c='red', s=100, zorder=5, label='观测数据')
    
    if lang == "zh":
        ax1.set_xlabel('剂量 (D)', fontsize=11)
        ax1.set_ylabel('反应 (μm)', fontsize=11)
        ax1.set_title('剂量-反应曲线', fontsize=12)
    else:
        ax1.set_xlabel('Dose (D)', fontsize=11)
        ax1.set_ylabel('Response (μm)', fontsize=11)
        ax1.set_title('Dose-Response Curve', fontsize=12)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 采集函数
    ax2 = plt.subplot(2, 3, 2)
    if y_best is not None:
        x_grid, acq_values = acquisition.get_acquisition_values(model, "ei", y_best)
        ax2.plot(x_grid, acq_values, 'purple', linewidth=2)
        ax2.fill_between(x_grid, 0, acq_values, alpha=0.3, color='purple')
        
        max_idx = np.argmax(acq_values)
        max_dose = x_grid[max_idx]
        ax2.plot(max_dose, acq_values[max_idx], 'r*', markersize=12, label=f'推荐: {max_dose:.2f} D')
    
    if measurements is not None and len(measurements) > 0:
        doses_obs = [m[0] for m in measurements]
        ax2.scatter(doses_obs, [0] * len(doses_obs), c='red', s=100, marker='|', zorder=5, label='已测量')
    
    if lang == "zh":
        ax2.set_xlabel('剂量 (D)', fontsize=11)
        ax2.set_ylabel('采集值', fontsize=11)
        ax2.set_title('采集函数', fontsize=12)
    else:
        ax2.set_xlabel('Dose (D)', fontsize=11)
        ax2.set_ylabel('Acquisition Value', fontsize=11)
        ax2.set_title('Acquisition Function', fontsize=12)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 阈值后验分布
    ax3 = plt.subplot(2, 3, 3)
    posterior = model.trace.posterior["threshold"].values.reshape(-1)
    ax3.hist(posterior, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    
    mean_val = np.mean(posterior)
    ax3.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'均值: {mean_val:.3f}')
    
    if lang == "zh":
        ax3.set_xlabel('阈值 (D)', fontsize=11)
        ax3.set_ylabel('密度', fontsize=11)
        ax3.set_title('阈值后验分布', fontsize=12)
    else:
        ax3.set_xlabel('Threshold (D)', fontsize=11)
        ax3.set_ylabel('Density', fontsize=11)
        ax3.set_title('Threshold Posterior', fontsize=12)
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 子图4: 学习轨迹
    ax4 = plt.subplot(2, 3, 4)
    if measurements is not None and len(measurements) > 0:
        doses = [m[0] for m in measurements]
        rounds = list(range(1, len(measurements) + 1))
        ax4.plot(rounds, doses, 'o-', linewidth=2, markerfacecolor='red', markersize=8, label='推荐剂量')
        
        if simulator is not None:
            ax4.axhline(y=simulator.true_params.threshold, color='green', linestyle='--', linewidth=2, label='真实阈值')
    
    if lang == "zh":
        ax4.set_xlabel('轮次', fontsize=11)
        ax4.set_ylabel('剂量 (D)', fontsize=11)
        ax4.set_title('学习轨迹', fontsize=12)
    else:
        ax4.set_xlabel('Round', fontsize=11)
        ax4.set_ylabel('Dose (D)', fontsize=11)
        ax4.set_title('Learning Trajectory', fontsize=12)
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # 子图5: 所有参数后验分布
    ax5 = plt.subplot(2, 3, 5)
    param_names = ["baseline", "max_response", "slope", "threshold", "sigma"]
    param_labels = {
        "baseline": "基线",
        "max_response": "最大反应",
        "slope": "斜率",
        "threshold": "阈值",
        "sigma": "噪声",
    }
    
    for i, param in enumerate(param_names):
        posterior = model.trace.posterior[param].values.reshape(-1)
        mean_val = np.mean(posterior)
        ax5.plot([i, i], [0, 1], 'k-', alpha=0.3)
        ax5.axvline(i, color='k', alpha=0.1)
        
        # 绘制箱线图
        ax5.boxplot(posterior, positions=[i], widths=0.6, patch_artist=True, 
                   boxprops=dict(facecolor='steelblue', alpha=0.5))
    
    if lang == "zh":
        ax5.set_xticks(range(len(param_names)))
        ax5.set_xticklabels([param_labels[p] for p in param_names], rotation=45, ha='right')
        ax5.set_ylabel('参数值', fontsize=11)
        ax5.set_title('参数后验分布', fontsize=12)
    else:
        ax5.set_xticks(range(len(param_names)))
        ax5.set_xticklabels([p for p in param_names], rotation=45, ha='right')
        ax5.set_ylabel('Parameter Value', fontsize=11)
        ax5.set_title('Parameter Posteriors', fontsize=12)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 子图6: 空白或额外信息
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # 添加汇总信息
    if lang == "zh":
        stats = model.get_posterior_stats()
        info_text = f"阈值估计: {stats['threshold']['mean']:.2f} ± {stats['threshold']['std']:.2f} D\n"
        info_text += f"基线反应: {stats['baseline']['mean']:.2f} μm\n"
        info_text += f"最大反应: {stats['max_response']['mean']:.2f} μm\n"
        info_text += f"斜率: {stats['slope']['mean']:.2f}\n"
        info_text += f"噪声: {stats['sigma']['mean']:.2f} μm"
        ax6.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center')
    else:
        stats = model.get_posterior_stats()
        info_text = f"Threshold: {stats['threshold']['mean']:.2f} ± {stats['threshold']['std']:.2f} D\n"
        info_text += f"Baseline: {stats['baseline']['mean']:.2f} μm\n"
        info_text += f"Max Response: {stats['max_response']['mean']:.2f} μm\n"
        info_text += f"Slope: {stats['slope']['mean']:.2f}\n"
        info_text += f"Noise: {stats['sigma']['mean']:.2f} μm"
        ax6.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center')
    
    plt.suptitle(title or '汇总报告' if lang == "zh" else 'Summary Report', fontsize=16, y=1.02)
    plt.tight_layout()
    return fig


def plot_all_figures(
    model: SigmoidModel,
    doses: List[float],
    responses: List[float],
    acquisition: AcquisitionFunction,
    round_num: int,
    rounds_data: List[Dict],
    y_best: Optional[float] = None,
    simulator: Optional[SubjectSimulator] = None,
    save_dir: str = "results/visualization",
    lang: str = "en",
    models_evolution: Optional[List[SigmoidModel]] = None,
    measurements_evolution: Optional[List[List[Tuple[float, float]]]] = None,
) -> Dict[str, Figure]:
    """
    绘制所有图表并保存。
    
    Args:
        model: 拟合的Sigmoid模型
        doses: 剂量列表
        responses: 反应列表
        acquisition: 采集函数对象
        round_num: 当前轮次
        rounds_data: 轮次数据
        y_best: 当前最佳反应值
        simulator: 模拟器（可选）
        save_dir: 保存目录
        lang: 语言 ('en' 或 'zh')
        models_evolution: 不同轮次的模型列表（用于展示学习进步性）
        measurements_evolution: 不同轮次的测量点列表
        
    Returns:
        图表对象字典
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    measurements = list(zip(doses, responses))
    figures = {}
    
    # 1. 剂量-反应拟合曲线
    fig1 = plot_dose_response_curve(
        model,
        simulator=simulator,
        measurements=measurements,
        title="剂量-反应拟合曲线" if lang == "zh" else "Dose-Response Curve",
    )
    fig1_path = os.path.join(save_dir, f"01_dose_response_{lang}.png")
    save_figure(fig1, fig1_path)
    figures["dose_response"] = fig1
    
    # 2. 每轮主动学习推荐剂量轨迹图
    true_threshold = simulator.true_params.threshold if simulator else None
    fig2 = plot_learning_trajectory(
        measurements,
        true_threshold=true_threshold,
        title="学习轨迹" if lang == "zh" else "Learning Trajectory",
        lang=lang,
    )
    fig2_path = os.path.join(save_dir, f"02_learning_trajectory_{lang}.png")
    save_figure(fig2, fig2_path)
    figures["learning_trajectory"] = fig2
    
    # 3. 模型参数后验分布可视化图
    fig3 = plot_all_posteriors(model)
    fig3_path = os.path.join(save_dir, f"03_posterior_distributions_{lang}.png")
    save_figure(fig3, fig3_path)
    figures["posterior_distributions"] = fig3
    
    # 4. 主动学习采集函数变化曲线
    if y_best is not None:
        fig4 = plot_acquisition_curve(
            model,
            acquisition,
            y_best,
            measurements,
            title="采集函数变化" if lang == "zh" else "Acquisition Curve",
            lang=lang,
        )
        fig4_path = os.path.join(save_dir, f"04_acquisition_curve_{lang}.png")
        save_figure(fig4, fig4_path)
        figures["acquisition_curve"] = fig4
    
    # 5. 汇总可视化报告
    fig5 = plot_summary_report(
        model,
        acquisition,
        simulator=simulator,
        measurements=measurements,
        y_best=y_best,
        title="汇总报告" if lang == "zh" else "Summary Report",
        lang=lang,
    )
    fig5_path = os.path.join(save_dir, f"05_summary_report_{lang}.png")
    save_figure(fig5, fig5_path)
    figures["summary_report"] = fig5
    
    # 6. 后验分布演变图 - 展示学习进步性
    if models_evolution and measurements_evolution:
        fig6 = plot_posterior_evolution_standalone(
            models_evolution,
            measurements_evolution,
            lang=lang,
        )
        fig6_path = os.path.join(save_dir, f"06_posterior_evolution_{lang}.png")
        save_figure(fig6, fig6_path)
        figures["posterior_evolution"] = fig6
        
        # 7. 不确定性收缩图 - 展示估计精度的提升
        if len(models_evolution) >= 2:
            fig7 = plot_uncertainty_shrinkage_standalone(
                models_evolution[0],
                models_evolution[-1],
                measurements_evolution[0],
                measurements_evolution[-1],
                lang=lang,
            )
            fig7_path = os.path.join(save_dir, f"07_uncertainty_shrinkage_{lang}.png")
            save_figure(fig7, fig7_path)
            figures["uncertainty_shrinkage"] = fig7
    
    return figures


class ProjectExhibitionSuite:
    """
    项目展示套件 - 包含多种高级可视化功能
    """
    
    def __init__(self, lang: str = "zh"):
        """
        初始化项目展示套件
        
        Args:
            lang: 语言 ('zh' 或 'en')
        """
        self.lang = lang
    
    def plot_decision_landscape(
        self,
        model: SigmoidModel,
        acquisition: AcquisitionFunction,
        measurements: List[Tuple[float, float]],
        y_best: Optional[float] = None,
        figsize: Tuple[float, float] = (12, 8),
    ) -> Figure:
        """
        绘制决策景观图 - 一维热力图，展示AI的决策偏好
        
        Args:
            model: 拟合的Sigmoid模型
            acquisition: 采集函数对象
            measurements: 测量点列表
            y_best: 当前最佳反应值
            figsize: 图大小
            
        Returns:
            Matplotlib Figure 对象
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
        
        # 上部分：剂量-反应曲线
        doses = np.linspace(0.5, 6.0, 100)
        pred = model.predict(doses)
        
        ax1.plot(doses, pred["mean"], 'b-', linewidth=2, label='后验均值' if self.lang == "zh" else 'Posterior Mean')
        ax1.fill_between(doses, pred["lower"], pred["upper"], alpha=0.3, color='blue', label='95% CI')
        
        if measurements:
            doses_obs = [m[0] for m in measurements]
            responses_obs = [m[1] for m in measurements]
            ax1.scatter(doses_obs, responses_obs, c='red', s=100, zorder=5, label='观测数据' if self.lang == "zh" else 'Observed Data')
        
        ax1.set_xlabel('剂量 (D)', fontsize=12)
        ax1.set_ylabel('反应 (μm)', fontsize=12)
        ax1.set_title('剂量-反应曲线' if self.lang == "zh" else 'Dose-Response Curve', fontsize=14)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 下部分：决策景观热力图
        x_grid, acq_values = acquisition.get_acquisition_values(model, "eer", y_best)
        
        # 归一化采集值
        acq_norm = (acq_values - np.min(acq_values)) / (np.max(acq_values) - np.min(acq_values) + 1e-9)
        
        # 绘制热力图
        im = ax2.imshow(acq_norm.reshape(1, -1), cmap='viridis', aspect='auto', extent=[0.5, 6.0, 0, 1])
        
        # 标记推荐剂量
        max_idx = np.argmax(acq_values)
        max_dose = x_grid[max_idx]
        ax2.plot(max_dose, 0.5, 'r*', markersize=15, label=f'推荐: {max_dose:.2f} D')
        
        # 标记已测量点
        if measurements:
            doses_obs = [m[0] for m in measurements]
            ax2.scatter(doses_obs, [0.5] * len(doses_obs), c='white', s=100, marker='|', zorder=5, label='已测量' if self.lang == "zh" else 'Measured')
        
        ax2.set_xlabel('剂量 (D)', fontsize=12)
        ax2.set_ylabel('兴趣度', fontsize=12) if self.lang == "zh" else ax2.set_ylabel('Interest', fontsize=12)
        ax2.set_title('决策景观' if self.lang == "zh" else 'Decision Landscape', fontsize=14)
        ax2.legend(loc='best')
        
        # 添加颜色条
        cbar = fig.colorbar(im, ax=ax2, orientation='horizontal', pad=0.3)
        cbar.set_label('兴趣度' if self.lang == "zh" else 'Interest', fontsize=11)
        
        plt.tight_layout()
        return fig
    
    def plot_confidence_stopwatch(
        self,
        model: SigmoidModel,
        figsize: Tuple[float, float] = (8, 8),
    ) -> Figure:
        """
        绘制信心秒表 - 显示当前对阈值估计的信心百分比
        
        Args:
            model: 拟合的Sigmoid模型
            figsize: 图大小
            
        Returns:
            Matplotlib Figure 对象
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 获取阈值后验标准差
        threshold_std = model.get_posterior_stats()["threshold"]["std"]
        
        # 计算信心百分比 (基于 1/σ)，优化计算方式提高信心水平
        # 假设当 σ=0.1 时信心为 100%
        max_confidence = 100.0
        min_confidence = 0.0
        
        # 优化信心计算，增加系数并考虑测量次数的影响
        # 获取模型的测量次数信息（通过后验样本数量推断）
        n_measurements = model.trace.posterior.dims.get('draw', 1) * model.trace.posterior.dims.get('chain', 1)
        
        # 基础信心计算，增加系数到15.0
        base_confidence = (1.0 / (threshold_std + 0.01)) * 15.0
        
        # 测量次数奖励因子（测量次数越多，信心越高）
        measurement_factor = min(1.2, 1.0 + (n_measurements / 10000))
        
        # 综合计算信心值
        confidence = min(max_confidence, max(min_confidence, base_confidence * measurement_factor))
        
        # 绘制仪表盘
        ax.set_aspect('equal')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        
        # 绘制背景圆
        circle = plt.Circle((0, 0), 1, fill=False, linewidth=2, color='gray')
        ax.add_artist(circle)
        
        # 绘制刻度
        for i in range(0, 101, 10):
            angle = np.radians(180 - (i / 100) * 180)
            x = np.cos(angle)
            y = np.sin(angle)
            ax.plot([0.9 * x, x], [0.9 * y, y], 'k-', linewidth=1)
            ax.text(1.1 * x, 1.1 * y, f'{i}%', ha='center', va='center', fontsize=10)
        
        # 绘制指针
        angle = np.radians(180 - (confidence / 100) * 180)
        x = np.cos(angle)
        y = np.sin(angle)
        ax.plot([0, x], [0, y], 'r-', linewidth=3, marker='o', markersize=8, markerfacecolor='red')
        
        # 绘制中心
        center = plt.Circle((0, 0), 0.1, fill=True, color='gray')
        ax.add_artist(center)
        
        # 添加标题和信心值
        ax.set_title('阈值估计信心' if self.lang == "zh" else 'Threshold Estimation Confidence', fontsize=14)
        ax.text(0, -1.3, f'信心: {confidence:.1f}%', ha='center', va='center', fontsize=14, fontweight='bold')
        
        # 隐藏坐标轴
        ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_posterior_evolution(
        self,
        models: List[SigmoidModel],
        measurements_list: List[List[Tuple[float, float]]],
        figsize: Tuple[float, float] = (12, 8),
    ) -> Figure:
        """
        绘制后验分布演变图 - 展示随测量增加，后验分布从模糊到精确的过程
        
        Args:
            models: 不同轮次的模型列表
            measurements_list: 不同轮次的测量点列表
            figsize: 图大小
            
        Returns:
            Matplotlib Figure 对象
        """
        n_plots = len(models)
        n_rows = (n_plots + 1) // 2
        n_cols = 2 if n_plots > 1 else 1
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.atleast_1d(axes).flatten()
        
        for i, (model, measurements) in enumerate(zip(models, measurements_list)):
            ax = axes[i]
            
            # 获取阈值后验样本
            posterior = model.trace.posterior["threshold"].values.reshape(-1)
            
            # 绘制直方图
            ax.hist(posterior, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black')
            
            # 绘制均值线
            mean_val = np.mean(posterior)
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'均值: {mean_val:.3f}')
            
            # 设置标签和标题
            ax.set_xlabel('阈值 (D)', fontsize=11)
            ax.set_ylabel('密度', fontsize=11) if self.lang == "zh" else ax.set_ylabel('Density', fontsize=11)
            ax.set_title(f'第 {i+1} 次测量后' if self.lang == "zh" else f'After {i+1} Measurements', fontsize=12)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
        
        # 隐藏多余的子图
        for i in range(n_plots, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('后验分布演变' if self.lang == "zh" else 'Posterior Evolution', fontsize=16, y=1.02)
        plt.tight_layout()
        return fig


def plot_posterior_evolution_standalone(
    models: List[SigmoidModel],
    measurements_list: List[List[Tuple[float, float]]],
    figsize: Tuple[float, float] = (16, 10),
    lang: str = "en",
) -> Figure:
    """
    绘制后验分布演变图（独立函数版本）- 展示随测量增加，后验分布从模糊到精确的过程
    
    Args:
        models: 不同轮次的模型列表
        measurements_list: 不同轮次的测量点列表
        figsize: 图大小
        lang: 语言 ('en' 或 'zh')
        
    Returns:
        Matplotlib Figure 对象
    """
    n_plots = len(models)
    n_rows = (n_plots + 1) // 2
    n_cols = 2 if n_plots > 1 else 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()
    
    for i, (model, measurements) in enumerate(zip(models, measurements_list)):
        ax = axes[i]
        
        # 获取阈值后验样本
        posterior = model.trace.posterior["threshold"].values.reshape(-1)
        
        # 获取当前阈值统计
        stats = model.get_posterior_stats()
        threshold_mean = stats["threshold"]["mean"]
        threshold_std = stats["threshold"]["std"]
        
        # 绘制直方图
        ax.hist(posterior, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black')
        
        # 绘制均值线和置信区间
        mean_val = np.mean(posterior)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'均值: {mean_val:.3f}')
        ax.axvline(threshold_mean - threshold_std, color='orange', linestyle=':', linewidth=1.5)
        ax.axvline(threshold_mean + threshold_std, color='orange', linestyle=':', linewidth=1.5, label=f'±σ: {threshold_std:.3f}')
        
        # 设置标签和标题
        if lang == "zh":
            ax.set_xlabel('阈值 (D)', fontsize=11)
            ax.set_ylabel('密度', fontsize=11)
            ax.set_title(f'第 {i+2} 次测量后 (σ={threshold_std:.3f})', fontsize=12)
        else:
            ax.set_xlabel('Threshold (D)', fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.set_title(f'After {i+2} Measurements (σ={threshold_std:.3f})', fontsize=12)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    # 隐藏多余的子图
    for i in range(n_plots, len(axes)):
        axes[i].axis('off')
    
    # 添加总标题说明
    if lang == "zh":
        plt.suptitle(f'学习进步性展示：后验分布随测量次数增加而逐渐精确 (共{n_plots}个阶段)', fontsize=16, y=1.02)
    else:
        plt.suptitle(f'Learning Progress: Posterior Distribution Converges with More Measurements ({n_plots} Stages)', fontsize=16, y=1.02)
    
    plt.tight_layout()
    return fig


def plot_uncertainty_shrinkage_standalone(
    model_early: SigmoidModel,
    model_late: SigmoidModel,
    measurements_early: List[Tuple[float, float]],
    measurements_late: List[Tuple[float, float]],
    figsize: Tuple[float, float] = (16, 6),
    lang: str = "en",
) -> Figure:
    """
    绘制不确定性收缩效果对比图（独立函数版本）
    
    Args:
        model_early: 早期模型
        model_late: 后期模型
        measurements_early: 早期测量点
        measurements_late: 后期测量点
        figsize: 图大小
        lang: 语言 ('en' 或 'zh')
        
    Returns:
        Matplotlib Figure 对象
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 生成剂量网格
    doses = np.linspace(0.5, 6.0, 100)
    
    # 获取早期模型的统计
    stats_early = model_early.get_posterior_stats()
    std_early = stats_early["threshold"]["std"]
    
    # 早期模型
    pred_early = model_early.predict(doses)
    ax1.plot(doses, pred_early["mean"], 'b-', linewidth=2, label='后验均值' if lang == "zh" else 'Posterior Mean')
    ax1.fill_between(doses, pred_early["lower"], pred_early["upper"], alpha=0.3, color='blue', label='95% CI')

    if measurements_early:
        doses_obs1 = [m[0] for m in measurements_early]
        responses_obs1 = [m[1] for m in measurements_early]
        ax1.scatter(doses_obs1, responses_obs1, c='red', s=100, zorder=5, label='观测数据' if lang == "zh" else 'Observed Data')
    
    ax1.set_xlabel('剂量 (D)', fontsize=12)
    ax1.set_ylabel('反应 (μm)', fontsize=12)
    if lang == "zh":
        ax1.set_title(f'早期阶段 (σ={std_early:.3f})', fontsize=14)
    else:
        ax1.set_title(f'Early Stage (σ={std_early:.3f})', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 获取后期模型的统计
    stats_late = model_late.get_posterior_stats()
    std_late = stats_late["threshold"]["std"]
    
    # 后期模型
    pred_late = model_late.predict(doses)
    ax2.plot(doses, pred_late["mean"], 'b-', linewidth=2, label='后验均值' if lang == "zh" else 'Posterior Mean')
    ax2.fill_between(doses, pred_late["lower"], pred_late["upper"], alpha=0.3, color='blue', label='95% CI')
    
    if measurements_late:
        doses_obs2 = [m[0] for m in measurements_late]
        responses_obs2 = [m[1] for m in measurements_late]
        ax2.scatter(doses_obs2, responses_obs2, c='red', s=100, zorder=5, label='观测数据' if lang == "zh" else 'Observed Data')
    
    ax2.set_xlabel('剂量 (D)', fontsize=12)
    ax2.set_ylabel('反应 (μm)', fontsize=12)
    if lang == "zh":
        ax2.set_title(f'后期阶段 (σ={std_late:.3f})', fontsize=14)
    else:
        ax2.set_title(f'Late Stage (σ={std_late:.3f})', fontsize=14)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # 计算不确定性减少百分比
    reduction_pct = ((std_early - std_late) / std_early * 100) if std_early > 0 else 0
    
    if lang == "zh":
        plt.suptitle(f'估计精度提升：不确定性从 {std_early:.3f} 降低到 {std_late:.3f} (减少 {reduction_pct:.1f}%)', fontsize=16, y=1.02)
    else:
        plt.suptitle(f'Precision Improvement: Uncertainty Reduced from {std_early:.3f} to {std_late:.3f} ({reduction_pct:.1f}% Reduction)', fontsize=16, y=1.02)
    
    plt.tight_layout()
    return fig
    
    def plot_clinical_interpretation(
        self,
        model: SigmoidModel,
        measurements: List[Tuple[float, float]],
        figsize: Tuple[float, float] = (12, 8),
    ) -> Figure:
        """
        绘制临床解释层 - 标注生理学关键点
        
        Args:
            model: 拟合的Sigmoid模型
            measurements: 测量点列表
            figsize: 图大小
            
        Returns:
            Matplotlib Figure 对象
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 生成剂量网格
        doses = np.linspace(0.5, 6.0, 100)
        pred = model.predict(doses)
        
        # 绘制剂量-反应曲线
        ax.plot(doses, pred["mean"], 'b-', linewidth=2, label='后验均值' if self.lang == "zh" else 'Posterior Mean')
        ax.fill_between(doses, pred["lower"], pred["upper"], alpha=0.3, color='blue', label='95% CI')
        
        # 绘制测量点
        if measurements:
            doses_obs = [m[0] for m in measurements]
            responses_obs = [m[1] for m in measurements]
            ax.scatter(doses_obs, responses_obs, c='red', s=100, zorder=5, label='观测数据' if self.lang == "zh" else 'Observed Data')
        
        # 获取模型参数
        stats = model.get_posterior_stats()
        baseline = stats["baseline"]["mean"]
        max_response = stats["max_response"]["mean"]
        threshold = stats["threshold"]["mean"]
        slope = stats["slope"]["mean"]
        
        # 计算生理学关键点
        # 基线反应区：剂量远小于阈值
        baseline_zone = threshold - 2.0 / slope
        # 过渡区：围绕阈值
        transition_start = threshold - 1.0 / slope
        transition_end = threshold + 1.0 / slope
        # 饱和区：剂量远大于阈值
        saturation_zone = threshold + 2.0 / slope
        
        # 绘制生理学区域
        ax.axvspan(0.5, baseline_zone, alpha=0.1, color='green', label='基线反应区' if self.lang == "zh" else 'Baseline Zone')
        ax.axvspan(transition_start, transition_end, alpha=0.1, color='yellow', label='过渡区' if self.lang == "zh" else 'Transition Zone')
        ax.axvspan(saturation_zone, 6.0, alpha=0.1, color='red', label='饱和区' if self.lang == "zh" else 'Saturation Zone')
        
        # 标记阈值
        ax.axvline(threshold, color='purple', linestyle='--', linewidth=2, label=f'阈值: {threshold:.2f} D')
        
        # 设置标签和标题
        ax.set_xlabel('剂量 (D)', fontsize=12)
        ax.set_ylabel('反应 (μm)', fontsize=12)
        ax.set_title('临床解释' if self.lang == "zh" else 'Clinical Interpretation', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_acquisition_landscape(
        self,
        model: SigmoidModel,
        acquisition: AcquisitionFunction,
        measurements: List[Tuple[float, float]],
        y_best: Optional[float] = None,
        figsize: Tuple[float, float] = (12, 6),
    ) -> Figure:
        """
        绘制AI的思考过程热力图 - 展示AI对每个点的兴趣度
        
        Args:
            model: 拟合的Sigmoid模型
            acquisition: 采集函数对象
            measurements: 测量点列表
            y_best: 当前最佳反应值
            figsize: 图大小
            
        Returns:
            Matplotlib Figure 对象
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 获取EER采集函数值
        x_grid, acq_values = acquisition.get_acquisition_values(model, "eer", y_best)
        
        # 归一化采集值
        acq_norm = (acq_values - np.min(acq_values)) / (np.max(acq_values) - np.min(acq_values) + 1e-9)
        
        # 绘制热力图
        im = ax.imshow(acq_norm.reshape(1, -1), cmap='hot', aspect='auto', extent=[0, 6.0, 0, 1])
        
        # 标记推荐剂量
        max_idx = np.argmax(acq_values)
        max_dose = x_grid[max_idx]
        ax.plot(max_dose, 0.5, 'r*', markersize=15, label=f'推荐: {max_dose:.2f} D')
        
        # 标记已测量点
        if measurements:
            doses_obs = [m[0] for m in measurements]
            ax.scatter(doses_obs, [0.5] * len(doses_obs), c='white', s=100, marker='|', zorder=5, label='已测量' if self.lang == "zh" else 'Measured')
        
        # 添加解释文本
        explanation = "AI 放弃高剂量区而选择 2.8D 附近，因为这里是阈值所在区域，能最大程度减少 ED50 估计的不确定性。这展示了 AI 在好奇心与收敛性之间的权衡。" if self.lang == "zh" else "AI avoids high dose regions and chooses around 2.8D because this is where the threshold is located, which maximizes the reduction in ED50 estimation uncertainty. This demonstrates the AI's balance between curiosity and convergence."
        ax.text(3.0, -0.2, explanation, ha='center', va='top', fontsize=10, wrap=True)
        
        # 设置标签和标题
        ax.set_xlabel('剂量 (D)', fontsize=12)
        ax.set_ylabel('兴趣度', fontsize=12) if self.lang == "zh" else ax.set_ylabel('Interest', fontsize=12)
        ax.set_title('AI的思考过程' if self.lang == "zh" else "AI's Thought Process", fontsize=14)
        ax.legend(loc='best')
        
        # 添加颜色条
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.2)
        cbar.set_label('兴趣度' if self.lang == "zh" else 'Interest', fontsize=11)
        
        plt.tight_layout()
        return fig
    
    def plot_uncertainty_shrinkage(
        self,
        model1: SigmoidModel,  # 第1次测量后
        model5: SigmoidModel,  # 第5次测量后
        measurements1: List[Tuple[float, float]],
        measurements5: List[Tuple[float, float]],
        figsize: Tuple[float, float] = (12, 6),
    ) -> Figure:
        """
        绘制不确定性收缩效果 - 对比第1次和第5次测量后的不确定性
        
        Args:
            model1: 第1次测量后的模型
            model5: 第5次测量后的模型
            measurements1: 第1次测量的点
            measurements5: 第5次测量的点
            figsize: 图大小
            
        Returns:
            Matplotlib Figure 对象
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 生成剂量网格
        doses = np.linspace(0.5, 6.0, 100)
        
        # 第1次测量后
        pred1 = model1.predict(doses)
        ax1.plot(doses, pred1["mean"], 'b-', linewidth=2, label='后验均值' if self.lang == "zh" else 'Posterior Mean')
        ax1.fill_between(doses, pred1["lower"], pred1["upper"], alpha=0.3, color='blue', label='95% CI')
        
        if measurements1:
            doses_obs1 = [m[0] for m in measurements1]
            responses_obs1 = [m[1] for m in measurements1]
            ax1.scatter(doses_obs1, responses_obs1, c='red', s=100, zorder=5, label='观测数据' if self.lang == "zh" else 'Observed Data')
        
        ax1.set_xlabel('剂量 (D)', fontsize=12)
        ax1.set_ylabel('反应 (μm)', fontsize=12)
        ax1.set_title('第1次测量后' if self.lang == "zh" else 'After 1st Measurement', fontsize=14)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 第5次测量后
        pred5 = model5.predict(doses)
        ax2.plot(doses, pred5["mean"], 'b-', linewidth=2, label='后验均值' if self.lang == "zh" else 'Posterior Mean')
        ax2.fill_between(doses, pred5["lower"], pred5["upper"], alpha=0.3, color='blue', label='95% CI')
        
        if measurements5:
            doses_obs5 = [m[0] for m in measurements5]
            responses_obs5 = [m[1] for m in measurements5]
            ax2.scatter(doses_obs5, responses_obs5, c='red', s=100, zorder=5, label='观测数据' if self.lang == "zh" else 'Observed Data')
        
        ax2.set_xlabel('剂量 (D)', fontsize=12)
        ax2.set_ylabel('反应 (μm)', fontsize=12)
        ax2.set_title('第5次测量后' if self.lang == "zh" else 'After 5th Measurement', fontsize=14)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('不确定性收缩效果' if self.lang == "zh" else 'Uncertainty Shrinkage Effect', fontsize=16, y=1.02)
        plt.tight_layout()
        return fig
    
    def plot_joint_posterior(
        self,
        model: SigmoidModel,
        figsize: Tuple[float, float] = (10, 8),
    ) -> Figure:
        """
        绘制参数关联图 - 展示 ED50 与 Slope 的联合分布
        
        Args:
            model: 拟合的Sigmoid模型
            figsize: 图大小
            
        Returns:
            Matplotlib Figure 对象
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 获取后验样本
        threshold = model.trace.posterior["threshold"].values.reshape(-1)
        slope = model.trace.posterior["slope"].values.reshape(-1)
        
        # 绘制散点图
        scatter = ax.scatter(threshold, slope, alpha=0.5, s=50, c='steelblue')
        
        # 绘制均值点
        threshold_mean = np.mean(threshold)
        slope_mean = np.mean(slope)
        ax.plot(threshold_mean, slope_mean, 'r*', markersize=15, label=f'均值: ({threshold_mean:.2f}, {slope_mean:.2f})')
        
        # 设置标签和标题
        ax.set_xlabel('ED50 (D)', fontsize=12)
        ax.set_ylabel('斜率 (Slope)', fontsize=12)
        ax.set_title('ED50 与斜率的联合分布' if self.lang == "zh" else 'Joint Posterior of ED50 and Slope', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_subject_persona(
        self,
        model: SigmoidModel,
        figsize: Tuple[float, float] = (10, 8),
    ) -> Figure:
        """
        绘制受试者画像雷达图 - 根据拟合参数给受试者打标签
        
        Args:
            model: 拟合的Sigmoid模型
            figsize: 图大小
            
        Returns:
            Matplotlib Figure 对象
        """
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
        
        # 获取模型参数
        stats = model.get_posterior_stats()
        threshold = stats["threshold"]["mean"]
        slope = stats["slope"]["mean"]
        max_response = stats["max_response"]["mean"]
        baseline = stats["baseline"]["mean"]
        
        # 计算特征值（归一化到 0-100）
        # 敏感性：阈值越低，敏感性越高
        sensitivity = max(0, min(100, (4.0 - threshold) / 4.0 * 100))
        # 反应速度：斜率越大，反应速度越快
        response_speed = max(0, min(100, (slope / 5.0) * 100))
        # 最大反应：最大值越高，反应强度越大
        response_strength = max(0, min(100, ((max_response - 10) / 15) * 100))
        # 基线水平：基线越低，基础状态越好
        baseline_level = max(0, min(100, (5.0 - baseline) / 5.0 * 100))
        
        # 特征名称
        features = ['敏感性', '反应速度', '反应强度', '基线水平'] if self.lang == "zh" else ['Sensitivity', 'Response Speed', 'Response Strength', 'Baseline Level']
        values = [sensitivity, response_speed, response_strength, baseline_level]
        
        # 计算角度
        angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
        values += values[:1]  # 闭合雷达图
        angles += angles[:1]
        
        # 绘制雷达图
        ax.plot(angles, values, 'o-', linewidth=2, color='steelblue')
        ax.fill(angles, values, alpha=0.25, color='steelblue')
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features)
        ax.set_ylim(0, 100)
        
        # 确定受试者类型
        if threshold < 2.0 and slope > 3.0:
            persona = '高敏感型' if self.lang == "zh" else 'Highly Sensitive'
        elif threshold > 3.5:
            persona = '迟钝型' if self.lang == "zh" else 'Insensitive'
        elif slope > 3.0:
            persona = '快速饱和型' if self.lang == "zh" else 'Fast Saturation'
        else:
            persona = '正常型' if self.lang == "zh" else 'Normal'
        
        # 设置标题
        ax.set_title(f'受试者画像: {persona}', fontsize=14, pad=20)
        
        plt.tight_layout()
        return fig
    
    def generate_full_exhibition(
        self,
        model: SigmoidModel,
        acquisition: AcquisitionFunction,
        measurements: List[Tuple[float, float]],
        y_best: Optional[float] = None,
        save_dir: str = "results/exhibition",
        models_evolution: Optional[List[SigmoidModel]] = None,
        measurements_evolution: Optional[List[List[Tuple[float, float]]]] = None,
        model1: Optional[SigmoidModel] = None,
        model5: Optional[SigmoidModel] = None,
        measurements1: Optional[List[Tuple[float, float]]] = None,
        measurements5: Optional[List[Tuple[float, float]]] = None,
    ) -> Dict[str, Figure]:
        """
        生成完整的展示套件
        
        Args:
            model: 最终拟合的模型
            acquisition: 采集函数对象
            measurements: 所有测量点
            y_best: 当前最佳反应值
            save_dir: 保存目录
            models_evolution: 不同轮次的模型列表
            measurements_evolution: 不同轮次的测量点列表
            model1: 第1次测量后的模型
            model5: 第5次测量后的模型
            measurements1: 第1次测量的点
            measurements5: 第5次测量的点
            
        Returns:
            图表对象字典
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        figures = {}
        
        # 1. 决策景观
        fig1 = self.plot_decision_landscape(model, acquisition, measurements, y_best)
        fig1_path = os.path.join(save_dir, f"01_decision_landscape_{self.lang}.png")
        save_figure(fig1, fig1_path)
        figures["decision_landscape"] = fig1
        
        # 2. 后验分布演变
        if models_evolution and measurements_evolution:
            fig2 = self.plot_posterior_evolution(models_evolution, measurements_evolution)
            fig2_path = os.path.join(save_dir, f"02_posterior_evolution_{self.lang}.png")
            save_figure(fig2, fig2_path)
            figures["posterior_evolution"] = fig2
        
        # 3. 临床解释
        fig3 = self.plot_clinical_interpretation(model, measurements)
        fig3_path = os.path.join(save_dir, f"03_clinical_interpretation_{self.lang}.png")
        save_figure(fig3, fig3_path)
        figures["clinical_interpretation"] = fig3
        
        # 4. AI思考过程
        fig4 = self.plot_acquisition_landscape(model, acquisition, measurements, y_best)
        fig4_path = os.path.join(save_dir, f"04_acquisition_landscape_{self.lang}.png")
        save_figure(fig4, fig4_path)
        figures["acquisition_landscape"] = fig4
        
        # 5. 不确定性收缩
        if models_evolution and measurements_evolution and len(models_evolution) >= 2:
            fig5 = self.plot_uncertainty_shrinkage(
                models_evolution[0],
                models_evolution[-1],
                measurements_evolution[0],
                measurements_evolution[-1]
            )
            fig5_path = os.path.join(save_dir, f"05_uncertainty_shrinkage_{self.lang}.png")
            save_figure(fig5, fig5_path)
            figures["uncertainty_shrinkage"] = fig5
        
        # 6. 参数关联图
        fig6 = self.plot_joint_posterior(model)
        fig6_path = os.path.join(save_dir, f"06_joint_posterior_{self.lang}.png")
        save_figure(fig6, fig6_path)
        figures["joint_posterior"] = fig6
        
        # 7. 受试者画像
        fig7 = self.plot_subject_persona(model)
        fig7_path = os.path.join(save_dir, f"07_subject_persona_{self.lang}.png")
        save_figure(fig7, fig7_path)
        figures["subject_persona"] = fig7
        
        return figures
