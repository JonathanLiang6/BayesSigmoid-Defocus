"""
可视化模块 - 绘制剂量-反应曲线和采集函数
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
    title: str = "剂量-反应曲线",
) -> Figure:
    """
    绘制后验剂量-反应曲线。
    
    Args:
        model: 已拟合的 Sigmoid 模型
        simulator: 模拟器（用于绘制真实曲线，可选）
        measurements: 测量点列表 [(dose, response), ...]
        dose_range: 剂量范围
        n_points: 曲线点数
        figsize: 图像大小
        title: 图表标题
        
    Returns:
        Matplotlib Figure 对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 生成剂量网格
    doses = np.linspace(dose_range[0], dose_range[1], n_points)
    
    # 预测
    pred = model.predict(doses)
    mean = pred["mean"]
    lower = pred["lower"]
    upper = pred["upper"]
    
    # 绘制后验均值
    ax.plot(doses, mean, 'b-', linewidth=2, label='后验均值')
    
    # 绘制置信区间
    ax.fill_between(doses, lower, upper, alpha=0.3, color='blue', label='95% 置信区间')
    
    # 绘制真实曲线（如果提供）
    if simulator is not None:
        true_doses, true_responses = simulator.get_true_curve(dose_range, n_points)
        ax.plot(true_doses, true_responses, 'g--', linewidth=2, label='真实曲线')
    
    # 绘制测量点
    if measurements is not None and len(measurements) > 0:
        doses_obs = [m[0] for m in measurements]
        responses_obs = [m[1] for m in measurements]
        ax.scatter(doses_obs, responses_obs, c='red', s=100, zorder=5, label='观测数据')
    
    # 设置标签和标题
    ax.set_xlabel('剂量 (D)', fontsize=12)
    ax.set_ylabel('反应 (μm)', fontsize=12)
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
    title: str = "采集函数",
) -> Figure:
    """
    绘制采集函数。
    
    Args:
        model: 已拟合的 Sigmoid 模型
        acquisition: 采集函数对象
        strategy: 策略 ("variance" 或 "ei")
        y_best: 当前最佳反应值（EI需要）
        measurements: 测量点列表
        figsize: 图像大小
        title: 图表标题
        
    Returns:
        Matplotlib Figure 对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 获取采集函数值
    x_grid, acq_values = acquisition.get_acquisition_values(model, strategy, y_best)
    
    # 绘制采集函数
    ax.plot(x_grid, acq_values, 'purple', linewidth=2)
    ax.fill_between(x_grid, 0, acq_values, alpha=0.3, color='purple')
    
    # 标记最大值
    max_idx = np.argmax(acq_values)
    max_dose = x_grid[max_idx]
    max_value = acq_values[max_idx]
    ax.plot(max_dose, max_value, 'r*', markersize=15, label=f'推荐剂量: {max_dose:.2f} D')
    
    # 标记已测量点
    if measurements is not None and len(measurements) > 0:
        doses_obs = [m[0] for m in measurements]
        ax.scatter(doses_obs, [0] * len(doses_obs), c='red', s=100, marker='|', 
                   zorder=5, label='已测量剂量')
    
    # 设置标签和标题
    strategy_name = "后验方差" if strategy == "variance" else "期望改进"
    ax.set_xlabel('剂量 (D)', fontsize=12)
    ax.set_ylabel('采集函数值', fontsize=12)
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
    绘制学习过程的组合图（剂量-反应曲线 + 采集函数）。
    
    Args:
        model: 已拟合的 Sigmoid 模型
        acquisition: 采集函数对象
        simulator: 模拟器（可选）
        measurements: 测量点列表
        strategy: 采集策略
        y_best: 当前最佳反应值
        dose_range: 剂量范围
        figsize: 图像大小
        
    Returns:
        Matplotlib Figure 对象
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 左图：剂量-反应曲线
    ax1 = axes[0]
    doses = np.linspace(dose_range[0], dose_range[1], 100)
    pred = model.predict(doses)
    
    ax1.plot(doses, pred["mean"], 'b-', linewidth=2, label='后验均值')
    ax1.fill_between(doses, pred["lower"], pred["upper"], alpha=0.3, color='blue', label='95% CI')
    
    if simulator is not None:
        true_doses, true_responses = simulator.get_true_curve(dose_range, 100)
        ax1.plot(true_doses, true_responses, 'g--', linewidth=2, label='真实曲线')
    
    if measurements is not None and len(measurements) > 0:
        doses_obs = [m[0] for m in measurements]
        responses_obs = [m[1] for m in measurements]
        ax1.scatter(doses_obs, responses_obs, c='red', s=100, zorder=5, label='观测数据')
    
    ax1.set_xlabel('剂量 (D)', fontsize=12)
    ax1.set_ylabel('反应 (μm)', fontsize=12)
    ax1.set_title('剂量-反应曲线', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 右图：采集函数
    ax2 = axes[1]
    x_grid, acq_values = acquisition.get_acquisition_values(model, strategy, y_best)
    
    ax2.plot(x_grid, acq_values, 'purple', linewidth=2)
    ax2.fill_between(x_grid, 0, acq_values, alpha=0.3, color='purple')
    
    max_idx = np.argmax(acq_values)
    max_dose = x_grid[max_idx]
    max_value = acq_values[max_idx]
    ax2.plot(max_dose, max_value, 'r*', markersize=15, label=f'推荐: {max_dose:.2f} D')
    
    if measurements is not None and len(measurements) > 0:
        doses_obs = [m[0] for m in measurements]
        ax2.scatter(doses_obs, [0] * len(doses_obs), c='red', s=100, marker='|', 
                   zorder=5, label='已测量')
    
    strategy_name = "后验方差" if strategy == "variance" else "期望改进"
    ax2.set_xlabel('剂量 (D)', fontsize=12)
    ax2.set_ylabel('采集函数值', fontsize=12)
    ax2.set_title(f'采集函数 ({strategy_name})', fontsize=14)
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
    绘制参数的后验分布。
    
    Args:
        model: 已拟合的 Sigmoid 模型
        param_name: 参数名称
        figsize: 图像大小
        title: 图表标题
        
    Returns:
        Matplotlib Figure 对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 获取后验样本
    posterior = model.trace.posterior[param_name].values.reshape(-1)
    
    # 绘制直方图
    ax.hist(posterior, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    
    # 绘制均值线
    mean_val = np.mean(posterior)
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'均值: {mean_val:.3f}')
    
    # 绘制中位数线
    median_val = np.median(posterior)
    ax.axvline(median_val, color='green', linestyle=':', linewidth=2, label=f'中位数: {median_val:.3f}')
    
    # 设置标签
    param_labels = {
        "baseline": "基线 (baseline)",
        "max_response": "最大反应 (max_response)",
        "slope": "斜率 (slope)",
        "threshold": "阈值 (threshold)",
        "sigma": "噪声标准差 (sigma)",
    }
    
    ax.set_xlabel(param_labels.get(param_name, param_name), fontsize=12)
    ax.set_ylabel('密度', fontsize=12)
    ax.set_title(title or f"{param_labels.get(param_name, param_name)} 的后验分布", fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_all_posteriors(
    model: SigmoidModel,
    figsize: Tuple[float, float] = (15, 10),
) -> Figure:
    """
    绘制所有参数的后验分布。
    
    Args:
        model: 已拟合的 Sigmoid 模型
        figsize: 图像大小
        
    Returns:
        Matplotlib Figure 对象
    """
    param_names = ["baseline", "max_response", "slope", "threshold", "sigma"]
    param_labels = {
        "baseline": "基线 (baseline)",
        "max_response": "最大反应 (max_response)",
        "slope": "斜率 (slope)",
        "threshold": "阈值 (threshold)",
        "sigma": "噪声标准差 (sigma)",
    }
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    for i, param in enumerate(param_names):
        ax = axes[i]
        posterior = model.trace.posterior[param].values.reshape(-1)
        
        ax.hist(posterior, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
        
        mean_val = np.mean(posterior)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'均值: {mean_val:.3f}')
        
        ax.set_xlabel(param_labels[param], fontsize=11)
        ax.set_ylabel('密度', fontsize=11)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    # 隐藏多余的子图
    axes[-1].axis('off')
    
    plt.suptitle('参数后验分布', fontsize=16, y=1.02)
    plt.tight_layout()
    return fig


def plot_simulation_results(
    results: List[dict],
    figsize: Tuple[float, float] = (14, 10),
) -> Figure:
    """
    绘制模拟研究结果。
    
    Args:
        results: 模拟结果列表
        figsize: 图像大小
        
    Returns:
        Matplotlib Figure 对象
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 提取数据
    n_measurements = [r["n_measurements"] for r in results]
    errors = [r["threshold_error"] for r in results]
    true_thresholds = [r["true_threshold"] for r in results]
    estimated_thresholds = [r["estimated_threshold"] for r in results]
    
    # 图1：测量次数分布
    ax1 = axes[0, 0]
    ax1.hist(n_measurements, bins=range(2, max(n_measurements)+2), alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_xlabel('测量次数', fontsize=12)
    ax1.set_ylabel('频数', fontsize=12)
    ax1.set_title('测量次数分布', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 图2：阈值估计误差分布
    ax2 = axes[0, 1]
    ax2.hist(errors, bins=30, alpha=0.7, color='coral', edgecolor='black')
    ax2.set_xlabel('阈值估计误差 (D)', fontsize=12)
    ax2.set_ylabel('频数', fontsize=12)
    ax2.set_title('阈值估计误差分布', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 图3：真实 vs 估计阈值
    ax3 = axes[1, 0]
    ax3.scatter(true_thresholds, estimated_thresholds, alpha=0.6, c='green')
    
    # 添加对角线
    min_val = min(min(true_thresholds), min(estimated_thresholds))
    max_val = max(max(true_thresholds), max(estimated_thresholds))
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想线')
    
    ax3.set_xlabel('真实阈值 (D)', fontsize=12)
    ax3.set_ylabel('估计阈值 (D)', fontsize=12)
    ax3.set_title('真实 vs 估计阈值', fontsize=14)
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    # 图4：测量次数 vs 误差
    ax4 = axes[1, 1]
    ax4.scatter(n_measurements, errors, alpha=0.6, c='purple')
    ax4.set_xlabel('测量次数', fontsize=12)
    ax4.set_ylabel('阈值估计误差 (D)', fontsize=12)
    ax4.set_title('测量次数 vs 估计误差', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('模拟研究结果', fontsize=16, y=1.02)
    plt.tight_layout()
    return fig


def save_figure(fig: Figure, filepath: str, dpi: int = 150) -> None:
    """
    保存图像到文件。
    
    Args:
        fig: Matplotlib Figure 对象
        filepath: 保存路径
        dpi: 分辨率
    """
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"图像已保存到: {filepath}")
