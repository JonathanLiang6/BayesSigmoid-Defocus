"""
学习过程可视化模块 - 学习轨迹、采集函数等
"""

from typing import Optional, Tuple, List, Dict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .model import SigmoidModel
from .acquisition import AcquisitionFunction
from .simulate import SubjectSimulator
from .basic_plots import save_figure

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_acquisition_function(
    model: SigmoidModel,
    acquisition: AcquisitionFunction,
    strategy: str = "variance",
    y_best: Optional[float] = None,
    measurements: Optional[List[Tuple[float, float]]] = None,
    figsize: Tuple[float, float] = (10, 4),
    title: str = "Acquisition Function",
) -> Figure:
    fig, ax = plt.subplots(figsize=figsize)
    x_grid, acq_values = acquisition.get_acquisition_values(model, strategy, y_best)
    ax.plot(x_grid, acq_values, 'purple', linewidth=2)
    ax.fill_between(x_grid, 0, acq_values, alpha=0.3, color='purple')
    
    max_idx = np.argmax(acq_values)
    max_dose = x_grid[max_idx]
    max_value = acq_values[max_idx]
    ax.plot(max_dose, max_value, 'r*', markersize=15, label=f'Recommended: {max_dose:.2f} D')
    
    if measurements is not None and len(measurements) > 0:
        doses_obs = [m[0] for m in measurements]
        ax.scatter(doses_obs, [0] * len(doses_obs), c='red', s=100, marker='|', zorder=5, label='Measured')
    
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
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    doses = np.linspace(dose_range[0], dose_range[1], 100)
    pred = model.predict(doses)
    
    doses_obs = []
    responses_obs = []
    if measurements is not None and len(measurements) > 0:
        doses_obs = [m[0] for m in measurements]
        responses_obs = [m[1] for m in measurements]
    
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
    
    ax2 = axes[1]
    x_grid, acq_values = acquisition.get_acquisition_values(model, strategy, y_best)
    ax2.plot(x_grid, acq_values, 'purple', linewidth=2)
    ax2.fill_between(x_grid, 0, acq_values, alpha=0.3, color='purple')
    
    max_idx = np.argmax(acq_values)
    max_dose = x_grid[max_idx]
    max_value = acq_values[max_idx]
    ax2.plot(max_dose, max_value, 'r*', markersize=15, label=f'Recommended: {max_dose:.2f} D')
    
    if len(doses_obs) > 0:
        ax2.scatter(doses_obs, [0] * len(doses_obs), c='red', s=100, marker='|', zorder=5, label='Measured')
    
    strategy_name = "Posterior Variance" if strategy == "variance" else "Expected Improvement"
    ax2.set_xlabel('Dose (D)', fontsize=12)
    ax2.set_ylabel('Acquisition Value', fontsize=12)
    ax2.set_title(f'Acquisition Function ({strategy_name})', fontsize=14)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_learning_trajectory(
    measurements: List[Tuple[float, float]],
    true_threshold: Optional[float] = None,
    figsize: Tuple[float, float] = (12, 6),
    title: str = "Learning Trajectory",
    lang: str = "en",
) -> Figure:
    fig, ax = plt.subplots(figsize=figsize)
    doses = [m[0] for m in measurements]
    responses = [m[1] for m in measurements]
    rounds = list(range(1, len(measurements) + 1))
    ax.plot(rounds, doses, 'o-', linewidth=2, markerfacecolor='red', markersize=8, label='推荐剂量' if lang == "zh" else 'Recommended Dose')
    
    if true_threshold is not None:
        ax.axhline(y=true_threshold, color='green', linestyle='--', linewidth=2, label='真实阈值' if lang == "zh" else 'True Threshold')
        ax.fill_between(rounds, true_threshold - 0.3, true_threshold + 0.3, alpha=0.2, color='green', label='±0.3D误差带' if lang == "zh" else '±0.3D Error Band')
        
        errors = [abs(d - true_threshold) for d in doses]
        ax2 = ax.twinx()
        ax2.plot(rounds, errors, 's--', color='orange', linewidth=1.5, markersize=6, alpha=0.7, label='估计误差' if lang == "zh" else 'Estimation Error')
        ax2.set_ylabel('估计误差 (D)', fontsize=11, color='orange') if lang == "zh" else ax2.set_ylabel('Estimation Error (D)', fontsize=11, color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        
        if len(errors) >= 2:
            reduction = (errors[0] - errors[-1]) / errors[0] * 100 if errors[0] > 0 else 0
            if lang == "zh":
                ax2.annotate(f'误差减少 {reduction:.1f}%', xy=(rounds[-1], errors[-1]), xytext=(rounds[-1]-0.5, errors[-1]+0.2), fontsize=10, color='orange')
            else:
                ax2.annotate(f'Error Reduced {reduction:.1f}%', xy=(rounds[-1], errors[-1]), xytext=(rounds[-1]-0.5, errors[-1]+0.2), fontsize=10, color='orange')
    
    if lang == "zh":
        ax.set_xlabel('轮次', fontsize=12)
        ax.set_ylabel('剂量 (D)', fontsize=12)
        ax.set_title('学习轨迹：推荐剂量逐渐收敛到真实阈值' if title is None else title, fontsize=14)
    else:
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel('Dose (D)', fontsize=12)
        ax.set_title('Learning Trajectory: Recommended Doses Converge to True Threshold' if title is None else title, fontsize=14)
    
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
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(errors, bins=30, alpha=0.7, color='coral', edgecolor='black')
    
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    ax.axvline(mean_error, color='red', linestyle='--', linewidth=2, label=f'均值: {mean_error:.3f}')
    
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
    fig, ax = plt.subplots(figsize=figsize)
    counts = np.bincount(n_measurements)
    bins = np.arange(len(counts))
    ax.bar(bins, counts, alpha=0.7, color='steelblue', edgecolor='black')
    
    if lang == "zh":
        ax.set_xlabel('测量次数', fontsize=12)
        ax.set_ylabel('受试者数量', fontsize=12)
        ax.set_title(title or '测量次数统计', fontsize=14)
    else:
        ax.set_xlabel('Number of Measurements', fontsize=12)
        ax.set_ylabel('Number of Subjects', fontsize=12)
        ax.set_title(title or 'Measurement Count', fontsize=14)
    
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
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(true_values, estimated_values, alpha=0.6, c='green')
    
    min_val = min(min(true_values), min(estimated_values))
    max_val = max(max(true_values), max(estimated_values))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美拟合')
    
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
    fig, ax = plt.subplots(figsize=figsize)
    x_grid, var_values = acquisition.get_acquisition_values(model, "variance")
    x_grid, ei_values = acquisition.get_acquisition_values(model, "ei", y_best)
    
    var_norm = (var_values - np.min(var_values)) / (np.max(var_values) - np.min(var_values) + 1e-9)
    ei_norm = (ei_values - np.min(ei_values)) / (np.max(ei_values) - np.min(ei_values) + 1e-9)
    
    ax.plot(x_grid, var_norm, 'b-', linewidth=2, label='后验方差')
    ax.plot(x_grid, ei_norm, 'g-', linewidth=2, label='期望改进')
    
    if measurements is not None and len(measurements) > 0:
        doses_obs = [m[0] for m in measurements]
        ax.scatter(doses_obs, [0] * len(doses_obs), c='red', s=100, marker='|', zorder=5, label='已测量')
    
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
    fig = plt.figure(figsize=figsize)
    
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
        ax5.plot([i, i], [0, 1], 'k-', alpha=0.3)
        ax5.axvline(i, color='k', alpha=0.1)
        ax5.boxplot(posterior, positions=[i], widths=0.6, patch_artist=True, boxprops=dict(facecolor='steelblue', alpha=0.5))
    
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
    
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    stats = model.get_posterior_stats()
    if lang == "zh":
        info_text = f"阈值估计: {stats['threshold']['mean']:.2f} ± {stats['threshold']['std']:.2f} D\n"
        info_text += f"基线反应: {stats['baseline']['mean']:.2f} μm\n"
        info_text += f"最大反应: {stats['max_response']['mean']:.2f} μm\n"
        info_text += f"斜率: {stats['slope']['mean']:.2f}\n"
        info_text += f"噪声: {stats['sigma']['mean']:.2f} μm"
    else:
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
    import os
    os.makedirs(save_dir, exist_ok=True)
    measurements = list(zip(doses, responses))
    figures = {}
    
    from .basic_plots import plot_dose_response_curve, plot_all_posteriors
    from .evolution_plots import plot_posterior_evolution_standalone, plot_uncertainty_shrinkage_standalone
    
    fig1 = plot_dose_response_curve(
        model,
        simulator=simulator,
        measurements=measurements,
        title="剂量-反应拟合曲线" if lang == "zh" else "Dose-Response Curve",
    )
    fig1_path = os.path.join(save_dir, f"01_dose_response_{lang}.png")
    save_figure(fig1, fig1_path)
    figures["dose_response"] = fig1
    
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
    
    fig3 = plot_all_posteriors(model)
    fig3_path = os.path.join(save_dir, f"03_posterior_distributions_{lang}.png")
    save_figure(fig3, fig3_path)
    figures["posterior_distributions"] = fig3
    
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
    
    if models_evolution and measurements_evolution:
        fig6 = plot_posterior_evolution_standalone(
            models_evolution,
            measurements_evolution,
            lang=lang,
        )
        fig6_path = os.path.join(save_dir, f"06_posterior_evolution_{lang}.png")
        save_figure(fig6, fig6_path)
        figures["posterior_evolution"] = fig6
        
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


__all__ = [
    'plot_acquisition_function',
    'plot_learning_process',
    'plot_learning_trajectory',
    'plot_error_distribution',
    'plot_measurement_count',
    'plot_true_vs_estimated',
    'plot_acquisition_curve',
    'plot_summary_report',
    'plot_all_figures',
]
