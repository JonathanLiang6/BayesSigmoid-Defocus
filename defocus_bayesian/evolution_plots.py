"""
后验演变可视化模块 - 后验分布演变和不确定性收缩
"""

from typing import Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .model import SigmoidModel

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_posterior_evolution_standalone(
    models: List[SigmoidModel],
    measurements_list: List[List[Tuple[float, float]]],
    figsize: Tuple[float, float] = (16, 10),
    lang: str = "en",
) -> Figure:
    n_plots = len(models)
    n_rows = (n_plots + 1) // 2
    n_cols = 2 if n_plots > 1 else 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()
    
    for i, (model, measurements) in enumerate(zip(models, measurements_list)):
        ax = axes[i]
        posterior = model.trace.posterior["threshold"].values.reshape(-1)
        stats = model.get_posterior_stats()
        threshold_mean = stats["threshold"]["mean"]
        threshold_std = stats["threshold"]["std"]
        
        ax.hist(posterior, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black')
        mean_val = np.mean(posterior)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'均值: {mean_val:.3f}')
        ax.axvline(threshold_mean - threshold_std, color='orange', linestyle=':', linewidth=1.5)
        ax.axvline(threshold_mean + threshold_std, color='orange', linestyle=':', linewidth=1.5, label=f'±σ: {threshold_std:.3f}')
        
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
    
    for i in range(n_plots, len(axes)):
        axes[i].axis('off')
    
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    doses = np.linspace(0.5, 6.0, 100)
    
    stats_early = model_early.get_posterior_stats()
    std_early = stats_early["threshold"]["std"]
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
    
    stats_late = model_late.get_posterior_stats()
    std_late = stats_late["threshold"]["std"]
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
    
    reduction_pct = ((std_early - std_late) / std_early * 100) if std_early > 0 else 0
    
    if lang == "zh":
        plt.suptitle(f'估计精度提升：不确定性从 {std_early:.3f} 降低到 {std_late:.3f} (减少 {reduction_pct:.1f}%)', fontsize=16, y=1.02)
    else:
        plt.suptitle(f'Precision Improvement: Uncertainty Reduced from {std_early:.3f} to {std_late:.3f} ({reduction_pct:.1f}% Reduction)', fontsize=16, y=1.02)
    
    plt.tight_layout()
    return fig


__all__ = [
    'plot_posterior_evolution_standalone',
    'plot_uncertainty_shrinkage_standalone',
]
