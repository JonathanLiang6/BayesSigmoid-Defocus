"""
双语可视化模块
实现中英文名双语标注的图表功能
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from matplotlib.gridspec import GridSpec
from matplotlib.table import Table

from .model import SigmoidModel
from .acquisition import AcquisitionFunction

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def plot_dose_response(
    model: SigmoidModel,
    doses: List[float],
    responses: List[float],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制剂量-反应拟合图
    
    Args:
        model: 已拟合的 Sigmoid 模型
        doses: 剂量列表
        responses: 反应列表
        save_path: 保存路径
        
    Returns:
         matplotlib Figure 对象
    """
    x_grid = np.linspace(0.5, 6.0, 100)
    pred = model.predict(x_grid)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制 95% 置信区间云图
    ax.fill_between(
        x_grid,
        pred['mean'] - 1.96 * pred['std'],
        pred['mean'] + 1.96 * pred['std'],
        alpha=0.2,
        color='blue',
        label='95% 置信区间'
    )
    
    # 绘制均值曲线
    ax.plot(x_grid, pred['mean'], 'b-', linewidth=2, label='Sigmoid 均值曲线')
    
    # 绘制观测点
    ax.scatter(doses, responses, color='red', s=50, zorder=5, label='观测点')
    
    # 设置标题和标签（双语）
    ax.set_title('剂量-反应拟合图 (Dose-Response Curve)', fontsize=14)
    ax.set_xlabel('离焦剂量 (D) / Defocus Dose (D)', fontsize=12)
    ax.set_ylabel('脉络膜厚度变化 (μm) / Choroidal Thickness Change (μm)', fontsize=12)
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_parameter_posteriors(
    model: SigmoidModel,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制参数后验分布图
    
    Args:
        model: 已拟合的 Sigmoid 模型
        save_path: 保存路径
        
    Returns:
         matplotlib Figure 对象
    """
    # 获取后验样本
    posterior = model.get_posterior_samples()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    # 绘制阈值 (ED50)
    axes[0].hist(posterior['threshold'], bins=30, alpha=0.7, color='blue')
    axes[0].set_title('阈值 (ED50) / Threshold (ED50)', fontsize=12)
    axes[0].set_xlabel('值 / Value', fontsize=10)
    axes[0].set_ylabel('频率 / Frequency', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 绘制斜率 (Slope)
    axes[1].hist(posterior['slope'], bins=30, alpha=0.7, color='green')
    axes[1].set_title('斜率 (Slope) / Slope', fontsize=12)
    axes[1].set_xlabel('值 / Value', fontsize=10)
    axes[1].set_ylabel('频率 / Frequency', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # 绘制最大反应 (Max Response)
    axes[2].hist(posterior['max_response'], bins=30, alpha=0.7, color='orange')
    axes[2].set_title('最大反应 / Max Response', fontsize=12)
    axes[2].set_xlabel('值 (μm) / Value (μm)', fontsize=10)
    axes[2].set_ylabel('频率 / Frequency', fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    # 绘制噪声 (Sigma)
    axes[3].hist(posterior['sigma'], bins=30, alpha=0.7, color='red')
    axes[3].set_title('噪声 (Sigma) / Noise (Sigma)', fontsize=12)
    axes[3].set_xlabel('值 (μm) / Value (μm)', fontsize=10)
    axes[3].set_ylabel('频率 / Frequency', fontsize=10)
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_acquisition_function(
    model: SigmoidModel,
    acquisition: AcquisitionFunction,
    round_num: int,
    y_best: Optional[float] = None,
    exclude_points: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制采集函数热力图
    
    Args:
        model: 已拟合的 Sigmoid 模型
        acquisition: 采集函数对象
        round_num: 当前轮次
        y_best: 当前最大反应值
        exclude_points: 已测量的点
        save_path: 保存路径
        
    Returns:
         matplotlib Figure 对象
    """
    # 根据轮次选择策略
    if round_num == 3:
        x_grid, acq_values = acquisition.posterior_variance(model)
        strategy = "后验方差 (Posterior Variance)"
    elif round_num >= 4:
        if y_best is None:
            raise ValueError("第4轮起需要提供 y_best")
        x_grid, acq_values = acquisition.expected_improvement(model, y_best)
        strategy = "期望改进 (Expected Improvement)"
    else:
        raise ValueError("轮次应从3开始")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制热力图
    im = ax.plot(x_grid, acq_values, 'r-', linewidth=2)
    ax.fill_between(x_grid, 0, acq_values, alpha=0.3, color='red')
    
    # 标记已测量的点
    if exclude_points is not None and len(exclude_points) > 0:
        for x_obs in exclude_points:
            ax.axvline(x=x_obs, color='gray', linestyle='--', alpha=0.5)
    
    # 设置标题和标签（双语）
    ax.set_title(f'采集函数热力图 (Acquisition Function) - {strategy}', fontsize=14)
    ax.set_xlabel('离焦剂量 (D) / Defocus Dose (D)', fontsize=12)
    ax.set_ylabel('采集函数值 / Acquisition Value', fontsize=12)
    
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_comparison_table(
    rounds_data: List[Dict[str, float]],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制阶段对比表格
    
    Args:
        rounds_data: 各轮次的数据，每个字典包含 'round', 'ed50', 'sigma' 等键
        save_path: 保存路径
        
    Returns:
         matplotlib Figure 对象
    """
    # 创建 DataFrame
    df = pd.DataFrame(rounds_data)
    
    # 重新排列列顺序
    if 'round' in df.columns:
        df = df[['round', 'ed50', 'sigma', 'best_dose', 'uncertainty']]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # 创建表格
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        colColours=['#f0f0f0'] * len(df.columns),
        cellLoc='center',
        loc='center'
    )
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # 设置标题（双语）
    plt.title('阶段对比表格 (Comparison Table)', fontsize=14)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_all_figures(
    model: SigmoidModel,
    doses: List[float],
    responses: List[float],
    acquisition: AcquisitionFunction,
    round_num: int,
    rounds_data: List[Dict[str, float]],
    y_best: Optional[float] = None,
    save_dir: Optional[str] = None
) -> Dict[str, plt.Figure]:
    """
    绘制所有图表
    
    Args:
        model: 已拟合的 Sigmoid 模型
        doses: 剂量列表
        responses: 反应列表
        acquisition: 采集函数对象
        round_num: 当前轮次
        rounds_data: 各轮次的数据
        y_best: 当前最大反应值
        save_dir: 保存目录
        
    Returns:
        包含所有图表的字典
    """
    figures = {}
    
    # 图 A：剂量-反应拟合图
    if save_dir:
        save_path = f"{save_dir}/dose_response_round_{round_num}.png"
    else:
        save_path = None
    figures['dose_response'] = plot_dose_response(
        model=model,
        doses=doses,
        responses=responses,
        save_path=save_path
    )
    
    # 图 B：参数后验分布图
    if save_dir:
        save_path = f"{save_dir}/parameter_posteriors_round_{round_num}.png"
    else:
        save_path = None
    figures['parameter_posteriors'] = plot_parameter_posteriors(
        model=model,
        save_path=save_path
    )
    
    # 图 C：采集函数热力图
    if round_num >= 3:
        if save_dir:
            save_path = f"{save_dir}/acquisition_function_round_{round_num}.png"
        else:
            save_path = None
        figures['acquisition_function'] = plot_acquisition_function(
            model=model,
            acquisition=acquisition,
            round_num=round_num,
            y_best=y_best,
            exclude_points=np.array(doses),
            save_path=save_path
        )
    
    # 图 D：阶段对比表格
    if save_dir:
        save_path = f"{save_dir}/comparison_table_round_{round_num}.png"
    else:
        save_path = None
    figures['comparison_table'] = plot_comparison_table(
        rounds_data=rounds_data,
        save_path=save_path
    )
    
    return figures
