"""
采集函数模块 - 主动学习策略实现
"""

import logging
from typing import Optional, Tuple
import numpy as np
from scipy import stats

from .model import SigmoidModel

logger = logging.getLogger(__name__)


class AcquisitionFunction:
    """
    主动学习采集函数。
    
    支持两种策略：
    1. 后验方差（探索）- 第3轮使用
    2. 期望改进（开发）- 第4轮起使用
    """
    
    def __init__(
        self,
        x_min: float = 0.5,
        x_max: float = 6.0,
        n_grid: int = 100,
    ):
        """
        初始化采集函数。
        
        Args:
            x_min: 最小剂量 (D)
            x_max: 最大剂量 (D)
            n_grid: 网格点数
        """
        self.x_min = x_min
        self.x_max = x_max
        self.n_grid = n_grid
        self.x_grid = np.linspace(x_min, x_max, n_grid)
        
    def posterior_variance(
        self,
        model: SigmoidModel,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        后验方差采集函数（纯探索）。
        
        选择预测方差最大的点，以最大化信息增益。
        
        Args:
            model: 已拟合的 Sigmoid 模型
            
        Returns:
            (网格点数组, 采集函数值数组)
        """
        pred = model.predict(self.x_grid)
        acq_values = pred["var"]  # 使用方差作为采集函数
        
        return self.x_grid.copy(), acq_values
    
    def expected_improvement(
        self,
        model: SigmoidModel,
        y_best: float,
        xi: float = 0.01,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        期望改进采集函数（探索+开发）。
        
        EI(x) = E[max(0, y(x) - y_best)]
        
        Args:
            model: 已拟合的 Sigmoid 模型
            y_best: 当前观测到的最大反应值
            xi: 探索参数（越大越倾向于探索）
            
        Returns:
            (网格点数组, 采集函数值数组)
        """
        pred = model.predict(self.x_grid)
        
        mu = pred["mean"]
        sigma = pred["std"]
        
        # 避免除零
        sigma = np.maximum(sigma, 1e-9)
        
        # 计算 EI
        improvement = mu - y_best - xi
        z = improvement / sigma
        
        # EI = (mu - y_best - xi) * Phi(z) + sigma * phi(z)
        ei = improvement * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
        
        # 处理数值问题
        ei = np.maximum(ei, 0)
        
        return self.x_grid.copy(), ei
    
    def recommend(
        self,
        model: SigmoidModel,
        round_num: int,
        y_best: Optional[float] = None,
        exclude_points: Optional[np.ndarray] = None,
        min_distance: float = 0.1,
    ) -> Tuple[float, str]:
        """
        推荐下一个测量剂量。
        
        Args:
            model: 已拟合的 Sigmoid 模型
            round_num: 当前轮次（从1开始）
            y_best: 当前最大反应值（第4轮起需要）
            exclude_points: 已测量的点（避免重复推荐）
            min_distance: 与已测量点的最小距离
            
        Returns:
            (推荐剂量, 使用的策略名称)
        """
        if round_num == 3:
            # 第3轮：使用方差采集函数
            x_grid, acq_values = self.posterior_variance(model)
            strategy = "后验方差"
            
        elif round_num >= 4:
            # 第4轮起：使用期望改进，根据轮次调整探索参数
            if y_best is None:
                raise ValueError("第4轮起需要提供 y_best 参数")
            # 轮次越晚，探索参数越小，越倾向于开发
            xi = 0.01 if round_num > 4 else 0.05
            x_grid, acq_values = self.expected_improvement(model, y_best, xi=xi)
            strategy = "期望改进"
            
        else:
            raise ValueError("轮次应从3开始（1-2轮为固定双点启动）")
        
        # 应用排除约束
        if exclude_points is not None and len(exclude_points) > 0:
            for x_obs in exclude_points:
                mask = np.abs(x_grid - x_obs) < min_distance
                acq_values[mask] = -np.inf
        
        # 找到采集函数最大的点
        best_idx = np.argmax(acq_values)
        recommended_dose = float(x_grid[best_idx])
        max_acq_value = float(acq_values[best_idx])
        
        # 记录采集函数的原始评分数据
        logger.info(f"采集函数策略: {strategy}")
        logger.info(f"最大采集函数值: {max_acq_value:.4f}")
        logger.info(f"推荐剂量: {recommended_dose:.2f} D")
        logger.debug(f"采集函数完整数据: x_grid={x_grid}, acq_values={acq_values}")
        
        return recommended_dose, strategy
    
    def get_acquisition_values(
        self,
        model: SigmoidModel,
        strategy: str = "variance",
        y_best: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取整个网格的采集函数值（用于可视化）。
        
        Args:
            model: 已拟合的 Sigmoid 模型
            strategy: 策略名称 ("variance" 或 "ei")
            y_best: 当前最大反应值（EI需要）
            
        Returns:
            (网格点数组, 采集函数值数组)
        """
        if strategy == "variance":
            return self.posterior_variance(model)
        elif strategy == "ei":
            if y_best is None:
                raise ValueError("EI策略需要提供 y_best")
            return self.expected_improvement(model, y_best)
        else:
            raise ValueError(f"未知策略: {strategy}")
