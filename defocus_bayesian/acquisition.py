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
        
        return self.x_grid, acq_values  # 避免不必要的数组复制
    
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
        
        return self.x_grid, ei  # 避免不必要的数组复制
    
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
        # 【优化点】融合后验方差+期望改进+EER，解决局部最优、推荐偏差问题
        if round_num == 3:
            # 第3轮：使用方差采集函数
            x_grid, var_values = self.posterior_variance(model)
            strategy = "后验方差"
            acq_values = var_values
            
        elif round_num >= 4:
            # 第4轮起：融合后验方差、期望改进和EER
            if y_best is None:
                raise ValueError("第4轮起需要提供 y_best 参数")
            
            # 计算三种采集函数
            x_grid, var_values = self.posterior_variance(model)
            _, ei_values = self.expected_improvement(model, y_best, xi=0.03)
            _, eer_values = self.expected_error_reduction(model)
            
            # 归一化
            var_norm = (var_values - np.min(var_values)) / (np.max(var_values) - np.min(var_values) + 1e-9)
            ei_norm = (ei_values - np.min(ei_values)) / (np.max(ei_values) - np.min(ei_values) + 1e-9)
            eer_norm = (eer_values - np.min(eer_values)) / (np.max(eer_values) - np.min(eer_values) + 1e-9)
            
            # 融合策略：轮次越晚，越重视EER和期望改进
            # 进一步增加EER的权重，使算法更加关注误差减少，提高精度
            weight_eer = min(0.55, 0.25 + (round_num - 3) * 0.10)
            weight_ei = min(0.25, 0.15 + (round_num - 3) * 0.03)
            weight_var = 1.0 - weight_eer - weight_ei
            
            acq_values = weight_eer * eer_norm + weight_ei * ei_norm + weight_var * var_norm
            strategy = "融合策略(EER)"
            
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
    
    def expected_error_reduction(
        self,
        model: SigmoidModel,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        预期误差减少采集函数（EER）。
        
        计算下一个测量点对 ED50 后验标准差的预期降幅。
        
        Args:
            model: 已拟合的 Sigmoid 模型
            
        Returns:
            (网格点数组, 采集函数值数组)
        """
        # 获取当前后验分布的阈值统计信息
        threshold_stats = model.get_posterior_stats()["threshold"]
        current_threshold_mean = threshold_stats["mean"]
        current_threshold_std = threshold_stats["std"]
        
        # 获取所有参数的后验标准差，用于更精确的EER计算
        all_stats = model.get_posterior_stats()
        
        # 从后验分布中获取参数样本
        posterior = model.trace.posterior
        baseline = posterior["baseline"].values.reshape(-1)
        max_response = posterior["max_response"].values.reshape(-1)
        slope = posterior["slope"].values.reshape(-1)
        threshold = posterior["threshold"].values.reshape(-1)
        sigma = posterior["sigma"].values.reshape(-1)
        
        n_samples = len(baseline)
        eer_values = np.zeros(len(self.x_grid))
        
        # 对每个候选剂量点计算 EER
        for i, x in enumerate(self.x_grid):
            # 模拟在该剂量点的反应
            def sigmoid(x, b, m, s, t):
                return b + (m - b) / (1 + np.exp(-s * (x - t)))
            
            # 计算每个样本的预测反应
            y_pred = sigmoid(x, baseline, max_response, slope, threshold)
            
            # 优化的EER计算方法 - 更精确地估计信息增益
            # 1. 基于距离当前阈值估计的远近（核心信息区）
            distance_to_threshold = np.abs(x - current_threshold_mean)
            
            # 2. Sigmoid曲线斜率 - 斜率越大的地方信息量越大
            sigmoid_slope_at_x = (max_response.mean() - baseline.mean()) * np.exp(-slope.mean() * (x - threshold.mean())) / (1 + np.exp(-slope.mean() * (x - threshold.mean())))**2
            slope_factor = np.abs(sigmoid_slope_at_x) / (np.abs(max_response.mean() - baseline.mean()) + 1e-9)
            
            # 3. 基于剂量点的预测不确定性
            pred = model.predict(np.array([x]))
            prediction_std = pred["std"][0]
            
            # 4. 综合计算EER - 多因素加权
            # 距离因子：距离阈值越近，预期信息增益越大
            distance_factor = np.exp(-distance_to_threshold ** 2 / 2.0)  # 高斯衰减
            
            # 不确定性因子：预测不确定性越高，潜在信息增益越大
            uncertainty_factor = prediction_std / (current_threshold_std + 1e-9)
            
            # 斜率因子：曲线斜率大的地方对阈值估计贡献大
            # 但在阈值附近斜率最大，这是最关键的区域
            
            # 5. 计算预期的误差减少量
            # 结合多种因素，更准确地预测信息增益
            expected_reduction = current_threshold_std * distance_factor * (1 + uncertainty_factor * 0.3) * (1 + slope_factor * 0.5)
            eer_values[i] = expected_reduction
        
        # 归一化EER值
        eer_values = (eer_values - np.min(eer_values)) / (np.max(eer_values) - np.min(eer_values) + 1e-9)
        
        return self.x_grid, eer_values
    
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
            strategy: 策略名称 ("variance", "ei" 或 "eer")
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
        elif strategy == "eer":
            return self.expected_error_reduction(model)
        else:
            raise ValueError(f"未知策略: {strategy}")
