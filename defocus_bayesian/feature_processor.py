"""
特征处理模块 - 用于特征标准化和预处理
"""

import logging
from typing import Optional, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class FeatureProcessor:
    """
    特征处理器，用于特征标准化和预处理。
    
    支持以下功能：
    1. 特征标准化（z-score 标准化）
    2. 特征范围缩放
    3. 特征验证
    """
    
    def __init__(self):
        """
        初始化特征处理器。
        """
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.min: Optional[np.ndarray] = None
        self.max: Optional[np.ndarray] = None
        self.is_fitted = False
    
    def fit(self, features: np.ndarray) -> "FeatureProcessor":
        """
        拟合特征处理器，计算标准化所需的统计量。
        
        Args:
            features: 特征数组，形状为 (n_samples, n_features)
            
        Returns:
            拟合后的特征处理器实例
        """
        if features.ndim != 2:
            raise ValueError("特征数组必须是二维的")
        
        # 计算均值和标准差（用于 z-score 标准化）
        self.mean = np.mean(features, axis=0)
        self.std = np.std(features, axis=0)
        
        # 计算最小值和最大值（用于范围缩放）
        self.min = np.min(features, axis=0)
        self.max = np.max(features, axis=0)
        
        # 处理标准差为 0 的情况
        self.std[self.std == 0] = 1.0
        
        # 处理范围为 0 的情况
        range_vals = self.max - self.min
        range_vals[range_vals == 0] = 1.0
        
        self.is_fitted = True
        logger.info(f"特征处理器拟合完成，特征维度: {features.shape[1]}")
        return self
    
    def transform(self, features: np.ndarray, method: str = "standardize") -> np.ndarray:
        """
        转换特征，应用标准化或范围缩放。
        
        Args:
            features: 特征数组，形状为 (n_samples, n_features)
            method: 转换方法，可选值: "standardize" (z-score 标准化) 或 "scale" (范围缩放到 [0, 1])
            
        Returns:
            转换后的特征数组
        """
        if not self.is_fitted:
            raise RuntimeError("特征处理器尚未拟合，请先调用 fit() 方法")
        
        if features.ndim != 2:
            raise ValueError("特征数组必须是二维的")
        
        if features.shape[1] != len(self.mean):
            raise ValueError(f"特征维度不匹配，期望 {len(self.mean)}，实际 {features.shape[1]}")
        
        if method == "standardize":
            # z-score 标准化
            return (features - self.mean) / self.std
        elif method == "scale":
            # 范围缩放到 [0, 1]
            return (features - self.min) / (self.max - self.min)
        else:
            raise ValueError(f"不支持的转换方法: {method}")
    
    def fit_transform(self, features: np.ndarray, method: str = "standardize") -> np.ndarray:
        """
        拟合并转换特征。
        
        Args:
            features: 特征数组，形状为 (n_samples, n_features)
            method: 转换方法，可选值: "standardize" (z-score 标准化) 或 "scale" (范围缩放到 [0, 1])
            
        Returns:
            转换后的特征数组
        """
        self.fit(features)
        return self.transform(features, method)
    
    def inverse_transform(self, transformed_features: np.ndarray, method: str = "standardize") -> np.ndarray:
        """
        逆转换特征，将标准化或缩放后的特征恢复为原始范围。
        
        Args:
            transformed_features: 转换后的特征数组
            method: 转换方法，必须与 transform 时使用的方法一致
            
        Returns:
            原始范围的特征数组
        """
        if not self.is_fitted:
            raise RuntimeError("特征处理器尚未拟合，请先调用 fit() 方法")
        
        if method == "standardize":
            # 逆 z-score 标准化
            return transformed_features * self.std + self.mean
        elif method == "scale":
            # 逆范围缩放
            return transformed_features * (self.max - self.min) + self.min
        else:
            raise ValueError(f"不支持的转换方法: {method}")
    
    def validate_features(self, features: np.ndarray, expected_features: List[str]) -> bool:
        """
        验证特征数组是否符合预期。
        
        Args:
            features: 特征数组
            expected_features: 预期的特征名称列表
            
        Returns:
            是否验证通过
        """
        if features.ndim != 2:
            logger.error("特征数组必须是二维的")
            return False
        
        if features.shape[1] != len(expected_features):
            logger.error(f"特征维度不匹配，期望 {len(expected_features)}，实际 {features.shape[1]}")
            return False
        
        # 检查是否有 NaN 或无穷值
        if np.isnan(features).any() or np.isinf(features).any():
            logger.error("特征数组包含 NaN 或无穷值")
            return False
        
        logger.info(f"特征验证通过，特征维度: {features.shape[1]}")
        return True
    
    def get_feature_stats(self) -> dict:
        """
        获取特征统计信息。
        
        Returns:
            包含特征统计信息的字典
        """
        if not self.is_fitted:
            raise RuntimeError("特征处理器尚未拟合，请先调用 fit() 方法")
        
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "min": self.min.tolist(),
            "max": self.max.tolist()
        }
