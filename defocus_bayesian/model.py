"""
Sigmoid 模型模块 - 基于 PyMC 的贝叶斯四参数逻辑斯蒂模型
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pymc as pm
import arviz as az

logger = logging.getLogger(__name__)


class SigmoidModel:
    """
    四参数 Sigmoid（逻辑斯蒂）模型的贝叶斯实现。
    
    模型公式：
        y = baseline + (max_response - baseline) / (1 + exp(-slope * (x - threshold))) + ε
        ε ~ N(0, sigma^2)
    """
    
    def __init__(
        self,
        n_chains: int = 4,
        tune: int = 2000,
        draws: int = 1000,
        target_accept: float = 0.85,
        random_seed: Optional[int] = None,
    ):
        """
        初始化 Sigmoid 模型。
        
        Args:
            n_chains: MCMC 链数
            tune: 调优步数
            draws: 后验样本数
            target_accept: 目标接受率
            random_seed: 随机种子
        """
        self.n_chains = n_chains
        self.tune = tune
        self.draws = draws
        self.target_accept = target_accept
        self.random_seed = random_seed
        
        self.model: Optional[pm.Model] = None
        self.trace: Optional[az.InferenceData] = None
        self.is_fitted = False
        
    def build_model(
        self,
        x_obs: np.ndarray,
        y_obs: np.ndarray,
    ) -> pm.Model:
        """
        构建 PyMC 模型。
        
        Args:
            x_obs: 观测剂量数组 (D)
            y_obs: 观测反应数组 (μm)
            
        Returns:
            PyMC 模型实例
        """
        with pm.Model() as model:
            # 先验分布（基于文献和预实验）
            baseline = pm.TruncatedNormal(
                "baseline",
                mu=0,
                sigma=5,
                lower=0,
                upper=10,
            )
            max_response = pm.TruncatedNormal(
                "max_response",
                mu=20,
                sigma=10,
                lower=0.01,
                upper=50,
            )
            slope = pm.LogNormal(
                "slope",
                mu=np.log(3),
                sigma=0.5,
            )
            threshold = pm.TruncatedNormal(
                "threshold",
                mu=2.5,
                sigma=1.0,
                lower=0.5,
                upper=6.0,
            )
            sigma = pm.HalfNormal("sigma", sigma=5)
            
            # Sigmoid 函数
            def sigmoid(x, b, m, s, t):
                return b + (m - b) / (1 + np.exp(-s * (x - t)))
            
            # 期望值
            mu = sigmoid(x_obs, baseline, max_response, slope, threshold)
            
            # 似然函数
            y_likelihood = pm.Normal(
                "y_likelihood",
                mu=mu,
                sigma=sigma,
                observed=y_obs,
            )
            
        self.model = model
        return model
    
    def fit(
        self,
        x_obs: np.ndarray,
        y_obs: np.ndarray,
    ) -> az.InferenceData:
        """
        拟合模型并执行 MCMC 采样。
        
        Args:
            x_obs: 观测剂量数组 (D)
            y_obs: 观测反应数组 (μm)
            
        Returns:
            Arviz InferenceData 对象
        """
        logger.info(f"开始 MCMC 采样: {len(x_obs)} 个观测点")
        
        # 构建模型
        self.build_model(x_obs, y_obs)
        
        # 执行采样
        with self.model:
            self.trace = pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.n_chains,
                target_accept=self.target_accept,
                random_seed=self.random_seed,
                progressbar=False,
            )
        
        self.is_fitted = True
        logger.info("MCMC 采样完成")
        
        return self.trace
    
    def predict(
        self,
        x_new: np.ndarray,
        return_stats: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        对新的剂量点进行预测。
        
        Args:
            x_new: 新的剂量点数组 (D)
            return_stats: 是否返回统计量（均值、标准差等）
            
        Returns:
            包含预测结果的字典
        """
        if not self.is_fitted or self.trace is None:
            raise RuntimeError("模型尚未拟合，请先调用 fit() 方法")
        
        # 从后验获取参数样本
        posterior = self.trace.posterior
        n_samples = self.n_chains * self.draws
        
        # 展平链维度
        baseline = posterior["baseline"].values.reshape(-1)
        max_response = posterior["max_response"].values.reshape(-1)
        slope = posterior["slope"].values.reshape(-1)
        threshold = posterior["threshold"].values.reshape(-1)
        
        # 计算每个后验样本的预测值
        x_new = np.atleast_1d(x_new)
        predictions = np.zeros((n_samples, len(x_new)))
        
        for i in range(n_samples):
            predictions[i, :] = self._sigmoid(
                x_new,
                baseline[i],
                max_response[i],
                slope[i],
                threshold[i],
            )
        
        result = {"samples": predictions}
        
        if return_stats:
            result["mean"] = np.mean(predictions, axis=0)
            result["std"] = np.std(predictions, axis=0)
            result["var"] = np.var(predictions, axis=0)
            result["lower"] = np.percentile(predictions, 2.5, axis=0)
            result["upper"] = np.percentile(predictions, 97.5, axis=0)
        
        return result
    
    def get_posterior_stats(self) -> Dict[str, Dict[str, float]]:
        """
        获取后验参数统计信息。
        
        Returns:
            各参数的后验统计量字典
        """
        if not self.is_fitted or self.trace is None:
            raise RuntimeError("模型尚未拟合")
        
        stats = {}
        param_names = ["baseline", "max_response", "slope", "threshold", "sigma"]
        
        for param in param_names:
            values = self.trace.posterior[param].values.reshape(-1)
            stats[param] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "median": float(np.median(values)),
                "lower_95": float(np.percentile(values, 2.5)),
                "upper_95": float(np.percentile(values, 97.5)),
            }
        
        return stats
    
    def check_convergence(self) -> Dict[str, Any]:
        """
        检查 MCMC 收敛性。
        
        Returns:
            收敛诊断结果字典
        """
        if not self.is_fitted or self.trace is None:
            raise RuntimeError("模型尚未拟合")
        
        # 计算 R-hat
        rhat = az.rhat(self.trace)
        ess = az.ess(self.trace)
        
        results = {
            "converged": True,
            "rhat": {},
            "ess": {},
            "warnings": [],
        }
        
        param_names = ["baseline", "max_response", "slope", "threshold", "sigma"]
        
        for param in param_names:
            rhat_val = float(rhat[param].values)
            ess_val = float(ess[param].values)
            
            results["rhat"][param] = rhat_val
            results["ess"][param] = ess_val
            
            if rhat_val > 1.05:
                results["converged"] = False
                results["warnings"].append(f"{param} 的 R-hat = {rhat_val:.3f} > 1.05")
            
            if ess_val < 200:
                results["warnings"].append(f"{param} 的 ESS = {ess_val:.0f} < 200")
        
        return results
    
    def find_best_dose(self, x_grid: np.ndarray) -> Tuple[float, float]:
        """
        在网格上找到最佳剂量（预测反应最大的点）。
        
        Args:
            x_grid: 剂量搜索网格
            
        Returns:
            (最佳剂量, 预测反应均值)
        """
        pred = self.predict(x_grid)
        best_idx = np.argmax(pred["mean"])
        return float(x_grid[best_idx]), float(pred["mean"][best_idx])
    
    def get_uncertainty_at_dose(self, dose: float) -> float:
        """
        获取指定剂量处的预测不确定性（标准差）。
        
        Args:
            dose: 剂量值 (D)
            
        Returns:
            预测标准差
        """
        pred = self.predict(np.array([dose]))
        return float(pred["std"][0])
    
    def save_posterior(self, filepath: str) -> None:
        """
        保存后验分布到文件。
        
        Args:
            filepath: 保存路径（.nc 格式）
        """
        if not self.is_fitted or self.trace is None:
            raise RuntimeError("模型尚未拟合")
        
        self.trace.to_netcdf(filepath)
        logger.info(f"后验分布已保存到: {filepath}")
    
    def load_posterior(self, filepath: str) -> None:
        """
        从文件加载后验分布。
        
        Args:
            filepath: 文件路径（.nc 格式）
        """
        self.trace = az.from_netcdf(filepath)
        self.is_fitted = True
        logger.info(f"后验分布已从 {filepath} 加载")
    
    @staticmethod
    def _sigmoid(
        x: np.ndarray,
        baseline: float,
        max_response: float,
        slope: float,
        threshold: float,
    ) -> np.ndarray:
        """
        计算 Sigmoid 函数值。
        
        Args:
            x: 输入剂量
            baseline: 基线反应
            max_response: 最大反应
            slope: 斜率参数
            threshold: 阈值（ED50）
            
        Returns:
            Sigmoid 函数值
        """
        return baseline + (max_response - baseline) / (1 + np.exp(-slope * (x - threshold)))
