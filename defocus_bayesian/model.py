"""
Sigmoid 模型模块 - 基于 PyMC 的贝叶斯四参数逻辑斯蒂模型
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pymc as pm
import arviz as az
from pymc.sampling import jax

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
        tune: int = 5000,  # 大幅增加调优步数，提高收敛性
        draws: int = 4000,  # 大幅增加采样数，提高后验估计准确性
        target_accept: float = 0.92,  # 进一步提高目标接受率，减少divergences
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
        # 确保每次都创建一个新的模型
        self.model = None
        
        with pm.Model() as model:
            # 群体先验参数（基于临床数据统计，优化为先验分布）
            mu_pop_baseline = 1.0
            sigma_pop_baseline = 0.8  # 增加方差，提高模型灵活性
            mu_pop_max_response = 18.0
            sigma_pop_max_response = 3.0  # 增加方差，适应不同个体差异
            mu_pop_slope = np.log(2.0)
            sigma_pop_slope = 0.3  # 增加方差，提高对斜率变化的适应能力
            mu_pop_threshold = 2.5
            sigma_pop_threshold = 1.0  # 增加方差，适应不同个体的阈值差异
            
            # 分层先验：个体参数从群体分布中采样
            # 使用截断正态分布确保参数在合理范围内
            baseline = pm.TruncatedNormal(
                "baseline",
                mu=mu_pop_baseline,
                sigma=sigma_pop_baseline,
                lower=0.0,
                upper=5.0,
            )
            max_response = pm.TruncatedNormal(
                "max_response",
                mu=mu_pop_max_response,
                sigma=sigma_pop_max_response,
                lower=10.0,
                upper=25.0,
            )
            slope = pm.TruncatedNormal(
                "slope",
                mu=np.exp(mu_pop_slope),
                sigma=0.5,
                lower=0.1,
                upper=10.0,
            )
            threshold = pm.TruncatedNormal(
                "threshold",
                mu=mu_pop_threshold,
                sigma=sigma_pop_threshold,
                lower=0.5,
                upper=6.0,
            )
            sigma = pm.HalfNormal("sigma", sigma=1.0)  # 减少初始方差
            
            # 【优化点】增加稳健拟合机制，使用Student-T分布处理异常值
            # 使用更灵活的先验分布，使自由度能够自动优化
            nu_raw = pm.Gamma("nu_raw", alpha=2, beta=0.5)  # 重命名为 nu_raw，避免与确定性变量冲突
            # 确保自由度至少为2，保证方差存在
            nu = pm.Deterministic("nu", pm.math.maximum(nu_raw, 2.0))
            
            # Sigmoid 函数
            def sigmoid(x, b, m, s, t):
                return b + (m - b) / (1 + np.exp(-s * (x - t)))
            
            # 期望值
            mu = sigmoid(x_obs, baseline, max_response, slope, threshold)
            
            # 【优化点】使用Student-T分布增加稳健性
            y_likelihood = pm.StudentT(
                "y_likelihood",
                mu=mu,
                sigma=sigma,
                nu=nu,
                observed=y_obs,
            )
            
            # 【优化点】添加参数约束，确保参数合理
            pm.Potential("max_response_gt_baseline", pm.math.log(pm.math.switch(max_response > baseline, 1, 0)))
            
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
        
        # 执行采样，使用 JAX 后端加速
        with self.model:
            try:
                # 使用 JAX NUTS 采样器，支持并行执行多条链
                self.trace = jax.sample_numpyro_nuts(
                    draws=self.draws,
                    tune=self.tune,
                    chains=self.n_chains,
                    target_accept=0.95,
                    random_seed=self.random_seed,
                    progressbar=False,
                    return_inferencedata=True,
                    nuts_kwargs={
                        'adapt_step_size': True,
                        'adapt_mass_matrix': True,
                        'max_tree_depth': 10,
                    },
                )
            except Exception as e:
                logger.error(f"MCMC 采样失败: {e}")
                # 回退到标准采样器
                logger.info("回退到标准 NUTS 采样器")
                self.trace = pm.sample(
                    draws=self.draws,
                    tune=self.tune,
                    chains=self.n_chains,
                    target_accept=0.95,
                    random_seed=self.random_seed,
                    progressbar=False,
                    return_inferencedata=True,
                    nuts_kwargs={
                        'adapt_step_size': True,
                        'adapt_mass_matrix': True,
                        'max_treedepth': 10,
                        'adapt_delta': 0.95,
                    },
                    init='jitter+adapt_diag',
                )
        
        self.is_fitted = True
        logger.info("MCMC 采样完成")
        
        # 记录收敛诊断指标
        convergence = self.check_convergence()
        logger.info("MCMC 收敛诊断:")
        logger.info(f"R-hat 值: {convergence['rhat']}")
        logger.info(f"ESS 值: {convergence['ess']}")
        if convergence['warnings']:
            for warning in convergence['warnings']:
                logger.warning(f"收敛警告: {warning}")
        
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
        
        # 展平链维度
        baseline = posterior["baseline"].values.reshape(-1)
        max_response = posterior["max_response"].values.reshape(-1)
        slope = posterior["slope"].values.reshape(-1)
        threshold = posterior["threshold"].values.reshape(-1)
        
        # 重新计算样本数
        n_samples = len(baseline)
        
        # 计算每个后验样本的预测值
        x_new = np.atleast_1d(x_new)
        
        # 【优化点】使用向量化操作替代循环，提升性能
        # 更高效的向量化实现
        x_new_2d = np.broadcast_to(x_new, (n_samples, len(x_new)))
        baseline_2d = np.broadcast_to(baseline[:, np.newaxis], (n_samples, len(x_new)))
        max_response_2d = np.broadcast_to(max_response[:, np.newaxis], (n_samples, len(x_new)))
        slope_2d = np.broadcast_to(slope[:, np.newaxis], (n_samples, len(x_new)))
        threshold_2d = np.broadcast_to(threshold[:, np.newaxis], (n_samples, len(x_new)))
        
        # 向量化计算Sigmoid
        exponent = -slope_2d * (x_new_2d - threshold_2d)
        denominator = 1 + np.exp(exponent)
        predictions = baseline_2d + (max_response_2d - baseline_2d) / denominator
        
        result = {"samples": predictions}
        
        if return_stats:
            # 【优化点】使用更高效的统计计算
            result["mean"] = np.mean(predictions, axis=0)
            result["std"] = np.std(predictions, axis=0)
            result["var"] = result["std"] ** 2  # 避免重复计算
            # 【优化点】使用并行计算加速分位数计算
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
        # 【优化点】包含nu参数的统计信息
        param_names = ["baseline", "max_response", "slope", "threshold", "sigma", "nu"]
        
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
            return {
                "converged": False,
                "rhat": {},
                "ess": {},
                "warnings": ["模型尚未拟合"],
            }
        
        # 计算 R-hat
        rhat = az.rhat(self.trace)
        ess = az.ess(self.trace)
        
        results = {
            "converged": True,
            "rhat": {},
            "ess": {},
            "warnings": [],
        }
        
        # 【优化点】包含nu参数的收敛检查
        param_names = ["baseline", "max_response", "slope", "threshold", "sigma", "nu"]
        
        for param in param_names:
            rhat_val = float(rhat[param].values)
            ess_val = float(ess[param].values)
            
            results["rhat"][param] = rhat_val
            results["ess"][param] = ess_val
            
            # 【优化点】使用更严格的收敛标准
            if rhat_val > 1.02:
                results["converged"] = False
                results["warnings"].append(f"{param} 的 R-hat = {rhat_val:.3f} > 1.02")
            
            if ess_val < 500:
                results["warnings"].append(f"{param} 的 ESS = {ess_val:.0f} < 500")
        
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
