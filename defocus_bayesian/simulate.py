"""
模拟数据生成器 - 用于测试和演示
"""

import logging
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SubjectParameters:
    """受试者真实参数"""
    baseline: float
    max_response: float
    slope: float
    threshold: float
    sigma: float


class SubjectSimulator:
    """
    受试者模拟器。
    
    生成具有真实 Sigmoid 响应的虚拟受试者，用于测试主动学习算法。
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        初始化模拟器。
        
        Args:
            random_seed: 随机种子
        """
        self.rng = np.random.RandomState(random_seed)
        self.true_params: Optional[SubjectParameters] = None
        
    def generate_subject(self) -> SubjectParameters:
        """
        从先验分布生成一个虚拟受试者的真实参数。
        
        先验分布与模型中使用的先验一致：
        - baseline: TruncatedNormal(0, 5), [0, 10]
        - max_response: TruncatedNormal(20, 10), (0, 50]
        - slope: LogNormal(ln(3), 0.5)
        - threshold: TruncatedNormal(2.5, 1.0), [0.5, 6.0]
        - sigma: HalfNormal(5)
        
        Returns:
            SubjectParameters 对象
        """
        # baseline: TruncatedNormal(0, 5), [0, 10]
        while True:
            baseline = self.rng.normal(0, 5)
            if 0 <= baseline <= 10:
                break
        
        # max_response: TruncatedNormal(20, 10), (0, 50]
        while True:
            max_response = self.rng.normal(20, 10)
            if 0.01 < max_response <= 50:
                break
        
        # slope: LogNormal(ln(3), 0.5)
        slope = self.rng.lognormal(np.log(3), 0.5)
        
        # threshold: TruncatedNormal(2.5, 1.0), [0.5, 6.0]
        while True:
            threshold = self.rng.normal(2.5, 1.0)
            if 0.5 <= threshold <= 6.0:
                break
        
        # sigma: HalfNormal(5)
        sigma = np.abs(self.rng.normal(0, 5))
        
        self.true_params = SubjectParameters(
            baseline=baseline,
            max_response=max_response,
            slope=slope,
            threshold=threshold,
            sigma=sigma,
        )
        
        logger.debug(f"生成受试者: threshold={threshold:.2f}, slope={slope:.2f}")
        
        return self.true_params
    
    def get_response(self, dose: float) -> float:
        """
        获取指定剂量的观测反应（带噪声）。
        
        Args:
            dose: 剂量值 (D)
            
        Returns:
            观测反应 (μm)
        """
        if self.true_params is None:
            raise RuntimeError("先生成受试者参数")
        
        # 计算真实响应
        true_response = self._sigmoid(
            dose,
            self.true_params.baseline,
            self.true_params.max_response,
            self.true_params.slope,
            self.true_params.threshold,
        )
        
        # 添加观测噪声
        noise = self.rng.normal(0, self.true_params.sigma)
        observed_response = true_response + noise
        
        return observed_response
    
    def get_response_function(self) -> Callable[[float], float]:
        """
        获取响应函数（用于主动学习）。
        
        Returns:
            响应函数
        """
        return self.get_response
    
    def get_true_curve(
        self,
        dose_range: Tuple[float, float] = (0.5, 6.0),
        n_points: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取真实剂量-反应曲线（无噪声）。
        
        Args:
            dose_range: 剂量范围
            n_points: 点数
            
        Returns:
            (剂量数组, 真实反应数组)
        """
        if self.true_params is None:
            raise RuntimeError("先生成受试者参数")
        
        doses = np.linspace(dose_range[0], dose_range[1], n_points)
        responses = self._sigmoid(
            doses,
            self.true_params.baseline,
            self.true_params.max_response,
            self.true_params.slope,
            self.true_params.threshold,
        )
        
        return doses, responses
    
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


class SimulationStudy:
    """
    模拟研究类。
    
    批量运行主动学习流程，评估算法性能。
    """
    
    def __init__(
        self,
        n_subjects: int = 100,
        random_seed: int = 42,
    ):
        """
        初始化模拟研究。
        
        Args:
            n_subjects: 受试者数量
            random_seed: 随机种子
        """
        self.n_subjects = n_subjects
        self.random_seed = random_seed
        self.results: List[Dict] = []
        
    def run(
        self,
        learner_factory,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        运行模拟研究。
        
        Args:
            learner_factory: 创建学习器的工厂函数
            verbose: 是否打印进度
            
        Returns:
            统计结果字典
        """
        from .learner import SigmoidActiveLearner
        
        self.results = []
        
        if verbose:
            print(f"\n开始模拟 {self.n_subjects} 个受试者...")
            print("=" * 60)
        
        for i in range(self.n_subjects):
            # 创建新的模拟器和学习器
            simulator = SubjectSimulator(random_seed=self.random_seed + i)
            simulator.generate_subject()
            
            learner = learner_factory()
            
            # 运行主动学习
            result = learner.learn(
                simulator.get_response_function(),
                verbose=False,
            )
            
            # 计算误差
            threshold_error = abs(result.threshold_estimate - simulator.true_params.threshold)
            
            # 记录结果
            self.results.append({
                "subject_id": i,
                "true_threshold": simulator.true_params.threshold,
                "estimated_threshold": result.threshold_estimate,
                "threshold_std": result.threshold_std,
                "threshold_error": threshold_error,
                "n_measurements": result.n_measurements,
                "best_dose": result.best_dose,
                "converged": result.converged,
                "measurements": result.measurements,
                "true_params": simulator.true_params,
            })
            
            if verbose and (i + 1) % 10 == 0:
                print(f"已完成: {i + 1}/{self.n_subjects}")
        
        # 计算统计量
        stats = self._compute_statistics()
        
        if verbose:
            print("=" * 60)
            self._print_statistics(stats)
        
        return stats
    
    def _compute_statistics(self) -> Dict[str, float]:
        """计算统计量"""
        n_measurements = [r["n_measurements"] for r in self.results]
        errors = [r["threshold_error"] for r in self.results]
        converged = [r["converged"] for r in self.results]
        
        # 收敛率：因不确定性达标而停止的比例
        # 这里简化为 converged 标志
        convergence_rate = sum(converged) / len(converged) * 100
        
        stats = {
            "n_subjects": self.n_subjects,
            "mean_measurements": np.mean(n_measurements),
            "std_measurements": np.std(n_measurements),
            "median_measurements": np.median(n_measurements),
            "min_measurements": np.min(n_measurements),
            "max_measurements": np.max(n_measurements),
            "rmse_threshold": np.sqrt(np.mean(np.array(errors) ** 2)),
            "mae_threshold": np.mean(errors),
            "std_error": np.std(errors),
            "convergence_rate": convergence_rate,
        }
        
        return stats
    
    def _print_statistics(self, stats: Dict[str, float]) -> None:
        """打印统计结果"""
        print("\n模拟结果统计:")
        print("-" * 40)
        print(f"受试者数量: {stats['n_subjects']}")
        print(f"平均测量次数: {stats['mean_measurements']:.2f} ± {stats['std_measurements']:.2f}")
        print(f"中位数测量次数: {stats['median_measurements']:.1f}")
        print(f"测量次数范围: [{stats['min_measurements']:.0f}, {stats['max_measurements']:.0f}]")
        print(f"阈值 RMSE: {stats['rmse_threshold']:.3f} D")
        print(f"阈值 MAE: {stats['mae_threshold']:.3f} D")
        print(f"收敛率 (MCMC): {stats['convergence_rate']:.1f}%")
        print("-" * 40)
    
    def compare_strategies(
        self,
        strategies: Dict[str, Callable],
        verbose: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        比较不同策略。
        
        Args:
            strategies: 策略名称到工厂函数的映射
            verbose: 是否打印结果
            
        Returns:
            各策略的统计结果
        """
        all_stats = {}
        
        for name, factory in strategies.items():
            if verbose:
                print(f"\n{'='*60}")
                print(f"策略: {name}")
                print(f"{'='*60}")
            
            stats = self.run(factory, verbose=verbose)
            all_stats[name] = stats
        
        if verbose:
            print(f"\n{'='*60}")
            print("策略对比:")
            print(f"{'='*60}")
            for name, stats in all_stats.items():
                print(f"\n{name}:")
                print(f"  平均测量次数: {stats['mean_measurements']:.2f}")
                print(f"  阈值 RMSE: {stats['rmse_threshold']:.3f} D")
                print(f"  收敛率: {stats['convergence_rate']:.1f}%")
        
        return all_stats
