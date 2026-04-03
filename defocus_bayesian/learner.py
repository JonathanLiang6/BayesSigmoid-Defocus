"""
主动学习器模块 - 主控制类
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

from .model import SigmoidModel
from .acquisition import AcquisitionFunction

logger = logging.getLogger(__name__)


@dataclass
class LearningResult:
    """主动学习结果数据类"""
    threshold_estimate: float
    threshold_std: float
    best_dose: float
    n_measurements: int
    stop_reason: str
    measurements: List[Tuple[float, float]]  # (dose, response) 列表
    converged: bool


class SigmoidActiveLearner:
    """
    Sigmoid 主动学习器。
    
    实现完整的主动学习流程：
    1. 双点启动（2.0D, 4.0D）
    2. 贝叶斯 MCMC 推断
    3. 主动学习推荐下一剂量
    4. 自动停止判断
    """
    
    def __init__(
        self,
        max_measurements: int = 5,
        uncertainty_threshold: float = 2.0,
        dose_range: Tuple[float, float] = (0.5, 6.0),
        n_grid: int = 100,
        mcmc_chains: int = 4,
        mcmc_tune: int = 2000,
        mcmc_draws: int = 1000,
        random_seed: Optional[int] = None,
    ):
        """
        初始化主动学习器。
        
        Args:
            max_measurements: 最大测量次数
            uncertainty_threshold: 不确定性阈值 (μm)，低于此值停止
            dose_range: 剂量范围 (min, max) D
            n_grid: 搜索网格点数
            mcmc_chains: MCMC 链数
            mcmc_tune: MCMC 调优步数
            mcmc_draws: MCMC 后验样本数
            random_seed: 随机种子
        """
        self.max_measurements = max_measurements
        self.uncertainty_threshold = uncertainty_threshold
        self.dose_range = dose_range
        self.n_grid = n_grid
        self.random_seed = random_seed
        
        # 初始化模型和采集函数
        self.model = SigmoidModel(
            n_chains=mcmc_chains,
            tune=mcmc_tune,
            draws=mcmc_draws,
            target_accept=0.85,
            random_seed=random_seed,
        )
        
        self.acquisition = AcquisitionFunction(
            x_min=dose_range[0],
            x_max=dose_range[1],
            n_grid=n_grid,
        )
        
        # 存储观测数据
        self.doses: List[float] = []
        self.responses: List[float] = []
        self.round_num: int = 0
        
    def reset(self) -> None:
        """重置学习器状态"""
        self.doses = []
        self.responses = []
        self.round_num = 0
        self.model = SigmoidModel(
            n_chains=self.model.n_chains,
            tune=self.model.tune,
            draws=self.model.draws,
            target_accept=self.model.target_accept,
            random_seed=self.random_seed,
        )
        
    def add_measurement(self, dose: float, response: float) -> None:
        """
        添加测量数据。
        
        Args:
            dose: 测量剂量 (D)
            response: 观测反应 (μm)
        """
        self.doses.append(dose)
        self.responses.append(response)
        self.round_num += 1
        logger.info(f"第 {self.round_num} 轮: 剂量 {dose:.2f} D → 反应 {response:.1f} μm")
        
    def get_starting_doses(self) -> List[float]:
        """
        获取双点启动的初始剂量。
        
        Returns:
            初始剂量列表 [2.0, 4.0]
        """
        return [2.0, 4.0]
        
    def should_stop(self) -> Tuple[bool, str]:
        """
        检查是否应该停止。
        
        停止条件：
        1. 最佳剂量处的后验标准差 < uncertainty_threshold
        2. 已测次数 >= max_measurements
        3. 最高剂量 >= 6.0 D 且最大反应 < 10 μm（低反应者）
        
        Returns:
            (是否停止, 停止原因)
        """
        # 条件2：最大测量次数
        if len(self.doses) >= self.max_measurements:
            return True, f"达到最大测量次数 ({self.max_measurements})"
        
        # 需要至少2个点才能拟合模型
        if len(self.doses) < 2:
            return False, ""
        
        # 拟合模型以检查其他条件
        try:
            self.model.fit(
                np.array(self.doses),
                np.array(self.responses),
            )
        except Exception as e:
            logger.warning(f"模型拟合失败: {e}")
            return False, ""
        
        # 找到最佳剂量
        x_grid = np.linspace(self.dose_range[0], self.dose_range[1], self.n_grid)
        best_dose, _ = self.model.find_best_dose(x_grid)
        
        # 条件1：不确定性达标
        uncertainty = self.model.get_uncertainty_at_dose(best_dose)
        if uncertainty < self.uncertainty_threshold:
            return True, f"最佳剂量处后验标准差={uncertainty:.2f} μm < {self.uncertainty_threshold} μm"
        
        # 条件3：低反应者
        max_dose = max(self.doses)
        max_response = max(self.responses)
        if max_dose >= 6.0 and max_response < 10:
            return True, f"最高剂量={max_dose:.1f}D 且最大反应={max_response:.1f}μm < 10μm（低反应者）"
        
        return False, ""
        
    def recommend_next_dose(self) -> Tuple[float, str]:
        """
        推荐下一个测量剂量。
        
        Returns:
            (推荐剂量, 策略名称)
        """
        if self.round_num < 2:
            # 双点启动
            starting_doses = self.get_starting_doses()
            dose = starting_doses[self.round_num]
            return dose, "双点启动"
        
        # 拟合模型
        self.model.fit(
            np.array(self.doses),
            np.array(self.responses),
        )
        
        # 使用采集函数推荐
        y_best = max(self.responses) if self.responses else None
        
        dose, strategy = self.acquisition.recommend(
            self.model,
            round_num=self.round_num + 1,  # 下一轮
            y_best=y_best,
            exclude_points=np.array(self.doses),
        )
        
        return dose, strategy
        
    def run_step(self, response: Optional[float] = None) -> Dict[str, Any]:
        """
        执行单步主动学习。
        
        Args:
            response: 上一步推荐剂量的观测反应（首次调用为None）
            
        Returns:
            包含下一步推荐或最终结果的字典
        """
        # 如果有观测反应，添加到数据
        if response is not None and len(self.doses) > 0:
            # 获取上一步推荐的剂量
            last_dose = self.doses[-1]
            self.responses[-1] = response
            logger.info(f"测量剂量 {last_dose:.2f} D → 反应 {response:.1f} μm")
        
        # 检查停止条件
        should_stop, stop_reason = self.should_stop()
        if should_stop:
            return {
                "finished": True,
                "stop_reason": stop_reason,
                "result": self.get_result(),
            }
        
        # 推荐下一剂量
        next_dose, strategy = self.recommend_next_dose()
        self.doses.append(next_dose)
        self.responses.append(0.0)  # 占位，等待实际测量
        self.round_num += 1
        
        return {
            "finished": False,
            "next_dose": next_dose,
            "strategy": strategy,
            "round": self.round_num,
        }
        
    def learn(
        self,
        response_func,
        verbose: bool = True,
    ) -> LearningResult:
        """
        执行完整的主动学习流程。
        
        Args:
            response_func: 响应函数，输入剂量返回反应
            verbose: 是否打印详细输出
            
        Returns:
            学习结果
        """
        self.reset()
        
        if verbose:
            print(f"\n开始主动学习流程...")
            print(f"最大测量次数: {self.max_measurements}")
            print(f"不确定性阈值: {self.uncertainty_threshold} μm")
            print("-" * 50)
        
        while True:
            # 推荐下一剂量
            next_dose, strategy = self.recommend_next_dose()
            
            if verbose:
                print(f"\n第 {self.round_num + 1} 轮")
                print(f"推荐剂量: {next_dose:.2f} D ({strategy})")
            
            # 获取观测反应
            response = response_func(next_dose)
            self.add_measurement(next_dose, response)
            
            if verbose:
                print(f"观测反应: {response:.1f} μm")
            
            # 检查停止条件
            should_stop, stop_reason = self.should_stop()
            if should_stop:
                if verbose:
                    print(f"\n停止: {stop_reason}")
                break
        
        result = self.get_result()
        
        if verbose:
            print("-" * 50)
            print(f"估计阈值: {result.threshold_estimate:.2f} ± {result.threshold_std:.2f} D")
            print(f"最佳剂量: {result.best_dose:.2f} D")
            print(f"总测量次数: {result.n_measurements}")
        
        return result
        
    def get_result(self) -> LearningResult:
        """
        获取当前学习结果。
        
        Returns:
            LearningResult 对象
        """
        if len(self.doses) < 2:
            raise RuntimeError("至少需要2个测量点")
        
        # 确保模型已拟合
        if not self.model.is_fitted:
            self.model.fit(
                np.array(self.doses),
                np.array(self.responses),
            )
        
        # 获取后验统计
        stats = self.model.get_posterior_stats()
        threshold_mean = stats["threshold"]["mean"]
        threshold_std = stats["threshold"]["std"]
        
        # 找到最佳剂量
        x_grid = np.linspace(self.dose_range[0], self.dose_range[1], self.n_grid)
        best_dose, _ = self.model.find_best_dose(x_grid)
        
        # 检查收敛性
        convergence = self.model.check_convergence()
        
        return LearningResult(
            threshold_estimate=threshold_mean,
            threshold_std=threshold_std,
            best_dose=best_dose,
            n_measurements=len(self.doses),
            stop_reason="",  # 由调用者填充
            measurements=list(zip(self.doses, self.responses)),
            converged=convergence["converged"],
        )
        
    def load_from_csv(
        self,
        filepath: str,
        dose_col: str = "dose",
        response_col: str = "response",
    ) -> None:
        """
        从 CSV 文件加载历史数据。
        
        Args:
            filepath: CSV 文件路径
            dose_col: 剂量列名
            response_col: 反应列名
        """
        import pandas as pd
        
        df = pd.read_csv(filepath)
        
        if dose_col not in df.columns:
            raise ValueError(f"CSV 中未找到剂量列: {dose_col}")
        if response_col not in df.columns:
            raise ValueError(f"CSV 中未找到反应列: {response_col}")
        
        self.reset()
        
        for _, row in df.iterrows():
            self.doses.append(float(row[dose_col]))
            self.responses.append(float(row[response_col]))
        
        self.round_num = len(self.doses)
        logger.info(f"从 CSV 加载了 {len(self.doses)} 条历史记录")
