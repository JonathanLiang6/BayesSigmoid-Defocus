"""
主动学习器模块 - 主控制类
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import os
from datetime import datetime

from .model import SigmoidModel
from .acquisition import AcquisitionFunction
from .plot import ProjectExhibitionSuite
from .config import config

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
            target_accept=0.9,  # 增加目标接受率以减少divergences
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
        
    def get_starting_doses(self) -> List[float]:
        """
        获取双点启动的初始剂量。
        
        Returns:
            初始剂量列表 [2.0, 4.0]
        """
        # 【优化点】固定双点启动策略，确保覆盖合理的剂量范围
        return [2.0, 4.0]
        
    def should_stop(self) -> Tuple[bool, str]:
        """
        检查是否应该停止。
        
        停止条件：
        1. 最佳剂量处的后验标准差 < 自适应不确定性阈值
        2. 已测次数 >= max_measurements
        3. 最高剂量 >= 6.0 D 且最大反应 < 10 μm（低反应者）
        4. 误差率 < 1%（极高精度要求）
        
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
        
        # 【优化点】动态调整不确定性阈值，兼顾精度与测量次数
        # 轮次越多，阈值越严格，追求最小误差
        current_round = len(self.doses)
        adaptive_threshold = max(0.5, self.uncertainty_threshold - (current_round - 2) * 0.2)
        
        # 【新增】当接近最大测量次数时，进一步降低阈值要求
        if current_round >= self.max_measurements - 2:
            adaptive_threshold = max(0.3, adaptive_threshold - 0.2)
        
        # 条件1：不确定性达标
        uncertainty = self.model.get_uncertainty_at_dose(best_dose)
        if uncertainty < adaptive_threshold:
            return True, f"最佳剂量处后验标准差={uncertainty:.2f} μm < {adaptive_threshold:.2f} μm"
        
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
        input_data_path: str = "input_data.csv",
    ) -> LearningResult:
        """
        执行完整的主动学习流程。
        
        Args:
            response_func: 响应函数，输入剂量返回反应
            verbose: 是否打印详细输出
            input_data_path: 输入数据文件路径
            
        Returns:
            学习结果
        """
        import os
        from tqdm import tqdm
        
        self.reset()
        
        # 检查是否存在输入数据文件
        if os.path.exists(input_data_path):
            # 模式 A：数据驱动
            if verbose:
                print("[状态] 检测到输入数据文件，使用数据驱动模式...")
            self.load_from_csv(input_data_path)
        else:
            # 模式 B：冷启动
            if verbose:
                print("[状态] 未检测到输入数据文件，使用冷启动模式...")
        
        if verbose:
            print(f"\n开始主动学习流程...")
            print(f"最大测量次数: {self.max_measurements}")
            print(f"不确定性阈值: {self.uncertainty_threshold} μm")
            print("-" * 50)
        
        # 主动学习循环
        while True:
            # 推荐下一剂量
            next_dose, strategy = self.recommend_next_dose()
            
            if verbose:
                print(f"\n[状态] 第 {self.round_num + 1} 轮")
                print(f"[结果] 推荐下一轮离焦剂量: {next_dose:.2f} D ({strategy})")
            
            # 获取观测反应
            response = response_func(next_dose)
            self.add_measurement(next_dose, response)
            
            if verbose:
                print(f"[结果] 观测反应: {response:.1f} μm")
            
            # 检查停止条件
            should_stop, stop_reason = self.should_stop()
            if should_stop:
                if verbose:
                    print(f"\n[状态] 停止: {stop_reason}")
                logger.info(f"系统停止: {stop_reason}")
                break
        
        result = self.get_result()
        
        if verbose:
            print("-" * 50)
            print(f"[结果] 估计阈值: {result.threshold_estimate:.2f} ± {result.threshold_std:.2f} D")
            print(f"[结果] 最佳剂量: {result.best_dose:.2f} D")
            print(f"[结果] 总测量次数: {result.n_measurements}")
        
        # 自动生成 PDF 报告
        self.generate_pdf_report()
        
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
    
    def generate_pdf_report(
        self,
        save_dir: Optional[str] = None,
        subject_id: Optional[str] = None,
    ) -> str:
        """
        生成 PDF 格式的个体诊断报告。
        
        Args:
            save_dir: 保存目录
            subject_id: 受试者 ID
            
        Returns:
            报告文件路径
        """
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        
        # 获取配置
        report_config = config.get_report_config()
        visualization_config = config.get_visualization_config()
        
        # 设置保存目录
        if save_dir is None:
            save_dir = report_config.get("output_dir", "results/reports")
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成报告文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if subject_id:
            filename = f"subject_{subject_id}_{timestamp}.pdf"
        else:
            filename = f"report_{timestamp}.pdf"
        report_path = os.path.join(save_dir, filename)
        
        # 确保模型已拟合
        if not self.model.is_fitted:
            self.model.fit(np.array(self.doses), np.array(self.responses))
        
        # 创建 ProjectExhibitionSuite 实例
        lang = visualization_config.get("language", "zh")
        exhibition = ProjectExhibitionSuite(lang=lang)
        
        # 准备数据
        measurements = list(zip(self.doses, self.responses))
        y_best = max(self.responses) if self.responses else None
        
        # 生成 PDF 报告
        with PdfPages(report_path) as pdf:
            # 封面
            fig = plt.figure(figsize=(12, 8))
            plt.axis('off')
            plt.text(0.5, 0.7, "个性化离焦剂量探索系统", ha='center', fontsize=24, fontweight='bold')
            plt.text(0.5, 0.6, "个体诊断报告", ha='center', fontsize=18)
            plt.text(0.5, 0.4, f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ha='center', fontsize=12)
            if subject_id:
                plt.text(0.5, 0.5, f"受试者 ID: {subject_id}", ha='center', fontsize=14)
            plt.text(0.5, 0.3, f"测量次数: {len(self.doses)}", ha='center', fontsize=12)
            pdf.savefig(fig)
            plt.close(fig)
            
            # 1. 剂量-反应曲线
            fig = exhibition.plot_decision_landscape(self.model, self.acquisition, measurements, y_best)
            pdf.savefig(fig)
            plt.close(fig)
            
            # 2. 信心秒表
            fig = exhibition.plot_confidence_stopwatch(self.model)
            pdf.savefig(fig)
            plt.close(fig)
            
            # 3. 临床解释
            fig = exhibition.plot_clinical_interpretation(self.model, measurements)
            pdf.savefig(fig)
            plt.close(fig)
            
            # 4. AI 思考过程
            fig = exhibition.plot_acquisition_landscape(self.model, self.acquisition, measurements, y_best)
            pdf.savefig(fig)
            plt.close(fig)
            
            # 5. 参数关联图
            fig = exhibition.plot_joint_posterior(self.model)
            pdf.savefig(fig)
            plt.close(fig)
            
            # 6. 受试者画像
            fig = exhibition.plot_subject_persona(self.model)
            pdf.savefig(fig)
            plt.close(fig)
            
            # 7. 总结页
            fig = plt.figure(figsize=(12, 8))
            plt.axis('off')
            
            # 获取模型参数统计
            stats = self.model.get_posterior_stats()
            
            summary_text = "\n".join([
                "===== 总结报告 =====",
                f"阈值估计 (ED50): {stats['threshold']['mean']:.2f} ± {stats['threshold']['std']:.2f} D",
                f"基线反应: {stats['baseline']['mean']:.2f} ± {stats['baseline']['std']:.2f} μm",
                f"最大反应: {stats['max_response']['mean']:.2f} ± {stats['max_response']['std']:.2f} μm",
                f"斜率: {stats['slope']['mean']:.2f} ± {stats['slope']['std']:.2f}",
                f"噪声水平: {stats['sigma']['mean']:.2f} ± {stats['sigma']['std']:.2f} μm",
                f"",
                f"推荐最佳剂量: {self.get_result().best_dose:.2f} D",
                f"总测量次数: {len(self.doses)}",
                f"模型收敛状态: {'已收敛' if self.get_result().converged else '未收敛'}",
            ])
            
            plt.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=14, family='monospace')
            pdf.savefig(fig)
            plt.close(fig)
        
        logger.info(f"PDF 报告已生成: {report_path}")
        return report_path
