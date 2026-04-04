"""
批量模拟测试脚本 - 评估主动学习算法性能

运行方式:
    python test_simulation.py [--n_subjects N] [--output_dir DIR]

选项:
    --n_subjects: 模拟受试者数量 (默认: 100)
    --output_dir: 输出目录 (默认: results/)
    --compare: 是否对比不同策略
"""

import argparse
import os
import logging
from datetime import datetime
from typing import Dict, Callable

from defocus_bayesian import SigmoidActiveLearner
from defocus_bayesian.simulate import SimulationStudy
from defocus_bayesian.plot import plot_simulation_results, save_figure


def create_default_learner() -> SigmoidActiveLearner:
    """创建默认学习器"""
    return SigmoidActiveLearner(
        max_measurements=5,
        uncertainty_threshold=2.0,
    )


def create_conservative_learner() -> SigmoidActiveLearner:
    """创建保守学习器（更严格的不确定性阈值）"""
    return SigmoidActiveLearner(
        max_measurements=5,
        uncertainty_threshold=1.5,
    )


def create_exploration_learner() -> SigmoidActiveLearner:
    """创建探索型学习器（更多测量次数）"""
    return SigmoidActiveLearner(
        max_measurements=6,
        uncertainty_threshold=1.5,
    )


def setup_logging(output_dir: str):
    """设置日志记录"""
    log_file = os.path.join(output_dir, "run.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        ]
    )
    
    # 屏蔽arviz的INFO级别输出
    logging.getLogger('arviz').setLevel(logging.WARNING)
    logging.getLogger('arviz.preview').setLevel(logging.WARNING)
    
    return log_file


def run_simulation(
    n_subjects: int = 100,
    output_dir: str = "results",
    compare_strategies: bool = False,
) -> Dict:
    """
    运行批量模拟。
    
    Args:
        n_subjects: 受试者数量
        output_dir: 输出目录
        compare_strategies: 是否对比策略
        
    Returns:
        统计结果字典
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建可视化子目录
    visualization_dir = os.path.join(output_dir, "visualization")
    os.makedirs(visualization_dir, exist_ok=True)
    
    # 设置日志
    log_file = setup_logging(output_dir)
    logger = logging.getLogger(__name__)
    
    print("=" * 60)
    print("批量模拟测试")
    print("=" * 60)
    print(f"受试者数量: {n_subjects}")
    print(f"输出目录: {output_dir}\n")
    
    # 记录到日志
    logger.info("=" * 60)
    logger.info("批量模拟测试")
    logger.info("=" * 60)
    logger.info(f"受试者数量: {n_subjects}")
    logger.info(f"输出目录: {output_dir}")
    
    # 创建模拟研究
    study = SimulationStudy(
        n_subjects=n_subjects,
        random_seed=42,
    )
    
    if compare_strategies:
        # 对比不同策略
        strategies: Dict[str, Callable] = {
            "默认策略 (unc=2.0, max=5)": create_default_learner,
            "保守策略 (unc=1.5, max=5)": create_conservative_learner,
            "探索策略 (unc=1.5, max=6)": create_exploration_learner,
        }
        
        all_stats = study.compare_strategies(strategies, verbose=False)
        results = study.results
        
        # 记录详细结果到日志
        for i, r in enumerate(study.results):
            logger.info(f"受试者 {i+1}/{n_subjects}: "
                       f"真实阈值={r['true_threshold']:.2f}D, "
                       f"估计={r['estimated_threshold']:.2f}±{r['threshold_std']:.2f}D, "
                       f"误差={r['threshold_error']:.2f}D, "
                       f"测量次数={r['n_measurements']}")
        
    else:
        # 运行默认策略 - 简化终端输出
        print("开始模拟...")
        
        study.results = []
        
        for i in range(n_subjects):
            from defocus_bayesian.simulate import SubjectSimulator
            
            # 创建新的模拟器和学习器
            simulator = SubjectSimulator(random_seed=42 + i)
            simulator.generate_subject()
            
            learner = create_default_learner()
            
            # 运行主动学习（不输出详细信息）
            result = learner.learn(
                simulator.get_response_function(),
                verbose=False,
            )
            
            # 计算误差
            threshold_error = abs(result.threshold_estimate - simulator.true_params.threshold)
            
            # 记录结果
            study.results.append({
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
            
            # 记录到日志
            logger.info(f"受试者 {i+1}/{n_subjects}: "
                       f"真实阈值={simulator.true_params.threshold:.2f}D, "
                       f"估计={result.threshold_estimate:.2f}±{result.threshold_std:.2f}D, "
                       f"误差={threshold_error:.2f}D, "
                       f"测量次数={result.n_measurements}")
            
            # 简单的进度显示
            if (i + 1) % 10 == 0 or (i + 1) == n_subjects:
                print(f"  已完成: {i + 1}/{n_subjects}")
        
        # 计算统计量
        stats = study._compute_statistics()
        all_stats = {"默认策略": stats}
    
    results = study.results
    
    # 绘制结果图
    if results:
        print("\n正在生成图表...")
        fig = plot_simulation_results(results)
        fig_path = os.path.join(output_dir, "visualization", "simulation_results.png")
        save_figure(fig, fig_path)
    
    # 保存详细结果到 CSV
    import pandas as pd
    
    results_df = pd.DataFrame([
        {
            "subject_id": r["subject_id"],
            "true_threshold": r["true_threshold"],
            "estimated_threshold": r["estimated_threshold"],
            "threshold_std": r["threshold_std"],
            "threshold_error": r["threshold_error"],
            "n_measurements": r["n_measurements"],
            "best_dose": r["best_dose"],
            "converged": r["converged"],
        }
        for r in results
    ])
    
    csv_path = os.path.join(output_dir, "simulation_results.csv")
    results_df.to_csv(csv_path, index=False, encoding='utf-8')
    
    # 保存统计摘要
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append("模拟测试统计摘要")
    summary_lines.append("=" * 60)
    
    for name, stats in all_stats.items():
        summary_lines.append(f"\n{name}:")
        summary_lines.append(f"  受试者数量: {stats['n_subjects']}")
        summary_lines.append(f"  平均测量次数: {stats['mean_measurements']:.2f} ± {stats['std_measurements']:.2f}")
        summary_lines.append(f"  中位数测量次数: {stats['median_measurements']:.1f}")
        summary_lines.append(f"  测量次数范围: [{stats['min_measurements']:.0f}, {stats['max_measurements']:.0f}]")
        summary_lines.append(f"  阈值 RMSE: {stats['rmse_threshold']:.3f} D")
        summary_lines.append(f"  阈值 MAE: {stats['mae_threshold']:.3f} D")
        summary_lines.append(f"  收敛率 (MCMC): {stats['convergence_rate']:.1f}%")
    
    summary_text = "\n".join(summary_lines)
    
    # 打印统计结果到终端
    print("\n" + summary_text)
    
    # 记录到日志
    logger.info("\n" + summary_text)
    
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    
    print(f"\n结果已保存到 {output_dir}/ 目录:")
    print(f"  - {os.path.basename(log_file)}")
    print(f"  - simulation_results.csv")
    print(f"  - visualization/simulation_results.png")
    print(f"  - summary.txt")
    
    return all_stats


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="批量模拟测试 - 评估主动学习算法性能",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python test_simulation.py
    python test_simulation.py --n_subjects 50
    python test_simulation.py --n_subjects 100 --compare
        """
    )
    
    parser.add_argument(
        "--n_subjects",
        type=int,
        default=100,
        help="模拟受试者数量 (默认: 100)",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="输出目录 (默认: results/)",
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="对比不同策略",
    )
    
    args = parser.parse_args()
    
    # 运行模拟
    run_simulation(
        n_subjects=args.n_subjects,
        output_dir=args.output_dir,
        compare_strategies=args.compare,
    )
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
