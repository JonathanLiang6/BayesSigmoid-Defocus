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
    
    print("=" * 70)
    print("批量模拟测试")
    print("=" * 70)
    print(f"受试者数量: {n_subjects}")
    print(f"输出目录: {output_dir}")
    print()
    
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
        
        all_stats = study.compare_strategies(strategies, verbose=True)
        
        # 保存结果
        results = study.results
        
    else:
        # 运行默认策略
        stats = study.run(create_default_learner, verbose=True)
        results = study.results
        all_stats = {"默认策略": stats}
    
    # 绘制结果图
    if results:
        print("\n生成可视化图表...")
        fig = plot_simulation_results(results)
        save_figure(fig, os.path.join(output_dir, "simulation_results.png"))
    
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
    results_df.to_csv(csv_path, index=False)
    print(f"\n详细结果已保存到: {csv_path}")
    
    # 保存统计摘要
    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("模拟测试统计摘要")
    summary_lines.append("=" * 70)
    
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
    print("\n" + summary_text)
    
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"\n统计摘要已保存到: {summary_path}")
    
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
    
    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
