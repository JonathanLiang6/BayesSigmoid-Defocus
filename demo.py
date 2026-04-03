"""
演示脚本 - 展示单个受试者的主动学习流程

运行方式:
    python demo.py
"""

import numpy as np

from defocus_bayesian import SubjectSimulator, SigmoidActiveLearner
from defocus_bayesian.plot import (
    plot_learning_process,
    plot_all_posteriors,
    save_figure,
)


def main():
    """主函数 - 演示主动学习流程"""
    
    print("=" * 60)
    print("个性化离焦剂量探索系统 - 演示")
    print("=" * 60)
    
    # 设置随机种子以保证可重复性
    random_seed = 42
    np.random.seed(random_seed)
    
    # 创建模拟受试者
    simulator = SubjectSimulator(random_seed=random_seed)
    true_params = simulator.generate_subject()
    
    print(f"\n生成虚拟受试者:")
    print(f"  真实阈值 (ED50): {true_params.threshold:.2f} D")
    print(f"  基线反应: {true_params.baseline:.2f} μm")
    print(f"  最大反应: {true_params.max_response:.2f} μm")
    print(f"  斜率: {true_params.slope:.2f}")
    print(f"  噪声标准差: {true_params.sigma:.2f} μm")
    
    # 创建主动学习器
    learner = SigmoidActiveLearner(
        max_measurements=5,
        uncertainty_threshold=2.0,
        random_seed=random_seed,
    )
    
    print("\n" + "=" * 60)
    print("开始主动学习流程")
    print("=" * 60)
    print(f"最大测量次数: {learner.max_measurements}")
    print(f"不确定性阈值: {learner.uncertainty_threshold} μm")
    
    # 逐步执行主动学习
    round_num = 0
    
    while True:
        round_num += 1
        
        # 推荐下一剂量
        next_dose, strategy = learner.recommend_next_dose()
        
        print(f"\n第 {round_num} 轮")
        print(f"  推荐剂量: {next_dose:.2f} D ({strategy})")
        
        # 获取观测反应
        response = simulator.get_response(next_dose)
        print(f"  观测反应: {response:.1f} μm")
        
        # 添加到学习器
        learner.add_measurement(next_dose, response)
        
        # 检查停止条件（从第2轮后开始检查）
        if round_num >= 2:
            should_stop, stop_reason = learner.should_stop()
            
            if should_stop:
                print(f"\n停止: {stop_reason}")
                break
            
            # 拟合模型并显示当前估计
            if learner.model.is_fitted or len(learner.doses) >= 2:
                try:
                    learner.model.fit(
                        np.array(learner.doses),
                        np.array(learner.responses),
                    )
                    stats = learner.model.get_posterior_stats()
                    threshold_est = stats["threshold"]["mean"]
                    threshold_std = stats["threshold"]["std"]
                    print(f"  当前阈值估计: {threshold_est:.2f} ± {threshold_std:.2f} D")
                except Exception as e:
                    print(f"  模型拟合跳过: {e}")
    
    # 获取最终结果
    result = learner.get_result()
    
    print("\n" + "=" * 60)
    print("最终结果")
    print("=" * 60)
    print(f"估计阈值: {result.threshold_estimate:.2f} ± {result.threshold_std:.2f} D")
    print(f"真实阈值: {true_params.threshold:.2f} D")
    print(f"估计误差: {abs(result.threshold_estimate - true_params.threshold):.2f} D")
    print(f"最佳剂量: {result.best_dose:.2f} D")
    print(f"总测量次数: {result.n_measurements}")
    print(f"MCMC 收敛: {'是' if result.converged else '否'}")
    
    # 绘制结果
    print("\n生成可视化图表...")
    
    # 确保模型已拟合
    if not learner.model.is_fitted:
        learner.model.fit(
            np.array(learner.doses),
            np.array(learner.responses),
        )
    
    # 绘制学习过程
    y_best = max(learner.responses)
    strategy = "ei" if round_num >= 4 else "variance"
    
    fig1 = plot_learning_process(
        learner.model,
        learner.acquisition,
        simulator=simulator,
        measurements=result.measurements,
        strategy=strategy,
        y_best=y_best,
    )
    save_figure(fig1, "learning_process.png")
    
    # 绘制后验分布
    fig2 = plot_all_posteriors(learner.model)
    save_figure(fig2, "posterior_distributions.png")
    
    print("\n演示完成!")
    print("生成的图表:")
    print("  - learning_process.png: 学习过程可视化")
    print("  - posterior_distributions.png: 参数后验分布")


if __name__ == "__main__":
    main()
