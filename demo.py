"""
演示脚本 - 展示单个受试者的主动学习流程

运行方式:
    python demo.py

功能说明:
    - 模拟个性化离焦剂量探索过程
    - 使用贝叶斯主动学习方法
    - 生成标准化的日志和可视化输出
"""

import os
import logging
from datetime import datetime
import numpy as np
from tqdm import tqdm

from defocus_bayesian import SubjectSimulator, SigmoidActiveLearner
from defocus_bayesian.plot import (
    plot_learning_process,
    plot_all_posteriors,
    save_figure,
)


def setup_logging(results_dir: str):
    """设置日志记录
    
    Args:
        results_dir: 结果目录路径
    
    Returns:
        日志文件路径
    """
    # 统一日志文件路径
    log_file = os.path.join(results_dir, "run.log")
    
    # 配置日志，使用覆盖模式
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


def main():
    """主函数 - 演示主动学习流程"""
    
    # 创建结果目录
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 创建可视化子目录
    visualization_dir = os.path.join(results_dir, "visualization")
    os.makedirs(visualization_dir, exist_ok=True)
    
    # 设置日志
    log_file = setup_logging(results_dir)
    logger = logging.getLogger(__name__)
    
    # 控制台输出标题
    print("=" * 60)
    print("个性化离焦剂量探索系统")
    print("=" * 60)
    
    # 设置随机种子以保证可重复性
    random_seed = 42
    np.random.seed(random_seed)
    
    # 创建模拟受试者
    simulator = SubjectSimulator(random_seed=random_seed)
    true_params = simulator.generate_subject()
    
    # 控制台输出真实阈值
    print(f"\n真实阈值: {true_params.threshold:.2f} D")
    
    # 记录详细信息到日志
    logger.info("=" * 60)
    logger.info("个性化离焦剂量探索系统 - 演示")
    logger.info("=" * 60)
    logger.info(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"随机种子: {random_seed}")
    logger.info(f"真实阈值 (ED50): {true_params.threshold:.2f} D")
    logger.info(f"基线反应: {true_params.baseline:.2f} μm")
    logger.info(f"最大反应: {true_params.max_response:.2f} μm")
    logger.info(f"斜率: {true_params.slope:.2f}")
    logger.info(f"噪声标准差: {true_params.sigma:.2f} μm")
    
    # 创建主动学习器
    learner = SigmoidActiveLearner(
        max_measurements=5,
        uncertainty_threshold=2.0,
        random_seed=random_seed,
    )
    
    # 控制台输出学习器配置
    print(f"\n最大测量次数: {learner.max_measurements}")
    print(f"停止阈值: {learner.uncertainty_threshold} μm\n")
    
    # 记录学习器配置到日志
    logger.info(f"最大测量次数: {learner.max_measurements}")
    logger.info(f"不确定性阈值: {learner.uncertainty_threshold} μm")
    logger.info("开始主动学习流程")
    
    # 逐步执行主动学习
    round_num = 0
    
    # 动态更新进度条，使用单行覆盖方式
    import sys
    
    def update_progress(round_num, max_rounds, dose, response):
        """动态更新进度条"""
        progress = round_num / max_rounds
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        percent = round(progress * 100, 1)
        
        # 使用回车符覆盖当前行
        sys.stdout.write(f'\r[进度] {round_num}/{max_rounds} 轮 | [{bar}] {percent}% | 剂量: {dose:.2f} D | 反应: {response:.1f} μm')
        sys.stdout.flush()
    
    round_num = 0
    while True:
        round_num += 1
        
        # 推荐下一剂量
        next_dose, strategy = learner.recommend_next_dose()
        
        # 获取观测反应
        response = simulator.get_response(next_dose)
        
        # 添加到学习器
        learner.add_measurement(next_dose, response)
        
        # 记录到日志
        logger.info(f"第 {round_num} 轮: 剂量 {next_dose:.2f} D, 反应 {response:.1f} μm, 策略: {strategy}")
        
        # 动态更新进度条
        update_progress(round_num, learner.max_measurements, next_dose, response)
        
        # 检查停止条件（从第2轮后开始检查）
        if round_num >= 2:
            should_stop, stop_reason = learner.should_stop()
            
            if should_stop:
                print(f"\n停止原因: {stop_reason}")
                logger.info(f"停止原因: {stop_reason}")
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
                    print(f"  当前估计阈值: {threshold_est:.2f} ± {threshold_std:.2f} D")
                    logger.info(f"  当前阈值估计: {threshold_est:.2f} ± {threshold_std:.2f} D")
                except Exception as e:
                    logger.warning(f"模型拟合跳过: {e}")
    
    # 获取最终结果
    result = learner.get_result()
    
    # 控制台输出最终结果
    print("\n" + "=" * 60)
    print("最终结果")
    print("=" * 60)
    print(f"估计阈值: {result.threshold_estimate:.2f} ± {result.threshold_std:.2f} D")
    print(f"真实阈值: {true_params.threshold:.2f} D")
    print(f"估计误差: {abs(result.threshold_estimate - true_params.threshold):.2f} D")
    print(f"最佳剂量: {result.best_dose:.2f} D")
    print(f"总测试次数: {result.n_measurements}")
    print(f"计算收敛: {'成功' if result.converged else '失败'}")
    
    # 记录最终结果到日志
    logger.info("=" * 60)
    logger.info("最终结果")
    logger.info("=" * 60)
    logger.info(f"估计阈值: {result.threshold_estimate:.2f} ± {result.threshold_std:.2f} D")
    logger.info(f"真实阈值: {true_params.threshold:.2f} D")
    logger.info(f"估计误差: {abs(result.threshold_estimate - true_params.threshold):.2f} D")
    logger.info(f"最佳剂量: {result.best_dose:.2f} D")
    logger.info(f"总测量次数: {result.n_measurements}")
    logger.info(f"MCMC 收敛: {result.converged}")
    logger.info(f"测量记录: {result.measurements}")
    
    # 绘制结果
    print("\n正在生成图表...")
    
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
    
    # 保存学习过程图表（按序号命名）
    fig1_path = os.path.join(results_dir, "visualization", "01_dose_response.png")
    save_figure(fig1, fig1_path)
    logger.info(f"生成图表: {os.path.basename(fig1_path)}")
    
    # 绘制后验分布
    fig2 = plot_all_posteriors(learner.model)
    fig2_path = os.path.join(results_dir, "visualization", "02_posterior_distributions.png")
    save_figure(fig2, fig2_path)
    logger.info(f"生成图表: {os.path.basename(fig2_path)}")
    
    # 控制台输出保存信息
    print(f"\n结果已保存到 {results_dir}/ 目录:")
    print(f"  - {os.path.basename(log_file)}")
    print(f"  - visualization/01_dose_response.png")
    print(f"  - visualization/02_posterior_distributions.png")


if __name__ == "__main__":
    main()
