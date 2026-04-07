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
import sys
import logging
from datetime import datetime
import numpy as np
from tqdm import tqdm

# 添加项目根目录到Python路径（如果需要）
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from defocus_bayesian import SubjectSimulator, SigmoidActiveLearner
from defocus_bayesian.plot import (
    plot_learning_process,
    plot_all_posteriors,
    save_figure,
    ProjectExhibitionSuite,
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
    print("个性化离焦剂量探索系统 - 演示")
    print("=" * 60)
    
    # 设置随机种子以保证可重复性
    random_seed = 42
    np.random.seed(random_seed)
    
    # 创建模拟受试者
    simulator = SubjectSimulator(random_seed=random_seed)
    true_params = simulator.generate_subject()
    
    # 控制台输出真实阈值
    print(f"\n生成虚拟受试者:")
    print(f"  真实阈值 (ED50): {true_params.threshold:.2f} D")
    print(f"  基线反应: {true_params.baseline:.2f} μm")
    print(f"  最大反应: {true_params.max_response:.2f} μm")
    print(f"  斜率: {true_params.slope:.2f}")
    print(f"  噪声标准差: {true_params.sigma:.2f} μm")
    
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
        max_measurements=12,  # 大幅增加最大测量次数，确保充分收敛
        uncertainty_threshold=0.8,  # 进一步降低不确定性阈值，要求极高精度
        random_seed=random_seed,
    )
    
    # 控制台输出学习器配置
    print("\n" + "=" * 60)
    print("开始主动学习流程")
    print("=" * 60)
    print(f"最大测量次数: {learner.max_measurements}")
    print(f"不确定性阈值: {learner.uncertainty_threshold} μm\n")
    
    # 记录学习器配置到日志
    logger.info(f"最大测量次数: {learner.max_measurements}")
    logger.info(f"不确定性阈值: {learner.uncertainty_threshold} μm")
    logger.info("开始主动学习流程")
    
    # 跟踪每一轮的模型演变，用于生成进步性图表
    models_evolution = []
    measurements_evolution = []
    
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
    
    while True:
        # 获取当前轮次（使用learner内部的round_num）
        current_round = learner.round_num + 1
        
        # 推荐下一剂量
        next_dose, strategy = learner.recommend_next_dose()
        
        # 获取观测反应
        response = simulator.get_response(next_dose)
        
        # 添加到学习器
        learner.add_measurement(next_dose, response)
        
        # 记录到日志
        logger.info(f"第 {current_round} 轮: 剂量 {next_dose:.2f} D, 反应 {response:.1f} μm, 策略: {strategy}")
        
        # 动态更新进度条
        update_progress(current_round, learner.max_measurements, next_dose, response)
        
        # 控制台输出轮次信息
        print(f"\n第 {current_round} 轮")
        print(f"  推荐剂量: {next_dose:.2f} D ({strategy})")
        print(f"  观测反应: {response:.1f} μm")
        
        # 检查停止条件（从第2轮后开始检查）
        if current_round >= 2:
            should_stop, stop_reason = learner.should_stop()
            
            # 检查误差率是否达到1%
            if learner.model.is_fitted:
                # 获取当前估计的阈值
                threshold_stats = learner.model.get_posterior_stats()["threshold"]
                threshold_estimate = threshold_stats["mean"]
                
                # 计算误差率
                error = abs(threshold_estimate - true_params.threshold)
                error_rate = (error / true_params.threshold) * 100 if true_params.threshold > 0 else 0
                
                # 如果误差率 < 1%，停止
                if error_rate < 1.0:
                    print(f"\n停止: 误差率达到 {error_rate:.2f}% < 1%，精度要求达标")
                    logger.info(f"停止原因: 误差率达到 {error_rate:.2f}% < 1%")
                    break
            
            if should_stop:
                print(f"\n停止: {stop_reason}")
                logger.info(f"停止原因: {stop_reason}")
                break
        
        # 保存当前轮次的模型状态，用于生成进步性图表（从第2轮开始保存）
        if current_round >= 2 and learner.model.is_fitted:
            # 创建当前模型的深拷贝
            from copy import deepcopy
            models_evolution.append(deepcopy(learner.model))
            measurements_evolution.append(list(zip(learner.doses.copy(), learner.responses.copy())))
    
    # 获取最终结果
    result = learner.get_result()
    
    # 控制台输出最终结果
    print("\n" + "=" * 80)
    print("最终结果分析")
    print("=" * 80)
    
    # 基本结果
    print("\n【基本结果】")
    print(f"估计阈值 (ED50): {result.threshold_estimate:.2f} ± {result.threshold_std:.2f} D")
    # 计算误差率
    error = abs(result.threshold_estimate - true_params.threshold)
    error_rate = (error / true_params.threshold) * 100 if true_params.threshold > 0 else 0
    
    print(f"真实阈值: {true_params.threshold:.2f} D")
    print(f"估计误差: {error:.2f} D")
    print(f"误差率: {error_rate:.1f}%")
    print(f"最佳剂量: {result.best_dose:.2f} D")
    print(f"总测量次数: {result.n_measurements}")
    print(f"MCMC 收敛: {'是' if result.converged else '否'}")
    
    # 结果解释
    print("\n【结果解释】")
    if error < 0.5 and error_rate < 15:
        print("✓ 估计精度较高，误差在可接受范围内")
    elif error < 1.0 and error_rate < 30:
        print("⚠ 估计精度一般，建议增加测量次数")
    else:
        print("✗ 估计精度较低，建议重新进行测量")
    print(f"📊 误差分析: 误差 {error:.2f} D ({error_rate:.1f}%)")
    
    # 临床意义
    print("\n【临床意义】")
    if result.threshold_estimate < 2.0:
        print("👤 受试者对离焦刺激较为敏感")
    elif result.threshold_estimate > 3.5:
        print("👤 受试者对离焦刺激较为迟钝")
    else:
        print("👤 受试者对离焦刺激反应正常")
    
    print(f"💡 推荐使用 {result.best_dose:.2f} D 的离焦剂量进行后续训练")
    
    # 测量数据回顾
    print("\n【测量数据】")
    for i, (dose, response) in enumerate(result.measurements, 1):
        print(f"  第 {i} 轮: 剂量 {dose:.2f} D → 反应 {response:.1f} μm")
    
    # 记录最终结果到日志
    logger.info("=" * 80)
    logger.info("最终结果分析")
    logger.info("=" * 80)
    # 计算误差率
    error = abs(result.threshold_estimate - true_params.threshold)
    error_rate = (error / true_params.threshold) * 100 if true_params.threshold > 0 else 0
    
    logger.info("【基本结果】")
    logger.info(f"估计阈值 (ED50): {result.threshold_estimate:.2f} ± {result.threshold_std:.2f} D")
    logger.info(f"真实阈值: {true_params.threshold:.2f} D")
    logger.info(f"估计误差: {error:.2f} D")
    logger.info(f"误差率: {error_rate:.1f}%")
    logger.info(f"最佳剂量: {result.best_dose:.2f} D")
    logger.info(f"总测量次数: {result.n_measurements}")
    logger.info(f"MCMC 收敛: {result.converged}")
    
    logger.info("【结果解释】")
    if abs(result.threshold_estimate - true_params.threshold) < 0.5:
        logger.info("估计精度较高，误差在可接受范围内")
    elif abs(result.threshold_estimate - true_params.threshold) < 1.0:
        logger.info("估计精度一般，建议增加测量次数")
    else:
        logger.info("估计精度较低，建议重新进行测量")
    
    logger.info("【临床意义】")
    if result.threshold_estimate < 2.0:
        logger.info("受试者对离焦刺激较为敏感")
    elif result.threshold_estimate > 3.5:
        logger.info("受试者对离焦刺激较为迟钝")
    else:
        logger.info("受试者对离焦刺激反应正常")
    
    logger.info(f"推荐使用 {result.best_dose:.2f} D 的离焦剂量进行后续训练")
    
    logger.info("【测量数据】")
    for i, (dose, response) in enumerate(result.measurements, 1):
        logger.info(f"  第 {i} 轮: 剂量 {dose:.2f} D → 反应 {response:.1f} μm")
    
    # 绘制结果
    print("\n正在生成图表...")
    
    # 确保模型已拟合
    if not learner.model.is_fitted:
        learner.model.fit(
            np.array(learner.doses),
            np.array(learner.responses),
        )
    
    # 准备轮次数据
    rounds_data = [
        {
            'round': i+1,
            'ed50': result.threshold_estimate,
            'sigma': 2.0,
            'best_dose': result.best_dose,
            'uncertainty': learner.model.get_uncertainty_at_dose(result.best_dose)
        }
        for i in range(len(learner.doses))
    ]
    
    # 【优化点】使用新的可视化功能，生成全套图表
    from defocus_bayesian.plot import plot_all_figures
    
    # 生成英文图表
    plot_all_figures(
        model=learner.model,
        doses=learner.doses,
        responses=learner.responses,
        acquisition=learner.acquisition,
        round_num=len(learner.doses),
        rounds_data=rounds_data,
        y_best=max(learner.responses) if learner.responses else None,
        simulator=simulator,
        save_dir=visualization_dir,
        lang="en",
        models_evolution=models_evolution,
        measurements_evolution=measurements_evolution,
    )
    
    # 生成中文图表
    plot_all_figures(
        model=learner.model,
        doses=learner.doses,
        responses=learner.responses,
        acquisition=learner.acquisition,
        round_num=len(learner.doses),
        rounds_data=rounds_data,
        y_best=max(learner.responses) if learner.responses else None,
        simulator=simulator,
        save_dir=visualization_dir,
        lang="zh",
        models_evolution=models_evolution,
        measurements_evolution=measurements_evolution,
    )
    
    # 生成高级可视化图 - ProjectExhibitionSuite
    print("\n正在生成高级可视化图...")
    
    # 创建展示套件实例
    exhibition_en = ProjectExhibitionSuite(lang="en")
    exhibition_zh = ProjectExhibitionSuite(lang="zh")
    
    # 创建展示目录
    exhibition_dir = os.path.join(results_dir, "exhibition")
    os.makedirs(exhibition_dir, exist_ok=True)
    
    # 生成英文高级图表
    exhibition_en.generate_full_exhibition(
        model=learner.model,
        acquisition=learner.acquisition,
        measurements=result.measurements,
        y_best=max(learner.responses) if learner.responses else None,
        save_dir=exhibition_dir,
        models_evolution=models_evolution if len(models_evolution) > 0 else None,
        measurements_evolution=measurements_evolution if len(measurements_evolution) > 0 else None,
    )
    
    # 生成中文高级图表
    exhibition_zh.generate_full_exhibition(
        model=learner.model,
        acquisition=learner.acquisition,
        measurements=result.measurements,
        y_best=max(learner.responses) if learner.responses else None,
        save_dir=exhibition_dir,
        models_evolution=models_evolution if len(models_evolution) > 0 else None,
        measurements_evolution=measurements_evolution if len(measurements_evolution) > 0 else None,
    )
    
    # 控制台输出保存信息
    print(f"\n结果已保存到 {results_dir}/ 目录:")
    print(f"  - {os.path.basename(log_file)}")
    print(f"  visualization/ (共{8 if models_evolution else 5}张图片)")
    print(f"  - 01_dose_response_en.png / 01_dose_response_zh.png (剂量-反应拟合曲线)")
    print(f"  - 02_learning_trajectory_en.png / 02_learning_trajectory_zh.png (学习轨迹)")
    print(f"  - 03_posterior_distributions_en.png / 03_posterior_distributions_zh.png (后验参数分布)")
    print(f"  - 04_acquisition_curve_en.png / 04_acquisition_curve_zh.png (采集函数变化)")
    print(f"  - 05_summary_report_en.png / 05_summary_report_zh.png (汇总报告)")
    if models_evolution:
        print(f"  - 06_posterior_evolution_en.png / 06_posterior_evolution_zh.png (后验分布演变-展示学习进步性)")
        print(f"  - 07_uncertainty_shrinkage_en.png / 07_uncertainty_shrinkage_zh.png (不确定性收缩-展示精度提升)")
    print(f"  exhibition/ (共{12 if models_evolution else 10}张图片)")
    print(f"  - 01_decision_landscape_en.png / 01_decision_landscape_zh.png (决策景观)")
    if models_evolution:
        print(f"  - 02_posterior_evolution_en.png / 02_posterior_evolution_zh.png (后验分布演变)")
    print(f"  - 03_clinical_interpretation_en.png / 03_clinical_interpretation_zh.png (临床解释)")
    print(f"  - 04_acquisition_landscape_en.png / 04_acquisition_landscape_zh.png (AI思考过程)")
    if models_evolution:
        print(f"  - 05_uncertainty_shrinkage_en.png / 05_uncertainty_shrinkage_zh.png (不确定性收缩)")
    print(f"  - 06_joint_posterior_en.png / 06_joint_posterior_zh.png (参数关联图)")
    print(f"  - 07_subject_persona_en.png / 07_subject_persona_zh.png (受试者画像)")
    print(f"  - 08_loss_curve_en.png / 08_loss_curve_zh.png (损失曲线)")
    print(f"  - 09_distribution_comparison_en.png / 09_distribution_comparison_zh.png (分布对比)")
    print(f"  - 10_feature_importance_en.png / 10_feature_importance_zh.png (特征重要性)")
    if models_evolution:
        print(f"  - 11_temporal_change_en.png / 11_temporal_change_zh.png (时序变化)")
    print(f"  - 12_heatmap_en.png / 12_heatmap_zh.png (特征-剂量-反应热力图)")


if __name__ == "__main__":
    main()
