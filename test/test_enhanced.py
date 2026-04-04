#!/usr/bin/env python3
"""
增强版系统测试脚本
实现结构化输出、日志管理和标准化可视化
"""

import os
import time
import logging
import numpy as np
from tqdm import tqdm
from defocus_bayesian import SigmoidActiveLearner, SubjectSimulator, plot_all_figures


def setup_logging():
    """
    设置日志配置
    """
    # 创建 results 目录
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # 创建 visualization 子目录
    visualization_dir = os.path.join('results', 'visualization')
    if not os.path.exists(visualization_dir):
        os.makedirs(visualization_dir)
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join('results', 'run.log'), encoding='utf-8'),
        ]
    )
    
    # 屏蔽arviz的INFO级别输出
    logging.getLogger('arviz').setLevel(logging.WARNING)
    logging.getLogger('arviz.preview').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)


def test_single_subject(logger):
    """
    测试单个受试者的主动学习过程
    """
    logger.info("=" * 80)
    logger.info("轮次1：测试单个受试者主动学习过程")
    logger.info("=" * 80)
    
    # 创建模拟器（模拟真实受试者的反应）
    simulator = SubjectSimulator(
        threshold=2.5,  # ED50
        slope=3.0,      # 斜率
        max_response=25.0,  # 最大反应
        baseline=0.0,   # 基线
        sigma=2.0,      # 测量噪声
    )
    
    logger.info(f"创建模拟受试者: 真实阈值={simulator.threshold:.2f} D")
    
    # 创建主动学习器
    learner = SigmoidActiveLearner(
        max_measurements=5,
        uncertainty_threshold=2.0,
        dose_range=(0.5, 6.0),
        n_grid=100,
        mcmc_chains=4,
        mcmc_tune=1000,
        mcmc_draws=500,
        random_seed=42,
    )
    
    logger.info(f"初始化主动学习器: 最大测量次数={learner.max_measurements}")
    
    # 定义响应函数（使用模拟器）
    def response_func(dose):
        return simulator.simulate_response(dose)
    
    # 执行主动学习
    logger.info("开始主动学习循环...")
    
    # 模拟轮次进度
    rounds = []
    
    # 手动执行每轮，以便添加进度条
    with tqdm(total=learner.max_measurements, desc="主动学习进度") as pbar:
        while True:
            # 推荐下一剂量
            next_dose, strategy = learner.recommend_next_dose()
            
            # 获取观测反应
            response = response_func(next_dose)
            learner.add_measurement(next_dose, response)
            
            # 记录轮次数据
            rounds.append({
                'round': learner.round_num,
                'dose': next_dose,
                'response': response,
                'strategy': strategy
            })
            
            # 打印控制台输出
            print(f"[轮次 {learner.round_num}] 剂量: {next_dose:.2f} D, 反应: {response:.1f} μm, 策略: {strategy}")
            
            # 检查停止条件
            should_stop, stop_reason = learner.should_stop()
            if should_stop:
                print(f"[停止] {stop_reason}")
                logger.info(f"停止条件: {stop_reason}")
                break
            
            # 更新进度条
            pbar.update(1)
    
    # 获取最终结果
    result = learner.get_result()
    
    # 打印最终结果
    print("\n" + "=" * 60)
    print("最终结果")
    print("=" * 60)
    print(f"估计阈值: {result.threshold_estimate:.2f} ± {result.threshold_std:.2f} D")
    print(f"最佳剂量: {result.best_dose:.2f} D")
    print(f"总测量次数: {result.n_measurements}")
    print(f"是否收敛: {result.converged}")
    print("=" * 60)
    
    # 记录最终结果到日志
    logger.info("\n最终结果:")
    logger.info(f"估计阈值: {result.threshold_estimate:.2f} ± {result.threshold_std:.2f} D")
    logger.info(f"最佳剂量: {result.best_dose:.2f} D")
    logger.info(f"总测量次数: {result.n_measurements}")
    logger.info(f"是否收敛: {result.converged}")
    
    # 生成可视化图表
    if learner.model.is_fitted:
        logger.info("\n生成可视化图表...")
        
        # 准备轮次数据
        rounds_data = []
        for i, r in enumerate(rounds):
            # 拟合模型获取当前轮次的阈值估计
            learner.model.fit(
                np.array([rounds[j]['dose'] for j in range(i+1)]),
                np.array([rounds[j]['response'] for j in range(i+1)])
            )
            stats = learner.model.get_posterior_stats()
            
            rounds_data.append({
                'round': i+1,
                'ed50': stats['threshold']['mean'],
                'sigma': stats['sigma']['mean'],
                'best_dose': result.best_dose,
                'uncertainty': learner.model.get_uncertainty_at_dose(result.best_dose)
            })
        
        # 绘制所有图表
        figures = plot_all_figures(
            model=learner.model,
            doses=[r['dose'] for r in rounds],
            responses=[r['response'] for r in rounds],
            acquisition=learner.acquisition,
            round_num=len(rounds),
            rounds_data=rounds_data,
            y_best=max([r['response'] for r in rounds]),
            save_dir=os.path.join('results', 'visualization')
        )
        
        logger.info(f"可视化图表已保存到: results/visualization")
    
    return result


def test_batch_simulation(logger, n_subjects=10):
    """
    测试批量模拟
    """
    logger.info("\n" + "=" * 80)
    logger.info(f"轮次2：批量模拟测试 - {n_subjects}个受试者")
    logger.info("=" * 80)
    
    results = []
    
    # 批量模拟进度条
    with tqdm(total=n_subjects, desc="批量模拟进度") as pbar:
        for i in range(n_subjects):
            # 创建模拟器
            simulator = SubjectSimulator(random_seed=i)
            simulator.generate_subject()
            
            # 创建主动学习器
            learner = SigmoidActiveLearner(
                max_measurements=5,
                uncertainty_threshold=2.0,
                random_seed=i,
            )
            
            # 定义响应函数
            def response_func(dose):
                return simulator.simulate_response(dose)
            
            # 执行主动学习
            result = learner.learn(
                response_func=response_func,
                verbose=False,
            )
            
            # 记录结果
            results.append({
                'subject_id': i+1,
                'true_threshold': simulator.threshold,
                'estimated_threshold': result.threshold_estimate,
                'threshold_error': abs(result.threshold_estimate - simulator.threshold),
                'n_measurements': result.n_measurements,
                'converged': result.converged
            })
            
            # 更新进度条
            pbar.update(1)
    
    # 计算统计结果
    if results:
        errors = [r['threshold_error'] for r in results]
        n_measurements = [r['n_measurements'] for r in results]
        converged = [r['converged'] for r in results]
        
        logger.info("\n批量模拟统计结果:")
        logger.info(f"受试者数量: {n_subjects}")
        logger.info(f"平均阈值误差: {np.mean(errors):.3f} D")
        logger.info(f"平均测量次数: {np.mean(n_measurements):.2f}")
        logger.info(f"收敛率: {sum(converged)/len(converged):.2f}")
        
        # 打印控制台输出
        print("\n" + "=" * 60)
        print("批量模拟统计结果")
        print("=" * 60)
        print(f"受试者数量: {n_subjects}")
        print(f"平均阈值误差: {np.mean(errors):.3f} D")
        print(f"平均测量次数: {np.mean(n_measurements):.2f}")
        print(f"收敛率: {sum(converged)/len(converged):.2f}")
        print("=" * 60)
    
    return results


def main():
    """
    主测试函数
    """
    # 设置日志
    logger = setup_logging()
    
    logger.info("开始增强版系统测试...")
    print("开始增强版系统测试...")
    print("=" * 80)
    
    # 测试单个受试者
    single_result = test_single_subject(logger)
    
    # 测试批量模拟
    batch_results = test_batch_simulation(logger, n_subjects=5)
    
    logger.info("\n" + "=" * 80)
    logger.info("测试完成！")
    logger.info("所有输出已保存到 results 目录")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("所有输出已保存到 results 目录")
    print("=" * 80)

if __name__ == "__main__":
    main()
