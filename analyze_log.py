"""
日志分析脚本 - 读取并展示run.log中的关键信息

运行方式:
    python analyze_log.py
"""

import os
import re
from datetime import datetime


def analyze_log():
    """分析run.log文件并展示关键信息"""
    log_file = os.path.join("results", "run.log")
    
    if not os.path.exists(log_file):
        print("\n❌ 错误: 未找到run.log文件")
        print("请先运行demo.py或其他测试脚本生成日志文件")
        return
    
    print("\n" + "=" * 60)
    print("📊 日志分析结果")
    print("=" * 60)
    
    # 读取日志文件
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"\n❌ 读取日志文件失败: {e}")
        return
    
    # 提取关键信息
    execution_time = None
    true_threshold = None
    rounds = []
    final_result = None
    errors = []
    warnings = []
    
    # 解析日志
    for line in lines:
        line = line.strip()
        
        # 提取执行时间
        if "执行时间:" in line:
            execution_time = line.split("执行时间:")[1].strip()
        
        # 提取真实阈值
        if "真实阈值 (ED50):" in line:
            try:
                true_threshold = float(line.split("真实阈值 (ED50):")[1].split("D")[0].strip())
            except ValueError:
                pass
        
        # 提取轮次信息
        round_match = re.match(r'.*第 (\d+) 轮: 剂量 (\d+\.\d+) D, 反应 (\d+\.\d+) μm.*', line)
        if round_match:
            try:
                round_num = int(round_match.group(1))
                dose = float(round_match.group(2))
                response = float(round_match.group(3))
                rounds.append({
                    'round': round_num,
                    'dose': dose,
                    'response': response
                })
            except (ValueError, IndexError):
                pass
        
        # 提取最终结果
        if "最终结果" in line and "=" in line:
            final_result = {}
        elif final_result is not None and "估计阈值:" in line:
            final_result['估计阈值'] = line.split("估计阈值:")[1].strip()
        elif final_result is not None and "真实阈值:" in line:
            final_result['真实阈值'] = line.split("真实阈值:")[1].strip()
        elif final_result is not None and "估计误差:" in line:
            final_result['估计误差'] = line.split("估计误差:")[1].strip()
        elif final_result is not None and "最佳剂量:" in line:
            final_result['最佳剂量'] = line.split("最佳剂量:")[1].strip()
        elif final_result is not None and "总测量次数:" in line:
            final_result['总测量次数'] = line.split("总测量次数:")[1].strip()
        elif final_result is not None and "MCMC 收敛:" in line:
            final_result['MCMC 收敛'] = line.split("MCMC 收敛:")[1].strip()
        
        # 提取错误和警告信息
        if "ERROR" in line:
            errors.append(line)
        elif "WARNING" in line:
            warnings.append(line)
    
    # 展示关键信息
    print("\n🔍 基本信息:")
    print(f"执行时间: {execution_time if execution_time else '未找到'}")
    print(f"真实阈值: {true_threshold if true_threshold else '未找到'} D")
    
    print("\n📈 轮次数据:")
    if rounds:
        print("轮次 | 测试剂量 (D) | 反应 (μm)")
        print("-" * 40)
        for r in rounds:
            print(f"{r['round']:2d}  | {r['dose']:12.2f} | {r['response']:8.1f}")
    else:
        print("未找到轮次数据")
    
    print("\n🎯 最终结果:")
    if final_result:
        for key, value in final_result.items():
            print(f"{key}: {value}")
    else:
        print("未找到最终结果")
    
    # 展示警告信息
    if warnings:
        print(f"\n⚠️  警告信息 ({len(warnings)} 个):")
        for warning in warnings[:5]:  # 只显示前5个警告
            print(f"  - {warning}")
        if len(warnings) > 5:
            print(f"  ... 还有 {len(warnings) - 5} 个警告未显示")
    
    # 展示错误信息
    print("\n❌ 错误信息:")
    if errors:
        print(f"发现 {len(errors)} 个错误:")
        for error in errors:
            print(f"  - {error}")
        
        # 尝试修复错误
        print("\n💡 错误修复建议:")
        for error in errors:
            if "模型拟合失败" in error:
                print("  - 模型拟合失败: 可能是数据不足或参数设置问题，建议增加测量次数或调整模型参数")
            elif "MCMC 收敛" in error:
                print("  - MCMC 收敛问题: 建议增加MCMC链数或调优步数")
            elif "采样失败" in error:
                print("  - 采样失败: 可能是参数设置不当，建议调整先验分布或增加调优步数")
            else:
                print(f"  - 其他错误: {error}")
    else:
        print("未发现错误")
    
    print("\n" + "=" * 60)
    print("✅ 日志分析完成")
    print("=" * 60)


if __name__ == "__main__":
    analyze_log()
