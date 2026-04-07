# 贝叶斯Sigmoid焦点优化模型

## 项目概述

本项目实现了一个基于贝叶斯主动学习的Sigmoid模型，用于优化焦点估计。通过智能的剂量推荐和后验分布估计，该模型能够以最少的测量次数获得准确的焦点阈值估计。

## 核心功能

### 1. 贝叶斯主动学习
- 使用马尔可夫链蒙特卡洛(MCMC)方法进行参数估计
- 基于预期误差减少(EER)的智能剂量推荐
- 动态调整的停止条件，确保估计精度
- 双点启动策略（2.0D, 4.0D），确保覆盖合理的剂量范围

### 2. 特征处理
- 支持多特征输入，包括屈光不正、眼轴长、年龄、性别、脉络膜厚度、血管指数等
- 自动特征标准化处理，确保模型稳定性和准确性
- 特征权重参数学习，使模型参数先验依赖于输入特征

### 3. 高级可视化
- **PDF报告**：自动生成个体诊断报告，包含：
  - 剂量-反应曲线
  - 信心秒表
  - 临床解释
  - AI思考过程
  - 参数关联图
  - 受试者画像
  - 总结报告

### 4. 精度优化
- 支持可配置的最大测量次数（默认5次）
- 高精度MCMC采样（5000步调优，4000步采样）
- 智能的剂量推荐算法，优先选择信息增益最大的点
- 动态调整的不确定性阈值，兼顾精度与测量次数

## 快速开始

### 环境要求
- Python 3.7+
- 依赖包：详见 requirements.txt

### 安装依赖
```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\Activate.ps1
# Linux/Mac
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 运行示例
```bash
python demo.py
```

运行后，结果将保存到 `results/` 目录，包括：
- `run.log` - 运行日志
- `visualization/` - 基本可视化图表
- `exhibition/` - 高级可视化图表

## 项目结构

```
├── defocus_bayesian/         # 核心模块
│   ├── __init__.py
│   ├── model.py              # Sigmoid模型实现
│   ├── acquisition.py        # 采集函数实现
│   ├── learner.py            # 主动学习器
│   ├── feature_processor.py  # 特征处理模块
│   ├── plot.py               # 可视化工具
│   ├── basic_plots.py        # 基础图表
│   ├── learning_plots.py     # 学习过程图表
│   ├── evolution_plots.py    # 后验演变图表
│   ├── exhibition.py         # 高级可视化
│   ├── simulate.py           # 受试者模拟器
│   ├── config.py             # 配置模块
├── demo.py                   # 演示脚本
├── README.md                 # 项目说明
├── REPORT.md                 # 报告解读指南
├── config.yaml               # 配置文件
├── requirements.txt          # 依赖包文件
└── .gitignore                # Git忽略文件
```

## 关键参数说明

### SigmoidActiveLearner 参数
- `max_measurements`：最大测量次数（默认5次）
- `uncertainty_threshold`：不确定性阈值（默认2.0）
- `dose_range`：剂量范围（默认(0.5, 6.0)）
- `n_grid`：搜索网格点数（默认100）
- `mcmc_chains`：MCMC链数（默认4）
- `mcmc_tune`：MCMC调优步数（默认2000）
- `mcmc_draws`：MCMC后验样本数（默认1000）
- `random_seed`：随机种子（默认None）

### SigmoidModel 参数
- `n_chains`：MCMC链数（默认4）
- `tune`：调优步数（默认5000）
- `draws`：后验样本数（默认4000）
- `target_accept`：目标接受率（默认0.92）
- `random_seed`：随机种子（默认None）

## 结果解释

运行后，控制台会显示详细的结果分析，包括：
- 估计的阈值 (ED50)
- 阈值的标准差
- 推荐的最佳剂量
- 总测量次数
- 停止原因
- 模型收敛状态

## 自定义数据

### 从CSV文件加载数据
```python
from defocus_bayesian.learner import SigmoidActiveLearner

learner = SigmoidActiveLearner()
learner.load_from_csv(
    'your_data.csv',
    dose_col='dose',
    response_col='response',
    feature_cols=['屈光不正', '眼轴长', '年龄', '性别', '脉络膜厚度', '血管指数']  # 可选
)
```

### 使用特征数据
```python
from defocus_bayesian.learner import SigmoidActiveLearner
import numpy as np

learner = SigmoidActiveLearner()

# 特征数据：屈光不正、眼轴长、年龄、性别、脉络膜厚度、血管指数
features = np.array([[-3.0, 25.0, 18, 0, 280.0, 0.75]])  # 性别：0=男, 1=女
learner.set_features(features, ['屈光不正', '眼轴长', '年龄', '性别', '脉络膜厚度', '血管指数'])
```

## 注意事项

1. **数据质量**：确保输入的剂量和反应数据是准确的，避免异常值
2. **特征数据**：特征数据应包含屈光不正、眼轴长、年龄、性别、脉络膜厚度、血管指数等相关指标
3. **计算资源**：MCMC采样可能需要较长时间，特别是当使用较多的链和样本时
4. **收敛性**：系统会自动检查模型的收敛性，如果收敛性不佳，可能需要增加采样次数
5. **文件路径**：确保CSV文件路径正确，并且文件格式符合要求
6. **环境配置**：建议使用虚拟环境安装依赖，避免版本冲突

## 许可证

本项目采用 MIT 许可证。