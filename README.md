# 个性化离焦剂量探索系统（贝叶斯 Sigmoid + 主动学习）

基于贝叶斯 Sigmoid（四参数逻辑斯蒂）模型和主动学习策略，为每个受试者动态推荐下一个离焦测量剂量。系统以最少实验次数（通常 2~5 次）估计个体离焦阈值（ED50）和最佳剂量。

## 核心特点

- **双点启动**（2.0D, 4.0D）避免单点方向误判
- **贝叶斯 MCMC 推断**（PyMC）量化不确定性
- **主动学习采集函数**：后验方差（探索） + 期望改进（开发）
- **自动停止条件**
- **无需 Jupyter**，纯 Python 脚本运行

## 安装

### 环境要求

- Python >= 3.9
- PyMC >= 5.10

### 使用虚拟环境安装（推荐）

#### 1. 创建虚拟环境

```bash
# 进入项目目录
cd BayesSigmoid-Defocus

# 创建虚拟环境（命名为 venv，你也可以使用其他名称）
python -m venv venv
```

#### 2. 激活虚拟环境

**Windows:**
```powershell
# PowerShell
venv\Scripts\Activate.ps1

# CMD
venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

激活成功后，命令行提示符前会显示 `(venv)`。

#### 3. 升级 pip

在虚拟环境中，首先升级 pip 到最新版本：

```bash
python -m pip install --upgrade pip
```

#### 4. 安装依赖包

```bash
pip install -r requirements.txt
```

或者手动安装最新版本：

```bash
# 安装 PyMC 及其依赖（最新版本）
pip install pymc

# 安装其他依赖
pip install numpy scipy matplotlib arviz pandas
```

#### 5. 验证安装

```bash
python -c "import pymc; print(f'PyMC version: {pymc.__version__}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
```

#### 6. 退出虚拟环境

```bash
deactivate
```

### 完整安装流程示例

```bash
# 1. 进入项目目录
cd e:\Projects_of_Liang\SingleProject\Python_Liang\BayesSigmoid-Defocus

# 2. 创建虚拟环境
python -m venv venv

# 3. 激活虚拟环境
venv\Scripts\activate

# 4. 升级 pip
python -m pip install --upgrade pip

# 5. 安装所有依赖
pip install -r requirements.txt

# 6. 运行演示
python demo.py

# 7. 退出虚拟环境
deactivate
```

### 常见问题

**Q: 为什么使用虚拟环境？**
A: 虚拟环境可以隔离项目依赖，避免不同项目之间的包版本冲突。

**Q: 如何删除虚拟环境？**
A: 直接删除 `venv` 文件夹即可：
```bash
deactivate  # 先退出虚拟环境
rmdir /s venv  # Windows
rm -rf venv    # macOS/Linux
```

**Q: 如何重新创建虚拟环境？**
A: 删除旧环境后，按照上述步骤重新创建即可。

**Q: PyMC 安装失败怎么办？**
A: PyMC 需要 C++ 编译器，可以尝试：
```bash
# 安装编译工具（Windows）
pip install msvc-runtime

# 或使用 conda（如果已安装 Anaconda）
conda install -c conda-forge pymc
```

## 快速开始

### 1. 运行演示脚本

```bash
python demo.py
```

演示脚本将：
1. 生成一个虚拟受试者
2. 执行主动学习流程
3. 输出每一步的推荐剂量和观测反应
4. 生成可视化图表

### 2. 运行批量模拟测试

```bash
# 默认运行 100 个受试者
python test_simulation.py

# 指定受试者数量
python test_simulation.py --n_subjects 50

# 对比不同策略
python test_simulation.py --compare

# 指定输出目录
python test_simulation.py --output_dir my_results
```

## 项目结构

```
defocus_bayesian/
├── __init__.py          # 包初始化
├── model.py             # SigmoidModel 类（PyMC 建模、拟合、预测）
├── acquisition.py       # 采集函数（后验方差、期望改进）
├── learner.py           # SigmoidActiveLearner 主类
├── simulate.py          # 模拟数据生成器
└── plot.py              # 可视化模块

demo.py                  # 演示脚本
test_simulation.py       # 批量模拟测试脚本
requirements.txt         # 依赖列表
README.md                # 本文件
```

## 使用示例

### 基本用法

```python
from defocus_bayesian import SubjectSimulator, SigmoidActiveLearner

# 创建模拟受试者
simulator = SubjectSimulator(random_seed=42)
simulator.generate_subject()

# 创建主动学习器
learner = SigmoidActiveLearner(
    max_measurements=5,
    uncertainty_threshold=2.0,
)

# 执行主动学习
result = learner.learn(simulator.get_response_function())

print(f"估计阈值: {result.threshold_estimate:.2f} ± {result.threshold_std:.2f} D")
print(f"最佳剂量: {result.best_dose:.2f} D")
print(f"测量次数: {result.n_measurements}")
```

### 分步执行

```python
from defocus_bayesian import SigmoidActiveLearner

learner = SigmoidActiveLearner()

# 第1轮：双点启动
next_dose, strategy = learner.recommend_next_dose()  # 返回 2.0 D
response = measure(next_dose)  # 你的测量函数
learner.add_measurement(next_dose, response)

# 第2轮：双点启动
next_dose, strategy = learner.recommend_next_dose()  # 返回 4.0 D
response = measure(next_dose)
learner.add_measurement(next_dose, response)

# 第3轮起：主动学习
next_dose, strategy = learner.recommend_next_dose()  # 基于采集函数
response = measure(next_dose)
learner.add_measurement(next_dose, response)

# 检查是否停止
should_stop, stop_reason = learner.should_stop()
```

### 从 CSV 加载历史数据

```python
learner = SigmoidActiveLearner()
learner.load_from_csv("data/subject1.csv", dose_col="dose", response_col="response")

# 继续主动学习
next_dose, strategy = learner.recommend_next_dose()
```

### 可视化

```python
from defocus_bayesian.plot import (
    plot_dose_response_curve,
    plot_acquisition_function,
    plot_all_posteriors,
    save_figure,
)

# 绘制剂量-反应曲线
fig = plot_dose_response_curve(
    learner.model,
    simulator=simulator,
    measurements=result.measurements,
)
save_figure(fig, "dose_response.png")

# 绘制采集函数
fig = plot_acquisition_function(
    learner.model,
    learner.acquisition,
    strategy="ei",
    y_best=max(result.measurements, key=lambda x: x[1])[1],
)
save_figure(fig, "acquisition.png")

# 绘制参数后验分布
fig = plot_all_posteriors(learner.model)
save_figure(fig, "posteriors.png")
```

## 模型与算法

### 剂量-反应模型（Sigmoid）

```
y = baseline + (max_response - baseline) / (1 + exp(-slope * (x - threshold))) + ε
ε ~ N(0, sigma^2)
```

### 参数先验

| 参数 | 先验分布 | 约束 |
|------|----------|------|
| baseline | Normal(0, 5) | [0, 10] |
| max_response | Normal(20, 10) | (0, 50] |
| slope | LogNormal(ln(3), 0.5) | 无 |
| threshold | Normal(2.5, 1.0) | [0.5, 6.0] |
| sigma | HalfNormal(5) | 无 |

### 贝叶斯推断

- **库**: PyMC (≥5.10)
- **MCMC 设置**:
  - 链数: 4
  - 调优步数: 2000
  - 后验样本数: 1000
  - 目标接受率: 0.85
  - 收敛诊断: R̂ < 1.05, ESS > 200

### 主动学习策略

**双点启动**（无历史数据时）：
- 第1轮: 2.0 D
- 第2轮: 4.0 D

**采集函数**（从第3轮开始）：
- 第3轮: 后验方差 `acq(x) = Var[y(x) | data]`
- 第4轮起: 期望改进 `acq_EI(x) = E[max(0, y(x) - y_best)]`

搜索空间: 0.5 D 到 6.0 D，离散网格 100 点

### 停止条件

满足任一即停止：
1. 最佳剂量处的后验标准差 < 2.0 μm
2. 已测次数 ≥ 5
3. 最高剂量 ≥ 6.0 D 且最大反应 < 10 μm（低反应者）

## 配置参数

### SigmoidActiveLearner 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| max_measurements | int | 5 | 最大测量次数 |
| uncertainty_threshold | float | 2.0 | 不确定性阈值 (μm) |
| dose_range | tuple | (0.5, 6.0) | 剂量范围 (D) |
| n_grid | int | 100 | 搜索网格点数 |
| mcmc_chains | int | 4 | MCMC 链数 |
| mcmc_tune | int | 2000 | MCMC 调优步数 |
| mcmc_draws | int | 1000 | MCMC 后验样本数 |
| random_seed | int | None | 随机种子 |

## 示例输出

### demo.py 输出示例

```
============================================================
个性化离焦剂量探索系统 - 演示
============================================================

生成虚拟受试者:
  真实阈值 (ED50): 2.87 D
  基线反应: 1.23 μm
  最大反应: 18.5 μm
  斜率: 2.45
  噪声标准差: 3.21 μm

============================================================
开始主动学习流程
============================================================
最大测量次数: 5
不确定性阈值: 2.0 μm

第 1 轮
  推荐剂量: 2.00 D (双点启动)
  观测反应: 3.2 μm

第 2 轮
  推荐剂量: 4.00 D (双点启动)
  观测反应: 14.5 μm
  当前阈值估计: 2.95 ± 0.45 D

第 3 轮
  推荐剂量: 3.30 D (后验方差)
  观测反应: 18.1 μm
  当前阈值估计: 2.85 ± 0.30 D

第 4 轮
  推荐剂量: 2.90 D (期望改进)
  观测反应: 17.8 μm

停止: 最佳剂量处后验标准差=1.8 μm < 2.0 μm

============================================================
最终结果
============================================================
估计阈值: 2.85 ± 0.25 D
真实阈值: 2.87 D
估计误差: 0.02 D
最佳剂量: 2.85 D
总测量次数: 4
MCMC 收敛: 是
```

### test_simulation.py 输出示例

```
======================================================
批量模拟测试
======================================================
受试者数量: 100
输出目录: results

开始模拟 100 个受试者...
======================================================
已完成: 10/100
已完成: 20/100
...
已完成: 100/100
======================================================

模拟结果统计:
----------------------------------------
受试者数量: 100
平均测量次数: 3.45 ± 0.78
中位数测量次数: 3.0
测量次数范围: [2, 5]
阈值 RMSE: 0.32 D
阈值 MAE: 0.28 D
收敛率 (MCMC): 87.0%
----------------------------------------
```

## 依赖列表

- pymc>=5.10
- numpy>=1.24
- scipy>=1.10
- matplotlib>=3.5
- arviz>=0.17
- pandas>=2.0
- typing-extensions>=4.0

## 许可证

MIT License

## 作者

个性化离焦剂量探索系统开发团队
