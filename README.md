# 离焦剂量-脉络膜反应贝叶斯分析系统

## 项目简介

本项目利用贝叶斯推断和主动学习策略，分析离焦剂量（D）与脉络膜反应（RDV15）之间的剂量-反应关系。系统基于四参数 Sigmoid 模型，通过 PyMC 实现贝叶斯统计推断，能够为每位受试者个性化推荐最佳离焦剂量，并提供完整的不确定性量化。

项目同时包含基于 React 的前端可视化界面，用于直观展示主动学习过程、剂量-反应曲线收敛过程以及预测结果的不确定性分布。

## 核心功能

### 贝叶斯建模
- **四参数 Sigmoid 模型**：使用 PyMC 构建，准确描述离焦剂量与脉络膜反应之间的非线性剂量-反应关系
- **特征整合**：支持整合患者多维临床特征（年龄、性别、眼轴长度、脉络膜厚度、CVI 等），实现个性化建模
- **不确定性量化**：提供后验预测分布和 94% HDI（最高密度区间），完整量化预测不确定性
- **稳健估计**：支持 Student-T 似然，降低异常值对模型拟合的影响

### 主动学习
- **智能剂量推荐**：通过 SigmoidActiveLearner 实现，最小化达到目标精度所需的测量次数
- **自适应采样**：在当前模型不确定度最大的位置推荐下一个测量剂量点
- **灵活停止条件**：支持不确定性阈值和剂量收敛双重停止条件

### 可视化展示
- **剂量-反应曲线**：后验均值曲线与 94% HDI 可信区间
- **特征重要性分析**：基于后验权重的临床特征重要性排序
- **残差诊断**：模型拟合质量评估
- **推荐剂量分布**：受试者群体最佳剂量分布直方图

### 前端交互
- **患者选择**：从预计算数据集中选择不同受试者
- **学习过程可视化**：实时观察推荐剂量震荡收敛过程
- **多维度图表**：准确率变化、MAE 趋势、后验分布、残差诊断等
- **进度控制**：可调节播放速度，支持暂停/继续/进度条拖动

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (React)                     │
│  ┌──────────┐  ┌────────────┐  ┌────────────────────────┐  │
│  │ 参数面板  │  │ 学习过程图  │  │   扩展可视化（8种图表）  │  │
│  └──────────┘  └────────────┘  └────────────────────────┘  │
└──────────────────────────────────┬──────────────────────────┘
                                   │ HTTP / REST API
┌──────────────────────────────────▼──────────────────────────┐
│                    FastAPI Backend                           │
│  ┌─────────────┐  ┌───────────┐  ┌──────────────────────┐  │
│  │  /patients  │  │ /learning │  │  Precomputed Cache   │  │
│  └─────────────┘  └───────────┘  └──────────────────────┘  │
└──────────────────────────────────┬──────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────┐
│                  Core Analysis Engine                       │
│  ┌─────────────┐  ┌───────────┐  ┌──────────────────────┐  │
│  │ SigmoidModel│  │  Learner  │  │  FeatureProcessor    │  │
│  │  (PyMC)     │  │ (Active)  │  │  (scikit-learn)      │  │
│  └─────────────┘  └───────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 快速开始

### 环境要求
- Python 3.10+
- Node.js 18+（前端开发）
- 建议 8GB 以上内存（贝叶斯采样需要）

### 后端安装与运行

1. **克隆仓库**
   ```bash
   git clone <repository-url>
   cd Medical
   ```

2. **创建虚拟环境并安装依赖**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate

   pip install -r requirements.txt
   ```

3. **生成测试数据（可选，首次使用）**
   ```bash
   python -m src.data_generator
   ```

4. **运行贝叶斯分析**
   ```bash
   python -m src.main
   ```
   分析结果将保存在 `data/results/` 和 `data/figures/` 目录下。

5. **启动 API 服务**
   ```bash
   python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
   ```
   API 文档：http://localhost:8000/docs

### 前端安装与运行

```bash
cd frontend
npm install
npm run dev
```

访问 http://localhost:5173 查看前端界面。

### Windows 一键启动

项目提供了 `start.bat` 脚本，可选择不同的启动模式：

```bash
start.bat
```

选项包括：生成数据、运行分析、启动后端、启动前端、同时启动、运行测试等。

## 项目结构

```
Medical/
├── data/
│   ├── raw_data.csv              # 原始数据
│   ├── results/                  # 模型输出结果
│   │   ├── cleaned_data.csv      # 清洗后数据
│   │   ├── posterior.nc          # 后验分布（NetCDF）
│   │   ├── fit_metrics.json      # 拟合指标
│   │   ├── feature_priors.json   # 特征先验配置
│   │   ├── learning_sessions.json  # 预计算学习会话
│   │   └── ...
│   └── figures/                  # 可视化图表输出
│       ├── dose_response_curve.png
│       ├── feature_importance.png
│       ├── residual_diagnostics.png
│       └── ...
├── frontend/                     # 前端项目（React + TypeScript）
│   ├── src/
│   │   ├── components/           # React 组件
│   │   ├── hooks/                # 自定义 Hooks
│   │   ├── engine/               # 模拟引擎
│   │   ├── types/                # TypeScript 类型
│   │   └── App.tsx
│   ├── package.json
│   └── README.md                 # 前端独立文档
├── src/                          # Python 源码
│   ├── api/
│   │   └── app.py                # FastAPI 服务
│   ├── defocus_bayesian/
│   │   ├── model.py              # SigmoidModel 贝叶斯模型
│   │   ├── learner.py            # SigmoidActiveLearner 主动学习
│   │   ├── feature_processor.py  # 特征处理器
│   │   └── __init__.py
│   ├── utils/
│   │   └── data_processor.py     # 数据清洗与处理
│   ├── visualization/
│   │   └── plots.py              # 可视化绘图函数
│   ├── config.py                 # 全局配置
│   ├── main.py                   # 主分析入口
│   └── data_generator.py         # 测试数据生成器
├── tests/                        # 单元测试
│   ├── test_data_processor.py
│   └── test_feature_processor.py
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
├── requirements.txt
└── start.bat                     # Windows 启动脚本
```

## 算法原理

### 四参数 Sigmoid 模型

剂量-反应关系采用经典的四参数 Sigmoid 函数描述：

```
RDV15(D) = baseline + (max_response - baseline) / (1 + exp(-slope * (D - threshold)))
```

| 参数 | 生理含义 |
|------|----------|
| baseline | 基线反应值（无刺激时的脉络膜反应） |
| max_response | 最大反应值（饱和剂量下的反应） |
| slope | 曲线陡峭程度（剂量敏感度） |
| threshold | 半最大效应剂量（ED50） |

### 贝叶斯先验设计

系统根据特征的临床重要性分配不同的先验权重：

- **核心特征**（先验尺度 0.30）：黄斑中心凹下脉络膜厚度
- **重要特征**（先验尺度 0.20）：脉络膜血管指数 (CVI)、眼轴长度
- **辅助特征**（先验尺度 0.08）：年龄、性别、验光离焦量等

所有特征在建模前进行 Z-score 标准化处理。

### 主动学习策略

主动学习通过迭代选择最有信息量的剂量点进行测量，以最少的测量次数达到目标精度：

1. 基于当前数据拟合贝叶斯模型
2. 在候选剂量网格上评估预测不确定性
3. 选择不确定性最大（或预期效用最高）的剂量点推荐测量
4. 将新测量结果加入数据集，重新拟合模型
5. 重复上述步骤直到满足停止条件

**停止条件**（满足任一即可）：
- 不确定性达标：后验标准差低于预设阈值
- 剂量收敛：连续两轮推荐剂量变化小于阈值
- 达到最大迭代轮次

## 数据格式

原始数据 CSV 文件应包含以下关键字段：

| 字段名 | 描述 | 类型 |
|--------|------|------|
| 被测者编号 | 受试者唯一标识 | string |
| 检测批次 | 随访批次（第一次/第二次） | string |
| 眼别 | 左眼/右眼 | string |
| 离焦剂量 (D) | 离焦量 | float |
| RDV15(D) | 脉络膜反应指标 | float |
| 脉络膜血管指数 (CVI) | 临床特征 | float |
| 黄斑中心凹下脉络膜厚度 (μm) | 临床特征 | float |
| 年龄 (岁) | 受试者年龄 | int |
| 性别 | 男/女 | string 或 int |
| 眼轴长度 (mm) | 临床特征 | float |
| 验光 - 离焦量 (近视度数，D) | 临床特征 | float |

## API 接口

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/patients` | GET | 获取所有患者列表 |
| `/api/patients/{patient_id}` | GET | 获取单个患者详情 |
| `/api/learning/run/{patient_id}` | POST | 启动学习会话（返回预计算数据） |
| `/api/learning/{session_id}` | GET | 获取学习会话状态 |
| `/api/learning/precache` | GET | 获取预计算缓存状态 |
| `/docs` | GET | Swagger API 文档 |

## 临床应用场景

- **近视防控**：离焦镜片的个性化参数配适
- **角膜塑形镜**：OK 镜参数优化与效果预测
- **临床研究**：剂量-反应关系的统计建模
- **个体化医疗**：基于患者特征的精准剂量推荐

## 注意事项

1. 数据中存在缺失值时，系统会删除含有缺失值的样本行（严格模式）
2. 所有数值特征在建模前进行 Z-score 标准化
3. 主动学习过程中的模拟测量基于噪声模型，临床使用时需替换为真实测量数据
4. 剂量推荐范围默认为 3.5-5.0 D（临床舒适区间），可在 `src/config.py` 中调整
5. 贝叶斯采样可能需要较长时间（5-15 分钟），取决于样本量和采样参数
6. 前端展示使用预计算数据，目的是演示学习过程，不保证高精度

## 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。
