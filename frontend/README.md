# 离焦剂量贝叶斯分析系统 - 前端

这是离焦剂量-脉络膜反应贝叶斯分析系统的前端可视化界面，用于直观展示主动学习过程和剂量-反应关系。

## 功能特性

### 核心功能
- **患者选择**：从后端 API 获取预计算的患者列表，支持选择不同受试者
- **学习过程可视化**：实时展示推荐剂量的震荡收敛过程
- **学习效果监控**：准确率上升曲线与 MAE 下降曲线的双 Y 轴展示
- **扩展可视化面板**：可切换查看后验分布、残差诊断、观测 vs 预测、推荐剂量分布等多种图表
- **实时状态监控**：当前轮次、准确率、MAE、不确定性、推荐剂量等关键指标
- **进度交互控制**：学习完成后可通过进度条回看任意历史轮次
- **播放速度调节**：支持 1x / 2x / 5x 倍速播放

### 界面组成
- **顶部导航**：系统标题与简介
- **参数面板**：患者信息展示与学习启动控制
- **剂量推荐区**：当前/最终推荐剂量、进度条、速度控制
- **学习效果图表**：准确率与 MAE 趋势
- **学习过程图表**：剂量-反应曲线收敛过程
- **扩展可视化**：多标签页切换的分析图表
- **状态监控面板**：8 项实时指标卡片

## 技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| React | 18+ | UI 框架 |
| TypeScript | 5+ | 类型系统 |
| Vite | 5+ | 构建工具 |
| ECharts | 5.5+ | 图表库 |
| Framer Motion | 11+ | 动画库 |

## 快速开始

### 环境要求
- Node.js 18+
- npm 或 yarn / pnpm

### 安装依赖

```bash
cd frontend
npm install
```

### 开发模式

```bash
npm run dev
```

访问 http://localhost:5173 查看应用。

### 生产构建

```bash
npm run build
```

构建产物输出到 `dist` 目录。

### 预览生产构建

```bash
npm run preview
```

## 项目结构

```
frontend/
├── public/               # 静态资源
├── src/
│   ├── components/       # React 组件
│   │   ├── Header.tsx
│   │   ├── ParameterPanel.tsx
│   │   ├── LearningProcessChart.tsx
│   │   ├── LearningEffectChart.tsx
│   │   ├── ExtendedVisualization.tsx
│   │   ├── DoseRecommendation.tsx
│   │   └── ...
│   ├── hooks/            # 自定义 Hooks
│   │   └── useSimulation.ts
│   ├── engine/           # 模拟引擎
│   ├── types/            # TypeScript 类型定义
│   │   └── index.ts
│   ├── App.tsx           # 主应用组件
│   ├── main.tsx          # 入口文件
│   └── index.css         # 全局样式
├── index.html
├── package.json
├── tsconfig.json
├── tsconfig.node.json
└── vite.config.ts
```

## 后端接口

前端默认从 `http://localhost:8000` 获取数据，主要接口：

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/patients` | GET | 获取患者列表 |
| `/api/patients/{id}` | GET | 获取单个患者详情 |
| `/api/learning/run/{patient_id}` | POST | 启动学习会话 |

如需修改后端地址，可在 `src/hooks/useSimulation.ts` 和 `src/App.tsx` 中调整 API 基础地址。

## 使用说明

### 基本流程

1. 确保后端 API 服务已启动（详见根目录 README）
2. 启动前端开发服务器
3. 在参数面板的患者下拉列表中选择一位患者
4. 点击「开始学习」按钮
5. 观察学习过程中剂量-反应曲线的收敛过程
6. 学习完成后，可拖动进度条回看任意历史轮次
7. 在扩展可视化区域切换不同图表查看分析结果

### 速度控制

- 点击 1x / 2x / 5x 按钮调节播放速度
- 学习过程中可随时暂停和继续

## 部署说明

### 静态部署

构建后的 `dist` 目录可直接部署到任意静态文件服务器：

- Nginx / Apache
- GitHub Pages
- Vercel / Netlify
- 或其他静态托管服务

### 本地预览（无后端）

前端也支持纯前端模式运行（使用内置的模拟数据），具体取决于 `useSimulation` hook 的实现方式。

## 相关文档

- 项目主 README：[../README.md](../README.md)
- 后端 API 文档：http://localhost:8000/docs

## 许可证

MIT License
