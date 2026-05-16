# 短波大规模共址天线台站多目标优化系统

## 项目简介

本项目实现了一个完整的短波（HF, 2-30 MHz）大规模共址天线台站多目标优化与数字孪生系统。系统针对包含50+副天线的大型短波台站，解决了天线布局优化、共址干扰抑制、动态频率分配等核心问题。

## 核心功能

### 1. 多目标优化
- **天线布局优化**：基于NSGA-II/III、MOEA/D等演化算法
- **代理辅助优化**：MPS多问题代理模型、K-RVEA算法
- **混合整数非线性规划**：处理离散-连续混合变量

### 2. 强化学习调度
- **MODDQN**：多目标双重深度Q网络
- **Attention-MO-PPO**：基于注意力机制的多智能体近端策略优化
- **动态频率分配**：实时响应电离层状态变化

### 3. 电磁仿真
- **隔离度计算**：基于天线方向图的空间耦合分析
- **传播预测**：ITU-R P.533天波、P.368地波传播模型
- **干扰评估**：三阶/五阶互调产物计算

### 4. 数字孪生
- **五层架构**：物理环境、设备模型、电磁态势、信道预测、智能决策
- **3D可视化**：电磁热力图、方向图波瓣、干涉云渲染

## 项目结构

```
src/sw_station/
├── models/          # 数据模型层
├── simulation/      # 电磁仿真引擎
├── optimization/    # 多目标优化算法
├── rl/              # 强化学习模块
└── digital_twin/    # 数字孪生平台
```

## 安装

```bash
# 克隆项目
git clone <repository-url>
cd MOO

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 开发模式安装
pip install -e ".[dev]"
```

## 快速开始

```python
from sw_station.models import AntennaPatternCube, StationDigitalTwin
from sw_station.simulation import EMSimulator
from sw_station.optimization import ShortwaveStationProblem

# 创建台站模型
station = StationDigitalTwin(n_antennas=50)

# 运行优化
problem = ShortwaveStationProblem(station)
# ... 详见 examples/
```

## 运行示例

```bash
# 运行优化示例
python examples/run_optimization.py

# 运行强化学习训练
python examples/run_rl_training.py

# 运行可视化演示
python examples/visualize_station.py
```

## 测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行带覆盖率的测试
pytest tests/ -v --cov=src/ --cov-report=html
```

## 技术文档

详细技术文档请参考 [Research_Report.md](Research_Report.md)

## 许可证

MIT License
