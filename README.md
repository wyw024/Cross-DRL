# 基于深度强化学习的自动驾驶车辆交叉口通行决策研究

**研究主题：** 基于 Q-learning、DQN 及其改进算法的仿真实验  
**仿真平台：** SUMO 1.28.0  
**Python：** 3.10.x  
**环境接口：** Gymnasium + TraCI

## 1. 项目目标与实验对象

本项目构建一个带固定周期信号灯的四岔路口仿真环境，以单辆受控主车为研究对象，比较四种纵向通行控制策略：

| 方法 | 类型 | 是否需要训练 |
|---|---|---|
| Rule-based | 人工规则控制基线 | 否 |
| Q-learning | 表格型强化学习 | 是 |
| DQN | 深度 Q 网络 | 是 |
| Double DQN | Double Q-learning 深度网络 | 是 |

主车从北侧进入，回合开始前指定直行、左转或右转路线；智能体在回合内选择纵向加速度动作。因此，本项目研究的是**给定路线下的交叉口纵向通行决策**，不是同时学习路线选择和横向转向控制。

## 2. 当前实现

- 单个四向信号灯交叉口，背景车辆随机发车并随机选择路线。
- 三档背景交通需求：`low`、`medium`、`high`，分别对应基准需求的 0.5、1.0、1.5 倍。
- 统一的 Gymnasium 环境接口、七维观测和五个离散纵向动作。
- Rule-based、Q-learning、DQN、Double DQN 四种策略。
- 统一评估协议：逐回合记录、汇总指标、Wilson 置信区间和相同测试种子。
- 离线对比、配对统计检验、结果完整性检查和可视化脚本。

## 3. 环境安装

1. 安装 SUMO 1.28.0，并配置 `SUMO_HOME` 指向 SUMO 安装目录。
2. 创建 Python 3.10 环境。
3. 在项目根目录安装依赖：

```bash
python -m pip install -r requirements.txt
```

4. 检查 SUMO 与 Python 环境：

```bash
python -c "import traci, sumolib, gymnasium, numpy, pandas, torch, matplotlib, scipy; print('dependencies OK')"
```

SUMO 本体需单独安装；`pip install` 不会安装 SUMO GUI。

## 4. 目录结构

```text
Cross-DRL-main/
├── algorithms/                 # Q-learning、DQN、Double DQN 及网络
├── baseline/                   # Rule-based 基线
├── environment/                # Gymnasium 环境与 SUMO 路网
├── training/                   # 三种学习算法的训练入口
├── evaluation/                 # 统一评估、结果检查、统计对比
├── visualization/              # 图表和 SUMO 演示视频
├── models/q_learning/           # Q-table 模型
├── results/
│   ├── rule_based/              # baseline 评估结果
│   ├── q_learning/              # Q-learning 模型、训练和评估记录
│   ├── dqn/                     # DQN checkpoint、训练和评估记录
│   ├── double_dqn/              # Double DQN checkpoint、训练和评估记录
│   ├── comparison/              # 算法对比 CSV 和统计检验
│   └── figures/                 # 自动生成的图表
└── docs/                        # 环境、标定和实验说明
```

## 5. 训练与评估

以下命令均从项目根目录运行。已有模型和结果时，不需要重复训练。

### 5.1 训练

```bash
python -m training.train_q_learning --episodes 300 --density medium
python -m training.train_dqn --algorithm dqn --episodes 300 --density medium
python -m training.train_dqn --algorithm double_dqn --episodes 300 --density medium
```

Rule-based 是非学习型基线，不训练。

### 5.2 统一评估

评估脚本默认使用 50 个回合、测试种子从 1000 开始。不同算法在相同 `density / route / seed` 下生成相同 SUMO 场景。若改变评估参数，四种算法必须使用相同设置重新评估。

```bash
python -m evaluation.evaluate_rule_based --episodes 50 --density all --route straight --seed 1000
python -m evaluation.evaluate_q_learning --episodes 50 --density all --route straight --seed 1000
python -m evaluation.evaluate_dqn --algorithm dqn --episodes 50 --density all --route straight --seed 1000
python -m evaluation.evaluate_dqn --algorithm double_dqn --episodes 50 --density all --route straight --seed 1000
```

评估输出到 `results/<algorithm>/evaluation_episodes.csv` 和 `evaluation_summary.csv`。逐回合文件保留测试 seed、SUMO seed、奖励、结果、安全事件、速度、停车、急刹和动作比例；汇总文件按交通密度和路线统计。

## 6. 结果检查、算法对比与可视化

### 6.1 检查结果完整性

```bash
python -m evaluation.check_results
```

该脚本检查必需字段、重复测试种子、比率范围，以及不同算法是否使用相同测试种子和相同 SUMO 场景种子。

### 6.2 算法对比与统计检验

```bash
python -m evaluation.compare_algorithms
```

可指定算法、密度和路线：

```bash
python -m evaluation.compare_algorithms --algorithms q_learning dqn double_dqn --density low medium --route straight
```

输出：
- `results/comparison/comparison_summary.csv`
- `results/comparison/pairwise_tests.csv`

由于不同算法使用相同测试场景，统计检验按 seed 配对：奖励和通过时间采用双侧 Wilcoxon signed-rank 检验，安全通过结果采用精确配对二项检验（McNemar 检验的条件二项形式）。所有检验的 p 值采用 Holm-Bonferroni 方法校正。通过时间只比较两种算法均成功通过的配对回合。

### 6.3 绘制图表

```bash
python -m visualization.plot_results
```

输出至 `results/figures/`，包括：
- 各密度下平均奖励、通过时间、停车步数、急刹步数；
- 安全通过率、碰撞率、闯红灯率、超时率；
- 逐回合奖励和成功通过时间箱线图；
- 五种动作的使用比例；
- Q-learning、DQN、Double DQN 的训练奖励曲线和 epsilon 曲线（存在训练日志时）。

### 6.4 录制 SUMO 场景演示

```bash
python -m visualization.record_demo --density medium
```

该视频是用于展示红灯等待和恢复信号后的通行过程的**场景演示**，不是训练策略的性能视频。运行需要 SUMO-GUI 和 OpenCV。

## 7. 当前评估数据说明

压缩包内现有评估数据为每种算法 150 回合：low、medium、high 各 50 回合，路线均为 `straight`，测试 seed 从 1000 起。该数据可用于当前直行场景下的四算法对比。

它不能代表 left/right 路线结果，也不能替代多随机训练种子实验。当前四种算法的安全通过率均为 100%，安全指标存在天花板效应，因此还应同时分析效率、停车、急刹和逐回合分布。

## 8. 指标解释与注意事项

- **安全通过率：** 无碰撞、无确认闯红灯、无传送且到达目标出口的回合比例。
- **碰撞率：** 至少发生一次碰撞的回合比例。
- **闯红灯率：** 环境确认红灯违规的回合比例。
- **超时率：** 达到最大步数且未通过的回合比例。
- **平均通过时间：** 成功回合中首次满足 `passed_intersection` 的时间均值。
- **停车步数：** 速度低于 0.1 m/s 的步数。
- **急刹步数：** 实际加速度不大于 -3 m/s² 的步数。
- **最小前车间距：** 作为诊断指标使用；极小值应结合 SUMO 间距定义、车道和车辆几何关系复核，不能单独作为碰撞或安全结论。

评估结果是当前仿真配置和测试种子集合下的结果，不应直接外推为真实道路安全性或普遍算法优劣。

## 9. 文档

- [SUMO 安装与运行](docs/sumo_setup.md)
- [环境状态、动作、奖励与结束条件](docs/environment_interface.md)
- [交通流标定与环境检查](docs/calibration.md)
- [实验协议、指标和分析建议](docs/experiment_protocol.md)
