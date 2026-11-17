# 基于深度强化学习的自动驾驶车辆交叉口通行决策研究

**研究主题：** 基于 Q-learning、DQN 及其改进算法的仿真实验  
**仿真平台：** SUMO 1.28.0　**Python：** 3.10.x　**环境接口：** Gymnasium + TraCI  
**版本：** v2（评估/绘图/训练流程优化版，修改明细见 [CHANGELOG_v2.md](CHANGELOG_v2.md)）

## 1. 项目目标与实验对象

本项目构建一个带固定周期信号灯的四岔路口仿真环境，以单辆受控主车为研究对象，比较四种纵向通行控制策略：

| 方法 | 类型 | 是否需要训练 |
|---|---|---|
| Rule-based | 人工规则控制基线 | 否 |
| Q-learning | 表格型强化学习 | 是 |
| DQN | 深度 Q 网络 | 是 |
| Double DQN | Double Q-learning 深度网络 | 是 |

主车从北侧进入，回合开始前指定**直行、左转或右转**路线；智能体在回合内选择纵向加速度动作。因此本项目研究的是**给定路线下的交叉口纵向通行决策**，不是同时学习路线选择和横向转向控制。

## 2. 当前实现

- 单个四向信号灯交叉口，背景车辆随机发车并随机选择路线。
- 四档背景交通需求：`low`（0.5 倍）、`medium`（1.0）、`high`（1.5）、`extreme`（2.0，压力测试，默认不包含）。
- 统一 Gymnasium 接口、七维观测、五个离散纵向动作。
- **训练**：支持混合密度、混合路线（直行/左转/右转）、多训练种子、自动 ε 衰减。
- **评估**：统一协议（相同测试种子、相同指标）；结果按 (density, route) **合并保存**，不再互相覆盖。
- **对比与绘图**：配对统计检验（含“实际意义”判定）、多训练种子均值±标准差、直行/左转/右转并排对比。

## 3. 环境安装

1. 安装 SUMO 1.28.0，并配置 `SUMO_HOME` 指向 SUMO 安装目录（Windows PowerShell 示例）：

```powershell
$env:SUMO_HOME = "C:\Program Files (x86)\SUMO"
```

2. 创建 Python 3.10 环境，在项目根目录安装依赖：

```bash
python -m pip install -r requirements.txt
```

3. 检查依赖：

```bash
python -c "import traci, sumolib, gymnasium, numpy, pandas, torch, matplotlib, scipy; print('dependencies OK')"
```

`pip install` 不会安装 SUMO 本体；SUMO 需单独安装。**以下命令均在项目根目录运行。**

## 4. 目录结构

```text
Cross-DRL-main/
├── algorithms/        # Q-learning、DQN、Double DQN 及网络
├── baseline/          # Rule-based 基线
├── environment/       # Gymnasium 环境与 SUMO 路网
├── training/          # 训练入口（common.py 为共用工具）
├── evaluation/        # 统一评估、结果检查、统计对比、多种子聚合
├── visualization/     # 图表和 SUMO 演示视频
├── run_experiments.py # 一键流程：训练→评估→检查→对比→绘图
├── models/q_learning/seed_<S>/          # Q-table（每个训练种子一个目录）
├── results/
│   ├── rule_based/                      # baseline 评估结果
│   ├── <q_learning|dqn|double_dqn>/seed_<S>/   # 每个训练种子：训练日志、模型、评估结果
│   ├── comparison/                      # 对比表、统计检验、多种子汇总
│   └── figures/                         # 自动生成的图表
└── docs/              # 环境、标定和实验说明
```

## 5. 快速开始

### 5.1 先跑一遍冒烟测试（确认整个流程可用）

```bash
python run_experiments.py --quick
```

`--quick` 使用 2 个训练种子、每次训练 30 回合、每个场景评估 5 回合，只用于确认流程和文件都能正常生成（通常几分钟到十几分钟，取决于机器），**其数值不能用作结论**。结束后在 `results/figures/` 查看图片。

### 5.2 正式实验（一条命令）

```bash
python run_experiments.py --seeds 42 43 44 --episodes 600 --eval-episodes 50 --eval-density all+extreme --jobs 3
```

`--jobs 3` 表示同时运行 3 个训练/评估进程（每个进程占用一个 SUMO 实例和一个 CPU 核）。可先用 `--dry-run` 只打印将要执行的命令。

### 5.3 分步运行（与一键命令等价）

```bash
# ① 训练：3 种算法 × 3 个种子（种子之间相互独立，可开多个终端并行）
python -m training.train_q_learning --seeds 42 43 44 --episodes 600
python -m training.train_dqn --algorithm dqn --seeds 42 43 44 --episodes 600
python -m training.train_dqn --algorithm double_dqn --seeds 42 43 44 --episodes 600

# ② 评估：直行+左转+右转，low/medium/high/extreme，每个组合 50 回合
python -m evaluation.evaluate_rule_based --density all+extreme --route all --episodes 50
python -m evaluation.evaluate_q_learning --all-seeds --density all+extreme --route all --episodes 50
python -m evaluation.evaluate_dqn --algorithm dqn --all-seeds --density all+extreme --route all --episodes 50
python -m evaluation.evaluate_dqn --algorithm double_dqn --all-seeds --density all+extreme --route all --episodes 50

# ③ 检查、聚合、对比、绘图
python -m evaluation.check_results
python -m evaluation.aggregate_seeds
python -m evaluation.compare_algorithms --density low medium high extreme --route straight left right
python -m visualization.plot_results
```

> **从 v1 升级的用户：** 请先把旧的 `results/` 和 `models/` 改名备份（如 `results_v1`、`models_v1`），再运行新流程。v1 的单模型文件仍可被读取，但与新的 `seed_<S>/` 目录混在一起容易造成混淆。

## 6. 左转 / 右转是怎么处理的

环境**一直支持**三条路线（`ego_route`：`straight` / `left` / `right`），v2 把它真正接入了整个流程：

| 环节 | v1 | v2 |
|---|---|---|
| 训练 | 只训练直行 | 默认每回合在三条路线中随机抽取（`--routes`） |
| 评估 | 可指定 `--route`，但结果会覆盖 | `--route all` 一次评估三条路线，结果合并保存 |
| 绘图 | 默认只画直行，路线缺失时报错 | 自动绘制所有已有路线；新增 `route_generalization.png` 并排对比 |
| 对比 | 只能比较直行 | `--route straight left right` |

两种实验设计可以并存，建议都做：

- **混合路线训练（默认）：** 训练时见过三条路线，评估三条路线都是“同分布测试”。
- **仅直行训练 → 左右转测试（泛化测试）：** 训练时加 `--routes straight`，评估时 `--route all`，考察策略对没见过的路线的迁移能力。注意模型的七维观测里**不含路线信息**，左转/右转只通过剩余距离、信号灯等间接体现。

> 注意：动作仍是纵向加速度，左右转时没有横向控制；左转还需要处理与对向车流的冲突，这部分由 SUMO 的路权规则决定。

## 7. 训练参数

### 7.1 Q-learning：`python -m training.train_q_learning`

| 参数 | 默认 | 含义与建议 |
|---|---|---|
| `--seeds 42 43 44` | 42 | 训练种子，每个种子独立训练一次。**至少 3 个，条件允许 5 个**，才能报告均值±标准差 |
| `--episodes` | 600 | 训练回合数。表格型一般 600–1500 足够；若 reward 曲线仍在上升就加大 |
| `--density` | mixed | `low/medium/high/extreme` 或 `mixed`（每回合在 low/medium/high 随机）。要复现 v1 用 `medium` |
| `--routes` | straight left right | 训练路线。仅直行：`--routes straight` |
| `--learning-rate` | 0.10 | 学习率 α。曲线抖动大可降到 0.05；收敛太慢可升到 0.2 |
| `--gamma` | 0.99 | 折扣因子。通常保持；回合很短时可试 0.95 |
| `--epsilon` / `--epsilon-min` | 1.0 / 0.05 | 初始 / 最低探索率 |
| `--epsilon-decay` | auto | `auto` 或固定系数（如 0.995）。见 7.3 |
| `--epsilon-reach` | 0.7 | `auto` 时在多少比例的回合后降到最低值 |
| `--save-interval` | 25 | 每隔多少回合保存一次模型 |

### 7.2 DQN / Double DQN：`python -m training.train_dqn --algorithm dqn|double_dqn`

在 7.1 的 `--seeds/--episodes/--density/--routes/--epsilon-*` 基础上，还有：

| 参数 | 默认 | 含义与建议 |
|---|---|---|
| `--learning-rate` | 1e-3 | Adam 学习率。loss 震荡或 reward 退化时降到 5e-4、3e-4 |
| `--batch-size` | 64 | 批大小。可试 128，训练更稳但更慢 |
| `--buffer-capacity` | 20000 | 经验池大小。回合数 ≥1500 时建议 50000–100000 |
| `--target-update-interval` | 200 | 目标网络每隔多少次梯度更新同步一次。不稳定时增大（500、1000）；收敛太慢时减小（100） |
| `--hidden-size` | 64 | 隐藏层宽度。改了之后评估时必须传相同的 `--hidden-size` |
| `--gamma` | 0.99 | 折扣因子 |

### 7.3 关于 ε（探索率）

ε 是每一步“随机选动作”的概率：ε=1 完全随机，ε→0 完全按 Q 值贪心。v1 每回合乘 0.995，300 回合后仍有 0.22，训练结束时还在大量随机探索。v2 默认 `--epsilon-decay auto`：在前 70% 的回合内平滑降到 `--epsilon-min`，后 30% 用接近贪心的策略“精修”。要复现 v1：`--epsilon-decay 0.995`。

### 7.4 训练/测试种子隔离

训练回合的环境种子是 `100000 + 训练种子×10000 + 回合号`，与测试种子（默认从 1000 起）不会重叠。v1 中 `42 + 回合号` 在训练超过约 960 回合后会与测试种子 1000–1049 重叠。

## 8. 评估参数（四个 `evaluation.evaluate_*` 通用）

| 参数 | 默认 | 含义与建议 |
|---|---|---|
| `--episodes` | 50 | 每个 (密度, 路线) 的回合数。想让置信区间更窄、差异更可信，用 100–200 |
| `--density` | all | `low/medium/high/extreme`；`all` = low+medium+high；`all+extreme` 再加 2.0 倍压力测试 |
| `--route` | straight | `straight/left/right`；**`all` = 三条路线全评估（推荐）** |
| `--seed` | 1000 | 测试种子起点。**四种算法必须相同**，否则不能配对比较 |
| `--max-steps` | 300 | 回合最大步数，四种算法必须相同 |
| `--train-seed S` | — | 评估训练种子 S 的模型（仅学习算法） |
| `--all-seeds` | — | 评估所有 `seed_<S>/` 目录里的模型 |
| `--overwrite` | 关 | 默认把新结果**合并**进已有 CSV；加此参数则整体覆盖 |
| `--output-dir` | 自动 | 指定输出目录（与 `--all-seeds` 不能同用） |
| `--model` | 自动 | 直接指定模型文件 |
| `--hidden-size` | 64 | 仅 DQN：必须与训练时一致 |

## 9. 对比、聚合、绘图参数

| 命令 | 参数 | 含义 |
|---|---|---|
| `evaluation.compare_algorithms` | `--route straight left right`、`--density ...` | 要检验的路线 / 密度 |
| | `--train-seed S` | 选用哪个训练种子的模型做配对检验（默认最小种子） |
| | `--min-reward-diff 0.3` / `--min-time-diff 2.0` | “有实际意义”的最小差异（奖励 / 通过时间 s）；`practically_meaningful` 列 = 校正后显著 **且** 差异超过阈值 |
| `evaluation.aggregate_seeds` | `--algorithms ...` | 汇总所有训练种子的均值与标准差 → `results/comparison/multi_seed_summary.csv`（**写报告/PPT 用这张表**） |
| `visualization.plot_results` | `--route all|straight|left|right` | 画哪条路线（默认全部已有路线） |
| | `--train-seed S` | 只画某个训练种子（默认汇总全部种子，带误差棒/阴影带） |
| | `--route-density medium` | `route_generalization.png` 使用的密度 |
| | `--smooth-window 20` | 训练曲线移动平均窗口 |

## 10. 输出的图表（`results/figures/`）

- `summary_metrics_<route>.png`：奖励、通过时间、停车步数、急刹步数（多种子时为均值±标准差）。
- `safety_rates_<route>.png`：安全通过率（含 95% Wilson 区间）；碰撞/闯红灯/超时/传送合并为一个“失败率”面板，全为 0 时给出文字说明而不是空图。
- `tradeoff_<route>.png`：通过时间 vs 急刹步数（效率—舒适性权衡，越靠左下越好）。
- `reward_boxplot_<route>.png`、`pass_time_s_boxplot_<route>.png`：逐回合分布。
- `action_distribution_<density>_<route>.png`：五种动作使用比例。
- `route_generalization.png`：直行/左转/右转并排对比。
- `training_reward_curves.png`、`training_steps_curves.png`、`training_loss_curves.png`：训练曲线（多种子带 ±std 阴影）。
- `training_epsilon_curves.png`：ε 的衰减过程。三种算法共用同一衰减时只画一条线。

## 11. 如何解读结果（重要）

- **奖励几乎由步数决定。** 奖励 ≈ 常数 − 约 0.058 × 步数（成功通过 +50、前进奖励总量固定、每步 −0.05），所以训练 reward 的上升主要来自“开得更快”，并在约 72.5 处触到上限，这是环境决定的，不是训练失败。请同时看 `training_steps_curves.png`。
- **安全指标有天花板。** 当前场景下四种算法常常 100% 安全通过，此时不能用安全指标区分算法；请看通过时间、停车、急刹，并用 `extreme` 密度和多训练种子检验。
- **统计显著 ≠ 差异大。** 配对检验的 p 值很小只说明差异稳定，需结合 `median_diff_a_minus_b` 与 `practically_meaningful` 判断。
- **单个训练种子不足以下结论。** 请至少 3 个训练种子，并以 `multi_seed_summary.csv` 的均值±标准差为准。

指标定义：

- **安全通过率：** 无碰撞、无确认闯红灯、无传送且到达目标出口的回合比例。
- **碰撞率 / 闯红灯率：** 发生过碰撞 / 环境确认红灯违规的回合比例。
- **超时率：** 达到最大步数且未通过的回合比例。
- **平均通过时间：** 成功回合中首次满足 `passed_intersection` 的时间均值。
- **停车步数 / 急刹步数：** 速度低于 0.1 m/s 的步数 / 实际加速度不大于 −3 m/s² 的步数。
- **最小前车间距：** 诊断指标；极小值需结合 SUMO 间距定义复核，不能单独作为碰撞或安全结论。

评估结果是当前仿真配置和测试种子集合下的结果，不应直接外推为真实道路安全性或普遍算法优劣。

## 12. 常见问题

| 现象 | 原因与处理 |
|---|---|
| 绘图报 `no numeric data to plot` | v1 中评估结果被后一次评估覆盖，只剩 left，而绘图默认画 straight。v2 已修复：评估合并保存、绘图缺少路线时给出明确提示。若仍遇到，说明确实没评估过该路线，运行 `--route all` 评估即可 |
| `compare_algorithms` 提示没有可检验的配对数据 | 所选路线/密度在某个算法的结果里不存在，先用 `--route all --density all+extreme` 评估 |
| `check_results` 提示缺少 (density, route) 组合 | 仅为提示；补评估即可 |
| 评估时报 `找不到模型` | 先训练；或用 `--train-seed` 指定存在的种子 |
| DQN 评估报 checkpoint 不匹配 | 训练时改过 `--hidden-size`，评估要传相同值 |
| 想复现 v1 的设置 | `--density medium --routes straight --episodes 300 --epsilon-decay 0.995` |

## 13. 录制 SUMO 场景演示

```bash
python -m visualization.record_demo --density medium
```

该视频是展示红灯等待和恢复信号后通行过程的**场景演示**，不是训练策略的性能视频。运行需要 SUMO-GUI 和 OpenCV。

## 14. 文档

- [修改说明（v2）](CHANGELOG_v2.md)
- [SUMO 安装与运行](docs/sumo_setup.md)
- [环境状态、动作、奖励与结束条件](docs/environment_interface.md)
- [交通流标定与环境检查](docs/calibration.md)
- [实验协议、指标和分析建议](docs/experiment_protocol.md)
