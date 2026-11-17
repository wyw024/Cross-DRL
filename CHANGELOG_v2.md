# v2 修改说明

本次修改针对上一轮分析中发现的四类问题：**评估结果被覆盖、绘图信息量低且有遮挡、训练设置与评估不匹配、算法差异看不出来**。
**未改动**：`algorithms/*`（Q-learning / DQN / Double DQN 的算法实现）、奖励函数、七维观测与五个动作、SUMO 路网、规则基线。因此 v1 与 v2 的算法本身完全一致，差别只在训练设置、评估/绘图流程与结果组织方式。

## 1. 问题 → 原因 → 修改

| # | 问题 | 原因 | 修改 |
|---|---|---|---|
| 1 | `safety_rates_*.png` 只有 0 和 1 | 四种算法在直行场景下全部 100% 安全通过，碰撞/闯红灯/超时全为 0，是真实结果（天花板效应），不是绘图错误 | 安全通过率保留并加 95% Wilson 区间；碰撞/闯红灯/超时/传送合并成一个“失败率”面板，全为 0 时显示说明文字而不是三张空图；新增 `tradeoff_<route>.png`（通过时间 vs 急刹）；新增 `extreme`（2.0 倍）压力测试密度 |
| 2 | `training_epsilon_curves.png` 只见一条线 | 三种算法 ε 衰减完全相同，曲线重合，后画的盖住前面的；同时 ε 衰减过慢，300 回合后仍有 0.22 | 绘图：完全重合时只画一条并标注“所有学习算法相同”；训练：新增 `--epsilon-decay auto`，在 70% 回合内降到 0.05；新增 `training_steps_curves.png`、`training_loss_curves.png` |
| 3 | 评估 left/right 后绘图报 `no numeric data to plot` | 评估把结果写进固定文件并**整体覆盖**，评估 left 后文件中只有 left；绘图默认画 straight，过滤出空表，`pivot.plot` 无数据可画 | 评估改为按 (algorithm, density, ego_route) **合并**写入（`--overwrite` 可恢复旧行为）；绘图默认画全部已有路线，指定的路线不存在时给出明确提示；`compare_algorithms` / `check_results` 对缺失路线给出提示；新增 `route_generalization.png` |
| 4 | 四个模型差不多，reward 卡在约 72.5 | reward ≈ 73.3 − 0.058×步数，上限由环境决定；任务太简单（直行、单一密度、全部成功）；单训练种子，噪声大于算法差异 | 训练默认混合密度 + 混合路线；多训练种子；ε 降到下限；增加 `extreme` 压力测试；多种子均值±标准差汇总；统计检验增加“实际意义”判定；训练/测试种子隔离 |

## 2. 新增 / 修改的文件

### 新增
| 文件 | 作用 |
|---|---|
| `run_experiments.py` | 一键流程：训练→评估→检查→聚合→对比→绘图；`--quick` 冒烟测试，`--jobs N` 并行，`--dry-run` 只打印命令 |
| `training/common.py` | 训练工具：混合场景采样 `ScenarioSampler`、`auto_epsilon_decay`、训练环境种子 `env_seed` |
| `training/__init__.py` | 使 `training` 成为正式包 |
| `evaluation/paths.py` | 统一管理 `results/<algo>/seed_<S>/` 等路径与种子发现，并兼容 v1 的旧目录 |
| `evaluation/aggregate_seeds.py` | 汇总所有训练种子的均值与标准差 → `results/comparison/multi_seed_summary.csv` |
| `tests/test_tools.py` | 8 个不依赖 SUMO / PyTorch 的单元测试（CSV 合并、种子隔离、ε 衰减、采样、路径发现） |
| `CHANGELOG_v2.md` | 本文件 |

### 修改
| 文件 | 修改内容 |
|---|---|
| `training/train_dqn.py`、`train_double_dqn.py` | 新增参数：`--seeds/--seed`、`--density mixed`、`--routes`、`--gamma`、`--hidden-size`、`--buffer-capacity`、`--epsilon-start/-min/-decay/-reach`；默认 600 回合；日志增加 density、ego_route；输出到 `results/<algo>/seed_<S>/`；训练环境种子与测试种子隔离 |
| `training/train_q_learning.py` | 同上（Q-learning 版）；模型输出到 `models/q_learning/seed_<S>/`；日志增加 density、ego_route |
| `evaluation/common.py` | 合并写入 `merge_csv`；`--overwrite`、`--train-seed`、`--all-seeds`；`--density` 增加 `extreme`、`all+extreme`；`run_evaluation_for_seeds` |
| `evaluation/evaluate_dqn.py`、`evaluate_q_learning.py` | 接入 `--train-seed/--all-seeds`，按训练种子加载模型、写结果 |
| `evaluation/evaluate_rule_based.py` | 仅更新说明（用法示例） |
| `evaluation/compare_algorithms.py` | `--train-seed`；`--min-reward-diff/--min-time-diff` 与 `practically_meaningful` 列；无差异时 p 值记为 1 而非空；缺失路线提示；`extreme` |
| `evaluation/check_results.py` | `--train-seed`；`--expect-routes/--expect-densities` 缺失组合提示（警告，不报错） |
| `visualization/plot_results.py` | 重写：见第 1 节；固定配色（颜色跟随算法）；图例放到图外；误差线不低于 0；多种子均值±标准差和训练曲线阴影带；两算法数值完全相同时标记大小不同，避免相互遮挡 |
| `environment/sumo_env.py` | 仅在 `DENSITY_SCALE` 中增加 `"extreme": 2.0`（及注释） |
| `main.py`、`visualization/record_demo.py` | `--density` 可选项增加 `extreme` |
| `README.md`、`docs/experiment_protocol.md`、`docs/environment_interface.md`、`docs/calibration.md` | 同步更新（运行步骤、参数表、调参建议、左右转说明、结果解读边界） |
| `.github/workflows/test.yml` | 增加无需 SUMO 的单元测试步骤 |

## 3. 行为变化与兼容性

1. **训练默认值变了**：`--density` 由 `medium` 变为 `mixed`，路线由仅直行变为三条路线随机，回合数由 300 变为 600，ε 衰减由 0.995 变为 `auto`。要复现 v1：`--density medium --routes straight --episodes 300 --epsilon-decay 0.995`。
2. **输出位置变了**：训练和评估结果在 `results/<algo>/seed_<S>/`，Q-table 在 `models/q_learning/seed_<S>/`。v1 的旧文件仍可读取（没有 `seed_*` 目录时回退到旧位置），但**建议先把旧 `results/`、`models/` 改名备份再跑**，避免新旧混在一起。
3. **评估默认合并而非覆盖**。需要整体替换时加 `--overwrite`。
4. **训练环境种子变了**（`100000 + 种子×10000 + 回合号`），因此即使使用 `--density medium --routes straight`，具体回合场景也与 v1 不同，逐回合曲线不会完全一样。
5. 由于 v1 的模型是单一密度、仅直行训练得到的，v1 与 v2 的结果**不能直接作为严格同条件的比较**。

## 4. 验证情况

已验证（在安装了 SUMO 1.28.0 与 PyTorch 的环境中）：
- 8 个新单元测试全部通过；仓库原有 5 个 SUMO 测试（路线、随机交通等）13 项全部通过。
- 冒烟全流程：2 个训练种子 × 3 种学习算法训练（8 回合）、规则基线 + 6 个模型在 `all+extreme` × `straight/left/right` 下评估（每组合 2 回合）、`check_results` / `aggregate_seeds` / `compare_algorithms` / `plot_results` 全部正常运行，并行（`--jobs 4`）正常。
- 评估合并与 `--overwrite` 行为符合预期；使用你仓库中 v1 的结果文件（仅含 left）运行绘图，不再报错，缺少 straight 时给出明确提示。

**未验证（需要你运行）**：
- 正式规模训练（600 回合 × 3 种子）的实际结果。混合训练、ε 降到下限、多种子是否能让算法差异显现，**取决于你的实际运行结果，我没有也不能预先保证**；`extreme` 密度下四种算法是否仍然 100% 安全，同样需要实测。
- 冒烟测试只用了 8 回合训练，其中 DQN 与 Double DQN 的数值几乎相同，这是训练太短所致，不代表正式结果。

## 5. 仍然存在的限制（未在 v2 处理）

- 奖励函数没有改动：奖励仍主要由通过步数决定，上限约 72.5；如要拉开算法差异，下一步可考虑在奖励中加入舒适性与安全距离项（会改变训练目标，需要重新训练全部算法）。
- 观测中不含路线信息；左转与对向车流的冲突完全由 SUMO 路权规则决定。
- 未引入 Dueling / 优先回放 / 软更新等 DQN 改进，保持与 v1 算法一致以便对照。
