# 实验协议、指标与结果分析

## 1. 实验对象

比较 Rule-based、Q-learning、DQN、Double DQN 四种纵向控制策略。Rule-based 为非学习基线。四种方法调用同一个 `IntersectionEnv`，采用相同的观测、动作、交通密度和评估流程。

## 2. 评估协议

`evaluation/common.py` 中的默认设置：

| 参数 | 默认值 |
|---|---|
| 每个密度/路线回合数 | 50 |
| 最大步数 | 300 |
| 测试 seed 起点 | 1000 |
| 密度 | low / medium / high |
| 路线 | straight / left / right |
| 动作数 | 5 |
| 停车阈值 | 0.1 m/s |
| 急刹阈值 | -3.0 m/s² |

每个算法对给定组合使用相同 seed。训练 seed 与测试 seed 分离。正式对比应保证评估回合数、最大步数、密度、路线、seed 起点、模型选择规则均一致。

## 3. 指标定义

| 指标 | 定义 | 解释 |
|---|---|---|
| mean_reward | 每回合累计奖励的均值 | 综合奖励目标 |
| std_reward | 回合奖励标准差 | 结果波动 |
| safe_pass_rate | outcome 为 passed 的比例 | 无碰撞、无确认闯红灯、无传送且成功通过 |
| collision_rate | 至少发生一次碰撞的比例 | 碰撞风险 |
| red_violation_rate | 确认闯红灯的比例 | 信号遵守情况 |
| timeout_rate | 未通过且达到最大步数的比例 | 通行失败/超时 |
| mean_pass_time_s | 成功回合的首次通过时间均值 | 通行效率 |
| mean_speed_m_s | 回合平均速度的均值 | 行驶速度表现 |
| mean_stopped_steps | 速度低于 0.1 m/s 的步数均值 | 停车等待 |
| mean_hard_brake_steps | 实际加速度不大于 -3 m/s² 的步数均值 | 急刹行为 |
| min_leader_gap_m | 所有记录中最小前车间距 | 诊断性跟车指标 |

安全通过率、碰撞率和红灯违规率提供 Wilson 95% 置信区间。置信区间反映有限回合样本的不确定性，不等同于真实道路安全概率。

## 4. 统计检验

不同算法在相同测试 seed 下运行相同 SUMO 场景，因此结果是配对样本。`evaluation.compare_algorithms` 使用：

- 累计奖励：双侧 Wilcoxon signed-rank 配对检验；
- 通过时间：双侧 Wilcoxon 配对检验，仅保留双方均成功通过的 seed；
- 安全通过：精确配对二项检验（McNemar 检验的条件形式）；
- 多重比较：对全部检验 p 值进行 Holm-Bonferroni 校正。

显著性阈值为校正后 p < 0.05。统计显著不等于实际效果差异足够大，应结合中位数差、分布图和置信区间解释。

## 5. 可视化

`python -m visualization.plot_results` 会生成：

- `summary_metrics_<route>.png`：奖励、通过时间、停车和急刹；
- `safety_rates_<route>.png`：安全通过、碰撞、闯红灯、超时比例；
- `reward_boxplot_<route>.png`：逐回合奖励分布；
- `pass_time_s_boxplot_<route>.png`：成功通过时间分布；
- `action_distribution_<density>_<route>.png`：动作比例；
- `training_reward_curves.png`：训练奖励移动平均；
- `training_epsilon_curves.png`：探索率变化（训练日志具备 epsilon 字段时）。

## 6. 当前压缩包数据覆盖范围

当前四种算法各自有 150 个评估回合：三种密度各 50 个，全部为 straight 路线。相同 seed 对应相同 `sumo_seed`。因此可以开展三种密度下的直行四算法配对比较；left/right 的算法比较仍需补充统一评估。

当前四种算法的安全通过率均为 100%，碰撞率、闯红灯率和超时率均为 0。这说明本批样本中安全结果相同，不代表算法在更广泛场景下安全性等价。应重点分析奖励、通过时间、停车、急刹和动作分布。

## 7. 结果解释的边界

- 单次训练得到的模型不足以刻画训练随机性；更严谨的算法比较应采用多个训练随机种子，报告不同训练模型的均值和方差。
- 相同测试 seed 只控制评估场景，不代表不同算法训练过程相同。
- 当前主车路线在回合开始前指定，动作仅控制纵向加速度；不能将结果表述为完整的自主路线选择。
- 最小前车间距接近 0 时，应复核具体逐回合轨迹、车辆长度、车道与 SUMO gap 语义。
- 仿真结果不能直接等同于真实道路部署效果。
