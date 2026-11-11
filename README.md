# Cross-DRL
基于深度强化学习的自动驾驶车辆交叉口通行决策研究——基于Q-learning、DQN及其改进算法的仿真实验 

## SUMO 仿真环境

- [环境搭建和运行](docs/sumo_setup.md)
- [主车状态、动作和 Gymnasium 接口草案](docs/environment_interface.md)
- [三档交通流初次测试记录](docs/calibration.md)

目前提供单路口、固定信号、背景车流和单辆受控主车，以及 Q-learning、DQN 和 Double DQN 的初步实现；环境与算法接口仍需组员共同确认。

## 算法

`baseline/rule_based.py` 提供无需训练的规则控制；`algorithms/q_learning.py` 提供离散状态 Q-learning。`algorithms/dqn.py` 和 `algorithms/double_dqn.py` 使用相同的 7 维观测、5 个动作，均含经验回放和目标网络。Double DQN 由在线网络选下一动作、目标网络评估该动作。时间截断时继续 bootstrap，真正终止时不 bootstrap。

在项目根目录运行（须先按[环境搭建说明](docs/sumo_setup.md)安装 SUMO 并设置 `SUMO_HOME`）：

```bash
python -m training.train_dqn --algorithm dqn --episodes 300
python -m training.train_double_dqn --episodes 300
```

默认使用无界面 SUMO 和 CPU；`--gui` 可用于演示。训练日志和模型写入 `results/dqn/` 或 `results/double_dqn/`，两种算法可使用相同的种子与交通密度进行比较。这些训练入口只生成模型与日志，不预设哪种算法效果更好。
