# SUMO 安装与项目运行

## 1. 软件环境

项目当前开发环境以 Windows、Python 3.10.x、SUMO 1.28.0 为基准。需要：

- SUMO 1.28.0（包含 `sumo`、`sumo-gui` 和 `tools`）；
- Python 3.10.x；
- TraCI、sumolib、Gymnasium、NumPy、Pandas；
- PyTorch（DQN / Double DQN）；
- Matplotlib、SciPy（结果可视化和配对统计）。

安装 SUMO 后，将 `SUMO_HOME` 设置为 SUMO 安装目录，例如：

```powershell
$env:SUMO_HOME = "C:\Program Files (x86)\SUMO"
```

确认 `$env:SUMO_HOME\tools` 存在。安装 Python 依赖：

```powershell
python -m pip install -r requirements.txt
```

SUMO 是独立软件，不会由 requirements.txt 安装。

## 2. 运行环境演示

在项目根目录执行：

```powershell
python main.py --gui --density medium --route straight
```

`main.py` 使用脚本动作演示加速、保持和减速，不加载训练模型，不应当作算法评估结果。无图形界面运行时去掉 `--gui`。

可单独运行 SUMO 背景车流统计：

```powershell
python run.py --gui --steps 1000 --seed 42
```

输出写入 `results/csv/queues.csv`。

## 3. 路网重建与检查

修改 `environment/network/` 中节点或道路后，在项目根目录执行：

```powershell
python build_network.py
```

运行测试：

```powershell
python -m unittest discover -s tests -v
```

这些测试会启动 SUMO，需正确配置 SUMO_HOME 和网络文件。

## 4. 训练与评估入口

```powershell
python -m training.train_q_learning --episodes 500 --density medium
python -m training.train_dqn --algorithm dqn --episodes 300 --density medium
python -m training.train_dqn --algorithm double_dqn --episodes 300 --density medium

python -m evaluation.evaluate_rule_based --episodes 50 --density all --route straight
python -m evaluation.evaluate_q_learning --episodes 50 --density all --route straight
python -m evaluation.evaluate_dqn --algorithm dqn --episodes 50 --density all --route straight
python -m evaluation.evaluate_dqn --algorithm double_dqn --episodes 50 --density all --route straight
```

Rule-based 是规则基线，不需要训练。训练脚本会覆盖对应训练日志或模型文件；评估脚本会覆盖对应评估 CSV。正式结果建议先备份。

## 5. 对比、检查与图表

评估 CSV 已存在时，以下操作不需要启动 SUMO：

```powershell
python -m evaluation.check_results
python -m evaluation.compare_algorithms
python -m visualization.plot_results
```

可将输出位置指定到其他目录：

```powershell
python -m evaluation.compare_algorithms --out results/final_comparison
python -m visualization.plot_results --out results/final_figures
```

对比脚本使用相同 seed 的配对检验，并对 p 值进行 Holm 校正。只有测试场景相同才适合做配对统计；脚本会核对 `sumo_seed`。

## 6. 视频演示

```powershell
python -m visualization.record_demo --density medium
```

需要 SUMO-GUI 和 OpenCV。该脚本通过临时控制信号灯展示停车等待场景，是环境演示，不是已训练策略的 rollout。

## 7. 常见问题

- **找不到 SUMO / TraCI：** 检查 SUMO_HOME，确认其 `tools` 目录存在，并重新打开终端。
- **`sumo-gui` 无法启动：** 检查 SUMO 安装和图形桌面环境；无界面评估可不使用 `--gui`。
- **找不到模型：** Q-learning 默认读取 `models/q_learning/q_table.npz`；DQN 系列默认读取 `results/<algorithm>/checkpoint.pt`。
- **评估结果覆盖：** 每次评估默认覆盖该算法目录下的两个评估 CSV；需要保留多组实验时使用 `--output-dir` 或先备份。
- **图表没有训练曲线：** Rule-based 没有训练过程；其他算法需存在 `training_log.csv`，且至少包含 `episode`、`reward` 字段。
