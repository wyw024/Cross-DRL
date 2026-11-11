# 交叉口车辆决策：SUMO 项目

第一阶段包含固定周期信号灯、四向背景车流、一辆红色受控主车，以及 Gymnasium 环境接口。Q-learning、DQN、Double DQN 等算法尚未实现。

## 环境和运行

使用 Python 3.10/3.11、SUMO、TraCI、Gymnasium 和 NumPy。DQN 阶段还需要 PyTorch。`requirements.txt` 列出计划使用的包；SUMO 本体需单独安装。运行前请激活自己的 Python 环境，确保 SUMO_HOME 指向本机 SUMO 安装目录。

在 PowerShell 运行：

```powershell
cd '你的Cross-DRL仓库目录'
python main.py --gui
```

主车显示为红色；示例按固定顺序演示加速、保持、减速，并不是训练好的策略。去掉 `--gui` 可无界面运行。原始背景车流统计仍可通过 `python run.py --gui` 运行，数据写入 `results/csv/queues.csv`。

若更改节点或道路后需要重建路网：`python build_network.py`。若缺少依赖，可在激活所需 Python 环境后运行 `python -m pip install -r requirements.txt`。

## 目录

```text
environment/network/  路网、车辆路线和仿真配置
environment/sumo_env.py  Gymnasium 主车环境
main.py               主车控制演示
run.py                原背景车流统计
build_network.py      生成路网
algorithms/           待实现：Q-learning、DQN、Double DQN
baseline/             待实现：规则控制
training/             待实现：训练脚本
evaluation/           待实现：评估脚本
visualization/        待实现：绘图和展示
results/              模型、日志、图表和 CSV
```

## 接口与路线

最新接口以 [主车环境接口说明](environment_interface.md) 为准。主车从北侧驶入，可在运行演示时通过 --route straight、--route left 或 --route right 选择直行、左转、右转。例如：

`powershell
python main.py --gui --route left
`

方向在回合开始前确定；模型的五个动作仍控制纵向加速度。SUMO_HOME 需要指向本机 SUMO 安装目录。
