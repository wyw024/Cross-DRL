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

## 接口与边界

供算法同学查看的完整接口约定见 [docs/environment_interface.md](docs/environment_interface.md)；三档车流的初次检查记录见 [docs/calibration.md](docs/calibration.md)。

运行 `python evaluation/calibrate.py` 可复查三档车流与受控红灯场景。运行 `python visualization/record_demo.py --density medium` 可生成演示视频和逐步安全诊断 CSV，需安装 `opencv-python` 和支持 GUI 的 SUMO；该演示使用受控初始红灯与固定动作，不是训练模型表现。

`obs, info = env.reset(seed=42)`；`obs, reward, terminated, truncated, info = env.step(action)`。

状态依次为自车速度、距停止线距离、信号灯状态、前车间距、前车速度、前车相对速度。均做了缩放；信号为红 0、黄 0.5、绿或无后续信号 1。动作 0–4 分别对应请求加速度 -3、-1、0、1、2 m/s²。SUMO 默认的跟驰和信号安全检查仍然生效，实际加速度可能不同。

背景交通密度在构造环境时设置：`IntersectionEnv(traffic_density='low')`，可选 `low`、`medium`、`high`，分别将原始流量乘以 0.5、1.0、1.5。也可以在 `reset(options={'traffic_density': 'high'})` 时单独选择，或运行 `python main.py --gui --density high`。这些比例是待校准的实验初值，不保证对应真实交通密度；背景流量采用固定生成率，不同种子不一定生成不同的发车时刻。

主车越过路口进入 `south_out` 后回合结束，`info['passed_intersection']` 为真。主车从入口进入路口内部车道时，代码记录越线前后的信号；前后均为红灯时记录违规并结束。若同一步信号发生变化或信号不可获取，`signal_boundary_uncertain` 为真，`red_light_violation` 为 `None`，该次不判为违规或成功奖励。黄灯不计入红灯违规。时间步长为 1 秒，切灯边界的精确先后不可确定。SUMO 默认安全规则保持启用，通常会阻止红灯越线；不能用零违规直接证明算法安全。

奖励目前包含前进距离、时间、急刹、碰撞、确认的红灯违规和成功通过项，属于初始教学设计；`safety_interventions` 尚未实现。主车异常消失和传送会结束回合，不算成功；`info` 包含碰撞、传送和是否通过等字段。正式训练前仍需验证奖励尺度和各类边界情况。
