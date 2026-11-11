# SUMO 主车环境接口（对接草案）

当前环境提供 Gymnasium 接口，供规则控制、Q-learning 和 DQN 等算法调用。请组员共同确认后再将接口作为稳定版本。

```python
from environment.sumo_env import IntersectionEnv

env = IntersectionEnv(traffic_density='medium', ego_route='left')
try:
    obs, info = env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(3)
finally:
    env.close()
```

## 主车方向与动作

主车从北侧进入。`ego_route` 可设为 `straight`（直行，南侧出口）、`left`（左转，东侧出口）或 `right`（右转，西侧出口），默认直行。每回合可在 `env.reset(options={'ego_route': 'right', 'density': 'low'})` 更换路线和密度；`info` 提供 `ego_route`、`route_id`、`exit_edge`。

路线在回合开始前确定，SUMO 根据路线完成转向；智能体每步仍只选择加速度，**尚不支持在路口自主选择方向**。当前七维状态中没有目标方向；训练中混合路线时应与算法同学商定如何提供目标信息，不能假定只靠当前状态就能区分转向意图。

动作编号 0–4 对应请求加速度 -3、-1、0、1、2 m/s²，时间步长 1 秒。SUMO 默认跟驰和信号安全约束会影响实际加速度。

## 七维观测

| 下标 | 内容 | 归一化范围 |
| --- | --- | --- |
| 0 | 自车速度 / 13.89 m/s | 0–1 |
| 1 | 距入口停止线距离 / 250 m；进入路口后为 0 | 0–1 |
| 2 | 当前路线对应信号：红 0、黄 0.5、绿或没有后续信号 1 | 0–1 |
| 3 | 前车距离 / 100 m；无前车时为 1 | 0–1 |
| 4 | 前车速度 / 13.89 m/s；无前车时为 1 | 0–1 |
| 5 | (自车速度 − 前车速度) / 13.89 m/s，裁剪 | −1–1 |
| 6 | `light_present`：当前代码中表示是否有前车 | 0 或 1 |

第 6 项的名称容易误解，但为兼容现有算法暂不改名。背景密度 `low`、`medium`、`high` 对应原流量的 0.5、1、1.5 倍；初始信号相位默认随机化。

## 奖励与结束

当前奖励包含每步前进奖励 `0.1 × 进展米数`、时间惩罚 `-0.05`、实际加速度不大于 -3 时的 `-0.1`、碰撞和确认闯红灯各 `-100`、成功通过 `+50`。这些值是初步设计，不代表已完成调优。

主车进入所选出口、到达、碰撞、传送、确认红灯违规或异常消失时 `terminated=True`；达到最大步数时 `truncated=True`。`info` 包含 `passed_intersection`、`collision`、`teleported`、`red_light_violation` 等字段。成功驶入出口不等于驶完全程。

安全保护介入次数目前尚未实现。红灯判断基于越线前缓存的信号；一秒仿真步内切灯的先后无法精确还原，正式安全评价前需补充边界测试并与组员确认。`evaluation/safety_log.py` 的加速度差异仅是排查线索，不能直接等同于安全介入。
