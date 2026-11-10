# SUMO 主车环境接口说明（供算法同学对接）

这是当前环境的接口草案，尚未与算法同学共同确认。运行 `from environment.sumo_env import IntersectionEnv`，构造 `IntersectionEnv(traffic_density='medium', max_steps=300)`。

```python
env = IntersectionEnv(traffic_density='medium')
try:
    obs, info = env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(3)
finally:
    env.close()
```

每步 1 秒；背景车辆由 SUMO 管理，主车从北向南直行，禁止主车换道；默认 SUMO speedMode 安全限制保持开启。`reset(options={'traffic_density':'high'})` 可覆盖该回合密度。

| 状态顺序 | 含义及范围 |
| --- | --- |
| 0 | 主车速度 / 13.89 m/s，裁剪到 [0,1] |
| 1 | 到停止线距离 / 250 m，裁剪到 [0,1]；通过停止线后为 0 |
| 2 | 前方信号：红 0，黄 0.5，绿或没有后续信号 1 |
| 3 | 与前车的 SUMO 间距 / 100 m，裁剪到 [0,1]；无前车为 1 |
| 4 | 前车速度 / 13.89 m/s；无前车时为主车速度 |
| 5 | (前车速度 − 主车速度) / 13.89，裁剪到 [-1,1] |

动作 0–4 对应请求加速度 -3、-1、0、1、2 m/s²；SUMO 安全规则与速度上下界可能使实际加速度不同。`info` 有 `requested_acceleration`、`actual_acceleration_m_s2`、未缩放速度、信号、前车间距等数据。

奖励当前为 `0.1*前进米数 - 0.05 - 0.1*急减速 - 100*碰撞 - 100*确认闯红灯 + 50*确认通过`。切灯边界不确定时不计红灯惩罚或成功奖励。这个奖励是初始参数；主车在当步消失时最后一步前进距离目前不能准确得到，应在算法实验前评估其影响。

`terminated`：成功驶入南侧出口、确认红灯违规、碰撞、传送或异常消失；`truncated`：达到 `max_steps`。通过后主车仍在出口道路，不等同于驶完全程。`info` 包含 `passed_intersection`、`red_light_violation`（布尔值或边界不确定时为 None）、`signal_boundary_uncertain`、`collision`、`teleported`。信号在越线的 1 秒步内切换时无法准确判断先后，不能将 None 当作未违规。需双方确认状态语义、奖励、终止条件及安全干预统计，后续再冻结接口。

`evaluation/safety_log.py` 记录请求和实际加速度，默认差值绝对值大于 0.5 m/s² 标记差异，并标注附近红黄灯、前车、低速和限速线索。这些只是用于人工分析的线索；差异不是安全干预次数，多个因素可以同时存在。
