"""Run a short smoke demonstration of the ego-car environment."""
import argparse
from environment.sumo_env import IntersectionEnv

parser = argparse.ArgumentParser(description='SUMO 主车控制演示')
parser.add_argument('--gui', action='store_true', help='使用 SUMO-GUI')
parser.add_argument('--density', choices=['low','medium','high','extreme'], default='medium', help='背景交通密度')
parser.add_argument('--route', choices=['straight','left','right'], default='straight', help='主车预设路线')
args = parser.parse_args()
env = IntersectionEnv(render_mode='human' if args.gui else None, traffic_density=args.density, ego_route=args.route)
try:
    observation, info = env.reset(seed=42)
    print('初始状态:', observation, info)
    total = 0.0  # 初始化累计奖励
    for step in range(300):
        # Demonstrate acceleration, coasting, and braking; this is not a trained policy.
        action = [3, 2, 1][(step // 10) % 3]  # 每 10 步改变一次整数值，让索引在 0、1、2 之间循环
        observation, reward, terminated, truncated, info = env.step(action)
        total += reward
        if step % 10 == 0 or terminated or truncated:  # 每 10 步输出一次/仿真任务正常终止/仿真任务被截断
            print(f'步数={step+1} 动作={action} 速度={observation[0]*13.89:.1f}m/s '
                  f'信号={observation[2]} 奖励={reward:.2f} 结果={info}')
        if terminated or truncated:
            break
    print(f'累计奖励: {total:.2f}')
finally:
    env.close()
