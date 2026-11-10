"""Observe raw ego state and the effect of a simple sequence of controls."""
import argparse
from environment.sumo_env import IntersectionEnv

parser = argparse.ArgumentParser(description='观察主车速度、信号灯和动作效果')
parser.add_argument('--gui', action='store_true', help='显示 SUMO 图形界面')
parser.add_argument('--density', choices=['low','medium','high'], default='medium', help='背景交通密度')
args = parser.parse_args()
env = IntersectionEnv(render_mode='human' if args.gui else None, max_steps=150, traffic_density=args.density)
try:
    _, info = env.reset(seed=42)
    print('主车已生成:', info, flush=True)
    for step in range(150):
        action = [3, 2, 1, 0][(step // 8) % 4]
        _, reward, terminated, truncated, info = env.step(action)
        print(f"第{step+1:03d}秒  请求加速度={info['requested_acceleration']:+.1f} "
              f"实际加速度={info['actual_acceleration_m_s2']} "
              f"速度={info.get('speed_m_s')} 信号={info.get('signal')} "
              f"距信号={info.get('distance_to_signal_m')} 前车间距={info.get('leader_gap_m')} "
              f"到达={info['arrived']} 碰撞={info['collision']}", flush=True)
        if terminated or truncated:
            break
finally:
    env.close()
