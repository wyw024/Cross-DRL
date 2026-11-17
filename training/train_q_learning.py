# -*- coding: utf-8 -*-
"""
训练 Q-learning 智能体。

在项目根目录运行：
    python -m training.train_q_learning
    python -m training.train_q_learning --seeds 42 43 44 --episodes 600

每个训练种子 S 的输出：
    models/q_learning/seed_<S>/q_table.npz       最终 Q-table（评估默认使用它）
    models/q_learning/seed_<S>/best_q_table.npz  单回合奖励最高时的 Q-table（仅供参考，噪声大）
    results/q_learning/seed_<S>/training_log.csv

复现 v1 设置（单一密度、仅直行）：
    python -m training.train_q_learning --density medium --routes straight --episodes 300 --epsilon-decay 0.995
"""

import argparse
import csv
from pathlib import Path

from algorithms.q_learning import QLearningAgent
from training.common import (DENSITY_CHOICES, ROUTES, ScenarioSampler, describe_schedule,
                             env_seed, resolve_epsilon_decay)

ROOT = Path(__file__).resolve().parents[1]

FIELDS = ["episode", "density", "ego_route", "reward", "steps", "epsilon",
          "passed_intersection", "collision", "red_light_violation", "truncated",
          "traffic_density"]


def train_one_seed(args, train_seed):
    # Imported here so that --help works without SUMO_HOME.
    from environment.sumo_env import IntersectionEnv

    model_dir = ROOT / "models" / "q_learning" / f"seed_{train_seed}"
    result_dir = ROOT / "results" / "q_learning" / f"seed_{train_seed}"
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    decay = resolve_epsilon_decay(args.epsilon_decay, args.epsilon, args.epsilon_min,
                                  args.episodes, args.epsilon_reach)
    agent = QLearningAgent(learning_rate=args.learning_rate, gamma=args.gamma,
                           epsilon=args.epsilon, epsilon_min=args.epsilon_min,
                           epsilon_decay=decay, seed=train_seed)
    sampler = ScenarioSampler(args.density, args.routes, train_seed)
    first_density = args.density if args.density in ("low", "medium", "high", "extreme") else "medium"
    env = IntersectionEnv(render_mode="human" if args.gui else None,
                          max_steps=args.max_steps, traffic_density=first_density,
                          gui_delay_ms=args.gui_delay)
    print(f"[q_learning | train seed {train_seed}] density={args.density} "
          f"routes={list(args.routes)} episodes={args.episodes}\n  "
          + describe_schedule(decay, args.epsilon, args.epsilon_min, args.episodes))

    log_path = result_dir / "training_log.csv"
    best_reward = float("-inf")
    try:
        with open(log_path, "w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
            writer.writeheader()
            for episode in range(1, args.episodes + 1):
                density, route = sampler.sample()
                observation, info = env.reset(seed=env_seed(train_seed, episode),
                                              options={"density": density, "ego_route": route})
                total_reward, steps = 0.0, 0
                passed = collision = red_light_violation = truncated_episode = False
                while True:
                    action = agent.select_action(observation, info=info, training=True)
                    next_observation, reward, terminated, truncated, next_info = env.step(action)
                    agent.learn(observation=observation, action=action, reward=reward,
                                next_observation=next_observation, terminated=terminated,
                                truncated=truncated)
                    observation, info = next_observation, next_info
                    total_reward += reward
                    steps += 1
                    passed = passed or bool(next_info.get("passed_intersection", False))
                    collision = collision or bool(next_info.get("collision", False))
                    # None = signal boundary uncertain; not counted as a violation
                    if next_info.get("red_light_violation") is True:
                        red_light_violation = True
                    if terminated or truncated:
                        truncated_episode = truncated
                        break

                agent.decay_epsilon()   # once per episode
                writer.writerow({
                    "episode": episode, "density": density, "ego_route": route,
                    "reward": round(total_reward, 4), "steps": steps,
                    "epsilon": round(agent.epsilon, 5),
                    "passed_intersection": int(passed), "collision": int(collision),
                    "red_light_violation": int(red_light_violation),
                    "truncated": int(truncated_episode), "traffic_density": density,
                })
                csv_file.flush()

                if total_reward > best_reward:
                    best_reward = total_reward
                    agent.save(model_dir / "best_q_table.npz")
                if episode % args.save_interval == 0:
                    agent.save(model_dir / "q_table.npz")
                if episode % 10 == 0 or episode == 1:
                    print(f"Seed {train_seed} Episode {episode:4d}/{args.episodes} [{density}/{route}] | "
                          f"Reward: {total_reward:8.2f} | Steps: {steps:3d} | "
                          f"Epsilon: {agent.epsilon:.4f} | Passed: {passed} | Collision: {collision}")
        agent.save(model_dir / "q_table.npz")
    finally:
        env.close()

    print(f"\n训练结束（种子 {train_seed}）\n训练记录：{log_path}\n"
          f"最终模型：{model_dir / 'q_table.npz'}\n最高单回合奖励：{best_reward:.2f}")
    return model_dir / "q_table.npz"


def train(args):
    return [train_one_seed(args, s) for s in args.seeds]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="训练随机种子，每个种子独立训练一次（默认 42）")
    parser.add_argument("--seed", type=int, default=None, help="单个训练种子的简写")
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--density", choices=DENSITY_CHOICES, default="mixed",
                        help="训练交通密度；mixed 表示每回合在 low/medium/high 中随机抽取")
    parser.add_argument("--routes", nargs="+", choices=ROUTES, default=list(ROUTES),
                        help="训练使用的主车路线；多条路线时每回合均匀随机抽取")
    parser.add_argument("--learning-rate", type=float, default=0.10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=1.0, help="初始探索率")
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", default="auto",
                        help="'auto'（在 epsilon-reach × episodes 回合后降到 epsilon-min）或每回合衰减系数，如 0.995")
    parser.add_argument("--epsilon-reach", type=float, default=0.7)
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--gui-delay", type=int, default=200)
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.seed is not None:
        args.seeds = [args.seed]
    if args.episodes <= 0 or args.max_steps <= 0 or args.save_interval <= 0:
        parser.error("episodes、max-steps、save-interval 必须为正数")
    try:
        resolve_epsilon_decay(args.epsilon_decay, args.epsilon, args.epsilon_min,
                              args.episodes, args.epsilon_reach)
    except ValueError as error:
        parser.error(str(error))
    return args


if __name__ == "__main__":
    train(parse_args())
