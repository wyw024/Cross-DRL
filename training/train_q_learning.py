
"""
训练Q-learning智能体。

在项目根目录运行：
python -m scripts.train_q_learning

例如：
python -m scripts.train_q_learning --episodes 500 --density medium
"""

import argparse
import csv
from pathlib import Path

from algorithms.q_learning import QLearningAgent
from environment.sumo_env import IntersectionEnv


ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "models" / "q_learning"
RESULT_DIR = ROOT / "results" / "q_learning"


def train(args):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    agent = QLearningAgent(
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
    )

    env = IntersectionEnv(
        render_mode="human" if args.gui else None,
        max_steps=args.max_steps,
        traffic_density=args.density,
        gui_delay_ms=args.gui_delay,
    )

    log_path = RESULT_DIR / "training_log.csv"
    best_reward = float("-inf")

    fieldnames = [
        "episode",
        "reward",
        "steps",
        "epsilon",
        "passed_intersection",
        "collision",
        "red_light_violation",
        "truncated",
        "traffic_density",
    ]

    try:
        with open(
            log_path,
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames,
            )
            writer.writeheader()

            for episode in range(1, args.episodes + 1):

                observation, info = env.reset(
                    seed=args.seed + episode
                )

                total_reward = 0.0
                steps = 0

                passed = False
                collision = False
                red_light_violation = False
                truncated_episode = False

                while True:

                    action = agent.select_action(
                        observation,
                        info=info,
                        training=True,
                    )

                    (
                        next_observation,
                        reward,
                        terminated,
                        truncated,
                        next_info,
                    ) = env.step(action)

                    agent.learn(
                        observation=observation,
                        action=action,
                        reward=reward,
                        next_observation=next_observation,
                        terminated=terminated,
                        truncated=truncated,
                    )

                    observation = next_observation
                    info = next_info

                    total_reward += reward
                    steps += 1

                    passed = passed or next_info.get(
                        "passed_intersection", False
                    )

                    collision = collision or next_info.get(
                        "collision", False
                    )

                    # None表示该次信号灯边界判定不确定，
                    # 不把它直接当作发生了闯红灯。
                    violation = next_info.get("red_light_violation")
                    if violation is True:
                        red_light_violation = True

                    if terminated or truncated:
                        truncated_episode = truncated
                        break

                # 一个episode结束后再衰减探索率
                agent.decay_epsilon()

                row = {
                    "episode": episode,
                    "reward": round(total_reward, 4),
                    "steps": steps,
                    "epsilon": round(agent.epsilon, 5),
                    "passed_intersection": int(passed),
                    "collision": int(collision),
                    "red_light_violation": int(red_light_violation),
                    "truncated": int(truncated_episode),
                    "traffic_density": args.density,
                }

                writer.writerow(row)
                csv_file.flush()

                if total_reward > best_reward:
                    best_reward = total_reward
                    agent.save(MODEL_DIR / "best_q_table.npz")

                if episode % args.save_interval == 0:
                    agent.save(MODEL_DIR / "q_table.npz")

                if episode % 10 == 0 or episode == 1:
                    print(
                        f"Episode {episode:4d}/{args.episodes} | "
                        f"Reward: {total_reward:8.2f} | "
                        f"Steps: {steps:3d} | "
                        f"Epsilon: {agent.epsilon:.4f} | "
                        f"Passed: {passed} | "
                        f"Collision: {collision}"
                    )

        agent.save(MODEL_DIR / "q_table.npz")

    finally:
        env.close()

    print("\n训练结束")
    print(f"训练记录：{log_path}")
    print(f"最终模型：{MODEL_DIR / 'q_table.npz'}")
    print(f"最高单回合奖励：{best_reward:.2f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="SUMO intersection Q-learning training"
    )

    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=300)

    parser.add_argument(
        "--density",
        choices=["low", "medium", "high"],
        default="medium",
    )

    parser.add_argument("--learning-rate", type=float, default=0.10)
    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-interval", type=int, default=25)

    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--gui-delay", type=int, default=200)

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
