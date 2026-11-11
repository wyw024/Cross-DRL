
"""
评估训练好的Q-learning智能体。

在项目根目录运行：
python -m scripts.evaluate_q_learning

指定交通密度：
python -m scripts.evaluate_q_learning --density high

打开SUMO GUI：
python -m scripts.evaluate_q_learning --gui
"""

import argparse
import csv
from pathlib import Path

from algorithms.q_learning import QLearningAgent
from environment.sumo_env import IntersectionEnv


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "q_learning" / "q_table.npz"
RESULT_DIR = ROOT / "results" / "q_learning"


def evaluate(args):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"找不到模型：{MODEL_PATH}\n请先运行训练脚本。"
        )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    agent = QLearningAgent(seed=args.seed)
    agent.load(MODEL_PATH)

    # 评估阶段不进行随机探索
    agent.epsilon = 0.0

    env = IntersectionEnv(
        render_mode="human" if args.gui else None,
        max_steps=args.max_steps,
        traffic_density=args.density,
        gui_delay_ms=args.gui_delay,
    )

    records = []

    try:
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
                    training=False,
                )

                (
                    next_observation,
                    reward,
                    terminated,
                    truncated,
                    next_info,
                ) = env.step(action)

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

                if next_info.get("red_light_violation") is True:
                    red_light_violation = True

                if terminated or truncated:
                    truncated_episode = truncated
                    break

            record = {
                "episode": episode,
                "reward": round(total_reward, 4),
                "steps": steps,
                "passed_intersection": int(passed),
                "collision": int(collision),
                "red_light_violation": int(red_light_violation),
                "truncated": int(truncated_episode),
                "traffic_density": args.density,
            }

            records.append(record)

            print(
                f"Episode {episode:3d}/{args.episodes} | "
                f"Reward: {total_reward:8.2f} | "
                f"Steps: {steps:3d} | "
                f"Passed: {passed} | "
                f"Collision: {collision} | "
                f"Red violation: {red_light_violation}"
            )

    finally:
        env.close()

    output_path = RESULT_DIR / f"evaluation_{args.density}.csv"

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(records[0].keys()),
        )
        writer.writeheader()
        writer.writerows(records)

    passed_count = sum(r["passed_intersection"] for r in records)
    collision_count = sum(r["collision"] for r in records)
    violation_count = sum(r["red_light_violation"] for r in records)

    episode_count = len(records)

    print("\n========== 评估汇总 ==========")
    print(f"交通密度：{args.density}")
    print(f"测试回合：{episode_count}")
    print(f"平均奖励：{sum(r['reward'] for r in records) / episode_count:.3f}")
    print(f"平均步数：{sum(r['steps'] for r in records) / episode_count:.2f}")
    print(f"通过路口比例：{passed_count / episode_count:.2%}")
    print(f"碰撞比例：{collision_count / episode_count:.2%}")
    print(f"闯红灯判定比例：{violation_count / episode_count:.2%}")
    print(f"评估结果：{output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SUMO Q-learning agent"
    )

    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=300)

    parser.add_argument(
        "--density",
        choices=["low", "medium", "high"],
        default="medium",
    )

    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--gui-delay", type=int, default=300)

    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
