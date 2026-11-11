"""Train DQN or Double DQN against the shared SUMO Gymnasium environment.

Run from repository root: python -m training.train_dqn --algorithm dqn
"""

import argparse
import csv
from pathlib import Path

from algorithms.dqn import DQNAgent
from algorithms.double_dqn import DoubleDQNAgent


def train(args):
    # Defer SUMO import so --help and algorithm unit checks work without SUMO_HOME.
    from environment.sumo_env import IntersectionEnv

    root = Path(__file__).resolve().parents[1]
    output = root / "results" / args.algorithm
    output.mkdir(parents=True, exist_ok=True)
    agent_type = DQNAgent if args.algorithm == "dqn" else DoubleDQNAgent
    agent = agent_type(seed=args.seed, learning_rate=args.learning_rate,
                       batch_size=args.batch_size,
                       target_update_interval=args.target_update_interval)
    env = IntersectionEnv(render_mode="human" if args.gui else None,
                          max_steps=args.max_steps, traffic_density=args.density)
    fields = ("episode", "reward", "steps", "loss", "epsilon",
              "passed_intersection", "collision", "red_light_violation", "truncated")
    try:
        with (output / "training_log.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for episode in range(1, args.episodes + 1):
                observation, info = env.reset(seed=args.seed + episode)
                total_reward, steps, losses = 0.0, 0, []
                while True:
                    action = agent.select_action(observation, info, training=True)
                    next_obs, reward, terminated, truncated, next_info = env.step(action)
                    loss = agent.learn(observation, action, reward, next_obs,
                                       terminated, truncated)
                    if loss is not None:
                        losses.append(loss)
                    total_reward += reward
                    steps += 1
                    observation, info = next_obs, next_info
                    if terminated or truncated:
                        break
                agent.decay_epsilon()
                writer.writerow({"episode": episode, "reward": total_reward,
                                 "steps": steps, "loss": sum(losses) / len(losses) if losses else "",
                                 "epsilon": agent.epsilon,
                                 "passed_intersection": int(bool(info.get("passed_intersection"))),
                                 "collision": int(bool(info.get("collision"))),
                                 "red_light_violation": int(info.get("red_light_violation") is True),
                                 "truncated": int(truncated)})
                file.flush()
                if episode % args.save_interval == 0:
                    agent.save(output / "checkpoint.pt")
                if episode == 1 or episode % 10 == 0:
                    print(f"{args.algorithm} episode {episode}/{args.episodes}: "
                          f"reward={total_reward:.2f}, steps={steps}, epsilon={agent.epsilon:.3f}")
        agent.save(output / "checkpoint.pt")
    finally:
        env.close()
    return output / "checkpoint.pt"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("dqn", "double_dqn"), default="dqn")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--density", choices=("low", "medium", "high"), default="medium")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--target-update-interval", type=int, default=200)
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    if args.episodes <= 0 or args.save_interval <= 0 or args.max_steps <= 0:
        parser.error("episodes, save-interval and max-steps must be positive")
    return args


if __name__ == "__main__":
    train(parse_args())
