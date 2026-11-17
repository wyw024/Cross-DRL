"""Train DQN or Double DQN against the shared SUMO Gymnasium environment.

Run from the repository root:
    python -m training.train_dqn --algorithm dqn
    python -m training.train_dqn --algorithm double_dqn --seeds 42 43 44 --episodes 600

Output (one directory per training seed S):
    results/<algorithm>/seed_<S>/training_log.csv
    results/<algorithm>/seed_<S>/checkpoint.pt

Reproduce the v1 setting (single density, straight route only):
    python -m training.train_dqn --density medium --routes straight --episodes 300 --epsilon-decay 0.995
"""

import argparse
import csv
from pathlib import Path

from algorithms.double_dqn import DoubleDQNAgent
from algorithms.dqn import DQNAgent
from training.common import (DENSITY_CHOICES, ROUTES, ScenarioSampler, describe_schedule,
                             env_seed, resolve_epsilon_decay)

FIELDS = ("episode", "density", "ego_route", "reward", "steps", "loss", "epsilon",
          "passed_intersection", "collision", "red_light_violation", "truncated")


def train_one_seed(args, train_seed):
    # Defer SUMO import so --help works without SUMO_HOME.
    from environment.sumo_env import IntersectionEnv

    root = Path(__file__).resolve().parents[1]
    output = root / "results" / args.algorithm / f"seed_{train_seed}"
    output.mkdir(parents=True, exist_ok=True)

    decay = resolve_epsilon_decay(args.epsilon_decay, args.epsilon_start, args.epsilon_min,
                                  args.episodes, args.epsilon_reach)
    agent_type = DQNAgent if args.algorithm == "dqn" else DoubleDQNAgent
    agent = agent_type(seed=train_seed, learning_rate=args.learning_rate, gamma=args.gamma,
                       epsilon=args.epsilon_start, epsilon_min=args.epsilon_min,
                       epsilon_decay=decay, hidden_size=args.hidden_size,
                       buffer_capacity=args.buffer_capacity, batch_size=args.batch_size,
                       target_update_interval=args.target_update_interval)
    sampler = ScenarioSampler(args.density, args.routes, train_seed)
    first_density = args.density if args.density in ("low", "medium", "high", "extreme") else "medium"
    env = IntersectionEnv(render_mode="human" if args.gui else None,
                          max_steps=args.max_steps, traffic_density=first_density)
    print(f"[{args.algorithm} | train seed {train_seed}] density={args.density} "
          f"routes={list(args.routes)} episodes={args.episodes}\n  "
          + describe_schedule(decay, args.epsilon_start, args.epsilon_min, args.episodes))
    try:
        with (output / "training_log.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            for episode in range(1, args.episodes + 1):
                density, route = sampler.sample()
                observation, info = env.reset(seed=env_seed(train_seed, episode),
                                              options={"density": density, "ego_route": route})
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
                writer.writerow({"episode": episode, "density": density, "ego_route": route,
                                 "reward": round(total_reward, 4), "steps": steps,
                                 "loss": sum(losses) / len(losses) if losses else "",
                                 "epsilon": round(agent.epsilon, 5),
                                 "passed_intersection": int(bool(info.get("passed_intersection"))),
                                 "collision": int(bool(info.get("collision"))),
                                 "red_light_violation": int(info.get("red_light_violation") is True),
                                 "truncated": int(truncated)})
                file.flush()
                if episode % args.save_interval == 0:
                    agent.save(output / "checkpoint.pt")
                if episode == 1 or episode % 10 == 0:
                    print(f"{args.algorithm} seed {train_seed} episode {episode}/{args.episodes} "
                          f"[{density}/{route}]: reward={total_reward:.2f}, steps={steps}, "
                          f"epsilon={agent.epsilon:.3f}")
        agent.save(output / "checkpoint.pt")
    finally:
        env.close()
    return output / "checkpoint.pt"


def train(args):
    paths = []
    for train_seed in args.seeds:
        paths.append(train_one_seed(args, train_seed))
    return paths


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algorithm", choices=("dqn", "double_dqn"), default="dqn")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="training seeds; each one is an independent run (default: 42)")
    parser.add_argument("--seed", type=int, default=None,
                        help="shortcut for a single training seed (same as --seeds N)")
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--density", choices=DENSITY_CHOICES, default="mixed",
                        help="training traffic density; 'mixed' samples low/medium/high every episode")
    parser.add_argument("--routes", nargs="+", choices=ROUTES, default=list(ROUTES),
                        help="ego routes used in training; several routes are sampled uniformly")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64,
                        help="must be passed again to evaluate_dqn --hidden-size if changed")
    parser.add_argument("--buffer-capacity", type=int, default=20_000)
    parser.add_argument("--target-update-interval", type=int, default=200,
                        help="hard target-network update every N gradient steps")
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", default="auto",
                        help="'auto' (reach epsilon-min after --epsilon-reach * episodes) or a "
                             "per-episode factor such as 0.995")
    parser.add_argument("--epsilon-reach", type=float, default=0.7,
                        help="fraction of episodes after which auto-decay reaches epsilon-min")
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--gui", action="store_true")
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.seed is not None:
        args.seeds = [args.seed]
    if args.episodes <= 0 or args.save_interval <= 0 or args.max_steps <= 0:
        parser.error("episodes, save-interval and max-steps must be positive")
    try:
        resolve_epsilon_decay(args.epsilon_decay, args.epsilon_start, args.epsilon_min,
                              args.episodes, args.epsilon_reach)
    except ValueError as error:
        parser.error(str(error))
    return args


if __name__ == "__main__":
    train(parse_args())
