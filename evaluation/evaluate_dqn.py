"""Evaluate a trained DQN or Double DQN agent with the shared protocol in evaluation/common.py.

Run from the repository root:
    python -m evaluation.evaluate_dqn --algorithm dqn
    python -m evaluation.evaluate_dqn --algorithm double_dqn --density all --episodes 50
    python -m evaluation.evaluate_double_dqn            # shortcut for --algorithm double_dqn

The checkpoint is written by training/train_dqn.py to results/<algorithm>/checkpoint.pt.
Output: results/<algorithm>/evaluation_episodes.csv and evaluation_summary.csv
"""

import argparse
from pathlib import Path

from evaluation.common import ROOT, add_common_args, check_args, run_evaluation


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument("--algorithm", choices=("dqn", "double_dqn"), default="dqn")
    parser.add_argument("--model", type=Path, default=None,
                        help="checkpoint path (default: results/<algorithm>/checkpoint.pt)")
    parser.add_argument("--hidden-size", type=int, default=64,
                        help="must match training; the training scripts always use 64")
    args = parser.parse_args()
    check_args(parser, args)

    model = args.model or ROOT / "results" / args.algorithm / "checkpoint.pt"
    if not model.exists():
        raise FileNotFoundError(
            f"找不到模型：{model}\n请先运行 python -m training.train_{args.algorithm}")

    # Import here so --help works without PyTorch / SUMO installed.
    from algorithms.dqn import DQNAgent
    from algorithms.double_dqn import DoubleDQNAgent

    agent_type = DQNAgent if args.algorithm == "dqn" else DoubleDQNAgent
    agent = agent_type(hidden_size=args.hidden_size, seed=args.seed)
    agent.load(model)            # raises ValueError if algorithm or network size does not match
    run_evaluation(agent, args.algorithm, args)


if __name__ == "__main__":
    main()
