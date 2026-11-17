"""Evaluate a trained DQN or Double DQN agent with the shared protocol in evaluation/common.py.

Run from the repository root:
    python -m evaluation.evaluate_dqn --algorithm dqn --all-seeds --route all
    python -m evaluation.evaluate_dqn --algorithm double_dqn --train-seed 42 --density all+extreme
    python -m evaluation.evaluate_double_dqn --all-seeds --route all      # shortcut for double_dqn

Checkpoints are written by training/train_dqn.py to results/<algorithm>/seed_<S>/checkpoint.pt.
Output: results/<algorithm>/seed_<S>/evaluation_episodes.csv and evaluation_summary.csv
Without --train-seed / --all-seeds the legacy file results/<algorithm>/checkpoint.pt is used.
New (density, route) rows are merged into existing CSV files; add --overwrite to replace them.
"""

import argparse
from pathlib import Path

from evaluation.common import (ROOT, add_common_args, add_train_seed_args, check_args,
                               run_evaluation_for_seeds)
from evaluation.paths import result_dir


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter))
    add_train_seed_args(parser)
    parser.add_argument("--algorithm", choices=("dqn", "double_dqn"), default="dqn")
    parser.add_argument("--model", type=Path, default=None,
                        help="checkpoint path (overrides --train-seed / --all-seeds lookup)")
    parser.add_argument("--hidden-size", type=int, default=64,
                        help="must match training (--hidden-size of train_dqn)")
    args = parser.parse_args()
    check_args(parser, args)
    if args.model is not None and args.all_seeds:
        parser.error("--model cannot be combined with --all-seeds")

    # Import here so --help works without PyTorch / SUMO installed.
    from algorithms.dqn import DQNAgent
    from algorithms.double_dqn import DoubleDQNAgent

    agent_type = DQNAgent if args.algorithm == "dqn" else DoubleDQNAgent

    def load_agent(train_seed):
        model = args.model or result_dir(ROOT / "results", args.algorithm, train_seed) / "checkpoint.pt"
        if not model.exists():
            raise FileNotFoundError(
                f"找不到模型：{model}\n请先运行 python -m training.train_{args.algorithm}")
        agent = agent_type(hidden_size=args.hidden_size, seed=args.seed)
        agent.load(model)        # raises ValueError if algorithm or network size does not match
        return agent

    run_evaluation_for_seeds(args, args.algorithm, load_agent)


if __name__ == "__main__":
    main()
