"""Evaluate a trained Q-learning agent with the shared protocol in evaluation/common.py.

Run from the repository root:
    python -m evaluation.evaluate_q_learning --all-seeds --route all
    python -m evaluation.evaluate_q_learning --train-seed 42 --density all+extreme
    python -m evaluation.evaluate_q_learning --model models/q_learning/best_q_table.npz   # legacy single model

Q-tables are written by training/train_q_learning.py to models/q_learning/seed_<S>/q_table.npz.
Output: results/q_learning/seed_<S>/evaluation_episodes.csv and evaluation_summary.csv
New (density, route) rows are merged into existing CSV files; add --overwrite to replace them.
"""

import argparse
from pathlib import Path

from algorithms.q_learning import QLearningAgent
from evaluation.common import (add_common_args, add_train_seed_args, check_args,
                               run_evaluation_for_seeds)
from evaluation.paths import q_table_path


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter))
    add_train_seed_args(parser)
    parser.add_argument("--model", type=Path, default=None,
                        help="Q-table .npz file (overrides --train-seed / --all-seeds lookup)")
    args = parser.parse_args()
    check_args(parser, args)
    if args.model is not None and args.all_seeds:
        parser.error("--model cannot be combined with --all-seeds")

    def load_agent(train_seed):
        model = args.model or q_table_path(train_seed)
        if not model.exists():
            raise FileNotFoundError(f"找不到模型：{model}\n请先运行 python -m training.train_q_learning")
        agent = QLearningAgent(seed=args.seed)
        agent.load(model)
        agent.epsilon = 0.0      # greedy evaluation; select_action(training=False) also ignores it
        return agent

    run_evaluation_for_seeds(args, "q_learning", load_agent)


if __name__ == "__main__":
    main()
