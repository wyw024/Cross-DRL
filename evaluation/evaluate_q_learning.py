"""Evaluate a trained Q-learning agent with the shared protocol in evaluation/common.py.

Run from the repository root:
    python -m evaluation.evaluate_q_learning
    python -m evaluation.evaluate_q_learning --model models/q_learning/best_q_table.npz --density medium

Output: results/q_learning/evaluation_episodes.csv and evaluation_summary.csv
"""

import argparse
from pathlib import Path

from algorithms.q_learning import QLearningAgent
from evaluation.common import ROOT, add_common_args, check_args, run_evaluation

DEFAULT_MODEL = ROOT / "models" / "q_learning" / "q_table.npz"


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="Q-table .npz file (default: models/q_learning/q_table.npz)")
    args = parser.parse_args()
    check_args(parser, args)

    if not args.model.exists():
        raise FileNotFoundError(f"找不到模型：{args.model}\n请先运行 python -m training.train_q_learning")

    agent = QLearningAgent(seed=args.seed)
    agent.load(args.model)
    agent.epsilon = 0.0          # greedy evaluation; select_action(training=False) also ignores it
    run_evaluation(agent, "q_learning", args)


if __name__ == "__main__":
    main()
