"""Evaluate the rule-based baseline with the shared protocol in evaluation/common.py.

Run from the repository root:
    python -m evaluation.evaluate_rule_based
    python -m evaluation.evaluate_rule_based --episodes 50 --density all --route straight
    python -m evaluation.evaluate_rule_based --density medium --gui

Output: results/rule_based/evaluation_episodes.csv and evaluation_summary.csv
"""

import argparse

from baseline.rule_based import RuleBasedAgent
from evaluation.common import add_common_args, check_args, run_evaluation


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter))
    args = parser.parse_args()
    check_args(parser, args)
    run_evaluation(RuleBasedAgent(), "rule_based", args)


if __name__ == "__main__":
    main()
