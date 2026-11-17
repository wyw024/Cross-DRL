"""Convenience entry point: python -m training.train_double_dqn --seeds 42 43 44"""

import sys
from training.train_dqn import parse_args, train


if __name__ == "__main__":
    sys.argv.extend(["--algorithm", "double_dqn"])   # last occurrence wins in argparse
    train(parse_args())
