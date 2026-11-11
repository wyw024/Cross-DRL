"""Convenience entry point: python -m training.train_double_dqn --episodes 300"""

import sys
from training.train_dqn import parse_args, train


if __name__ == "__main__":
    sys.argv.extend(["--algorithm", "double_dqn"])
    train(parse_args())
