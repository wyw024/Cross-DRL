"""Convenience entry point: python -m evaluation.evaluate_double_dqn"""

import sys

from evaluation.evaluate_dqn import main


if __name__ == "__main__":
    sys.argv.extend(["--algorithm", "double_dqn"])   # last occurrence wins in argparse
    main()
