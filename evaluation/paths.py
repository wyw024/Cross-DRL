"""Shared helpers that decide where results of each algorithm / training seed live.

Layout (v2)
-----------
results/rule_based/                      non-learning baseline, always a single run
results/<algo>/seed_<S>/                 one directory per training seed S (q_learning, dqn, double_dqn)
results/<algo>/                          legacy single-run layout (still readable)
models/q_learning/seed_<S>/q_table.npz   Q-table of training seed S
results/<algo>/seed_<S>/checkpoint.pt    DQN / Double DQN checkpoint of training seed S
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEARNING_ALGORITHMS = ("q_learning", "dqn", "double_dqn")
ALL_ALGORITHMS = ("rule_based",) + LEARNING_ALGORITHMS

_SEED_DIR = re.compile(r"^seed_(\d+)$")


def result_dir(results_root, algorithm, train_seed=None):
    """Directory that holds training_log.csv / evaluation_*.csv of one run."""
    root = Path(results_root) / algorithm
    if algorithm == "rule_based" or train_seed is None:
        return root
    return root / f"seed_{int(train_seed)}"


def discover_train_seeds(results_root, algorithm):
    """Sorted training seeds that have a seed_<S> directory for this algorithm."""
    root = Path(results_root) / algorithm
    if algorithm == "rule_based" or not root.is_dir():
        return []
    seeds = []
    for child in root.iterdir():
        match = _SEED_DIR.match(child.name)
        if match and child.is_dir():
            seeds.append(int(match.group(1)))
    return sorted(seeds)


def run_dirs(results_root, algorithm, train_seed=None, need=("evaluation_summary.csv",)):
    """List of (train_seed | None, directory) that actually contain the files in `need`.

    train_seed given   -> only that seed (rule_based ignores it)
    train_seed None    -> every seed_<S> directory if there are any, otherwise the legacy
                          directory results/<algo>/
    """
    def complete(folder):
        return all((folder / name).is_file() for name in need)

    if algorithm == "rule_based":
        folder = result_dir(results_root, algorithm)
        return [(None, folder)] if complete(folder) else []
    if train_seed is not None:
        folder = result_dir(results_root, algorithm, train_seed)
        return [(int(train_seed), folder)] if complete(folder) else []
    found = [(s, result_dir(results_root, algorithm, s))
             for s in discover_train_seeds(results_root, algorithm)]
    found = [(s, f) for s, f in found if complete(f)]
    if found:
        return found
    legacy = result_dir(results_root, algorithm)
    return [(None, legacy)] if complete(legacy) else []


def q_table_path(train_seed=None, models_root=None):
    base = Path(models_root) if models_root else ROOT / "models" / "q_learning"
    if train_seed is None:
        return base / "q_table.npz"
    return base / f"seed_{int(train_seed)}" / "q_table.npz"
