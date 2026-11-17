# -*- coding: utf-8 -*-
"""Aggregate evaluation summaries over training seeds (mean and std across trained models).

Run from the repository root:
    python -m evaluation.aggregate_seeds
    python -m evaluation.aggregate_seeds --algorithms q_learning dqn double_dqn

Reads  results/<algo>/seed_<S>/evaluation_summary.csv   (or the legacy results/<algo>/ file)
Writes results/comparison/multi_seed_summary.csv

The std columns are the standard deviation ACROSS trained models (ddof=1). They are empty when only
one trained model exists. This is the number to put next to a mean in a report: it contains the
randomness of training, which a single model's per-episode std does not.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.paths import ALL_ALGORITHMS, ROOT, run_dirs

METRICS = ("mean_reward", "safe_pass_rate", "collision_rate", "red_violation_rate",
           "timeout_rate", "mean_pass_time_s", "mean_speed_m_s", "mean_stopped_steps",
           "mean_hard_brake_steps")
DENSITY_ORDER = ("low", "medium", "high", "extreme")
ROUTE_ORDER = ("straight", "left", "right")


def load_summaries(results_root, algorithms, train_seed=None):
    """Concatenate every evaluation_summary.csv found; adds algorithm and train_seed columns."""
    frames = []
    for algo in algorithms:
        for seed, folder in run_dirs(results_root, algo, train_seed, ("evaluation_summary.csv",)):
            frame = pd.read_csv(folder / "evaluation_summary.csv", encoding="utf-8-sig")
            frame["algorithm"] = algo
            frame["train_seed"] = np.nan if seed is None else float(seed)
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def aggregate(summary):
    """Mean and across-model std of every metric per (algorithm, density, ego_route)."""
    if summary.empty:
        return pd.DataFrame()
    rows = []
    for (algo, density, route), group in summary.groupby(["algorithm", "density", "ego_route"], sort=False):
        row = {"algorithm": algo, "density": density, "ego_route": route,
               "n_models": int(len(group)),
               "episodes_per_model": int(group["episodes"].iloc[0]) if "episodes" in group else np.nan}
        for metric in METRICS:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            row[metric] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["_d"] = frame["density"].map({d: i for i, d in enumerate(DENSITY_ORDER)}).fillna(99)
    frame["_r"] = frame["ego_route"].map({r: i for i, r in enumerate(ROUTE_ORDER)}).fillna(99)
    frame["_a"] = frame["algorithm"].map({a: i for i, a in enumerate(ALL_ALGORITHMS)}).fillna(99)
    frame = frame.sort_values(["_r", "_d", "_a"]).drop(columns=["_d", "_r", "_a"])
    return frame.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algorithms", nargs="+", default=list(ALL_ALGORITHMS),
                        choices=list(ALL_ALGORITHMS))
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "comparison")
    args = parser.parse_args()

    summary = load_summaries(args.results_dir, args.algorithms)
    if summary.empty:
        raise SystemExit("没有找到任何 evaluation_summary.csv，请先完成评估。")
    table = aggregate(summary)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "multi_seed_summary.csv"
    table.to_csv(path, index=False, encoding="utf-8-sig")
    shown = [c for c in ("algorithm", "density", "ego_route", "n_models", "mean_reward",
                         "mean_reward_std", "mean_pass_time_s", "mean_pass_time_s_std",
                         "safe_pass_rate", "collision_rate") if c in table.columns]
    print(table[shown].round(3).to_string(index=False))
    print(f"\n已写入：{path}")


if __name__ == "__main__":
    main()
