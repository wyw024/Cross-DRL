# -*- coding: utf-8 -*-
"""Offline integrity checks for the saved four-algorithm evaluation results."""
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS = ("rule_based", "q_learning", "dqn", "double_dqn")
REQUIRED_EPISODE_COLUMNS = {
    "density", "ego_route", "episode", "seed", "sumo_seed", "reward", "outcome",
    "passed", "collision", "red_light_violation", "teleported", "truncated",
    "pass_time_s", "stopped_steps", "hard_brake_steps",
}
REQUIRED_SUMMARY_COLUMNS = {
    "density", "ego_route", "episodes", "mean_reward", "safe_pass_rate",
    "collision_rate", "red_violation_rate", "timeout_rate",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS), choices=list(ALGORITHMS))
    args = parser.parse_args()

    loaded = {}
    errors = []
    for algo in args.algorithms:
        folder = args.results_dir / algo
        ep_path, sum_path = folder / "evaluation_episodes.csv", folder / "evaluation_summary.csv"
        if not ep_path.exists() or not sum_path.exists():
            errors.append(f"{algo}: 缺少评估 CSV")
            continue
        ep, summary = pd.read_csv(ep_path), pd.read_csv(sum_path)
        missing_ep = REQUIRED_EPISODE_COLUMNS - set(ep.columns)
        missing_sum = REQUIRED_SUMMARY_COLUMNS - set(summary.columns)
        if missing_ep:
            errors.append(f"{algo}: episodes 缺少字段 {sorted(missing_ep)}")
        if missing_sum:
            errors.append(f"{algo}: summary 缺少字段 {sorted(missing_sum)}")
        if ep.duplicated(["density", "ego_route", "seed"]).any():
            errors.append(f"{algo}: 存在重复测试 seed")
        for col in ("safe_pass_rate", "collision_rate", "red_violation_rate", "timeout_rate"):
            if col in summary and not summary[col].dropna().between(0, 1).all():
                errors.append(f"{algo}: {col} 超出 [0, 1]")
        if {"density", "ego_route", "seed", "sumo_seed"}.issubset(ep.columns):
            loaded[algo] = ep
        print(f"{algo:12s} episodes={len(ep):4d} summary_rows={len(summary):2d} "
              f"densities={sorted(ep['density'].dropna().unique())} "
              f"routes={sorted(ep['ego_route'].dropna().unique())}")

    if len(loaded) >= 2:
        base_name = next(iter(loaded))
        base = loaded[base_name]
        keys = ["density", "ego_route", "seed"]
        for algo, frame in loaded.items():
            if algo == base_name:
                continue
            joined = base[keys + ["sumo_seed"]].merge(
                frame[keys + ["sumo_seed"]], on=keys, how="outer",
                suffixes=("_base", "_other"), indicator=True
            )
            if not (joined["_merge"] == "both").all():
                errors.append(f"{algo}: 与 {base_name} 的测试 seed 不一致")
            matched = joined[joined["_merge"] == "both"]
            if not (matched["sumo_seed_base"] == matched["sumo_seed_other"]).all():
                errors.append(f"{algo}: 与 {base_name} 的 SUMO 场景 seed 不一致")

    if errors:
        print("\n检查未通过：")
        for error in errors:
            print(" -", error)
        raise SystemExit(1)
    print("\n检查通过：文件结构、指标范围及跨算法测试场景一致性均符合要求。")


if __name__ == "__main__":
    main()
