# -*- coding: utf-8 -*-
"""Compare saved evaluation results for rule-based, Q-learning, DQN and Double DQN.

Run from repository root:
    python -m evaluation.compare_algorithms --route straight left right
    python -m evaluation.compare_algorithms --train-seed 42 --density low medium --route straight

With several trained seeds the paired tests compare ONE trained model per algorithm (they need the
same test episodes). Pick it with --train-seed (default: the smallest available seed). Use
evaluation.aggregate_seeds for the mean/std over all trained seeds.
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.paths import discover_train_seeds, result_dir

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALGORITHMS = ("rule_based", "q_learning", "dqn", "double_dqn")
DENSITIES = ("low", "medium", "high", "extreme")
ROUTES = ("straight", "left", "right")
SUMMARY_FILE = "evaluation_summary.csv"
EPISODE_FILE = "evaluation_episodes.csv"


def pick_folder(result_root, algorithm, train_seed):
    """One result directory per algorithm: explicit seed, else smallest seed_<S>, else legacy."""
    if algorithm == "rule_based":
        return result_dir(result_root, algorithm)
    if train_seed is not None:
        return result_dir(result_root, algorithm, train_seed)
    seeds = discover_train_seeds(result_root, algorithm)
    if seeds:
        if len(seeds) > 1:
            print(f"[INFO] {algorithm}: 有多个训练种子 {seeds}，配对检验使用 seed {seeds[0]}"
                  f"（用 --train-seed 指定其他种子）。")
        return result_dir(result_root, algorithm, seeds[0])
    return result_dir(result_root, algorithm)


def load_results(algorithms, result_root, train_seed=None):
    summaries, episodes = {}, {}
    for algorithm in algorithms:
        folder = pick_folder(result_root, algorithm, train_seed)
        summary_path, episode_path = folder / SUMMARY_FILE, folder / EPISODE_FILE
        if not summary_path.is_file() or not episode_path.is_file():
            print(f"[WARN] {algorithm}: 缺少评估 CSV，跳过。")
            continue
        summary = pd.read_csv(summary_path)
        episode = pd.read_csv(episode_path)
        summary["algorithm"] = algorithm
        episode["algorithm"] = algorithm
        summaries[algorithm], episodes[algorithm] = summary, episode
    if len(summaries) < 2:
        raise SystemExit("至少需要两个算法同时具备 summary 和 episodes 文件。")
    return summaries, episodes


def validate_results(episodes):
    """Check the common-seed design before performing paired tests."""
    required = {"density", "ego_route", "seed", "sumo_seed", "reward",
                "outcome", "passed", "pass_time_s"}
    for algorithm, frame in episodes.items():
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{algorithm} 缺少必要字段：{sorted(missing)}")
        if frame.duplicated(["density", "ego_route", "seed"]).any():
            raise ValueError(f"{algorithm} 存在重复的 density/route/seed 记录。")
    base = next(iter(episodes))
    for algorithm, frame in episodes.items():
        if algorithm == base:
            continue
        keys = ["density", "ego_route", "seed"]
        left = episodes[base][keys + ["sumo_seed"]]
        right = frame[keys + ["sumo_seed"]]
        merged = left.merge(right, on=keys, suffixes=("_base", "_other"), how="outer", indicator=True)
        if not (merged["_merge"] == "both").all():
            print(f"[WARN] {base} 与 {algorithm} 的测试种子集合不完全一致；配对检验只使用交集。")
        matched = merged[merged["_merge"] == "both"]
        if not (matched["sumo_seed_base"] == matched["sumo_seed_other"]).all():
            raise ValueError(f"{base} 与 {algorithm} 相同测试 seed 对应的 SUMO 场景不一致，不能进行配对比较。")


def _paired_test(x, y):
    """Two-sided paired Wilcoxon test; returns p=None when no informative pairs."""
    from scipy.stats import wilcoxon
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return len(x), None
    if np.allclose(x, y):
        return len(x), 1.0           # identical results: no evidence of a difference
    try:
        result = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox", method="auto")
        return len(x), float(result.pvalue)
    except ValueError:
        return len(x), None


def _paired_binary_test(a, b):
    """Exact McNemar/binomial test for paired binary success outcomes."""
    from scipy.stats import binomtest
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    discordant_a = int(np.sum((a == 1) & (b == 0)))
    discordant_b = int(np.sum((a == 0) & (b == 1)))
    n = discordant_a + discordant_b
    if n == 0:
        return len(a), discordant_a, discordant_b, 1.0   # both algorithms succeed/fail on the same seeds
    p = binomtest(min(discordant_a, discordant_b), n=n, p=0.5, alternative="two-sided").pvalue
    return len(a), discordant_a, discordant_b, float(p)


def holm_adjust(p_values):
    """Holm-Bonferroni adjusted p-values; None values remain None."""
    valid = [(i, p) for i, p in enumerate(p_values) if p is not None and np.isfinite(p)]
    adjusted = [None] * len(p_values)
    ordered = sorted(valid, key=lambda item: item[1])
    running = 0.0
    m = len(ordered)
    for rank, (index, p) in enumerate(ordered):
        value = min(1.0, (m - rank) * p)
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def build_pairwise_tests(episodes, algorithms, densities, routes,
                         min_reward_diff=0.3, min_time_diff=2.0):
    rows = []
    for algo_a, algo_b in itertools.combinations(algorithms, 2):
        a_all, b_all = episodes[algo_a], episodes[algo_b]
        for density in densities:
            for route in routes:
                keys = ["seed"]
                a = a_all[(a_all.density == density) & (a_all.ego_route == route)]
                b = b_all[(b_all.density == density) & (b_all.ego_route == route)]
                paired = a.merge(b, on=keys, suffixes=("_a", "_b"), how="inner")
                if paired.empty:
                    continue
                for metric in ("reward", "pass_time_s"):
                    xa = pd.to_numeric(paired[f"{metric}_a"], errors="coerce")
                    xb = pd.to_numeric(paired[f"{metric}_b"], errors="coerce")
                    n, p = _paired_test(xa, xb)
                    rows.append({
                        "metric": metric, "density": density, "ego_route": route,
                        "algo_a": algo_a, "algo_b": algo_b, "n_pairs": n,
                        "median_a": float(xa.median()) if xa.notna().any() else np.nan,
                        "median_b": float(xb.median()) if xb.notna().any() else np.nan,
                        "median_diff_a_minus_b": float((xa - xb).median()) if (xa - xb).notna().any() else np.nan,
                        "p_raw": p,
                    })
                n, discordant_a, discordant_b, p = _paired_binary_test(
                    paired["passed_a"].fillna(0), paired["passed_b"].fillna(0)
                )
                rows.append({
                    "metric": "safe_pass_binary", "density": density, "ego_route": route,
                    "algo_a": algo_a, "algo_b": algo_b, "n_pairs": n,
                    "discordant_a_success": discordant_a,
                    "discordant_b_success": discordant_b,
                    "p_raw": p,
                })
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["p_holm"] = holm_adjust(frame["p_raw"].tolist())
    frame["significant_0.05_holm"] = frame["p_holm"].apply(
        lambda p: bool(pd.notna(p) and p < 0.05)
    )

    def meaningful(row):
        """Significant AND big enough to matter (|median diff| above the chosen thresholds)."""
        if not row["significant_0.05_holm"]:
            return False
        if row["metric"] == "reward":
            return abs(row["median_diff_a_minus_b"]) >= min_reward_diff
        if row["metric"] == "pass_time_s":
            return abs(row["median_diff_a_minus_b"]) >= min_time_diff
        return True

    frame["practically_meaningful"] = frame.apply(meaningful, axis=1)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithms", nargs="+", default=list(DEFAULT_ALGORITHMS),
                        choices=list(DEFAULT_ALGORITHMS))
    parser.add_argument("--density", nargs="+", default=list(DENSITIES),
                        choices=list(DENSITIES))
    parser.add_argument("--route", nargs="+", default=["straight"],
                        choices=list(ROUTES), help="e.g. --route straight left right")
    parser.add_argument("--train-seed", type=int, default=None,
                        help="trained model to compare (default: smallest available seed)")
    parser.add_argument("--min-reward-diff", type=float, default=0.3,
                        help="smallest median reward difference counted as practically meaningful")
    parser.add_argument("--min-time-diff", type=float, default=2.0,
                        help="smallest median pass-time difference (s) counted as practically meaningful")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "comparison")
    args = parser.parse_args()

    summaries, episodes = load_results(args.algorithms, args.results_dir, args.train_seed)
    algorithms = [a for a in args.algorithms if a in summaries]
    validate_results(episodes)

    summary = pd.concat([summaries[a] for a in algorithms], ignore_index=True)
    summary = summary[
        summary["density"].isin(args.density) & summary["ego_route"].isin(args.route)
    ].copy()
    args.out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out / "comparison_summary.csv", index=False, encoding="utf-8-sig")

    present_routes = set(summary["ego_route"].unique()) if not summary.empty else set()
    for route in args.route:
        if route not in present_routes:
            print(f"[WARN] 评估结果中没有 route={route}；请先运行 "
                  f"python -m evaluation.evaluate_<algo> --route {route}（结果会合并保存，不会覆盖其他路线）。")
    filtered_episodes = {
        a: frame[frame["density"].isin(args.density) & frame["ego_route"].isin(args.route)].copy()
        for a, frame in episodes.items()
    }
    tests = build_pairwise_tests(filtered_episodes, algorithms, args.density, args.route,
                                 args.min_reward_diff, args.min_time_diff)
    if not tests.empty:
        tests.to_csv(args.out / "pairwise_tests.csv", index=False, encoding="utf-8-sig")
    elif (args.out / "pairwise_tests.csv").exists():
        print("[WARN] 本次没有可检验的配对数据，保留旧的 pairwise_tests.csv 不变。")

    print("\n=== 汇总结果 ===")
    cols = [c for c in ("algorithm", "density", "ego_route", "episodes", "mean_reward",
                        "safe_pass_rate", "collision_rate", "red_violation_rate",
                        "mean_pass_time_s", "mean_stopped_steps", "mean_hard_brake_steps",
                        "min_leader_gap_m") if c in summary.columns]
    print(summary[cols].to_string(index=False))
    if tests.empty:
        print("\n没有可进行的配对检验。")
    else:
        print(f"\n配对检验：{len(tests)} 项；Holm 校正后显著：{int(tests['significant_0.05_holm'].sum())} 项；"
              f"其中有实际意义（差异超过阈值）：{int(tests['practically_meaningful'].sum())} 项。")
        print(tests[["metric", "density", "ego_route", "algo_a", "algo_b", "n_pairs",
                     "median_diff_a_minus_b", "p_holm", "significant_0.05_holm",
                     "practically_meaningful"]].to_string(index=False))
        print("注意：pass_time_s 只对两算法均成功通过的配对回合进行检验。")
    print(f"\n输出目录：{args.out}")


if __name__ == "__main__":
    main()
