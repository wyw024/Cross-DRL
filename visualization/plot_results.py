# -*- coding: utf-8 -*-
"""Create publication/report figures from saved evaluation and training CSV files."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 后端：不弹窗，直接保存图片
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS = ("rule_based", "q_learning", "dqn", "double_dqn")
DENSITIES = ("low", "medium", "high")


def read_data(results_dir, algorithms):
    summaries, episodes, training = [], [], []
    for algo in algorithms:
        folder = results_dir / algo
        sp, ep, tp = folder / "evaluation_summary.csv", folder / "evaluation_episodes.csv", folder / "training_log.csv"
        if sp.exists():
            frame = pd.read_csv(sp)
            frame["algorithm"] = algo
            summaries.append(frame)
        if ep.exists():
            frame = pd.read_csv(ep)
            frame["algorithm"] = algo
            episodes.append(frame)
        if tp.exists():
            frame = pd.read_csv(tp)
            frame["algorithm"] = algo
            training.append(frame)
    if not summaries or not episodes:
        raise FileNotFoundError("缺少 evaluation_summary.csv 或 evaluation_episodes.csv，请先完成评估。")
    return pd.concat(summaries, ignore_index=True), pd.concat(episodes, ignore_index=True), (
        pd.concat(training, ignore_index=True) if training else pd.DataFrame()
    )


def plot_summary(summary, out_dir, route):
    data = summary[summary["ego_route"] == route].copy()
    metrics = [
        ("mean_reward", "Mean episode reward", "Reward"),
        ("mean_pass_time_s", "Mean pass time", "Time (s)"),
        ("mean_stopped_steps", "Mean stopped steps", "Steps"),
        ("mean_hard_brake_steps", "Mean hard-brake steps", "Steps"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (metric, title, ylabel) in zip(axes.flat, metrics):
        if metric not in data.columns:
            ax.set_visible(False)
            continue
        pivot = data.pivot(index="density", columns="algorithm", values=metric)
        pivot = pivot.reindex([d for d in DENSITIES if d in pivot.index])
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Traffic density")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"Algorithm comparison — route: {route}")
    fig.savefig(out_dir / f"summary_metrics_{route}.png", dpi=220)
    plt.close(fig)

    rate_metrics = [
        ("safe_pass_rate", "Safe pass rate"),
        ("collision_rate", "Collision rate"),
        ("red_violation_rate", "Red-light violation rate"),
        ("timeout_rate", "Timeout rate"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, (metric, title) in zip(axes.flat, rate_metrics):
        if metric not in data.columns:
            ax.set_visible(False)
            continue
        pivot = data.pivot(index="density", columns="algorithm", values=metric)
        pivot = pivot.reindex([d for d in DENSITIES if d in pivot.index])
        pivot.plot(kind="bar", ax=ax, ylim=(0, 1))
        ax.set_title(title)
        ax.set_xlabel("Traffic density")
        ax.set_ylabel("Rate")
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"Safety and outcome rates — route: {route}")
    fig.savefig(out_dir / f"safety_rates_{route}.png", dpi=220)
    plt.close(fig)


def plot_episode_distributions(episodes, out_dir, route):
    data = episodes[episodes["ego_route"] == route]
    for metric, title, ylabel in (
        ("reward", "Episode reward distribution", "Reward"),
        ("pass_time_s", "Successful pass-time distribution", "Time (s)"),
    ):
        if metric not in data.columns:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
        for ax, density in zip(axes, DENSITIES):
            subset = data[data["density"] == density]
            labels, values = [], []
            for algo in ALGORITHMS:
                vals = pd.to_numeric(
                    subset.loc[subset["algorithm"] == algo, metric], errors="coerce"
                ).dropna().to_numpy()
                if len(vals):
                    labels.append(algo)
                    values.append(vals)
            if values:
                ax.boxplot(values, showmeans=True)
                ax.set_xticks(range(1, len(labels) + 1), labels=labels)
            ax.set_title(density)
            ax.set_ylabel(ylabel)
            ax.tick_params(axis="x", rotation=25)
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle(title + f" — route: {route}")
        fig.savefig(out_dir / f"{metric}_boxplot_{route}.png", dpi=220)
        plt.close(fig)


def plot_action_distribution(summary, out_dir, route):
    cols = [f"action_{i}_ratio" for i in range(5)]
    if not set(cols).issubset(summary.columns):
        return
    data = summary[summary["ego_route"] == route].copy()
    for density in DENSITIES:
        subset = data[data["density"] == density].set_index("algorithm")
        if subset.empty:
            continue
        ax = subset[cols].plot(kind="bar", stacked=True, figsize=(9, 5))
        ax.set_title(f"Action distribution — {density} — {route}")
        ax.set_xlabel("Algorithm")
        ax.set_ylabel("Action ratio")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=15)
        ax.legend([f"Action {i}" for i in range(5)], title="Action")
        ax.grid(axis="y", alpha=0.2)
        ax.figure.tight_layout()
        ax.figure.savefig(out_dir / f"action_distribution_{density}_{route}.png", dpi=220)
        plt.close(ax.figure)


def plot_training(training, out_dir):
    if training.empty or "episode" not in training.columns or "reward" not in training.columns:
        print("[INFO] 未发现可绘制的训练日志。")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for algo, frame in training.groupby("algorithm"):
        frame = frame.sort_values("episode").copy()
        reward = pd.to_numeric(frame["reward"], errors="coerce")
        frame["reward_smooth"] = reward.rolling(20, min_periods=1).mean()
        ax.plot(frame["episode"], frame["reward_smooth"], label=algo)
    ax.set_title("Training reward (20-episode moving average)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Moving-average reward")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_reward_curves.png", dpi=220)
    plt.close(fig)

    if "epsilon" in training.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        for algo, frame in training.groupby("algorithm"):
            frame = frame.sort_values("episode")
            if frame["epsilon"].notna().any():
                ax.plot(frame["episode"], frame["epsilon"], label=algo)
        ax.set_title("Exploration rate during training")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Epsilon")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "training_epsilon_curves.png", dpi=220)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS), choices=list(ALGORITHMS))
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "figures")
    parser.add_argument("--route", choices=("straight", "left", "right", "all"), default="straight")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    summary, episodes, training = read_data(args.results_dir, args.algorithms)
    routes = sorted(summary["ego_route"].dropna().unique()) if args.route == "all" else [args.route]
    for route in routes:
        plot_summary(summary, args.out, route)
        plot_episode_distributions(episodes, args.out, route)
        plot_action_distribution(summary, args.out, route)
    plot_training(training, args.out)
    print(f"图表已保存到：{args.out}")


if __name__ == "__main__":
    main()
