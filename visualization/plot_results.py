# -*- coding: utf-8 -*-
"""Create report figures from the saved evaluation and training CSV files.

Run from the repository root:
    python -m visualization.plot_results                    # every route found in the data
    python -m visualization.plot_results --route left
    python -m visualization.plot_results --train-seed 42    # one trained model instead of all seeds

If several training seeds exist, bars show the mean over trained models with the across-model std as
the error bar, and training curves show mean +/- std bands. Figures written to results/figures/:

  summary_metrics_<route>.png     reward, pass time, stopped steps, hard-brake steps
  safety_rates_<route>.png        safe pass rate (with 95% CI) and failure rate
  tradeoff_<route>.png            pass time vs. hard-brake steps (efficiency / comfort trade-off)
  reward_boxplot_<route>.png, pass_time_s_boxplot_<route>.png
  action_distribution_<density>_<route>.png
  route_generalization.png        straight / left / right side by side
  training_reward_curves.png, training_steps_curves.png, training_loss_curves.png,
  training_epsilon_curves.png
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no window, just save images
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.aggregate_seeds import aggregate
from evaluation.common import wilson_interval
from evaluation.paths import ALL_ALGORITHMS, ROOT, run_dirs

ALGORITHMS = ALL_ALGORITHMS
DENSITIES = ("low", "medium", "high", "extreme")
ROUTES = ("straight", "left", "right")

# Fixed colour per algorithm (categorical slots 1-3 of the validated reference palette, all-pairs
# safe; the non-learning baseline is a neutral grey). Colour follows the algorithm, never its rank.
COLORS = {"rule_based": "#6b6a66", "q_learning": "#2a78d6", "dqn": "#eb6834", "double_dqn": "#1baf7a"}
LABELS = {"rule_based": "Rule-based", "q_learning": "Q-learning", "dqn": "DQN", "double_dqn": "Double DQN"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
MARKERS = {"low": "o", "medium": "s", "high": "^", "extreme": "D"}
# Marker sizes differ per algorithm so that two algorithms with identical results still show up as
# concentric markers instead of one hiding the other.
MARKER_SIZE = {"rule_based": 8, "q_learning": 10, "dqn": 15, "double_dqn": 8}

plt.rcParams.update({
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "font.size": 10, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
})


# --------------------------------------------------------------------------- data
def read_data(results_dir, algorithms, train_seed=None):
    summaries, episodes, training = [], [], []
    for algo in algorithms:
        for seed, folder in run_dirs(results_dir, algo, train_seed,
                                     ("evaluation_summary.csv", "evaluation_episodes.csv")):
            for name, bucket in (("evaluation_summary.csv", summaries),
                                 ("evaluation_episodes.csv", episodes)):
                frame = pd.read_csv(folder / name, encoding="utf-8-sig")
                frame["algorithm"] = algo
                frame["train_seed"] = np.nan if seed is None else float(seed)
                bucket.append(frame)
        for seed, folder in run_dirs(results_dir, algo, train_seed, ("training_log.csv",)):
            if algo == "rule_based":
                continue
            frame = pd.read_csv(folder / "training_log.csv", encoding="utf-8-sig")
            frame["algorithm"] = algo
            frame["train_seed"] = np.nan if seed is None else float(seed)
            training.append(frame)
    if not summaries or not episodes:
        raise FileNotFoundError("缺少 evaluation_summary.csv 或 evaluation_episodes.csv，请先完成评估。")
    return (pd.concat(summaries, ignore_index=True), pd.concat(episodes, ignore_index=True),
            pd.concat(training, ignore_index=True) if training else pd.DataFrame())


def ordered(values, order):
    return [v for v in order if v in set(values)]


def algos_in(frame):
    return ordered(frame["algorithm"].unique(), ALGORITHMS)


def algo_legend(fig, algos):
    """One legend below the figure, never on top of the bars."""
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[a]) for a in algos]
    fig.legend(handles, [LABELS[a] for a in algos], loc="outside lower center",
               ncol=len(algos), fontsize=9)


def savefig(fig, path):
    fig.savefig(path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- bars
def grouped_bars(ax, table, metric, densities, algos, ylabel, ylim=None):
    """table = aggregate() output filtered to one route. Bars = mean, error bars = across-model std."""
    width = 0.8 / max(len(algos), 1)
    x = np.arange(len(densities))
    for i, algo in enumerate(algos):
        means, errs = [], []
        for density in densities:
            row = table[(table["algorithm"] == algo) & (table["density"] == density)]
            means.append(float(row[metric].iloc[0]) if len(row) and metric in row else np.nan)
            err = float(row[f"{metric}_std"].iloc[0]) if len(row) and f"{metric}_std" in row else np.nan
            errs.append(0.0 if np.isnan(err) else err)
        means_arr, errs_arr = np.asarray(means, dtype=float), np.asarray(errs, dtype=float)
        lower = np.minimum(errs_arr, np.nan_to_num(means_arr))      # a count/time never goes below 0
        ax.bar(x + (i - (len(algos) - 1) / 2) * width, means, width * 0.92, yerr=[lower, errs_arr],
               capsize=2.5, color=COLORS[algo], label=LABELS[algo],
               error_kw={"elinewidth": 1, "ecolor": MUTED})
    ax.set_xticks(x, densities)
    ax.set_xlabel("Traffic density")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    if ylim:
        ax.set_ylim(*ylim)


def plot_summary(summary, out_dir, route):
    data = summary[summary["ego_route"] == route]
    table = aggregate(data)
    densities = ordered(data["density"].unique(), DENSITIES)
    algos = algos_in(data)
    n_models = int(table["n_models"].max()) if len(table) else 1
    note = f" (mean ± std over {n_models} trained models)" if n_models > 1 else ""

    metrics = [("mean_reward", "Mean episode reward", "Reward"),
               ("mean_pass_time_s", "Mean pass time", "Time (s)"),
               ("mean_stopped_steps", "Mean stopped steps", "Steps"),
               ("mean_hard_brake_steps", "Mean hard-brake steps", "Steps")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (metric, title, ylabel) in zip(axes.flat, metrics):
        if metric not in table.columns or table[metric].dropna().empty:
            ax.set_visible(False)
            continue
        grouped_bars(ax, table, metric, densities, algos, ylabel)
        ax.set_title(title)
    algo_legend(fig, algos)
    fig.suptitle(f"Algorithm comparison — route: {route}{note}")
    savefig(fig, out_dir / f"summary_metrics_{route}.png")

    # Safety: the pass rate always gets a panel with its Wilson CI. Collision / red-light / timeout /
    # teleport are folded into ONE failure-rate panel; if they are all zero we say so instead of
    # drawing four empty charts.
    fail_cols = [c for c in ("collision_rate", "red_violation_rate", "timeout_rate", "teleport_rate")
                 if c in data.columns]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    width = 0.8 / max(len(algos), 1)
    x = np.arange(len(densities))
    pooled = data.groupby(["algorithm", "density"])
    for i, algo in enumerate(algos):
        vals, lows, highs = [], [], []
        for density in densities:
            if (algo, density) in pooled.groups:
                g = pooled.get_group((algo, density))
                total = int(g["episodes"].sum())
                hits = int(round(float((g["safe_pass_rate"] * g["episodes"]).sum())))
                lo, hi = wilson_interval(hits, total)       # pooled over trained models
                vals.append(hits / total)
                lows.append(lo)
                highs.append(hi)
            else:
                vals.append(np.nan); lows.append(np.nan); highs.append(np.nan)
        vals, lows, highs = map(np.asarray, (vals, lows, highs))
        pos = x + (i - (len(algos) - 1) / 2) * width
        ax1.bar(pos, vals, width * 0.92, color=COLORS[algo], label=LABELS[algo],
                yerr=[np.clip(vals - lows, 0, None), np.clip(highs - vals, 0, None)],
                capsize=2.5, error_kw={"elinewidth": 1, "ecolor": MUTED})
    ax1.set_xticks(x, densities)
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Safe pass rate (bars: rate, whiskers: 95% Wilson CI)")
    ax1.set_xlabel("Traffic density")
    ax1.set_ylabel("Rate")
    ax1.grid(axis="y", color=GRID, linewidth=0.7)
    ax1.set_axisbelow(True)
    algo_legend(fig, algos)

    fail = data.copy()
    fail["failure_rate"] = fail[fail_cols].fillna(0).sum(axis=1).clip(upper=1.0) if fail_cols else 0.0
    ftable = (fail.groupby(["algorithm", "density"])["failure_rate"].mean().reset_index())
    if float(ftable["failure_rate"].max() if len(ftable) else 0.0) == 0.0:
        ax2.axis("off")
        ax2.text(0.5, 0.55, "No failures in any evaluated episode\n"
                 "(collision, red-light violation, timeout, teleport = 0)\n\n"
                 "Safety is at its ceiling here, so algorithms cannot be\n"
                 "separated by safety. Compare efficiency (pass time,\n"
                 "stops, hard braking) or use the extreme-density stress test.",
                 ha="center", va="center", fontsize=10, color=MUTED, transform=ax2.transAxes)
    else:
        for i, algo in enumerate(algos):
            vals = [float(ftable[(ftable.algorithm == algo) & (ftable.density == d)]["failure_rate"].mean())
                    if len(ftable[(ftable.algorithm == algo) & (ftable.density == d)]) else np.nan
                    for d in densities]
            ax2.bar(x + (i - (len(algos) - 1) / 2) * width, vals, width * 0.92,
                    color=COLORS[algo], label=LABELS[algo])
        ax2.set_xticks(x, densities)
        ax2.set_title("Failure rate (collision + red light + timeout + teleport)")
        ax2.set_xlabel("Traffic density")
        ax2.set_ylabel("Rate")
        ax2.grid(axis="y", color=GRID, linewidth=0.7)
        ax2.set_axisbelow(True)
    fig.suptitle(f"Safety and outcome — route: {route}")
    savefig(fig, out_dir / f"safety_rates_{route}.png")


def plot_tradeoff(summary, out_dir, route):
    table = aggregate(summary[summary["ego_route"] == route])
    need = {"mean_pass_time_s", "mean_hard_brake_steps"}
    if table.empty or not need.issubset(table.columns):
        return
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    for algo in algos_in(table):
        sub = table[table["algorithm"] == algo]
        for _, row in sub.iterrows():
            ax.errorbar(row["mean_pass_time_s"], row["mean_hard_brake_steps"],
                        xerr=None if np.isnan(row.get("mean_pass_time_s_std", np.nan)) else row["mean_pass_time_s_std"],
                        yerr=None if np.isnan(row.get("mean_hard_brake_steps_std", np.nan)) else row["mean_hard_brake_steps_std"],
                        fmt=MARKERS.get(row["density"], "o"), color=COLORS[algo],
                        markersize=MARKER_SIZE[algo], markeredgecolor="#fcfcfb", markeredgewidth=1.5,
                        ecolor=GRID, capsize=2, zorder=5 - ALGORITHMS.index(algo) * 0.5)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=COLORS[a], label=LABELS[a])
               for a in algos_in(table)]
    handles += [plt.Line2D([], [], marker=MARKERS[d], linestyle="", color=MUTED, label=d)
                for d in ordered(table["density"].unique(), DENSITIES)]
    ax.legend(handles=handles, fontsize=8, ncol=2)
    ax.set_xlabel("Mean pass time (s)   ← faster")
    ax.set_ylabel("Mean hard-brake steps   ← smoother")
    ax.set_title(f"Efficiency vs. comfort — route: {route} (lower-left is better)")
    ax.grid(color=GRID, linewidth=0.7)
    savefig(fig, out_dir / f"tradeoff_{route}.png")


def plot_episode_distributions(episodes, out_dir, route):
    data = episodes[episodes["ego_route"] == route]
    densities = ordered(data["density"].unique(), DENSITIES)
    if not densities:
        return
    for metric, title, ylabel in (("reward", "Episode reward distribution", "Reward"),
                                  ("pass_time_s", "Successful pass-time distribution", "Time (s)")):
        if metric not in data.columns:
            continue
        fig, axes = plt.subplots(1, len(densities), figsize=(4.6 * len(densities), 4.5),
                                 constrained_layout=True, squeeze=False)
        for ax, density in zip(axes[0], densities):
            subset = data[data["density"] == density]
            labels, values = [], []
            for algo in ALGORITHMS:
                vals = pd.to_numeric(subset.loc[subset["algorithm"] == algo, metric],
                                     errors="coerce").dropna().to_numpy()
                if len(vals):
                    labels.append(algo)
                    values.append(vals)
            if values:
                box = ax.boxplot(values, showmeans=True, patch_artist=True,
                                 medianprops={"color": INK}, meanprops={"markerfacecolor": INK,
                                                                         "markeredgecolor": INK})
                for patch, algo in zip(box["boxes"], labels):
                    patch.set_facecolor(COLORS[algo]); patch.set_alpha(0.85)
                ax.set_xticks(range(1, len(labels) + 1), [LABELS[a] for a in labels])
            ax.set_title(density)
            ax.set_ylabel(ylabel)
            ax.tick_params(axis="x", rotation=25)
            ax.grid(axis="y", color=GRID, linewidth=0.7)
        fig.suptitle(f"{title} — route: {route} (pooled over trained models)")
        savefig(fig, out_dir / f"{metric}_boxplot_{route}.png")


def plot_action_distribution(summary, out_dir, route):
    cols = [f"action_{i}_ratio" for i in range(5)]
    if not set(cols).issubset(summary.columns):
        return
    data = summary[summary["ego_route"] == route]
    palette = ["#e34948", "#eda100", "#c3c2b7", "#1baf7a", "#2a78d6"]   # brake -> accelerate
    names = ["-3.0 m/s²", "-1.0 m/s²", "0.0 m/s²", "+1.0 m/s²", "+2.0 m/s²"]
    for density in ordered(data["density"].unique(), DENSITIES):
        subset = data[data["density"] == density].groupby("algorithm")[cols].mean()
        subset = subset.reindex(algos_in(data))
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        bottom = np.zeros(len(subset))
        for col, color, name in zip(cols, palette, names):
            ax.bar([LABELS[a] for a in subset.index], subset[col].to_numpy(), bottom=bottom,
                   color=color, edgecolor="#fcfcfb", linewidth=1.5, label=name)
            bottom += subset[col].to_numpy()
        ax.set_title(f"Action distribution — {density} — {route}")
        ax.set_ylabel("Action ratio")
        ax.set_ylim(0, 1)
        ax.legend(title="Requested acceleration", fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        savefig(fig, out_dir / f"action_distribution_{density}_{route}.png")


def plot_route_generalization(summary, out_dir, density):
    data = summary[summary["density"] == density]
    routes = ordered(data["ego_route"].unique(), ROUTES)
    if len(routes) < 2:
        print(f"[INFO] 只有一条路线的评估结果，跳过 route_generalization.png"
              f"（用 --route all 评估后再画）。")
        return
    table = aggregate(data)
    algos = algos_in(data)
    metrics = [("mean_reward", "Mean episode reward", "Reward"),
               ("mean_pass_time_s", "Mean pass time", "Time (s)"),
               ("mean_hard_brake_steps", "Mean hard-brake steps", "Steps")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    width = 0.8 / len(algos)
    x = np.arange(len(routes))
    for ax, (metric, title, ylabel) in zip(axes, metrics):
        for i, algo in enumerate(algos):
            means, errs = [], []
            for route in routes:
                row = table[(table.algorithm == algo) & (table.ego_route == route)]
                means.append(float(row[metric].iloc[0]) if len(row) else np.nan)
                e = float(row[f"{metric}_std"].iloc[0]) if len(row) else np.nan
                errs.append(0.0 if np.isnan(e) else e)
            means_arr, errs_arr = np.asarray(means, dtype=float), np.asarray(errs, dtype=float)
            lower = np.minimum(errs_arr, np.nan_to_num(means_arr))
            ax.bar(x + (i - (len(algos) - 1) / 2) * width, means, width * 0.92, yerr=[lower, errs_arr],
                   capsize=2.5, color=COLORS[algo], label=LABELS[algo],
                   error_kw={"elinewidth": 1, "ecolor": MUTED})
        ax.set_xticks(x, routes)
        ax.set_xlabel("Ego route")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
    algo_legend(fig, algos)
    fig.suptitle(f"Straight / left / right comparison — density: {density}")
    savefig(fig, out_dir / "route_generalization.png")


# --------------------------------------------------------------------------- training
def smooth(values, window):
    return pd.Series(pd.to_numeric(values, errors="coerce")).rolling(window, min_periods=1).mean().to_numpy()


def curve_with_band(ax, frames, column, window, color, label):
    """frames: list of per-seed DataFrames. Draws the mean curve and a ±std band if there are >= 2."""
    curves = []
    for frame in frames:
        frame = frame.sort_values("episode")
        if column in frame.columns and frame[column].notna().any():
            curves.append(pd.Series(smooth(frame[column], window), index=frame["episode"].to_numpy()))
    if not curves:
        return False
    stacked = pd.concat(curves, axis=1)
    mean = stacked.mean(axis=1)
    ax.plot(mean.index, mean.to_numpy(), color=color, label=label, linewidth=1.8)
    if stacked.shape[1] > 1:
        std = stacked.std(axis=1, ddof=1).fillna(0)
        ax.fill_between(mean.index, (mean - std).to_numpy(), (mean + std).to_numpy(),
                        color=color, alpha=0.18, linewidth=0)
    return True


def plot_training(training, out_dir, window=20):
    if training.empty or "episode" not in training.columns or "reward" not in training.columns:
        print("[INFO] 未发现可绘制的训练日志。")
        return
    algos = ordered(training["algorithm"].unique(), ALGORITHMS)
    per_algo = {a: [g for _, g in training[training.algorithm == a].groupby(
        training["train_seed"].fillna(-1))] for a in algos}
    n_seeds = max(len(v) for v in per_algo.values())
    band = f"mean ± std over {n_seeds} training seeds" if n_seeds > 1 else "single training seed"

    for column, title, ylabel, fname in (
            ("reward", f"Training reward ({window}-episode moving average, {band})", "Reward",
             "training_reward_curves.png"),
            ("steps", f"Episode length in steps ({window}-episode moving average, {band})", "Steps",
             "training_steps_curves.png"),
            ("loss", f"Training loss ({window}-episode moving average, {band})", "Smooth-L1 loss",
             "training_loss_curves.png")):
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        drawn = False
        for algo in algos:
            drawn |= curve_with_band(ax, per_algo[algo], column, window, COLORS[algo], LABELS[algo])
        if not drawn:
            plt.close(fig)
            continue
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(color=GRID, linewidth=0.7)
        ax.legend()
        if column == "steps":
            ax.text(0.99, 0.97, "Each extra step costs ≈0.05–0.06 reward, so shorter episodes score higher",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8, color=MUTED)
        savefig(fig, out_dir / fname)

    # epsilon: it is a hyper-parameter schedule, not a result. If every algorithm shares the same
    # schedule the curves coincide, so draw ONE line instead of three hidden ones.
    if "epsilon" in training.columns:
        curves = {}
        for algo in algos:
            first = per_algo[algo][0].sort_values("episode")
            if first["epsilon"].notna().any():
                curves[algo] = first.set_index("episode")["epsilon"]
        if curves:
            fig, ax = plt.subplots(figsize=(10, 4.6), constrained_layout=True)
            base_name = next(iter(curves))
            identical = all(len(c) == len(curves[base_name]) and
                            np.allclose(c.to_numpy(), curves[base_name].to_numpy(), atol=1e-4)
                            for c in curves.values())
            if identical:
                c = curves[base_name]
                ax.plot(c.index, c.to_numpy(), color=INK, linewidth=2,
                        label="All learning algorithms (identical schedule)")
            else:
                for algo, c in curves.items():
                    ax.plot(c.index, c.to_numpy(), color=COLORS[algo], linewidth=1.8, label=LABELS[algo])
            floor = min(float(c.min()) for c in curves.values())
            ax.axhline(floor, color=MUTED, linewidth=0.8, linestyle=":")
            ax.text(0.99, 0.5, f"final epsilon = {floor:.3f}", transform=ax.transAxes, ha="right",
                    fontsize=8, color=MUTED)
            ax.set_title("Exploration rate ε during training (probability of a random action)")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Epsilon")
            ax.grid(color=GRID, linewidth=0.7)
            ax.legend()
            savefig(fig, out_dir / "training_epsilon_curves.png")


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS), choices=list(ALGORITHMS))
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "figures")
    parser.add_argument("--route", choices=(*ROUTES, "all"), default="all",
                        help="route(s) to plot; 'all' = every route present in the data")
    parser.add_argument("--train-seed", type=int, default=None,
                        help="plot only the model trained with this seed (default: all seeds)")
    parser.add_argument("--route-density", default="medium",
                        help="density used in route_generalization.png")
    parser.add_argument("--smooth-window", type=int, default=20,
                        help="moving-average window (episodes) of the training curves")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    summary, episodes, training = read_data(args.results_dir, args.algorithms, args.train_seed)
    available = ordered(summary["ego_route"].dropna().unique(), ROUTES)
    if args.route == "all":
        routes = available
    elif args.route in available:
        routes = [args.route]
    else:
        raise SystemExit(
            f"评估结果中没有 route={args.route}（现有路线：{available or '无'}）。\n"
            f"请先评估：python -m evaluation.evaluate_<algo> --route {args.route}\n"
            f"（新版评估会把结果合并保存，不会覆盖其他路线。）")
    for route in routes:
        plot_summary(summary, args.out, route)
        plot_tradeoff(summary, args.out, route)
        plot_episode_distributions(episodes, args.out, route)
        plot_action_distribution(summary, args.out, route)
    plot_route_generalization(summary, args.out, args.route_density)
    plot_training(training, args.out, args.smooth_window)
    print(f"图表已保存到：{args.out}（路线：{routes}）")


if __name__ == "__main__":
    main()
