"""Shared evaluation protocol for rule-based, Q-learning, DQN and Double DQN agents.

Every evaluate_*.py script calls run_evaluation(), so all algorithms are scored with
the same test seeds, the same per-episode fields and the same summary columns.

Metric definitions
------------------
outcome            collision > red_violation > teleported > passed > timeout > other
safe_pass_rate     share of episodes whose outcome is "passed" (reached the exit with no
                   collision, no red-light violation and no teleport)
collision_rate     share of episodes with a collision at any step
red_violation_rate share of episodes where the environment confirmed a red-light violation
timeout_rate       share of episodes truncated at max_steps without passing
pass_time_s        seconds until passed_intersection first became True (passed episodes only)
stopped_steps      steps with speed < 0.1 m/s
hard_brake_steps   steps with actual acceleration <= -3 m/s^2 (the threshold the reward uses)
min_leader_gap_m   smallest same-lane leader gap seen by the ego car
Rates carry a 95% Wilson interval, which stays sensible for small episode counts.

v2 changes
----------
* Results are MERGED into the existing CSV files by (algorithm, density, ego_route) instead of
  overwriting them, so evaluating --route left no longer deletes the straight results.
  Use --overwrite to restore the old "replace the whole file" behaviour.
* --density accepts extreme (2.0x traffic, a stress test) and all+extreme.
* --train-seed S / --all-seeds select models trained with seed S (results/<algo>/seed_<S>/).
"""

import csv
import math
from pathlib import Path

import numpy as np

from evaluation.paths import discover_train_seeds, result_dir

ROOT = Path(__file__).resolve().parents[1]

DENSITIES = ("low", "medium", "high")
STRESS_DENSITIES = ("extreme",)          # 2.0x background traffic, not part of "all"
DENSITY_ORDER = DENSITIES + STRESS_DENSITIES
ROUTES = ("straight", "left", "right")

DEFAULT_EPISODES = 50
DEFAULT_MAX_STEPS = 300
# Test seeds start far above any training seed (training uses seed + episode, ~43..800),
# and are identical for every algorithm, so results are directly comparable.
DEFAULT_TEST_SEED = 1000

ACTION_COUNT = 5
STOP_SPEED_M_S = 0.1
HARD_BRAKE_M_S2 = -3.0

EPISODE_FIELDS = [
    "algorithm", "density", "ego_route", "episode", "seed", "sumo_seed",
    "reward", "steps", "sim_time_s", "outcome",
    "passed", "collision", "red_light_violation", "teleported", "truncated",
    "signal_uncertain", "pass_time_s", "mean_speed_m_s", "stopped_steps",
    "hard_brake_steps", "min_leader_gap_m",
] + [f"action_{i}_ratio" for i in range(ACTION_COUNT)]

SUMMARY_FIELDS = [
    "algorithm", "density", "ego_route", "episodes",
    "mean_reward", "std_reward", "mean_steps",
    "safe_pass_rate", "safe_pass_ci95_low", "safe_pass_ci95_high",
    "collision_rate", "collision_ci95_low", "collision_ci95_high",
    "red_violation_rate", "red_violation_ci95_low", "red_violation_ci95_high",
    "timeout_rate", "teleport_rate", "signal_uncertain_rate",
    "mean_pass_time_s", "pass_time_episodes",
    "mean_speed_m_s", "mean_stopped_steps", "mean_hard_brake_steps", "min_leader_gap_m",
] + [f"action_{i}_ratio" for i in range(ACTION_COUNT)]


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def add_common_args(parser):
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                        help="episodes per (density, route) combination")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--density", choices=[*DENSITY_ORDER, "all", "all+extreme"], default="all",
                        help="all = low/medium/high; all+extreme additionally runs the 2.0x stress density")
    parser.add_argument("--route", choices=[*ROUTES, "all"], default="straight",
                        help="ego route; 'all' evaluates straight, left and right")
    parser.add_argument("--seed", type=int, default=DEFAULT_TEST_SEED,
                        help="first test seed; episode i uses seed + i for every algorithm")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--gui-delay", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="default: results/<algorithm>/ (or .../seed_<S>/ with --train-seed)")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace the CSV files instead of merging new (density, route) rows into them")
    return parser


def add_train_seed_args(parser):
    """Only for learning algorithms (rule_based has nothing to select)."""
    parser.add_argument("--train-seed", type=int, default=None,
                        help="evaluate the model trained with this seed (results/<algo>/seed_<S>/)")
    parser.add_argument("--all-seeds", action="store_true",
                        help="evaluate every trained seed that has a seed_<S> directory")
    return parser


def selected_train_seeds(args, algorithm):
    """[None] = legacy single model; otherwise the list of training seeds to evaluate."""
    if getattr(args, "all_seeds", False):
        seeds = discover_train_seeds(ROOT / "results", algorithm)
        if not seeds:
            raise FileNotFoundError(
                f"没有找到 results/{algorithm}/seed_<S>/ 目录；请先训练，或去掉 --all-seeds")
        return seeds
    if getattr(args, "train_seed", None) is not None:
        return [args.train_seed]
    return [None]


def densities_from_arg(value):
    if value == "all":
        return DENSITIES
    if value == "all+extreme":
        return DENSITIES + STRESS_DENSITIES
    return (value,)


def check_args(parser, args):
    if args.episodes <= 0 or args.max_steps <= 0:
        parser.error("--episodes and --max-steps must be positive")
    if getattr(args, "all_seeds", False) and args.output_dir is not None:
        parser.error("--output-dir cannot be combined with --all-seeds")


# ---------------------------------------------------------------------------
# one episode
# ---------------------------------------------------------------------------

def classify_outcome(collision, red_violation, teleported, passed, truncated):
    if collision:
        return "collision"
    if red_violation:
        return "red_violation"
    if teleported:
        return "teleported"
    if passed:
        return "passed"
    if truncated:
        return "timeout"
    return "other"


def run_episode(env, agent, algorithm, density, route, episode, seed, step_seconds):
    """Run one greedy (no exploration) episode and return a record with EPISODE_FIELDS."""
    if hasattr(agent, "reset"):
        agent.reset()

    observation, info = env.reset(seed=seed, options={"density": density, "ego_route": route})
    sumo_seed = info.get("sumo_seed")

    total_reward, steps = 0.0, 0
    passed = collision = red_violation = teleported = uncertain = False
    pass_step = None
    speeds, gaps = [], []
    stopped = hard_brakes = 0
    action_counts = [0] * ACTION_COUNT
    terminated = truncated = False

    while not (terminated or truncated):
        action = int(agent.select_action(observation, info, training=False))
        action_counts[action] += 1
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1

        if info.get("passed_intersection") and not passed:
            passed, pass_step = True, steps
        collision = collision or bool(info.get("collision"))
        red_violation = red_violation or info.get("red_light_violation") is True
        teleported = teleported or bool(info.get("teleported"))
        uncertain = uncertain or bool(info.get("signal_boundary_uncertain"))

        speed = info.get("speed_m_s")
        if speed is not None:
            speeds.append(float(speed))
            stopped += int(speed < STOP_SPEED_M_S)
        acceleration = info.get("actual_acceleration_m_s2")
        if acceleration is not None and acceleration <= HARD_BRAKE_M_S2:
            hard_brakes += 1
        gap = getattr(env, "current_leader_gap", None)   # not exposed through info
        if gap is not None:
            gaps.append(float(gap))

    record = {
        "algorithm": algorithm, "density": density, "ego_route": route,
        "episode": episode, "seed": seed, "sumo_seed": sumo_seed,
        "reward": round(total_reward, 4), "steps": steps,
        "sim_time_s": round(steps * step_seconds, 3),
        "outcome": classify_outcome(collision, red_violation, teleported, passed, truncated),
        "passed": int(passed), "collision": int(collision),
        "red_light_violation": int(red_violation), "teleported": int(teleported),
        "truncated": int(bool(truncated)), "signal_uncertain": int(uncertain),
        "pass_time_s": round(pass_step * step_seconds, 3) if pass_step else None,
        "mean_speed_m_s": round(float(np.mean(speeds)), 4) if speeds else None,
        "stopped_steps": stopped, "hard_brake_steps": hard_brakes,
        "min_leader_gap_m": round(min(gaps), 3) if gaps else None,
    }
    for i, count in enumerate(action_counts):
        record[f"action_{i}_ratio"] = round(count / steps, 4) if steps else 0.0
    return record


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def wilson_interval(successes, total, z=1.96):
    if total == 0:
        return None, None
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)


def _mean(values):
    values = [v for v in values if v is not None]
    return round(float(np.mean(values)), 4) if values else None


def summarize(records, algorithm, density, route):
    rows = [r for r in records if r["density"] == density and r["ego_route"] == route]
    n = len(rows)
    if n == 0:
        return None
    count = lambda key, value=1: sum(r[key] == value for r in rows)
    safe, collisions, reds = count("outcome", "passed"), count("collision"), count("red_light_violation")
    rewards = [r["reward"] for r in rows]
    gaps = [r["min_leader_gap_m"] for r in rows if r["min_leader_gap_m"] is not None]
    pass_times = [r["pass_time_s"] for r in rows if r["outcome"] == "passed"]

    summary = {
        "algorithm": algorithm, "density": density, "ego_route": route, "episodes": n,
        "mean_reward": round(float(np.mean(rewards)), 4),
        "std_reward": round(float(np.std(rewards, ddof=1)), 4) if n > 1 else 0.0,
        "mean_steps": round(float(np.mean([r["steps"] for r in rows])), 3),
        "safe_pass_rate": round(safe / n, 4),
        "collision_rate": round(collisions / n, 4),
        "red_violation_rate": round(reds / n, 4),
        "timeout_rate": round(count("outcome", "timeout") / n, 4),
        "teleport_rate": round(count("teleported") / n, 4),
        "signal_uncertain_rate": round(count("signal_uncertain") / n, 4),
        "mean_pass_time_s": _mean(pass_times), "pass_time_episodes": len(pass_times),
        "mean_speed_m_s": _mean([r["mean_speed_m_s"] for r in rows]),
        "mean_stopped_steps": _mean([r["stopped_steps"] for r in rows]),
        "mean_hard_brake_steps": _mean([r["hard_brake_steps"] for r in rows]),
        "min_leader_gap_m": round(min(gaps), 3) if gaps else None,
    }
    for name, hits in (("safe_pass", safe), ("collision", collisions), ("red_violation", reds)):
        summary[f"{name}_ci95_low"], summary[f"{name}_ci95_high"] = wilson_interval(hits, n)
    for i in range(ACTION_COUNT):
        summary[f"action_{i}_ratio"] = _mean([r[f"action_{i}_ratio"] for r in rows])
    return summary


def print_summary(summaries):
    pct = lambda v: "n/a" if v is None else f"{v:.1%}"
    num = lambda v, fmt="{:.2f}": "n/a" if v is None else fmt.format(v)
    print("\n" + "=" * 78)
    for s in summaries:
        print(f"[{s['algorithm']}] density={s['density']} route={s['ego_route']} "
              f"episodes={s['episodes']}")
        print(f"  平均奖励 {num(s['mean_reward'])} ± {num(s['std_reward'])}    "
              f"平均步数 {num(s['mean_steps'], '{:.1f}')}")
        print(f"  安全通过率 {pct(s['safe_pass_rate'])} "
              f"(95%CI {pct(s['safe_pass_ci95_low'])}~{pct(s['safe_pass_ci95_high'])})")
        print(f"  碰撞率 {pct(s['collision_rate'])}    闯红灯率 {pct(s['red_violation_rate'])}    "
              f"超时率 {pct(s['timeout_rate'])}    传送率 {pct(s['teleport_rate'])}")
        print(f"  平均通过用时 {num(s['mean_pass_time_s'])} s ({s['pass_time_episodes']} 回合)    "
              f"平均速度 {num(s['mean_speed_m_s'])} m/s")
        print(f"  停车步数 {num(s['mean_stopped_steps'], '{:.1f}')}    "
              f"急刹步数 {num(s['mean_hard_brake_steps'], '{:.1f}')}    "
              f"最小前车间距 {num(s['min_leader_gap_m'])} m")
    print("=" * 78)


def save_csv(rows, fieldnames, path):
    """Write rows to path, replacing the file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fieldnames})


MERGE_KEY = ("algorithm", "density", "ego_route")


def _sort_key(row):
    density = row.get("density")
    route = row.get("ego_route")
    try:
        episode = int(float(row.get("episode") or 0))
    except ValueError:
        episode = 0
    return (DENSITY_ORDER.index(density) if density in DENSITY_ORDER else 99,
            ROUTES.index(route) if route in ROUTES else 99, episode)


def merge_csv(rows, fieldnames, path, key=MERGE_KEY):
    """Insert `rows` into the CSV at `path`.

    Existing rows whose (algorithm, density, ego_route) also appear in `rows` are replaced;
    every other existing row is kept. Returns the number of rows that were kept from the old file.
    """
    path = Path(path)
    new_keys = {tuple(str(r.get(k)) for k in key) for r in rows}
    kept = []
    if path.is_file():
        with path.open("r", newline="", encoding="utf-8-sig") as file:
            for old in csv.DictReader(file):
                if tuple(str(old.get(k)) for k in key) not in new_keys:
                    kept.append(old)
    merged = sorted(kept + [dict(r) for r in rows], key=_sort_key)
    save_csv(merged, fieldnames, path)
    return len(kept)


# ---------------------------------------------------------------------------
# entry point shared by every evaluate_*.py
# ---------------------------------------------------------------------------

def run_evaluation(agent, algorithm, args, output_dir=None):
    # Imported here so that importing this module (e.g. in unit tests) needs no SUMO.
    from environment.sumo_env import IntersectionEnv, STEP_LENGTH

    densities = densities_from_arg(args.density)
    routes = ROUTES if args.route == "all" else (args.route,)
    if output_dir is None:
        output_dir = Path(args.output_dir) if args.output_dir else ROOT / "results" / algorithm
    output_dir = Path(output_dir)

    env = IntersectionEnv(render_mode="human" if args.gui else None,
                          max_steps=args.max_steps, traffic_density=densities[0],
                          gui_delay_ms=args.gui_delay)
    records = []
    try:
        for density in densities:
            for route in routes:
                for index in range(args.episodes):
                    record = run_episode(env, agent, algorithm, density, route,
                                         episode=index + 1, seed=args.seed + index,
                                         step_seconds=STEP_LENGTH)
                    records.append(record)
                    print(f"[{algorithm}|{density}|{route}] episode {index + 1}/{args.episodes} "
                          f"seed={record['seed']} outcome={record['outcome']} "
                          f"reward={record['reward']:.2f} steps={record['steps']}")
    finally:
        env.close()

    summaries = [s for d in densities for r in routes
                 if (s := summarize(records, algorithm, d, r)) is not None]
    episodes_path = output_dir / "evaluation_episodes.csv"
    summary_path = output_dir / "evaluation_summary.csv"
    if getattr(args, "overwrite", False):
        save_csv(records, EPISODE_FIELDS, episodes_path)
        save_csv(summaries, SUMMARY_FIELDS, summary_path)
        kept = 0
    else:
        merge_csv(records, EPISODE_FIELDS, episodes_path)
        kept = merge_csv(summaries, SUMMARY_FIELDS, summary_path)
    print_summary(summaries)
    print(f"逐回合结果：{episodes_path}")
    print(f"汇总结果：  {summary_path}")
    if kept:
        print(f"（已与文件中原有的其他 density/route 结果合并，保留了 {kept} 行汇总；"
              f"加 --overwrite 可整体覆盖）")
    return records, summaries


def run_evaluation_for_seeds(args, algorithm, load_agent):
    """Evaluate one model per selected training seed.

    load_agent(train_seed) -> agent. train_seed is None for the legacy single model.
    """
    for train_seed in selected_train_seeds(args, algorithm):
        agent = load_agent(train_seed)
        if args.output_dir:
            out = Path(args.output_dir)
        else:
            out = result_dir(ROOT / "results", algorithm, train_seed)
        label = "legacy model" if train_seed is None else f"train seed {train_seed}"
        print(f"\n===== {algorithm} | {label} -> {out} =====")
        run_evaluation(agent, algorithm, args, output_dir=out)
