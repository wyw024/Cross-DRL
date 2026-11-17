# -*- coding: utf-8 -*-
"""One-command experiment pipeline: train -> evaluate -> check -> compare -> plot.

Run from the repository root (SUMO_HOME must be set):
    python run_experiments.py --quick                      # smoke test of the whole pipeline (a few minutes)
    python run_experiments.py --seeds 42 43 44 --jobs 3    # full experiment, 3 trainings in parallel
    python run_experiments.py --stages eval report         # re-evaluate and re-plot existing models
    python run_experiments.py --dry-run                    # only print the commands

Every step is an ordinary `python -m ...` command, so each can also be run by hand (see README).
"""

import argparse
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LEARNING = ("q_learning", "dqn", "double_dqn")

def module_for(algo):
    return "training.train_q_learning" if algo == "q_learning" else "training.train_dqn"

def build_commands(args):
    seeds = [str(s) for s in args.seeds]
    train, evaluate, report = [], [], []
    # train
    for algo in LEARNING:
        for seed in seeds:
            cmd = [module_for(algo)]
            if algo != "q_learning":
                cmd += ["--algorithm", algo]
            cmd += ["--seed", seed, "--episodes", str(args.episodes), "--density", args.train_density,
                    "--routes", *args.train_routes, "--epsilon-reach", str(args.epsilon_reach),
                    "--epsilon-decay", str(args.epsilon_decay)]
            cmd += shlex.split(args.train_args)  # 允许用户额外指定训练参数
            train.append(cmd)
    # evaluate
    common = ["--episodes", str(args.eval_episodes), "--density", args.eval_density,
              "--route", "all", "--seed", str(args.test_seed)]
    evaluate.append(["evaluation.evaluate_rule_based", *common])
    for algo in LEARNING:
        for seed in seeds:
            module = "evaluation.evaluate_q_learning" if algo == "q_learning" else "evaluation.evaluate_dqn"
            cmd = [module] + ([] if algo == "q_learning" else ["--algorithm", algo])
            evaluate.append(cmd + ["--train-seed", seed, *common])
    # report
    densities = ["low", "medium", "high"] + (["extreme"] if args.eval_density == "all+extreme" else [])
    report.append(["evaluation.check_results", "--train-seed", seeds[0]])
    report.append(["evaluation.aggregate_seeds"])
    report.append(["evaluation.compare_algorithms", "--train-seed", seeds[0], "--density", *densities,
                   "--route", "straight", "left", "right"])
    report.append(["visualization.plot_results"])
    return {"train": train, "eval": evaluate, "report": report}

def run_command(cmd, dry_run):  # 执行一条 Python 模块命令
    shown = "python -m " + " ".join(cmd)
    print(f"\n>>> {shown}", flush=True)
    if dry_run:
        return 0
    return subprocess.run([sys.executable, "-m", *cmd], cwd=ROOT).returncode

def run_stage(name, commands, jobs, dry_run):
    print(f"\n==================== stage: {name} ({len(commands)} commands) ====================")
    if jobs > 1 and not dry_run and len(commands) > 1:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            codes = list(pool.map(lambda c: run_command(c, dry_run), commands))
    else:
        codes = [run_command(c, dry_run) for c in commands]
    failed = [c for c, code in zip(commands, codes) if code != 0]
    if failed:
        print("\n以下命令失败：")
        for c in failed:
            print("  python -m " + " ".join(c))
        raise SystemExit(1)

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", nargs="+", choices=("train", "eval", "report"),
                        default=["train", "eval", "report"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                        help="training seeds (independent runs per algorithm)")
    parser.add_argument("--episodes", type=int, default=600, help="training episodes per run")
    parser.add_argument("--train-density", default="mixed",
                        choices=("low", "medium", "high", "extreme", "mixed"))
    parser.add_argument("--train-routes", nargs="+", default=["straight", "left", "right"],
                        choices=("straight", "left", "right"))
    parser.add_argument("--epsilon-decay", default="auto")
    parser.add_argument("--epsilon-reach", type=float, default=0.7)
    parser.add_argument("--train-args", default="",
                        help='extra arguments for every training command, e.g. "--learning-rate 5e-4"')
    parser.add_argument("--eval-episodes", type=int, default=50, help="episodes per (density, route)")
    parser.add_argument("--eval-density", default="all",
                        choices=("low", "medium", "high", "extreme", "all", "all+extreme"))
    parser.add_argument("--test-seed", type=int, default=1000)
    parser.add_argument("--jobs", type=int, default=1,
                        help="commands run in parallel inside train/eval stages (each uses one SUMO process)")
    parser.add_argument("--quick", action="store_true",
                        help="smoke test: 2 seeds, 30 training episodes, 5 evaluation episodes")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.seeds, args.episodes, args.eval_episodes = [42, 43], 30, 5
    plan = build_commands(args)
    for stage in ("train", "eval", "report"):
        if stage in args.stages:
            run_stage(stage, plan[stage], args.jobs if stage != "report" else 1, args.dry_run)
    print("\n全部完成。图表：results/figures/  对比表：results/comparison/")

if __name__ == "__main__":
    main()
