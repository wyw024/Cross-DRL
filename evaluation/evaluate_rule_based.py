# -*- coding: utf-8 -*-
"""
Evaluate Rule-based driving policy in SUMO.

功能：
1. 在 SUMO 环境中批量测试 Rule-based Agent。
2. 支持 low / medium / high 三种交通流密度。
3. 记录每个 episode 的运行结果。
4. 计算总体评价指标。
5. 将结果保存为 CSV 文件。

运行示例：

1. 测试 medium 密度，运行 50 个 episode：
   python -m scripts.evaluate_rule_based --episodes 50 --density medium

2. 测试全部三种交通流密度：
   python -m scripts.evaluate_rule_based --episodes 50 --density all

3. 使用 SUMO GUI 观察运行过程：
   python -m scripts.evaluate_rule_based --episodes 3 --density medium --gui

4. 修改最大仿真步数：
   python -m scripts.evaluate_rule_based --episodes 50 --max-steps 500

输出：
    results/rule_based/
        rule_based_episodes.csv
        rule_based_summary.csv
"""

import argparse
import csv
import math
import os
from datetime import datetime

import numpy as np

from environment.sumo_env import IntersectionEnv
from baseline.rule_based import RuleBasedAgent


# =============================================================
# 1. 基本配置
# =============================================================

DEFAULT_EPISODES = 50

DEFAULT_MAX_STEPS = 300

DEFAULT_SEED = 42

DEFAULT_DENSITIES = ["low", "medium", "high"]

# 当前环境每执行一次 step，SUMO 前进 1 秒。
# 如果以后修改 SUMO 的 step-length，需要同步修改此参数。
DEFAULT_STEP_SECONDS = 1.0

# 输出目录
DEFAULT_OUTPUT_DIR = os.path.join(
    "results",
    "rule_based"
)


# =============================================================
# 2. 工具函数：读取 info 中的布尔状态
# =============================================================

def extract_flag(info, candidate_keys):
    """
    从 info 字典中读取某项事件状态。

    参数
    ----------
    info:
        SUMO 环境 step() 或 reset() 返回的 info 字典。

    candidate_keys:
        可能表示同一事件的字段名称列表。

    返回
    ----------
    found:
        是否找到了对应字段。

    value:
        True / False / None。

    说明：
        有些环境会将 red_light_violation 设置为 None，
        表示无法确定是否闯红灯。
        此时不能将 None 当成 False。
    """

    if not isinstance(info, dict):
        return False, None

    for key in candidate_keys:

        if key not in info:
            continue

        value = info[key]

        if value is None:
            return True, None

        # 布尔值
        if isinstance(value, (bool, np.bool_)):
            return True, bool(value)

        # 数值型事件计数
        if isinstance(value, (int, float, np.integer, np.floating)):
            if not np.isfinite(value):
                return True, None

            return True, bool(value > 0)

        # 字符串型状态
        if isinstance(value, str):
            normalized = value.strip().lower()

            if normalized in ("true", "yes", "1", "success"):
                return True, True

            if normalized in ("false", "no", "0", "none", ""):
                return True, False

    return False, None


# =============================================================
# 3. 工具函数：读取车辆速度
# =============================================================

def extract_speed(info, observation):
    """
    优先从 info 中读取真实速度。

    如果 info 没有速度字段，则根据 observation[0] 还原。

    环境定义：
        observation[0] = speed / 13.89
    """

    if isinstance(info, dict):

        for key in ("speed_m_s", "speed"):

            value = info.get(key)

            if value is None:
                continue

            try:
                value = float(value)

                if np.isfinite(value) and value >= 0:
                    return value

            except (TypeError, ValueError):
                pass

    obs = np.asarray(observation, dtype=np.float32)

    if obs.shape == (7,) and np.isfinite(obs[0]):
        return max(0.0, float(obs[0]) * 13.89)

    return None


# =============================================================
# 4. 单个 episode 的评估
# =============================================================

def evaluate_one_episode(
    env,
    agent,
    episode,
    density,
    seed,
    step_seconds=DEFAULT_STEP_SECONDS
):
    """
    执行并评估一个 episode。

    一个 episode 的流程：

        env.reset()
            |
            v
        获取 observation 和 info
            |
            v
        RuleBasedAgent.select_action()
            |
            v
        env.step(action)
            |
            v
        记录奖励、速度和安全事件
            |
            v
        terminated 或 truncated？
            |
            +---- 否：继续
            |
            +---- 是：计算 episode 指标

    返回：
        一个字典，包含该 episode 的评价结果。
    """

    # 每个 episode 开始前重置控制器统计量
    agent.reset()

    # 重置 SUMO 环境
    observation, info = env.reset(seed=seed)

    total_reward = 0.0

    episode_steps = 0

    speed_records = []

    action_records = []

    # 事件状态：
    # None 表示整个 episode 中都没有获得有效记录。
    passed = None
    collision = None
    red_light_violation = None

    # 首次成功通过路口的时间
    pass_time_s = None

    # 记录每项指标是否曾经从环境中获得过
    passed_found = False
    collision_found = False
    red_violation_found = False

    terminated = False
    truncated = False

    red_violation_seen_true = False
    red_violation_seen_unknown = False
    red_violation_seen_false = False

    while not (terminated or truncated):

        # -----------------------------------------------------
        # 4.1 根据当前状态选择动作
        # -----------------------------------------------------

        action = agent.select_action(
            observation,
            info
        )

        action_records.append(int(action))

        # -----------------------------------------------------
        # 4.2 执行动作
        # -----------------------------------------------------

        (
            next_observation,
            reward,
            terminated,
            truncated,
            next_info
        ) = env.step(action)

        total_reward += float(reward)

        episode_steps += 1

        # -----------------------------------------------------
        # 4.3 记录车辆速度
        # -----------------------------------------------------

        speed = extract_speed(
            next_info,
            next_observation
        )

        if speed is not None:
            speed_records.append(speed)

        # -----------------------------------------------------
        # 4.4 记录路口通行情况
        # -----------------------------------------------------

        if isinstance(next_info, dict):

            if "passed_intersection" in next_info:

                value = next_info["passed_intersection"]

                if value is not None:

                    passed_found = True

                    if bool(value):

                        # 一旦成功通过，就保持 True
                        passed = True

                        # 只记录第一次成功通过的时间
                        if pass_time_s is None:
                            pass_time_s = episode_steps * step_seconds

                    elif passed is None:

                        # 环境明确返回 False，表示当前尚未成功通过
                        passed = False

            else:

                # 兼容其他环境字段
                found, value = extract_flag(
                    next_info,
                    (
                        "passed",
                        "successful_passage",
                        "success",
                        "crossed"
                    )
                )

                if found:

                    passed_found = True

                    if value is not None:

                        if bool(value):

                            passed = True

                            if pass_time_s is None:
                                pass_time_s = episode_steps * step_seconds

                        elif passed is None:

                            passed = False
        
        # -----------------------------------------------------
        # 4.5 记录碰撞情况
        # -----------------------------------------------------

        found, value = extract_flag(
            next_info,
            (
                "collision",
                "collided",
                "has_collision",
                "collision_occurred"
            )
        )

        if found:
            collision_found = True

            if value is not None:
                collision = bool(
                    (collision is True) or value
                )

        # -----------------------------------------------------
        # 4.6 记录闯红灯情况
        # -----------------------------------------------------

        found, value = extract_flag(
            next_info,
            (
                "red_light_violation",
                "red_light_violated",
                "ran_red_light"
            )
        )

        if found:
            red_violation_found = True

            if value is True:
                red_violation_seen_true = True

            elif value is None:
                red_violation_seen_unknown = True

            elif value is False:
                red_violation_seen_false = True

        # -----------------------------------------------------
        # 4.7 更新状态，进入下一步
        # -----------------------------------------------------

        observation = next_observation
        info = next_info

    # =========================================================
    # 确定整个 episode 的闯红灯状态
    # =========================================================

    if red_violation_seen_true:
        red_light_violation = True

    elif red_violation_seen_unknown:
        red_light_violation = None

    elif red_violation_seen_false:
        red_light_violation = False

    else:
        red_light_violation = None

    # =========================================================
    # 5. episode 结束，计算统计指标
    # =========================================================

    mean_speed = (
        float(np.mean(speed_records))
        if speed_records
        else float("nan")
    )

    max_speed = (
        float(np.max(speed_records))
        if speed_records
        else float("nan")
    )

    min_speed = (
        float(np.min(speed_records))
        if speed_records
        else float("nan")
    )

    # 假设每一步对应 step_seconds 秒
    simulation_time_s = episode_steps * step_seconds

    # 统计各动作使用次数
    action_counts = {
        f"action_{i}_count": action_records.count(i)
        for i in range(5)
    }

    # 计算动作比例
    action_ratios = {
        f"action_{i}_ratio": (
            action_records.count(i) / episode_steps
            if episode_steps > 0
            else 0.0
        )
        for i in range(5)
    }

    result = {
        "episode": episode,
        "seed": seed,
        "density": density,

        "total_reward": total_reward,
        "episode_steps": episode_steps,
        "simulation_time_s": simulation_time_s,

        # 首次通过路口的时间
        "pass_time_s": (
            pass_time_s
            if passed is True
            else None
        ),

        "mean_speed_m_s": mean_speed,
        "max_speed_m_s": max_speed,
        "min_speed_m_s": min_speed,

        "passed": passed,
        "passed_observed": passed_found,

        "collision": collision,
        "collision_observed": collision_found,

        "red_light_violation": red_light_violation,
        "red_violation_observed": red_violation_found,

        "terminated": bool(terminated),
        "truncated": bool(truncated),

        "action_count": len(action_records),
    }

    result.update(action_counts)
    result.update(action_ratios)

    return result


# =============================================================
# 6. 计算总体评价指标
# =============================================================

def calculate_summary(episode_results, density):
    """
    根据多个 episode 的结果，计算总体指标。

    主要指标：
        mean_reward
        mean_steps
        mean_speed_m_s
        mean_travel_time_s
        pass_rate
        collision_rate
        red_light_violation_rate

    对于环境没有提供的事件字段，不会假装其发生次数为 0，
    而是将对应统计结果设为 NaN。
    """

    results = [
        item
        for item in episode_results
        if item["density"] == density
    ]

    if not results:
        return {}

    rewards = np.array(
        [item["total_reward"] for item in results],
        dtype=np.float64
    )

    steps = np.array(
        [item["episode_steps"] for item in results],
        dtype=np.float64
    )

    speeds = np.array(
        [item["mean_speed_m_s"] for item in results],
        dtype=np.float64
    )

    # Episode 持续时间
    episode_times = np.array(
        [item["simulation_time_s"] for item in results],
        dtype=np.float64
    )

    # 只统计成功通过路口的 episode
    pass_times = [
        item["pass_time_s"]
        for item in results
        if item["passed"] is True
        and item["pass_time_s"] is not None
    ]

    # 只统计环境明确提供的事件数据
    passed_values = [
        item["passed"]
        for item in results
        if item["passed"] is not None
    ]

    collision_values = [
        item["collision"]
        for item in results
        if item["collision"] is not None
    ]

    red_values = [
        item["red_light_violation"]
        for item in results
        if item["red_light_violation"] is not None
    ]

    def safe_mean(values):
        values = np.asarray(values, dtype=np.float64)

        if len(values) == 0:
            return float("nan")

        return float(np.mean(values))

    def safe_rate(values):
        if len(values) == 0:
            return float("nan")

        return float(np.mean(values))

    summary = {
        "density": density,

        "episodes": len(results),

        "mean_reward": safe_mean(rewards),
        "std_reward": (
            float(np.std(rewards, ddof=1))
            if len(rewards) > 1
            else 0.0
        ),

        "mean_steps": safe_mean(steps),
        "mean_speed_m_s": safe_mean(speeds),
        "mean_episode_time_s": safe_mean(episode_times),

        "mean_pass_time_s": safe_mean(pass_times),

        "pass_time_observed_episodes": len(pass_times),

        "pass_rate": safe_rate(passed_values),
        "pass_observed_episodes": len(passed_values),

        "collision_rate": safe_rate(collision_values),
        "collision_observed_episodes": len(collision_values),

        "red_light_violation_rate": safe_rate(red_values),
        "red_violation_observed_episodes": len(red_values),

        "total_collisions": (
            int(sum(collision_values))
            if collision_values
            else None
        ),

        "total_red_light_violations": (
            int(sum(red_values))
            if red_values
            else None
        ),
    }

    return summary


# =============================================================
# 7. 保存 episode 结果
# =============================================================

def save_episode_results(results, output_path):
    """
    将每个 episode 的详细结果保存为 CSV。
    """

    if not results:
        print("没有 episode 结果，跳过保存。")
        return

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    fieldnames = list(results[0].keys())

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nEpisode 结果已保存：{output_path}")


# =============================================================
# 8. 保存总体汇总
# =============================================================

def save_summary_results(summaries, output_path):
    """
    将不同交通流密度的总体指标保存为 CSV。
    """

    if not summaries:
        print("没有总体汇总结果，跳过保存。")
        return

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    fieldnames = list(summaries[0].keys())

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(summaries)

    print(f"总体汇总已保存：{output_path}")


# =============================================================
# 9. 打印实验结果
# =============================================================

def print_summary(summaries):
    """
    在终端输出不同交通流密度下的评价指标。
    """

    print("\n")
    print("=" * 85)
    print("                 RULE-BASED EVALUATION SUMMARY")
    print("=" * 85)

    for item in summaries:

        print(f"\n交通流密度：{item['density']}")

        print("-" * 60)

        print(f"测试 episode 数量：{item['episodes']}")

        print(
            f"平均累计奖励："
            f"{item['mean_reward']:.3f}"
        )

        print(
            f"平均仿真步数："
            f"{item['mean_steps']:.2f}"
        )

        print(
            f"平均速度："
            f"{item['mean_speed_m_s']:.3f} m/s"
        )

        print(
            f"平均 Episode 时间："
            f"{item['mean_episode_time_s']:.2f} s"
        )

        if not math.isnan(item["mean_pass_time_s"]):
            print(
                f"平均路口通过时间："
                f"{item['mean_pass_time_s']:.2f} s "
                f"({item['pass_time_observed_episodes']} 个成功通过的 episode)"
            )
        else:
            print("平均路口通过时间：没有成功通过的有效 episode")

        if not math.isnan(item["pass_rate"]):
            print(
                f"成功通过率："
                f"{item['pass_rate'] * 100:.2f}% "
                f"({item['pass_observed_episodes']} 个有效 episode)"
            )
        else:
            print("成功通过率：环境未提供可用的通过状态")

        if not math.isnan(item["collision_rate"]):
            print(
                f"碰撞率："
                f"{item['collision_rate'] * 100:.2f}% "
                f"({item['collision_observed_episodes']} 个有效 episode)"
            )
        else:
            print("碰撞率：环境未提供可用的碰撞状态")

        if not math.isnan(item["red_light_violation_rate"]):
            print(
                f"闯红灯率："
                f"{item['red_light_violation_rate'] * 100:.2f}% "
                f"({item['red_violation_observed_episodes']} 个有效 episode)"
            )
        else:
            print("闯红灯率：环境未提供可用的闯红灯状态")

    print("\n" + "=" * 85)


# =============================================================
# 10. 命令行参数
# =============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Evaluate Rule-based Agent in SUMO."
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
        help="每种交通流密度测试的 episode 数量。"
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="每个 episode 最大仿真步数。"
    )

    parser.add_argument(
        "--density",
        type=str,
        default="all",
        choices=["low", "medium", "high", "all"],
        help="交通流密度。"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="随机种子起始值。"
    )

    parser.add_argument(
        "--step-seconds",
        type=float,
        default=DEFAULT_STEP_SECONDS,
        help="每个仿真 step 对应的秒数。"
    )

    parser.add_argument(
        "--gui",
        action="store_true",
        help="使用 SUMO GUI 显示车辆运行过程。"
    )

    parser.add_argument(
        "--gui-delay",
        type=int,
        default=200,
        help="SUMO GUI 每步延迟，单位 ms。"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="结果输出目录。"
    )

    return parser.parse_args()


# =============================================================
# 11. 主函数
# =============================================================

def main():

    args = parse_args()

    if args.episodes <= 0:
        raise ValueError("--episodes 必须大于 0")

    if args.max_steps <= 0:
        raise ValueError("--max-steps 必须大于 0")

    if args.step_seconds <= 0:
        raise ValueError("--step-seconds 必须大于 0")

    # 确定需要测试的交通流密度
    if args.density == "all":
        densities = DEFAULT_DENSITIES
    else:
        densities = [args.density]

    print("\n")
    print("=" * 85)
    print("                  RULE-BASED EVALUATION")
    print("=" * 85)

    print(f"测试 episode 数量：{args.episodes}")
    print(f"最大仿真步数：{args.max_steps}")
    print(f"交通流密度：{densities}")
    print(f"随机种子：{args.seed}")
    print(f"是否使用 GUI：{args.gui}")
    print(f"结果目录：{args.output_dir}")

    # 保存全部 episode 结果
    all_episode_results = []

    # 保存全部密度的总体汇总
    all_summaries = []

    # =========================================================
    # 逐个测试交通流密度
    # =========================================================

    for density in densities:

        print("\n")
        print("#" * 85)
        print(f"开始测试交通流密度：{density}")
        print("#" * 85)

        # 创建 SUMO 环境
        env = IntersectionEnv(
            render_mode="human" if args.gui else None,
            max_steps=args.max_steps,
            traffic_density=density,
            gui_delay_ms=args.gui_delay,
        )

        # 创建 Rule-based Agent
        agent = RuleBasedAgent()

        try:

            for episode in range(1, args.episodes + 1):

                # 每个 episode 使用不同随机种子
                episode_seed = args.seed + episode - 1

                print(
                    f"\n[{density}] "
                    f"Episode {episode}/{args.episodes} "
                    f"(seed={episode_seed})"
                )

                result = evaluate_one_episode(
                    env=env,
                    agent=agent,
                    episode=episode,
                    density=density,
                    seed=episode_seed,
                    step_seconds=args.step_seconds
                )

                all_episode_results.append(result)

                # 实时输出当前 episode 结果
                print(
                    f"Reward={result['total_reward']:.2f} | "
                    f"Steps={result['episode_steps']} | "
                    f"MeanSpeed={result['mean_speed_m_s']:.2f} m/s | "
                    f"Passed={result['passed']} | "
                    f"Collision={result['collision']} | "
                    f"RedViolation={result['red_light_violation']}"
                )

        finally:
            # 无论正常结束还是发生异常，都关闭 SUMO
            env.close()

        # 当前交通流密度的总体指标
        summary = calculate_summary(
            all_episode_results,
            density
        )

        all_summaries.append(summary)

    # # =========================================================
    # # 保存结果
    # # =========================================================

    # os.makedirs(
    #     args.output_dir,
    #     exist_ok=True
    # )

    # # 生成时间标记，防止覆盖以前的实验结果
    # timestamp = datetime.now().strftime(
    #     "%Y%m%d_%H%M%S"
    # )

    # episode_path = os.path.join(
    #     args.output_dir,
    #     f"rule_based_episodes_{timestamp}.csv"
    # )

    # summary_path = os.path.join(
    #     args.output_dir,
    #     f"rule_based_summary_{timestamp}.csv"
    # )

    # save_episode_results(
    #     all_episode_results,
    #     episode_path
    # )

    # save_summary_results(
    #     all_summaries,
    #     summary_path
    # )

    # =========================================================
    # 打印最终结果
    # =========================================================

    print_summary(all_summaries)

    print("\nRule-based 评估全部完成！")


if __name__ == "__main__":
    main()