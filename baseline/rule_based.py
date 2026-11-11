# -*- coding: utf-8 -*-
"""
Rule-based driving policy for SUMO intersection environment.

本模块实现一个基于人工驾驶规则的控制器，不需要训练模型。

输入：
    observation: 环境返回的 7 维状态向量
    info:        环境返回的辅助信息（可选）

输出：
    action:      0~4 之间的离散动作编号

动作对应关系：
    0 -> -3.0 m/s^2，强制动
    1 -> -1.0 m/s^2，轻制动
    2 ->  0.0 m/s^2，保持速度
    3 -> +1.0 m/s^2，轻微加速
    4 -> +2.0 m/s^2，强加速
"""

import numpy as np


class RuleBasedAgent:
    """
    基于人工规则的自动驾驶控制器。

    控制优先级：
        1. 前车安全：避免追尾
        2. 红灯控制：尽可能在停止线前停车
        3. 黄灯控制：根据停车距离决定停车或通过
        4. 正常行驶：根据目标速度加速、减速或保持速度

    使用方式：
        agent = RuleBasedAgent()
        action = agent.select_action(observation, info)
    """

    # ---------------------------------------------------------
    # 1. 动作定义
    # ---------------------------------------------------------

    ACTION_HARD_BRAKE = 0
    ACTION_SOFT_BRAKE = 1
    ACTION_KEEP_SPEED = 2
    ACTION_SOFT_ACCELERATE = 3
    ACTION_HARD_ACCELERATE = 4

    # ---------------------------------------------------------
    # 2. 初始化规则参数
    # ---------------------------------------------------------

    def __init__(
        self,
        target_speed_m_s=10.0,
        speed_limit_m_s=13.89,
        comfortable_decel_m_s2=1.5,
        hard_decel_m_s2=3.0,
        stop_margin_m=3.0,
        hard_brake_gap_m=5.0,
        soft_brake_gap_m=12.0,
        hard_brake_ttc_s=1.0,
        soft_brake_ttc_s=2.0,
    ):
        """
        初始化 Rule-based 控制器。

        参数说明
        ----------
        target_speed_m_s:
            正常行驶时希望达到的目标速度，单位 m/s。

        speed_limit_m_s:
            速度上限，单位 m/s。
            ≈60km/h

        comfortable_decel_m_s2:
            舒适制动减速度的绝对值，单位 m/s^2。
            例如 1.5 表示以 -1.5 m/s^2 减速。

        hard_decel_m_s2:
            强制动减速度的绝对值，单位 m/s^2。

        stop_margin_m:
            停止线前预留的安全距离，单位 m。

        hard_brake_gap_m:
            与前车距离小于该值时，优先强制动。

        soft_brake_gap_m:
            与前车距离小于该值时，优先轻制动。

        hard_brake_ttc_s:
            碰撞时间小于该值时，优先强制动。

        soft_brake_ttc_s:
            碰撞时间小于该值时，优先轻制动。
        """

        self.target_speed_m_s = float(target_speed_m_s)
        self.speed_limit_m_s = float(speed_limit_m_s)

        self.comfortable_decel_m_s2 = float(comfortable_decel_m_s2)
        self.hard_decel_m_s2 = float(hard_decel_m_s2)

        self.stop_margin_m = float(stop_margin_m)

        self.hard_brake_gap_m = float(hard_brake_gap_m)
        self.soft_brake_gap_m = float(soft_brake_gap_m)

        self.hard_brake_ttc_s = float(hard_brake_ttc_s)
        self.soft_brake_ttc_s = float(soft_brake_ttc_s)

        # 记录控制器执行次数，方便调试和统计
        self.action_count = 0

    # ---------------------------------------------------------
    # 3. 从状态向量和 info 中读取车辆信息
    # ---------------------------------------------------------

    @staticmethod
    def _read_number(info, key, default=None):
        """
        安全读取 info 字典中的数值。

        如果字段不存在、为 None 或者不是有效数字，
        就返回 default。
        """

        if not isinstance(info, dict):
            return default

        value = info.get(key, default)

        if value is None:
            return default

        try:
            value = float(value)
        except (TypeError, ValueError):
            return default

        if not np.isfinite(value):
            return default

        return value

    def _parse_observation(self, observation, info=None):
        """
        解析环境的 7 维 observation。

        observation 的排列顺序：
            0 speed
            1 remaining
            2 signal
            3 gap
            4 leader_speed
            5 relative_speed
            6 light_present

        返回：
            包含实际物理量的字典。
        """

        obs = np.asarray(observation, dtype=np.float32)

        if obs.shape != (7,):
            raise ValueError(
                "RuleBasedAgent 需要 7 维 observation，"
                f"当前输入维度为 {obs.shape}"
            )

        if not np.all(np.isfinite(obs)):
            raise ValueError("observation 中包含 NaN 或 Inf")

        # 按照环境定义的范围进行裁剪
        obs = np.clip(
            obs,
            [0, 0, 0, 0, 0, -1, 0],
            [1, 1, 1, 1, 1, 1, 1]
        )

        # 将归一化状态还原为实际物理量
        speed_m_s = float(obs[0] * 13.89)

        distance_to_signal_m = float(obs[1] * 250.0)

        signal_value = float(obs[2])

        leader_gap_m = float(obs[3] * 100.0)

        leader_speed_m_s = float(obs[4] * 13.89)

        relative_speed_m_s = float(obs[5] * 13.89)

        light_present = bool(obs[6] >= 0.5)

        # 优先使用环境 info 中的实际物理量。
        # 如果 info 没有对应字段，则使用 observation 还原的数值。
        speed_m_s = self._read_number(
            info, "speed_m_s", speed_m_s
        )

        distance_to_signal_m = self._read_number(
            info, "distance_to_signal_m", distance_to_signal_m
        )

        leader_gap_from_info = self._read_number(
            info, "leader_gap_m", None
        )

        leader_speed_m_s = self._read_number(
            info, "leader_speed_m_s", leader_speed_m_s
        )

        # light_present 以 observation 为基础。
        # 如果 info 中提供了该字段，则优先使用 info。
        if isinstance(info, dict) and "light_present" in info:
            light_present = bool(info["light_present"])

        # 判断前方是否存在车辆。
        #
        # 如果 info 中有 leader_id，则以 leader_id 为准。
        # 如果没有，则根据前车距离进行近似判断。
        if isinstance(info, dict) and "leader_id" in info:
            has_leader = info["leader_id"] is not None
        elif leader_gap_from_info is not None:
            leader_gap_m = leader_gap_from_info
            has_leader = leader_gap_m < 99.5
        else:
            has_leader = leader_gap_m < 99.0

        # 前车距离优先采用 info 中的真实距离
        if leader_gap_from_info is not None:
            leader_gap_m = leader_gap_from_info

        return {
            "speed_m_s": max(0.0, speed_m_s),
            "distance_to_signal_m": max(
                0.0, distance_to_signal_m
            ),
            "signal_value": signal_value,
            "leader_gap_m": max(0.0, leader_gap_m),
            "leader_speed_m_s": max(
                0.0, leader_speed_m_s
            ),
            "relative_speed_m_s": relative_speed_m_s,
            "light_present": light_present,
            "has_leader": has_leader,
        }

    # ---------------------------------------------------------
    # 4. 判断前车风险
    # ---------------------------------------------------------

    def _leader_safety_action(self, state):
        """
        根据前车距离和相对速度，判断是否需要减速。

        TTC = 前车距离 / 接近速度

        其中：
            接近速度 = 自车速度 - 前车速度

        只有自车比前车快时，才存在追尾接近趋势。

        返回：
            如果需要制动，返回动作编号；
            如果没有明显追尾风险，返回 None。
        """

        if not state["has_leader"]:
            return None

        gap = state["leader_gap_m"]
        ego_speed = state["speed_m_s"]
        leader_speed = state["leader_speed_m_s"]

        closing_speed = ego_speed - leader_speed

        # 距离已经非常近，直接强制动
        if gap <= self.hard_brake_gap_m:
            return self.ACTION_HARD_BRAKE

        # 自车正在接近前车，计算 TTC
        if closing_speed > 0.1:
            ttc = gap / closing_speed

            if ttc <= self.hard_brake_ttc_s:
                return self.ACTION_HARD_BRAKE

            if ttc <= self.soft_brake_ttc_s:
                return self.ACTION_SOFT_BRAKE

        # 前车距离较近，即使 TTC 尚未达到阈值，
        # 也先轻微制动，避免持续逼近。
        if gap <= self.soft_brake_gap_m:
            return self.ACTION_SOFT_BRAKE

        return None

    # ---------------------------------------------------------
    # 5. 计算停车距离
    # ---------------------------------------------------------

    @staticmethod
    def _stopping_distance(speed_m_s, deceleration_m_s2):
        """
        根据运动学公式计算理论制动距离。

        v^2 = 2 * a * s

        因此：
            s = v^2 / (2 * a)

        speed_m_s:
            当前速度，单位 m/s。

        deceleration_m_s2:
            减速度的绝对值，单位 m/s^2。
        """

        if speed_m_s <= 0:
            return 0.0

        if deceleration_m_s2 <= 0:
            return float("inf")

        return (
            speed_m_s ** 2
            / (2.0 * deceleration_m_s2)
        )

    # ---------------------------------------------------------
    # 6. 红灯控制
    # ---------------------------------------------------------

    def _red_light_action(self, state):
        """
        红灯时的驾驶规则。

        基本原则：
            1. 根据当前速度计算停车距离。
            2. 考虑停止线前的安全预留距离。
            3. 根据实际剩余距离决定轻制动或强制动。

        注意：
            这是基于简化运动学公式的启发式控制，
            不能保证在所有情况下都能安全停车。
        """

        speed = state["speed_m_s"]
        distance = state["distance_to_signal_m"]

        # 已经接近停止线时，优先强制动
        if distance <= self.stop_margin_m:
            return self.ACTION_HARD_BRAKE

        stopping_distance = self._stopping_distance(
            speed,
            self.comfortable_decel_m_s2
        )

        required_distance = stopping_distance + self.stop_margin_m

        # 距离不足以舒适停车，使用强制动
        if distance <= required_distance:
            return self.ACTION_HARD_BRAKE

        # 距离足够，先轻制动
        return self.ACTION_SOFT_BRAKE

    # ---------------------------------------------------------
    # 7. 黄灯控制
    # ---------------------------------------------------------

    def _yellow_light_action(self, state):
        """
        黄灯时的驾驶规则。

        黄灯策略：
            如果距离足够，能够舒适停车，则停车。
            如果已经进入难以安全停车的区域，则保持速度。

        这里的“保持速度”只是一个简化策略，
        并不代表任何情况下都应该通过路口。
        """

        speed = state["speed_m_s"]
        distance = state["distance_to_signal_m"]

        stopping_distance = self._stopping_distance(
            speed,
            self.comfortable_decel_m_s2
        )

        required_distance = stopping_distance + self.stop_margin_m

        if distance > required_distance:
            return self.ACTION_SOFT_BRAKE

        # 已经太接近停止线，不建议突然大幅制动
        return self.ACTION_KEEP_SPEED

    # ---------------------------------------------------------
    # 8. 正常行驶控制
    # ---------------------------------------------------------

    def _cruise_action(self, state):
        """
        没有需要处理的交通灯或前车风险时，
        根据目标速度控制车辆。

        控制逻辑：
            速度明显低于目标速度：加速
            速度略低于目标速度：轻微加速
            速度达到目标速度：保持速度
            速度接近上限：轻微制动
        """

        speed = state["speed_m_s"]

        # 超过速度上限，轻微制动
        if speed >= self.speed_limit_m_s:
            return self.ACTION_SOFT_BRAKE

        # 明显低于目标速度，适当加速
        if speed < self.target_speed_m_s - 2.0:
            return self.ACTION_HARD_ACCELERATE

        # 略低于目标速度，轻微加速
        if speed < self.target_speed_m_s - 0.5:
            return self.ACTION_SOFT_ACCELERATE

        # 达到目标速度，保持速度
        return self.ACTION_KEEP_SPEED

    # ---------------------------------------------------------
    # 9. 主决策函数
    # ---------------------------------------------------------

    def select_action(
        self,
        observation,
        info=None,
        training=False
    ):
        """
        根据当前环境状态选择动作。

        参数：
            observation:
                环境返回的 7 维状态向量。

            info:
                环境返回的辅助信息，可选。

            training:
                为了兼容 QLearningAgent 的调用接口而保留。
                Rule-based 不需要训练，因此不使用该参数。

        返回：
            action:
                0~4 之间的整数。
        """

        self.action_count += 1

        state = self._parse_observation(
            observation,
            info
        )

        # =====================================================
        # 优先级 1：前车安全
        # =====================================================

        leader_action = self._leader_safety_action(state)

        if leader_action is not None:
            return int(leader_action)

        # =====================================================
        # 优先级 2：交通灯控制
        # =====================================================

        if state["light_present"]:

            signal = state["signal_value"]

            # 红灯
            if signal < 0.25:
                return int(
                    self._red_light_action(state)
                )

            # 黄灯
            elif signal < 0.75:
                return int(
                    self._yellow_light_action(state)
                )

            # 绿灯
            else:
                return int(
                    self._cruise_action(state)
                )

        # =====================================================
        # 优先级 3：没有交通灯，正常行驶
        # =====================================================

        return int(
            self._cruise_action(state)
        )

    # ---------------------------------------------------------
    # 10. 重置控制器
    # ---------------------------------------------------------

    def reset(self):
        """
        重置控制器内部统计信息。

        每次开始新的 episode 时可以调用。
        """

        self.action_count = 0