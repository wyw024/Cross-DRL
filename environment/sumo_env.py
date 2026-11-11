
# -*- coding: utf-8 -*-
"""
SUMO Reinforcement Learning Environment
---------------------------------------
四岔路口自动驾驶车辆纵向控制环境。

功能：
1. 基于 Gymnasium 封装 SUMO / TraCI
2. 五种纵向加速度动作
3. 七维连续状态观测
4. 支持 low / medium / high 三种交通密度
5. reset 时随机化初始交通信号灯相位
6. 区分主车和背景交通车辆
7. 检测碰撞、传送、闯红灯和通过路口
8. 提供用于 Q-learning / DQN 的统一接口

状态空间：
0. speed              当前速度 / 最高速度
1. remaining_distance 剩余距离 / 250m
2. signal              红灯=0，黄灯=0.5，绿灯=1
3. gap                 前车距离 / 100m
4. leader_speed        前车速度 / 最高速度
5. relative_speed      相对速度 / 最高速度
6. light_present       是否存在前车（字段名沿用现有接口）

动作空间：
0: -3.0 m/s²
1: -1.0 m/s²
2:  0.0 m/s²
3: +1.0 m/s²
4: +2.0 m/s²

注意：
- 仿真步长默认为 1 秒。
- 主车从 north_in 进入，可预设直行、左转或右转；动作仍控制纵向加速度。
- 信号灯 TLS ID 默认为 junction。
"""

import os
import sys
import math
import random
import subprocess
import xml.etree.ElementTree as ET

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# ============================================================
# 1. SUMO / TraCI 环境配置
# ============================================================

if "SUMO_HOME" not in os.environ:
    raise EnvironmentError(
        "未检测到 SUMO_HOME 环境变量，请先配置 SUMO 环境。"
    )

SUMO_HOME = os.environ["SUMO_HOME"]

TOOLS = os.path.join(SUMO_HOME, "tools")

if TOOLS not in sys.path:
    sys.path.append(TOOLS)

import traci
import sumolib


# ============================================================
# 2. 全局参数
# ============================================================

# SUMO 配置文件
SUMO_CONFIG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "network",
    "intersection.sumocfg"
)

# 交通信号灯 ID
TLS_ID = "junction"

# 主车 ID
EGO_ID = "ego"

# 主车路线 ID 与目标出口（从北侧驶入）
EGO_ROUTES = {
    "straight": ("n_s", "south_out"),
    "left": ("n_e", "east_out"),
    "right": ("n_w", "west_out"),
}

# 主车车辆类型
EGO_TYPE_ID = "car"

# 仿真步长
STEP_LENGTH = 1.0

# 最大速度
MAX_SPEED = 13.89

# 路口前最大观测距离
MAX_REMAINING_DISTANCE = 250.0

# 前车观测距离
MAX_LEADER_DISTANCE = 100.0

# episode 最大步数
DEFAULT_MAX_STEPS = 300

# 交通密度对应的 SUMO scale
DENSITY_SCALE = {
    "low": 0.5,
    "medium": 1.0,
    "high": 1.5
}

# 五种动作对应的加速度
ACTION_ACCELERATIONS = {
    0: -3.0,
    1: -1.0,
    2: 0.0,
    3: 1.0,
    4: 2.0
}

# 信号灯相位持续时间
# 必须与 network.net.xml 中的 tlLogic 一致
SIGNAL_PHASE_DURATIONS = [42, 3, 42, 3]

SIGNAL_CYCLE_LENGTH = sum(SIGNAL_PHASE_DURATIONS)

# 信号灯相位类型
# 0: 南北绿灯
# 1: 南北黄灯
# 2: 东西绿灯
# 3: 东西黄灯


# ============================================================
# 3. Gymnasium 环境
# ============================================================

class IntersectionEnv(gym.Env):

    metadata = {
        "render_modes": ["human"],
        "render_fps": 1
    }

    def __init__(
        self,
        render_mode=None,
        max_steps=DEFAULT_MAX_STEPS,
        traffic_density="medium",
        gui_delay_ms=500,
        gui_quit_on_end=False,
        randomize_signal_phase=True,
        ego_route="straight"
    ):

        super().__init__()

        # ----------------------------------------------------
        # 基本参数
        # ----------------------------------------------------

        self.render_mode = render_mode
        self.max_steps = int(max_steps)

        self.traffic_density = traffic_density

        if ego_route not in EGO_ROUTES:
            raise ValueError(f"ego_route 必须是 {list(EGO_ROUTES)}")
        self.default_ego_route = ego_route
        self.ego_route = ego_route
        self.route_id, self.exit_edge = EGO_ROUTES[ego_route]

        self.gui_delay_ms = int(gui_delay_ms)

        self.gui_quit_on_end = bool(gui_quit_on_end)

        self.randomize_signal_phase = bool(
            randomize_signal_phase
        )

        if traffic_density not in DENSITY_SCALE:
            raise ValueError(
                f"traffic_density 必须是 {list(DENSITY_SCALE.keys())}"
            )

        if self.max_steps <= 0:
            raise ValueError("max_steps 必须大于 0")

        if not os.path.exists(SUMO_CONFIG):
            raise FileNotFoundError(
                f"找不到 SUMO 配置文件：{SUMO_CONFIG}"
            )

        # ----------------------------------------------------
        # 动作空间
        # ----------------------------------------------------

        self.action_space = spaces.Discrete(5)

        # ----------------------------------------------------
        # 状态空间
        # ----------------------------------------------------

        self.observation_space = spaces.Box(
            low=np.array(
                [0, 0, 0, 0, 0, -1, 0],
                dtype=np.float32
            ),
            high=np.array(
                [1, 1, 1, 1, 1, 1, 1],
                dtype=np.float32
            ),
            dtype=np.float32
        )

        # ----------------------------------------------------
        # 仿真状态
        # ----------------------------------------------------

        self.connection = None

        self.step_count = 0

        self.episode_reward = 0.0

        self.ego_departed = False

        self.ego_arrived = False

        self.ego_passed_intersection = False

        self.ego_collision = False

        self.ego_teleported = False

        self.red_light_violation = False

        self.signal_boundary_uncertain = False

        self.has_entered_junction = False

        self.initial_phase_offset_s = None

        self.insertion_delay_steps = 0

        self.previous_speed = None

        self.previous_position = None

        self.previous_distance = None

        # ----------------------------------------------------
        # 背景交通统计
        # ----------------------------------------------------

        self.total_background_departed = 0

        self.total_background_arrived = 0

        self.background_vehicle_count = 0

        # ----------------------------------------------------
        # 信号灯越线信息
        # ----------------------------------------------------

        self.signal_before_crossing = None

        self.signal_after_crossing = None

        self.crossing_signal_uncertain = False

        self.crossing_red_violation = False

        # ----------------------------------------------------
        # 路口状态
        # ----------------------------------------------------

        self.current_signal = None

        self.current_leader_gap = None

        self.current_leader_speed = None

        # ----------------------------------------------------
        # SUMO 启动命令
        # ----------------------------------------------------

        self.sumo_binary = sumolib.checkBinary(
            "sumo-gui" if render_mode == "human" else "sumo"
        )

    # ========================================================
    # 4. SUMO 启动与关闭
    # ========================================================

    def _start_sumo(self, sumo_seed):

        if self.connection is not None:
            self._close_sumo()

        command = [
            self.sumo_binary,
            "-c", SUMO_CONFIG,
            "--step-length", str(STEP_LENGTH),
            "--seed", str(sumo_seed),
            "--scale",
            str(DENSITY_SCALE[self.traffic_density]),
            "--no-step-log", "true",
            "--duration-log.disable", "true",
            "--collision.action", "warn",
            "--time-to-teleport", "300"
        ]

        if self.render_mode == "human":

            command.extend([
                "--start",
                "--quit-on-end",
                "false",
                "--delay",
                str(self.gui_delay_ms)
            ])

        else:

            command.extend([
                "--no-warnings", "true"
            ])

        traci.start(
            command,
            label="intersection_env"
        )

        self.connection = traci.getConnection(
            "intersection_env"
        )

    def _close_sumo(self):

        if self.connection is not None:

            try:
                self.connection.close()
            except Exception:
                pass

            self.connection = None

    # ========================================================
    # 5. 信号灯状态
    # ========================================================

    @staticmethod
    def _signal_class(signal_char):

        """
        将 SUMO 信号灯字符映射为统一类别。

        G / g -> green
        y     -> yellow
        r     -> red
        其他  -> unknown
        """

        if signal_char in ("G", "g"):

            return "green"

        elif signal_char == "y":

            return "yellow"

        elif signal_char == "r":

            return "red"

        return "unknown"

    def _get_tls_state(self, tls_id=TLS_ID):

        """
        获取当前信号灯状态字符串。
        """

        if self.connection is None:
            return None

        try:

            state = (
                self.connection.trafficlight
                .getRedYellowGreenState(tls_id)
            )

            return state

        except traci.TraCIException:

            return None

    def _get_signal_for_ego(self):

        """
        获取主车当前车道对应的信号灯状态。

        getNextTLS() 返回：
        (tlsID, tlsIndex, distance, state)
        """

        if self.connection is None:
            return None, None, None

        try:

            next_tls = (
                self.connection.vehicle
                .getNextTLS(EGO_ID)
            )

            if not next_tls:

                return None, None, None

            tls_id, link_index, distance, state = next_tls[0]

            return tls_id, int(link_index), state

        except traci.TraCIException:

            return None, None, None

    def _get_signal_value(self):

        """
        获取用于 observation 的信号灯数值。

        red    = 0
        yellow = 0.5
        green  = 1
        no TLS = 1
        """

        tls_id, link_index, state = self._get_signal_for_ego()

        if tls_id is None:

            self.current_signal = None

            return 1.0

        signal_class = self._signal_class(state)

        if signal_class == "red":

            self.current_signal = "red"

            return 0.0

        elif signal_class == "yellow":

            self.current_signal = "yellow"

            return 0.5

        elif signal_class == "green":

            self.current_signal = "green"

            return 1.0

        self.current_signal = "unknown"

        return 0.5

    # ========================================================
    # 6. 随机化初始信号灯相位
    # ========================================================

    def _randomize_initial_signal_phase(self):

        """
        随机化信号灯初始相位。

        不通过推进 SUMO 仿真来实现随机化，
        而是直接设置 TLS 当前 phase 和剩余持续时间。

        这样不会在 episode 开始前额外生成背景车辆。
        """

        if not self.randomize_signal_phase:

            self.initial_phase_offset_s = 0

            return

        try:

            offset = int(
                self.np_random.integers(
                    0,
                    SIGNAL_CYCLE_LENGTH
                )
            )

            self.initial_phase_offset_s = offset

            elapsed = offset

            phase_index = 0

            remaining_duration = None

            for index, duration in enumerate(
                SIGNAL_PHASE_DURATIONS
            ):

                if elapsed < duration:

                    phase_index = index

                    remaining_duration = duration - elapsed

                    break

                elapsed -= duration

            if remaining_duration is None:

                phase_index = 0

                remaining_duration = SIGNAL_PHASE_DURATIONS[0]

            self.connection.trafficlight.setPhase(
                TLS_ID,
                phase_index
            )

            self.connection.trafficlight.setPhaseDuration(
                TLS_ID,
                float(remaining_duration)
            )

        except traci.TraCIException as error:

            raise RuntimeError(
                "随机化信号灯相位失败。"
                "请检查 TLS_ID 和 network.net.xml 中的相位配置。"
            ) from error

    # ========================================================
    # 7. 背景车辆统计
    # ========================================================

    def _update_traffic_counters(self):

        """
        累计统计背景车辆进入和离开数量。

        主车 ego 不计入背景车辆。
        """

        if self.connection is None:
            return

        try:

            departed_ids = (
                self.connection.simulation
                .getDepartedIDList()
            )

            arrived_ids = (
                self.connection.simulation
                .getArrivedIDList()
            )

            self.total_background_departed += sum(
                1 for vehicle_id in departed_ids
                if vehicle_id != EGO_ID
            )

            self.total_background_arrived += sum(
                1 for vehicle_id in arrived_ids
                if vehicle_id != EGO_ID
            )

            current_ids = (
                self.connection.vehicle.getIDList()
            )

            self.background_vehicle_count = sum(
                1 for vehicle_id in current_ids
                if vehicle_id != EGO_ID
            )

        except traci.TraCIException:

            pass

    def _get_background_vehicle_count(self):

        """
        获取当前背景车辆数量。
        """

        if self.connection is None:
            return 0

        try:

            vehicle_ids = (
                self.connection.vehicle.getIDList()
            )

            return sum(
                1 for vehicle_id in vehicle_ids
                if vehicle_id != EGO_ID
            )

        except traci.TraCIException:

            return 0

    # ========================================================
    # 8. 车辆道路信息
    # ========================================================

    def _get_ego_road_id(self):

        try:

            return (
                self.connection.vehicle
                .getRoadID(EGO_ID)
            )

        except traci.TraCIException:

            return None

    def _get_ego_lane_id(self):

        try:

            return (
                self.connection.vehicle
                .getLaneID(EGO_ID)
            )

        except traci.TraCIException:

            return None

    def _get_remaining_distance(self):

        """
        获取主车到停止线的剩余距离。

        主车从 north_in 进入，目标出口由 ego_route 决定。
        """

        try:

            road_id = self._get_ego_road_id()

            lane_id = self._get_ego_lane_id()

            if road_id is None:

                return MAX_REMAINING_DISTANCE

            if road_id.startswith(":"):

                return 0.0

            if road_id == "north_in":

                lane_position = (
                    self.connection.vehicle
                    .getLanePosition(EGO_ID)
                )

                lane_length = (
                    self.connection.lane
                    .getLength(lane_id)
                )

                remaining = lane_length - lane_position

                return float(
                    np.clip(
                        remaining,
                        0,
                        MAX_REMAINING_DISTANCE
                    )
                )

            if road_id == self.exit_edge:

                return 0.0

            return MAX_REMAINING_DISTANCE

        except traci.TraCIException:

            return MAX_REMAINING_DISTANCE

    # ========================================================
    # 9. 前车信息
    # ========================================================

    def _get_leader_info(self):

        """
        获取同车道前车信息。

        返回：
        gap         前车距离
        leader_speed 前车速度
        """

        try:

            leader = (
                self.connection.vehicle
                .getLeader(
                    EGO_ID,
                    MAX_LEADER_DISTANCE
                )
            )

            if leader is None:

                self.current_leader_gap = None

                self.current_leader_speed = None

                return (
                    MAX_LEADER_DISTANCE,
                    MAX_SPEED,
                    0.0
                )

            leader_id, gap = leader

            leader_speed = (
                self.connection.vehicle
                .getSpeed(leader_id)
            )

            self.current_leader_gap = float(gap)

            self.current_leader_speed = float(
                leader_speed
            )

            return (
                float(gap),
                float(leader_speed),
                1.0
            )

        except traci.TraCIException:

            self.current_leader_gap = None

            self.current_leader_speed = None

            return (
                MAX_LEADER_DISTANCE,
                MAX_SPEED,
                0.0
            )

    # ========================================================
    # 10. 状态观测
    # ========================================================

    def _observation(self):

        """
        构建七维 observation。

        顺序必须与 state_discretizer.py 保持一致。
        """

        if EGO_ID not in self.connection.vehicle.getIDList():

            return np.zeros(
                7,
                dtype=np.float32
            )

        # 当前速度
        speed = (
            self.connection.vehicle
            .getSpeed(EGO_ID)
        )

        # 剩余距离
        remaining = self._get_remaining_distance()

        # 信号灯
        signal = self._get_signal_value()

        # 前车信息
        gap, leader_speed, light_present = (
            self._get_leader_info()
        )

        # 相对速度
        relative_speed = speed - leader_speed

        # 归一化
        observation = np.array(
            [
                speed / MAX_SPEED,

                remaining / MAX_REMAINING_DISTANCE,

                signal,

                gap / MAX_LEADER_DISTANCE,

                leader_speed / MAX_SPEED,

                relative_speed / MAX_SPEED,

                light_present
            ],
            dtype=np.float32
        )

        # 限制范围
        observation[0] = np.clip(
            observation[0],
            0,
            1
        )

        observation[1] = np.clip(
            observation[1],
            0,
            1
        )

        observation[2] = np.clip(
            observation[2],
            0,
            1
        )

        observation[3] = np.clip(
            observation[3],
            0,
            1
        )

        observation[4] = np.clip(
            observation[4],
            0,
            1
        )

        observation[5] = np.clip(
            observation[5],
            -1,
            1
        )

        observation[6] = np.clip(
            observation[6],
            0,
            1
        )

        return observation

    # ========================================================
    # 11. 越线检测
    # ========================================================

    def _check_crossing(self, before_road, after_road):

        """
        检查主车是否从 north_in 越过停止线。

        仅在 crossing 发生的当前 step 检查信号灯状态。
        """

        self.crossing_signal_uncertain = False

        self.crossing_red_violation = False

        self.signal_before_crossing = None

        self.signal_after_crossing = None

        if before_road is None or after_road is None:

            return False

        crossing = (
            before_road == "north_in"
            and (
                after_road.startswith(":")
                or after_road == self.exit_edge
            )
        )

        if not crossing:

            return False

        self.has_entered_junction = True

        # 获取越线前信号灯
        before_tls = self._get_signal_for_ego()

        # 注意：调用此函数时主车已经执行 simulationStep，
        # 因此 getNextTLS 读取的是越线后的状态。
        #
        # 越线前状态由 step() 在 simulationStep 前缓存，
        # 这里使用缓存值进行判断。

        before_state = getattr(
            self,
            "_cached_signal_before_step",
            None
        )

        after_state = None

        if before_tls is not None:

            after_state = before_tls[2]

        self.signal_before_crossing = before_state

        self.signal_after_crossing = after_state

        before_class = self._signal_class(
            before_state
        ) if before_state is not None else "unknown"

        after_class = self._signal_class(
            after_state
        ) if after_state is not None else "unknown"

        # 只有越线前状态明确为红灯时，才判定闯红灯。
        if before_class == "red":

            self.crossing_red_violation = True

            self.red_light_violation = True

        # 状态不确定单独记录，不直接判定闯红灯。
        elif before_class == "unknown":

            self.crossing_signal_uncertain = True

            self.signal_boundary_uncertain = True

        return True

    # ========================================================
    # 12. reset
    # ========================================================

    def reset(
        self,
        seed=None,
        options=None
    ):

        super().reset(seed=seed)

        options = options or {}

        route = options.get("ego_route", self.default_ego_route)
        if route not in EGO_ROUTES:
            raise ValueError(f"ego_route 必须是 {list(EGO_ROUTES)}")
        self.ego_route = route
        self.route_id, self.exit_edge = EGO_ROUTES[route]

        # ----------------------------------------------------
        # 更新交通密度
        # ----------------------------------------------------

        density = options.get(
            "density",
            self.traffic_density
        )

        if density not in DENSITY_SCALE:

            raise ValueError(
                f"density 必须是 {list(DENSITY_SCALE.keys())}"
            )

        self.traffic_density = density

        # ----------------------------------------------------
        # 重置 episode 统计
        # ----------------------------------------------------

        self.step_count = 0

        self.episode_reward = 0.0

        self.ego_departed = False

        self.ego_arrived = False

        self.ego_passed_intersection = False

        self.ego_collision = False

        self.ego_teleported = False

        self.red_light_violation = False

        self.signal_boundary_uncertain = False

        self.has_entered_junction = False

        self.initial_phase_offset_s = None

        self.insertion_delay_steps = 0

        self.previous_speed = None

        self.previous_position = None

        self.previous_distance = None

        self.total_background_departed = 0

        self.total_background_arrived = 0

        self.background_vehicle_count = 0

        self.signal_before_crossing = None

        self.signal_after_crossing = None

        self.crossing_signal_uncertain = False

        self.crossing_red_violation = False

        self._cached_signal_before_step = None

        # ----------------------------------------------------
        # 启动 SUMO
        # ----------------------------------------------------

        # 每次回合从 Gymnasium 随机生成器产生 SUMO 种子：显式 seed 可复现，
        # reset(seed=None) 延续随机序列，避免每回合重复同一背景车流。
        self.sumo_seed = int(self.np_random.integers(0, 2**31 - 1))
        self._start_sumo(self.sumo_seed)

        # ----------------------------------------------------
        # 随机初始信号灯相位
        # ----------------------------------------------------

        self._randomize_initial_signal_phase()

        # ----------------------------------------------------
        # 插入主车：检查路线并添加
        # ----------------------------------------------------

        # 获取当前 SUMO 已加载的路线 ID
        existing_routes = set(
            self.connection.route.getIDList()
        )

        # 获取当前路网中的道路 ID
        existing_edges = set(
            self.connection.edge.getIDList()
        )

        # 检查主车所需道路是否存在
        required_edges = [
            "north_in",
            self.exit_edge
        ]

        missing_edges = [
            edge
            for edge in required_edges
            if edge not in existing_edges
        ]

        if missing_edges:

            self._close_sumo()

            raise RuntimeError(
                f"主车路线包含不存在的道路：{missing_edges}"
            )

        # 如果路线不存在，才动态添加
        if self.route_id not in existing_routes:

            try:

                self.connection.route.add(
                    self.route_id,
                    required_edges
                )

                print(
                    f"[SUMO INFO] "
                    f"成功创建主车路线：{self.route_id}"
                )

            except traci.TraCIException as error:

                self._close_sumo()

                raise RuntimeError(
                    f"创建主车路线失败：{self.route_id}。"
                    f"请检查 north_in 和 {self.exit_edge} 是否连通。"
                ) from error

        else:

            print(
                f"[SUMO INFO] "
                f"使用已有主车路线：{self.route_id}"
            )

        # 插入主车
        try:

            self.connection.vehicle.add(
                vehID=EGO_ID,
                routeID=self.route_id,
                typeID=EGO_TYPE_ID,
                depart="now",
                departLane="best",
                departSpeed="0"
            )

            # 禁止主车自动换道
            self.connection.vehicle.setLaneChangeMode(
                EGO_ID,
                0
            )

        except traci.TraCIException as error:

            self._close_sumo()

            raise RuntimeError(
                f"主车插入失败："
                f"vehicle={EGO_ID}, "
                f"route={self.route_id}, "
                f"type={EGO_TYPE_ID}"
            ) from error

        # ----------------------------------------------------
        # 等待主车进入路网
        # ----------------------------------------------------

        ego_found = False

        for wait_step in range(60):

            self.connection.simulationStep()

            self._update_traffic_counters()

            vehicle_ids = (
                self.connection.vehicle.getIDList()
            )

            if EGO_ID in vehicle_ids:

                ego_found = True

                self.ego_departed = True

                self.insertion_delay_steps = wait_step + 1

                break

        if not ego_found:

            self._close_sumo()

            raise RuntimeError(
                "主车在 60 个仿真步内未能进入路网。"
                "请检查 routes.rou.xml 和 network.net.xml。"
            )

        # ----------------------------------------------------
        # 初始化速度和距离
        # ----------------------------------------------------

        self.previous_speed = (
            self.connection.vehicle
            .getSpeed(EGO_ID)
        )

        self.previous_position = (
            self.connection.vehicle
            .getPosition(EGO_ID)
        )

        self.previous_distance = (
            self._get_remaining_distance()
        )

        self.background_vehicle_count = (
            self._get_background_vehicle_count()
        )

        # ----------------------------------------------------
        # 返回 observation
        # ----------------------------------------------------

        observation = self._observation()

        info = {

            "ego_route": self.ego_route,
            "route_id": self.route_id,
            "exit_edge": self.exit_edge,

            "traffic_density": self.traffic_density,

            "sumo_seed": self.sumo_seed,

            "traffic_scale": DENSITY_SCALE[
                self.traffic_density
            ],

            "initial_phase_offset_s": (
                self.initial_phase_offset_s
            ),

            "insertion_delay_steps": (
                self.insertion_delay_steps
            ),

            "background_vehicle_count": (
                self.background_vehicle_count
            ),

            "traffic_vehicle_count": (
                self.background_vehicle_count
            ),

            "total_background_departed": (
                self.total_background_departed
            ),

            "total_background_arrived": (
                self.total_background_arrived
            ),

            "ego_departed": self.ego_departed,

            "ego_present": True
        }

        return observation, info

    # ========================================================
    # 13. step
    # ========================================================

    def step(self, action):

        if self.connection is None:

            raise RuntimeError(
                "SUMO 尚未启动，请先调用 reset()。"
            )

        if not self.action_space.contains(action):

            raise ValueError(
                f"无效动作：{action}"
            )

        if EGO_ID not in self.connection.vehicle.getIDList():

            observation = np.zeros(
                7,
                dtype=np.float32
            )

            info = {
                "ego_present": False,
                "passed_intersection": self.ego_passed_intersection,
                "collision": self.ego_collision,
                "red_light_violation": self.red_light_violation
            }

            return observation, -100.0, True, False, info

        self.step_count += 1

        # ----------------------------------------------------
        # 记录 step 前状态
        # ----------------------------------------------------

        before_road = self._get_ego_road_id()

        before_speed = (
            self.connection.vehicle
            .getSpeed(EGO_ID)
        )

        before_position = (
            self.connection.vehicle
            .getPosition(EGO_ID)
        )

        before_distance = (
            self._get_remaining_distance()
        )

        # 越线前信号灯状态
        before_tls = self._get_signal_for_ego()

        if before_tls is not None:

            self._cached_signal_before_step = before_tls[2]

        else:

            self._cached_signal_before_step = None

        # ----------------------------------------------------
        # 执行动作
        # ----------------------------------------------------

        requested_acceleration = (
            ACTION_ACCELERATIONS[int(action)]
        )

        self.connection.vehicle.setAcceleration(
            EGO_ID,
            float(requested_acceleration),
            STEP_LENGTH
        )

        # ----------------------------------------------------
        # 推进仿真
        # ----------------------------------------------------

        self.connection.simulationStep()

        # 更新背景交通统计
        self._update_traffic_counters()

        # ----------------------------------------------------
        # 获取 step 后状态
        # ----------------------------------------------------

        vehicle_ids = (
            self.connection.vehicle.getIDList()
        )

        ego_present = EGO_ID in vehicle_ids

        after_road = None

        after_speed = 0.0

        after_position = None

        after_distance = 0.0

        if ego_present:

            after_road = self._get_ego_road_id()

            after_speed = (
                self.connection.vehicle
                .getSpeed(EGO_ID)
            )

            after_position = (
                self.connection.vehicle
                .getPosition(EGO_ID)
            )

            after_distance = (
                self._get_remaining_distance()
            )

        # ----------------------------------------------------
        # 实际加速度
        # ----------------------------------------------------

        actual_acceleration = (
            after_speed - before_speed
        ) / STEP_LENGTH if ego_present else 0.0

        # ----------------------------------------------------
        # 距离进展
        # ----------------------------------------------------

        progress = (
            before_distance - after_distance
        ) if ego_present else 0.0

        progress = float(
            np.clip(
                progress,
                -MAX_SPEED * STEP_LENGTH,
                MAX_SPEED * STEP_LENGTH
            )
        )

        # ----------------------------------------------------
        # 越线检测
        # ----------------------------------------------------

        crossed_stop_line = False

        if ego_present:

            crossed_stop_line = self._check_crossing(
                before_road,
                after_road
            )

        # ----------------------------------------------------
        # 通过路口检测
        # ----------------------------------------------------

        arrived = False

        passed = False

        if ego_present:

            if after_road == self.exit_edge:

                passed = self.has_entered_junction

        else:

            try:

                arrived_ids = (
                    self.connection.simulation
                    .getArrivedIDList()
                )

                arrived = EGO_ID in arrived_ids

            except traci.TraCIException:

                arrived = False

        if arrived:

            self.ego_arrived = True

        if passed:

            self.ego_passed_intersection = True

        # ----------------------------------------------------
        # 碰撞检测
        # ----------------------------------------------------

        collision_ids = []

        try:

            collision_list = (
                self.connection.simulation
                .getCollidingVehiclesIDList()
            )

            collision_ids = list(collision_list)

        except traci.TraCIException:

            pass

        collided = (
            EGO_ID in collision_ids
        )

        if collided:

            self.ego_collision = True

        # ----------------------------------------------------
        # 传送检测
        # ----------------------------------------------------

        teleported = False

        try:

            starting_teleport_ids = (
                self.connection.simulation
                .getStartingTeleportIDList()
            )

            ending_teleport_ids = (
                self.connection.simulation
                .getEndingTeleportIDList()
            )

            teleported = (
                EGO_ID in starting_teleport_ids
                or EGO_ID in ending_teleport_ids
            )

        except traci.TraCIException:

            pass

        if teleported:

            self.ego_teleported = True

        # ----------------------------------------------------
        # 当前背景车辆数量
        # ----------------------------------------------------

        self.background_vehicle_count = (
            self._get_background_vehicle_count()
        )

        # ----------------------------------------------------
        # 终止条件
        # ----------------------------------------------------

        terminated = bool(

            self.ego_arrived

            or self.ego_passed_intersection

            or self.ego_collision

            or self.ego_teleported

            or self.red_light_violation

            or not ego_present

        )

        truncated = bool(
            self.step_count >= self.max_steps
            and not terminated
        )

        # ----------------------------------------------------
        # 奖励函数
        # ----------------------------------------------------

        reward = 0.0

        # 前进奖励
        reward += 0.1 * progress

        # 每步时间惩罚
        reward -= 0.05

        # 急刹车惩罚
        if actual_acceleration <= -3.0:

            reward -= 0.1

        # 碰撞惩罚
        if collided:

            reward -= 100.0

        # 闯红灯惩罚
        if self.crossing_red_violation:

            reward -= 100.0

        # 成功通过奖励
        if (
            self.ego_passed_intersection
            and not self.red_light_violation
        ):

            reward += 50.0

        # ----------------------------------------------------
        # 更新累计奖励
        # ----------------------------------------------------

        self.episode_reward += reward

        # ----------------------------------------------------
        # 构建 observation
        # ----------------------------------------------------

        if ego_present:

            observation = self._observation()

        else:

            observation = np.zeros(
                7,
                dtype=np.float32
            )

        # ----------------------------------------------------
        # 获取当前信号灯
        # ----------------------------------------------------

        current_signal = self.current_signal

        # ----------------------------------------------------
        # 构建 info
        # ----------------------------------------------------

        info = {

            "ego_route": self.ego_route,
            "route_id": self.route_id,
            "exit_edge": self.exit_edge,

            # 基本信息
            "step": self.step_count,

            "traffic_density": self.traffic_density,

            "sumo_seed": self.sumo_seed,

            "traffic_scale": DENSITY_SCALE[
                self.traffic_density
            ],

            # 主车状态
            "ego_present": ego_present,

            "ego_departed": self.ego_departed,

            "arrived": self.ego_arrived,

            "passed_intersection": (
                self.ego_passed_intersection
            ),

            # 安全信息
            "collision": self.ego_collision,

            "teleported": self.ego_teleported,

            "red_light_violation": (
                self.red_light_violation
            ),

            "crossing_red_violation": (
                self.crossing_red_violation
            ),

            "signal_boundary_uncertain": (
                self.signal_boundary_uncertain
            ),

            "crossing_signal_uncertain": (
                self.crossing_signal_uncertain
            ),

            "crossed_stop_line": crossed_stop_line,

            "signal_before_crossing": (
                self.signal_before_crossing
            ),

            "signal_after_crossing": (
                self.signal_after_crossing
            ),

            "current_signal": current_signal,

            # 运动信息
            "progress_m": progress,

            "speed_m_s": after_speed,

            "requested_acceleration": (
                requested_acceleration
            ),

            "actual_acceleration_m_s2": (
                actual_acceleration
            ),

            "remaining_distance_m": (
                after_distance
            ),

            # 交通流信息
            "traffic_vehicle_count": (
                self.background_vehicle_count
            ),

            "background_vehicle_count": (
                self.background_vehicle_count
            ),

            "total_background_departed": (
                self.total_background_departed
            ),

            "total_background_arrived": (
                self.total_background_arrived
            ),

            # 其他
            "safety_interventions": None,

            "episode_reward": self.episode_reward,

            "terminated": terminated,

            "truncated": truncated
        }

        # ----------------------------------------------------
        # GUI 控制
        # ----------------------------------------------------

        if self.render_mode == "human":

            try:

                self.connection.gui.setSchema(
                    "View #0",
                    "real world"
                )

            except traci.TraCIException:

                pass

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info
        )

    # ========================================================
    # 14. render
    # ========================================================

    def render(self):

        if self.render_mode == "human":

            return None

        return None

    # ========================================================
    # 15. close
    # ========================================================

    def close(self):

        self._close_sumo()

    # ========================================================
    # 16. 其他辅助方法
    # ========================================================

    def get_episode_statistics(self):

        """
        获取当前 episode 的统计信息。
        """

        return {

            "traffic_density": self.traffic_density,

            "initial_phase_offset_s": (
                self.initial_phase_offset_s
            ),

            "insertion_delay_steps": (
                self.insertion_delay_steps
            ),

            "episode_reward": (
                self.episode_reward
            ),

            "episode_steps": (
                self.step_count
            ),

            "passed_intersection": (
                self.ego_passed_intersection
            ),

            "arrived": (
                self.ego_arrived
            ),

            "collision": (
                self.ego_collision
            ),

            "teleported": (
                self.ego_teleported
            ),

            "red_light_violation": (
                self.red_light_violation
            ),

            "signal_boundary_uncertain": (
                self.signal_boundary_uncertain
            ),

            "background_vehicle_count": (
                self.background_vehicle_count
            ),

            "total_background_departed": (
                self.total_background_departed
            ),

            "total_background_arrived": (
                self.total_background_arrived
            )
        }
