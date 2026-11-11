
"""
将 IntersectionEnv 的7维连续状态离散化，供表格型 Q-learning 使用。

状态顺序必须与 sumo_env.py 完全一致：
0 speed
1 remaining
2 signal
3 gap
4 leader_speed
5 relative_speed
6 light_present
"""

import numpy as np


class StateDiscretizer:
    """使用固定阈值将7维连续观测映射为离散状态。"""

    def __init__(self):
        # 速度、前车速度：归一化范围 [0, 1]
        self.speed_bins = np.array(
            [0.15, 0.35, 0.55, 0.75], dtype=np.float32
        )

        # 距离信号灯：归一化范围 [0, 1]
        self.remaining_bins = np.array(
            [0.04, 0.10, 0.20, 0.40, 0.70], dtype=np.float32
        )

        # 信号灯：红灯0、黄灯0.5、绿灯1
        self.signal_bins = np.array(
            [0.25, 0.75], dtype=np.float32
        )

        # 与前车距离：归一化范围 [0, 1]
        self.gap_bins = np.array(
            [0.05, 0.10, 0.20, 0.40], dtype=np.float32
        )

        # 相对速度：归一化范围 [-1, 1]
        self.relative_speed_bins = np.array(
            [-0.50, -0.20, 0.20, 0.50], dtype=np.float32
        )

        # 每个维度的离散状态数量
        self.state_shape = (
            5,  # speed
            6,  # remaining
            3,  # signal
            5,  # gap
            5,  # leader_speed
            5,  # relative_speed
            2,  # light_present
        )

    @staticmethod
    def _digitize(value, bins):
        """将一个连续数值转换为离散编号。"""
        return int(np.digitize(value, bins))

    def discretize(self, observation):
        """
        参数：
            observation: shape=(7,) 的环境观测

        返回：
            tuple，例如 (2, 2, 0, 1, 1, 2, 1)
        """
        obs = np.asarray(observation, dtype=np.float32)

        if obs.shape != (7,):
            raise ValueError(
                f"状态必须是7维，当前收到的形状为 {obs.shape}"
            )

        if not np.all(np.isfinite(obs)):
            raise ValueError("状态中包含 NaN 或 Inf")

        # 按环境 observation_space 的范围进行保护性截断
        obs = np.clip(
            obs,
            np.array([0, 0, 0, 0, 0, -1, 0], dtype=np.float32),
            np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
        )

        speed = self._digitize(obs[0], self.speed_bins)
        remaining = self._digitize(obs[1], self.remaining_bins)
        signal = self._digitize(obs[2], self.signal_bins)
        gap = self._digitize(obs[3], self.gap_bins)
        leader_speed = self._digitize(obs[4], self.speed_bins)
        relative_speed = self._digitize(
            obs[5], self.relative_speed_bins
        )

        # light_present 是二值特征：
        # 0 = 前方没有信号灯
        # 1 = 前方存在信号灯
        light_present = int(obs[6] >= 0.5)

        state = (
            speed,
            remaining,
            signal,
            gap,
            leader_speed,
            relative_speed,
            light_present,
        )

        return state

    def get_state_shape(self):
        """返回Q-table前7个维度的大小。"""
        return self.state_shape

    def get_num_states(self):
        """返回离散状态组合总数。"""
        return int(np.prod(self.state_shape))
