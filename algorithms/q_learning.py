
"""
Tabular Q-learning agent for the SUMO intersection environment.
"""

from pathlib import Path
import json

import numpy as np

from algorithms.state_discretizer import StateDiscretizer


class QLearningAgent:
    """表格型 Q-learning 智能体。"""

    def __init__(
        self,
        learning_rate=0.10,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        action_count=5,
        seed=42,
    ):
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.action_count = action_count

        self.rng = np.random.default_rng(seed)
        self.discretizer = StateDiscretizer()

        # Q-table：7维离散状态 + 1维动作
        self.q_table = np.zeros(
            self.discretizer.get_state_shape() + (self.action_count,),
            dtype=np.float32,
        )

        self.update_count = 0

    def select_action(self, observation, info=None, training=False):
        """
        根据当前观测选择动作。

        training=True：
            使用 epsilon-greedy，允许探索。

        training=False：
            始终选择当前Q值最大的动作。
        """
        state = self.discretizer.discretize(observation)

        # 探索：随机选择一个动作
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.action_count))

        # 利用：选择Q值最大的动作
        q_values = self.q_table[state]

        # 多个动作Q值相同时，随机选择其中一个
        best_actions = np.flatnonzero(
            np.isclose(q_values, np.max(q_values))
        )

        return int(self.rng.choice(best_actions))

    def learn(
        self,
        observation,
        action,
        reward,
        next_observation,
        terminated,
        truncated=False,
    ):
        """
        执行一次Q-learning更新。

        terminated=True：
            真正的终止状态，不再进行未来奖励估计。

        truncated=True：
            时间步数达到上限等截断情况。
            默认仍然保留下一状态的价值估计。
        """
        state = self.discretizer.discretize(observation)
        next_state = self.discretizer.discretize(next_observation)

        if not 0 <= int(action) < self.action_count:
            raise ValueError(f"无效动作：{action}")

        current_q = self.q_table[state + (int(action),)]

        if terminated:
            target = float(reward)
        else:
            next_max_q = float(np.max(self.q_table[next_state]))
            target = float(reward) + self.gamma * next_max_q

        # Q-learning 更新公式
        self.q_table[state + (int(action),)] = (
            current_q
            + self.learning_rate * (target - current_q)
        )

        self.update_count += 1

    def decay_epsilon(self):
        """每轮训练结束后降低探索率。"""
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

    def save(self, path):
        """
        保存Q-table和超参数。

        例如：
            models/q_learning/q_table.npz
            models/q_learning/q_table.json
        """
        path = Path(path)

        if path.suffix != ".npz":
            path = path.with_suffix(".npz")

        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            path,
            q_table=self.q_table,
        )

        metadata = {
            "algorithm": "Q-learning",
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "action_count": self.action_count,
            "state_shape": list(self.discretizer.get_state_shape()),
            "update_count": self.update_count,
        }

        metadata_path = path.with_suffix(".json")

        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=4, ensure_ascii=False)

        print(f"Q-table已保存：{path}")
        print(f"模型参数已保存：{metadata_path}")

    def load(self, path):
        """从NPZ文件加载Q-table，并读取对应JSON参数。"""
        path = Path(path)

        if path.suffix != ".npz":
            path = path.with_suffix(".npz")

        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在：{path}")

        metadata_path = path.with_suffix(".json")

        with np.load(path) as data:
            loaded_q_table = data["q_table"]

        expected_shape = (
            self.discretizer.get_state_shape()
            + (self.action_count,)
        )

        if loaded_q_table.shape != expected_shape:
            raise ValueError(
                f"Q-table形状不匹配："
                f"期望 {expected_shape}，实际 {loaded_q_table.shape}"
            )

        self.q_table = loaded_q_table.astype(np.float32)

        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as file:
                metadata = json.load(file)

            if metadata.get("action_count") != self.action_count:
                raise ValueError("模型动作数量与当前环境不一致")

            if tuple(metadata.get("state_shape", [])) != self.discretizer.get_state_shape():
                raise ValueError("模型状态离散化维度与当前版本不一致")

            self.learning_rate = metadata["learning_rate"]
            self.gamma = metadata["gamma"]
            self.epsilon = metadata["epsilon"]
            self.epsilon_min = metadata["epsilon_min"]
            self.epsilon_decay = metadata["epsilon_decay"]
            self.update_count = metadata.get("update_count", 0)

        print(f"Q-table已加载：{path}")
