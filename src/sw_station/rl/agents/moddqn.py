"""MODDQN 多目标双重深度 Q 网络实现"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


@dataclass
class MODDQNConfig:
    """MODDQN 配置"""
    # 学习率
    learning_rate: float = 3e-4
    # 折扣因子
    gamma: float = 0.99
    # 经验回放缓冲区大小
    buffer_size: int = 100000
    # 批次大小
    batch_size: int = 64
    # 目标网络更新频率
    target_update_freq: int = 1000
    # 探索率
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 10000
    # 网络隐藏层大小
    hidden_size: int = 256
    # 目标数量
    n_objectives: int = 3
    # 设备
    device: str = "auto"


class MultiObjectiveQNetwork(nn.Module):
    """
    多目标 Q 网络

    输出多个 Q 值，分别对应不同的优化目标。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_objectives: int = 3,
        hidden_size: int = 256,
    ):
        """
        初始化网络

        Parameters
        ----------
        state_dim : int
            状态维度
        action_dim : int
            动作维度
        n_objectives : int
            目标数量
        hidden_size : int
            隐藏层大小
        """
        super().__init__()

        self.n_objectives = n_objectives

        # 共享特征提取层
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # 每个目标独立的 Q 值头
        self.q_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, action_dim),
            )
            for _ in range(n_objectives)
        ])

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Parameters
        ----------
        state : torch.Tensor
            状态张量, shape: (batch, state_dim)

        Returns
        -------
        torch.Tensor
            Q 值张量, shape: (batch, n_objectives, action_dim)
        """
        features = self.feature_layer(state)

        q_values = []
        for head in self.q_heads:
            q_values.append(head(features))

        # 堆叠为 (batch, n_objectives, action_dim)
        return torch.stack(q_values, dim=1)


class PrioritizedReplayBuffer:
    """
    优先经验回放缓冲区

    根据 TD 误差设置采样优先级。
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        alpha: float = 0.6,
    ):
        """
        初始化缓冲区

        Parameters
        ----------
        capacity : int
            缓冲区容量
        state_dim : int
            状态维度
        alpha : float
            优先级指数
        """
        self.capacity = capacity
        self.alpha = alpha

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.priorities = np.zeros(capacity, dtype=np.float32)

        self.position = 0
        self.size = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        priority: Optional[float] = None,
    ) -> None:
        """
        添加经验

        Parameters
        ----------
        state : np.ndarray
            状态
        action : int
            动作
        reward : float
            奖励
        next_state : np.ndarray
            下一状态
        done : bool
            是否结束
        priority : float, optional
            优先级
        """
        if priority is None:
            priority = self.priorities[:self.size].max() if self.size > 0 else 1.0

        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)
        self.priorities[self.position] = priority

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple:
        """
        按优先级采样

        Parameters
        ----------
        batch_size : int
            批次大小

        Returns
        -------
        tuple
            (states, actions, rewards, next_states, dones, indices, weights)
        """
        # 计算采样概率
        priorities = self.priorities[:self.size] ** self.alpha
        probs = priorities / priorities.sum()

        # 采样
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)

        # 重要性采样权重
        weights = (self.size * probs[indices]) ** (-0.4)
        weights = weights / weights.max()

        return (
            torch.FloatTensor(self.states[indices]),
            torch.LongTensor(self.actions[indices]),
            torch.FloatTensor(self.rewards[indices]),
            torch.FloatTensor(self.next_states[indices]),
            torch.FloatTensor(self.dones[indices]),
            indices,
            torch.FloatTensor(weights),
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """更新优先级"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6

    def __len__(self) -> int:
        return self.size


class MODDQNAgent:
    """
    MODDQN 智能体

    实现 Double DQN 架构的多目标 Q 学习。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[MODDQNConfig] = None,
    ):
        """
        初始化 MODDQN 智能体

        Parameters
        ----------
        state_dim : int
            状态维度
        action_dim : int
            动作维度
        config : MODDQNConfig, optional
            配置
        """
        self.config = config or MODDQNConfig()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # 设备选择
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        # 主网络
        self.q_network = MultiObjectiveQNetwork(
            state_dim, action_dim,
            self.config.n_objectives,
            self.config.hidden_size,
        ).to(self.device)

        # 目标网络
        self.target_network = MultiObjectiveQNetwork(
            state_dim, action_dim,
            self.config.n_objectives,
            self.config.hidden_size,
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # 优化器
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.config.learning_rate,
        )

        # 经验回放
        self.replay_buffer = PrioritizedReplayBuffer(
            self.config.buffer_size, state_dim
        )

        # 探索率
        self.epsilon = self.config.epsilon_start
        self.steps_done = 0

    def select_action(
        self,
        state: np.ndarray,
        evaluate: bool = False,
    ) -> int:
        """
        选择动作

        Parameters
        ----------
        state : np.ndarray
            状态
        evaluate : bool
            是否评估模式

        Returns
        -------
        int
            选择的动作
        """
        if not evaluate and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)

            # 聚合多目标 Q 值（简单求和）
            aggregated_q = q_values.sum(dim=1)  # (1, action_dim)
            return aggregated_q.argmax(dim=1).item()

    def update(self) -> Optional[float]:
        """
        更新网络

        Returns
        -------
        float, optional
            损失值
        """
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        # 采样
        states, actions, rewards, next_states, dones, indices, weights = \
            self.replay_buffer.sample(self.config.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)

        # 当前 Q 值
        current_q = self.q_network(states)
        # 选择对应动作的 Q 值
        current_q = current_q.gather(
            2, actions.unsqueeze(1).unsqueeze(2).expand(-1, self.config.n_objectives, 1)
        ).squeeze(2)

        # Double DQN: 用主网络选择动作，用目标网络评估
        with torch.no_grad():
            next_q_main = self.q_network(next_states)
            next_actions = next_q_main.sum(dim=1).argmax(dim=1)

            next_q_target = self.target_network(next_states)
            next_q = next_q_target.gather(
                2, next_actions.unsqueeze(1).unsqueeze(2).expand(-1, self.config.n_objectives, 1)
            ).squeeze(2)

            target_q = rewards.unsqueeze(1) + self.config.gamma * next_q * (1 - dones.unsqueeze(1))

        # 计算损失
        td_errors = (current_q - target_q).abs()
        loss = (weights.unsqueeze(1) * td_errors ** 2).mean()

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新优先级
        priorities = td_errors.sum(dim=1).detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, priorities)

        # 更新探索率
        self.steps_done += 1
        self.epsilon = self.config.epsilon_end + \
            (self.config.epsilon_start - self.config.epsilon_end) * \
            np.exp(-self.steps_done / self.config.epsilon_decay)

        # 更新目标网络
        if self.steps_done % self.config.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def save(self, path: str) -> None:
        """保存模型"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'epsilon': self.epsilon,
        }, path)

    def load(self, path: str) -> None:
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint['steps_done']
        self.epsilon = checkpoint['epsilon']
