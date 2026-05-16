"""Attention-MO-PPO 多智能体近端策略优化实现"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


@dataclass
class AttentionPPOConfig:
    """Attention-MO-PPO 配置"""
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 10
    batch_size: int = 64
    hidden_size: int = 256
    attention_heads: int = 4
    n_objectives: int = 3
    # 多目标奖励权重
    reward_weights: dict = None  # 若为None，自动从env配置继承


class AttentionModule(nn.Module):
    """
    注意力机制模块

    允许智能体关注其他智能体的状态和动作意图。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_heads: int = 4,
    ):
        """
        初始化注意力模块

        Parameters
        ----------
        input_dim : int
            输入维度
        hidden_dim : int
            隐藏维度
        n_heads : int
            注意力头数
        """
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        # Q, K, V 投影
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)

        # 输出投影
        self.output = nn.Linear(hidden_dim, hidden_dim)

        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query_input: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播

        Parameters
        ----------
        query_input : torch.Tensor
            查询输入, shape: (batch, input_dim)
        context : torch.Tensor
            上下文输入, shape: (batch, n_agents, input_dim)

        Returns
        -------
        torch.Tensor
            注意力输出, shape: (batch, hidden_dim)
        """
        batch_size = query_input.size(0)

        # 计算 Q
        Q = self.query(query_input).unsqueeze(1)  # (batch, 1, hidden_dim)

        # 计算 K, V
        K = self.key(context)  # (batch, n_agents, hidden_dim)
        V = self.value(context)  # (batch, n_agents, hidden_dim)

        # 多头注意力
        Q = Q.view(batch_size, 1, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # 注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)

        # 加权求和
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, 1, -1)
        attended = attended.squeeze(1)

        # 输出投影
        output = self.output(attended)

        # 残差连接和层归一化
        output = self.layer_norm(output + query_input)

        return output


class ActorCriticWithAttention(nn.Module):
    """
    带注意力机制的 Actor-Critic 网络
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_objectives: int = 3,
        hidden_size: int = 256,
        attention_heads: int = 4,
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
        attention_heads : int
            注意力头数
        """
        super().__init__()

        self.n_objectives = n_objectives
        self.action_dim = action_dim

        # 特征提取
        self.feature_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # 注意力模块
        self.attention = AttentionModule(
            hidden_size, hidden_size, attention_heads
        )

        # Actor 头（策略网络）
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_dim),
        )

        # Critic 头（价值网络）- 每个目标一个
        self.critics = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1),
            )
            for _ in range(n_objectives)
        ])

    def forward(
        self,
        state: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Parameters
        ----------
        state : torch.Tensor
            状态, shape: (batch, state_dim)
        context : torch.Tensor, optional
            其他智能体上下文, shape: (batch, n_agents, state_dim)

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (动作 logits, 状态价值)
        """
        # 特征编码
        features = self.feature_encoder(state)

        # 注意力（如果有上下文）
        if context is not None:
            # 将上下文编码为特征
            context_features = self.feature_encoder(
                context.view(-1, state.size(-1))
            ).view(context.size(0), context.size(1), -1)
            features = self.attention(features, context_features)

        # Actor 输出
        action_logits = self.actor(features)

        # Critic 输出
        values = torch.cat([critic(features) for critic in self.critics], dim=-1)

        return action_logits, values

    def get_action(
        self,
        state: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> tuple[int, float, float]:
        """
        获取动作

        Parameters
        ----------
        state : torch.Tensor
            状态
        context : torch.Tensor, optional
            上下文

        Returns
        -------
        tuple[int, float, float]
            (动作, 对数概率, 价值)
        """
        with torch.no_grad():
            logits, values = self.forward(state.unsqueeze(0), context)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), values.squeeze(0)


class RolloutBuffer:
    """经验存储缓冲区"""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.multi_rewards = []  # 多目标奖励

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
        multi_reward: Optional[np.ndarray] = None,
    ):
        """添加经验"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        if multi_reward is not None:
            self.multi_rewards.append(multi_reward)

    def clear(self):
        """清空缓冲区"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.multi_rewards.clear()

    def compute_returns(self, gamma: float, gae_lambda: float) -> tuple:
        """计算回报和优势"""
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(self.actions)
        old_log_probs = torch.FloatTensor(self.log_probs)

        # 计算 GAE
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)

        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae

        returns = advantages + values

        return (
            states,
            actions,
            old_log_probs,
            torch.FloatTensor(returns),
            torch.FloatTensor(advantages),
        )


class AttentionMO_PPOAgent:
    """
    Attention-MO-PPO 智能体

    基于注意力机制的多目标近端策略优化。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[AttentionPPOConfig] = None,
    ):
        """
        初始化智能体

        Parameters
        ----------
        state_dim : int
            状态维度
        action_dim : int
            动作维度
        config : AttentionPPOConfig, optional
            配置
        """
        self.config = config or AttentionPPOConfig()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 网络
        self.network = ActorCriticWithAttention(
            state_dim, action_dim,
            self.config.n_objectives,
            self.config.hidden_size,
            self.config.attention_heads,
        ).to(self.device)

        # 优化器
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
        )

        # 经验缓冲
        self.buffer = RolloutBuffer()

    def select_action(
        self,
        state: np.ndarray,
        context: Optional[np.ndarray] = None,
    ) -> tuple[int, float, np.ndarray]:
        """
        选择动作

        Parameters
        ----------
        state : np.ndarray
            状态
        context : np.ndarray, optional
            其他智能体上下文

        Returns
        -------
        tuple[int, float, np.ndarray]
            (动作, 对数概率, 价值)
        """
        state_tensor = torch.FloatTensor(state).to(self.device)
        context_tensor = torch.FloatTensor(context).to(self.device) if context is not None else None

        action, log_prob, values = self.network.get_action(state_tensor, context_tensor)

        return action, log_prob, values.cpu().numpy()

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ):
        """存储经验"""
        self.buffer.add(state, action, reward, value, log_prob, done)

    def update(self) -> dict:
        """
        更新网络

        Returns
        -------
        dict
            训练信息
        """
        # 计算回报
        states, actions, old_log_probs, returns, advantages = \
            self.buffer.compute_returns(self.config.gamma, self.config.gae_lambda)

        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        # 多轮更新
        for _ in range(self.config.n_epochs):
            # 随机打乱
            indices = np.random.permutation(len(states))

            for start in range(0, len(states), self.config.batch_size):
                end = start + self.config.batch_size
                batch_idx = indices[start:end]

                # 前向传播
                logits, values = self.network(states[batch_idx])
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)

                # 新的对数概率
                new_log_probs = dist.log_prob(actions[batch_idx])
                entropy = dist.entropy().mean()

                # 策略损失（PPO 裁剪）
                ratio = torch.exp(new_log_probs - old_log_probs[batch_idx])
                surr1 = ratio * advantages[batch_idx]
                surr2 = torch.clamp(
                    ratio,
                    1 - self.config.clip_epsilon,
                    1 + self.config.clip_epsilon,
                ) * advantages[batch_idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                # 价值损失（多目标）- 每个目标独立计算
                # returns 是标量，分解为多目标回报
                n_obj = values.size(1)
                rw = self.config.reward_weights or {"throughput": 0.5, "delay": 0.3, "interference": 0.2}
                w = torch.tensor(
                    [rw.get("throughput", 0.5), rw.get("delay", 0.3), rw.get("interference", 0.2)],
                    device=values.device
                )
                multi_returns = returns[batch_idx].unsqueeze(1) * w.unsqueeze(0)
                value_loss = F.mse_loss(values, multi_returns)

                # 总损失
                loss = (
                    policy_loss
                    + self.config.value_coeff * value_loss
                    - self.config.entropy_coeff * entropy
                )

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()

        # 清空缓冲区
        self.buffer.clear()

        n_updates = self.config.n_epochs * max(1, len(states) // self.config.batch_size)
        return {
            "loss": total_loss / n_updates,
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }

    def save(self, path: str) -> None:
        """保存模型"""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
