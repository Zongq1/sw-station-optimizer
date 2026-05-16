"""
短波台站强化学习训练示例

演示如何使用 MODDQN 和 Attention-MO-PPO 进行动态调度训练。
"""

import numpy as np
import time
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sw_station.rl.env import ShortwaveStationEnv
from sw_station.rl.agents.moddqn import MODDQNAgent, MODDQNConfig


def flatten_observation(obs: dict) -> np.ndarray:
    """将观测字典展平为向量"""
    arrays = []
    for key, value in obs.items():
        arrays.append(value.flatten())
    return np.concatenate(arrays)


def train_moddqn():
    """训练 MODDQN 智能体"""
    print("=" * 60)
    print("MODDQN 强化学习训练演示")
    print("=" * 60)

    # 创建环境
    n_antennas = 10
    n_channels = 50

    env = ShortwaveStationEnv(
        n_antennas=n_antennas,
        n_channels=n_channels,
        n_pending_tasks=5,
        max_steps=200,
    )

    # 计算状态和动作维度
    obs, _ = env.reset()
    state_dim = sum(v.size for v in obs.values())
    action_dim = n_antennas * n_channels * 10  # 简化动作空间

    # 使用 MultiDiscrete 动作的实际维度
    action_dims = [n_antennas, n_channels, 10]
    total_action_dim = n_antennas  # 简化：只选择天线

    print(f"环境配置:")
    print(f"  天线数量: {n_antennas}")
    print(f"  信道数量: {n_channels}")
    print(f"  状态维度: {state_dim}")
    print()

    # 创建智能体
    config = MODDQNConfig(
        learning_rate=1e-3,
        gamma=0.99,
        buffer_size=10000,
        batch_size=32,
        target_update_freq=100,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=5000,
        hidden_size=128,
    )

    agent = MODDQNAgent(
        state_dim=state_dim,
        action_dim=total_action_dim,
        config=config,
    )

    print("开始训练...")
    print()

    # 训练循环
    n_episodes = 10
    rewards_history = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        state = flatten_observation(obs)

        total_reward = 0
        steps = 0

        while True:
            # 选择动作
            action = agent.select_action(state)

            # 执行动作（简化动作到完整动作）
            full_action = np.array([action, action % n_channels, 5])
            next_obs, reward, terminated, truncated, info = env.step(full_action)

            next_state = flatten_observation(next_obs)
            done = terminated or truncated

            # 存储经验
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # 更新网络
            loss = agent.update()

            total_reward += reward
            steps += 1
            state = next_state

            if done:
                break

        rewards_history.append(total_reward)

        if (episode + 1) % 2 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(f"Episode {episode + 1}/{n_episodes}: "
                  f"Reward={total_reward:.2f}, "
                  f"Avg={avg_reward:.2f}, "
                  f"Steps={steps}, "
                  f"Epsilon={agent.epsilon:.3f}")

    print()
    print("训练完成！")
    print(f"最终平均奖励: {np.mean(rewards_history[-5:]):.2f}")


def train_random_baseline():
    """随机策略基准"""
    print("=" * 60)
    print("随机策略基准演示")
    print("=" * 60)

    env = ShortwaveStationEnv(
        n_antennas=10,
        n_channels=50,
        max_steps=100,
    )

    n_episodes = 5
    rewards = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0

        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        rewards.append(total_reward)
        print(f"Episode {episode + 1}: Reward={total_reward:.2f}")

    print(f"\n随机策略平均奖励: {np.mean(rewards):.2f}")


if __name__ == "__main__":
    print("短波台站多目标优化系统 - 强化学习训练演示")
    print()

    # 运行随机基准
    train_random_baseline()

    print("\n" + "=" * 60 + "\n")

    # 运行 MODDQN 训练
    train_moddqn()
