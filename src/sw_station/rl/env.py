"""短波台站调度强化学习环境 - ShortwaveStationEnv"""

from __future__ import annotations

from typing import Optional, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..models.station import StationDigitalTwin, CommunicationTask
from ..models.channel import ChannelState, IonosphericState
from ..simulation.em_simulator import EMSimulator
from ..config import SystemConfig


class ShortwaveStationEnv(gym.Env):
    """
    短波台站调度强化学习环境

    将台站任务调度建模为马尔可夫决策过程（MDP），支持多目标优化。

    状态空间：
        - 天线状态：每副天线的工作状态、频率、功率
        - 信道状态：MUF、LUF、SNR
        - 任务队列：待分配任务的特征

    动作空间：
        - 天线 ID
        - 频道 ID
        - 功率等级

    奖励函数：
        - r1: 吞吐量奖励
        - r2: 延迟惩罚
        - r3: 同址干扰惩罚
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        config: Optional[SystemConfig] = None,
        n_antennas: int = 50,
        n_channels: int = 100,
        n_pending_tasks: int = 10,
        max_steps: int = 1000,
        render_mode: Optional[str] = None,
    ):
        """
        初始化强化学习环境

        Parameters
        ----------
        config : SystemConfig, optional
            系统配置
        n_antennas : int
            天线数量
        n_channels : int
            信道数量
        n_pending_tasks : int
            待处理任务数量
        max_steps : int
            最大步数
        render_mode : str, optional
            渲染模式
        """
        super().__init__()

        self.config = config or SystemConfig.default()
        self.n_antennas = n_antennas
        self.n_channels = n_channels
        self.n_pending_tasks = n_pending_tasks
        self.max_steps = max_steps
        self.render_mode = render_mode

        # 台站模型
        self.station = StationDigitalTwin(n_antennas)
        self.em_simulator = EMSimulator()

        # 电离层状态
        self.ionosphere = IonosphericState()

        # 动作空间：(天线ID, 频道ID, 功率等级)
        self.action_space = spaces.MultiDiscrete([
            n_antennas,
            n_channels,
            10,  # 功率等级
        ])

        # 状态空间
        self.observation_space = spaces.Dict({
            # 天线状态：[是否发射, 当前频率归一化, 当前功率归一化]
            "antenna_status": spaces.Box(
                low=0, high=1, shape=(n_antennas, 3), dtype=np.float32
            ),
            # 信道状态：[质量评分, 可用度, SNR归一化]
            "channel_quality": spaces.Box(
                low=0, high=1, shape=(n_channels, 3), dtype=np.float32
            ),
            # 任务队列：[目标方位角, 目标仰角, 优先级, 紧急度]
            "pending_tasks": spaces.Box(
                low=0, high=1, shape=(n_pending_tasks, 4), dtype=np.float32
            ),
            # 电离层状态：[MUF归一化, LUF归一化, 太阳活动]
            "ionosphere": spaces.Box(
                low=0, high=1, shape=(4,), dtype=np.float32
            ),
        })

        # 环境状态
        self.current_step = 0
        self.task_queue: list[CommunicationTask] = []
        self.completed_tasks: list[CommunicationTask] = []
        self.failed_tasks: list[CommunicationTask] = []

        # 统计信息
        self.total_throughput = 0.0
        self.total_delay = 0.0
        self.total_interference = 0.0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """
        重置环境

        Returns
        -------
        tuple[dict, dict]
            (初始观测, 信息字典)
        """
        super().reset(seed=seed)

        # 重置台站
        self.station = StationDigitalTwin(self.n_antennas)
        self.station.initialize_random_layout()

        # 重置电离层
        self.ionosphere = IonosphericState()
        self.ionosphere.solar_sunspot_number = self.np_random.uniform(20, 150)

        # 生成初始任务队列
        self.task_queue = self._generate_tasks(self.n_pending_tasks)
        self.completed_tasks = []
        self.failed_tasks = []

        # 重置统计
        self.current_step = 0
        self.total_throughput = 0.0
        self.total_delay = 0.0
        self.total_interference = 0.0

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """
        执行一步动作

        Parameters
        ----------
        action : np.ndarray
            动作 [天线ID, 频道ID, 功率等级]

        Returns
        -------
        tuple
            (观测, 奖励, 终止, 截断, 信息)
        """
        antenna_id, channel_id, power_level = action

        # 解析动作
        power_dbm = 10 + power_level * 3  # 10-37 dBm

        # 获取当前任务
        if len(self.task_queue) == 0:
            # 没有待处理任务，生成新任务
            self.task_queue = self._generate_tasks(self.n_pending_tasks)

        current_task = self.task_queue[0]

        # 执行调度
        success = self._execute_scheduling(
            antenna_id, channel_id, power_dbm, current_task
        )

        # 计算奖励
        reward, reward_info = self._calculate_reward(
            antenna_id, channel_id, power_dbm, current_task, success
        )

        # 更新状态
        if success:
            current_task.status = "completed"
            current_task.assigned_antenna = f"ANT_{antenna_id:03d}"
            current_task.assigned_frequency = self._channel_to_frequency(channel_id)
            self.completed_tasks.append(current_task)
        else:
            current_task.status = "failed"
            self.failed_tasks.append(current_task)

        self.task_queue.pop(0)

        # 生成新任务
        new_tasks = self._generate_tasks(1)
        self.task_queue.extend(new_tasks)

        # 更新电离层状态
        self._update_ionosphere()

        # 更新步数
        self.current_step += 1

        # 检查终止条件
        terminated = self.current_step >= self.max_steps
        truncated = False

        obs = self._get_observation()
        info = self._get_info()
        info.update(reward_info)

        return obs, reward, terminated, truncated, info

    def _execute_scheduling(
        self,
        antenna_id: int,
        channel_id: int,
        power_dbm: float,
        task: CommunicationTask,
    ) -> bool:
        """
        执行调度动作

        Parameters
        ----------
        antenna_id : int
            天线 ID
        channel_id : int
            信道 ID
        power_dbm : float
            发射功率 (dBm)
        task : CommunicationTask
            通信任务

        Returns
        -------
        bool
            是否成功
        """
        antenna = self.station.antennas[antenna_id]
        frequency = self._channel_to_frequency(channel_id)

        # 检查天线是否已被占用
        if antenna.is_transmitting:
            return False

        # 检查频率是否在 MUF 范围内
        muf = self.ionosphere.muf_3000
        if frequency > muf:
            return False

        # 检查同址干扰
        for i, ant in enumerate(self.station.antennas):
            if i == antenna_id or not ant.is_transmitting:
                continue

            isolation = self.em_simulator.calculate_isolation(
                antenna, ant, frequency
            )
            interference = power_dbm - isolation

            if interference > self.config.interference.max_allowed_interference:
                return False

        # 执行分配
        antenna.is_transmitting = True
        antenna.current_frequency = frequency
        antenna.current_power = power_dbm

        return True

    def _calculate_reward(
        self,
        antenna_id: int,
        channel_id: int,
        power_dbm: float,
        task: CommunicationTask,
        success: bool,
    ) -> tuple[float, dict]:
        """
        计算多目标奖励

        Parameters
        ----------
        antenna_id, channel_id : int
            动作参数
        power_dbm : float
            发射功率
        task : CommunicationTask
            通信任务
        success : bool
            是否成功

        Returns
        -------
        tuple[float, dict]
            (总奖励, 奖励详情)
        """
        # 吞吐量奖励
        if success:
            r1 = 1.0 + task.priority * 0.1
        else:
            r1 = -0.1

        # 延迟惩罚
        r2 = -0.01 * self.current_step / self.max_steps

        # 同址干扰惩罚
        r3 = self._calculate_interference_penalty(antenna_id, power_dbm)

        # 标量化
        w1, w2, w3 = self.config.rl.reward_weights
        total_reward = w1 * r1 + w2 * r2 + w3 * r3

        reward_info = {
            "reward_throughput": r1,
            "reward_delay": r2,
            "reward_interference": r3,
            "success": success,
        }

        return total_reward, reward_info

    def _calculate_interference_penalty(
        self,
        antenna_id: int,
        power_dbm: float,
    ) -> float:
        """计算同址干扰惩罚"""
        penalty = 0.0
        antenna = self.station.antennas[antenna_id]

        for i, ant in enumerate(self.station.antennas):
            if i == antenna_id or not ant.is_transmitting:
                continue

            isolation = self.em_simulator.calculate_isolation(
                antenna, ant, 15.0
            )
            interference = power_dbm - isolation

            if interference > self.config.interference.max_allowed_interference:
                excess = interference - self.config.interference.max_allowed_interference
                penalty -= excess / 10.0

        return penalty

    def _get_observation(self) -> dict[str, np.ndarray]:
        """构建观测向量"""
        # 天线状态
        antenna_status = np.zeros((self.n_antennas, 3), dtype=np.float32)
        for i, ant in enumerate(self.station.antennas):
            antenna_status[i, 0] = 1.0 if ant.is_transmitting else 0.0
            antenna_status[i, 1] = (ant.current_frequency or 0) / 30.0
            antenna_status[i, 2] = (ant.current_power or 0) / 50.0

        # 信道质量（简化）
        channel_quality = np.random.uniform(0.3, 1.0, (self.n_channels, 3)).astype(np.float32)

        # 任务队列
        pending_tasks = np.zeros((self.n_pending_tasks, 4), dtype=np.float32)
        for i, task in enumerate(self.task_queue[:self.n_pending_tasks]):
            pending_tasks[i, 0] = task.target_azimuth / 360.0
            pending_tasks[i, 1] = task.target_elevation / 90.0
            pending_tasks[i, 2] = task.priority / 5.0
            pending_tasks[i, 3] = 1.0  # 紧急度

        # 电离层状态
        ionosphere = np.array([
            self.ionosphere.muf_3000 / 30.0,
            self.ionosphere.fof2 / 15.0,
            self.ionosphere.solar_sunspot_number / 200.0,
            0.5,  # 占位
        ], dtype=np.float32)

        return {
            "antenna_status": antenna_status,
            "channel_quality": channel_quality,
            "pending_tasks": pending_tasks,
            "ionosphere": ionosphere,
        }

    def _get_info(self) -> dict[str, Any]:
        """获取信息字典"""
        return {
            "step": self.current_step,
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "total_throughput": self.total_throughput,
            "ionosphere_muf": self.ionosphere.muf_3000,
        }

    def _generate_tasks(self, n_tasks: int) -> list[CommunicationTask]:
        """生成随机通信任务"""
        tasks = []
        for _ in range(n_tasks):
            task = CommunicationTask(
                task_id=f"TASK_{self.current_step}_{len(tasks)}",
                target_azimuth=self.np_random.uniform(0, 360),
                target_elevation=self.np_random.uniform(5, 60),
                freq_range=(2.0, 30.0),
                priority=int(self.np_random.integers(1, 6)),
            )
            tasks.append(task)
        return tasks

    def _channel_to_frequency(self, channel_id: int) -> float:
        """信道 ID 转换为频率"""
        freq_min = self.config.frequency.freq_min
        freq_max = self.config.frequency.freq_max
        return freq_min + (freq_max - freq_min) * channel_id / self.n_channels

    def _update_ionosphere(self) -> None:
        """更新电离层状态"""
        # 简化的昼夜变化
        hour = (self.current_step / 100) % 24
        if 6 <= hour <= 18:
            self.ionosphere.fof2 = 8.0 + 2.0 * np.sin(np.pi * (hour - 6) / 12)
        else:
            self.ionosphere.fof2 = 6.0

    def render(self) -> Optional[str]:
        """渲染环境状态"""
        if self.render_mode == "ansi":
            return (
                f"Step: {self.current_step}, "
                f"Completed: {len(self.completed_tasks)}, "
                f"Failed: {len(self.failed_tasks)}"
            )
        return None

    def close(self) -> None:
        """关闭环境"""
        pass
