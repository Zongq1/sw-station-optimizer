"""奖励函数设计 - MultiObjectiveReward"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..models.station import AntennaDevice, CommunicationTask
from ..simulation.em_simulator import EMSimulator


@dataclass
class RewardWeights:
    """奖励权重配置"""
    throughput: float = 0.5
    delay: float = 0.3
    interference: float = 0.2


class MultiObjectiveReward:
    """
    多目标奖励计算器

    计算三个维度的奖励：
    - r1: 吞吐量/通信成功率
    - r2: 延迟惩罚
    - r3: 同址干扰惩罚
    """

    def __init__(
        self,
        weights: Optional[RewardWeights] = None,
        em_simulator: Optional[EMSimulator] = None,
    ):
        """
        初始化奖励计算器

        Parameters
        ----------
        weights : RewardWeights, optional
            奖励权重
        em_simulator : EMSimulator, optional
            电磁仿真器
        """
        self.weights = weights or RewardWeights()
        self.em_simulator = em_simulator or EMSimulator()

    def calculate(
        self,
        antenna: AntennaDevice,
        task: CommunicationTask,
        frequency: float,
        power_dbm: float,
        success: bool,
        delay_steps: int,
        active_antennas: list[AntennaDevice],
    ) -> tuple[float, dict]:
        """
        计算综合奖励

        Parameters
        ----------
        antenna : AntennaDevice
            分配的天线
        task : CommunicationTask
            通信任务
        frequency : float
            分配的频率 (MHz)
        power_dbm : float
            发射功率 (dBm)
        success : bool
            是否成功
        delay_steps : int
            延迟步数
        active_antennas : list[AntennaDevice]
            当前活跃的天线列表

        Returns
        -------
        tuple[float, dict]
            (总奖励, 奖励详情)
        """
        # 吞吐量奖励
        r1 = self._throughput_reward(antenna, task, frequency, success)

        # 延迟惩罚
        r2 = self._delay_penalty(delay_steps)

        # 同址干扰惩罚
        r3 = self._interference_penalty(antenna, power_dbm, active_antennas)

        # 标量化
        total = (
            self.weights.throughput * r1
            + self.weights.delay * r2
            + self.weights.interference * r3
        )

        details = {
            "throughput_reward": r1,
            "delay_penalty": r2,
            "interference_penalty": r3,
            "total_reward": total,
        }

        return total, details

    def _throughput_reward(
        self,
        antenna: AntennaDevice,
        task: CommunicationTask,
        frequency: float,
        success: bool,
    ) -> float:
        """
        计算吞吐量奖励

        Parameters
        ----------
        antenna : AntennaDevice
            天线
        task : CommunicationTask
            任务
        frequency : float
            频率
        success : bool
            是否成功

        Returns
        -------
        float
            吞吐量奖励
        """
        if not success:
            return -0.1

        # 基础奖励
        base_reward = 1.0

        # 增益匹配度奖励
        if antenna.pattern is not None:
            gain = antenna.get_gain_at(
                frequency, task.target_azimuth, task.target_elevation
            )
            peak_gain = antenna.pattern.get_peak_gain(frequency)[0]
            gain_ratio = gain / peak_gain if peak_gain > 0 else 0.5
            base_reward *= (0.5 + 0.5 * gain_ratio)

        # 优先级加成
        priority_bonus = task.priority * 0.1
        base_reward += priority_bonus

        return base_reward

    def _delay_penalty(self, delay_steps: int) -> float:
        """
        计算延迟惩罚

        Parameters
        ----------
        delay_steps : int
            延迟步数

        Returns
        -------
        float
            延迟惩罚（负值）
        """
        # 指数衰减的延迟惩罚
        return -0.01 * delay_steps

    def _interference_penalty(
        self,
        antenna: AntennaDevice,
        power_dbm: float,
        active_antennas: list[AntennaDevice],
    ) -> float:
        """
        计算同址干扰惩罚

        Parameters
        ----------
        antenna : AntennaDevice
            发射天线
        power_dbm : float
            发射功率
        active_antennas : list[AntennaDevice]
            活跃天线列表

        Returns
        -------
        float
            干扰惩罚（负值）
        """
        penalty = 0.0
        max_allowed = -130.0  # dBm

        for other in active_antennas:
            if other.id == antenna.id:
                continue

            isolation = self.em_simulator.calculate_isolation(
                antenna, other, 15.0
            )
            interference = power_dbm - isolation

            if interference > max_allowed:
                excess = interference - max_allowed
                penalty -= excess / 10.0

        return penalty


class ShapedReward(MultiObjectiveReward):
    """
    奖励塑形版本

    添加额外的引导信号加速学习。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev_completed = 0

    def calculate(
        self,
        antenna: AntennaDevice,
        task: CommunicationTask,
        frequency: float,
        power_dbm: float,
        success: bool,
        delay_steps: int,
        active_antennas: list[AntennaDevice],
        completed_tasks: int,
    ) -> tuple[float, dict]:
        """
        计算塑形奖励
        """
        base_reward, details = super().calculate(
            antenna, task, frequency, power_dbm, success,
            delay_steps, active_antennas
        )

        # 进度奖励
        progress = completed_tasks - self.prev_completed
        if progress > 0:
            base_reward += 0.5 * progress
        self.prev_completed = completed_tasks

        # 频率效率奖励（使用 FOT 附近频率）
        fot = 0.85 * 20.0  # 简化 FOT
        freq_efficiency = 1.0 - abs(frequency - fot) / fot
        base_reward += 0.1 * max(0, freq_efficiency)

        details["total_reward_shaped"] = base_reward
        return base_reward, details
