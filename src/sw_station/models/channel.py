"""信道状态模型 - ChannelState 数据结构"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class PropagationMode(Enum):
    """传播模式枚举"""
    GROUND_WAVE = "ground_wave"    # 地波传播
    SKY_WAVE = "sky_wave"          # 天波传播
    MIXED = "mixed"                # 混合传播
    LINE_OF_SIGHT = "los"          # 视距传播


class TimeOfDay(Enum):
    """时段枚举"""
    DAY = "day"        # 白天
    NIGHT = "night"    # 夜间
    DAWN = "dawn"      # 黎明
    DUSK = "dusk"      # 黄昏


@dataclass
class IonosphericState:
    """
    电离层状态参数

    基于 ITU-R P.1239 和 ITU-R P.533-14 规范
    """
    # F2层临界频率 foF2 (MHz)
    fof2: float = 8.0
    # F2层最大可用频率因子 M(3000)F2
    m3000f2: float = 3.0
    # F2层虚高 (km)
    h_prime_f2: float = 300.0
    # F1层临界频率 foF1 (MHz) - 夜间为0
    fof1: float = 5.0
    # E层临界频率 foE (MHz)
    foe: float = 3.0
    # E层虚高 (km)
    h_prime_e: float = 110.0
    # 太阳黑子数 R12
    solar_sunspot_number: float = 50.0
    # 太阳10.7cm射电通量 (sfu)
    solar_flux_107: float = 100.0

    @property
    def muf_3000(self) -> float:
        """F2层 3000km 最高可用频率"""
        return self.fof2 * self.m3000f2

    @property
    def typical_muf(self) -> float:
        """典型最高可用频率（考虑季节和时段修正）"""
        return self.muf_3000 * 0.85


@dataclass
class ChannelState:
    """
    信道状态数据结构

    描述特定频率-路径组合的信道状态，用于短波通信链路评估。
    """
    # 频率 (MHz)
    frequency: float
    # 最高可用频率 MUF (MHz)
    muf: float
    # 最低可用频率 LUF (MHz)
    luf: float
    # 接收信噪比 SNR (dB)
    snr: float
    # 信道可用度 (0-1)
    availability: float
    # 传播模式
    propagation_mode: PropagationMode
    # 传播距离 (km)
    distance: float = 0.0
    # 传播跳数
    number_of_hops: int = 1
    # 链路余量 (dB)
    link_margin: float = 0.0
    # 多径时延扩展 (ms)
    multipath_delay_spread: float = 0.0
    # 多普勒频移 (Hz)
    doppler_shift: float = 0.0
    # 电离层吸收损耗 (dB)
    ionospheric_absorption: float = 0.0
    # 时间戳
    timestamp: Optional[float] = None

    @property
    def is_usable(self) -> bool:
        """信道是否可用"""
        return self.availability > 0.5 and self.snr > 0

    @property
    def is_within_muf(self) -> bool:
        """频率是否在 MUF 以下"""
        return self.frequency < self.muf

    @property
    def is_above_luf(self) -> bool:
        """频率是否在 LUF 以上"""
        return self.frequency > self.luf

    @property
    def is_in_operating_range(self) -> bool:
        """频率是否在可用工作范围内 (LUF < f < MUF)"""
        return self.is_above_luf and self.is_within_muf

    @property
    def fot(self) -> float:
        """最佳工作频率 FOT (MHz)"""
        return self.muf * 0.85

    @property
    def capacity_estimate(self) -> float:
        """信道容量估算 (bps) - 基于 Shannon 公式"""
        bandwidth_hz = 3000  # 典型短波信道带宽 3kHz
        snr_linear = 10 ** (self.snr / 10)
        return bandwidth_hz * np.log2(1 + snr_linear)

    def quality_score(self) -> float:
        """
        计算信道质量综合评分

        Returns
        -------
        float
            质量评分 (0-1)
        """
        # SNR 归一化 (假设 0-40 dB 范围)
        snr_score = np.clip(self.snr / 40.0, 0, 1)

        # 可用度评分
        avail_score = self.availability

        # 链路余量评分 (假设 0-20 dB 范围)
        margin_score = np.clip(self.link_margin / 20.0, 0, 1)

        # 综合评分
        return 0.4 * snr_score + 0.3 * avail_score + 0.3 * margin_score


@dataclass
class LinkBudget:
    """
    链路预算计算

    用于评估短波通信链路的功率预算。
    """
    # 发射功率 (dBm)
    tx_power: float = 30.0  # 1W
    # 发射天线增益 (dBi)
    tx_antenna_gain: float = 10.0
    # 接收天线增益 (dBi)
    rx_antenna_gain: float = 10.0
    # 自由空间路径损耗 (dB)
    free_space_path_loss: float = 0.0
    # 电离层吸收 (dB)
    ionospheric_absorption: float = 0.0
    # 地面反射损耗 (dB)
    ground_reflection_loss: float = 0.0
    # 极化失配损耗 (dB)
    polarization_mismatch: float = 0.0
    # 馈线损耗 (dB)
    feedline_loss: float = 2.0
    # 系统噪声温度 (K)
    system_noise_temp: float = 1000.0
    # 所需 SNR (dB)
    required_snr: float = 10.0

    @property
    def eirp(self) -> float:
        """有效全向辐射功率 EIRP (dBm)"""
        return self.tx_power + self.tx_antenna_gain - self.feedline_loss

    @property
    def received_power(self) -> float:
        """接收功率 (dBm)"""
        total_loss = (
            self.free_space_path_loss
            + self.ionospheric_absorption
            + self.ground_reflection_loss
            + self.polarization_mismatch
            + self.feedline_loss
        )
        return self.eirp - total_loss + self.rx_antenna_gain

    @property
    def noise_power(self) -> float:
        """噪声功率 (dBm)"""
        # kTB, k=1.38e-23, B=3000Hz
        k = 1.38e-23
        B = 3000
        noise_watts = k * self.system_noise_temp * B
        return 10 * np.log10(noise_watts * 1000)  # 转换为 dBm

    @property
    def link_margin(self) -> float:
        """链路余量 (dB)"""
        return self.received_power - self.noise_power - self.required_snr

    @property
    def is_link_feasible(self) -> bool:
        """链路是否可行"""
        return self.link_margin > 0


@dataclass
class PropagationPath:
    """
    传播路径描述

    描述短波天波传播的完整路径参数。
    """
    # 发射端坐标 (lat, lon)
    tx_location: tuple[float, float]
    # 接收端坐标 (lat, lon)
    rx_location: tuple[float, float]
    # 大圆距离 (km)
    great_circle_distance: float
    # 传播跳数
    number_of_hops: int
    # F2层反射点位置
    reflection_points: list[tuple[float, float]] = field(default_factory=list)
    # 出射仰角 (度)
    elevation_angle: float = 10.0
    # 传播模式
    propagation_mode: PropagationMode = PropagationMode.SKY_WAVE

    @property
    def distance_per_hop(self) -> float:
        """每跳距离 (km)"""
        if self.number_of_hops == 0:
            return 0.0
        return self.great_circle_distance / self.number_of_hops

    @classmethod
    def estimate_hops(cls, distance_km: float, elevation_angle: float = 10.0,
                      f2_height_km: float = 300.0) -> int:
        """
        估算所需跳数

        Parameters
        ----------
        distance_km : float
            大圆距离 (km)
        elevation_angle : float
            出射仰角 (度)
        f2_height_km : float
            F2层高度 (km)

        Returns
        -------
        int
            估算跳数
        """
        # 简化估算：单跳距离
        elevation_rad = np.radians(elevation_angle)
        single_hop_distance = 2 * f2_height_km / np.tan(elevation_rad)

        if single_hop_distance <= 0:
            return 1

        n_hops = max(1, int(np.ceil(distance_km / single_hop_distance)))
        return n_hops
