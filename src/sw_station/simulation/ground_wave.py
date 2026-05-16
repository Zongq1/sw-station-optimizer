"""地波传播预测 - ITU-R P.368 简化实现"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class GroundWaveResult:
    """地波传播计算结果"""
    frequency: float        # MHz
    distance: float         # km
    field_strength: float   # dB(uV/m)
    path_loss: float        # dB
    max_range: float        # km
    is_feasible: bool


class GroundWavePropagation:
    """
    地波传播预测引擎

    基于 ITU-R P.368-9 规范的简化实现，用于预测短波地波传播特性。
    地波传播适用于近距离通信（通常 < 100 km）。
    """

    # 地球半径 (km)
    EARTH_RADIUS = 6371.0

    def __init__(
        self,
        ground_conductivity: float = 0.01,  # S/m
        ground_permittivity: float = 13.0,
    ):
        """
        初始化地波传播引擎

        Parameters
        ----------
        ground_conductivity : float
            地表电导率 (S/m)
        ground_permittivity : float
            地表相对介电常数
        """
        self.ground_conductivity = ground_conductivity
        self.ground_permittivity = ground_permittivity

    def calculate_field_strength(
        self,
        frequency: float,
        distance: float,
        tx_power_dbw: float = 0.0,
        tx_gain_dbi: float = 0.0,
    ) -> float:
        """
        计算地波场强

        Parameters
        ----------
        frequency : float
            频率 (MHz)
        distance : float
            距离 (km)
        tx_power_dbw : float
            发射功率 (dBW)
        tx_gain_dbi : float
            发射天线增益 (dBi)

        Returns
        -------
        float
            场强 (dB(uV/m))
        """
        # 标准化距离参数
        d_km = max(distance, 0.001)

        # 基于 Millington 方法的简化模型
        # 场强随距离衰减，受地面电导率影响

        # 自由空间场强参考值
        # E = 106.9 - 20*log10(d) + P_tx + G_tx (dB(uV/m) at 1 MHz, 1 kW)
        e_ref = 106.9 - 20 * np.log10(d_km) + tx_power_dbw + tx_gain_dbi

        # 频率修正
        freq_correction = -20 * np.log10(frequency)

        # 地面损耗修正
        ground_correction = self._ground_loss_correction(frequency, d_km)

        field_strength = e_ref + freq_correction + ground_correction

        return field_strength

    def _ground_loss_correction(
        self,
        frequency: float,
        distance: float,
    ) -> float:
        """
        计算地面损耗修正

        Parameters
        ----------
        frequency : float
            频率 (MHz)
        distance : float
            距离 (km)

        Returns
        -------
        float
            损耗修正 (dB)
        """
        # 简化模型
        sigma = self.ground_conductivity

        # 归一化参数
        # 高电导率（海水）损耗小，低电导率（干燥地面）损耗大
        if sigma > 1.0:  # 海水
            base_loss = -5.0
        elif sigma > 0.01:  # 良好地面
            base_loss = -15.0
        elif sigma > 0.001:  # 一般地面
            base_loss = -25.0
        else:  # 干燥地面
            base_loss = -35.0

        # 频率越高，地面损耗越大
        freq_factor = 10 * np.log10(frequency)

        # 距离因子
        dist_factor = 5 * np.log10(distance)

        return base_loss + freq_factor + dist_factor

    def calculate_path_loss(
        self,
        frequency: float,
        distance: float,
    ) -> float:
        """
        计算地波传播路径损耗

        Parameters
        ----------
        frequency : float
            频率 (MHz)
        distance : float
            距离 (km)

        Returns
        -------
        float
            路径损耗 (dB)
        """
        # 地波路径损耗模型
        # 近距离：接近自由空间损耗
        # 远距离：损耗增加更快

        # 自由空间损耗
        fspl = 32.4 + 20 * np.log10(frequency) + 20 * np.log10(distance)

        # 附加地面损耗
        ground_excess_loss = self._excess_loss(frequency, distance)

        return fspl + ground_excess_loss

    def _excess_loss(self, frequency: float, distance: float) -> float:
        """
        计算相对于自由空间的附加损耗

        Parameters
        ----------
        frequency : float
            频率 (MHz)
        distance : float
            距离 (km)

        Returns
        -------
        float
            附加损耗 (dB)
        """
        sigma = self.ground_conductivity

        # 数值距离参数
        nu = (np.pi * distance * 1e3 * frequency * 1e6) / (3e8)
        nu = max(nu, 0.01)

        # 基于地面电导率的衰减系数
        alpha = 1.0 / (1.0 + sigma * 1e6 / (frequency * 1e6))

        # 附加损耗
        excess = alpha * 10 * np.log10(1 + nu)

        return max(excess, 0.0)

    def calculate_max_range(
        self,
        frequency: float,
        tx_power_dbw: float = 0.0,
        tx_gain_dbi: float = 0.0,
        min_field_strength: float = 20.0,  # dB(uV/m)
    ) -> float:
        """
        计算最大传播距离

        Parameters
        ----------
        frequency : float
            频率 (MHz)
        tx_power_dbw : float
            发射功率 (dBW)
        tx_gain_dbi : float
            天线增益 (dBi)
        min_field_strength : float
            最小可用场强 (dB(uV/m))

        Returns
        -------
        float
            最大传播距离 (km)
        """
        # 二分法搜索最大距离
        d_min = 0.1
        d_max = 500.0

        for _ in range(50):
            d_mid = (d_min + d_max) / 2
            e = self.calculate_field_strength(
                frequency, d_mid, tx_power_dbw, tx_gain_dbi
            )

            if e > min_field_strength:
                d_min = d_mid
            else:
                d_max = d_mid

        return (d_min + d_max) / 2

    def evaluate(
        self,
        frequency: float,
        distance: float,
        tx_power_dbw: float = 0.0,
        tx_gain_dbi: float = 0.0,
        min_snr: float = 10.0,
    ) -> GroundWaveResult:
        """
        评估地波传播

        Parameters
        ----------
        frequency : float
            频率 (MHz)
        distance : float
            距离 (km)
        tx_power_dbw : float
            发射功率 (dBW)
        tx_gain_dbi : float
            天线增益 (dBi)
        min_snr : float
            最小可用信噪比 (dB)

        Returns
        -------
        GroundWaveResult
            评估结果
        """
        field_strength = self.calculate_field_strength(
            frequency, distance, tx_power_dbw, tx_gain_dbi
        )

        path_loss = self.calculate_path_loss(frequency, distance)

        max_range = self.calculate_max_range(
            frequency, tx_power_dbw, tx_gain_dbi
        )

        # 简化可用性判断
        is_feasible = field_strength > 20.0  # dB(uV/m)

        return GroundWaveResult(
            frequency=frequency,
            distance=distance,
            field_strength=field_strength,
            path_loss=path_loss,
            max_range=max_range,
            is_feasible=is_feasible,
        )

    def frequency_scan(
        self,
        distance: float,
        freq_range: tuple[float, float] = (2.0, 30.0),
        n_points: int = 50,
        tx_power_dbw: float = 0.0,
        tx_gain_dbi: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        频率扫描分析

        Parameters
        ----------
        distance : float
            传播距离 (km)
        freq_range : tuple
            频率范围 (MHz)
        n_points : int
            扫描点数
        tx_power_dbw : float
            发射功率 (dBW)
        tx_gain_dbi : float
            天线增益 (dBi)

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (频率数组, 场强数组)
        """
        frequencies = np.linspace(freq_range[0], freq_range[1], n_points)
        field_strengths = np.zeros(n_points)

        for i, freq in enumerate(frequencies):
            field_strengths[i] = self.calculate_field_strength(
                freq, distance, tx_power_dbw, tx_gain_dbi
            )

        return frequencies, field_strengths

    def distance_scan(
        self,
        frequency: float,
        dist_range: tuple[float, float] = (1.0, 200.0),
        n_points: int = 50,
        tx_power_dbw: float = 0.0,
        tx_gain_dbi: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        距离扫描分析

        Parameters
        ----------
        frequency : float
            工作频率 (MHz)
        dist_range : tuple
            距离范围 (km)
        n_points : int
            扫描点数
        tx_power_dbw : float
            发射功率 (dBW)
        tx_gain_dbi : float
            天线增益 (dBi)

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (距离数组, 场强数组)
        """
        distances = np.linspace(dist_range[0], dist_range[1], n_points)
        field_strengths = np.zeros(n_points)

        for i, dist in enumerate(distances):
            field_strengths[i] = self.calculate_field_strength(
                frequency, dist, tx_power_dbw, tx_gain_dbi
            )

        return distances, field_strengths
