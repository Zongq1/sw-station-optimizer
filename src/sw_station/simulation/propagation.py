"""天波传播预测 - ITU-R P.533-14 简化实现"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..models.channel import (
    ChannelState,
    IonosphericState,
    LinkBudget,
    PropagationMode,
    PropagationPath,
)


@dataclass
class PropagationResult:
    """传播计算结果"""
    frequency: float           # MHz
    muf: float                 # MHz
    luf: float                 # MHz
    path_loss: float           # dB
    received_power: float      # dBm
    snr: float                 # dB
    availability: float        # 0-1
    number_of_hops: int
    elevation_angle: float     # 度
    link_margin: float         # dB
    ionospheric_absorption: float  # dB
    propagation_mode: PropagationMode


class SkyWavePropagation:
    """
    天波传播预测引擎

    基于 ITU-R P.533-14 规范的简化实现，用于预测短波天波传播特性。
    """

    # 地球半径 (km)
    EARTH_RADIUS = 6371.0
    # F2层典型高度 (km)
    F2_LAYER_HEIGHT = 300.0
    # E层典型高度 (km)
    E_LAYER_HEIGHT = 110.0

    def __init__(self):
        """初始化传播预测引擎"""
        pass

    def calculate_muf(
        self,
        distance_km: float,
        ionosphere: IonosphericState,
        n_hops: Optional[int] = None,
    ) -> float:
        """
        计算最高可用频率 (MUF)

        MUF = foF2 * M(3000)F2 * factor

        Parameters
        ----------
        distance_km : float
            传播距离 (km)
        ionosphere : IonosphericState
            电离层状态
        n_hops : int, optional
            传播跳数

        Returns
        -------
        float
            MUF (MHz)
        """
        # 基础 MUF
        base_muf = ionosphere.muf_3000

        # 距离修正因子
        if n_hops is None:
            n_hops = PropagationPath.estimate_hops(distance_km)

        # 每跳距离
        hop_distance = distance_km / max(n_hops, 1)

        # 距离因子（距离越远，MUF 越高）
        distance_factor = 1.0 + 0.1 * np.log10(hop_distance / 3000.0)
        distance_factor = np.clip(distance_factor, 0.8, 1.2)

        # 时段修正 - 基于电离层 foF2 的昼夜变化
        # 白天 foF2 高 -> MUF 高，夜间 foF2 低 -> MUF 低
        # 利用 ionosphere 已有的 foF2 和 muf_3000 关系
        # time_factor 基于 foF2 相对于典型值的比例
        typical_fof2 = 8.0  # 典型 foF2 (MHz)
        time_factor = np.clip(ionosphere.fof2 / typical_fof2, 0.6, 1.4)

        muf = base_muf * distance_factor * time_factor

        return max(muf, ionosphere.fof2)

    def calculate_luf(
        self,
        distance_km: float,
        ionosphere: IonosphericState,
        tx_power_dbm: float = 30.0,
        required_snr_db: float = 10.0,
        rx_sensitivity_dbm: float = -120.0,
    ) -> float:
        """
        计算最低可用频率 (LUF)

        LUF 由电离层吸收和接收机灵敏度共同决定。
        频率越低，电离层吸收越大，当接收功率低于灵敏度时即为 LUF。

        Parameters
        ----------
        distance_km : float
            传播距离 (km)
        ionosphere : IonosphericState
            电离层状态
        tx_power_dbm : float
            发射功率 (dBm)
        required_snr_db : float
            所需信噪比 (dB)
        rx_sensitivity_dbm : float
            接收机灵敏度 (dBm)

        Returns
        -------
        float
            LUF (MHz)
        """
        # 二分法搜索 LUF：找到使接收功率 = 灵敏度的频率
        f_low, f_high = 2.0, 15.0

        for _ in range(30):
            f_mid = (f_low + f_high) / 2

            # 计算该频率的路径损耗
            path_loss = self.calculate_path_loss(f_mid, distance_km, ionosphere)

            # 接收功率
            rx_power = tx_power_dbm - path_loss

            # 所需最小接收功率 = 灵敏度 + 所需SNR + 噪声余量
            min_rx_power = rx_sensitivity_dbm + required_snr_db

            if rx_power > min_rx_power:
                f_high = f_mid  # 频率可以更低
            else:
                f_low = f_mid   # 需要更高频率

        return max((f_low + f_high) / 2, 2.0)

    def calculate_path_loss(
        self,
        frequency: float,
        distance_km: float,
        ionosphere: IonosphericState,
        elevation_angle: float = 10.0,
    ) -> float:
        """
        计算天波传播路径损耗

        Parameters
        ----------
        frequency : float
            工作频率 (MHz)
        distance_km : float
            传播距离 (km)
        ionosphere : IonosphericState
            电离层状态
        elevation_angle : float
            出射仰角 (度)

        Returns
        -------
        float
            总路径损耗 (dB)
        """
        # 自由空间损耗
        fspl = self._free_space_loss(frequency, distance_km)

        # 电离层吸收损耗
        iono_absorption = self._ionospheric_absorption(
            frequency, distance_km, ionosphere
        )

        # 地面反射损耗（多跳时）
        n_hops = PropagationPath.estimate_hops(distance_km, elevation_angle)
        ground_loss = self._ground_reflection_loss(n_hops)

        # 极化耦合损耗 - 基于法拉第旋转效应
        # 低频时法拉第旋转大，极化损耗可达 10-20 dB
        # 高频时旋转小，损耗接近 0
        freq_factor = max(0.1, 1.0 - frequency / 30.0)
        solar_factor = 1.0 + 0.5 * (ionosphere.solar_sunspot_number / 200.0)
        polarization_loss = 6.0 * freq_factor * solar_factor

        # 多径衰落余量
        multipath_margin = self._multipath_fading_margin(distance_km)

        total_loss = (
            fspl + iono_absorption + ground_loss
            + polarization_loss + multipath_margin
        )

        return total_loss

    def _free_space_loss(self, frequency: float, distance: float) -> float:
        """计算自由空间路径损耗"""
        # FSPL = 32.4 + 20*log10(f_MHz) + 20*log10(d_km)
        if distance <= 0 or frequency <= 0:
            return 0.0
        return 32.4 + 20 * np.log10(frequency) + 20 * np.log10(distance)

    def _ionospheric_absorption(
        self,
        frequency: float,
        distance: float,
        ionosphere: IonosphericState,
    ) -> float:
        """
        计算电离层吸收损耗

        基于 ITU-R P.533 非偏离吸收模型：
        L_a = (sec(phi_i) / (f + f_L)^n) * A

        其中 phi_i 是电离层入射角，f_L 是地磁旋频率，n≈1.5-2
        """
        n_hops = PropagationPath.estimate_hops(distance)

        # 吸收系数与 foF2 正相关（电子密度越高吸收越大）
        # 与频率负相关（近似 f^(-1.5) 到 f^(-2)）
        # 使用 (foF2/f)^2 作为归一化吸收因子
        freq_ratio = ionosphere.fof2 / max(frequency, 1.0)
        absorption_per_hop = 15.0 * freq_ratio ** 1.5  # dB/hop

        # 太阳活动修正
        solar_factor = 1.0 + 0.3 * (ionosphere.solar_sunspot_number / 100.0)

        # E 层吸收（低频时更显著）
        e_layer_factor = ionosphere.foe / max(frequency, 1.0)
        e_absorption = 5.0 * e_layer_factor * solar_factor

        total_absorption = (absorption_per_hop * solar_factor + e_absorption) * n_hops

        return max(total_absorption, 0.0)

    def _ground_reflection_loss(self, n_hops: int) -> float:
        """计算地面反射损耗"""
        # 每次地面反射约 1-3 dB 损耗
        loss_per_hop = 2.0  # dB
        # 多跳需要 n-1 次地面反射
        return loss_per_hop * max(n_hops - 1, 0)

    def _multipath_fading_margin(self, distance: float) -> float:
        """计算多径衰落余量"""
        # 距离越远，多径效应越明显
        base_margin = 3.0  # dB
        distance_factor = 1.0 + 0.5 * np.log10(distance / 1000.0)
        return base_margin * distance_factor

    def calculate_link_budget(
        self,
        frequency: float,
        distance_km: float,
        ionosphere: IonosphericState,
        tx_power_dbm: float = 30.0,
        tx_gain_dbi: float = 10.0,
        rx_gain_dbi: float = 10.0,
        required_snr_db: float = 10.0,
    ) -> LinkBudget:
        """
        计算完整链路预算

        Parameters
        ----------
        frequency : float
            工作频率 (MHz)
        distance_km : float
            传播距离 (km)
        ionosphere : IonosphericState
            电离层状态
        tx_power_dbm : float
            发射功率 (dBm)
        tx_gain_dbi : float
            发射天线增益 (dBi)
        rx_gain_dbi : float
            接收天线增益 (dBi)
        required_snr_db : float
            所需信噪比 (dB)

        Returns
        -------
        LinkBudget
            链路预算结果
        """
        path_loss = self.calculate_path_loss(
            frequency, distance_km, ionosphere
        )

        iono_absorption = self._ionospheric_absorption(
            frequency, distance_km, ionosphere
        )

        n_hops = PropagationPath.estimate_hops(distance_km)
        ground_loss = self._ground_reflection_loss(n_hops)

        return LinkBudget(
            tx_power=tx_power_dbm,
            tx_antenna_gain=tx_gain_dbi,
            rx_antenna_gain=rx_gain_dbi,
            free_space_path_loss=self._free_space_loss(frequency, distance_km),
            ionospheric_absorption=iono_absorption,
            ground_reflection_loss=ground_loss,
            polarization_mismatch=3.0,
            feedline_loss=2.0,
            required_snr=required_snr_db,
        )

    def evaluate_channel(
        self,
        frequency: float,
        distance_km: float,
        ionosphere: IonosphericState,
        tx_power_dbm: float = 30.0,
        tx_gain_dbi: float = 10.0,
        rx_gain_dbi: float = 10.0,
    ) -> ChannelState:
        """
        评估信道状态

        Parameters
        ----------
        frequency : float
            工作频率 (MHz)
        distance_km : float
            传播距离 (km)
        ionosphere : IonosphericState
            电离层状态
        tx_power_dbm : float
            发射功率 (dBm)
        tx_gain_dbi, rx_gain_dbi : float
            天线增益 (dBi)

        Returns
        -------
        ChannelState
            信道状态
        """
        muf = self.calculate_muf(distance_km, ionosphere)
        luf = self.calculate_luf(distance_km, ionosphere, tx_power_dbm)

        link_budget = self.calculate_link_budget(
            frequency, distance_km, ionosphere,
            tx_power_dbm, tx_gain_dbi, rx_gain_dbi,
        )

        # SNR 计算
        snr = link_budget.link_margin

        # 可用度评估
        if frequency > muf or frequency < luf:
            availability = 0.0
        else:
            # 基于链路余量的可用度
            availability = self._calculate_availability(link_budget.link_margin)

        # 传播模式判断
        if distance_km < 100:
            prop_mode = PropagationMode.GROUND_WAVE
        elif distance_km < 300 and frequency < luf * 1.2:
            prop_mode = PropagationMode.MIXED  # 地波和天波共存
        elif frequency > muf * 0.85:
            prop_mode = PropagationMode.SKY_WAVE  # 接近 MUF，天波传播
        else:
            prop_mode = PropagationMode.SKY_WAVE  # 正常天波传播

        # 跳数
        n_hops = PropagationPath.estimate_hops(distance_km)

        # 多径时延
        multipath_delay = self._estimate_multipath_delay(distance_km, n_hops)

        return ChannelState(
            frequency=frequency,
            muf=muf,
            luf=luf,
            snr=snr,
            availability=availability,
            propagation_mode=prop_mode,
            distance=distance_km,
            number_of_hops=n_hops,
            link_margin=link_budget.link_margin,
            multipath_delay_spread=multipath_delay,
            ionospheric_absorption=link_budget.ionospheric_absorption,
        )

    def _calculate_availability(self, link_margin: float) -> float:
        """
        基于链路余量计算可用度

        使用 sigmoid 函数将链路余量映射到 [0, 1]
        """
        # 余量为 0 时可用度约 0.5，余量 10dB 时约 0.95
        return 1.0 / (1.0 + np.exp(-0.3 * link_margin))

    def _estimate_multipath_delay(self, distance: float, n_hops: int) -> float:
        """
        估算多径时延扩展 (ms)

        基于各跳几何路径差异：
        - 单跳模式下多径主要来自高角/低角射线
        - 多跳模式下各跳路径长度差异累积
        """
        if n_hops <= 1:
            # 单跳：高角和低角射线的路径差
            # 典型 0.5-2 ms
            return 0.5 + 1.5 * distance / 3000.0

        # 多跳：每跳的路径差累积
        hop_distance = distance / n_hops
        # 每跳多径时延 ~ hop_distance / (c * sin(el)) 差异
        base_per_hop = 0.3  # ms/hop
        return base_per_hop * n_hops * (1 + hop_distance / 2000.0)

    def find_optimal_frequency(
        self,
        distance_km: float,
        ionosphere: IonosphericState,
        freq_range: tuple[float, float] = (2.0, 30.0),
        n_points: int = 100,
    ) -> tuple[float, float]:
        """
        寻找最佳工作频率

        Parameters
        ----------
        distance_km : float
            传播距离 (km)
        ionosphere : IonosphericState
            电离层状态
        freq_range : tuple
            频率搜索范围 (MHz)
        n_points : int
            搜索点数

        Returns
        -------
        tuple[float, float]
            (最佳频率 MHz, 预期 SNR dB)
        """
        frequencies = np.linspace(freq_range[0], freq_range[1], n_points)
        muf = self.calculate_muf(distance_km, ionosphere)
        luf = self.calculate_luf(distance_km, ionosphere)

        # 过滤可用频率
        usable_mask = (frequencies >= luf) & (frequencies <= muf)
        usable_freqs = frequencies[usable_mask]

        if len(usable_freqs) == 0:
            # 没有可用频率，返回 LUF
            return luf, -np.inf

        # 在可用频率中找 SNR 最高的
        best_freq = usable_freqs[len(usable_freqs) // 2]  # 默认中间频率
        best_snr = -np.inf

        for freq in usable_freqs:
            channel = self.evaluate_channel(freq, distance_km, ionosphere)
            if channel.snr > best_snr:
                best_snr = channel.snr
                best_freq = freq

        return float(best_freq), float(best_snr)
