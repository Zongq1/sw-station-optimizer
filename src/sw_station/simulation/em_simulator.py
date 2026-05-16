"""电磁仿真接口 - EMSimulator 隔离度计算与电磁耦合分析"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..models.antenna import AntennaPatternCube
from ..models.station import AntennaDevice


class EMSimulator:
    """
    电磁仿真接口封装

    提供基于天线方向图和空间位置的隔离度计算，
    以及地面反射、自由空间路径损耗等电磁传播特性计算。
    """

    # 物理常数
    SPEED_OF_LIGHT = 3e8  # m/s

    def __init__(
        self,
        ground_conductivity: float = 0.01,  # S/m
        ground_permittivity: float = 13.0,
        iip3: float = 10.0,
    ):
        """
        初始化电磁仿真器

        Parameters
        ----------
        ground_conductivity : float
            地表电导率 (S/m)
        ground_permittivity : float
            地表相对介电常数
        iip3 : float
            三阶输入截点 (dBm)
        """
        self.ground_conductivity = ground_conductivity
        self.ground_permittivity = ground_permittivity
        self.iip3 = iip3  # dBm, 三阶输入截点

    def calculate_isolation(
        self,
        tx_antenna: AntennaDevice,
        rx_antenna: AntennaDevice,
        frequency: float,
    ) -> float:
        """
        计算两天线间的空间隔离度

        Parameters
        ----------
        tx_antenna : AntennaDevice
            发射天线
        rx_antenna : AntennaDevice
            接收天线
        frequency : float
            工作频率 (MHz)

        Returns
        -------
        float
            隔离度 (dB)
        """
        # 计算距离和方向
        distance = tx_antenna.distance_to(rx_antenna)
        if distance < 1e-3:
            return 0.0  # 同位置，隔离度为0

        az_tx, el_tx = tx_antenna.direction_to(rx_antenna)
        az_rx, el_rx = rx_antenna.direction_to(tx_antenna)

        # 获取天线增益
        g_tx = tx_antenna.get_gain_at(frequency, az_tx, el_tx)
        g_rx = rx_antenna.get_gain_at(frequency, az_rx, el_rx)

        # 自由空间路径损耗
        fspl = self.calculate_fspl(frequency, distance)

        # 地面反射损耗
        ground_loss = self._ground_reflection_loss(
            frequency, distance, el_tx, el_rx
        )

        # 总隔离度 = FSPL - G_tx - G_rx + 地面损耗
        isolation = fspl - g_tx - g_rx + ground_loss

        return max(isolation, 0.0)  # 隔离度非负

    def calculate_isolation_batch(
        self,
        antennas: list[AntennaDevice],
        frequency: float,
    ) -> np.ndarray:
        """
        批量计算所有天线对的隔离度矩阵

        Parameters
        ----------
        antennas : list[AntennaDevice]
            天线设备列表
        frequency : float
            工作频率 (MHz)

        Returns
        -------
        np.ndarray
            隔离度矩阵, shape: (n, n)
            isolation[i, j] = 从天线j到天线i的隔离度
        """
        n = len(antennas)
        isolation_matrix = np.full((n, n), np.inf)

        for i in range(n):
            for j in range(n):
                if i != j:
                    isolation_matrix[i, j] = self.calculate_isolation(
                        antennas[j], antennas[i], frequency
                    )

        return isolation_matrix

    def calculate_fspl(
        self,
        frequency_mhz: float,
        distance_m: float,
    ) -> float:
        """
        计算自由空间路径损耗

        FSPL = 20*log10(4*pi*d/lambda)

        Parameters
        ----------
        frequency_mhz : float
            频率 (MHz)
        distance_m : float
            距离 (米)

        Returns
        -------
        float
            路径损耗 (dB)
        """
        wavelength = self.SPEED_OF_LIGHT / (frequency_mhz * 1e6)
        if wavelength < 1e-10 or distance_m < 1e-10:
            return 0.0
        return 20 * np.log10(4 * np.pi * distance_m / wavelength)

    def _ground_reflection_loss(
        self,
        frequency: float,
        distance: float,
        el_tx: float,
        el_rx: float,
    ) -> float:
        """计算地面反射损耗 - 基于 Fresnel 反射系数"""
        sigma = self.ground_conductivity
        epsilon_r = self.ground_permittivity

        # 相对介电常数（复数）
        # epsilon_c = epsilon_r - j*sigma/(omega*epsilon_0)
        omega = 2 * np.pi * frequency * 1e6
        epsilon_0 = 8.854e-12
        epsilon_c_imag = sigma / (omega * epsilon_0)

        # 入射角（取平均仰角）
        theta_i = np.radians(max(1.0, min(el_tx, el_rx)))
        cos_theta = np.cos(theta_i)
        sin_theta = np.sin(theta_i)

        # 垂直极化 Fresnel 反射系数
        sqrt_term = np.sqrt(epsilon_r - sin_theta**2 + 1j * epsilon_c_imag)
        R_v = (epsilon_r * cos_theta - sqrt_term) / (epsilon_r * cos_theta + sqrt_term)

        # 反射损耗 = -20*log10(|R|)，低仰角时接近 0 dB（全反射），高仰角时较大
        reflection_coeff = abs(R_v)
        reflection_coeff = np.clip(reflection_coeff, 0.01, 1.0)
        ground_loss = -20 * np.log10(reflection_coeff)

        # 距离因子：距离越远，地面反射路径影响越小
        dist_factor = 1.0 / (1.0 + distance / 1000.0)

        return max(ground_loss * dist_factor, 0.0)

    def calculate_mutual_coupling(
        self,
        antenna1: AntennaDevice,
        antenna2: AntennaDevice,
        frequency: float,
    ) -> complex:
        """
        计算两天线间的互耦系数（简化模型）

        Parameters
        ----------
        antenna1, antenna2 : AntennaDevice
            天线设备
        frequency : float
            频率 (MHz)

        Returns
        -------
        complex
            互耦系数 S21
        """
        isolation_db = self.calculate_isolation(antenna1, antenna2, frequency)
        # S21 幅值
        s21_mag = 10 ** (-isolation_db / 20)
        # 相位由传播路径长度决定: phi = 2*pi*d/lambda
        distance = antenna1.distance_to(antenna2)
        wavelength = self.SPEED_OF_LIGHT / (frequency * 1e6)
        s21_phase = 2 * np.pi * distance / wavelength
        return s21_mag * np.exp(1j * s21_phase)

    def calculate_near_field_boundary(
        self,
        frequency_mhz: float,
        max_dimension: float = 20.0,
    ) -> float:
        """
        计算近场-远场边界距离

        远场条件: r > 2*D^2/lambda

        Parameters
        ----------
        frequency_mhz : float
            频率 (MHz)
        max_dimension : float
            天线最大物理尺寸 (米)

        Returns
        -------
        float
            远场边界距离 (米)
        """
        wavelength = self.SPEED_OF_LIGHT / (frequency_mhz * 1e6)
        return 2 * max_dimension**2 / wavelength

    def check_mutual_coupling_regime(
        self,
        antenna1: AntennaDevice,
        antenna2: AntennaDevice,
        frequency: float,
    ) -> str:
        """
        判断两天线间的耦合区域

        Parameters
        ----------
        antenna1, antenna2 : AntennaDevice
            天线设备
        frequency : float
            频率 (MHz)

        Returns
        -------
        str
            "near_field" 或 "far_field"
        """
        distance = antenna1.distance_to(antenna2)
        max_dim = max(antenna1.max_dimension, antenna2.max_dimension)
        boundary = self.calculate_near_field_boundary(frequency, max_dim)
        return "near_field" if distance < boundary else "far_field"

    def calculate_intermodulation_power(
        self,
        tx_powers: list[float],
        isolations: list[float],
        order: int = 3,
    ) -> float:
        """
        计算互调产物功率（简化模型）

        Parameters
        ----------
        tx_powers : list[float]
            发射功率列表 (dBm)
        isolations : list[float]
            对应的隔离度列表 (dB)
        order : int
            互调阶数

        Returns
        -------
        float
            互调产物功率 (dBm)
        """
        if len(tx_powers) < 2:
            return -np.inf

        # 简化模型：P_IM = P1 + P2 - (n-1)*IIP3
        iip3 = self.iip3

        # 考虑隔离度后的等效输入功率
        effective_powers = [p - iso for p, iso in zip(tx_powers, isolations)]

        # 三阶互调估算
        if order == 3:
            im_power = 2 * effective_powers[0] + effective_powers[1] - 2 * iip3
        elif order == 5:
            im_power = 3 * effective_powers[0] + 2 * effective_powers[1] - 4 * iip3
        else:
            im_power = -np.inf

        return im_power


class FrequencyDependentEMSimulator(EMSimulator):
    """
    频率依赖的电磁仿真器

    扩展基础仿真器，支持在多个频率点上进行仿真。
    """

    def calculate_isolation_vs_frequency(
        self,
        tx_antenna: AntennaDevice,
        rx_antenna: AntennaDevice,
        frequencies: np.ndarray,
    ) -> np.ndarray:
        """
        计算频率扫描的隔离度曲线

        Parameters
        ----------
        tx_antenna, rx_antenna : AntennaDevice
            天线设备
        frequencies : np.ndarray
            频率点数组 (MHz)

        Returns
        -------
        np.ndarray
            隔离度数组 (dB)
        """
        isolations = np.zeros(len(frequencies))
        for i, freq in enumerate(frequencies):
            isolations[i] = self.calculate_isolation(tx_antenna, rx_antenna, freq)
        return isolations

    def calculate_isolation_matrix_multi_freq(
        self,
        antennas: list[AntennaDevice],
        frequencies: np.ndarray,
    ) -> dict[float, np.ndarray]:
        """
        计算多频率的隔离度矩阵

        Parameters
        ----------
        antennas : list[AntennaDevice]
            天线列表
        frequencies : np.ndarray
            频率点数组 (MHz)

        Returns
        -------
        dict[float, np.ndarray]
            频率到隔离度矩阵的映射
        """
        result = {}
        for freq in frequencies:
            result[float(freq)] = self.calculate_isolation_batch(antennas, freq)
        return result

    def find_worst_frequency(
        self,
        tx_antenna: AntennaDevice,
        rx_antenna: AntennaDevice,
        freq_range: tuple[float, float] = (2.0, 30.0),
        n_points: int = 100,
    ) -> tuple[float, float]:
        """
        找到隔离度最差的频率

        Parameters
        ----------
        tx_antenna, rx_antenna : AntennaDevice
            天线设备
        freq_range : tuple
            频率范围 (MHz)
        n_points : int
            采样点数

        Returns
        -------
        tuple[float, float]
            (最差频率 MHz, 最差隔离度 dB)
        """
        frequencies = np.linspace(freq_range[0], freq_range[1], n_points)
        isolations = self.calculate_isolation_vs_frequency(
            tx_antenna, rx_antenna, frequencies
        )
        min_idx = np.argmin(isolations)
        return float(frequencies[min_idx]), float(isolations[min_idx])
