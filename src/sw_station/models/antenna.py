"""天线设备模型 - AntennaPatternCube 三维方向图数据结构"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class AntennaType(Enum):
    """天线类型枚举"""
    YAGI = "yagi"                      # 八木-宇田天线
    LOG_PERIODIC = "log_periodic"      # 对数周期天线
    DIPOLE = "dipole"                  # 偶极天线
    MONOPOLE = "monopole"              # 单极天线（鞭状）
    RHOMBIC = "rhombic"                # 菱形天线
    CAGE = "cage"                      # 笼形天线
    ARRAY_VERTICAL = "array_vertical"  # 垂直阵列
    ARRAY_HORIZONTAL = "array_horizontal"  # 水平阵列


# 各类型天线的典型最大物理尺寸 (米)
ANTENNA_MAX_DIMENSIONS: dict[AntennaType, float] = {
    AntennaType.YAGI: 12.0,
    AntennaType.LOG_PERIODIC: 8.0,
    AntennaType.DIPOLE: 5.0,
    AntennaType.MONOPOLE: 10.0,
    AntennaType.RHOMBIC: 60.0,
    AntennaType.CAGE: 15.0,
    AntennaType.ARRAY_VERTICAL: 20.0,
    AntennaType.ARRAY_HORIZONTAL: 25.0,
}


@dataclass
class AntennaPatternCube:
    """
    天线三维方向图数据结构

    存储天线在不同频率、方位角、仰角下的增益响应，
    形成一个三维张量 PatternCube = G(f, az, el)。

    Parameters
    ----------
    antenna_id : str
        天线唯一标识符
    antenna_type : AntennaType
        天线类型
    frequencies : np.ndarray
        频率采样点数组 (MHz), shape: (n_freq,)
    azimuths : np.ndarray
        方位角采样点数组 (度), shape: (n_azimuth,)
    elevations : np.ndarray
        仰角采样点数组 (度), shape: (n_elevation,)
    gain_pattern : np.ndarray
        三维增益方向图 (dBi), shape: (n_freq, n_azimuth, n_elevation)
    impedance : np.ndarray, optional
        输入阻抗频谱 (Ohm), shape: (n_freq,) 或 (n_freq, 2) [实部, 虚部]
    polarization : np.ndarray, optional
        极化特性矩阵, shape: (n_freq, n_azimuth, n_elevation, 2) [theta, phi]
    """

    antenna_id: str
    antenna_type: AntennaType
    frequencies: np.ndarray  # MHz
    azimuths: np.ndarray     # 度 [0, 360)
    elevations: np.ndarray   # 度 [0, 90]
    gain_pattern: np.ndarray # dBi, shape: (n_freq, n_az, n_el)
    impedance: Optional[np.ndarray] = None
    polarization: Optional[np.ndarray] = None

    # 内部插值器缓存
    _interpolator: Optional[RegularGridInterpolator] = field(
        default=None, repr=False, init=False
    )

    def __post_init__(self):
        """验证数据维度一致性"""
        n_freq = len(self.frequencies)
        n_az = len(self.azimuths)
        n_el = len(self.elevations)

        expected_shape = (n_freq, n_az, n_el)
        if self.gain_pattern.shape != expected_shape:
            raise ValueError(
                f"gain_pattern shape {self.gain_pattern.shape} "
                f"不匹配预期 {expected_shape}"
            )

        # 构建插值器
        self._build_interpolator()

    def _build_interpolator(self) -> None:
        """构建三线性插值器"""
        self._interpolator = RegularGridInterpolator(
            (self.frequencies, self.azimuths, self.elevations),
            self.gain_pattern,
            method="linear",
            bounds_error=False,
            fill_value=None,  # 外推
        )

    def get_gain(self, freq: float, az: float, el: float) -> float:
        """
        获取指定频率和方向的天线增益（三线性插值）

        Parameters
        ----------
        freq : float
            工作频率 (MHz)
        az : float
            方位角 (度)
        el : float
            仰角 (度)

        Returns
        -------
        float
            天线增益 (dBi)
        """
        if self._interpolator is None:
            self._build_interpolator()

        # 归一化方位角到 [0, 360)
        az = az % 360.0

        point = np.array([[freq, az, el]])
        return float(self._interpolator(point)[0])

    def get_gain_batch(self, freq: float, az_array: np.ndarray,
                       el_array: np.ndarray) -> np.ndarray:
        """
        批量获取增益值

        Parameters
        ----------
        freq : float
            工作频率 (MHz)
        az_array : np.ndarray
            方位角数组 (度)
        el_array : np.ndarray
            仰角数组 (度)

        Returns
        -------
        np.ndarray
            增益数组 (dBi)
        """
        if self._interpolator is None:
            self._build_interpolator()

        n = len(az_array)
        points = np.column_stack([
            np.full(n, freq),
            az_array % 360.0,
            el_array,
        ])
        return self._interpolator(points)

    def get_peak_gain(self, freq: float) -> tuple[float, float, float]:
        """
        获取指定频率下的峰值增益及其方向

        Parameters
        ----------
        freq : float
            工作频率 (MHz)

        Returns
        -------
        tuple[float, float, float]
            (峰值增益 dBi, 方位角 度, 仰角 度)
        """
        # 找到最近的频率索引
        freq_idx = np.argmin(np.abs(self.frequencies - freq))

        # 在该频率切片上找最大值
        gain_slice = self.gain_pattern[freq_idx]
        max_idx = np.unravel_index(np.argmax(gain_slice), gain_slice.shape)

        peak_gain = gain_slice[max_idx]
        peak_az = self.azimuths[max_idx[0]]
        peak_el = self.elevations[max_idx[1]]

        return float(peak_gain), float(peak_az), float(peak_el)

    def get_front_to_back_ratio(self, freq: float, az: float, el: float) -> float:
        """
        计算指定方向的前后比

        Parameters
        ----------
        freq : float
            工作频率 (MHz)
        az : float
            主瓣方位角 (度)
        el : float
            主瓣仰角 (度)

        Returns
        -------
        float
            前后比 (dB)
        """
        forward_gain = self.get_gain(freq, az, el)
        backward_gain = self.get_gain(freq, (az + 180) % 360, el)
        return forward_gain - backward_gain

    def get_beamwidth(self, freq: float, az: float, el: float,
                      level_db: float = -3.0) -> tuple[float, float]:
        """
        估算指定方向的波束宽度

        Parameters
        ----------
        freq : float
            工作频率 (MHz)
        az : float
            主瓣方位角 (度)
        el : float
            主瓣仰角 (度)
        level_db : float
            波束宽度定义电平 (dB), 默认 -3dB

        Returns
        -------
        tuple[float, float]
            (方位面波束宽度 度, 仰角面波束宽度 度)
        """
        peak_gain = self.get_gain(freq, az, el)
        threshold = peak_gain + level_db  # level_db 为负值

        # 方位面波束宽度（固定仰角）
        az_sweep = np.linspace(0, 360, 361)
        el_sweep = np.full_like(az_sweep, el)
        gains_az = self.get_gain_batch(freq, az_sweep, el_sweep)

        # 找到主瓣附近超过阈值的范围
        above_threshold = gains_az >= threshold
        az_bw = self._estimate_beamwidth_from_mask(az_sweep, above_threshold, az)

        # 仰角面波束宽度（固定方位角）
        el_sweep2 = np.linspace(0, 90, 91)
        az_sweep2 = np.full_like(el_sweep2, az)
        gains_el = self.get_gain_batch(freq, az_sweep2, el_sweep2)

        above_threshold_el = gains_el >= threshold
        el_bw = self._estimate_beamwidth_from_mask(el_sweep2, above_threshold_el, el)

        return az_bw, el_bw

    def _estimate_beamwidth_from_mask(
        self, angles: np.ndarray, mask: np.ndarray, center: float
    ) -> float:
        """从布尔掩码估算波束宽度"""
        # 简单估算：计算 True 区域的角度跨度
        true_indices = np.where(mask)[0]
        if len(true_indices) < 2:
            return 0.0

        # 找到包含中心的连续 True 区域
        center_idx = np.argmin(np.abs(angles - center))
        # 向两侧扩展
        left = center_idx
        while left > 0 and mask[left - 1]:
            left -= 1
        right = center_idx
        while right < len(mask) - 1 and mask[right + 1]:
            right += 1

        return float(angles[right] - angles[left])

    @property
    def n_frequencies(self) -> int:
        return len(self.frequencies)

    @property
    def n_azimuths(self) -> int:
        return len(self.azimuths)

    @property
    def n_elevations(self) -> int:
        return len(self.elevations)

    @classmethod
    def create_synthetic(
        cls,
        antenna_id: str,
        antenna_type: AntennaType,
        freq_range: tuple[float, float] = (2.0, 30.0),
        n_freq: int = 29,
        n_az: int = 36,
        n_el: int = 18,
        peak_gain: float = 10.0,
        beamwidth_az: float = 60.0,
        beamwidth_el: float = 30.0,
    ) -> "AntennaPatternCube":
        """
        创建合成天线方向图（用于测试和演示）

        Parameters
        ----------
        antenna_id : str
            天线ID
        antenna_type : AntennaType
            天线类型
        freq_range : tuple
            频率范围 (MHz)
        n_freq, n_az, n_el : int
            各维度采样点数
        peak_gain : float
            峰值增益 (dBi)
        beamwidth_az : float
            方位面波束宽度 (度)
        beamwidth_el : float
            仰角面波束宽度 (度)

        Returns
        -------
        AntennaPatternCube
            合成天线方向图
        """
        frequencies = np.linspace(freq_range[0], freq_range[1], n_freq)
        azimuths = np.linspace(0, 360, n_az, endpoint=False)
        elevations = np.linspace(0, 90, n_el)

        # 生成方向图数据
        gain_pattern = np.zeros((n_freq, n_az, n_el))

        for fi, f in enumerate(frequencies):
            for ai, az in enumerate(azimuths):
                for ei, el in enumerate(elevations):
                    # 频率依赖的增益缩放
                    freq_factor = 1.0 + 0.1 * np.log2(f / 15.0)

                    # 方位面波瓣（高斯近似）
                    az_dev = min(abs(az - 0), abs(az - 360))
                    az_factor = np.exp(-2.77 * (az_dev / beamwidth_az) ** 2)

                    # 仰角面波瓣
                    el_factor = np.exp(-2.77 * ((el - 15) / beamwidth_el) ** 2)

                    gain_pattern[fi, ai, ei] = (
                        peak_gain * freq_factor * az_factor * el_factor
                    )

        # 添加一些随机扰动使其更真实
        noise = np.random.normal(0, 0.5, gain_pattern.shape)
        gain_pattern = np.clip(gain_pattern + noise, -30, peak_gain + 2)

        return cls(
            antenna_id=antenna_id,
            antenna_type=antenna_type,
            frequencies=frequencies,
            azimuths=azimuths,
            elevations=elevations,
            gain_pattern=gain_pattern,
        )


def create_default_antenna_library() -> dict[str, AntennaPatternCube]:
    """
    创建默认天线库（用于测试和演示）

    Returns
    -------
    dict[str, AntennaPatternCube]
        天线ID到天线模型的映射
    """
    antennas = {}

    # 对数周期天线 - 宽带
    antennas["LP_01"] = AntennaPatternCube.create_synthetic(
        "LP_01", AntennaType.LOG_PERIODIC,
        peak_gain=8.0, beamwidth_az=70.0, beamwidth_el=40.0,
    )

    # 八木天线 - 高增益定向
    antennas["YAGI_01"] = AntennaPatternCube.create_synthetic(
        "YAGI_01", AntennaType.YAGI,
        peak_gain=14.0, beamwidth_az=30.0, beamwidth_el=25.0,
    )

    # 笼形天线 - 全向
    antennas["CAGE_01"] = AntennaPatternCube.create_synthetic(
        "CAGE_01", AntennaType.CAGE,
        peak_gain=3.0, beamwidth_az=360.0, beamwidth_el=60.0,
    )

    # 菱形天线 - 远距离定向
    antennas["RHOMBIC_01"] = AntennaPatternCube.create_synthetic(
        "RHOMBIC_01", AntennaType.RHOMBIC,
        peak_gain=18.0, beamwidth_az=20.0, beamwidth_el=15.0,
    )

    # 偶极天线
    antennas["DIP_01"] = AntennaPatternCube.create_synthetic(
        "DIP_01", AntennaType.DIPOLE,
        peak_gain=2.15, beamwidth_az=360.0, beamwidth_el=78.0,
    )

    return antennas
