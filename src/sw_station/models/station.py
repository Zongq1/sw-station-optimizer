"""台站数字孪生主模型 - StationDigitalTwin 与 AntennaDevice"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .antenna import AntennaPatternCube, AntennaType
from .channel import ChannelState, PropagationMode, IonosphericState
from .interference import IsolationMatrix, InterferenceEvent


@dataclass
class AntennaDevice:
    """
    天线设备孪生模型

    将物理天线设备抽象为数字孪生体，包含位置、姿态和电磁特性。
    """
    # 设备唯一标识
    id: str
    # 天线类型
    antenna_type: AntennaType
    # 三维位置坐标 [x, y, z] (米)
    position: np.ndarray
    # 方位角 (度, [0, 360))
    azimuth: float = 0.0
    # 仰角 (度, [0, 90])
    elevation: float = 0.0
    # 天线方向图数据
    pattern: Optional[AntennaPatternCube] = None
    # 是否处于发射状态
    is_transmitting: bool = False
    # 当前工作频率 (MHz)
    current_frequency: Optional[float] = None
    # 当前发射功率 (dBm)
    current_power: float = 30.0
    # 分组归属
    group_id: int = 0
    # 馈线损耗 (dB)
    feedline_loss: float = 2.0

    def __post_init__(self):
        """数据验证"""
        if self.position.shape != (3,):
            raise ValueError(f"position 必须是 3D 向量, 实际为 {self.position.shape}")
        self.position = np.asarray(self.position, dtype=float)

    @property
    def xyz(self) -> tuple[float, float, float]:
        """位置坐标元组"""
        return tuple(self.position.tolist())

    @property
    def is_receiving(self) -> bool:
        """是否处于接收状态"""
        return not self.is_transmitting and self.current_frequency is not None

    def get_gain_at(self, freq: float, az: float, el: float) -> float:
        """
        获取指定方向的天线增益

        Parameters
        ----------
        freq : float
            频率 (MHz)
        az : float
            相对于天线朝向的方位角 (度)
        el : float
            仰角 (度)

        Returns
        -------
        float
            增益 (dBi)
        """
        if self.pattern is None:
            return 0.0

        # 转换为绝对方位角
        abs_az = (az + self.azimuth) % 360.0
        return self.pattern.get_gain(freq, abs_az, el)

    @property
    def max_dimension(self) -> float:
        """天线最大物理尺寸 (米)"""
        from .antenna import ANTENNA_MAX_DIMENSIONS
        return ANTENNA_MAX_DIMENSIONS.get(self.antenna_type, 20.0)

    def distance_to(self, other: "AntennaDevice") -> float:
        """
        计算到另一副天线的距离

        Parameters
        ----------
        other : AntennaDevice
            另一副天线

        Returns
        -------
        float
            距离 (米)
        """
        return float(np.linalg.norm(self.position - other.position))

    def direction_to(self, other: "AntennaDevice") -> tuple[float, float]:
        """
        计算到另一副天线的方向

        Parameters
        ----------
        other : AntennaDevice
            另一副天线

        Returns
        -------
        tuple[float, float]
            (方位角 度, 仰角 度)
        """
        diff = other.position - self.position
        distance = np.linalg.norm(diff)

        if distance < 1e-6:
            return 0.0, 0.0

        # 方位角
        azimuth = np.degrees(np.arctan2(diff[1], diff[0])) % 360.0
        # 仰角
        horizontal_dist = np.sqrt(diff[0]**2 + diff[1]**2)
        elevation = np.degrees(np.arctan2(diff[2], horizontal_dist))

        return float(azimuth), float(elevation)


@dataclass
class CommunicationTask:
    """
    通信任务描述

    描述一个待分配的通信任务。
    """
    task_id: str
    # 目标方位角 (度)
    target_azimuth: float
    # 目标仰角 (度)
    target_elevation: float
    # 要求的频率范围 (MHz)
    freq_range: tuple[float, float]
    # 要求的最小发射功率 (dBm)
    min_power: float = 20.0
    # 要求的可靠性等级 (0-1)
    reliability_level: float = 0.9
    # 优先级 (越高越优先)
    priority: int = 1
    # 任务状态
    status: str = "pending"  # pending, assigned, completed, failed
    # 分配的天线ID
    assigned_antenna: Optional[str] = None
    # 分配的频率
    assigned_frequency: Optional[float] = None

    def score_for_antenna(self, antenna: AntennaDevice, frequency: float) -> float:
        """
        计算天线-频率组合对本任务的匹配评分

        Parameters
        ----------
        antenna : AntennaDevice
            候选天线
        frequency : float
            候选频率 (MHz)

        Returns
        -------
        float
            匹配评分 (越高越好)
        """
        if antenna.pattern is None:
            return 0.0

        # 增益匹配度
        gain = antenna.get_gain_at(
            frequency, self.target_azimuth, self.target_elevation
        )
        peak_gain = antenna.pattern.get_peak_gain(frequency)[0]
        gain_ratio = gain / peak_gain if peak_gain > 0 else 0

        # 频率范围匹配度
        freq_in_range = self.freq_range[0] <= frequency <= self.freq_range[1]
        freq_score = 1.0 if freq_in_range else 0.5

        return gain_ratio * freq_score


class StationDigitalTwin:
    """
    台站数字孪生主模型

    整合天线设备、隔离度矩阵、信道状态和任务队列，
    提供台站全局状态管理和优化接口。
    """

    def __init__(self, n_antennas: int = 50):
        """
        初始化台站数字孪生

        Parameters
        ----------
        n_antennas : int
            天线数量
        """
        self.n_antennas = n_antennas

        # 天线设备列表
        self.antennas: list[AntennaDevice] = []

        # 隔离度矩阵
        self.isolation_matrix = IsolationMatrix(n_antennas)

        # 信道状态字典 {(tx_idx, rx_idx, freq): ChannelState}
        self.channel_states: dict[tuple[int, int, float], ChannelState] = {}

        # 电离层状态
        self.ionospheric_state = IonosphericState()

        # 通信任务队列
        self.task_queue: list[CommunicationTask] = []

        # 干扰事件历史
        self.interference_log: list[InterferenceEvent] = []

        # 时间步
        self.time_step: float = 0.0

    def add_antenna(self, antenna: AntennaDevice) -> int:
        """
        添加天线设备

        Parameters
        ----------
        antenna : AntennaDevice
            天线设备

        Returns
        -------
        int
            天线索引
        """
        self.antennas.append(antenna)
        return len(self.antennas) - 1

    def get_antenna(self, idx: int) -> AntennaDevice:
        """获取指定索引的天线"""
        return self.antennas[idx]

    def get_antenna_by_id(self, antenna_id: str) -> Optional[AntennaDevice]:
        """根据ID获取天线"""
        for ant in self.antennas:
            if ant.id == antenna_id:
                return ant
        return None

    def get_antenna_index(self, antenna_id: str) -> Optional[int]:
        """获取天线索引"""
        for i, ant in enumerate(self.antennas):
            if ant.id == antenna_id:
                return i
        return None

    def initialize_random_layout(
        self,
        boundary: tuple[float, float, float, float] = (0, 2000, 0, 2000),
        antenna_types: Optional[list[AntennaType]] = None,
    ) -> None:
        """
        随机初始化天线布局

        Parameters
        ----------
        boundary : tuple
            台站边界 (x_min, x_max, y_min, y_max) (米)
        antenna_types : list[AntennaType], optional
            天线类型列表，默认使用混合类型
        """
        if antenna_types is None:
            antenna_types = [
                AntennaType.LOG_PERIODIC,
                AntennaType.YAGI,
                AntennaType.CAGE,
                AntennaType.RHOMBIC,
                AntennaType.DIPOLE,
            ]

        x_min, x_max, y_min, y_max = boundary

        self.antennas = []
        for i in range(self.n_antennas):
            # 随机选择天线类型
            ant_type = antenna_types[i % len(antenna_types)]

            # 随机位置
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            z = np.random.uniform(10, 50)  # 架设高度 10-50m

            # 随机朝向
            azimuth = np.random.uniform(0, 360)
            elevation = np.random.uniform(0, 30)

            # 创建天线设备
            antenna = AntennaDevice(
                id=f"ANT_{i:03d}",
                antenna_type=ant_type,
                position=np.array([x, y, z]),
                azimuth=azimuth,
                elevation=elevation,
            )

            self.antennas.append(antenna)

    def encode_layout_to_vector(self) -> np.ndarray:
        """
        将天线布局编码为优化变量向量

        Returns
        -------
        np.ndarray
            变量向量, shape: (n_antennas * 4,)
            每个天线 4 个变量: [x, y, z, azimuth]
        """
        vector = np.zeros(self.n_antennas * 4)
        for i, ant in enumerate(self.antennas):
            vector[i * 4:i * 4 + 3] = ant.position
            vector[i * 4 + 3] = ant.azimuth
        return vector

    def decode_vector_to_layout(self, vector: np.ndarray) -> None:
        """
        从优化变量向量解码天线布局

        Parameters
        ----------
        vector : np.ndarray
            变量向量, shape: (n_antennas * 4,)
        """
        for i in range(self.n_antennas):
            self.antennas[i].position = vector[i * 4:i * 4 + 3]
            self.antennas[i].azimuth = vector[i * 4 + 3] % 360.0

    def get_active_transmitters(self) -> list[int]:
        """获取所有活跃发射机的索引"""
        return [i for i, ant in enumerate(self.antennas) if ant.is_transmitting]

    def get_active_receivers(self) -> list[int]:
        """获取所有活跃接收机的索引"""
        return [i for i, ant in enumerate(self.antennas) if ant.is_receiving]

    def update_state(self, time_delta: float = 1.0) -> None:
        """
        更新台站状态

        Parameters
        ----------
        time_delta : float
            时间步长 (秒)
        """
        self.time_step += time_delta

        # 更新电离层状态（简化模型）
        self._update_ionospheric_state()

        # 更新信道状态
        self._update_channel_states()

    def _update_ionospheric_state(self) -> None:
        """更新电离层状态（昼夜变化简化模型）"""
        # 简化：基于时间的昼夜变化
        hour = (self.time_step / 3600) % 24

        if 6 <= hour <= 18:
            # 白天
            self.ionospheric_state.fof2 = 8.0 + 2.0 * np.sin(np.pi * (hour - 6) / 12)
            self.ionospheric_state.fof1 = 5.0
        else:
            # 夜间
            self.ionospheric_state.fof2 = 6.0
            self.ionospheric_state.fof1 = 0.0

    def _update_channel_states(self) -> None:
        """更新信道状态缓存"""
        # 清除过期状态
        self.channel_states.clear()

    def predict_interference_for_allocation(
        self,
        allocations: list[dict],
        max_allowed_interference_dbm: float = -130.0,
    ) -> tuple[float, list[InterferenceEvent]]:
        """
        预测提议分配方案的干扰情况

        Parameters
        ----------
        allocations : list[dict]
            分配方案列表
        max_allowed_interference_dbm : float
            最大允许干扰功率

        Returns
        -------
        tuple[float, list[InterferenceEvent]]
            (总干扰惩罚, 干扰事件列表)
        """
        return self.isolation_matrix.evaluate_station_interference(
            allocations, max_allowed_interference_dbm
        )

    def get_station_statistics(self) -> dict:
        """
        获取台站统计信息

        Returns
        -------
        dict
            统计信息字典
        """
        positions = np.array([ant.position for ant in self.antennas])

        return {
            "n_antennas": self.n_antennas,
            "n_transmitting": len(self.get_active_transmitters()),
            "n_receiving": len(self.get_active_receivers()),
            "layout_center": positions.mean(axis=0).tolist(),
            "layout_extent": (positions.max(axis=0) - positions.min(axis=0)).tolist(),
            "isolation_stats": self.isolation_matrix.get_isolation_statistics(),
            "n_tasks": len(self.task_queue),
            "n_pending_tasks": sum(1 for t in self.task_queue if t.status == "pending"),
            "time_step": self.time_step,
        }

    @classmethod
    def create_default_station(cls, n_antennas: int = 50) -> "StationDigitalTwin":
        """
        创建默认台站配置

        Parameters
        ----------
        n_antennas : int
            天线数量

        Returns
        -------
        StationDigitalTwin
            默认配置的台站
        """
        station = cls(n_antennas)
        station.initialize_random_layout()
        return station
