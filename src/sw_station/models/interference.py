"""干扰事件模型 - IsolationMatrix 隔离度矩阵与干扰评估"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class InterferenceEvent:
    """
    干扰事件记录

    记录一次具体的同址干扰事件。
    """
    timestamp: float
    tx_antenna_id: str
    rx_antenna_id: str
    tx_frequency: float      # MHz
    rx_frequency: float      # MHz
    tx_power: float          # dBm
    interference_power: float  # dBm
    allowed_power: float     # dBm
    isolation: float         # dB
    is_violation: bool = False

    def __post_init__(self):
        """计算是否违规"""
        self.is_violation = self.interference_power > self.allowed_power

    @property
    def margin(self) -> float:
        """干扰余量 (dB) - 正值表示安全"""
        return self.allowed_power - self.interference_power

    @property
    def severity(self) -> float:
        """干扰严重程度 (dB) - 正值表示超限"""
        return max(0, self.interference_power - self.allowed_power)


@dataclass
class IntermodProduct:
    """
    互调产物

    描述三阶或五阶互调产物。
    """
    order: int              # 阶数 (3, 5, ...)
    frequency: float        # 互调产物频率 (MHz)
    power: float            # 互调产物功率 (dBm)
    source_tx1: str         # 源发射天线1 ID
    source_tx2: str         # 源发射天线2 ID
    source_freq1: float     # 源频率1 (MHz)
    source_freq2: float     # 源频率2 (MHz)
    affected_rx: str        # 受影响的接收天线 ID

    @classmethod
    def calculate_im3_freqs(cls, f1: float, f2: float) -> list[float]:
        """
        计算三阶互调产物频率

        Parameters
        ----------
        f1, f2 : float
            源频率 (MHz)

        Returns
        -------
        list[float]
            三阶互调产物频率列表 [2*f1-f2, 2*f2-f1]
        """
        return [2 * f1 - f2, 2 * f2 - f1]

    @classmethod
    def calculate_im5_freqs(cls, f1: float, f2: float) -> list[float]:
        """
        计算五阶互调产物频率

        Parameters
        ----------
        f1, f2 : float
            源频率 (MHz)

        Returns
        -------
        list[float]
            五阶互调产物频率列表
        """
        return [
            3 * f1 - 2 * f2,
            3 * f2 - 2 * f1,
            2 * f1 - 3 * f2,  # 可能为负，实际中需要过滤
            2 * f2 - 3 * f1,
        ]


class IsolationMatrix:
    """
    隔离度矩阵管理器

    管理台站内所有天线对之间的隔离度数据，提供干扰评估和约束检查功能。
    """

    def __init__(self, n_antennas: int):
        """
        初始化隔离度矩阵

        Parameters
        ----------
        n_antennas : int
            天线数量
        """
        self.n_antennas = n_antennas
        # 隔离度矩阵 (dB), shape: (n_antennas, n_antennas)
        # isolation_matrix[i, j] = 从天线j到天线i的隔离度
        self.isolation_matrix = np.full((n_antennas, n_antennas), 100.0)
        # 对角线设为无穷大（自身隔离度）
        np.fill_diagonal(self.isolation_matrix, np.inf)

        # 频率依赖的隔离度缓存
        self._freq_dependent_cache: dict[float, np.ndarray] = {}

        # 活跃发射机集合
        self.active_transmitters: dict[int, dict] = {}

        # 干扰事件历史
        self.interference_history: list[InterferenceEvent] = []

    def set_isolation(self, tx_idx: int, rx_idx: int, isolation_db: float,
                      frequency: Optional[float] = None) -> None:
        """
        设置两天线间的隔离度

        Parameters
        ----------
        tx_idx : int
            发射天线索引
        rx_idx : int
            接收天线索引
        isolation_db : float
            隔离度 (dB)
        frequency : float, optional
            工作频率 (MHz)，如果提供则存储频率相关的隔离度
        """
        if tx_idx == rx_idx:
            return

        if frequency is not None:
            if frequency not in self._freq_dependent_cache:
                self._freq_dependent_cache[frequency] = np.full(
                    (self.n_antennas, self.n_antennas), 100.0
                )
                np.fill_diagonal(self._freq_dependent_cache[frequency], np.inf)
            self._freq_dependent_cache[frequency][rx_idx, tx_idx] = isolation_db

        self.isolation_matrix[rx_idx, tx_idx] = isolation_db

    def get_isolation(self, tx_idx: int, rx_idx: int,
                      frequency: Optional[float] = None) -> float:
        """
        获取两天线间的隔离度

        Parameters
        ----------
        tx_idx : int
            发射天线索引
        rx_idx : int
            接收天线索引
        frequency : float, optional
            工作频率 (MHz)

        Returns
        -------
        float
            隔离度 (dB)
        """
        if frequency is not None and frequency in self._freq_dependent_cache:
            return float(self._freq_dependent_cache[frequency][rx_idx, tx_idx])
        return float(self.isolation_matrix[rx_idx, tx_idx])

    def update_from_simulation(self, isolation_data: np.ndarray,
                               frequency: Optional[float] = None) -> None:
        """
        从仿真结果批量更新隔离度矩阵

        Parameters
        ----------
        isolation_data : np.ndarray
            隔离度矩阵数据, shape: (n_antennas, n_antennas)
        frequency : float, optional
            对应的工作频率
        """
        if isolation_data.shape != (self.n_antennas, self.n_antennas):
            raise ValueError(
                f"数据维度 {isolation_data.shape} 不匹配 "
                f"({self.n_antennas}, {self.n_antennas})"
            )

        if frequency is not None:
            self._freq_dependent_cache[frequency] = isolation_data.copy()
        else:
            self.isolation_matrix = isolation_data.copy()
            np.fill_diagonal(self.isolation_matrix, np.inf)

    def calculate_interference_power(
        self,
        tx_idx: int,
        rx_idx: int,
        tx_power_dbm: float,
        frequency: Optional[float] = None,
        filter_rejection_db: float = 0.0,
        freq_offset_mhz: float = 0.0,
    ) -> float:
        """
        计算接收机受到的干扰功率

        P_int = P_tx - Isolation - L_filter(Δf) + P_IMPs

        Parameters
        ----------
        tx_idx : int
            发射天线索引
        rx_idx : int
            接收天线索引
        tx_power_dbm : float
            发射功率 (dBm)
        frequency : float, optional
            工作频率
        filter_rejection_db : float
            滤波器抑制度 (dB)
        freq_offset_mhz : float
            收发频率偏置 (MHz)

        Returns
        -------
        float
            干扰功率 (dBm)
        """
        isolation = self.get_isolation(tx_idx, rx_idx, frequency)

        # 频率偏置增加的隔离度（简化模型）
        offset_isolation = 20 * np.log10(1 + abs(freq_offset_mhz)) if freq_offset_mhz > 0 else 0

        interference = tx_power_dbm - isolation - filter_rejection_db - offset_isolation
        return interference

    def check_interference_constraint(
        self,
        tx_idx: int,
        rx_idx: int,
        tx_power_dbm: float,
        max_allowed_interference_dbm: float,
        frequency: Optional[float] = None,
        filter_rejection_db: float = 0.0,
    ) -> bool:
        """
        检查干扰约束是否满足

        Parameters
        ----------
        tx_idx, rx_idx : int
            发射/接收天线索引
        tx_power_dbm : float
            发射功率 (dBm)
        max_allowed_interference_dbm : float
            最大允许干扰功率 (dBm)
        frequency : float, optional
            工作频率
        filter_rejection_db : float
            滤波器抑制度

        Returns
        -------
        bool
            True 表示满足约束（安全）
        """
        interference = self.calculate_interference_power(
            tx_idx, rx_idx, tx_power_dbm, frequency, filter_rejection_db
        )
        return interference <= max_allowed_interference_dbm

    def evaluate_station_interference(
        self,
        active_links: list[dict],
        max_allowed_interference_dbm: float,
        filter_rejection_db: float = 60.0,
    ) -> tuple[float, list[InterferenceEvent]]:
        """
        评估整个台站的干扰状况

        Parameters
        ----------
        active_links : list[dict]
            活跃链路列表，每个链路包含:
            - tx_idx: 发射天线索引
            - rx_idx: 接收天线索引
            - tx_power: 发射功率 (dBm)
            - frequency: 工作频率 (MHz)
        max_allowed_interference_dbm : float
            最大允许干扰功率 (dBm)
        filter_rejection_db : float
            滤波器抑制度 (dB)

        Returns
        -------
        tuple[float, list[InterferenceEvent]]
            (总干扰惩罚, 干扰事件列表)
        """
        events = []
        total_penalty = 0.0
        timestamp = 0.0  # 可以从外部传入

        for i, link_i in enumerate(active_links):
            for j, link_j in enumerate(active_links):
                if i == j:
                    continue

                tx_i = link_i["tx_idx"]
                rx_j = link_j["rx_idx"]

                if tx_i == rx_j:
                    continue  # 跳过自干扰

                freq_offset = abs(link_i.get("frequency", 0) - link_j.get("frequency", 0))

                interference = self.calculate_interference_power(
                    tx_idx=tx_i,
                    rx_idx=rx_j,
                    tx_power_dbm=link_i["tx_power"],
                    frequency=link_j.get("frequency"),
                    filter_rejection_db=filter_rejection_db,
                    freq_offset_mhz=freq_offset,
                )

                event = InterferenceEvent(
                    timestamp=timestamp,
                    tx_antenna_id=f"ant_{tx_i}",
                    rx_antenna_id=f"ant_{rx_j}",
                    tx_frequency=link_i.get("frequency", 0),
                    rx_frequency=link_j.get("frequency", 0),
                    tx_power=link_i["tx_power"],
                    interference_power=interference,
                    allowed_power=max_allowed_interference_dbm,
                    isolation=self.get_isolation(tx_i, rx_j),
                )

                events.append(event)

                if event.is_violation:
                    total_penalty += event.severity

        return total_penalty, events

    def calculate_im_products(
        self,
        active_links: list[dict],
        receiver_idx: int,
        receiver_freq_mhz: float,
        im_order: int = 3,
    ) -> list[IntermodProduct]:
        """
        计算指定接收机处的互调产物

        Parameters
        ----------
        active_links : list[dict]
            活跃发射链路
        receiver_idx : int
            接收天线索引
        receiver_freq_mhz : float
            接收频率 (MHz)
        im_order : int
            互调阶数 (3 或 5)

        Returns
        -------
        list[IntermodProduct]
            落入接收通带的互调产物列表
        """
        im_products = []
        bandwidth = 0.003  # 接收机带宽 3kHz

        for i, link_i in enumerate(active_links):
            for j, link_j in enumerate(active_links):
                if i >= j:
                    continue

                f1 = link_i.get("frequency", 0)
                f2 = link_j.get("frequency", 0)

                if im_order == 3:
                    im_freqs = IntermodProduct.calculate_im3_freqs(f1, f2)
                elif im_order == 5:
                    im_freqs = IntermodProduct.calculate_im5_freqs(f1, f2)
                else:
                    continue

                for im_freq in im_freqs:
                    if im_freq <= 0:
                        continue

                    # 检查是否落入接收通带
                    if abs(im_freq - receiver_freq_mhz) < bandwidth:
                        # 估算互调产物功率（简化模型）
                        tx_power_1 = link_i["tx_power"]
                        tx_power_2 = link_j["tx_power"]
                        iso_1 = self.get_isolation(link_i["tx_idx"], receiver_idx)
                        iso_2 = self.get_isolation(link_j["tx_idx"], receiver_idx)

                        # 三阶互调功率估算
                        im_power = (
                            tx_power_1 - iso_1 + tx_power_2 - iso_2
                            - 30 * (im_order - 1)  # 互调抑制
                        )

                        im_products.append(IntermodProduct(
                            order=im_order,
                            frequency=im_freq,
                            power=im_power,
                            source_tx1=f"ant_{link_i['tx_idx']}",
                            source_tx2=f"ant_{link_j['tx_idx']}",
                            source_freq1=f1,
                            source_freq2=f2,
                            affected_rx=f"ant_{receiver_idx}",
                        ))

        return im_products

    def get_worst_case_isolation(self) -> tuple[int, int, float]:
        """
        获取最差隔离度的天线对

        Returns
        -------
        tuple[int, int, float]
            (发射天线索引, 接收天线索引, 隔离度 dB)
        """
        # 将对角线设为大值以便找到最小值
        temp_matrix = self.isolation_matrix.copy()
        np.fill_diagonal(temp_matrix, np.inf)

        min_idx = np.unravel_index(np.argmin(temp_matrix), temp_matrix.shape)
        rx_idx, tx_idx = min_idx
        isolation = temp_matrix[min_idx]

        return int(tx_idx), int(rx_idx), float(isolation)

    def get_isolation_statistics(self) -> dict[str, float]:
        """
        获取隔离度统计信息

        Returns
        -------
        dict[str, float]
            包含 min, max, mean, std 等统计值
        """
        # 排除对角线
        mask = ~np.eye(self.n_antennas, dtype=bool)
        values = self.isolation_matrix[mask]

        return {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
        }
