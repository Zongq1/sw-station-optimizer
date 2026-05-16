"""数字孪生引擎核心 - DigitalTwinEngine"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from enum import Enum
import time

import numpy as np

from ..models.station import StationDigitalTwin, AntennaDevice
from ..models.channel import ChannelState, IonosphericState
from ..models.interference import IsolationMatrix, InterferenceEvent
from ..simulation.em_simulator import EMSimulator
from ..simulation.propagation import SkyWavePropagation


class TwinLayer(Enum):
    """数字孪生五层架构"""
    PHYSICAL_ENVIRONMENT = "PET"  # 物理环境与几何孪生层
    PHYSICAL_DEVICE = "PDT"       # 物理设备与天线模型孪生层
    ELECTROMAGNETIC = "EST"       # 电磁兼容与空间态势孪生层
    CHANNEL = "CT"                # 通信传播与虚拟信道预测层
    DECISION = "DT"               # 联合调度与智能决策优化层


@dataclass
class TwinEvent:
    """孪生事件"""
    timestamp: float
    event_type: str
    source_layer: TwinLayer
    data: dict
    priority: int = 0


class DigitalTwinEngine:
    """
    数字孪生引擎

    管理五层孪生架构的状态同步和事件驱动更新。
    """

    def __init__(
        self,
        station: Optional[StationDigitalTwin] = None,
        update_interval: float = 1.0,
    ):
        """
        初始化数字孪生引擎

        Parameters
        ----------
        station : StationDigitalTwin, optional
            台站模型
        update_interval : float
            更新间隔 (秒)
        """
        self.station = station or StationDigitalTwin.create_default_station()
        self.update_interval = update_interval

        # 仿真器
        self.em_simulator = EMSimulator()
        self.propagation_engine = SkyWavePropagation()

        # 干扰阈值
        self.interference_threshold = -130.0  # dBm

        # 状态
        self.current_time = 0.0
        self.is_running = False

        # 事件队列
        self.event_queue: list[TwinEvent] = []
        self.event_handlers: dict[str, list[Callable]] = {}

        # 历史记录
        self.state_history: list[dict] = []
        self.interference_history: list[InterferenceEvent] = []

        # 层状态缓存
        self._layer_states: dict[TwinLayer, dict] = {
            layer: {} for layer in TwinLayer
        }

    def start(self) -> None:
        """启动引擎"""
        self.is_running = True
        self._initialize_layers()

    def stop(self) -> None:
        """停止引擎"""
        self.is_running = False

    def _initialize_layers(self) -> None:
        """初始化各层状态"""
        # PET: 物理环境 - 加载地形和电导率数据
        positions = np.array([ant.position for ant in self.station.antennas])
        self._layer_states[TwinLayer.PHYSICAL_ENVIRONMENT] = {
            "terrain_loaded": True,
            "terrain_resolution": 10.0,  # 米
            "elevation_range": (
                float(positions[:, 2].min()),
                float(positions[:, 2].max()),
            ),
            "buildings_loaded": True,
            "n_buildings": 0,  # 简化：无建筑物数据
            "conductivity_map": True,
            "ground_conductivity": self.em_simulator.ground_conductivity,
            "ground_permittivity": self.em_simulator.ground_permittivity,
            "boundary": (
                float(positions[:, 0].min()),
                float(positions[:, 0].max()),
                float(positions[:, 1].min()),
                float(positions[:, 1].max()),
            ),
        }

        # PDT: 物理设备
        self._layer_states[TwinLayer.PHYSICAL_DEVICE] = {
            "n_antennas": self.station.n_antennas,
            "antenna_types": [a.antenna_type.value for a in self.station.antennas],
        }

        # EST: 电磁态势
        self._update_electromagnetic_state()

        # CT: 信道预测
        self._update_channel_state()

        # DT: 决策优化层
        self._layer_states[TwinLayer.DECISION] = {
            "optimization_active": False,
            "rl_agent_active": False,
            "last_optimization_time": 0.0,
            "last_rl_decision_time": 0.0,
            "optimization_result": None,
            "rl_action_history": [],
            "n_decisions_made": 0,
        }

    def update(self, time_delta: float = 1.0) -> dict:
        """
        更新孪生状态

        Parameters
        ----------
        time_delta : float
            时间步长

        Returns
        -------
        dict
            更新后的状态摘要
        """
        if not self.is_running:
            return {}

        self.current_time += time_delta

        # 处理事件队列
        self._process_events()

        # 更新各层
        self.station.update_state(time_delta)
        self._update_electromagnetic_state()
        self._update_channel_state()

        # 记录历史
        state_snapshot = self.get_state_snapshot()
        self.state_history.append(state_snapshot)

        return state_snapshot

    def _update_electromagnetic_state(self) -> None:
        """更新电磁态势层"""
        # 计算当前活跃天线对的隔离度
        active_indices = [
            i for i, ant in enumerate(self.station.antennas)
            if ant.is_transmitting
        ]

        if len(active_indices) > 1:
            interference_events = []
            for i in active_indices:
                for j in active_indices:
                    if i >= j:
                        continue

                    ant_i = self.station.antennas[i]
                    ant_j = self.station.antennas[j]

                    # 使用天线实际工作频率
                    freq = 15.0
                    if ant_i.current_frequency and ant_j.current_frequency:
                        freq = (ant_i.current_frequency + ant_j.current_frequency) / 2
                    elif ant_i.current_frequency:
                        freq = ant_i.current_frequency
                    elif ant_j.current_frequency:
                        freq = ant_j.current_frequency

                    isolation = self.em_simulator.calculate_isolation(
                        ant_i, ant_j, freq
                    )

                    # 检查是否违规
                    interference = ant_i.current_power - isolation
                    if interference > self.interference_threshold:
                        event = InterferenceEvent(
                            timestamp=self.current_time,
                            tx_antenna_id=ant_i.id,
                            rx_antenna_id=ant_j.id,
                            tx_frequency=ant_i.current_frequency or 0,
                            rx_frequency=ant_j.current_frequency or 0,
                            tx_power=ant_i.current_power,
                            interference_power=interference,
                            allowed_power=self.interference_threshold,
                            isolation=isolation,
                        )
                        interference_events.append(event)

            self._layer_states[TwinLayer.ELECTROMAGNETIC] = {
                "active_transmitters": len(active_indices),
                "interference_events": len(interference_events),
                "worst_interference": max(
                    (e.interference_power for e in interference_events),
                    default=-np.inf,
                ),
            }

            # 触发告警事件
            violations = [e for e in interference_events if e.is_violation]
            if violations:
                self._emit_event(TwinEvent(
                    timestamp=self.current_time,
                    event_type="interference_violation",
                    source_layer=TwinLayer.ELECTROMAGNETIC,
                    data={"violations": len(violations)},
                    priority=1,
                ))

    def _update_channel_state(self) -> None:
        """更新信道预测层 - 调用传播引擎"""
        ionosphere = self.station.ionospheric_state
        hour = (self.current_time / 3600) % 24

        # 更新电离层状态的昼夜变化
        if 6 <= hour <= 18:
            ionosphere.fof2 = 8.0 + 4.0 * np.sin(np.pi * (hour - 6) / 12)
        else:
            ionosphere.fof2 = 5.0

        # 调用传播引擎计算典型路径的 MUF/LUF
        typical_distance = 1000.0  # km
        muf = self.propagation_engine.calculate_muf(typical_distance, ionosphere)
        luf = self.propagation_engine.calculate_luf(typical_distance, ionosphere)

        # 计算最佳频率
        optimal_freq, optimal_snr = self.propagation_engine.find_optimal_frequency(
            typical_distance, ionosphere
        )

        self._layer_states[TwinLayer.CHANNEL] = {
            "muf": muf,
            "luf": luf,
            "optimal_frequency": optimal_freq,
            "optimal_snr": optimal_snr,
            "fof2": ionosphere.fof2,
            "fof1": ionosphere.fof1,
            "foe": ionosphere.foe,
            "solar_flux": ionosphere.solar_flux_107,
            "h_prime_f2": ionosphere.h_prime_f2,
            "time_of_day": "day" if 6 <= hour <= 18 else "night",
        }

    def _process_events(self) -> None:
        """处理事件队列"""
        # 按优先级排序
        self.event_queue.sort(key=lambda e: e.priority, reverse=True)

        while self.event_queue:
            event = self.event_queue.pop(0)
            self._handle_event(event)

    def _handle_event(self, event: TwinEvent) -> None:
        """处理单个事件"""
        handlers = self.event_handlers.get(event.event_type, [])
        for handler in handlers:
            handler(event)

    def _emit_event(self, event: TwinEvent) -> None:
        """发送事件"""
        self.event_queue.append(event)

    def activate_optimization(self) -> None:
        """激活优化决策"""
        self._layer_states[TwinLayer.DECISION]["optimization_active"] = True
        self._layer_states[TwinLayer.DECISION]["last_optimization_time"] = self.current_time

    def activate_rl_agent(self) -> None:
        """激活 RL 决策"""
        self._layer_states[TwinLayer.DECISION]["rl_agent_active"] = True
        self._layer_states[TwinLayer.DECISION]["last_rl_decision_time"] = self.current_time

    def record_decision(self, action: dict) -> None:
        """记录决策动作"""
        dt_state = self._layer_states[TwinLayer.DECISION]
        dt_state["rl_action_history"].append({
            "time": self.current_time,
            "action": action,
        })
        dt_state["n_decisions_made"] += 1
        # 保留最近 100 条
        if len(dt_state["rl_action_history"]) > 100:
            dt_state["rl_action_history"] = dt_state["rl_action_history"][-100:]

    def register_handler(self, event_type: str, handler: Callable) -> None:
        """
        注册事件处理器

        Parameters
        ----------
        event_type : str
            事件类型
        handler : Callable
            处理函数
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)

    def get_state_snapshot(self) -> dict:
        """获取当前状态快照"""
        return {
            "timestamp": self.current_time,
            "station_stats": self.station.get_station_statistics(),
            "layer_states": {k.value: v for k, v in self._layer_states.items()},
        }

    def get_layer_state(self, layer: TwinLayer) -> dict:
        """获取指定层状态"""
        return self._layer_states.get(layer, {})

    def query_interference(
        self,
        tx_antenna_id: str,
        rx_antenna_id: str,
        frequency: float,
    ) -> float:
        """
        查询两天线间的干扰

        Parameters
        ----------
        tx_antenna_id, rx_antenna_id : str
            天线 ID
        frequency : float
            频率 (MHz)

        Returns
        -------
        float
            干扰功率 (dBm)
        """
        tx_ant = self.station.get_antenna_by_id(tx_antenna_id)
        rx_ant = self.station.get_antenna_by_id(rx_antenna_id)

        if tx_ant is None or rx_ant is None:
            return -np.inf

        isolation = self.em_simulator.calculate_isolation(tx_ant, rx_ant, frequency)
        return tx_ant.current_power - isolation

    def predict_propagation(
        self,
        frequency: float,
        distance_km: float,
    ) -> ChannelState:
        """
        预测传播状态

        Parameters
        ----------
        frequency : float
            频率 (MHz)
        distance_km : float
            距离 (km)

        Returns
        -------
        ChannelState
            信道状态预测
        """
        return self.propagation_engine.evaluate_channel(
            frequency, distance_km, self.station.ionospheric_state
        )

    def export_history(self) -> list[dict]:
        """导出历史记录"""
        return self.state_history.copy()
