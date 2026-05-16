"""数据同步机制 - DataSynchronizer"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from enum import Enum
import json
import time

import numpy as np

from ..models.station import StationDigitalTwin
from ..models.channel import IonosphericState


class DataSource(Enum):
    """数据源类型"""
    WSPR = "wspr"              # 弱信号传播报告
    PSK_REPORTER = "pskreporter"  # PSK Reporter
    RBN = "rbn"                # 反向信标网络
    IONOSONDE = "ionosonde"    # 电离层探测仪
    SOLAR = "solar"            # 太阳活动数据
    MANUAL = "manual"          # 手动输入


@dataclass
class DataRecord:
    """数据记录"""
    timestamp: float
    source: DataSource
    data_type: str
    payload: dict
    quality: float = 1.0  # 数据质量 0-1


class DataSynchronizer:
    """
    数据同步器

    管理外部数据源的摄取、存储和同步。
    支持 WSPR、PSKReporter、RBN 等业余无线电监测数据。
    """

    def __init__(
        self,
        station: Optional[StationDigitalTwin] = None,
        buffer_size: int = 10000,
    ):
        """
        初始化数据同步器

        Parameters
        ----------
        station : StationDigitalTwin, optional
            台站模型
        buffer_size : int
            缓冲区大小
        """
        self.station = station
        self.buffer_size = buffer_size

        # 数据缓冲区
        self.data_buffer: list[DataRecord] = []
        self.processed_count = 0

        # 数据处理器
        self.processors: dict[DataSource, Callable] = {
            DataSource.WSPR: self._process_wspr,
            DataSource.PSK_REPORTER: self._process_psk_reporter,
            DataSource.RBN: self._process_rbn,
            DataSource.IONOSONDE: self._process_ionosonde,
            DataSource.SOLAR: self._process_solar,
        }

        # 统计信息
        self.stats = {
            "total_records": 0,
            "by_source": {s.value: 0 for s in DataSource},
            "last_update": 0.0,
        }

    def ingest(self, record: DataRecord) -> None:
        """
        摄入数据记录

        Parameters
        ----------
        record : DataRecord
            数据记录
        """
        self.data_buffer.append(record)
        self.stats["total_records"] += 1
        self.stats["by_source"][record.source.value] += 1

        # 缓冲区管理
        if len(self.data_buffer) > self.buffer_size:
            self.data_buffer.pop(0)

        # 自动处理
        self._auto_process(record)

    def ingest_batch(self, records: list[DataRecord]) -> int:
        """
        批量摄入数据

        Parameters
        ----------
        records : list[DataRecord]
            数据记录列表

        Returns
        -------
        int
            成功处理的记录数
        """
        count = 0
        for record in records:
            try:
                self.ingest(record)
                count += 1
            except Exception as e:
                print(f"Error ingesting record: {e}")
        return count

    def _auto_process(self, record: DataRecord) -> None:
        """自动处理新数据"""
        processor = self.processors.get(record.source)
        if processor:
            processor(record)
            self.processed_count += 1
            self.stats["last_update"] = time.time()

    def _process_wspr(self, record: DataRecord) -> None:
        """
        处理 WSPR 数据

        WSPR 数据包含：呼叫号、频率、功率、SNR、网格定位等
        """
        payload = record.payload

        # 提取关键信息
        callsign = payload.get("callsign", "")
        frequency = payload.get("frequency", 0)  # MHz
        snr = payload.get("snr", 0)  # dB
        power = payload.get("power", 0)  # dBm
        grid = payload.get("grid", "")

        # 更新信道状态（如果有台站引用）
        if self.station and frequency > 0:
            # 简化：记录 SNR 数据用于后续分析
            pass

    def _process_psk_reporter(self, record: DataRecord) -> None:
        """
        处理 PSK Reporter 数据

        PSK Reporter 数据包含：发送者、接收者、频率、模式、SNR 等
        """
        payload = record.payload

        sender = payload.get("sender", "")
        receiver = payload.get("receiver", "")
        frequency = payload.get("frequency", 0)
        snr = payload.get("snr", 0)
        mode = payload.get("mode", "")

        # 用于传播路径分析
        if self.station and frequency > 0:
            pass

    def _process_rbn(self, record: DataRecord) -> None:
        """
        处理 RBN (反向信标网络) 数据

        RBN 数据包含：信标呼号、接收站、频率、信噪比等
        """
        payload = record.payload

        beacon = payload.get("beacon", "")
        receiver = payload.get("receiver", "")
        frequency = payload.get("frequency", 0)
        snr = payload.get("snr", 0)

        # 用于实时传播监控
        if self.station and frequency > 0:
            pass

    def _process_ionosonde(self, record: DataRecord) -> None:
        """
        处理电离层探测数据

        电离层数据包含：foF2, MUF, 层高等参数
        """
        payload = record.payload

        if self.station:
            # 更新电离层状态
            ionosphere = self.station.ionospheric_state

            if "fof2" in payload:
                ionosphere.fof2 = payload["fof2"]
            if "m3000f2" in payload:
                ionosphere.m3000f2 = payload["m3000f2"]
            if "h_prime_f2" in payload:
                ionosphere.h_prime_f2 = payload["h_prime_f2"]

    def _process_solar(self, record: DataRecord) -> None:
        """
        处理太阳活动数据

        太阳数据包含：太阳黑子数、10.7cm 射电通量等
        """
        payload = record.payload

        if self.station:
            ionosphere = self.station.ionospheric_state

            if "sunspot_number" in payload:
                ionosphere.solar_sunspot_number = payload["sunspot_number"]
            if "flux_107" in payload:
                ionosphere.solar_flux_107 = payload["flux_107"]

    def get_recent_data(
        self,
        source: Optional[DataSource] = None,
        n_records: int = 100,
    ) -> list[DataRecord]:
        """
        获取最近的数据记录

        Parameters
        ----------
        source : DataSource, optional
            数据源过滤
        n_records : int
            返回记录数

        Returns
        -------
        list[DataRecord]
            数据记录列表
        """
        if source:
            filtered = [r for r in self.data_buffer if r.source == source]
        else:
            filtered = self.data_buffer

        return filtered[-n_records:]

    def get_statistics(self) -> dict:
        """获取统计信息"""
        return {
            **self.stats,
            "buffer_size": len(self.data_buffer),
            "processed_count": self.processed_count,
        }

    def create_sample_wspr_data(self, n_records: int = 10) -> list[DataRecord]:
        """
        创建示例 WSPR 数据（用于测试）

        Parameters
        ----------
        n_records : int
            记录数量

        Returns
        -------
        list[DataRecord]
            示例数据
        """
        records = []
        for i in range(n_records):
            record = DataRecord(
                timestamp=time.time() - (n_records - i) * 120,
                source=DataSource.WSPR,
                data_type="propagation",
                payload={
                    "callsign": f"W1AW{i:03d}",
                    "frequency": np.random.uniform(7.0, 14.0),
                    "snr": np.random.randint(-20, 20),
                    "power": np.random.choice([1, 5, 10, 20, 30]),
                    "grid": "FN31",
                },
                quality=np.random.uniform(0.5, 1.0),
            )
            records.append(record)
        return records

    def create_sample_ionosonde_data(self) -> DataRecord:
        """
        创建示例电离层探测数据

        Returns
        -------
        DataRecord
            示例数据
        """
        return DataRecord(
            timestamp=time.time(),
            source=DataSource.IONOSONDE,
            data_type="ionosphere",
            payload={
                "fof2": np.random.uniform(5, 12),
                "m3000f2": np.random.uniform(2.5, 4.0),
                "h_prime_f2": np.random.uniform(250, 350),
                "foe": np.random.uniform(2, 5),
            },
            quality=1.0,
        )

    def export_data(self, filepath: str) -> None:
        """导出数据到 JSON 文件"""
        data = [
            {
                "timestamp": r.timestamp,
                "source": r.source.value,
                "data_type": r.data_type,
                "payload": r.payload,
                "quality": r.quality,
            }
            for r in self.data_buffer
        ]

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def import_data(self, filepath: str) -> int:
        """从 JSON 文件导入数据"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        records = []
        for item in data:
            record = DataRecord(
                timestamp=item["timestamp"],
                source=DataSource(item["source"]),
                data_type=item["data_type"],
                payload=item["payload"],
                quality=item.get("quality", 1.0),
            )
            records.append(record)

        return self.ingest_batch(records)
