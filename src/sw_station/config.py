"""全局配置模块"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class StationConfig:
    """台站物理配置"""
    n_antennas: int = 50
    # 台站边界 [x_min, x_max, y_min, y_max] (米)
    boundary: tuple[float, float, float, float] = (0, 2000, 0, 2000)
    # 地表电导率 (S/m)
    ground_conductivity: float = 0.01
    # 地表相对介电常数
    ground_permittivity: float = 13.0
    # 最大馈线长度 (米)
    max_cable_length: float = 500.0


@dataclass
class FrequencyConfig:
    """频率配置"""
    # 短波频段范围 (MHz)
    freq_min: float = 2.0
    freq_max: float = 30.0
    # 信道数量
    n_channels: int = 100
    # 信道带宽 (kHz)
    channel_bandwidth: float = 3.0


@dataclass
class OptimizationConfig:
    """优化算法配置"""
    # 种群大小
    population_size: int = 100
    # 最大迭代代数
    max_generations: int = 500
    # 交叉概率
    crossover_prob: float = 0.9
    # 变异概率
    mutation_prob: float = 0.1
    # 目标函数权重 [覆盖, 干扰, 成本]
    objective_weights: tuple[float, float, float] = (0.4, 0.4, 0.2)


@dataclass
class RLConfig:
    """强化学习配置"""
    # 学习率
    learning_rate: float = 3e-4
    # 折扣因子
    gamma: float = 0.99
    # 经验回放缓冲区大小
    buffer_size: int = 100000
    # 批次大小
    batch_size: int = 64
    # 目标网络更新频率
    target_update_freq: int = 1000
    # 探索率
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 10000
    # 奖励权重 [吞吐量, 延迟, 干扰]
    reward_weights: tuple[float, float, float] = (0.5, 0.3, 0.2)


@dataclass
class InterferenceConfig:
    """干扰约束配置"""
    # 接收机灵敏度 (dBm)
    receiver_sensitivity: float = -120.0
    # 最大允许干扰功率 (dBm)
    max_allowed_interference: float = -130.0
    # 干扰余量 (dB)
    interference_margin: float = 10.0
    # 滤波器抑制度 (dB)
    filter_rejection: float = 60.0


@dataclass
class VisualizationConfig:
    """可视化配置"""
    # 热力图分辨率
    heatmap_resolution: int = 50
    # 颜色映射
    colormap: str = "jet"
    # 透明度
    alpha: float = 0.6


@dataclass
class SystemConfig:
    """系统总配置"""
    station: StationConfig = field(default_factory=StationConfig)
    frequency: FrequencyConfig = field(default_factory=FrequencyConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    interference: InterferenceConfig = field(default_factory=InterferenceConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    @classmethod
    def default(cls) -> "SystemConfig":
        """创建默认配置"""
        return cls()

    @classmethod
    def from_dict(cls, config_dict: dict) -> "SystemConfig":
        """从字典创建配置"""
        return cls(
            station=StationConfig(**config_dict.get("station", {})),
            frequency=FrequencyConfig(**config_dict.get("frequency", {})),
            optimization=OptimizationConfig(**config_dict.get("optimization", {})),
            rl=RLConfig(**config_dict.get("rl", {})),
            interference=InterferenceConfig(**config_dict.get("interference", {})),
            visualization=VisualizationConfig(**config_dict.get("visualization", {})),
        )
