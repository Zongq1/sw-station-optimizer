"""天线方向图查看器 - AntennaPatternViewer"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from ...models.antenna import AntennaPatternCube, AntennaDevice


class AntennaPatternViewer:
    """
    天线方向图 3D 查看器

    支持频率动态缩放、多天线叠加显示。
    """

    def __init__(self):
        """初始化查看器"""
        pass

    def plot_3d_pattern(
        self,
        antenna: AntennaPatternCube,
        frequency: float = 15.0,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        绘制 3D 方向图

        Parameters
        ----------
        antenna : AntennaPatternCube
            天线方向图
        frequency : float
            频率 (MHz)
        title : str, optional
            标题
        save_path : str, optional
            保存路径
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib not available")
            return

        # 获取最近的频率索引
        freq_idx = np.argmin(np.abs(antenna.frequencies - frequency))

        # 获取该频率的方向图
        gain_pattern = antenna.gain_pattern[freq_idx]

        # 生成网格
        az_grid, el_grid = np.meshgrid(
            np.radians(antenna.azimuths),
            antenna.elevations,
            indexing='ij',
        )

        # 转换为笛卡尔坐标（球面投影）
        gain_linear = 10 ** (gain_pattern / 20)
        r = gain_linear / gain_linear.max()

        x = r * np.cos(el_grid) * np.cos(az_grid)
        y = r * np.cos(el_grid) * np.sin(az_grid)
        z = r * np.sin(el_grid)

        # 绘图
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 颜色映射
        colors = cm.jet((gain_pattern - gain_pattern.min()) /
                        (gain_pattern.max() - gain_pattern.min() + 1e-10))

        ax.plot_surface(x, y, z, facecolors=colors, alpha=0.8)

        # 设置标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'{antenna.antenna_id} @ {frequency}MHz')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_polar_pattern(
        self,
        antenna: AntennaPatternCube,
        frequency: float = 15.0,
        elevation: float = 15.0,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        绘制极坐标方向图

        Parameters
        ----------
        antenna : AntennaPatternCube
            天线方向图
        frequency : float
            频率 (MHz)
        elevation : float
            仰角切面 (度)
        title : str, optional
            标题
        save_path : str, optional
            保存路径
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib not available")
            return

        # 获取增益
        azimuths = np.linspace(0, 360, 361)
        gains = [antenna.get_gain(frequency, az, elevation) for az in azimuths]

        # 归一化
        gains = np.array(gains)
        gains_normalized = gains - gains.max()

        # 绘图
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})

        theta = np.radians(azimuths)
        ax.plot(theta, gains_normalized, linewidth=2)
        ax.fill(theta, gains_normalized, alpha=0.3)

        # 设置
        ax.set_ylim(-30, 0)
        ax.set_title(f'{antenna.antenna_id} @ {frequency}MHz, el={elevation}°')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_frequency_comparison(
        self,
        antenna: AntennaPatternCube,
        frequencies: list[float],
        elevation: float = 15.0,
        save_path: Optional[str] = None,
    ) -> None:
        """
        绘制多频率对比图

        Parameters
        ----------
        antenna : AntennaPatternCube
            天线方向图
        frequencies : list[float]
            频率列表
        elevation : float
            仰角
        save_path : str, optional
            保存路径
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib not available")
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        azimuths = np.linspace(0, 360, 361)

        for freq in frequencies:
            gains = [antenna.get_gain(freq, az, elevation) for az in azimuths]
            ax.plot(azimuths, gains, label=f'{freq}MHz', linewidth=2)

        ax.set_xlabel('Azimuth (degrees)')
        ax.set_ylabel('Gain (dBi)')
        ax.set_title(f'{antenna.antenna_id} - Frequency Comparison')
        ax.legend()
        ax.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_multiple_antennas(
        self,
        antennas: list[AntennaPatternCube],
        frequency: float = 15.0,
        elevation: float = 15.0,
        save_path: Optional[str] = None,
    ) -> None:
        """
        绘制多天线对比图

        Parameters
        ----------
        antennas : list[AntennaPatternCube]
            天线列表
        frequency : float
            频率
        elevation : float
            仰角
        save_path : str, optional
            保存路径
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib not available")
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        azimuths = np.linspace(0, 360, 361)

        for antenna in antennas:
            gains = [antenna.get_gain(freq, az, elevation) for az in azimuths]
            ax.plot(azimuths, gains, label=antenna.antenna_id, linewidth=2)

        ax.set_xlabel('Azimuth (degrees)')
        ax.set_ylabel('Gain (dBi)')
        ax.set_title(f'Antenna Comparison @ {frequency}MHz')
        ax.legend()
        ax.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
