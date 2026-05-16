"""3D电磁热力图可视化 - ElectromagneticHeatmap"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from ...models.station import StationDigitalTwin, AntennaDevice
from ...simulation.em_simulator import EMSimulator


class ElectromagneticHeatmap:
    """
    3D 电磁热力图

    使用体素渲染技术可视化三维空间的电磁强度分布。
    """

    def __init__(
        self,
        station: StationDigitalTwin,
        em_simulator: Optional[EMSimulator] = None,
        resolution: int = 50,
    ):
        """
        初始化热力图

        Parameters
        ----------
        station : StationDigitalTwin
            台站模型
        em_simulator : EMSimulator, optional
            电磁仿真器
        resolution : int
            体素分辨率
        """
        self.station = station
        self.em_simulator = em_simulator or EMSimulator()
        self.resolution = resolution

        # 计算空间边界
        positions = np.array([a.position for a in station.antennas])
        self.x_range = (positions[:, 0].min() - 50, positions[:, 0].max() + 50)
        self.y_range = (positions[:, 1].min() - 50, positions[:, 1].max() + 50)
        self.z_range = (0, 100)  # 高度范围

        # 生成网格
        self.x_grid = np.linspace(*self.x_range, resolution)
        self.y_grid = np.linspace(*self.y_range, resolution)
        self.z_grid = np.linspace(*self.z_range, resolution // 2)

    def calculate_field_strength(
        self,
        frequency: float = 15.0,
        tx_antenna_idx: int = 0,
    ) -> np.ndarray:
        """
        计算指定发射天线的三维场强分布

        Parameters
        ----------
        frequency : float
            工作频率 (MHz)
        tx_antenna_idx : int
            发射天线索引

        Returns
        -------
        np.ndarray
            场强分布, shape: (nx, ny, nz)
        """
        tx_antenna = self.station.antennas[tx_antenna_idx]
        nx, ny, nz = len(self.x_grid), len(self.y_grid), len(self.z_grid)
        field = np.zeros((nx, ny, nz))

        for i, x in enumerate(self.x_grid):
            for j, y in enumerate(self.y_grid):
                for k, z in enumerate(self.z_grid):
                    # 计算距离
                    point = np.array([x, y, z])
                    distance = np.linalg.norm(point - tx_antenna.position)

                    if distance < 1.0:
                        field[i, j, k] = 0
                        continue

                    # 计算方向
                    direction = (point - tx_antenna.position) / distance
                    az = np.degrees(np.arctan2(direction[1], direction[0])) % 360
                    el = np.degrees(np.arcsin(direction[2]))

                    # 获取天线增益
                    gain = tx_antenna.get_gain_at(frequency, az, el)

                    # 计算场强（简化模型）
                    fspl = 32.4 + 20 * np.log10(frequency) + 20 * np.log10(distance / 1000)
                    field[i, j, k] = tx_antenna.current_power + gain - fspl

        return field

    def calculate_interference_map(
        self,
        frequency: float = 15.0,
    ) -> np.ndarray:
        """
        计算同址干扰场强分布

        Parameters
        ----------
        frequency : float
            工作频率 (MHz)

        Returns
        -------
        np.ndarray
            干扰场强分布
        """
        nx, ny, nz = len(self.x_grid), len(self.y_grid), len(self.z_grid)
        total_field = np.zeros((nx, ny, nz))

        # 累加所有发射天线的贡献
        for ant_idx, antenna in enumerate(self.station.antennas):
            if antenna.is_transmitting:
                field = self.calculate_field_strength(frequency, ant_idx)
                # 功率叠加（线性域）
                total_field += 10 ** (field / 10)

        # 转回 dB
        total_field = 10 * np.log10(np.maximum(total_field, 1e-10))

        return total_field

    def plot_2d_slice(
        self,
        z_height: float = 10.0,
        frequency: float = 15.0,
        tx_antenna_idx: Optional[int] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        绘制指定高度的 2D 切片

        Parameters
        ----------
        z_height : float
            高度 (米)
        frequency : float
            频率 (MHz)
        tx_antenna_idx : int, optional
            发射天线索引，None 表示叠加所有
        save_path : str, optional
            保存路径
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib not available")
            return

        # 找到最近的 z 索引
        z_idx = np.argmin(np.abs(self.z_grid - z_height))

        if tx_antenna_idx is not None:
            field = self.calculate_field_strength(frequency, tx_antenna_idx)
        else:
            field = self.calculate_interference_map(frequency)

        slice_2d = field[:, :, z_idx]

        # 绘图
        fig, ax = plt.subplots(figsize=(12, 10))

        X, Y = np.meshgrid(self.x_grid, self.y_grid)
        c = ax.pcolormesh(X, Y, slice_2d.T, cmap='jet', shading='auto')
        fig.colorbar(c, ax=ax, label='Field Strength (dBm)')

        # 绘制天线位置
        for antenna in self.station.antennas:
            color = 'red' if antenna.is_transmitting else 'white'
            ax.plot(antenna.position[0], antenna.position[1], 'o',
                    color=color, markersize=8, markeredgecolor='black')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Electromagnetic Field at z={z_height}m, f={frequency}MHz')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_3d_volume(
        self,
        frequency: float = 15.0,
        threshold: float = -80.0,
        save_path: Optional[str] = None,
    ) -> None:
        """
        绘制 3D 体渲染

        Parameters
        ----------
        frequency : float
            频率 (MHz)
        threshold : float
            显示阈值 (dBm)
        save_path : str, optional
            保存路径
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib not available")
            return

        field = self.calculate_interference_map(frequency)

        # 提取高于阈值的点
        mask = field > threshold
        x_idx, y_idx, z_idx = np.where(mask)

        if len(x_idx) == 0:
            print("No points above threshold")
            return

        x = self.x_grid[x_idx]
        y = self.y_grid[y_idx]
        z = self.z_grid[z_idx]
        colors = field[mask]

        # 绘图
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(x, y, z, c=colors, cmap='jet', s=5, alpha=0.3)
        fig.colorbar(scatter, ax=ax, label='Field Strength (dBm)')

        # 绘制天线
        for antenna in self.station.antennas:
            color = 'red' if antenna.is_transmitting else 'blue'
            ax.scatter(*antenna.position, c=color, s=100, marker='^',
                      edgecolors='black')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'3D Electromagnetic Field Distribution, f={frequency}MHz')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
