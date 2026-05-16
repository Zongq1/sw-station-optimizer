"""隔离度干涉云可视化 - InterferenceCloudRenderer"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from ...models.station import StationDigitalTwin, AntennaDevice
from ...simulation.em_simulator import EMSimulator


class InterferenceCloudRenderer:
    """
    隔离度干涉云渲染器

    将天线间的电磁耦合关系以 3D 干连线的形式可视化。
    """

    def __init__(
        self,
        station: StationDigitalTwin,
        em_simulator: Optional[EMSimulator] = None,
    ):
        """
        初始化渲染器

        Parameters
        ----------
        station : StationDigitalTwin
            台站模型
        em_simulator : EMSimulator, optional
            电磁仿真器
        """
        self.station = station
        self.em_simulator = em_simulator or EMSimulator()

    def calculate_interference_cloud(
        self,
        frequency: float = 15.0,
        resolution: int = 20,
    ) -> dict:
        """
        计算干涉云数据

        Parameters
        ----------
        frequency : float
            频率 (MHz)
        resolution : int
            空间分辨率

        Returns
        -------
        dict
            干涉云数据
        """
        n_antennas = self.station.n_antennas

        # 计算隔离度矩阵
        isolation_matrix = np.zeros((n_antennas, n_antennas))
        for i in range(n_antennas):
            for j in range(n_antennas):
                if i != j:
                    isolation_matrix[i, j] = self.em_simulator.calculate_isolation(
                        self.station.antennas[i],
                        self.station.antennas[j],
                        frequency,
                    )

        # 生成干涉云数据
        cloud_points = []
        cloud_colors = []

        for i in range(n_antennas):
            for j in range(i + 1, n_antennas):
                ant_i = self.station.antennas[i]
                ant_j = self.station.antennas[j]

                isolation = isolation_matrix[i, j]

                # 生成连线上的采样点
                n_samples = resolution
                for t in np.linspace(0, 1, n_samples):
                    point = ant_i.position * (1 - t) + ant_j.position * t

                    # 添加垂直于连线的扰动
                    direction = ant_j.position - ant_i.position
                    distance = np.linalg.norm(direction)

                    # 扰动强度与隔离度相关（隔离度越低，扰动越大）
                    perturbation_scale = max(0, 50 - isolation / 2)

                    perturbation = np.random.normal(0, perturbation_scale, 3)
                    perturbation[2] = abs(perturbation[2])  # z 方向非负

                    cloud_point = point + perturbation
                    cloud_points.append(cloud_point)

                    # 颜色编码：红色=高干扰，蓝色=低干扰
                    interference_level = 1.0 / (1.0 + np.exp(-isolation / 20))
                    cloud_colors.append(interference_level)

        return {
            "points": np.array(cloud_points),
            "colors": np.array(cloud_colors),
            "isolation_matrix": isolation_matrix,
        }

    def plot_interference_cloud(
        self,
        frequency: float = 15.0,
        show_connections: bool = True,
        threshold: float = -100.0,
        save_path: Optional[str] = None,
    ) -> None:
        """
        绘制干涉云

        Parameters
        ----------
        frequency : float
            频率 (MHz)
        show_connections : bool
            是否显示天线间连线
        threshold : float
            显示阈值 (dB)
        save_path : str, optional
            保存路径
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib not available")
            return

        cloud_data = self.calculate_interference_cloud(frequency)

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制干涉云点
        points = cloud_data["points"]
        colors = cloud_data["colors"]

        if len(points) > 0:
            scatter = ax.scatter(
                points[:, 0], points[:, 1], points[:, 2],
                c=colors, cmap='RdYlBu_r', s=10, alpha=0.3,
            )
            fig.colorbar(scatter, ax=ax, label='Interference Level')

        # 绘制天线
        for i, antenna in enumerate(self.station.antennas):
            color = 'red' if antenna.is_transmitting else 'blue'
            ax.scatter(
                *antenna.position,
                c=color, s=100, marker='^',
                edgecolors='black', linewidths=1,
            )
            ax.text(
                antenna.position[0], antenna.position[1],
                antenna.position[2] + 5,
                antenna.id, fontsize=8,
            )

        # 绘制连线
        if show_connections:
            isolation_matrix = cloud_data["isolation_matrix"]
            for i in range(self.station.n_antennas):
                for j in range(i + 1, self.station.n_antennas):
                    if isolation_matrix[i, j] < threshold:
                        ant_i = self.station.antennas[i]
                        ant_j = self.station.antennas[j]

                        ax.plot(
                            [ant_i.position[0], ant_j.position[0]],
                            [ant_i.position[1], ant_j.position[1]],
                            [ant_i.position[2], ant_j.position[2]],
                            'r--', alpha=0.5, linewidth=1,
                        )

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Interference Cloud @ {frequency}MHz')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_isolation_matrix(
        self,
        frequency: float = 15.0,
        save_path: Optional[str] = None,
    ) -> None:
        """
        绘制隔离度矩阵热力图

        Parameters
        ----------
        frequency : float
            频率 (MHz)
        save_path : str, optional
            保存路径
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib not available")
            return

        n = self.station.n_antennas
        isolation_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    isolation_matrix[i, j] = self.em_simulator.calculate_isolation(
                        self.station.antennas[i],
                        self.station.antennas[j],
                        frequency,
                    )

        fig, ax = plt.subplots(figsize=(12, 10))

        im = ax.imshow(isolation_matrix, cmap='RdYlBu_r', aspect='auto')
        fig.colorbar(im, ax=ax, label='Isolation (dB)')

        ax.set_xlabel('Antenna Index')
        ax.set_ylabel('Antenna Index')
        ax.set_title(f'Isolation Matrix @ {frequency}MHz')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_worst_pairs(
        self,
        frequency: float = 15.0,
        n_pairs: int = 10,
        save_path: Optional[str] = None,
    ) -> None:
        """
        绘制最差隔离度天线对

        Parameters
        ----------
        frequency : float
            频率 (MHz)
        n_pairs : int
            显示的天线对数量
        save_path : str, optional
            保存路径
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib not available")
            return

        # 计算所有天线对的隔离度
        pairs = []
        for i in range(self.station.n_antennas):
            for j in range(i + 1, self.station.n_antennas):
                isolation = self.em_simulator.calculate_isolation(
                    self.station.antennas[i],
                    self.station.antennas[j],
                    frequency,
                )
                pairs.append((i, j, isolation))

        # 排序，取最差的
        pairs.sort(key=lambda x: x[2])
        worst_pairs = pairs[:n_pairs]

        # 绘图
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制所有天线（灰色）
        for antenna in self.station.antennas:
            ax.scatter(
                *antenna.position,
                c='gray', s=50, alpha=0.3,
            )

        # 绘制最差天线对（红色连线）
        for i, j, isolation in worst_pairs:
            ant_i = self.station.antennas[i]
            ant_j = self.station.antennas[j]

            ax.plot(
                [ant_i.position[0], ant_j.position[0]],
                [ant_i.position[1], ant_j.position[1]],
                [ant_i.position[2], ant_j.position[2]],
                'r-', linewidth=2, alpha=0.7,
            )

            # 高亮天线
            ax.scatter(
                *ant_i.position, c='red', s=100, marker='o',
                edgecolors='black',
            )
            ax.scatter(
                *ant_j.position, c='red', s=100, marker='o',
                edgecolors='black',
            )

            # 标注隔离度
            mid_point = (ant_i.position + ant_j.position) / 2
            ax.text(
                mid_point[0], mid_point[1], mid_point[2],
                f'{isolation:.1f}dB', fontsize=8,
            )

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Worst {n_pairs} Isolation Pairs @ {frequency}MHz')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
