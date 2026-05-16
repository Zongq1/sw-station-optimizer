"""短波台站多目标优化问题定义 - ShortwaveStationProblem"""

from __future__ import annotations

from typing import Optional

import numpy as np
from pymoo.core.problem import Problem
from scipy.spatial import ConvexHull

from ..models.station import StationDigitalTwin
from ..simulation.em_simulator import EMSimulator
from ..config import SystemConfig


class ShortwaveStationProblem(Problem):
    """
    短波台站多目标优化问题

    将台站天线布局优化建模为 pymoo Problem，支持 NSGA-II/III、MOEA/D 等算法。

    优化变量：
        - 每副天线 4 个变量: [x, y, z, azimuth]
        - 总变量数: n_antennas * 4

    目标函数：
        - f1: 最大化全局通信覆盖（取负值转最小化）
        - f2: 最小化同址干扰风险
        - f3: 最小化台站建设成本

    约束条件：
        - g: 隔离度约束（所有天线对需满足隔离度要求）
    """

    def __init__(
        self,
        station: Optional[StationDigitalTwin] = None,
        config: Optional[SystemConfig] = None,
        n_antennas: int = 50,
        boundary: tuple[float, float, float, float] = (0, 2000, 0, 2000),
    ):
        """
        初始化优化问题

        Parameters
        ----------
        station : StationDigitalTwin, optional
            台站模型，如果不提供则创建默认模型
        config : SystemConfig, optional
            系统配置
        n_antennas : int
            天线数量
        boundary : tuple
            台站边界 (x_min, x_max, y_min, y_max)
        """
        self.config = config or SystemConfig.default()
        self.station = station or StationDigitalTwin.create_default_station(n_antennas)
        self.boundary = boundary
        self.n_antennas = self.station.n_antennas

        # 电磁仿真器
        self.em_simulator = EMSimulator(
            ground_conductivity=self.config.station.ground_conductivity,
            ground_permittivity=self.config.station.ground_permittivity,
        )

        # 优化目标数量
        n_obj = 4

        # 约束数量：所有天线对的隔离度约束
        n_constr = self.n_antennas * (self.n_antennas - 1) // 2

        # 变量边界
        x_min, x_max, y_min, y_max = boundary
        xl = np.zeros(self.n_antennas * 4)
        xu = np.zeros(self.n_antennas * 4)

        for i in range(self.n_antennas):
            # x, y 坐标
            xl[i * 4] = x_min
            xu[i * 4] = x_max
            xl[i * 4 + 1] = y_min
            xu[i * 4 + 1] = y_max
            # z 坐标（高度）
            xl[i * 4 + 2] = 5.0    # 最低 5m
            xu[i * 4 + 2] = 80.0   # 最高 80m
            # 方位角
            xl[i * 4 + 3] = 0.0
            xu[i * 4 + 3] = 360.0

        super().__init__(
            n_var=self.n_antennas * 4,
            n_obj=n_obj,
            n_constr=n_constr,
            xl=xl,
            xu=xu,
        )

    def _evaluate(self, X: np.ndarray, out: np.ndarray, *args, **kwargs) -> None:
        """
        评估种群个体

        Parameters
        ----------
        X : np.ndarray
            决策变量矩阵, shape: (pop_size, n_var)
        out : dict
            输出字典，包含 "F" (目标值) 和 "G" (约束值)
        """
        pop_size = X.shape[0]
        F = np.zeros((pop_size, self.n_obj))
        G = np.zeros((pop_size, self.n_constr))

        for idx in range(pop_size):
            # 解码变量到台站布局
            self._decode_layout(X[idx])

            # 计算目标函数
            f1_coverage = self._calculate_coverage()
            f2_interference = self._calculate_interference_risk()
            f3_cost = self._calculate_cost()
            f4_spectrum = self._calculate_spectrum_efficiency()

            # 转为最小化问题
            F[idx, 0] = -f1_coverage      # 最大化覆盖 -> 最小化负覆盖
            F[idx, 1] = f2_interference    # 最小化干扰
            F[idx, 2] = f3_cost            # 最小化成本
            F[idx, 3] = -f4_spectrum       # 最大化频谱效率

            # 计算约束
            G[idx] = self._calculate_constraints()

        out["F"] = F
        out["G"] = G

    def _decode_layout(self, x: np.ndarray) -> None:
        """
        将决策变量解码为台站布局

        Parameters
        ----------
        x : np.ndarray
            决策变量向量, shape: (n_var,)
        """
        for i in range(self.n_antennas):
            self.station.antennas[i].position = np.array([
                x[i * 4],
                x[i * 4 + 1],
                x[i * 4 + 2],
            ])
            self.station.antennas[i].azimuth = x[i * 4 + 3] % 360.0

    def _calculate_coverage(self) -> float:
        """
        计算全局通信覆盖评分

        考虑天线增益的方向匹配度和覆盖均匀性。

        Returns
        -------
        float
            覆盖评分 (0-1, 越高越好)
        """
        # 简化模型：评估天线在不同方向的增益覆盖
        test_azimuths = np.linspace(0, 360, 36, endpoint=False)
        test_elevations = np.linspace(5, 60, 12)

        coverage_map = np.zeros((len(test_azimuths), len(test_elevations)))

        # 多频率评估覆盖
        test_frequencies = [5.0, 10.0, 15.0, 20.0, 25.0]

        for ant in self.station.antennas:
            if ant.pattern is not None:
                for freq in test_frequencies:
                    for ai, az in enumerate(test_azimuths):
                        for ei, el in enumerate(test_elevations):
                            gain = ant.get_gain_at(freq, az, el)
                            coverage_map[ai, ei] += 10 ** (gain / 10)

        # 归一化频率数量
        coverage_map /= len(test_frequencies)

        # 归一化
        if coverage_map.max() > 0:
            coverage_map = coverage_map / coverage_map.max()

        # 覆盖评分：平均覆盖度 * 覆盖均匀性
        mean_coverage = coverage_map.mean()
        coverage_std = coverage_map.std()
        uniformity = 1.0 / (1.0 + coverage_std)

        return float(mean_coverage * uniformity)

    def _calculate_interference_risk(self) -> float:
        """
        计算同址干扰风险评分

        基于天线间的隔离度计算潜在干扰风险。

        Returns
        -------
        float
            干扰风险评分 (越低越好)
        """
        total_risk = 0.0
        max_interference = self.config.interference.max_allowed_interference

        # 计算隔离度矩阵
        for i in range(self.n_antennas):
            for j in range(i + 1, self.n_antennas):
                ant_i = self.station.antennas[i]
                ant_j = self.station.antennas[j]

                # 使用天线实际工作频率，无则用多频率平均
                if ant_i.current_frequency and ant_j.current_frequency:
                    freq = (ant_i.current_frequency + ant_j.current_frequency) / 2
                else:
                    from ..config import DEFAULT_FREQUENCY_MHZ
                    freq = DEFAULT_FREQUENCY_MHZ

                isolation = self.em_simulator.calculate_isolation(
                    ant_i, ant_j, freq
                )

                # 使用天线实际功率，无则用典型值
                tx_power = ant_i.current_power if ant_i.is_transmitting else 30.0
                interference = tx_power - isolation

                # 超限部分作为风险
                if interference > max_interference:
                    risk = interference - max_interference
                    total_risk += risk

        return float(total_risk)

    def _calculate_cost(self) -> float:
        """
        计算台站建设成本评分

        考虑占地面积、馈线长度等因素。

        Returns
        -------
        float
            成本评分 (越低越好)
        """
        positions = np.array([ant.position for ant in self.station.antennas])

        # 占地面积（使用凸包）
        positions_2d = positions[:, :2]
        try:
            hull = ConvexHull(positions_2d)
            area = hull.volume  # 2D 凸包的 volume 就是面积
        except Exception:
            # 退化情况用边界框
            x_range = positions[:, 0].max() - positions[:, 0].min()
            y_range = positions[:, 1].max() - positions[:, 1].min()
            area = x_range * y_range

        # 归一化面积
        x_boundary = self.boundary[1] - self.boundary[0]
        y_boundary = self.boundary[3] - self.boundary[2]
        max_area = x_boundary * y_boundary
        area_score = area / max_area

        # 馈线长度估算 - 基于最小生成树拓扑
        # 机房位置（边界中心）
        center = np.array([
            (self.boundary[0] + self.boundary[1]) / 2,
            (self.boundary[2] + self.boundary[3]) / 2,
        ])

        # Prim 最小生成树算法
        n = len(positions)
        visited = [False] * n
        min_edge = np.full(n, np.inf)
        min_edge[0] = 0
        total_cable = 0.0

        for _ in range(n):
            # 找未访问节点中最小边
            u = -1
            for v in range(n):
                if not visited[v] and (u == -1 or min_edge[v] < min_edge[u]):
                    u = v

            visited[u] = True
            total_cable += min_edge[u]

            # 更新邻接边（到机房的距离也作为一条边）
            for v in range(n):
                if not visited[v]:
                    # 天线间距离
                    dist = np.linalg.norm(positions[u, :2] - positions[v, :2])
                    min_edge[v] = min(min_edge[v], dist)

        # 加上每副天线到机房的馈线（树的根连接）
        for i in range(n):
            dist_to_center = np.linalg.norm(positions[i, :2] - center)
            total_cable += dist_to_center * 0.1  # 10% 冗余连接

        max_cable = self.n_antennas * np.sqrt(max_area) / 2
        cable_score = total_cable / max(max_cable, 1)

        # 综合成本评分
        return float(0.6 * area_score + 0.4 * cable_score)

    def _calculate_constraints(self) -> np.ndarray:
        """
        计算隔离度约束

        Returns
        -------
        np.ndarray
            约束值数组，g <= 0 表示满足约束
        """
        constraints = []
        max_interference = self.config.interference.max_allowed_interference
        filter_rejection = self.config.interference.filter_rejection

        for i in range(self.n_antennas):
            for j in range(i + 1, self.n_antennas):
                ant_i = self.station.antennas[i]
                ant_j = self.station.antennas[j]

                # 计算隔离度 - 使用实际频率
                from ..config import DEFAULT_FREQUENCY_MHZ
                freq = ant_i.current_frequency if ant_i.current_frequency else DEFAULT_FREQUENCY_MHZ

                isolation = self.em_simulator.calculate_isolation(
                    ant_i, ant_j, freq
                )

                # 计算干扰功率 - 使用实际功率
                tx_power = ant_i.current_power if ant_i.is_transmitting else 30.0
                interference = tx_power - isolation

                # 约束：干扰功率 <= 最大允许值
                # g = interference - max_allowed <= 0
                g = interference - max_interference
                constraints.append(g)

        return np.array(constraints)

    def _calculate_spectrum_efficiency(self) -> float:
        """
        计算频谱利用效率

        基于天线覆盖的频率范围和方向多样性。
        """
        # 评估天线覆盖的频率范围利用率
        covered_freqs = set()
        for ant in self.station.antennas:
            if ant.pattern is not None:
                for freq in ant.pattern.frequencies:
                    covered_freqs.add(round(freq, 0))

        # 频率覆盖度
        total_freq_bins = 29  # 2-30 MHz, 1 MHz 分辨率
        freq_coverage = len(covered_freqs) / total_freq_bins

        # 方向多样性 - 天线朝向的均匀性
        azimuths = np.array([ant.azimuth for ant in self.station.antennas])
        az_hist, _ = np.histogram(azimuths, bins=12, range=(0, 360))
        az_uniformity = 1.0 - az_hist.std() / (az_hist.mean() + 1e-10)

        return float(0.6 * freq_coverage + 0.4 * max(0, az_uniformity))

    def decode_solution(self, x: np.ndarray) -> StationDigitalTwin:
        """
        将优化解解码为台站模型

        Parameters
        ----------
        x : np.ndarray
            优化变量向量

        Returns
        -------
        StationDigitalTwin
            解码后的台站模型
        """
        self._decode_layout(x)
        return self.station


class SimplifiedStationProblem(Problem):
    """
    简化版台站优化问题（用于快速测试）

    减少变量和约束维度，适合算法验证。
    """

    def __init__(self, n_antennas: int = 10):
        """
        初始化简化问题

        Parameters
        ----------
        n_antennas : int
            天线数量（建议较小值）
        """
        self.n_antennas = n_antennas

        # 简化：每副天线只有 x, y 两个变量
        n_var = n_antennas * 2
        n_obj = 2  # 覆盖和干扰两个目标
        n_constr = 0

        xl = np.zeros(n_var)
        xu = np.zeros(n_var)
        for i in range(n_antennas):
            xl[i * 2] = 0
            xu[i * 2] = 1000
            xl[i * 2 + 1] = 0
            xu[i * 2 + 1] = 1000

        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_constr=n_constr,
            xl=xl,
            xu=xu,
        )

    def _evaluate(self, X: np.ndarray, out: np.ndarray, *args, **kwargs) -> None:
        """评估种群"""
        pop_size = X.shape[0]
        F = np.zeros((pop_size, self.n_obj))

        for idx in range(pop_size):
            positions = X[idx].reshape(-1, 2)

            # 目标1：最大化覆盖（最小化负覆盖）
            coverage = self._simple_coverage(positions)
            F[idx, 0] = -coverage

            # 目标2：最小化干扰
            interference = self._simple_interference(positions)
            F[idx, 1] = interference

        out["F"] = F

    def _simple_coverage(self, positions: np.ndarray) -> float:
        """简化覆盖计算"""
        # 计算位置分散度
        centroid = positions.mean(axis=0)
        distances = np.linalg.norm(positions - centroid, axis=1)
        spread = distances.std()
        return float(spread / 500.0)  # 归一化

    def _simple_interference(self, positions: np.ndarray) -> float:
        """简化干扰计算"""
        # 计算最小间距
        min_dist = np.inf
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                min_dist = min(min_dist, dist)

        # 距离越小，干扰越大
        return float(max(0, 100 - min_dist) / 100.0)
