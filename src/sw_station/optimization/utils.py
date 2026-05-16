"""优化工具函数"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class OptimizationResult:
    """
    优化结果数据结构

    存储多目标优化的完整结果信息。
    """
    # 帕累托前沿目标值
    pareto_front: np.ndarray
    # 帕累托前沿解
    pareto_solutions: np.ndarray
    # 所有评估的目标值
    all_objectives: np.ndarray
    # 所有评估的解
    all_solutions: np.ndarray
    # 迭代代数
    n_generations: int
    # 超体积指标
    hypervolume: float
    # 执行时间 (秒)
    execution_time: float
    # 额外信息
    metadata: dict = field(default_factory=dict)

    @property
    def n_pareto(self) -> int:
        """帕累托前沿解的数量"""
        return len(self.pareto_front)

    @property
    def n_objectives(self) -> int:
        """目标数量"""
        return self.pareto_front.shape[1] if len(self.pareto_front) > 0 else 0

    @property
    def n_variables(self) -> int:
        """变量数量"""
        return self.pareto_solutions.shape[1] if len(self.pareto_solutions) > 0 else 0

    def get_best_solution(self, objective_idx: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """
        获取指定目标的最优解

        Parameters
        ----------
        objective_idx : int
            目标索引

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (最优解, 最优目标值)
        """
        best_idx = np.argmin(self.pareto_front[:, objective_idx])
        return self.pareto_solutions[best_idx], self.pareto_front[best_idx]

    def get_knee_point(self) -> tuple[np.ndarray, np.ndarray]:
        """
        获取膝点解（各目标平衡的解）

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (膝点解, 膝点目标值)
        """
        # 归一化目标值
        obj_min = self.pareto_front.min(axis=0)
        obj_max = self.pareto_front.max(axis=0)
        obj_range = obj_max - obj_min
        obj_range[obj_range < 1e-10] = 1.0

        normalized = (self.pareto_front - obj_min) / obj_range

        # 找到距离理想点最近的解
        ideal_point = np.zeros(self.n_objectives)
        distances = np.linalg.norm(normalized - ideal_point, axis=1)
        knee_idx = np.argmin(distances)

        return self.pareto_solutions[knee_idx], self.pareto_front[knee_idx]

    def summary(self) -> dict:
        """
        生成结果摘要

        Returns
        -------
        dict
            摘要信息
        """
        return {
            "n_pareto_solutions": self.n_pareto,
            "n_objectives": self.n_objectives,
            "n_variables": self.n_variables,
            "n_generations": self.n_generations,
            "hypervolume": self.hypervolume,
            "execution_time": self.execution_time,
            "pareto_front_range": {
                f"obj_{i}": {
                    "min": float(self.pareto_front[:, i].min()),
                    "max": float(self.pareto_front[:, i].max()),
                    "mean": float(self.pareto_front[:, i].mean()),
                }
                for i in range(self.n_objectives)
            },
        }


def is_pareto_efficient(costs: np.ndarray) -> np.ndarray:
    """
    判断哪些解是帕累托有效的

    Parameters
    ----------
    costs : np.ndarray
        目标值矩阵, shape: (n_solutions, n_objectives)
        假设所有目标都是最小化

    Returns
    -------
    np.ndarray
        布尔数组，True 表示帕累托有效
    """
    n_solutions = len(costs)
    is_efficient = np.ones(n_solutions, dtype=bool)

    for i in range(n_solutions):
        if not is_efficient[i]:
            continue

        # 检查是否被其他解支配
        for j in range(n_solutions):
            if i == j:
                continue

            # 检查 j 是否支配 i
            if np.all(costs[j] <= costs[i]) and np.any(costs[j] < costs[i]):
                is_efficient[i] = False
                break

    return is_efficient


def extract_pareto_front(
    objectives: np.ndarray,
    solutions: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    提取帕累托前沿

    Parameters
    ----------
    objectives : np.ndarray
        目标值矩阵
    solutions : np.ndarray, optional
        解矩阵

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (帕累托前沿目标值, 帕累托前沿解)
    """
    is_efficient = is_pareto_efficient(objectives)
    pareto_front = objectives[is_efficient]

    if solutions is not None:
        pareto_solutions = solutions[is_efficient]
    else:
        pareto_solutions = np.where(is_efficient)[0].reshape(-1, 1)

    return pareto_front, pareto_solutions


def calculate_hypervolume(
    pareto_front: np.ndarray,
    reference_point: Optional[np.ndarray] = None,
) -> float:
    """
    计算超体积指标

    Parameters
    ----------
    pareto_front : np.ndarray
        帕累托前沿目标值
    reference_point : np.ndarray, optional
        参考点，默认为各目标最大值的 1.1 倍

    Returns
    -------
    float
        超体积值
    """
    if len(pareto_front) == 0:
        return 0.0

    n_objectives = pareto_front.shape[1]

    if reference_point is None:
        reference_point = pareto_front.max(axis=0) * 1.1

    # 简化实现：2D 情况下的超体积计算
    if n_objectives == 2:
        return _hypervolume_2d(pareto_front, reference_point)
    else:
        # 高维情况使用近似方法
        return _hypervolume_approximate(pareto_front, reference_point)


def _hypervolume_2d(
    pareto_front: np.ndarray,
    reference_point: np.ndarray,
) -> float:
    """计算 2D 超体积"""
    # 按第一个目标排序
    sorted_front = pareto_front[pareto_front[:, 0].argsort()]

    hv = 0.0
    prev_x = 0.0

    for point in sorted_front:
        x, y = point
        width = x - prev_x
        height = reference_point[1] - y
        hv += width * height
        prev_x = x

    return max(hv, 0.0)


def _hypervolume_approximate(
    pareto_front: np.ndarray,
    reference_point: np.ndarray,
    n_samples: int = 10000,
) -> float:
    """
    近似计算高维超体积

    使用蒙特卡洛采样方法。
    """
    n_objectives = pareto_front.shape[1]

    # 确定采样边界
    lower_bound = np.zeros(n_objectives)
    upper_bound = reference_point

    # 生成随机采样点
    samples = np.random.uniform(
        lower_bound,
        upper_bound,
        size=(n_samples, n_objectives),
    )

    # 检查每个采样点是否被帕累托前沿支配
    dominated_count = 0
    for sample in samples:
        for point in pareto_front:
            if np.all(point <= sample):
                dominated_count += 1
                break

    # 超体积 = 被支配比例 * 总体积
    total_volume = np.prod(upper_bound - lower_bound)
    hv = (dominated_count / n_samples) * total_volume

    return hv


def calculate_spacing(pareto_front: np.ndarray) -> float:
    """
    计算间距指标

    衡量帕累托前沿解的分布均匀性。

    Parameters
    ----------
    pareto_front : np.ndarray
        帕累托前沿

    Returns
    -------
    float
        间距值（越小越均匀）
    """
    n = len(pareto_front)
    if n <= 1:
        return 0.0

    # 计算每个解到最近邻的距离
    min_distances = np.zeros(n)
    for i in range(n):
        distances = np.linalg.norm(
            pareto_front[i] - pareto_front, axis=1
        )
        distances[i] = np.inf  # 排除自身
        min_distances[i] = distances.min()

    # 间距 = 最近距离的标准差
    return float(min_distances.std())


def calculate_igd(
    pareto_front: np.ndarray,
    true_front: np.ndarray,
) -> float:
    """
    计算反向世代距离 (IGD)

    Parameters
    ----------
    pareto_front : np.ndarray
        近似帕累托前沿
    true_front : np.ndarray
        真实帕累托前沿

    Returns
    -------
    float
        IGD 值（越小越好）
    """
    if len(true_front) == 0:
        return np.inf

    total_distance = 0.0
    for point in true_front:
        distances = np.linalg.norm(pareto_front - point, axis=1)
        total_distance += distances.min()

    return total_distance / len(true_front)
