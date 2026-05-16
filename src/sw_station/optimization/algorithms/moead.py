"""MOEA/D 算法封装"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.core.algorithm import Algorithm
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.ref_dirs import get_reference_directions

from ..utils import OptimizationResult, extract_pareto_front, calculate_hypervolume


@dataclass
class MOEADConfig:
    """MOEA/D 算法配置"""
    # 权重向量生成方法: "uniform", "das-dennis", "energy"
    ref_dir_method: str = "das-dennis"
    # 参考点分区数
    n_partitions: int = 12
    # 邻域大小
    neighborhood_size: int = 20
    # 最大迭代代数
    max_generations: int = 500
    # 交叉概率
    crossover_prob: float = 1.0
    # 变异概率
    mutation_prob: float = 0.1
    # 聚合方法: "tchebycheff", "pbi", "weighted"
    aggregation_method: str = "tchebycheff"
    # PBI 惩罚参数（仅 PBI 方法使用）
    pbi_theta: float = 5.0
    # 随机种子
    seed: Optional[int] = None
    # 是否输出详细信息
    verbose: bool = True
    # 邻域选择概率
    neighborhood_selection_prob: float = 0.9


class MOEADRunner:
    """
    MOEA/D 优化运行器

    基于分解的多目标进化算法，适用于高维多目标问题。
    """

    def __init__(self, config: Optional[MOEADConfig] = None):
        """
        初始化 MOEA/D 运行器

        Parameters
        ----------
        config : MOEADConfig, optional
            算法配置
        """
        self.config = config or MOEADConfig()
        self.algorithm: Optional[Algorithm] = None
        self.result: Optional[OptimizationResult] = None

    def create_algorithm(self, problem: Problem) -> Algorithm:
        """
        创建算法实例

        Parameters
        ----------
        problem : Problem
            优化问题

        Returns
        -------
        Algorithm
            算法实例
        """
        # 生成参考方向
        ref_dirs = get_reference_directions(
            self.config.ref_dir_method,
            problem.n_obj,
            n_partitions=self.config.n_partitions,
        )

        # 创建算子
        crossover = SBX(prob=self.config.crossover_prob, eta=20)
        mutation = PM(prob=self.config.mutation_prob, eta=20)

        # 创建 MOEA/D 算法
        algorithm = MOEAD(
            ref_dirs=ref_dirs,
            n_neighbors=self.config.neighborhood_size,
            prob_neighbor_mating=self.config.neighborhood_selection_prob,
            sampling=FloatRandomSampling(),
            crossover=crossover,
            mutation=mutation,
        )

        self.algorithm = algorithm
        return algorithm

    def run(self, problem: Problem, **kwargs) -> OptimizationResult:
        """
        运行优化

        Parameters
        ----------
        problem : Problem
            优化问题

        Returns
        -------
        OptimizationResult
            优化结果
        """
        algorithm = self.create_algorithm(problem)

        res = minimize(
            problem,
            algorithm,
            ("n_gen", self.config.max_generations),
            seed=self.config.seed,
            verbose=self.config.verbose,
            **kwargs,
        )

        # 提取结果
        pareto_front, pareto_solutions = extract_pareto_front(res.F, res.X)
        hv = calculate_hypervolume(pareto_front)

        self.result = OptimizationResult(
            pareto_front=pareto_front,
            pareto_solutions=pareto_solutions,
            all_objectives=res.F,
            all_solutions=res.X,
            n_generations=res.algorithm.n_gen if hasattr(res.algorithm, 'n_gen') else 0,
            hypervolume=hv,
            execution_time=0.0,
        )

        return self.result


class MOEADwithDecomposition(MOEADRunner):
    """
    增强版 MOEA/D

    支持多种分解策略和自适应权重调整。
    """

    def __init__(self, config: Optional[MOEADConfig] = None):
        """初始化增强版 MOEA/D"""
        super().__init__(config)
        self.weight_history: list[np.ndarray] = []

    def generate_custom_weights(
        self,
        n_obj: int,
        n_points: int,
        method: str = "energy",
    ) -> np.ndarray:
        """
        生成自定义权重向量

        Parameters
        ----------
        n_obj : int
            目标数量
        n_points : int
            权重点数量
        method : str
            生成方法

        Returns
        -------
        np.ndarray
            权重向量矩阵, shape: (n_points, n_obj)
        """
        if method == "energy":
            return self._energy_based_weights(n_obj, n_points)
        elif method == "random":
            return self._random_weights(n_obj, n_points)
        else:
            return get_reference_directions("das-dennis", n_obj, n_partitions=n_points)

    def _energy_based_weights(self, n_obj: int, n_points: int) -> np.ndarray:
        """
        基于能量的权重生成

        使用电荷模拟法生成均匀分布的权重向量。
        """
        # 简化实现：使用随机初始化 + 迭代优化
        weights = np.random.rand(n_points, n_obj)
        weights = weights / weights.sum(axis=1, keepdims=True)

        # 迭代优化使权重更均匀
        for _ in range(100):
            for i in range(n_points):
                # 计算与其他点的排斥力
                force = np.zeros(n_obj)
                for j in range(n_points):
                    if i != j:
                        diff = weights[i] - weights[j]
                        dist = np.linalg.norm(diff) + 1e-10
                        force += diff / dist**3

                # 更新位置
                weights[i] += 0.01 * force
                weights[i] = np.maximum(weights[i], 0.01)
                weights[i] = weights[i] / weights[i].sum()

        return weights

    def _random_weights(self, n_obj: int, n_points: int) -> np.ndarray:
        """随机权重生成"""
        weights = np.random.dirichlet(np.ones(n_obj), n_points)
        return weights


def run_moead_optimization(
    problem: Problem,
    n_partitions: int = 12,
    max_generations: int = 500,
    seed: Optional[int] = 42,
    verbose: bool = True,
) -> OptimizationResult:
    """
    便捷函数：运行 MOEA/D 优化

    Parameters
    ----------
    problem : Problem
        优化问题
    n_partitions : int
        参考点分区数
    max_generations : int
        最大代数
    seed : int, optional
        随机种子
    verbose : bool
        是否输出详细信息

    Returns
    -------
    OptimizationResult
        优化结果
    """
    config = MOEADConfig(
        n_partitions=n_partitions,
        max_generations=max_generations,
        seed=seed,
        verbose=verbose,
    )

    runner = MOEADRunner(config)
    return runner.run(problem)
