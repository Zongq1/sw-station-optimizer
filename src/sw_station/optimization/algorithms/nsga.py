"""NSGA-II/III 算法封装"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.algorithm import Algorithm
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.ref_dirs import get_reference_directions

from ..utils import OptimizationResult, extract_pareto_front, calculate_hypervolume


@dataclass
class NSGAConfig:
    """NSGA 算法配置"""
    # 种群大小
    population_size: int = 100
    # 最大迭代代数
    max_generations: int = 500
    # 交叉概率
    crossover_prob: float = 0.9
    # 交叉分布指数
    crossover_eta: float = 15.0
    # 变异概率
    mutation_prob: float = 0.1
    # 变异分布指数
    mutation_eta: float = 20.0
    # 随机种子
    seed: Optional[int] = None
    # 是否输出详细信息
    verbose: bool = True
    # NSGA-III 参考点分区数
    n_partitions: int = 12
    # 算法变体: "nsga2" 或 "nsga3"
    variant: str = "nsga3"


class NSGARunner:
    """
    NSGA-II/III 优化运行器

    封装 pymoo 的 NSGA 算法，提供简化的运行接口。
    """

    def __init__(self, config: Optional[NSGAConfig] = None):
        """
        初始化 NSGA 运行器

        Parameters
        ----------
        config : NSGAConfig, optional
            算法配置
        """
        self.config = config or NSGAConfig()
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
        # 创建变异和交叉算子
        crossover = SBX(
            prob=self.config.crossover_prob,
            eta=self.config.crossover_eta,
        )
        mutation = PM(
            prob=self.config.mutation_prob,
            eta=self.config.mutation_eta,
        )

        if self.config.variant == "nsga3":
            # 生成参考方向
            ref_dirs = get_reference_directions(
                "das-dennis",
                problem.n_obj,
                n_partitions=self.config.n_partitions,
            )

            algorithm = NSGA3(
                pop_size=self.config.population_size,
                ref_dirs=ref_dirs,
                sampling=FloatRandomSampling(),
                crossover=crossover,
                mutation=mutation,
                eliminate_duplicates=True,
            )
        else:
            algorithm = NSGA2(
                pop_size=self.config.population_size,
                sampling=FloatRandomSampling(),
                crossover=crossover,
                mutation=mutation,
                eliminate_duplicates=True,
            )

        self.algorithm = algorithm
        return algorithm

    def run(
        self,
        problem: Problem,
        callback: Optional[Callable] = None,
    ) -> OptimizationResult:
        """
        运行优化

        Parameters
        ----------
        problem : Problem
            优化问题
        callback : Callable, optional
            每代回调函数

        Returns
        -------
        OptimizationResult
            优化结果
        """
        algorithm = self.create_algorithm(problem)

        # 运行优化
        minimize_kwargs = {
            "seed": self.config.seed,
            "verbose": self.config.verbose,
        }
        if callback is not None:
            minimize_kwargs["callback"] = callback

        res = minimize(
            problem,
            algorithm,
            ("n_gen", self.config.max_generations),
            **minimize_kwargs,
        )

        # 提取结果
        pareto_front, pareto_solutions = extract_pareto_front(res.F, res.X)

        # 计算超体积
        hv = calculate_hypervolume(pareto_front)

        self.result = OptimizationResult(
            pareto_front=pareto_front,
            pareto_solutions=pareto_solutions,
            all_objectives=res.F,
            all_solutions=res.X,
            n_generations=res.algorithm.n_gen if hasattr(res.algorithm, 'n_gen') else 0,
            hypervolume=hv,
            execution_time=0.0,  # 需要外部计时
        )

        return self.result

    def run_with_callback(
        self,
        problem: Problem,
        on_generation: Optional[Callable[[int, np.ndarray], None]] = None,
    ) -> OptimizationResult:
        """
        带回调的优化运行

        Parameters
        ----------
        problem : Problem
            优化问题
        on_generation : Callable, optional
            每代回调，参数为 (generation, objectives)

        Returns
        -------
        OptimizationResult
            优化结果
        """
        generation_counter = [0]

        def callback(algorithm):
            generation_counter[0] += 1
            if on_generation and generation_counter[0] % 10 == 0:
                on_generation(generation_counter[0], algorithm.pop.get("F"))

        return self.run(problem, callback=callback)


class AdaptiveNSGARunner(NSGARunner):
    """
    自适应 NSGA 运行器

    实现自适应遗传算子，根据进化进度动态调整参数。
    """

    def __init__(self, config: Optional[NSGAConfig] = None):
        """初始化自适应 NSGA 运行器"""
        super().__init__(config)
        self.initial_crossover_eta = self.config.crossover_eta
        self.initial_mutation_eta = self.config.mutation_eta

    def _adapt_parameters(self, generation: int, max_generations: int) -> None:
        """
        自适应调整参数

        Parameters
        ----------
        generation : int
            当前代数
        max_generations : int
            最大代数
        """
        progress = generation / max_generations

        # 前期：大变异率促进探索
        # 后期：小变异率促进收敛
        self.config.mutation_prob = 0.2 * (1 - progress) + 0.05 * progress

        # 交叉和变异分布指数自适应
        self.config.crossover_eta = self.initial_crossover_eta * (1 + progress)
        self.config.mutation_eta = self.initial_mutation_eta * (1 + 2 * progress)

    def run(self, problem: Problem, **kwargs) -> OptimizationResult:
        """运行自适应优化"""
        # 创建带自适应参数的算法
        algorithm = self.create_algorithm(problem)

        # 自定义回调
        def adaptive_callback(algorithm):
            gen = algorithm.n_gen
            self._adapt_parameters(gen, self.config.max_generations)

            # 更新算子参数
            if hasattr(algorithm, 'crossover'):
                algorithm.crossover.eta = self.config.crossover_eta
            if hasattr(algorithm, 'mutation'):
                algorithm.mutation.eta = self.config.mutation_eta

        return super().run(problem, callback=adaptive_callback)


def run_nsga_optimization(
    problem: Problem,
    variant: str = "nsga3",
    population_size: int = 100,
    max_generations: int = 500,
    seed: Optional[int] = 42,
    verbose: bool = True,
) -> OptimizationResult:
    """
    便捷函数：运行 NSGA 优化

    Parameters
    ----------
    problem : Problem
        优化问题
    variant : str
        算法变体 ("nsga2" 或 "nsga3")
    population_size : int
        种群大小
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
    config = NSGAConfig(
        population_size=population_size,
        max_generations=max_generations,
        seed=seed,
        verbose=verbose,
        variant=variant,
    )

    runner = NSGARunner(config)
    return runner.run(problem)
