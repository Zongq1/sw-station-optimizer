"""K-RVEA 算法 - 基于 Kriging 模型的参考向量引导进化算法"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np

from .mps import GaussianProcess
from ...utils import OptimizationResult, extract_pareto_front, calculate_hypervolume


@dataclass
class KRVEAConfig:
    """K-RVEA 配置"""
    # 种群大小
    population_size: int = 100
    # 最大迭代代数
    max_generations: int = 500
    # 最大真实评估次数
    max_evaluations: int = 1000
    # 初始采样大小
    initial_sample_size: int = 50
    # 参考向量更新频率
    ref_vector_update_freq: int = 50
    # 每批评估数量
    batch_size: int = 10
    # 随机种子
    seed: Optional[int] = None


class ReferenceVectorAdaptor:
    """
    自适应参考向量生成器

    根据种群分布动态调整参考向量方向。
    """

    def __init__(self, n_objectives: int, n_vectors: int):
        """
        初始化参考向量适配器

        Parameters
        ----------
        n_objectives : int
            目标数量
        n_vectors : int
            参考向量数量
        """
        self.n_objectives = n_objectives
        self.n_vectors = n_vectors
        self.reference_vectors = self._generate_uniform_vectors()

    def _generate_uniform_vectors(self) -> np.ndarray:
        """生成均匀分布的参考向量"""
        if self.n_objectives == 2:
            angles = np.linspace(0, np.pi / 2, self.n_vectors)
            vectors = np.column_stack([np.cos(angles), np.sin(angles)])
        elif self.n_objectives == 3:
            # 3D 简化：使用球面均匀分布
            vectors = []
            for i in range(self.n_vectors):
                phi = np.arccos(1 - 2 * (i + 0.5) / self.n_vectors)
                theta = np.pi * (1 + 5**0.5) * i
                x = np.sin(phi) * np.cos(theta)
                y = np.sin(phi) * np.sin(theta)
                z = np.cos(phi)
                vectors.append([abs(x), abs(y), abs(z)])
            vectors = np.array(vectors)
            vectors = vectors / vectors.sum(axis=1, keepdims=True)
        else:
            # 高维：使用 Das-Dennis 方法生成系统化参考向量
            from pymoo.util.ref_dirs import get_reference_directions
            try:
                n_partitions = max(1, int(self.n_vectors ** (1.0 / self.n_objectives)))
                vectors = get_reference_directions(
                    "das-dennis", self.n_objectives, n_partitions=n_partitions
                )
                # 如果数量不匹配，截断或补充
                if len(vectors) > self.n_vectors:
                    vectors = vectors[:self.n_vectors]
                elif len(vectors) < self.n_vectors:
                    extra = self.rng.dirichlet(
                        np.ones(self.n_objectives), self.n_vectors - len(vectors)
                    )
                    vectors = np.vstack([vectors, extra])
            except Exception:
                vectors = self.rng.dirichlet(np.ones(self.n_objectives), self.n_vectors)

        return vectors

    def adapt(self, population_objectives: np.ndarray) -> None:
        """
        根据种群分布自适应调整参考向量

        Parameters
        ----------
        population_objectives : np.ndarray
            种群目标值, shape: (pop_size, n_objectives)
        """
        # 归一化目标值
        obj_min = population_objectives.min(axis=0)
        obj_max = population_objectives.max(axis=0)
        obj_range = obj_max - obj_min
        obj_range[obj_range < 1e-10] = 1.0

        normalized = (population_objectives - obj_min) / obj_range

        # 计算种群中心方向
        center = normalized.mean(axis=0)
        center = center / center.sum()  # 归一化

        # 计算种群散布方向（PCA 主方向）
        if len(normalized) > self.n_objectives:
            cov = np.cov(normalized.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # 调整参考向量向主方向偏移
            principal_dir = eigenvectors[:, -1]
            principal_dir = np.abs(principal_dir)
            principal_dir = principal_dir / principal_dir.sum()

            # 混合均匀分布和主方向
            alpha = 0.3  # 调整强度
            for i in range(self.n_vectors):
                self.reference_vectors[i] = (
                    (1 - alpha) * self.reference_vectors[i]
                    + alpha * principal_dir
                )
                # 归一化
                self.reference_vectors[i] = np.abs(self.reference_vectors[i])
                self.reference_vectors[i] /= self.reference_vectors[i].sum()


class KRVEAOptimizer:
    """
    K-RVEA 优化器

    基于 Kriging 模型和自适应参考向量引导的进化算法。
    """

    def __init__(self, config: Optional[KRVEAConfig] = None):
        """
        初始化 K-RVEA 优化器

        Parameters
        ----------
        config : KRVEAConfig, optional
            算法配置
        """
        self.config = config or KRVEAConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.gp_models: list[GaussianProcess] = []
        self.ref_vector_adaptor: Optional[ReferenceVectorAdaptor] = None
        self.X_history: list[np.ndarray] = []
        self.y_history: list[np.ndarray] = []

    def _initialize_models(self, n_objectives: int, n_vars: int) -> None:
        """初始化代理模型"""
        self.gp_models = [
            GaussianProcess(
                length_scale=1.0,
                signal_variance=1.0,
                noise_variance=0.01,
            )
            for _ in range(n_objectives)
        ]

        self.ref_vector_adaptor = ReferenceVectorAdaptor(
            n_objectives, self.config.population_size
        )

    def _update_models(self) -> None:
        """更新代理模型"""
        if len(self.X_history) == 0:
            return

        X = np.array(self.X_history)
        y = np.array(self.y_history)

        for i, gp in enumerate(self.gp_models):
            gp.fit(X, y[:, i])

    def _select_by_reference_vectors(
        self,
        candidates: np.ndarray,
        n_select: int,
    ) -> np.ndarray:
        """
        基于参考向量选择候选解

        Parameters
        ----------
        candidates : np.ndarray
            候选解目标值
        n_select : int
            选择数量

        Returns
        -------
        np.ndarray
            选中解的索引
        """
        n_candidates = len(candidates)
        ref_vectors = self.ref_vector_adaptor.reference_vectors

        # 归一化候选解
        obj_min = candidates.min(axis=0)
        obj_max = candidates.max(axis=0)
        obj_range = obj_max - obj_min
        obj_range[obj_range < 1e-10] = 1.0
        normalized = (candidates - obj_min) / obj_range

        # 计算每个候选解到每个参考向量的角度
        angles = np.zeros((n_candidates, len(ref_vectors)))
        for i in range(n_candidates):
            for j, ref in enumerate(ref_vectors):
                cos_angle = np.dot(normalized[i], ref) / (
                    np.linalg.norm(normalized[i]) * np.linalg.norm(ref) + 1e-10
                )
                angles[i, j] = np.arccos(np.clip(cos_angle, -1, 1))

        # 每个参考向量分配最近的候选解
        selected = set()
        for j in range(len(ref_vectors)):
            if len(selected) >= n_select:
                break

            # 找到距离该参考向量最近的未选中候选解
            sorted_indices = np.argsort(angles[:, j])
            for idx in sorted_indices:
                if idx not in selected:
                    selected.add(idx)
                    break

        # 如果还不够，补充距离参考向量最远的候选解
        while len(selected) < n_select:
            remaining = [i for i in range(n_candidates) if i not in selected]
            if not remaining:
                break
            # 选择与已选解平均角度最大的候选
            if len(selected) > 0:
                avg_angles = angles[list(selected)].mean(axis=0)
                best_remaining = remaining[np.argmax(avg_angles[remaining])]
                selected.add(best_remaining)
            else:
                selected.add(remaining[0])

        return np.array(list(selected))[:n_select]

    def optimize(
        self,
        objective_func: Callable,
        bounds: list[tuple[float, float]],
        n_objectives: int = 3,
        n_iterations: int = 100,
    ) -> OptimizationResult:
        """
        运行 K-RVEA 优化

        Parameters
        ----------
        objective_func : Callable
            目标函数
        bounds : list[tuple]
            变量边界
        n_objectives : int
            目标数量
        n_iterations : int
            迭代次数

        Returns
        -------
        OptimizationResult
            优化结果
        """
        n_vars = len(bounds)
        start_time = time.perf_counter()

        # 初始化
        self._initialize_models(n_objectives, n_vars)

        # 初始采样
        X_init = self.rng.uniform(
            [b[0] for b in bounds],
            [b[1] for b in bounds],
            size=(self.config.initial_sample_size, n_vars),
        )

        y_init = np.array([objective_func(x) for x in X_init])
        self.X_history.extend(X_init)
        self.y_history.extend(y_init)

        # 更新模型
        self._update_models()

        # 迭代优化
        for iteration in range(n_iterations):
            # 生成候选解（使用模型预测引导）
            candidates = self._generate_candidates(bounds, 200)

            # 预测目标值
            predicted_objectives = self._predict_objectives(candidates)

            # 自适应调整参考向量
            if iteration % self.config.ref_vector_update_freq == 0:
                self.ref_vector_adaptor.adapt(predicted_objectives)

            # 基于参考向量选择候选解
            selected_idx = self._select_by_reference_vectors(
                predicted_objectives, self.config.batch_size
            )

            X_selected = candidates[selected_idx]

            # 真实评估
            y_selected = np.array([objective_func(x) for x in X_selected])

            # 更新历史
            self.X_history.extend(X_selected)
            self.y_history.extend(y_selected)

            # 更新模型
            self._update_models()

            if (iteration + 1) % 10 == 0:
                print(f"K-RVEA Iteration {iteration + 1}/{n_iterations}")

        # 提取帕累托前沿
        X_all = np.array(self.X_history)
        y_all = np.array(self.y_history)

        pareto_front, pareto_solutions = extract_pareto_front(y_all, X_all)
        hv = calculate_hypervolume(pareto_front)

        elapsed = time.perf_counter() - start_time

        return OptimizationResult(
            pareto_front=pareto_front,
            pareto_solutions=pareto_solutions,
            all_objectives=y_all,
            all_solutions=X_all,
            n_generations=n_iterations,
            hypervolume=hv,
            execution_time=elapsed,
        )

    def _generate_candidates(
        self,
        bounds: list[tuple[float, float]],
        n_candidates: int,
    ) -> np.ndarray:
        """生成候选解"""
        n_vars = len(bounds)

        # 混合策略：50% 随机 + 50% 基于历史解的扰动
        n_random = n_candidates // 2
        n_perturbation = n_candidates - n_random

        # 随机候选
        random_candidates = self.rng.uniform(
            [b[0] for b in bounds],
            [b[1] for b in bounds],
            size=(n_random, n_vars),
        )

        # 扰动候选 - GP 引导的候选解生成
        if len(self.X_history) > 0:
            X_array = np.array(self.X_history)
            scale = np.array([b[1] - b[0] for b in bounds])

            # GP 引导的候选解生成
            # 选择不确定性大的区域（探索）和预测值好的区域（开发）
            y_array = np.array(self.y_history)

            # 选择 Pareto 前沿解进行扰动（开发）
            from ...utils import is_pareto_efficient
            pareto_mask = is_pareto_efficient(y_array)
            pareto_indices = np.where(pareto_mask)[0]

            if len(pareto_indices) > 0:
                n_exploit = n_perturbation // 2
                n_explore = n_perturbation - n_exploit

                # 开发：在 Pareto 解附近扰动
                exploit_indices = self.rng.choice(pareto_indices, n_exploit, replace=True)
                exploit_perturbation = self.rng.normal(0, 0.05, (n_exploit, n_vars))
                exploit_candidates = X_array[exploit_indices] + exploit_perturbation * scale

                # 探索：在不确定性大的区域采样
                explore_candidates = self.rng.uniform(
                    [b[0] for b in bounds],
                    [b[1] for b in bounds],
                    size=(n_explore, n_vars),
                )

                perturbed_candidates = np.vstack([exploit_candidates, explore_candidates])
            else:
                # 没有 Pareto 解，全部随机
                perturbed_candidates = self.rng.uniform(
                    [b[0] for b in bounds],
                    [b[1] for b in bounds],
                    size=(n_perturbation, n_vars),
                )

            # 裁剪到边界
            for i, (lb, ub) in enumerate(bounds):
                perturbed_candidates[:, i] = np.clip(
                    perturbed_candidates[:, i], lb, ub
                )
        else:
            perturbed_candidates = random_candidates[:n_perturbation]

        return np.vstack([random_candidates, perturbed_candidates])

    def _predict_objectives(self, X: np.ndarray) -> np.ndarray:
        """预测目标值"""
        predictions = []
        for gp in self.gp_models:
            mean, _ = gp.predict(X)
            predictions.append(mean)
        return np.column_stack(predictions)
