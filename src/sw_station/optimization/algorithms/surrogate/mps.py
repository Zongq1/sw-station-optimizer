"""MPS 多问题代理模型 - 代理辅助多目标优化"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize as scipy_minimize

from ...utils import OptimizationResult, extract_pareto_front, calculate_hypervolume


@dataclass
class GaussianProcess:
    """
    简化版高斯过程回归模型

    用于代理模型预测和不确定性估计。
    """
    # 核函数参数
    length_scale: float = 1.0
    signal_variance: float = 1.0
    noise_variance: float = 0.01

    # 训练数据
    X_train: Optional[np.ndarray] = None
    y_train: Optional[np.ndarray] = None

    # 预计算矩阵
    _K_inv: Optional[np.ndarray] = field(default=None, repr=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        训练高斯过程（含超参数优化）

        Parameters
        ----------
        X : np.ndarray
            训练输入, shape: (n_samples, n_features)
        y : np.ndarray
            训练输出, shape: (n_samples,)
        """
        self.X_train = X.copy()
        self.y_train = y.copy()

        # 通过最小化负边际似然优化超参数
        if len(X) >= 3:
            self._optimize_hyperparameters(X, y)

        # 计算核矩阵
        K = self._kernel(X, X)
        K += self.noise_variance * np.eye(len(X))

        # 求逆（添加正则化）
        try:
            self._K_inv = np.linalg.inv(K + 1e-6 * np.eye(len(X)))
        except np.linalg.LinAlgError:
            self._K_inv = np.linalg.pinv(K)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        预测均值和方差

        Parameters
        ----------
        X : np.ndarray
            预测输入, shape: (n_test, n_features)

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (均值, 方差)
        """
        if self.X_train is None or self._K_inv is None:
            # 未训练，返回先验
            return np.zeros(len(X)), np.ones(len(X))

        # 计算核向量
        K_star = self._kernel(X, self.X_train)

        # 预测均值
        y_mean = K_star @ self._K_inv @ self.y_train

        # 预测方差
        K_star_star = self._kernel(X, X)
        y_var = np.diag(K_star_star - K_star @ self._K_inv @ K_star.T)
        y_var = np.maximum(y_var, 1e-10)

        return y_mean, y_var

    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        计算核矩阵（RBF 核）

        Parameters
        ----------
        X1, X2 : np.ndarray
            输入矩阵

        Returns
        -------
        np.ndarray
            核矩阵
        """
        # 计算欧氏距离
        dists = np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2, axis=2)
        return self.signal_variance * np.exp(-0.5 * dists / self.length_scale**2)

    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> None:
        """通过最大化边际似然优化超参数"""
        from scipy.optimize import minimize as scipy_minimize_opt

        def neg_log_likelihood(params):
            ls, sv, nv = np.exp(params)  # 对数空间优化保证正值
            old_ls, old_sv, old_nv = self.length_scale, self.signal_variance, self.noise_variance
            self.length_scale, self.signal_variance, self.noise_variance = ls, sv, nv

            K = self._kernel(X, X)
            K += nv * np.eye(len(X))

            try:
                L = np.linalg.cholesky(K + 1e-6 * np.eye(len(X)))
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
                log_det = 2 * np.sum(np.log(np.diag(L)))
                nll = 0.5 * y @ alpha + 0.5 * log_det
            except np.linalg.LinAlgError:
                nll = 1e10

            self.length_scale, self.signal_variance, self.noise_variance = old_ls, old_sv, old_nv
            return nll

        x0 = np.log([self.length_scale, self.signal_variance, self.noise_variance])
        try:
            result = scipy_minimize_opt(neg_log_likelihood, x0, method='Nelder-Mead')
            if result.success:
                ls, sv, nv = np.exp(result.x)
                self.length_scale = np.clip(ls, 0.01, 100.0)
                self.signal_variance = np.clip(sv, 0.01, 100.0)
                self.noise_variance = np.clip(nv, 1e-6, 1.0)
        except Exception:
            pass


@dataclass
class SourceModel:
    """源代理模型"""
    model_id: str
    problem_signature: np.ndarray  # 问题特征签名
    gp_model: GaussianProcess
    weight: float = 1.0


class MPSOptimizer:
    """
    MPS 多问题代理优化器

    通过知识迁移框架，利用历史代理模型加速当前问题的优化。
    """

    def __init__(
        self,
        n_objectives: int = 3,
        max_evaluations: int = 1000,
        initial_sample_size: int = 50,
        batch_size: int = 10,
    ):
        """
        初始化 MPS 优化器

        Parameters
        ----------
        n_objectives : int
            目标数量
        max_evaluations : int
            最大真实评估次数
        initial_sample_size : int
            初始采样大小
        batch_size : int
            每批推荐的候选解数量
        """
        self.n_objectives = n_objectives
        self.max_evaluations = max_evaluations
        self.initial_sample_size = initial_sample_size
        self.batch_size = batch_size

        # 源模型库
        self.source_models: list[SourceModel] = []

        # 目标问题的代理模型（每个目标一个）
        self.target_models: list[GaussianProcess] = [
            GaussianProcess() for _ in range(n_objectives)
        ]

        # 元回归模型权重
        self.meta_weights: Optional[np.ndarray] = None

        # 评估历史
        self.X_history: list[np.ndarray] = []
        self.y_history: list[np.ndarray] = []

    def add_source_model(
        self,
        model_id: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        problem_signature: np.ndarray,
    ) -> None:
        """
        添加源代理模型

        Parameters
        ----------
        model_id : str
            模型ID
        X_train : np.ndarray
            训练输入
        y_train : np.ndarray
            训练输出（单目标）
        problem_signature : np.ndarray
            问题特征签名（用于相似度计算）
        """
        gp = GaussianProcess()
        gp.fit(X_train, y_train)

        source = SourceModel(
            model_id=model_id,
            problem_signature=problem_signature,
            gp_model=gp,
        )
        self.source_models.append(source)

    def _calculate_source_weights(
        self,
        target_signature: np.ndarray,
    ) -> np.ndarray:
        """
        计算源模型的自适应权重

        基于目标问题与源问题的相似度。

        Parameters
        ----------
        target_signature : np.ndarray
            目标问题特征签名

        Returns
        -------
        np.ndarray
            权重数组
        """
        if len(self.source_models) == 0:
            return np.array([])

        weights = np.zeros(len(self.source_models))

        for i, source in enumerate(self.source_models):
            # 基于特征距离的相似度
            dist = np.linalg.norm(
                target_signature - source.problem_signature
            )
            # 高斯权重
            weights[i] = np.exp(-0.5 * dist**2)

        # 归一化
        total = weights.sum()
        if total > 0:
            weights /= total
        else:
            weights = np.ones(len(self.source_models)) / len(self.source_models)

        return weights

    def _meta_regression_predict(
        self,
        X: np.ndarray,
        target_idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        元回归预测

        堆叠源模型和目标模型的预测。

        Parameters
        ----------
        X : np.ndarray
            预测输入
        target_idx : int
            目标索引

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (均值, 方差)
        """
        if len(self.source_models) == 0:
            # 无源模型，直接使用目标模型
            return self.target_models[target_idx].predict(X)

        # 获取源模型预测
        source_means = []
        source_vars = []

        for source in self.source_models:
            mean, var = source.gp_model.predict(X)
            source_means.append(mean)
            source_vars.append(var)

        source_means = np.array(source_means)
        source_vars = np.array(source_vars)

        # 获取目标模型预测
        target_mean, target_var = self.target_models[target_idx].predict(X)

        # 使用目标问题的变量统计作为签名
        if len(self.X_history) > 0:
            X_all = np.array(self.X_history)
            # 签名 = [变量均值, 变量标准差, 变量范围]
            target_signature = np.concatenate([
                X_all.mean(axis=0)[:5],  # 前5维均值
                X_all.std(axis=0)[:5],   # 前5维标准差
            ])
        else:
            target_signature = np.zeros(10)

        weights = self._calculate_source_weights(target_signature)

        # 堆叠预测
        stacked_mean = np.zeros_like(target_mean)
        stacked_var = np.zeros_like(target_var)

        for i, w in enumerate(weights):
            stacked_mean += w * source_means[i]
            stacked_var += w**2 * source_vars[i]

        # 目标模型权重 - 基于交叉验证误差
        if len(self.X_history) >= 5:
            # 简化留一法：用最后 20% 数据估计目标模型误差
            n_val = max(1, len(self.X_history) // 5)
            X_val = np.array(self.X_history[-n_val:])
            y_val = np.array(self.y_history[-n_val:])
            target_pred, _ = self.target_models[target_idx].predict(X_val)
            target_mse = np.mean((target_pred - y_val[:, target_idx])**2)

            # 源模型误差
            source_pred_mse = 0.0
            for i, source in enumerate(self.source_models):
                pred, _ = source.gp_model.predict(X_val)
                source_pred_mse += weights[i] * np.mean((pred - y_val[:, target_idx])**2)

            # 自适应权重：误差小的模型权重高
            total_error = target_mse + source_pred_mse + 1e-10
            alpha = np.clip(source_pred_mse / total_error, 0.1, 0.9)
        else:
            alpha = 0.5
        final_mean = alpha * target_mean + (1 - alpha) * stacked_mean
        final_var = alpha * target_var + (1 - alpha) * stacked_var

        return final_mean, final_var

    def expected_improvement(
        self,
        X: np.ndarray,
        best_value: float,
        target_idx: int,
    ) -> np.ndarray:
        """
        计算期望增量 (EI)

        Parameters
        ----------
        X : np.ndarray
            候选点
        best_value : float
            当前最优值
        target_idx : int
            目标索引

        Returns
        -------
        np.ndarray
            EI 值
        """
        mean, var = self._meta_regression_predict(X, target_idx)
        std = np.sqrt(var)

        # EI 公式
        z = (best_value - mean) / std
        ei = (best_value - mean) * norm.cdf(z) + std * norm.pdf(z)

        return ei

    def select_next_points(
        self,
        bounds: list[tuple[float, float]],
        n_points: int = 10,
    ) -> np.ndarray:
        """
        选择下一批评估点

        Parameters
        ----------
        bounds : list[tuple]
            变量边界
        n_points : int
            选择点数

        Returns
        -------
        np.ndarray
            推荐点, shape: (n_points, n_vars)
        """
        n_vars = len(bounds)
        best_points = []

        for obj_idx in range(self.n_objectives):
            # 当前最优值
            if len(self.y_history) > 0:
                y_array = np.array(self.y_history)
                best_val = y_array[:, obj_idx].min()
            else:
                best_val = 0.0

            # 多起点 L-BFGS-B 优化找 EI 最大的点
            lb = np.array([b[0] for b in bounds])
            ub = np.array([b[1] for b in bounds])

            # 随机起点
            n_starts = min(50, 10 * n_vars)
            start_points = np.random.uniform(lb, ub, size=(n_starts, n_vars))

            ei_optimal_points = []
            for x0 in start_points:
                try:
                    result = scipy_minimize(
                        lambda x: -self.expected_improvement(
                            x.reshape(1, -1), best_val, obj_idx
                        )[0],
                        x0,
                        method='L-BFGS-B',
                        bounds=list(zip(lb, ub)),
                    )
                    if result.success:
                        ei_optimal_points.append(result.x)
                except Exception:
                    pass

            if len(ei_optimal_points) < n_points:
                # 补充随机候选
                extra = np.random.uniform(lb, ub, size=(n_points - len(ei_optimal_points), n_vars))
                ei_optimal_points.extend(extra)

            best_points.append(np.array(ei_optimal_points[:n_points]))

        # 合并所有目标的推荐点
        all_points = np.vstack(best_points)

        # 去重
        unique_points = np.unique(all_points, axis=0)

        return unique_points[:n_points]

    def update(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray,
    ) -> None:
        """
        更新代理模型

        Parameters
        ----------
        X_new : np.ndarray
            新增输入
        y_new : np.ndarray
            新增输出
        """
        if isinstance(X_new, np.ndarray):
            for x in X_new:
                self.X_history.append(x)
        else:
            self.X_history.extend(X_new)
        if isinstance(y_new, np.ndarray):
            for y in y_new:
                self.y_history.append(y)
        else:
            self.y_history.extend(y_new)

        # 重新训练目标模型
        X_all = np.array(self.X_history)
        y_all = np.array(self.y_history)

        for obj_idx in range(self.n_objectives):
            self.target_models[obj_idx].fit(X_all, y_all[:, obj_idx])

    def optimize(
        self,
        objective_func: Callable,
        bounds: list[tuple[float, float]],
        n_iterations: int = 100,
    ) -> OptimizationResult:
        """
        运行 MPS 优化

        Parameters
        ----------
        objective_func : Callable
            目标函数
        bounds : list[tuple]
            变量边界
        n_iterations : int
            迭代次数

        Returns
        -------
        OptimizationResult
            优化结果
        """
        n_vars = len(bounds)

        # 初始采样
        X_init = np.random.uniform(
            [b[0] for b in bounds],
            [b[1] for b in bounds],
            size=(self.initial_sample_size, n_vars),
        )

        y_init = np.array([objective_func(x) for x in X_init])
        self.update(X_init, y_init)

        # 迭代优化
        for iteration in range(n_iterations):
            # 选择下一批点
            X_next = self.select_next_points(bounds, self.batch_size)

            # 真实评估
            y_next = np.array([objective_func(x) for x in X_next])

            # 更新模型
            self.update(X_next, y_next)

            if (iteration + 1) % 10 == 0:
                print(f"MPS Iteration {iteration + 1}/{n_iterations}")

        # 提取帕累托前沿
        X_all = np.array(self.X_history)
        y_all = np.array(self.y_history)

        pareto_front, pareto_solutions = extract_pareto_front(y_all, X_all)
        hv = calculate_hypervolume(pareto_front)

        return OptimizationResult(
            pareto_front=pareto_front,
            pareto_solutions=pareto_solutions,
            all_objectives=y_all,
            all_solutions=X_all,
            n_generations=n_iterations,
            hypervolume=hv,
            execution_time=0.0,
        )
