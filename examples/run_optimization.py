"""
短波台站多目标优化示例

演示如何使用 NSGA-III 和 MOEA/D 算法进行天线布局优化。
"""

import numpy as np
import time
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sw_station.config import SystemConfig
from sw_station.models.station import StationDigitalTwin
from sw_station.optimization.problem import ShortwaveStationProblem, SimplifiedStationProblem
from sw_station.optimization.algorithms.nsga import NSGARunner, NSGAConfig
from sw_station.optimization.algorithms.moead import MOEADRunner, MOEADConfig


def run_simple_optimization():
    """运行简化版优化（快速演示）"""
    print("=" * 60)
    print("简化版短波台站优化演示")
    print("=" * 60)

    # 创建简化问题
    n_antennas = 10
    problem = SimplifiedStationProblem(n_antennas=n_antennas)

    print(f"问题规模: {n_antennas} 副天线")
    print(f"变量维度: {problem.n_var}")
    print(f"目标数量: {problem.n_obj}")
    print()

    # 配置 NSGA-II
    config = NSGAConfig(
        population_size=50,
        max_generations=100,
        variant="nsga2",
        seed=42,
        verbose=True,
    )

    # 运行优化
    print("运行 NSGA-II 优化...")
    start_time = time.time()

    runner = NSGARunner(config)
    result = runner.run(problem)

    elapsed = time.time() - start_time

    print()
    print("优化完成！")
    print(f"耗时: {elapsed:.2f} 秒")
    print(f"帕累托前沿解数量: {result.n_pareto}")
    print(f"超体积指标: {result.hypervolume:.4f}")
    print()

    # 显示部分帕累托解
    print("帕累托前沿解（前5个）:")
    for i in range(min(5, result.n_pareto)):
        obj = result.pareto_front[i]
        print(f"  解 {i+1}: 覆盖={-obj[0]:.4f}, 干扰={obj[1]:.4f}")


def run_full_optimization():
    """运行完整优化（较慢）"""
    print("=" * 60)
    print("完整短波台站优化演示")
    print("=" * 60)

    # 创建台站
    config = SystemConfig.default()
    config.station.n_antennas = 20  # 减少天线数量以加快演示

    station = StationDigitalTwin.create_default_station(
        n_antennas=config.station.n_antennas
    )

    # 创建优化问题
    problem = ShortwaveStationProblem(
        station=station,
        config=config,
        n_antennas=config.station.n_antennas,
    )

    print(f"问题规模: {config.station.n_antennas} 副天线")
    print(f"变量维度: {problem.n_var}")
    print(f"目标数量: {problem.n_obj}")
    print(f"约束数量: {problem.n_constr}")
    print()

    # 配置 NSGA-III
    config_nsga = NSGAConfig(
        population_size=100,
        max_generations=200,
        variant="nsga3",
        seed=42,
        verbose=True,
    )

    # 运行优化
    print("运行 NSGA-III 优化...")
    start_time = time.time()

    runner = NSGARunner(config_nsga)
    result = runner.run(problem)

    elapsed = time.time() - start_time

    print()
    print("优化完成！")
    print(f"耗时: {elapsed:.2f} 秒")
    print(f"帕累托前沿解数量: {result.n_pareto}")
    print(f"超体积指标: {result.hypervolume:.4f}")
    print()

    # 显示最优解
    print("各目标最优解:")
    for i in range(problem.n_obj):
        best_sol, best_obj = result.get_best_solution(i)
        print(f"  目标 {i+1}: {best_obj:.4f}")

    # 显示膝点解
    knee_sol, knee_obj = result.get_knee_point()
    print(f"\n膝点解: {knee_obj}")


def run_moead_optimization():
    """运行 MOEA/D 优化"""
    print("=" * 60)
    print("MOEA/D 优化演示")
    print("=" * 60)

    # 创建简化问题
    problem = SimplifiedStationProblem(n_antennas=10)

    # 配置 MOEA/D
    config = MOEADConfig(
        n_partitions=12,
        max_generations=100,
        seed=42,
        verbose=True,
    )

    # 运行优化
    print("运行 MOEA/D 优化...")
    start_time = time.time()

    runner = MOEADRunner(config)
    result = runner.run(problem)

    elapsed = time.time() - start_time

    print()
    print("优化完成！")
    print(f"耗时: {elapsed:.2f} 秒")
    print(f"帕累托前沿解数量: {result.n_pareto}")
    print(f"超体积指标: {result.hypervolume:.4f}")


if __name__ == "__main__":
    print("短波台站多目标优化系统 - 优化演示")
    print()

    # 运行简化版优化
    run_simple_optimization()

    print("\n" + "=" * 60 + "\n")

    # 运行 MOEA/D 优化
    run_moead_optimization()

    print("\n" + "=" * 60 + "\n")

    # 运行完整优化（可选，较慢）
    # run_full_optimization()
