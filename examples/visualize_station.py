"""
短波台站可视化示例

演示如何使用数字孪生和可视化模块。
"""

import numpy as np
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# 输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'visualizations')
os.makedirs(OUTPUT_DIR, exist_ok=True)

from sw_station.models.station import StationDigitalTwin
from sw_station.models.antenna import create_default_antenna_library
from sw_station.simulation.em_simulator import EMSimulator
from sw_station.digital_twin.twin_engine import DigitalTwinEngine
from sw_station.digital_twin.visualization.interference_cloud import InterferenceCloudRenderer


def demo_station_creation():
    """演示台站创建"""
    print("=" * 60)
    print("台站创建演示")
    print("=" * 60)

    # 创建台站
    station = StationDigitalTwin.create_default_station(n_antennas=20)

    print(f"台站配置:")
    print(f"  天线数量: {station.n_antennas}")

    # 显示天线信息
    print("\n天线列表（前5个）:")
    for i, ant in enumerate(station.antennas[:5]):
        print(f"  {ant.id}: 类型={ant.antenna_type.value}, "
              f"位置=({ant.position[0]:.0f}, {ant.position[1]:.0f}, {ant.position[2]:.0f})")

    # 获取统计信息
    stats = station.get_station_statistics()
    print(f"\n台站统计:")
    print(f"  布局中心: ({stats['layout_center'][0]:.0f}, {stats['layout_center'][1]:.0f})")
    print(f"  布局范围: {stats['layout_extent'][0]:.0f} x {stats['layout_extent'][1]:.0f} m")


def demo_em_simulation():
    """演示电磁仿真"""
    print("\n" + "=" * 60)
    print("电磁仿真演示")
    print("=" * 60)

    # 创建台站和仿真器
    station = StationDigitalTwin.create_default_station(n_antennas=10)
    simulator = EMSimulator()

    print("计算隔离度矩阵...")

    # 计算隔离度
    n = station.n_antennas
    isolations = []

    for i in range(min(5, n)):
        for j in range(i + 1, min(5, n)):
            iso = simulator.calculate_isolation(
                station.antennas[i], station.antennas[j], 15.0
            )
            isolations.append(iso)
            print(f"  {station.antennas[i].id} <-> {station.antennas[j].id}: {iso:.1f} dB")

    if isolations:
        print(f"\n隔离度统计:")
        print(f"  最小值: {min(isolations):.1f} dB")
        print(f"  最大值: {max(isolations):.1f} dB")
        print(f"  平均值: {np.mean(isolations):.1f} dB")


def demo_digital_twin():
    """演示数字孪生引擎"""
    print("\n" + "=" * 60)
    print("数字孪生引擎演示")
    print("=" * 60)

    # 创建台站
    station = StationDigitalTwin.create_default_station(n_antennas=15)

    # 设置一些天线为发射状态
    station.antennas[0].is_transmitting = True
    station.antennas[0].current_frequency = 10.0
    station.antennas[0].current_power = 30.0

    station.antennas[3].is_transmitting = True
    station.antennas[3].current_frequency = 15.0
    station.antennas[3].current_power = 25.0

    # 创建孪生引擎
    engine = DigitalTwinEngine(station)
    engine.start()

    print("运行数字孪生仿真...")

    # 运行几步
    for step in range(5):
        state = engine.update(1.0)

        if step % 2 == 0:
            print(f"\nStep {step}:")
            print(f"  活跃发射机: {state['station_stats']['n_transmitting']}")

            # 检查电磁态势
            em_state = engine.get_layer_state(
                engine._layer_states.__class__.__name__
            )

    # 获取最终状态
    final_state = engine.get_state_snapshot()
    print("\n最终状态:")
    print(f"  时间步: {final_state['timestamp']}")
    print(f"  台站统计: {final_state['station_stats']}")

    engine.stop()


def demo_propagation():
    """演示传播预测"""
    print("\n" + "=" * 60)
    print("传播预测演示")
    print("=" * 60)

    from sw_station.simulation.propagation import SkyWavePropagation
    from sw_station.models.channel import IonosphericState

    # 创建传播引擎
    engine = SkyWavePropagation()
    ionosphere = IonosphericState()

    # 测试不同距离
    distances = [500, 1000, 2000, 3000]

    print("传播预测结果:")
    print(f"{'距离(km)':<10} {'MUF(MHz)':<12} {'LUF(MHz)':<12} {'最佳频率':<12}")
    print("-" * 50)

    for dist in distances:
        muf = engine.calculate_muf(dist, ionosphere)
        luf = engine.calculate_luf(dist, ionosphere)
        fot = 0.85 * muf

        print(f"{dist:<10} {muf:<12.1f} {luf:<12.1f} {fot:<12.1f}")

    # 信道评估
    print("\n信道评估 (1000 km, 10 MHz):")
    channel = engine.evaluate_channel(10.0, 1000.0, ionosphere)

    print(f"  MUF: {channel.muf:.1f} MHz")
    print(f"  LUF: {channel.luf:.1f} MHz")
    print(f"  SNR: {channel.snr:.1f} dB")
    print(f"  可用度: {channel.availability:.2%}")
    print(f"  链路余量: {channel.link_margin:.1f} dB")


if __name__ == "__main__":
    print("短波台站多目标优化系统 - 可视化演示")
    print(f"输出目录: {OUTPUT_DIR}")
    print()

    # 台站创建演示
    demo_station_creation()

    # 电磁仿真演示
    demo_em_simulation()

    # 数字孪生演示
    demo_digital_twin()

    # 传播预测演示
    demo_propagation()

    # 生成可视化图表
    print("\n" + "=" * 60)
    print("生成可视化图表...")
    print("=" * 60)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # 创建台站
        station = StationDigitalTwin.create_default_station(n_antennas=10)

        # 1. 台站布局图
        fig, ax = plt.subplots(figsize=(10, 10))
        for ant in station.antennas:
            color = 'red' if ant.is_transmitting else 'blue'
            marker = '^' if ant.is_transmitting else 'o'
            ax.scatter(ant.position[0], ant.position[1], c=color, marker=marker, s=100)
            ax.annotate(ant.id, (ant.position[0], ant.position[1]), fontsize=8, ha='center', va='bottom')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Station Antenna Layout')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        path = os.path.join(OUTPUT_DIR, 'station_layout.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {path}")

        # 2. 隔离度矩阵热力图
        from sw_station.simulation.em_simulator import EMSimulator
        simulator = EMSimulator()
        n = station.n_antennas
        iso_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    iso_matrix[i, j] = simulator.calculate_isolation(
                        station.antennas[i], station.antennas[j], 15.0
                    )
                else:
                    iso_matrix[i, j] = 0

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(iso_matrix, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([a.id for a in station.antennas], rotation=45, fontsize=8)
        ax.set_yticklabels([a.id for a in station.antennas], fontsize=8)
        ax.set_title('Isolation Matrix (dB)')
        plt.colorbar(im, ax=ax)
        path = os.path.join(OUTPUT_DIR, 'isolation_matrix.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {path}")

        # 3. 天线极坐标方向图
        from sw_station.models.antenna import AntennaType, AntennaPatternCube
        pattern = AntennaPatternCube.create_synthetic("DEMO", AntennaType.YAGI, peak_gain=10.0)

        azimuths = np.linspace(0, 360, 361)
        gains = [pattern.get_gain(15.0, az, 15.0) for az in azimuths]
        gains = np.array(gains)
        gains_norm = gains - gains.max()

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
        theta = np.radians(azimuths)
        ax.plot(theta, gains_norm, linewidth=2)
        ax.fill(theta, gains_norm, alpha=0.3)
        ax.set_ylim(-30, 0)
        ax.set_title('Yagi Antenna Polar Pattern (15MHz, el=15°)')
        path = os.path.join(OUTPUT_DIR, 'antenna_pattern_polar.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {path}")

        print("\n可视化完成！")

    except Exception as e:
        print(f"可视化生成失败: {e}")

    print("\n" + "=" * 60)
    print("演示完成！")
