"""电磁仿真器测试"""

import numpy as np
import pytest

from sw_station.models.station import AntennaDevice, StationDigitalTwin
from sw_station.models.antenna import AntennaType, AntennaPatternCube
from sw_station.simulation.em_simulator import EMSimulator, FrequencyDependentEMSimulator


class TestEMSimulator:
    """EMSimulator 测试"""

    def setup_method(self):
        """测试前准备"""
        self.simulator = EMSimulator()

        # 创建两个测试天线
        self.antenna1 = AntennaDevice(
            id="ANT_001",
            antenna_type=AntennaType.YAGI,
            position=np.array([0, 0, 30]),
            azimuth=0.0,
            elevation=0.0,
            pattern=AntennaPatternCube.create_synthetic(
                "ANT_001", AntennaType.YAGI, peak_gain=10.0
            ),
        )

        self.antenna2 = AntennaDevice(
            id="ANT_002",
            antenna_type=AntennaType.YAGI,
            position=np.array([100, 0, 30]),
            azimuth=180.0,
            elevation=0.0,
            pattern=AntennaPatternCube.create_synthetic(
                "ANT_002", AntennaType.YAGI, peak_gain=10.0
            ),
        )

    def test_fspl_calculation(self):
        """测试自由空间路径损耗计算"""
        # 100m 距离，15 MHz
        fspl = self.simulator.calculate_fspl(15.0, 100.0)

        # 预期值：32.4 + 20*log10(15) + 20*log10(0.1) ≈ 56 dB
        expected = 32.4 + 20 * np.log10(15) + 20 * np.log10(0.1)
        assert abs(fspl - expected) < 1.0

    def test_isolation_calculation(self):
        """测试隔离度计算"""
        isolation = self.simulator.calculate_isolation(
            self.antenna1, self.antenna2, 15.0
        )

        # 隔离度应为正值
        assert isolation > 0

    def test_isolation_batch(self):
        """测试批量隔离度计算"""
        antennas = [self.antenna1, self.antenna2]
        matrix = self.simulator.calculate_isolation_batch(antennas, 15.0)

        assert matrix.shape == (2, 2)
        assert matrix[0, 0] == np.inf  # 对角线
        assert matrix[1, 1] == np.inf
        assert matrix[0, 1] > 0
        assert matrix[1, 0] > 0

    def test_near_field_boundary(self):
        """测试近场边界计算"""
        boundary = self.simulator.calculate_near_field_boundary(15.0)

        assert boundary > 0

    def test_coupling_regime(self):
        """测试耦合区域判断"""
        # 近距离
        regime = self.simulator.check_mutual_coupling_regime(
            self.antenna1, self.antenna2, 15.0
        )
        assert regime in ["near_field", "far_field"]


class TestFrequencyDependentEMSimulator:
    """频率依赖电磁仿真器测试"""

    def test_frequency_scan(self):
        """测试频率扫描"""
        simulator = FrequencyDependentEMSimulator()

        antenna1 = AntennaDevice(
            id="ANT_001",
            antenna_type=AntennaType.YAGI,
            position=np.array([0, 0, 30]),
            pattern=AntennaPatternCube.create_synthetic(
                "ANT_001", AntennaType.YAGI
            ),
        )
        antenna2 = AntennaDevice(
            id="ANT_002",
            antenna_type=AntennaType.YAGI,
            position=np.array([100, 0, 30]),
            pattern=AntennaPatternCube.create_synthetic(
                "ANT_002", AntennaType.YAGI
            ),
        )

        frequencies = np.array([5, 10, 15, 20, 25])
        isolations = simulator.calculate_isolation_vs_frequency(
            antenna1, antenna2, frequencies
        )

        assert len(isolations) == 5
        assert all(i > 0 for i in isolations)

    def test_worst_frequency(self):
        """测试最差频率查找"""
        simulator = FrequencyDependentEMSimulator()

        antenna1 = AntennaDevice(
            id="ANT_001",
            antenna_type=AntennaType.YAGI,
            position=np.array([0, 0, 30]),
            pattern=AntennaPatternCube.create_synthetic(
                "ANT_001", AntennaType.YAGI
            ),
        )
        antenna2 = AntennaDevice(
            id="ANT_002",
            antenna_type=AntennaType.YAGI,
            position=np.array([100, 0, 30]),
            pattern=AntennaPatternCube.create_synthetic(
                "ANT_002", AntennaType.YAGI
            ),
        )

        freq, isolation = simulator.find_worst_frequency(
            antenna1, antenna2
        )

        assert 2.0 <= freq <= 30.0
        assert isolation > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
