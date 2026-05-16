"""天线模型测试"""

import numpy as np
import pytest

from sw_station.models.antenna import (
    AntennaPatternCube,
    AntennaType,
    create_default_antenna_library,
)


class TestAntennaPatternCube:
    """AntennaPatternCube 测试"""

    def test_create_synthetic(self):
        """测试创建合成天线"""
        antenna = AntennaPatternCube.create_synthetic(
            "TEST_01", AntennaType.YAGI,
            peak_gain=10.0, beamwidth_az=60.0, beamwidth_el=30.0,
        )

        assert antenna.antenna_id == "TEST_01"
        assert antenna.antenna_type == AntennaType.YAGI
        assert antenna.gain_pattern.shape == (
            len(antenna.frequencies),
            len(antenna.azimuths),
            len(antenna.elevations),
        )

    def test_get_gain(self):
        """测试增益查询"""
        antenna = AntennaPatternCube.create_synthetic(
            "TEST_01", AntennaType.YAGI, peak_gain=10.0
        )

        # 查询峰值方向
        gain = antenna.get_gain(15.0, 0.0, 15.0)
        assert gain > 0

        # 查询不同方向
        gain_side = antenna.get_gain(15.0, 90.0, 15.0)
        assert gain_side < gain  # 侧向增益应较低

    def test_get_peak_gain(self):
        """测试峰值增益查询"""
        antenna = AntennaPatternCube.create_synthetic(
            "TEST_01", AntennaType.YAGI, peak_gain=10.0
        )

        peak_gain, peak_az, peak_el = antenna.get_peak_gain(15.0)
        assert peak_gain > 0
        assert 0 <= peak_az < 360
        assert 0 <= peak_el <= 90

    def test_get_front_to_back_ratio(self):
        """测试前后比计算"""
        antenna = AntennaPatternCube.create_synthetic(
            "TEST_01", AntennaType.YAGI, peak_gain=10.0
        )

        fb_ratio = antenna.get_front_to_back_ratio(15.0, 0.0, 15.0)
        assert fb_ratio > 0  # 前向增益应高于后向

    def test_batch_gain(self):
        """测试批量增益查询"""
        antenna = AntennaPatternCube.create_synthetic(
            "TEST_01", AntennaType.YAGI
        )

        az_array = np.array([0, 90, 180, 270])
        el_array = np.array([15, 15, 15, 15])

        gains = antenna.get_gain_batch(15.0, az_array, el_array)
        assert len(gains) == 4

    def test_default_library(self):
        """测试默认天线库"""
        library = create_default_antenna_library()

        assert len(library) > 0
        assert "LP_01" in library
        assert "YAGI_01" in library
        assert "CAGE_01" in library


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
