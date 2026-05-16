"""干扰模型测试"""

import numpy as np
import pytest

from sw_station.models.interference import (
    IsolationMatrix,
    InterferenceEvent,
    IntermodProduct,
)


class TestIsolationMatrix:
    """IsolationMatrix 测试"""

    def test_initialization(self):
        """测试初始化"""
        matrix = IsolationMatrix(5)

        assert matrix.n_antennas == 5
        assert matrix.isolation_matrix.shape == (5, 5)

    def test_set_get_isolation(self):
        """测试设置和获取隔离度"""
        matrix = IsolationMatrix(5)

        matrix.set_isolation(0, 1, 50.0)
        isolation = matrix.get_isolation(0, 1)

        assert isolation == 50.0

    def test_interference_calculation(self):
        """测试干扰功率计算"""
        matrix = IsolationMatrix(5)
        matrix.set_isolation(0, 1, 60.0)

        # 发射功率 30 dBm，隔离度 60 dB
        interference = matrix.calculate_interference_power(
            tx_idx=0, rx_idx=1,
            tx_power_dbm=30.0,
            filter_rejection_db=0.0,
        )

        assert interference == 30.0 - 60.0  # -30 dBm

    def test_constraint_check(self):
        """测试约束检查"""
        matrix = IsolationMatrix(5)
        matrix.set_isolation(0, 1, 100.0)

        # 发射功率 30 dBm，隔离度 100 dB → 干扰功率 = -70 dBm
        # -70 > -130，不满足约束
        result = matrix.check_interference_constraint(
            tx_idx=0, rx_idx=1,
            tx_power_dbm=30.0,
            max_allowed_interference_dbm=-130.0,
        )
        assert result is False

        # 高隔离度场景：隔离度 200 dB → 干扰功率 = 30 - 200 = -170 dBm
        matrix.set_isolation(0, 1, 200.0)
        result = matrix.check_interference_constraint(
            tx_idx=0, rx_idx=1,
            tx_power_dbm=30.0,
            max_allowed_interference_dbm=-130.0,
        )
        assert result is True

    def test_station_evaluation(self):
        """测试台站干扰评估"""
        matrix = IsolationMatrix(3)

        # 设置隔离度
        matrix.set_isolation(0, 1, 50.0)
        matrix.set_isolation(0, 2, 60.0)
        matrix.set_isolation(1, 2, 70.0)

        active_links = [
            {"tx_idx": 0, "rx_idx": 1, "tx_power": 30.0, "frequency": 15.0},
            {"tx_idx": 1, "rx_idx": 2, "tx_power": 30.0, "frequency": 15.5},
        ]

        penalty, events = matrix.evaluate_station_interference(
            active_links, -130.0
        )

        assert isinstance(penalty, float)
        assert isinstance(events, list)

    def test_worst_case_isolation(self):
        """测试最差隔离度查询"""
        matrix = IsolationMatrix(3)

        matrix.set_isolation(0, 1, 40.0)  # 最差
        matrix.set_isolation(0, 2, 80.0)
        matrix.set_isolation(1, 2, 60.0)

        tx, rx, isolation = matrix.get_worst_case_isolation()

        assert isolation == 40.0
        assert (tx == 0 and rx == 1) or (tx == 1 and rx == 0)

    def test_statistics(self):
        """测试统计信息"""
        matrix = IsolationMatrix(3)

        matrix.set_isolation(0, 1, 50.0)
        matrix.set_isolation(0, 2, 60.0)
        matrix.set_isolation(1, 2, 70.0)

        stats = matrix.get_isolation_statistics()

        assert "min" in stats
        assert "max" in stats
        assert "mean" in stats


class TestIntermodProduct:
    """IntermodProduct 测试"""

    def test_im3_frequencies(self):
        """测试三阶互调频率计算"""
        freqs = IntermodProduct.calculate_im3_freqs(10.0, 12.0)

        assert len(freqs) == 2
        assert 8.0 in freqs  # 2*10 - 12
        assert 14.0 in freqs  # 2*12 - 10

    def test_im5_frequencies(self):
        """测试五阶互调频率计算"""
        freqs = IntermodProduct.calculate_im5_freqs(10.0, 12.0)

        assert len(freqs) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
