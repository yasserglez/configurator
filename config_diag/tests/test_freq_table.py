import numpy as np

from ..freq_table import FrequencyTable


class TestFrequencyTable(object):

    def setup(self):
        np.random.seed(42)
        self.data = np.random.randint(1, 6, 40).reshape(10, 4)

    def test_init(self):
        freq_table = FrequencyTable(self.data)
        assert (freq_table.data == self.data).all()

    def test_freq_count(self):
        freq_table = FrequencyTable(self.data)
        assert freq_table.freq_count({2: 5}) == 0
        assert freq_table.freq_count({0: 4}) == 4
        assert freq_table.freq_count({0: 4, 1: 5}) == 2
        assert freq_table.freq_count({0: 4, 2: 3, 3: 5}) == 2
        assert freq_table.freq_count({0: 4, 1: 5, 2: 3, 3: 5}) == 1

    def test_cached_freq_count(self):
        freq_table = FrequencyTable(self.data, cache_size=10)
        first_time = freq_table.freq_count({0: 4, 1: 5})
        second_time = freq_table.freq_count({0: 4, 1: 5})
        assert first_time == second_time

    def test_joint_prob(self):
        freq_table = FrequencyTable(self.data)
        assert freq_table.joint_prob({2: 5}) == 0.0
        assert freq_table.joint_prob({0: 4}) == 0.4
        assert freq_table.joint_prob({0: 4, 1: 5}) == 0.2
        assert freq_table.joint_prob({0: 4, 2: 3, 3: 5}) == 0.2
        assert freq_table.joint_prob({0: 4, 1: 5, 2: 3, 3: 5}) == 0.1

    def test_cond_prob(self):
        freq_table = FrequencyTable(self.data)
        assert freq_table.cond_prob({0: 3}, {1: 5}) == 0.4
        assert freq_table.cond_prob({0: 3, 2: 1}, {1: 5}) == 0.2
        assert freq_table.cond_prob({0: 1}, {1: 5, 2: 4}) == 0.0
        assert freq_table.cond_prob({0: 3, 1: 5}, {2: 1, 3: 2}) == 1.0
