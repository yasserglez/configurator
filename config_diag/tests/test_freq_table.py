import os

import numpy as np

from ..freq_table import FrequencyTable


class TestFrequencyTable(object):

    def setup(self):
        tests_dir = os.path.abspath(os.path.dirname(__file__))
        self.data = np.genfromtxt(os.path.join(tests_dir, "titanic.csv"),
                                  skip_header=1, dtype=np.dtype(str),
                                  delimiter=",")

    def test_init(self):
        freq_table = FrequencyTable(self.data)
        np.testing.assert_array_equal(freq_table.data, self.data)

    def test_count_freq(self):
        freq_table = FrequencyTable(self.data)
        x = {0: "1st", 1: "Male", 2: "Child", 3: "No"}
        assert freq_table.count_freq(x) == 0
        x = {0: "1st", 1: "Male", 2: "Adult", 3: "No"}
        assert freq_table.count_freq(x) == 118

    def test_count_freq_cached(self):
        freq_table = FrequencyTable(self.data, cache_size=10)
        x = {0: "1st", 1: "Male", 2: "Child", 3: "No"}
        first_time = freq_table.count_freq(x)
        second_time = freq_table.count_freq(x)
        assert first_time == second_time
        x = {0: "1st", 1: "Male", 2: "Adult", 3: "No"}
        first_time = freq_table.count_freq(x)
        second_time = freq_table.count_freq(x)
        assert first_time == second_time
