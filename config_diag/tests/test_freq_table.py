from numpy.testing import assert_array_equal

from .examples import load_titanic
from ..freq_table import FrequencyTable


class TestFrequencyTable(object):

    def setup(self):
        self.data = load_titanic()

    def test_init(self):
        freq_table = FrequencyTable(self.data)
        assert_array_equal(freq_table.data, self.data)

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
