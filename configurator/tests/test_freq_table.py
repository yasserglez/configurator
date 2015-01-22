from numpy.testing import assert_array_equal, assert_almost_equal

from .examples import load_titanic
from ..freq_table import FrequencyTable


class TestFrequencyTable(object):

    var_sample = load_titanic()

    def test_init(self):
        freq_table = FrequencyTable(self.var_sample)
        assert_array_equal(freq_table.var_sample, self.var_sample)
        var_values = [["1st", "2nd", "3rd", "Crew"],
                      ["Male", "Female"],
                      ["Child", "Adult"],
                      ["Yes", "No"]]
        assert len(freq_table.var_values) == len(var_values)
        for i in range(len(var_values)):
            assert set(freq_table.var_values[i]) == set(var_values[i])

    def test_count_freq(self):
        freq_table = FrequencyTable(self.var_sample)
        x = {0: "1st", 1: "Male", 2: "Child", 3: "No"}
        assert freq_table.count_freq(x) == 0
        x = {0: "1st", 1: "Male", 2: "Adult", 3: "No"}
        assert freq_table.count_freq(x) == 118

    def test_count_freq_cached(self):
        freq_table = FrequencyTable(self.var_sample, cache_size=10)
        x = {0: "1st", 1: "Male", 2: "Child", 3: "No"}
        first_time = freq_table.count_freq(x)
        second_time = freq_table.count_freq(x)
        assert first_time == second_time
        x = {0: "1st", 1: "Male", 2: "Adult", 3: "No"}
        first_time = freq_table.count_freq(x)
        second_time = freq_table.count_freq(x)
        assert first_time == second_time

    def test_cond_prob_without_smoothing(self):
        freq_table = FrequencyTable(self.var_sample)
        prob = freq_table.cond_prob({3: "Yes"}, {2: "Adult"}, False)
        assert_almost_equal(prob, 0.3126195)
        prob = freq_table.cond_prob({3: "Yes"}, {2: "Adult", 0: "1st"}, False)
        assert_almost_equal(prob, 0.6175549)

    def test_cond_prob_with_smoothing(self):
        freq_table = FrequencyTable(self.var_sample)
        prob = freq_table.cond_prob({2: "Child"}, {0: "Crew"})
        assert_almost_equal(prob, 1 / (885 + 2))
