import pytest
from numpy.testing import assert_almost_equal

from configurator.util import get_config_values
from configurator.freq_table import FrequencyTable


@pytest.fixture(scope="module")
def freq_table(titanic_data):
    var_sample, var_values = titanic_data, get_config_values(titanic_data)
    freq_table = FrequencyTable(var_sample, var_values)
    return freq_table


class TestFrequencyTable(object):

    def test_count_freq(self, freq_table):
        x = {0: "1st", 1: "Male", 2: "Child", 3: "No"}
        assert freq_table.count_freq(x) == 0
        x = {0: "1st", 1: "Male", 2: "Adult", 3: "No"}
        assert freq_table.count_freq(x) == 118

    def test_count_freq_cached(self, freq_table):
        x = {0: "1st", 1: "Male", 2: "Child", 3: "No"}
        first_time = freq_table.count_freq(x)
        second_time = freq_table.count_freq(x)
        assert first_time == second_time
        x = {0: "1st", 1: "Male", 2: "Adult", 3: "No"}
        first_time = freq_table.count_freq(x)
        second_time = freq_table.count_freq(x)
        assert first_time == second_time

    def test_cond_prob_without_smoothing(self, freq_table):
        prob = freq_table.cond_prob({3: "Yes"}, {2: "Adult"}, False)
        assert_almost_equal(prob, 0.3126195)
        prob = freq_table.cond_prob({3: "Yes"}, {2: "Adult", 0: "1st"}, False)
        assert_almost_equal(prob, 0.6175549)

    def test_cond_prob_with_smoothing(self, freq_table):
        prob = freq_table.cond_prob({2: "Child"}, {0: "Crew"})
        assert_almost_equal(prob, 1 / (885 + 2))