import pytest
from numpy.testing import assert_almost_equal, assert_raises

from configurator.util import get_domains
from configurator.freq_table import FrequencyTable


@pytest.fixture(scope="module")
def freq_table(titanic_data):
    domains, sample = get_domains(titanic_data), titanic_data
    freq_table = FrequencyTable(domains, sample)
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

    def test_count_freq_no_sample(self):
        domains, sample = [[0, 1]], None
        freq_table = FrequencyTable(domains, sample)
        assert freq_table.count_freq({0: 0}) == 0

    def test_cond_prob_without_smoothing(self, freq_table):
        prob = freq_table.cond_prob({3: "Yes"}, {2: "Adult"}, False)
        assert_almost_equal(prob, 0.3126195)
        prob = freq_table.cond_prob({3: "Yes"}, {2: "Adult", 0: "1st"}, False)
        assert_almost_equal(prob, 0.6175549)

    def test_cond_prob_with_smoothing(self, freq_table):
        prob = freq_table.cond_prob({2: "Child"}, {0: "Crew"})
        assert_almost_equal(prob, 1 / (885 + 2))

    def test_cond_prob_no_sample(self):
        domains, sample = [[0, 1], [0, 1, 2]], None
        freq_table = FrequencyTable(domains, sample)
        assert freq_table.cond_prob({0: 0}, {}, True) == 0.5
        assert freq_table.cond_prob({0: 0}, {1: 0}, True) == 0.5
        assert_raises(ZeroDivisionError, freq_table.cond_prob,
                      {0: 0}, {}, False)
