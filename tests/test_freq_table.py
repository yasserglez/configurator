from numpy.testing import assert_almost_equal

from configurator.util import get_var_domains
from configurator.freq_table import FrequencyTable


class TestFrequencyTable(object):

    def test_count_freq(self, titanic_sample):
        var_domains, sample = get_var_domains(titanic_sample), titanic_sample
        freq_table = FrequencyTable(var_domains, sample)
        x = {0: "1st", 1: "Male", 2: "Child", 3: "No"}
        assert freq_table.count_freq(x) == 0
        x = {0: "1st", 1: "Male", 2: "Adult", 3: "No"}
        assert freq_table.count_freq(x) == 118

    def test_count_freq_cached(self, titanic_sample):
        var_domains, sample = get_var_domains(titanic_sample), titanic_sample
        freq_table = FrequencyTable(var_domains, sample, cache_size=10)
        x = {0: "1st", 1: "Male", 2: "Child", 3: "No"}
        first_time = freq_table.count_freq(x)
        second_time = freq_table.count_freq(x)
        assert first_time == second_time
        x = {0: "1st", 1: "Male", 2: "Adult", 3: "No"}
        first_time = freq_table.count_freq(x)
        second_time = freq_table.count_freq(x)
        assert first_time == second_time

    def test_cond_prob(self, titanic_sample):
        var_domains, sample = get_var_domains(titanic_sample), titanic_sample
        freq_table = FrequencyTable(var_domains, sample)
        prob = freq_table.cond_prob({3: "Yes"}, {2: "Adult"})
        assert_almost_equal(prob, 0.3126195)
        prob = freq_table.cond_prob({3: "Yes"}, {2: "Adult", 0: "1st"})
        assert_almost_equal(prob, 0.6175549)

    def test_cond_prob_uniform(self):
        var_domains, sample = [[0, 1], [0, 1, 2]], None
        freq_table = FrequencyTable(var_domains, sample)
        assert freq_table.cond_prob({0: 0}, {}) == 0.5
        assert freq_table.cond_prob({0: 0}, {1: 0}) == 0.5
