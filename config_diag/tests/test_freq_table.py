import string

import numpy as np
import pandas as pd

from ..freq_table import FrequencyTable


class TestFrequencyTable:

    def setup(self):
        np.random.seed(42)
        rows, cols = 10, 4
        data = np.random.randint(1, 6, rows * cols).reshape(rows, cols)
        self.df = pd.DataFrame(data, columns=list(string.ascii_letters[:cols]))

    def test_init(self):
        freq_tab = FrequencyTable(self.df)
        pd.util.testing.assert_frame_equal(freq_tab.df, self.df)

    def test_freq(self):
        freq_tab = FrequencyTable(self.df)
        assert freq_tab.freq({"c": 5}) == 0
        assert freq_tab.freq({"a": 4}) == 4
        assert freq_tab.freq({"a": 4, "b": 5}) == 2
        assert freq_tab.freq({"a": 4, "c": 3, "d": 5}) == 2
        assert freq_tab.freq({"a": 4, "b": 5, "c": 3, "d": 5}) == 1

    def test_joint_prob(self):
        freq_tab = FrequencyTable(self.df)
        assert freq_tab.joint_prob({"c": 5}) == 0.0
        assert freq_tab.joint_prob({"a": 4}) == 0.4
        assert freq_tab.joint_prob({"a": 4, "b": 5}) == 0.2
        assert freq_tab.joint_prob({"a": 4, "c": 3, "d": 5}) == 0.2
        assert freq_tab.joint_prob({"a": 4, "b": 5, "c": 3, "d": 5}) == 0.1

    def test_cond_prob(self):
        freq_tab = FrequencyTable(self.df)
        assert freq_tab.cond_prob({"a": 3}, {"b": 5}) == 0.4
        assert freq_tab.cond_prob({"a": 3, "c": 1}, {"b": 5}) == 0.2
        assert freq_tab.cond_prob({"a": 1}, {"b": 5, "c": 4}) == 0.0
        assert freq_tab.cond_prob({"a": 3, "b": 5}, {"c": 1, "d": 2}) == 1.0
