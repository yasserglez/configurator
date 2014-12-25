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
        assert freq_tab.freq({"C": 5}) == 0
        assert freq_tab.freq({"A": 4}) == 4
        assert freq_tab.freq({"A": 4, "B": 5}) == 2
        assert freq_tab.freq({"A": 4, "C": 3, "D": 5}) == 3
        assert freq_tab.freq({"A": 4, "B": 5, "C": 3, "D": 5}) == 1

    def test_joint_prob(self):
        freq_tab = FrequencyTable(self.df)
        assert freq_tab.joint_prob({"C": 5}) == 0
        assert freq_tab.joint_prob({"A": 4}) == 0.4
        assert freq_tab.joint_prob({"A": 4, "B": 5}) == 0.2
        assert freq_tab.joint_prob({"A": 4, "C": 3, "D": 5}) == 0.3
        assert freq_tab.joint_prob({"A": 4, "B": 5, "C": 3, "D": 5}) == 0.1

    def test_cond_prob(self):
        freq_tab = FrequencyTable(self.df)
        assert freq_tab.cond_prob({"A": 3}, {"B": 5}) == 0.4
        assert freq_tab.cond_prob({"A": 3, "C": 1}, {"B": 5}) == 0.2
        assert freq_tab.cond_prob({"A": 1}, {"B": 5, "C": 4}) == 0
        assert freq_tab.cond_prob({"A": 3, "B": 5}, {"C": 1, "D": 2}) == 0.25
