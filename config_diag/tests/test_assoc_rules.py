
import os

import numpy as np

from ..assoc_rules import AssociationRuleMiner


class TestAssociationRuleMiner:

    def setup(self):
        tests_dir = os.path.abspath(os.path.dirname(__file__))
        self.data = np.genfromtxt(os.path.join(tests_dir, "Titanic.csv"),
                                  skip_header=1, dtype=np.dtype(str),
                                  delimiter=",")

    def test_init(self):
        miner = AssociationRuleMiner(self.data)
        np.testing.assert_array_equal(miner.data, self.data)

    def test_mine_assoc_rlues(self):
        miner = AssociationRuleMiner(self.data)
        rules = miner.mine_assoc_rules(min_support=0.5, min_confidence=0.95)
        assert len(rules) == 3
        # Rule #1
        assert rules[0].lhs == {1: "Male", 3: "No"}
        assert rules[0].rhs == {2: "Adult"}
        np.testing.assert_almost_equal(rules[0].support, 0.6038164)
        np.testing.assert_almost_equal(rules[0].confidence, 0.9743402)
        # Rule #2
        assert rules[1].lhs == {3: "No"}
        assert rules[1].rhs == {2: "Adult"}
        np.testing.assert_almost_equal(rules[1].support, 0.6533394)
        np.testing.assert_almost_equal(rules[1].confidence, 0.9651007)
        # Rule #3
        assert rules[2].lhs == {1: "Male"}
        assert rules[2].rhs == {2: "Adult"}
        np.testing.assert_almost_equal(rules[2].support, 0.7573830)
        np.testing.assert_almost_equal(rules[2].confidence, 0.9630272)
