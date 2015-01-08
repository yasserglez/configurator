from numpy.testing import (assert_raises, assert_array_equal,
                           assert_almost_equal)

from .examples import load_titanic
from ..assoc_rules import AssociationRuleMiner


class TestAssociationRuleMiner(object):

    def setup(self):
        self.data = load_titanic()

    def test_init(self):
        miner = AssociationRuleMiner(self.data)
        assert_array_equal(miner.data, self.data)

    def _test_mine_assoc_rlues(self, algorithm):
        miner = AssociationRuleMiner(self.data)
        rules = miner.mine_assoc_rules(min_support=0.5, min_confidence=0.95,
                                       algorithm=algorithm)
        assert len(rules) == 3
        rules.sort(key=lambda rule: rule.confidence, reverse=True)
        # Rule #1
        assert rules[0].lhs == {1: "Male", 3: "No"}
        assert rules[0].rhs == {2: "Adult"}
        assert_almost_equal(rules[0].support, 0.6038164)
        assert_almost_equal(rules[0].confidence, 0.9743402)
        # Rule #2
        assert rules[1].lhs == {3: "No"}
        assert rules[1].rhs == {2: "Adult"}
        assert_almost_equal(rules[1].support, 0.6533394)
        assert_almost_equal(rules[1].confidence, 0.9651007)
        # Rule #3
        assert rules[2].lhs == {1: "Male"}
        assert rules[2].rhs == {2: "Adult"}
        assert_almost_equal(rules[2].support, 0.7573830)
        assert_almost_equal(rules[2].confidence, 0.9630272)

    def test_mine_assoc_rules_apriori(self):
        self._test_mine_assoc_rlues('apriori')

    def test_mine_assoc_rules_fpgrowth(self):
        self._test_mine_assoc_rlues('fp-growth')

    def test_mine_assoc_rules_invalid(self):
        miner = AssociationRuleMiner(self.data)
        assert_raises(ValueError, miner.mine_assoc_rules,
                                 algorithm='invalid')
