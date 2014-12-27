
import numpy as np

from ..assoc_rules import AssociationRuleMiner


class TestAssociationRuleMiner:

    def setup(self):
        # Sample dataset from http://www.jstatsoft.org/v14/i15
        self.data = np.array([1, 1, 0, 0,
                              0, 1, 0, 1,
                              1, 1, 1, 0,
                              0, 0, 1, 0]).reshape(4, 4)

    def test_init(self):
        miner = AssociationRuleMiner(self.data)
        assert (miner.data == self.data).all()

    def _test_mine_assoc_rules(self, algorithm):
        miner = AssociationRuleMiner(self.data)
        rules = miner.mine_assoc_rules(min_support=0.5, min_confidence=0.5,
                                       algorithm=algorithm)
        assert len(rules) == 2
        # Rule #1
        assert rules[0].support == 0.5
        assert rules[0].confidence == 1.0
        assert rules[0].lhs == {0: 1}
        assert rules[0].rhs == {1: 1}
        # Rule #2
        assert rules[1].support == 0.5
        assert rules[1].confidence == 0.6666667
        assert rules[1].lhs == {1: 1}
        assert rules[1].rhs == {0: 1}

    def test_mine_assoc_rules_apriori(self):
        self._test_mine_assoc_rules('apriori')

    def test_mine_assoc_rules_fpgrowth(self):
        self._test_mine_assoc_rules('fp-growth')
