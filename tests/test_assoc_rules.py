import pytest
from numpy.testing import assert_almost_equal

from configurator.assoc_rules import AssociationRule, AssociationRuleMiner


@pytest.fixture(scope="module")
def rule():
    lhs = {1: "a", 2: "b"}
    rhs = {3: "c", 4: "d"}
    support = 0.5
    confidence = 0.9
    rule = AssociationRule(lhs, rhs, support, confidence)
    return rule


@pytest.fixture(scope="module")
def miner(titanic_data):
    miner = AssociationRuleMiner(titanic_data)
    return miner


class TestAssociationRule(object):

    def test_is_lhs_compatible(self, rule):
        assert rule.is_lhs_compatible({1: "a", 2: "b"})
        assert not rule.is_lhs_compatible({1: "z", 2: "b"})

    def test_is_rhs_compatible(self, rule):
        assert rule.is_rhs_compatible({1: "a", 2: "b"})
        assert rule.is_rhs_compatible({1: "a", 2: "b", 3: "c"})
        assert not rule.is_rhs_compatible({1: "a", 2: "b", 3: "z"})
        assert not rule.is_rhs_compatible({1: "a", 2: "b", 3: "c", 4: "d"})

    def test_is_applicable(self, rule):
        assert rule.is_applicable({1: "a", 2: "b", 3: "c"})
        assert not rule.is_applicable({1: "z", 2: "b", 3: "c"})

    def test_apply_rule(self, rule):
        observation = {1: "a", 2: "b", 3: "c"}
        rule.apply_rule(observation)
        assert observation == {1: "a", 2: "b", 3: "c", 4: "d"}


class TestAssociationRuleMiner(object):

    def test_mine_assoc_rlues(self, miner):
        rules = miner.mine_assoc_rules(min_support=0.5, min_confidence=0.95)
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
