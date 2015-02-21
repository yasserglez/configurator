from numpy.testing import assert_raises

from configurator.dialogs import DialogBuilder
from configurator.rules import Rule


class TestDialogBuilder(object):

    def test_rules_constraints_one_required(self):
        domains = [[0, 1], [0, 1, 2]]
        assert_raises(ValueError, DialogBuilder, domains)

    def test_rules_constraints_mutually_exclusive(self):
        domains = [[0, 1], [0, 1, 2]]
        rules = [Rule({0: 0}, {1: 0})]
        constraints = [((0, 1), lambda _, x: x[0] != 0 or x[1] == 0)]
        assert_raises(ValueError, DialogBuilder, domains, rules, constraints)
