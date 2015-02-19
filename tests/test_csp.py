import pytest

import igraph

from configurator.csp import CSP


@pytest.fixture(scope="function")
def australia():
    # Chapter 5 of Russell, S. J. and Norvig, P.,
    # Artificial Intelligence: A Modern Approach, 2nd edition.
    # http://aima.cs.berkeley.edu/2nd-ed/newchap05.pdf
    regions = ("WA", "NT", "SA", "Q", "NSW", "V", "T")
    domain = {region: [0, 1, 2] for region in regions}
    must_have_distinct_colors = lambda _, color: color[0] != color[1]
    constraints = [
        (("WA", "NT"), must_have_distinct_colors),
        (("WA", "SA"), must_have_distinct_colors),
        (("SA", "NT"), must_have_distinct_colors),
        (("SA", "Q"), must_have_distinct_colors),
        (("NT", "Q"), must_have_distinct_colors),
        (("SA", "NSW"), must_have_distinct_colors),
        (("Q", "NSW"), must_have_distinct_colors),
        (("SA", "V"), must_have_distinct_colors),
        (("NSW", "V"), must_have_distinct_colors),
    ]
    csp = CSP(domain, constraints)
    return csp


@pytest.fixture(scope="function")
def australia_without_SA():
    # Constraint graph in Figure 5.1 (b).
    regions = ("WA", "NT", "Q", "NSW", "V", "T")
    domain = {region: [0, 1] for region in regions}
    must_have_distinct_colors = lambda _, color: color[0] != color[1]
    constraints = [
        (("WA", "NT"), must_have_distinct_colors),
        (("NT", "Q"), must_have_distinct_colors),
        (("Q", "NSW"), must_have_distinct_colors),
        (("NSW", "V"), must_have_distinct_colors),
    ]
    csp = CSP(domain, constraints)
    return csp


class TestCSP(object):

    def test_solve(self, australia):
        solution = australia.solve()
        for var_name, var_value in solution.items():
            assert var_value in australia.domain[var_name]
        for var_names, constrain_fun in australia.constraints:
            var_values = [solution[var_name] for var_name in var_names]
            assert constrain_fun(var_names, var_values)

    def test_assign_variable(self, australia):
        australia.assign_variable("T", 0, prune_domain=False)
        assert australia.get_assignment() == {"T": 0}

    def test_is_acyclic(self):
        empty = igraph.Graph(6)
        tree = igraph.Graph.Tree(6, 2**6 - 1)
        full = igraph.Graph.Full(6)
        assert CSP._is_acyclic(empty)
        assert CSP._is_acyclic(tree)
        assert CSP._is_acyclic(empty + tree)
        assert not CSP._is_acyclic(full)
        assert not CSP._is_acyclic(empty + full)
        assert not CSP._is_acyclic(tree + full)

    def test_is_tree_csp(self, australia, australia_without_SA):
        assert not australia.is_tree_csp
        assert australia_without_SA.is_tree_csp

    def test_prune_domain(self, australia):
        # Using the consistent assignment on page 138.
        csp = australia
        csp.assign_variable("T", 0)
        assert csp.pruned_domain["WA"] == [0, 1, 2]
        assert csp.pruned_domain["NT"] == [0, 1, 2]
        assert csp.pruned_domain["SA"] == [0, 1, 2]
        assert csp.pruned_domain["Q"] == [0, 1, 2]
        assert csp.pruned_domain["NSW"] == [0, 1, 2]
        assert csp.pruned_domain["V"] == [0, 1, 2]
        assert csp.pruned_domain["T"] == [0]
        assert csp.get_assignment() == {"T": 0}
        csp.assign_variable("WA", 0)
        assert csp.pruned_domain["WA"] == [0]
        assert csp.pruned_domain["NT"] == [1, 2]
        assert csp.pruned_domain["SA"] == [1, 2]
        assert csp.pruned_domain["Q"] == [0]
        assert csp.pruned_domain["NSW"] == [1, 2]
        assert csp.pruned_domain["V"] == [0]
        assert csp.pruned_domain["T"] == [0]
        assert csp.get_assignment() == {"WA": 0, "Q": 0, "V": 0, "T": 0}
        csp.assign_variable("NT", 1)
        assert csp.pruned_domain["WA"] == [0]
        assert csp.pruned_domain["NT"] == [1]
        assert csp.pruned_domain["SA"] == [2]
        assert csp.pruned_domain["Q"] == [0]
        assert csp.pruned_domain["NSW"] == [1]
        assert csp.pruned_domain["V"] == [0]
        assert csp.pruned_domain["T"] == [0]
        assert csp.get_assignment() == {"WA": 0, "NT": 1, "SA": 2,
                                        "Q": 0, "NSW": 1, "V": 0, "T": 0}

    def test_prune_domain_in_tree_csp(self, australia_without_SA):
        csp = australia_without_SA
        csp.assign_variable("T", 0)
        assert csp.pruned_domain["WA"] == [0, 1]
        assert csp.pruned_domain["NT"] == [0, 1]
        assert csp.pruned_domain["Q"] == [0, 1]
        assert csp.pruned_domain["NSW"] == [0, 1]
        assert csp.pruned_domain["V"] == [0, 1]
        assert csp.pruned_domain["T"] == [0]
        assert csp.get_assignment() == {"T": 0}
        csp.assign_variable("WA", 0)
        assert csp.pruned_domain["WA"] == [0]
        assert csp.pruned_domain["NT"] == [1]
        assert csp.pruned_domain["Q"] == [0]
        assert csp.pruned_domain["NSW"] == [1]
        assert csp.pruned_domain["V"] == [0]
        assert csp.pruned_domain["T"] == [0]
        assert csp.get_assignment() == {"WA": 0, "NT": 1, "Q": 0,
                                        "NSW": 1, "V": 0, "T": 0}
