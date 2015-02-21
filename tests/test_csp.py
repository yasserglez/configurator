import pytest

import igraph

from configurator.csp import CSP


@pytest.fixture(scope="function")
def australia():
    # Chapter 5 of Russell, S. J. and Norvig, P.,
    # Artificial Intelligence: A Modern Approach, 2nd edition.
    # http://aima.cs.berkeley.edu/2nd-ed/newchap05.pdf
    regions = ("WA", "NT", "Q", "NSW", "V", "T", "SA")
    domains = [[0, 1, 2] for region in regions]
    must_have_distinct_colors = lambda _, color: color[0] != color[1]
    constraints = [
        ((0, 1), must_have_distinct_colors),
        ((0, 6), must_have_distinct_colors),
        ((6, 1), must_have_distinct_colors),
        ((6, 2), must_have_distinct_colors),
        ((1, 2), must_have_distinct_colors),
        ((6, 3), must_have_distinct_colors),
        ((2, 3), must_have_distinct_colors),
        ((6, 4), must_have_distinct_colors),
        ((3, 4), must_have_distinct_colors),
    ]
    csp = CSP(domains, constraints)
    return csp


@pytest.fixture(scope="function")
def australia_without_SA():
    # Constraint graph in Figure 5.1 (b).
    regions = ("WA", "NT", "Q", "NSW", "V", "T")
    domains = [[0, 1] for region in regions]
    must_have_distinct_colors = lambda _, color: color[0] != color[1]
    constraints = [
        ((0, 1), must_have_distinct_colors),
        ((1, 2), must_have_distinct_colors),
        ((2, 3), must_have_distinct_colors),
        ((3, 4), must_have_distinct_colors),
    ]
    csp = CSP(domains, constraints)
    return csp


class TestCSP(object):

    def test_solve(self, australia):
        solution = australia.solve()
        for var_name, var_value in solution.items():
            assert var_value in australia.domains[var_name]
        for var_names, constrain_fun in australia.constraints:
            var_values = [solution[var_name] for var_name in var_names]
            assert constrain_fun(var_names, var_values)

    def test_assign_variable(self, australia):
        australia.assign_variable(5, 0, prune_domains=False)
        assert australia.get_assignment() == {5: 0}

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

    def test_prune_domains(self, australia):
        # Using the consistent assignment on page 138.
        csp = australia
        csp.assign_variable(5, 0)
        assert csp.pruned_domains[0] == [0, 1, 2]
        assert csp.pruned_domains[1] == [0, 1, 2]
        assert csp.pruned_domains[6] == [0, 1, 2]
        assert csp.pruned_domains[2] == [0, 1, 2]
        assert csp.pruned_domains[3] == [0, 1, 2]
        assert csp.pruned_domains[4] == [0, 1, 2]
        assert csp.pruned_domains[5] == [0]
        assert csp.get_assignment() == {5: 0}
        csp.assign_variable(0, 0)
        assert csp.pruned_domains[0] == [0]
        assert csp.pruned_domains[1] == [1, 2]
        assert csp.pruned_domains[6] == [1, 2]
        assert csp.pruned_domains[2] == [0]
        assert csp.pruned_domains[3] == [1, 2]
        assert csp.pruned_domains[4] == [0]
        assert csp.pruned_domains[5] == [0]
        assert csp.get_assignment() == {0: 0, 2: 0, 4: 0, 5: 0}
        csp.assign_variable(1, 1)
        assert csp.pruned_domains[0] == [0]
        assert csp.pruned_domains[1] == [1]
        assert csp.pruned_domains[6] == [2]
        assert csp.pruned_domains[2] == [0]
        assert csp.pruned_domains[3] == [1]
        assert csp.pruned_domains[4] == [0]
        assert csp.pruned_domains[5] == [0]
        assert csp.get_assignment() == {0: 0, 1: 1, 6: 2, 2:
                                        0, 3: 1, 4: 0, 5: 0}

    def test_prune_domains_in_tree_csp(self, australia_without_SA):
        csp = australia_without_SA
        csp.assign_variable(5, 0)
        assert csp.pruned_domains[0] == [0, 1]
        assert csp.pruned_domains[1] == [0, 1]
        assert csp.pruned_domains[2] == [0, 1]
        assert csp.pruned_domains[3] == [0, 1]
        assert csp.pruned_domains[4] == [0, 1]
        assert csp.pruned_domains[5] == [0]
        assert csp.get_assignment() == {5: 0}
        csp.assign_variable(0, 0)
        assert csp.pruned_domains[0] == [0]
        assert csp.pruned_domains[1] == [1]
        assert csp.pruned_domains[2] == [0]
        assert csp.pruned_domains[3] == [1]
        assert csp.pruned_domains[4] == [0]
        assert csp.pruned_domains[5] == [0]
        assert csp.get_assignment() == {0: 0, 1: 1, 2: 0,
                                        3: 1, 4: 0, 5: 0}
