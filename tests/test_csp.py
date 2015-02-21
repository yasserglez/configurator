import pytest

import igraph

from configurator.csp import CSP


@pytest.fixture(scope="function")
def australia():
    # Chapter 5 of Russell, S. J. and Norvig, P.,
    # Artificial Intelligence: A Modern Approach, 2nd edition.
    # http://aima.cs.berkeley.edu/2nd-ed/newchap05.pdf
    regions = ("WA", "NT", "Q", "NSW", "V", "T", "SA")
    domain = [["red", "green", "blue"] for region in regions]
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
    csp = CSP(domain, constraints)
    return csp


@pytest.fixture(scope="function")
def australia_without_SA():
    # Constraint graph in Figure 5.1 (b).
    regions = ("WA", "NT", "Q", "NSW", "V", "T")
    domain = [["red", "green"] for region in regions]
    must_have_distinct_colors = lambda _, color: color[0] != color[1]
    constraints = [
        ((0, 1), must_have_distinct_colors),
        ((1, 2), must_have_distinct_colors),
        ((2, 3), must_have_distinct_colors),
        ((3, 4), must_have_distinct_colors),
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

    def test_reset(self, australia):
        csp = australia
        assert csp.pruned_domain == csp.domain
        csp.assign_variable(0, "red")
        assert csp.pruned_domain != csp.domain
        csp.reset()
        assert csp.pruned_domain == csp.domain

    def test_assign_variable(self, australia):
        australia.assign_variable(5, "red", prune_domain=False)
        assert australia.assignment == {5: "red"}

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
        csp.assign_variable(5, "red")
        assert csp.pruned_domain[0] == ["red", "green", "blue"]
        assert csp.pruned_domain[1] == ["red", "green", "blue"]
        assert csp.pruned_domain[6] == ["red", "green", "blue"]
        assert csp.pruned_domain[2] == ["red", "green", "blue"]
        assert csp.pruned_domain[3] == ["red", "green", "blue"]
        assert csp.pruned_domain[4] == ["red", "green", "blue"]
        assert csp.pruned_domain[5] == ["red"]
        assert csp.assignment == {5: "red"}
        csp.assign_variable(0, "red")
        assert csp.pruned_domain[0] == ["red"]
        assert csp.pruned_domain[1] == ["green", "blue"]
        assert csp.pruned_domain[6] == ["green", "blue"]
        assert csp.pruned_domain[2] == ["red"]
        assert csp.pruned_domain[3] == ["green", "blue"]
        assert csp.pruned_domain[4] == ["red"]
        assert csp.pruned_domain[5] == ["red"]
        assert csp.assignment == {0: "red", 2: "red", 4: "red", 5: "red"}
        csp.assign_variable(1, "green")
        assert csp.pruned_domain[0] == ["red"]
        assert csp.pruned_domain[1] == ["green"]
        assert csp.pruned_domain[6] == ["blue"]
        assert csp.pruned_domain[2] == ["red"]
        assert csp.pruned_domain[3] == ["green"]
        assert csp.pruned_domain[4] == ["red"]
        assert csp.pruned_domain[5] == ["red"]
        assert csp.assignment == {0: "red", 1: "green", 6: "blue", 2: "red",
                                  3: "green", 4: "red", 5: "red"}

    def test_prune_domain_in_tree_csp(self, australia_without_SA):
        csp = australia_without_SA
        csp.assign_variable(5, "red")
        assert csp.pruned_domain[0] == ["red", "green"]
        assert csp.pruned_domain[1] == ["red", "green"]
        assert csp.pruned_domain[2] == ["red", "green"]
        assert csp.pruned_domain[3] == ["red", "green"]
        assert csp.pruned_domain[4] == ["red", "green"]
        assert csp.pruned_domain[5] == ["red"]
        assert csp.assignment == {5: "red"}
        csp.assign_variable(0, "red")
        assert csp.pruned_domain[0] == ["red"]
        assert csp.pruned_domain[1] == ["green"]
        assert csp.pruned_domain[2] == ["red"]
        assert csp.pruned_domain[3] == ["green"]
        assert csp.pruned_domain[4] == ["red"]
        assert csp.pruned_domain[5] == ["red"]
        assert csp.assignment == {0: "red", 1: "green", 2: "red", 3:
                                  "green", 4: "red", 5: "red"}
