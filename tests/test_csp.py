import pytest

import igraph

from configurator.csp import CSP


@pytest.fixture(scope="function")
def australia():
    # Chapter 5 of Russell, S. J. and Norvig, P.,
    # Artificial Intelligence: A Modern Approach, 2nd edition.
    # http://aima.cs.berkeley.edu/2nd-ed/newchap05.pdf
    regions = ("WA", "NT", "Q", "NSW", "V", "T", "SA")
    var_domains = [["red", "green", "blue"] for region in regions]
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
    csp = CSP(var_domains, constraints)
    return csp


class TestCSP(object):

    def test_is_consistent(self, australia):
        assert australia.is_consistent()
        australia.var_domains[0] = ["red"]
        australia.var_domains[1] = ["red"]
        assert not australia.is_consistent()

    def test_solve(self, australia):
        num_solutions = 0
        for solution in australia.solve():
            num_solutions += 1
            for var_name, var_value in solution.items():
                assert var_value in australia.var_domains[var_name]
            for var_names, constrain_fun in australia.constraints:
                var_values = [solution[var_name] for var_name in var_names]
                assert constrain_fun(var_names, var_values)
        assert num_solutions == 18

    def test_reset(self, australia):
        csp = australia
        assert csp.pruned_var_domains == csp.var_domains
        csp.assign_variable(0, "red")
        assert csp.pruned_var_domains != csp.var_domains
        csp.reset()
        assert csp.pruned_var_domains == csp.var_domains

    def test_assign_variable(self, australia):
        australia.assign_variable(5, "red")
        assert australia.assignment == {5: "red"}

    def test_global_consistency(self, australia):
        csp = australia
        # Using the consistent assignment on page 138.
        csp.assign_variable(5, "red", "global")
        assert csp.pruned_var_domains[0] == ["red", "green", "blue"]
        assert csp.pruned_var_domains[1] == ["red", "green", "blue"]
        assert csp.pruned_var_domains[2] == ["red", "green", "blue"]
        assert csp.pruned_var_domains[3] == ["red", "green", "blue"]
        assert csp.pruned_var_domains[4] == ["red", "green", "blue"]
        assert csp.pruned_var_domains[5] == ["red"]
        assert csp.pruned_var_domains[6] == ["red", "green", "blue"]
        assert csp.assignment == {5: "red"}
        csp.assign_variable(0, "red", "global")
        assert csp.pruned_var_domains[0] == ["red"]
        assert csp.pruned_var_domains[1] == ["green", "blue"]
        assert csp.pruned_var_domains[2] == ["red"]
        assert csp.pruned_var_domains[3] == ["green", "blue"]
        assert csp.pruned_var_domains[4] == ["red"]
        assert csp.pruned_var_domains[5] == ["red"]
        assert csp.pruned_var_domains[6] == ["green", "blue"]
        assert csp.assignment == {0: "red", 2: "red", 4: "red", 5: "red"}
        csp.assign_variable(1, "green", "global")
        assert csp.pruned_var_domains[0] == ["red"]
        assert csp.pruned_var_domains[1] == ["green"]
        assert csp.pruned_var_domains[2] == ["red"]
        assert csp.pruned_var_domains[3] == ["green"]
        assert csp.pruned_var_domains[4] == ["red"]
        assert csp.pruned_var_domains[5] == ["red"]
        assert csp.pruned_var_domains[6] == ["blue"]
        assert csp.assignment == {0: "red", 1: "green", 6: "blue", 2: "red",
                                  3: "green", 4: "red", 5: "red"}

    def test_local_consistency(self, australia):
        csp = australia
        # Using the consistent assignment on page 138.
        csp.assign_variable(5, "red", "local")
        assert csp.pruned_var_domains[0] == ["red", "green", "blue"]
        assert csp.pruned_var_domains[1] == ["red", "green", "blue"]
        assert csp.pruned_var_domains[2] == ["red", "green", "blue"]
        assert csp.pruned_var_domains[3] == ["red", "green", "blue"]
        assert csp.pruned_var_domains[4] == ["red", "green", "blue"]
        assert csp.pruned_var_domains[5] == ["red"]
        assert csp.pruned_var_domains[6] == ["red", "green", "blue"]
        assert csp.assignment == {5: "red"}
        csp.assign_variable(0, "red", "local")
        assert csp.pruned_var_domains[0] == ["red"]
        assert csp.pruned_var_domains[1] == ["green", "blue"]
        assert csp.pruned_var_domains[2] == ["red", "green", "blue"]
        assert csp.pruned_var_domains[3] == ["red", "green", "blue"]
        assert csp.pruned_var_domains[4] == ["red", "green", "blue"]
        assert csp.pruned_var_domains[5] == ["red"]
        assert csp.pruned_var_domains[6] == ["green", "blue"]
        assert csp.assignment == {0: "red", 5: "red"}
        csp.assign_variable(1, "green", "local")
        assert csp.pruned_var_domains[0] == ["red"]
        assert csp.pruned_var_domains[1] == ["green"]
        assert csp.pruned_var_domains[2] == ["red"]
        assert csp.pruned_var_domains[3] == ["green"]
        assert csp.pruned_var_domains[4] == ["red"]
        assert csp.pruned_var_domains[5] == ["red"]
        assert csp.pruned_var_domains[6] == ["blue"]
        assert csp.assignment == {0: "red", 1: "green", 6: "blue", 2: "red",
                                  3: "green", 4: "red", 5: "red"}

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
