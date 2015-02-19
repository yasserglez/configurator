import pytest

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


class TestCSP(object):

    def test_solve(self, australia):
        solution = australia.solve()
        for var_name, var_value in solution.items():
            assert var_value in australia.domain[var_name]
        for var_names, constrain_fun in australia.constraints:
            var_values = [solution[var_name] for var_name in var_names]
            assert constrain_fun(var_names, var_values)

    def test_assign_variable(self, australia):
        australia.assign_variable("T", 0, enforce_consistency=False)
        assert australia.get_assignment() == {"T": 0}

    def test_enforce_consistency(self, australia):
        # Using the consistent assignment on page 138.
        australia.assign_variable("T", 0)
        assert australia.pruned_domain["WA"] == [0, 1, 2]
        assert australia.pruned_domain["NT"] == [0, 1, 2]
        assert australia.pruned_domain["SA"] == [0, 1, 2]
        assert australia.pruned_domain["Q"] == [0, 1, 2]
        assert australia.pruned_domain["NSW"] == [0, 1, 2]
        assert australia.pruned_domain["V"] == [0, 1, 2]
        assert australia.pruned_domain["T"] == [0]
        assert australia.get_assignment() == {"T": 0}
        australia.assign_variable("WA", 0)
        assert australia.pruned_domain["WA"] == [0]
        assert australia.pruned_domain["NT"] == [1, 2]
        assert australia.pruned_domain["SA"] == [1, 2]
        assert australia.pruned_domain["Q"] == [0]
        assert australia.pruned_domain["NSW"] == [1, 2]
        assert australia.pruned_domain["V"] == [0]
        assert australia.pruned_domain["T"] == [0]
        assert australia.get_assignment() == {"WA": 0, "Q": 0, "V": 0, "T": 0}
        australia.assign_variable("NT", 1)
        assert australia.pruned_domain["WA"] == [0]
        assert australia.pruned_domain["NT"] == [1]
        assert australia.pruned_domain["SA"] == [2]
        assert australia.pruned_domain["Q"] == [0]
        assert australia.pruned_domain["NSW"] == [1]
        assert australia.pruned_domain["V"] == [0]
        assert australia.pruned_domain["T"] == [0]
        assert australia.get_assignment() == {"WA": 0, "NT": 1, "SA": 2,
                                              "Q": 0, "NSW": 1, "V": 0, "T": 0}
