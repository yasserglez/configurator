import pytest

from configurator.csp import CSP


@pytest.fixture(scope="module")
def australia():
    # Chapter 5 of Russell, S. J. and Norvig, P.,
    # Artificial Intelligence: A Modern Approach, 2nd edition.
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
