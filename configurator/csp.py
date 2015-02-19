"""Constraint satisfaction problems.
"""

from simpleai.search import (CspProblem, backtrack,
                             MOST_CONSTRAINED_VARIABLE,
                             LEAST_CONSTRAINING_VALUE)


class CSP(object):
    """Constraint satisfaction problem.

    Arguments:
        domain: A dictionary mapping each variable name (string) to an
            iterable with their values (integers).
        constraints: A list of tuples with two components each: i) a
            tuple with the names of the variables involved in the
            constraint, and ii) a function that checks the constraint.
            The constraint functions will receive a variables tuple
            and a values tuple, both containing only the restricted
            variable names and their values (in the same order
            provided in :obj:`constraints`). The function should
            return :obj:`True` if the values satisfy the constraint,
            :obj:`False` otherwise. The constraints must be normalized
            (i.e. two different constraints shouldn't involve the same
            set of variables).

    All the arguments are available as instance attributes.
    """

    def __init__(self, domain, constraints):
        self.domain = domain
        # Ensure that the constraints are normalized.
        constraint_support = set()
        for var_names, constraint_fun in constraints:
            var_names = frozenset(var_names)
            if var_names in constraint_support:
                raise ValueError("The constraints must be normalized")
            constraint_support.add(var_names)
        self.constraints = constraints
        self._reset_state()

    def solve(self):
        """Find a solution to the constraint satisfaction problem.

        Returns:
            A dictionary with the values assigned to the variables if
            a solution was found, :obj:`None` otherwise.
        """
        variables = tuple(self.domain.keys())
        csp = CspProblem(variables, self.domain, self.constraints)
        solution = backtrack(csp, variable_heuristic=MOST_CONSTRAINED_VARIABLE,
                             value_heuristic=LEAST_CONSTRAINING_VALUE,
                             inference=True)
        return solution

    # The following are internal methods used to keep track of the
    # assignments and ensure global consistency after each variable is
    # set when the dialog is being used.

    def _assign_variable(self, var_index, var_value):
        pass

    def _get_assignment(self):
        pass

    def _ensure_consistency(self):
        pass

    def _reset_state(self):
        pass

    def _push_state(self):
        pass

    def _pop_state(self):
        pass
