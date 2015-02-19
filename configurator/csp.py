"""Constraint satisfaction problems.
"""

import copy
import pprint
import logging

import igraph
from simpleai.search import (CspProblem, backtrack,
                             MOST_CONSTRAINED_VARIABLE,
                             LEAST_CONSTRAINING_VALUE)


log = logging.getLogger(__name__)


class CSP(object):
    """Constraint satisfaction problem.

    Arguments:
        domain: A dictionary mapping each variable name (string) to an
            iterable with their values (strings or integers). All the
            variables must be domain consistent.
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
        # simpleai expects a list with the domain values
        self.domain = {v: list(set(var_values))
                       for v, var_values in domain.items()}
        self._variables = tuple(self.domain.keys())  # cached for simpleai
        # Ensure that the constraints are normalized.
        constraint_support = set()
        for var_names, constraint_fun in constraints:
            var_names = frozenset(var_names)
            if var_names in constraint_support:
                raise ValueError("The constraints must be normalized")
            constraint_support.add(var_names)
        self.constraints = constraints
        # Initialization for the internal methods.
        self._is_tree_csp = self._compute_is_tree_csp()
        self.reset()

    def solve(self):
        """Find a consistent assignment of the variables.

        Returns:
            A dictionary with the values assigned to the variables if
            a solution was found, :obj:`None` otherwise.
        """
        solution = self._backtracking_solver(self.domain)
        return solution

    def _backtracking_solver(self, domain):
        # Backtracking solver maintaining arc consistency (using the
        # AC-3 algorithm), with the minimum-remaining-values heuristic
        # for variable selection and the least-constraining-value
        # heuristic for value selection.
        csp = CspProblem(self._variables, domain, self.constraints)
        solution = backtrack(csp, variable_heuristic=MOST_CONSTRAINED_VARIABLE,
                             value_heuristic=LEAST_CONSTRAINING_VALUE,
                             inference=True)
        return solution

    # The following are internal methods used to keep track of the
    # assignments and enforce global consistency after each variable
    # is assigned when the dialog is being used.

    def reset(self):
        self._assignment = {}
        self.pruned_domain = copy.deepcopy(self.domain)

    def get_assignment(self):
        return self._assignment

    def assign_variable(self, v, var_value, enforce_consistency=True):
        if v in self._assignment:
            raise ValueError("The variable is already assigned")
        if var_value not in self.pruned_domain[v]:
            raise ValueError("Invalid assignment in the current state")
        self._assignment[v] = var_value
        if enforce_consistency:
            self.enforce_consistency()

    def enforce_consistency(self):
        log.debug("enforcing global consistency")
        log.debug("initial assignment:\n%s", pprint.pformat(self._assignment))
        log.debug("initial domain:\n%s", pprint.pformat(self.pruned_domain))
        # Start with unary constraints.
        for v, var_value in self._assignment.items():
            self.pruned_domain[v] = [var_value]
        log.debug("after enforcing unary constraints:\n%s",
                  pprint.pformat(self.pruned_domain))
        # Then, check that all possible answers for the next question
        # lead to a consistent assignment.
        base_domain = copy.deepcopy(self.pruned_domain.copy())
        for v, var_values in self.pruned_domain.items():
            if len(var_values) > 1:
                tmp_domain = base_domain.copy()  # shallow copy is enough
                consistent_values = []
                for var_value in var_values:
                    tmp_domain[v] = [var_value]
                    if self._backtracking_solver(tmp_domain) is None:
                        log.debug("invalid value %d for %r",
                                  var_value, v)
                    else:
                        consistent_values.append(var_value)
                # Empty the list and then refill it, this doesn't
                # affect existing references to the list.
                del self.pruned_domain[v][:]
                self.pruned_domain[v].extend(consistent_values)
        # If the domain of a variable was reduced to a single value,
        # set it back in the assignment.
        for v, var_values in self.pruned_domain.items():
            assert len(var_values) > 0
            if len(var_values) == 1 and v not in self._assignment:
                self._assignment[v] = self.pruned_domain[v][0]
        log.debug("final assignment:\n%s", pprint.pformat(self._assignment))
        log.debug("final domain:\n%s", pprint.pformat(self.pruned_domain))

    def _compute_is_tree_csp(self):
        is_tree_csp = self._is_binary_csp() and self._has_acyclic_network()
        log.debug("it %s a tree CSP", "is" if is_tree_csp else "isn't")
        return is_tree_csp

    def _is_binary_csp(self):
        for var_names, constraint_fun in self.constraints:
            if len(var_names) != 2:
                return False
        return True

    def _has_acyclic_network(self):
        # _is_binary_csp must be called first.
        network = igraph.Graph(len(self.domain))
        edges = []
        var_index = {v: i for i, v in enumerate(self.domain.keys())}
        for var_names, constraint_fun in self.constraints:
            edges.append([var_index[v] for v in var_names])
        network.add_edges(edges)
        is_acyclic = self._is_acyclic(network)
        return is_acyclic

    @staticmethod
    def _is_acyclic(graph):
        # Check if a given undirected graph contains no cycles.
        assert not graph.is_directed()
        visited = [False] * graph.vcount()
        parent = [None] * graph.vcount()
        while not all(visited):
            stack = [visited.index(False)]
            while stack:
                v = stack.pop()
                visited[v] = True
                for w in graph.vs[v].neighbors():
                    if w.index != parent[v]:
                        if visited[w.index]:
                            return False
                        stack.append(w.index)
                        parent[w.index] = v
        return True
