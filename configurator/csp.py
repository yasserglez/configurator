"""Constraint satisfaction problems.
"""

import copy
import pprint
import logging

import igraph
from simpleai.search import CspProblem
from simpleai.search.arc import arc_consistency_3
from simpleai.search.csp import (backtrack, MOST_CONSTRAINED_VARIABLE,
                                 LEAST_CONSTRAINING_VALUE)


log = logging.getLogger(__name__)


class CSP(object):
    """Finite-domain constraint satisfaction problem.

    Arguments:
        domain: A list with one entry for each variable containing a
            sequence with all the possible values of the variable.
            All the variables must be domain-consistent (i.e. there
            must exist at least one consistent configuration in which
            each value value occurs).
        constraints: A list of tuples with two components each: i) a
            tuple with the indices of the variables involved in the
            constraint, and ii) a function that checks the constraint.
            The constraint functions will receive a variables tuple
            and a values tuple, both containing only the restricted
            variable indices and their values (in the same order
            provided in `constraints`). The function should return
            `True` if the values satisfy the constraint, `False`
            otherwise. The constraints must be normalized (i.e. two
            different constraints shouldn't involve the same set of
            variables).

    All the arguments are available as instance attributes.
    """

    def __init__(self, domain, constraints):
        self.domain = [list(var_domain) for var_domain in domain]
        self._variables = list(range(len(self.domain)))  # cached for simpleai
        # Ensure that the constraints are normalized.
        constraint_support = set()
        for var_indices, constraint_fun in constraints:
            var_indices = frozenset(var_indices)
            if var_indices in constraint_support:
                raise ValueError("The constraints must be normalized")
            constraint_support.add(var_indices)
        self.constraints = constraints
        self.is_tree_csp = self._compute_is_tree_csp()
        self.reset()

    def solve(self):
        """Find a consistent assignment of the variables.

        Returns:
            A dictionary with the values assigned to the variables if
            a solution was found, `None` otherwise.
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
        self.assignment = {}
        self.pruned_domain = copy.deepcopy(self.domain)

    def assign_variable(self, var_index, var_value, prune_domain=True):
        if var_index in self.assignment:
            raise ValueError("The variable is already assigned")
        if var_value not in self.pruned_domain[var_index]:
            raise ValueError("Invalid assignment in the current state")
        log.debug("assignning variable %d to %r", var_index, var_value)
        log.debug("initial assignment:\n%s", pprint.pformat(self.assignment))
        self.assignment[var_index] = var_value
        if prune_domain:
            self.prune_domain()
            # If the domain of a variable was reduced to a single
            # value, set it back in the assignment.
            for var_index, var_values in enumerate(self.pruned_domain):
                assert len(var_values) > 0
                if len(var_values) == 1 and var_index not in self.assignment:
                    var_value = self.pruned_domain[var_index][0]
                    self.assignment[var_index] = var_value
        log.debug("final assignment:\n%s", pprint.pformat(self.assignment))

    def prune_domain(self):
        log.debug("pruning the domain")
        log.debug("initial domain:\n%s", pprint.pformat(self.pruned_domain))
        # Enforce unary constraints.
        for var_index, var_value in self.assignment.items():
            self.pruned_domain[var_index] = [var_value]
        log.debug("after enforcing unary constraints:\n%s",
                  pprint.pformat(self.pruned_domain))
        log.debug("enforcing global consistency")
        if self.is_tree_csp:
            # Arc consistency is equivalent to global consistency in
            # normalized, tree-structured, binary CSPs.
            log.debug("it's a tree CSP, enforcing arc consistency")
            arc_consistency_3(self.pruned_domain, self.constraints)
        else:
            self._enforce_global_consistency()
        log.debug("finished enforcing global consistency")
        log.debug("finished pruning the domain")
        log.debug("final domain:\n%s", pprint.pformat(self.pruned_domain))

    def _enforce_global_consistency(self):
        # Check that all possible answers for the next question lead
        # to a consistent assignment.
        for var_index, var_values in enumerate(self.pruned_domain):
            if len(var_values) > 1:
                tmp_domain = self.pruned_domain.copy()
                consistent_values = []
                for var_value in var_values:
                    tmp_domain[var_index] = [var_value]
                    if self._backtracking_solver(tmp_domain) is None:
                        log.debug("invalid value %r for %d",
                                  var_value, var_index)
                    else:
                        consistent_values.append(var_value)
                self.pruned_domain[var_index] = consistent_values

    def _compute_is_tree_csp(self):
        is_tree_csp = self._is_binary_csp() and self._has_acyclic_network()
        log.debug("it %s a tree CSP", "is" if is_tree_csp else "isn't")
        return is_tree_csp

    def _is_binary_csp(self):
        for var_indices, _ in self.constraints:
            if len(var_indices) != 2:
                return False
        return True

    def _has_acyclic_network(self):
        # _is_binary_csp must be called first.
        network = igraph.Graph(len(self.domain))
        edges = []
        for var_indices, _ in self.constraints:
            edges.append([var_index for var_index in var_indices])
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
