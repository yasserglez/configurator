"""Constraint satisfaction problems.
"""

import pprint
import logging
import itertools

import igraph


log = logging.getLogger(__name__)


class CSP(object):
    """Finite-domain constraint satisfaction problem.

    Arguments:
        var_domains: A list with one entry for each variable
            containing a sequence with all the possible values of the
            variable. All the variables must be domain-consistent.
        constraints: A list of tuples with two components each: i) a
            tuple with the indices of the variables involved in the
            constraint, and ii) a function that checks the constraint.
            The constraint functions will receive an indices tuple and
            a values tuple, both containing only the restricted
            variable indices and their values (in the same order
            provided in `constraints`). The function should return
            `True` if the values satisfy the constraint, `False`
            otherwise. The constraints must be normalized (i.e. two
            different constraints shouldn't involve the same set of
            variables).

    All the arguments are available as instance attributes.
    """

    def __init__(self, var_domains, constraints):
        self.var_domains = [list(var_domain) for var_domain in var_domains]
        self.constraints = constraints
        self._is_binary = True
        self._constraints_index = {}
        for constraint_vars, constraint_fun in constraints:
            if len(constraint_vars) != 2:
                self._is_binary = False
            constraint_key = frozenset(constraint_vars)
            if constraint_key in self._constraints_index:
                raise ValueError("The constraints must be normalized")
            self._constraints_index[constraint_key] = constraint_vars, constraint_fun
        self._is_tree_csp = self._compute_is_tree_csp()
        log.debug("it %s a tree CSP", "is" if self._is_tree_csp else "isn't")
        self.reset()

    def solve(self):
        """Find a consistent assignment of the variables.

        Returns:
            A dictionary with the values assigned to the variables if
            a solution was found, `None` otherwise.
        """
        return _backtracking_search({}, self.var_domains,
                                    self._constraints_index)

    # The following are internal methods used to keep track of the
    # assignments and enforce global or local consistency after each
    # variable is assigned when the dialog is being used.

    def _compute_is_tree_csp(self):
        if not self._is_binary:
            return False
        network = igraph.Graph(len(self.var_domains))
        edges = []
        for constraint_vars, constraint_fun in self._constraints_index.values():
            edges.append(constraint_vars)
        network.add_edges(edges)
        return self._is_acyclic(network)

    @staticmethod
    def _is_acyclic(graph):
        # Run DFS to look for back edges.
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

    def reset(self):
        self.assignment = {}
        self.pruned_var_domains = self.var_domains.copy()

    def assign_variable(self, var_index, var_value, consistency="global"):
        if var_index in self.assignment:
            raise ValueError("The variable is already assigned")
        if var_value not in self.pruned_var_domains[var_index]:
            raise ValueError("Invalid assignment in the current state")
        if log.isEnabledFor(logging.DEBUG):
            log.debug("assigning variable %d to %r", var_index, var_value)
            log.debug("initial assignment:\n%s",
                      pprint.pformat(self.assignment))
            log.debug("initial domains:\n%s",
                      pprint.pformat(self.pruned_var_domains))
        self.assignment[var_index] = var_value
        self.pruned_var_domains[var_index] = [var_value]
        if len(self.assignment) < len(self.var_domains):
            if consistency == "global":
                log.debug("enforcing global consistency")
                if self._is_tree_csp:
                    # Arc consistency implies global consistency in an
                    # acyclic network of binary constraints.
                    _arc_consistency_3(self.pruned_var_domains,
                                       self._constraints_index)
                else:
                    self.enforce_global_consistency()
            elif consistency == "local":
                log.debug("enforcing local consistency")
                _arc_consistency_3(self.pruned_var_domains,
                                   self._constraints_index)
            # If the domain of a variable was reduced to a single
            # value, set it back in the assignment.
            for var_index, var_values in enumerate(self.pruned_var_domains):
                if len(var_values) == 1 and var_index not in self.assignment:
                    var_value = self.pruned_var_domains[var_index][0]
                    self.assignment[var_index] = var_value
        if log.isEnabledFor(logging.DEBUG):
            log.debug("final assignment:\n%s",
                      pprint.pformat(self.assignment))
            log.debug("final domains:\n%s",
                      pprint.pformat(self.pruned_var_domains))

    def enforce_global_consistency(self):
        # Check that all possible answers to the remaining questions
        # lead to at least one consistent assignment.
        _arc_consistency_3(self.pruned_var_domains,
                           self._constraints_index)
        for var_index in range(len(self.var_domains)):
            if len(self.pruned_var_domains[var_index]) > 1:
                consistent_values = []
                for var_value in self.pruned_var_domains[var_index]:
                    tmp_var_domains = self.pruned_var_domains.copy()
                    tmp_var_domains[var_index] = [var_value]
                    if _backtracking_search({}, tmp_var_domains,
                                            self._constraints_index):
                        consistent_values.append(var_value)
                if (len(consistent_values) <
                        len(self.pruned_var_domains[var_index])):
                    self.pruned_var_domains[var_index] = consistent_values
                    _arc_consistency_3(self.pruned_var_domains,
                                       self._constraints_index)


# Backtracking search maintaining arc consistency (using GAC-3), with
# the minimum-remaining-values heuristic for variable selection.
def _backtracking_search(assignment, var_domains, constraints_index):
    if len(assignment) == len(var_domains):
        return assignment
    unassigned_vars = [v for v in range(len(var_domains))
                       if v not in assignment]
    var_index = _most_constrained_var(unassigned_vars, var_domains)
    for var_value in var_domains[var_index]:
        new_assignment = assignment.copy()
        new_assignment[var_index] = var_value
        if not _has_conflicts(new_assignment, constraints_index):
            new_var_domains = var_domains.copy()
            new_var_domains[var_index] = [var_value]
            if _arc_consistency_3(new_var_domains, constraints_index):
                solution = _backtracking_search(new_assignment,
                                                new_var_domains,
                                                constraints_index)
                if solution:
                    return solution
    return None


def _most_constrained_var(unassigned_vars, var_domains):
    # Choose the variable with fewer values available.
    return min(unassigned_vars, key=lambda v: len(var_domains[v]))


def _has_conflicts(assignment, constraints_index):
    # Check if the given assignment generates at least one conflict.
    for constraint_vars, constraint_fun in constraints_index.values():
        if all(v in assignment for v in constraint_vars):
            if not _call_constraint(assignment, constraint_vars, constraint_fun):
                return True
    return False


def _call_constraint(assignment, constraint_vars, constraint_fun):
    var_values = [assignment[v] for v in constraint_vars]
    return constraint_fun(constraint_vars, var_values)


def _revise(var_domains, var_index, constraint_vars, constraint_fun):
    # Remove the (locally) inconsistent values for the domain of var_index.
    other_var_indices, other_var_domains = [], []
    for constraint_var_index in constraint_vars:
        if constraint_var_index != var_index:
            other_var_indices.append(constraint_var_index)
            other_var_domains.append(var_domains[constraint_var_index])
    # For each possible value of var_index, check that there is at
    # least one consistent assignment of the rest of the variables
    # (other_var_indices) that participate in the constraint.
    revised_domain = []
    assignment = {}
    for var_value in var_domains[var_index]:
        # Set the variable to the current value in the assignment.
        assignment[var_index] = var_value
        # Complete the assignment with every possible assignment of
        # the remaining variables that participate in the constraint.
        for other_var_values in itertools.product(*other_var_domains):
            for other_var_index, other_var_value in \
                    zip(other_var_indices, other_var_values):
                assignment[other_var_index] = other_var_value
            if _call_constraint(assignment, constraint_vars, constraint_fun):
                revised_domain.append(var_value)
                break  # found one match, move on with the next value
    changed = len(revised_domain) != len(var_domains[var_index])
    if changed:
        var_domains[var_index] = revised_domain
    return changed


def _arc_consistency_3(var_domains, constraints_index):
    # (Generalized) arc consistency using AC-3.
    pending_arcs = set()
    for constraint_key in constraints_index.keys():
        for var_index in constraint_key:
            pending_arcs.add((var_index, constraint_key))
    while pending_arcs:
        var_index, constraint_key = pending_arcs.pop()
        if _revise(var_domains, var_index, *constraints_index[constraint_key]):
            if len(var_domains[var_index]) == 0:  # domain wipe-out
                return False
            # We must check any other constraint where var_index participates.
            for other_constraint_key in constraints_index.keys():
                if (other_constraint_key != constraint_key and
                        var_index in other_constraint_key):
                    for other_var_index in other_constraint_key:
                        if other_var_index != var_index:
                            pending_arcs.add((other_var_index, other_constraint_key))
    return True
