"""Constraint satisfaction problems.
"""

import pprint
import logging


log = logging.getLogger(__name__)


class CSP(object):
    """Finite-domain constraint satisfaction problem.

    Arguments:
        var_domains: A list with one entry for each variable
            containing a sequence with all the possible values of the
            variable. All the variables must be domain-consistent
            (i.e. there must exist at least one consistent
            configuration in which each value value occurs).
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
        self.is_binary = True
        self._constraints_index = {}
        for var_indices, constraint_fun in constraints:
            if len(var_indices) != 2:
                self.is_binary = False
            key = frozenset(var_indices)
            if key in self._constraints_index:
                raise ValueError("The constraints must be normalized")
            self._constraints_index[key] = var_indices, constraint_fun
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

    def reset(self):
        self.assignment = {}
        self.pruned_var_domains = self.var_domains.copy()

    def assign_variable(self, var_index, var_value, consistency="global"):
        if var_index in self.assignment:
            raise ValueError("The variable is already assigned")
        if var_value not in self.pruned_var_domains[var_index]:
            raise ValueError("Invalid assignment in the current state")
        log.debug("assignning variable %d to %r", var_index, var_value)
        log.debug("initial assignment:\n%s", pprint.pformat(self.assignment))
        log.debug("initial domains:\n%s",
                  pprint.pformat(self.pruned_var_domains))
        self.assignment[var_index] = var_value
        self.pruned_var_domains[var_index] = [var_value]
        if len(self.assignment) < len(self.var_domains):
            if consistency == "global":
                self.enforce_global_consistency()
            elif consistency == "local":
                if not self.is_binary:
                    raise ValueError("Local consistency only " +
                                     "implemented for binary CSPs")
                self.enforce_local_consistency()
            # If the domain of a variable was reduced to a single
            # value, set it back in the assignment.
            for var_index, var_values in enumerate(self.pruned_var_domains):
                assert len(var_values) > 0
                if len(var_values) == 1 and var_index not in self.assignment:
                    var_value = self.pruned_var_domains[var_index][0]
                    self.assignment[var_index] = var_value
        log.debug("final assignment:\n%s", pprint.pformat(self.assignment))
        log.debug("final domains:\n%s",
                  pprint.pformat(self.pruned_var_domains))

    def enforce_global_consistency(self):
        # Check that all possible answers to the remaining questions
        # lead to at least one consistent assignment.
        log.debug("enforcing global consistency")
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

    def enforce_local_consistency(self):
        # Singleton arc consistency. Based on Figure 2 of Romuald
        # Debruyne and Christian Bessiere (1997). Some practicable
        # filtering techniques for the constraint satisfaction
        # problem, IJCAI'97, 412-417.
        log.debug("enforcing local consistency")
        _arc_consistency_3(self.pruned_var_domains,
                           self._constraints_index)
        changed = True
        while changed:
            changed = False
            for var_index in range(len(self.var_domains)):
                if len(self.pruned_var_domains[var_index]) > 1:
                    consistent_values = []
                    for var_value in self.pruned_var_domains[var_index]:
                        tmp_var_domains = self.pruned_var_domains.copy()
                        tmp_var_domains[var_index] = [var_value]
                        if _arc_consistency_3(tmp_var_domains,
                                              self._constraints_index):
                            consistent_values.append(var_value)
                    if (len(consistent_values) <
                            len(self.pruned_var_domains[var_index])):
                        self.pruned_var_domains[var_index] = consistent_values
                        # Ensuring arc consistency after all values
                        # are checked (instead of doing it when an
                        # inconsistent value is detected) in order to
                        # avoid changing self.pruned_var_domains[var_index]
                        # while it's being iterated.
                        _arc_consistency_3(self.pruned_var_domains,
                                           self._constraints_index)
                        changed = True


# Backtracking search maintaining arc consistency (using AC-3), with
# minimum-remaining-values heuristic for variable selection.
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


def _most_constrained_var(unassigned_vars, domains):
    # Choose the variable with fewer values available.
    return min(unassigned_vars, key=lambda v: len(domains[v]))


def _has_conflicts(assignment, constraints_index):
    # Check if the given assignment generates at least one conflict.
    for var_indices, constraint_fun in constraints_index.values():
        if all(v in assignment for v in var_indices):
            if not _call_constraint(assignment, var_indices, constraint_fun):
                return True
    return False


def _call_constraint(assignment, var_indices, constraint_fun):
    var_values = [assignment[v] for v in var_indices]
    return constraint_fun(var_indices, var_values)


def _remove_inconsistent_values(var_domains, arc, constraints_index):
    # Given the arc (x, y), remove the values from X's domain that
    # don't meet the constraint between X and Y.
    xi, xj = arc
    var_indices, constraint_fun = constraints_index[frozenset(arc)]
    assignment = {}
    consistent_values = []
    for xi_value in var_domains[xi]:
        assignment[xi] = xi_value
        for xj_value in var_domains[xj]:
            assignment[xj] = xj_value
            if _call_constraint(assignment, var_indices, constraint_fun):
                consistent_values.append(xi_value)
                break  # We are looking for at least one match.
    removed = len(consistent_values) < len(var_domains[xi])
    if removed:
        var_domains[xi] = consistent_values
    return removed


def _arc_consistency_3(var_domains, constraints_index):
    # The arc-consistency algorithm AC3.
    all_arcs = set()
    for var_indices, constraint_fun in constraints_index.values():
        # Non-binary constraints are ignored.
        if len(var_indices) == 2:
            xi, xj = var_indices
            all_arcs.add((xi, xj))
            all_arcs.add((xj, xi))
    pending_arcs = all_arcs.copy()
    while pending_arcs:
        arc = pending_arcs.pop()
        xi, xj = arc
        if _remove_inconsistent_values(var_domains, arc, constraints_index):
            if len(var_domains[xi]) == 0:
                return False
            pending_arcs.update((xk, xi) for xk, y in all_arcs if y == xi)
    return True
