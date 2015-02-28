"""Constraint satisfaction problems.
"""

import copy
import pprint
import logging

import igraph


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
        solution = self._backtracking_solver(self.var_domains)
        return solution

    def _backtracking_solver(self, var_domains):
        # Backtracking solver maintaining arc consistency (using the
        # AC-3 algorithm), with the minimum-remaining-values heuristic
        # for variable selection and the least-constraining-value
        # heuristic for value selection.
        vars = list(range(len(self.var_domains)))
        solution = backtrack(vars, var_domains, self.constraints)
        return solution

    # The following are internal methods used to keep track of the
    # assignments and enforce global consistency after each variable
    # is assigned when the dialog is being used.

    def reset(self):
        self.assignment = {}
        self.pruned_var_domains = copy.deepcopy(self.var_domains)

    def assign_variable(self, var_index, var_value, prune_var_domains=True):
        if var_index in self.assignment:
            raise ValueError("The variable is already assigned")
        if var_value not in self.pruned_var_domains[var_index]:
            raise ValueError("Invalid assignment in the current state")
        log.debug("assignning variable %d to %r", var_index, var_value)
        log.debug("initial assignment:\n%s", pprint.pformat(self.assignment))
        self.assignment[var_index] = var_value
        if len(self.assignment) < len(self.var_domains) and prune_var_domains:
            self.prune_var_domains()
            # If the domain of a variable was reduced to a single
            # value, set it back in the assignment.
            for var_index, var_values in enumerate(self.pruned_var_domains):
                assert len(var_values) > 0
                if len(var_values) == 1 and var_index not in self.assignment:
                    var_value = self.pruned_var_domains[var_index][0]
                    self.assignment[var_index] = var_value
        log.debug("final assignment:\n%s", pprint.pformat(self.assignment))

    def prune_var_domains(self):
        log.debug("pruning the variable domains")
        log.debug("initial domains:\n%s",
                  pprint.pformat(self.pruned_var_domains))
        # Enforce unary constraints.
        for var_index, var_value in self.assignment.items():
            self.pruned_var_domains[var_index] = [var_value]
        log.debug("after enforcing unary constraints:\n%s",
                  pprint.pformat(self.pruned_var_domains))
        log.debug("enforcing global consistency")
        if self.is_tree_csp:
            # Arc consistency is equivalent to global consistency in
            # normalized, tree-structured, binary CSPs.
            log.debug("it's a tree CSP, enforcing arc consistency")
            arc_consistency_3(self.pruned_var_domains, self.constraints)
        else:
            self._enforce_global_consistency()
        log.debug("finished enforcing global consistency")
        log.debug("finished pruning the variable domains")
        log.debug("final domains:\n%s",
                  pprint.pformat(self.pruned_var_domains))

    def _enforce_global_consistency(self):
        # Check that all possible answers for the next question lead
        # to a consistent assignment.
        for var_index, var_values in enumerate(self.pruned_var_domains):
            if len(var_values) > 1:
                tmp_var_domains = self.pruned_var_domains.copy()
                consistent_values = []
                for var_value in var_values:
                    tmp_var_domains[var_index] = [var_value]
                    if self._backtracking_solver(tmp_var_domains) is None:
                        log.debug("invalid value %r for %d",
                                  var_value, var_index)
                    else:
                        consistent_values.append(var_value)
                self.pruned_var_domains[var_index] = consistent_values

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
        network = igraph.Graph(len(self.var_domains))
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


# Portions of the following code were derived from the
# https://github.com/simpleai-team/simpleai.


def backtrack(variables, domains, constraints):
    '''
    Backtracking search.
    variable_heuristic is the heuristic for variable choosing, can be
    MOST_CONSTRAINED_VARIABLE, HIGHEST_DEGREE_VARIABLE, or blank for simple
    ordered choosing.
    value_heuristic is the heuristic for value choosing, can be
    LEAST_CONSTRAINING_VALUE or blank for simple ordered choosing.
    '''
    assignment = {}
    domains = copy.deepcopy(domains)
    return _backtracking(variables, constraints,
                         assignment,
                         domains)


def _backtracking(variables, constraints, assignment, domains):
    '''
    Internal recursive backtracking algorithm.
    '''
    if len(assignment) == len(variables):
        return assignment

    variable_chooser = _most_constrained_variable_chooser
    values_sorter = _least_constraining_values_sorter

    pending = [v for v in variables
               if v not in assignment]
    variable = variable_chooser(pending, domains)

    values = values_sorter(constraints, assignment, variable, domains)

    for value in values:
        new_assignment = copy.deepcopy(assignment)
        new_assignment[variable] = value

        if not _count_conflicts(constraints, new_assignment):
            new_domains = copy.deepcopy(domains)
            new_domains[variable] = [value]

            if arc_consistency_3(new_domains, constraints):
                result = _backtracking(variables, constraints,
                                       new_assignment,
                                       new_domains)
                if result:
                    return result

    return None



def _most_constrained_variable_chooser(variables, domains):
    '''
    Choose the variable that has less available values.
    '''
    # the variable with fewer values available
    return sorted(variables, key=lambda v: len(domains[v]))[0]


def _least_constraining_values_sorter(constraints, assignment, variable, domains):
    '''
    Sort values based on how many conflicts they generate if assigned.
    '''
    values = sorted(domains[variable][:],
                    key=lambda v: _count_conflicts(constraints, assignment,
                                                   variable, v))
    return values


def _count_conflicts(constraints, assignment, variable=None, value=None):
    '''
    Count the number of violated constraints on a given assignment.
    '''
    return len(_find_conflicts(constraints, assignment, variable, value))


def _call_constraint(assignment, neighbors, constraint):
    variables, values = zip(*[(n, assignment[n])
                              for n in neighbors])
    return constraint(variables, values)


def _find_conflicts(constraints, assignment, variable=None, value=None):
    '''
    Find violated constraints on a given assignment, with the possibility
    of specifying a new variable and value to add to the assignment before
    checking.
    '''
    if variable is not None and value is not None:
        assignment = copy.deepcopy(assignment)
        assignment[variable] = value

    conflicts = []
    for neighbors, constraint in constraints:
        # if all the neighbors on the constraint have values, check if conflict
        if all(n in assignment for n in neighbors):
            if not _call_constraint(assignment, neighbors, constraint):
                conflicts.append((neighbors, constraint))

    return conflicts


def revise(domains, arc, constraints):
    """
    Given the arc X, Y (variables), removes the values from X's domain that
    do not meet the constraint between X and Y.

    That is, given x1 in X's domain, x1 will be removed from the domain, if
    there is no value y in Y's domain that makes constraint(X,Y) True, for
    those constraints affecting X and Y.
    """
    x, y = arc
    related_constraints = [(neighbors, constraint)
                           for neighbors, constraint in constraints
                           if set(arc) == set(neighbors)]

    modified = False

    for neighbors, constraint in related_constraints:
        for x_value in domains[x]:
            constraint_results = (_call_constraint({x: x_value, y: y_value},
                                                   neighbors, constraint)
                                  for y_value in domains[y])

            if not any(constraint_results):
                domains[x].remove(x_value)
                modified = True

    return modified


def all_arcs(constraints):
    """
    For each constraint ((X, Y), const) adds:
        ((X, Y), const)
        ((Y, X), const)
    """
    arcs = set()

    for neighbors, constraint in constraints:
        if len(neighbors) == 2:
            x, y = neighbors
            list(map(arcs.add, ((x, y), (y, x))))

    return arcs


def arc_consistency_3(domains, constraints):
    """
    Makes a CSP problem arc consistent.

    Ignores any constraint that is not binary.
    """
    arcs = list(all_arcs(constraints))
    pending_arcs = set(arcs)

    while pending_arcs:
        x, y = pending_arcs.pop()
        if revise(domains, (x, y), constraints):
            if len(domains[x]) == 0:
                return False
            pending_arcs = pending_arcs.union((x2, y2) for x2, y2 in arcs
                                              if y2 == x)
    return True
