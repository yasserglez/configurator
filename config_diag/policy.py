"""
Configuration Dialogs Based on Policies
"""

import itertools

import igraph

from . import ConfigDialog, ConfigDialogBuilder
from .mdp import MDP, EpisodicMDP, PolicyIteration, ValueIteration


class PolicyConfigDialog(ConfigDialog):
    """Adaptive configuration dialog based on a policy.
    """

    def __init__(self, policy):
        """Initialize a new instance.

        Arguments:
            policy: The MDP policy, i.e. a dict mapping state indexes
                to action indexes.
        """
        super().__init__()
        self._policy = policy


class MDPDialogBuilder(ConfigDialogBuilder):
    """Adpative configuration dialog builder using MDPs.
    """

    def __init__(self, mdp_algorithm="policy-iteration",
                 mdp_max_iter=1000,
                 mdp_discard_states=None,
                 mdp_partial_assoc_rules=None,
                 mdp_collapse_terminals=None,
                 **kwargs):
        """Initialize a new instance.

        Arguments:
            mdp_algorithm: Algorithm for solving the MDP. Possible
                values are: 'policy-iteration' (default) and
                'value-iteration'.
            mdp_max_iter: The maximum number of iterations of the
                algorithm used to solve the MDP (default: 1000).
            mdp_discard_states: Indicates whether states that can't be
                reached from the initial state after applying the
                association rules should be discarded.
            mdp_partial_assoc_rules: Indicates whether the association
                rules can be applied when some of the variables in the
                right-hand-side are already set to the correct values
                (the opposite is to require that all variables in the
                left-hand-side are unknown).
            mdp_collapse_terminals: Indicates whether all terminal
                states should be collapsed into a single state.

        See ConfigDialogBuilder for the remaining arguments.
        """
        super().__init__(**kwargs)
        if mdp_algorithm == "policy-iteration":
            self._solver = PolicyIteration(max_iter=mdp_max_iter,
                                           eval_max_iter=100)
        elif mdp_algorithm == "value-iteration":
            self._solver = ValueIteration(max_iter=mdp_max_iter)
        else:
            raise ValueError("Invalid mdp_algorithm value")
        self._mdp_discard_states = mdp_discard_states
        self._mdp_partial_assoc_rules = mdp_partial_assoc_rules
        self._mdp_collapse_terminals = mdp_collapse_terminals

    def build_dialog(self):
        """Construct an adaptive configuration dialog.

        Returns:
            A PolicyConfigDialog instance.
        """
        mdp = self._build_mdp()

    def _build_mdp(self):
        # Build the initial graph.
        graph = self._build_graph()
        # Update the graph using the association rules.
        rules = self._mine_rules()
        self._update_graph(graph, rules)

    def _build_graph(self):
        # Build the graph that is used to compute the transition and
        # reward matrices of the MDP. The initial graph built here is
        # updated later using the association rules in _update_graph.
        self._logger.debug("building the initial graph")
        graph = igraph.Graph(directed=True)
        self._add_graph_nodes(graph)
        self._add_graph_edges(graph)
        self._logger.debug("built a graph with %d nodes and %d edges",
                           graph.vcount(), graph.ecount())
        self._logger.debug("finished building the initial graph")
        return graph

    def _add_graph_nodes(self, graph):
        # Add one node for each possible configuration state.
        self._logger.debug("adding nodes")
        config_values = [[None] + values for values in self._config_values]
        num_vars = len(self._config_values)
        for state_values in itertools.product(*config_values):
            # Since None goes first in config_values, the first state
            # will have all the variables set to None (empty dict).
            # It will be the initial state in EpisodicMDP.
            state = {var_index: var_value
                     for var_index, var_value in enumerate(state_values)
                     if var_value is not None}
            if len(state) == num_vars:
                # It's a terminal state. Add the vertex only if
                # terminal states shouldn't be collapsed.
                if not self._mdp_collapse_terminals:
                    graph.add_vertex(state=state)
            else:
                # Intermediate state, add the vertex.
                graph.add_vertex(state=state)
        if self._mdp_collapse_terminals:
            # If the terminal states were collapsed, add a single
            # state (with the state dict set to None) where all the
            # configuration values are known. It will be the terminal
            # state in EpisodicMDP.
            graph.add_vertex(state=None)
        self._logger.debug("finishing adding nodes")

    def _add_graph_edges(self, graph):
        # Add the initial graph edges: loop edges for the known
        # variables and one-step transitions.
        self._logger.debug("adding edges")
        for v in graph.vs:
            # Loop edges:
            self._add_loop_edges(graph, v)
            # One-step transitions:
            criterion = lambda w: self._is_one_step_transition(v, w)
            for w in graph.vs.select(criterion):
                self._add_one_step_edge(graph, v, w)
        self._logger.debug("finished adding edges")

    def _add_loop_edges(self, graph, v):
        # Add loop edges for the known variables.
        if v["state"] is None:
            # Collapsed terminal state. All variables are known.
            known_vars = range(len(self._config_values))
        else:
            # Non-collapsed terminal or intermediate state.
            known_vars = v["state"].keys()
        for var_index in known_vars:
            graph.add_edge(v, v, action=var_index, reward=0, prob=1.0)

    def _is_one_step_transition(self, v, w):
        # Check v and w represent a one-step transition, i.e.
        # transitions from states in which (k - 1) variables are known
        # to states where k variables are known. Additionally, v and w
        # have to differ only in one variable (the one that becomes
        # known after the question is asked).
        num_known_vars_in_v = self._count_known_vars(v)
        num_known_vars_in_w = self._count_known_vars(w)
        if num_known_vars_in_v + 1 == num_known_vars_in_w:
            if w["state"] is None:
                # Collapsed terminal state.
                return True
            else:
                v_set = set(v["state"].items())
                w_set = set(w["state"].items())
                return len(v_set & w_set) == num_known_vars_in_v
        return False

    def _add_one_step_edge(self, graph, v, w):
        # Add the one-step transition edges. Each edge is labelled with:
        # - the action corresponding to the variable that becomes known,
        # - the conditional probability of the variable taking the
        #   value, given the previous user responses, and
        # - a reward of 1 (variables whose values become known).
        num_vars = len(self._config_values)
        v_vars = set(v["state"].keys())
        w_vars = (set(range(num_vars)) if w["state"] is None
                  else set(w["state"].keys()))
        var_index = next(iter(w_vars - v_vars))
        if w["state"] is None:
            # w is a collapsed terminal state. Add a single edge with
            # probability one and reward one. Whatever the user
            # answers is going to take her/him to the terminal state.
            graph.add_edge(v, w, action=var_index, reward=1, prob=1.0)
        else:
            var_value = w["state"][var_index]
            prob = self._cond_prob({var_index: var_value}, v["state"])
            graph.add_edge(v, w, action=var_index,
                           reward=1, prob=prob)

    def _update_graph(self, graph, rules):
        # Update the graph using the association rules.
        self._logger.debug("updating the graph using the association rules")
        inaccessible_vertices = []
        for rule in rules:
            # It doesn't make sense to apply the rule in a terminal
            # state, so the S selection is restricted to non-terminals.
            for v_s in graph.vs.select(self._is_non_terminal):
                s = v_s["state"]  # S
                if not (v_s.index not in inaccessible_vertices and
                        self._update_graph_cond_a(rule, s) and
                        self._update_graph_cond_bii(rule, s)):
                    continue  # skip this vertex
                for v_sp in graph.vs:
                    sp = v_sp["state"]  # S'
                    if not (v_sp.index not in inaccessible_vertices and
                            self._update_graph_cond_a(rule, sp) and
                            self._update_graph_cond_bi(rule, sp) and
                            self._update_graph_cond_c(rule, s, sp)):
                        continue  # skip this vertex
                    # If we've reached this far, the rule can be used to
                    # "shortcut" from S to S'. After creating the "shortcut",
                    # S becomes inaccessible and it's added to a list of
                    # vertices to be deleted at the end.
                    self._update_graph_shortcut(graph, v_s, v_sp)
                    inaccessible_vertices.append(v_s.index)
                    break  # a match for S was found, don't keep looking
        # Delete the inaccesible vertices.
        if self._mdp_discard_states:
            graph.delete_vertices(inaccessible_vertices)
        self._logger.debug("found %d applications of the %d rules",
                           len(inaccessible_vertices), len(rules))
        self._logger.debug("turned into a graph with %d nodes and %d edges",
                           graph.vcount(), graph.ecount())
        self._logger.debug("finished updating the graph")

    def _update_graph_cond_a(self, rule, s_or_sp):
        # (a) variables in lhs have known and same options in S and S'.
        # When called with S', it could be a collapsed terminal state
        # which will satisfy any rule.
        return (s_or_sp is None or self._is_rule_lhs_compatible(rule, s_or_sp))

    def _update_graph_cond_bi(self, rule, sp):
        # (b-i) variables in the rhs appear with all values exactly
        # the same as in the rule in S'. The collapsed terminal state
        # satisfies any rule.
        if sp is not None:
            for var_index, var_value in rule.rhs.items():
                if var_index not in sp or sp[var_index] != var_value:
                    return False
        return True

    def _update_graph_cond_bii(self, rule, s):
        if self._mdp_partial_assoc_rules:
            # (b-ii) variables in the rhs appear with the same values
            # or set to unknown in S. At least one must be set to
            # unknown in S.
            return self._is_rule_rhs_compatible(rule, s)
        else:
            # (b-ii) variables in the rhs appear with all values set
            # to unknown in S.
            return all((var_index not in s for var_index in rule.rhs.keys()))

    def _update_graph_cond_c(self, rule, s, sp):
        # (c) all other variables not mentioned in the rule are set to
        # the same values for both S and S'.
        all_vars = set(range(len(self._config_values)))
        mentioned_vars = set(rule.lhs.keys()) | set(rule.rhs.keys())
        if sp is None:
            # S' is a collapsed terminal state and it has all the
            # variable combinations. We only need to check for S.
            for var_index in (all_vars - mentioned_vars):
                if var_index not in s:
                    return False
        else:
            for var_index in (all_vars - mentioned_vars):
                if not (var_index in s and
                        var_index in sp and
                        s[var_index] == sp[var_index]):
                    return False
        return True

    def _update_graph_shortcut(self, graph, v_s, v_sp):
        # Create a "shortcut" from S to S'. Take all the edges that
        # are targeting S and move them so that they now target S'.
        # The reward of those edges is incremented by the difference
        # in the number of known variables between S' and S.
        old_edges = []
        for e in graph.es.select(_target=v_s.index):
            reward = e["reward"] + (self._count_known_vars(v_sp) -
                                    self._count_known_vars(v_s))
            graph.add_edge(e.source, target=v_sp, action=e["action"],
                           prob=e["prob"], reward=reward)
            old_edges.append(e.index)
        graph.delete_edges(old_edges)

    def _count_known_vars(self, v):
        # Return the number of variables whose value is known.
        num_known_vars = (len(self._config_values) if v["state"] is None
                          else len(v["state"]))
        return num_known_vars

    def _is_terminal(self, v):
        # Check if v representes a terminal state.
        return self._count_known_vars(v) == len(self._config_values)

    def _is_non_terminal(self, v):
        # Check if v represents a terminal state.
        return not self._is_terminal(v)
