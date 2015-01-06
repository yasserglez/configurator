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
                 mdp_collapse_terminals=None,
                 **kwargs):
        """Initialize a new instance.

        Arguments:
            mdp_algorithm: Algorithm for solving the MDP. Possible
                values are: 'policy-iteration' (default) and
                'value-iteration'.
            mdp_max_iter: The maximum number of iterations of the
                algorithm used to solve the MDP (default: 1000).
            mdp_collapse_terminals: Indicates whether all terminal
                states (all configuration variables are known) should
                be collapsed into a single state.

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
        rules = self._mine_assoc_rules()
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
        num_vars = len(config_values)
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
            # Regular, intermediate state.
            known_vars = v["state"].keys()
        for var_index in known_vars:
            graph.add_edge(v, v, action=var_index, reward=0, prob=1.0)

    def _is_one_step_transition(self, v, w):
        # Check v and w represent a one-step transition, i.e.
        # transitions from states in which (k - 1) variables are known
        # to states where k variables are known. Additionally, v and w
        # have to differ only in one variable (the one that becomes
        # known after the question is asked).
        num_vars = len(self._config_values)
        state_len = lambda vertex: (num_vars if vertex["state"] is None
                                    else len(vertex["state"]))
        v_state_len, w_state_len = state_len(v), state_len(w)
        if v_state_len + 1 == w_state_len:
            v_set = set(v["state"].items())
            if w["state"] is None:
                # Collapsed terminal state.
                return True
            else:
                w_set = set(w["state"].items())
                return len(v_set & w_set) == v_state_len
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
        self._logger.debug("finished updating the graph")
