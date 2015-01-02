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
            self._solver = PolicyIteration(max_iter=mdp_max_iter)
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
        # Add one node for each possible configuration state.
        config_values = [[None] + values for values in self._config_values]
        for state_values in itertools.product(*config_values):
            # Since None goes first in config_values, the first state
            # will have all the variables set to None (empty dict).
            state = {var_index: var_value
                     for var_index, var_value in enumerate(state_values)
                     if var_value is not None}
            # If the flag is set, do not add multiple terminal states.
            if not (self._mdp_collapse_terminals and
                    len(state) == len(self._config_values)):
                graph.add_vertex(state=state)
        if self._mdp_collapse_terminals:
            # If the terminal states were collapsed, add a single
            # state where all the configuration values are known.
            graph.add_vertex(state=None)
        self._logger.debug("created a graph with %d nodes", graph.vcount())
        self._logger.debug("finished building the initial graph")
        return graph

    def _update_graph(self, graph, rules):
        # Update the graph using the association rules.
        self._logger.debug("updating the graph using the association rules")
        self._logger.debug("finished updating the graph")
