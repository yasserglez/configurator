"""
Configurators Based on Policies
"""

import itertools

import igraph
from scipy import sparse

from . import Configurator, TrivialConfigurator, ConfiguratorBuilder
from .mdp import MDP, EpisodicMDP, PolicyIteration, ValueIteration


class PolicyConfigurator(Configurator):
    """Configurator based on a policy.

    Attributes:
        rules: A list of AssociationRule instances.
        policy: The MDP policy, i.e. a dict mapping every possible
            configuration state to a variable index.

    See Configurator for other attributes.
    """

    def __init__(self, config_values, rules, policy):
        """Initialize a new instance.

        Arguments:
            rules: A list of AssociationRule instances.
            policy: The MDP policy, i.e. a dict mapping configuration
                states to variable indices. The configuration states
                are represented as frozensets of (index, value) tuples
                for each variable.

        See Configurator for the remaining arguments.
        """
        super().__init__(config_values)
        self.rules = rules
        self.policy = policy

    def set_answer(self, var_index, var_value):
        """Set the value of a configuration variable.

        See Configurator for more information.
        """
        super().set_answer(var_index, var_value)
        for rule in self.rules:
            if rule.is_applicable(self.config):
                rule.apply_rule(self.config)

    def get_next_question(self):
        """Get the question that should be asked next.
        """
        next_var_index = self.policy[frozenset(self.config.items())]
        return next_var_index


class MDPConfiguratorBuilder(ConfiguratorBuilder):
    """Configurator builder using MDPs.
    """

    def __init__(self, mdp_algorithm="policy-iteration",
                 mdp_max_iter=1000,
                 mdp_discard_states=True,
                 mdp_partial_assoc_rules=True,
                 mdp_collapse_terminals=True,
                 mdp_validate=False,
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
                association rules should be discarded (default: True).
            mdp_partial_assoc_rules: Indicates whether the association
                rules can be applied when some of the variables in the
                right-hand-side are already set to the correct values.
                (the opposite is to require that all variables in the
                left-hand-side are unknown) (default: True).
            mdp_collapse_terminals: Indicates whether all terminal
                states should be collapsed into a single state
                (default: True).
            mdp_validate: Indicates whether the resulting MDP
                transition and reward matrices should be validated
                (default: False).

        See ConfiguratorBuilder for the remaining arguments.
        """
        super().__init__(**kwargs)
        if mdp_algorithm == "policy-iteration":
            self._solver = PolicyIteration(max_iter=mdp_max_iter)
        elif mdp_algorithm == "value-iteration":
            self._solver = ValueIteration(max_iter=mdp_max_iter)
        else:
            raise ValueError("Invalid mdp_algorithm value")
        self._mdp_discard_states = mdp_discard_states
        self._mdp_partial_assoc_rules = mdp_partial_assoc_rules
        self._mdp_collapse_terminals = mdp_collapse_terminals
        self._mdp_validate = mdp_validate

    def build_configurator(self):
        """Construct a configurator.

        Returns:
            A PolicyConfigurator instance if at least one association
            rule is discovered, otherwise a TrivialConfigurator instance.
        """
        self._logger.debug("building the MDP")
        # Build the initial graph.
        graph = self._build_graph()
        # Update the graph using the association rules.
        rules = self._mine_rules()
        if rules:
            self._update_graph(graph, rules)
            # Transform the graph into MDP components.
            mdp = self._transform_graph_to_mdp(graph)
            self._logger.debug("finished building the MDP")
            self._logger.debug("solving the MDP")
            policy = self._solver.solve(mdp)
            self._logger.debug("finished solving the MDP")
            # Create the PolicyConfigurator instance.
            policy = {frozenset(graph.vs[s]["state"].items()): a
                      for s, a in policy.items()}
            configurator = PolicyConfigurator(self._config_values,
                                              rules, policy)
        else:
            # There are no rules. Build a trivial configurator.
            configurator = TrivialConfigurator(self._config_values)
        return configurator

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

    def _transform_graph_to_mdp(self, graph):
        # Generate the transition and reward matrices from the graph.
        self._logger.debug("transforming the graph into the MDP")
        S, A = graph.vcount(), len(self._config_values)
        self._logger.debug("the MDP has %d states and %d actions", S, A)
        transitions = [sparse.lil_matrix((S, S)) for a in range(A)]
        rewards = [sparse.lil_matrix((S, S)) for a in range(A)]
        for e in graph.es:
            a, i, j = e["action"], e.source, e.target
            if e["prob"] > 0:
                transitions[a][i, j] = e["prob"]
            if e["reward"] != 0:
                rewards[a][i, j] = e["reward"]
        # pymdptoolbox seems to work only with csr_matrix, but
        # lil_matrix matrices can be built faster.
        transitions = list(map(lambda m: m.tocsr(), transitions))
        rewards = list(map(lambda m: m.tocsr(), rewards))
        if self._mdp_collapse_terminals:
            initial_state = 0
            terminal_state = S - 1
            mdp = EpisodicMDP(transitions, rewards,
                              discount_factor=1.0,
                              initial_state=initial_state,
                              terminal_state=terminal_state,
                              validate=self._mdp_validate)
        else:
            mdp = MDP(transitions, rewards, discount_factor=1.0,
                      validate=self._mdp_validate)
        self._logger.debug("finished transforming the graph into the MDP")
        return mdp

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
            state_len = len(state)  # cached to be used in igraph's queries
            if state_len != num_vars or not self._mdp_collapse_terminals:
                # If it's a terminal state the vertex is added only if
                # terminal states shouldn't be collapsed.
                graph.add_vertex(state=state, state_len=state_len)
        if self._mdp_collapse_terminals:
            # If the terminal states were collapsed, add a single
            # state (with the state dict set to None) where all the
            # configuration values are known. It will be the terminal
            # state in EpisodicMDP.
            graph.add_vertex(state=None, state_len=num_vars)
        self._logger.debug("finishing adding nodes")

    def _add_graph_edges(self, graph):
        # Add the initial graph edges: loop edges for the known
        # variables and one-step transitions.
        self._logger.debug("adding edges")
        self._add_loop_edges(graph)
        self._add_one_step_edges(graph)
        self._logger.debug("finished adding edges")

    def _add_loop_edges(self, graph):
        self._logger.debug("adding loop edges")
        # Add loop edges for the known variables.
        num_vars = len(self._config_values)
        edges, actions, rewards = [], [], []
        for v in graph.vs:
            if v["state_len"] == num_vars:
                # In a terminal state the rewards are set to zero so
                # the MDP can be solved with a discount factor of one.
                known_vars = range(num_vars)
                reward = 0
            else:
                # In an intermediate state the rewards are set to -1:
                # asked one question but got no new information.
                known_vars = v["state"].keys()
                reward = -1
            for var_index in known_vars:
                edges.append((v.index, v.index))
                actions.append(var_index)
                rewards.append(reward)
        # This is faster than using graph.add_edge one at a time.
        base_eid = graph.ecount()
        graph.add_edges(edges)
        for i in range(len(edges)):
            graph.es[base_eid + i]["action"] = actions[i]
            graph.es[base_eid + i]["reward"] = rewards[i]
            graph.es[base_eid + i]["prob"] = 1.0
        self._logger.debug("finished adding loop edges")

    def _add_one_step_edges(self, graph):
        self._logger.debug("adding one-step edges")
        # Add the one-step transition edges. Each edge is labelled with:
        # - the action corresponding to the variable that becomes known,
        # - the conditional probability of the variable taking the
        #   value, given the previous user responses, and
        # - a reward of zero (we asked one question and we got a
        #   response, no variables were automatically discovered).
        num_vars = len(self._config_values)
        edges, actions, probs = [], [], []
        for v in graph.vs(state_len_ne=num_vars):
            for w in graph.vs.select(state_len=v["state_len"] + 1):
                if self._is_one_step_transition(v, w):
                    edges.append((v.index, w.index))
                    v_config = set(v["state"].keys())
                    w_config = (set(range(num_vars)) if w["state"] is None
                                else set(w["state"].keys()))
                    var_index = next(iter(w_config - v_config))
                    actions.append(var_index)
                    if w["state"] is None:
                        # w is a collapsed terminal state. Add an edge with
                        # probability one, whatever the user answers is
                        # going to take her/him to the terminal state.
                        probs.append(1.0)
                    else:
                        var_value = w["state"][var_index]
                        prob = self._cond_prob({var_index: var_value}, v["state"])
                        probs.append(prob)
        # This is faster than using graph.add_edge one at a time.
        base_eid = graph.ecount()
        graph.add_edges(edges)
        for i in range(len(edges)):
            graph.es[base_eid + i]["action"] = actions[i]
            graph.es[base_eid + i]["reward"] = 0
            graph.es[base_eid + i]["prob"] = probs[i]
        self._logger.debug("finished adding one-step edges")

    def _is_one_step_transition(self, v, w):
        # Check v and w represent a one-step transition, i.e. from a
        # state in which k-1 variables are known to a state where k
        # variables are known (checked before the function is called).
        # Also, v and w have to differ only in one variable (the one
        # that becomes known after the question is asked).
        w_state = w["state"]
        if w_state is None:
            # Collapsed terminal state.
            return True
        else:
            for var_index, var_value in v["state"].items():
                if (var_index not in w_state or
                        w_state[var_index] != var_value):
                    return False
            return True

    def _update_graph(self, graph, rules):
        # Update the graph using the association rules.
        self._logger.debug("updating the graph using the association rules")
        inaccessible_vertices = []
        for rule in rules:
            # It doesn't make sense to apply the rule in a terminal
            # state, so the S selection is restricted to non-terminals.
            for v_s in graph.vs.select(state_len_ne=len(self._config_values)):
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
        # Remove the inaccesible vertices.
        if self._mdp_discard_states:
            graph.delete_vertices(inaccessible_vertices)
        self._logger.debug("found %d applications of %d rules",
                           len(inaccessible_vertices), len(rules))
        self._logger.debug("turned into a graph with %d nodes and %d edges",
                           graph.vcount(), graph.ecount())
        self._logger.debug("finished updating the graph")

    def _update_graph_cond_a(self, rule, s_or_sp):
        # (a) variables in lhs have known and same options in S and S'.
        # When called with S', it could be a collapsed terminal state
        # which will satisfy any rule.
        return (s_or_sp is None or rule.is_lhs_compatible(s_or_sp))

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
            return rule.is_rhs_compatible(s)
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
        # are targeting S and rewire them so that they now target S'.
        rewired_edges = []
        for e in graph.es.select(_target=v_s.index):
            # The reward is incremented by the difference in the
            # number of known variables between S' and S (i.e. the
            # variables that are automaticallly discovered).
            reward = e["reward"] + (v_sp["state_len"] - v_s["state_len"])
            if v_sp["state"] is None:
                # Eliminate multi-edges pointing to the collapsed
                # terminal state by combining them into a single edge.
                try:
                    # If the edge already exists, just add the probabilities.
                    ep = graph.es.find(_source=e.source, _target=v_sp.index,
                                       action=e["action"])
                    ep["prob"] += e["prob"]
                except ValueError:
                    # If it doesn't exist. add the edge.
                    graph.add_edge(e.source, target=v_sp, action=e["action"],
                                   prob=e["prob"], reward=reward)
            else:
                # Edge pointing to an intermediate state or an individual
                # terminal state when they are not collapsed.
                graph.add_edge(e.source, target=v_sp, action=e["action"],
                               prob=e["prob"], reward=reward)
            # The edge was rewired, add it to a list of edges to be removed.
            rewired_edges.append(e.index)
        # Remove the edges that were rewired.
        graph.delete_edges(rewired_edges)
