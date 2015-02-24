import numpy as np
from numpy.testing import assert_raises

from configurator.dp import DPDialogBuilder, EpisodicMDP


class TestDPDialogBuilder(object):

    def _test_builder(self, algorithm, improv, email_client):
        builder = DPDialogBuilder(var_domains=email_client.var_domains,
                                  rules=email_client.rules,
                                  sample=email_client.sample,
                                  dp_algorithm=algorithm,
                                  dp_discard_states=improv,
                                  dp_partial_rules=improv,
                                  dp_aggregate_terminals=improv,
                                  validate=True)
        dialog = builder.build_dialog()
        for config, num_questions in email_client.scenarios:
            dialog.reset()
            for i in range(num_questions):
                var_index = dialog.get_next_question()
                dialog.set_answer(var_index, config[var_index])
            assert dialog.is_complete()
            assert dialog.config == config

    def _test_builder_without_improv(self, algorithm, email_client):
        self._test_builder(algorithm, False, email_client)

    def _test_builder_with_improv(self, algorithm, email_client):
        self._test_builder(algorithm, True, email_client)

    def test_value_iteration_without_improv(self, email_client):
        self._test_builder_without_improv("value-iteration", email_client)

    def test_policy_iteration_without_improv(self, email_client):
        self._test_builder_without_improv("policy-iteration", email_client)

    def test_value_iteration_with_improv(self, email_client):
        self._test_builder_with_improv("value-iteration", email_client)

    def test_policy_iteration_with_improv(self, email_client):
        self._test_builder_with_improv("policy-iteration", email_client)


class TestEpisodicMDP(object):

    def test_validate(self, grid_world):
        P = grid_world.transitions
        R = grid_world.rewards
        gamma = grid_world.discount_factor
        S = grid_world.num_states
        S0 = grid_world.initial_state
        Sn = grid_world.terminal_state
        A = grid_world.num_actions
        invalid_P = np.random.rand(A, S, S)
        assert_raises(ValueError, EpisodicMDP, invalid_P, R, gamma, S0, Sn)
        invalid_R = np.random.rand(A, S, S)
        assert_raises(ValueError, EpisodicMDP, P, invalid_R, gamma, S0, Sn)
        invalid_gamma = 2
        assert_raises(ValueError, EpisodicMDP, P, R, invalid_gamma, S0, Sn)

    def test_policy_iteration(self, grid_world):
        P = grid_world.transitions
        R = grid_world.rewards
        gamma = grid_world.discount_factor
        mdp = EpisodicMDP(P, R, gamma)
        policy = mdp.policy_iteration()
        assert all([policy[i] == grid_world.policy[i]
                    for i in range(len(grid_world.policy))])

    def test_value_iteration(self, grid_world):
        P = grid_world.transitions
        R = grid_world.rewards
        gamma = grid_world.discount_factor
        mdp = EpisodicMDP(P, R, gamma)
        policy = mdp.value_iteration()
        assert all([policy[i] == grid_world.policy[i]
                    for i in range(len(grid_world.policy))])
